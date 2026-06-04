from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from new_scanner.classification.datasets import ID_TO_LABEL, LABEL_TO_ID
from new_scanner.classification.rag_training import _build_augmented_text
from new_scanner.models import EvidenceChunk
from new_scanner.pipeline import ResearchBaselinePipeline


@dataclass(slots=True)
class EvalRecord:
    record_id: str
    source_name: str
    dataset_family: str
    text: str
    label: str
    title: str


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "dataset" / "multisource_human_benchmark"
    model_dir = root / "artifacts" / "multisource_human_rag_classification"
    latest_model_dir = resolve_latest_model_dir(model_dir)
    config_path = root / "config_multisource_human_db.yaml"
    output_root = root / "artifacts" / "multisource_human_eval"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ResearchBaselinePipeline.from_config(config_path)
    pipeline.ensure_index()

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(latest_model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(latest_model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    datasets = {
        "overall": load_eval_records(dataset_dir / "eval_overall.csv"),
        "reuters": load_eval_records(dataset_dir / "eval_reuters.csv"),
        "gossipcop_human": load_eval_records(dataset_dir / "eval_gossipcop_human.csv"),
        "politifact_human": load_eval_records(dataset_dir / "eval_politifact_human.csv"),
    }

    summary: dict[str, object] = {
        "created_at": datetime.now().isoformat(),
        "run_id": run_id,
        "model_dir": str(latest_model_dir),
        "vector_db_config": str(config_path),
        "datasets": {},
    }

    for dataset_name, records in datasets.items():
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        result = evaluate_records(
            records=records,
            pipeline=pipeline,
            tokenizer=tokenizer,
            model=model,
            device=device,
            retrieval_top_k=5,
        )
        (dataset_output_dir / "summary.json").write_text(
            json.dumps(result["summary"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_predictions_csv(dataset_output_dir / "predictions.csv", result["predictions"])
        summary["datasets"][dataset_name] = result["summary"]

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def resolve_latest_model_dir(root: Path) -> Path:
    run_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {root}")
    return run_dirs[-1] / "model"


def load_eval_records(path: Path) -> list[EvalRecord]:
    rows: list[EvalRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                continue
            rows.append(
                EvalRecord(
                    record_id=(row.get("record_id") or "").strip(),
                    source_name=(row.get("source_name") or "").strip(),
                    dataset_family=(row.get("dataset_family") or "").strip(),
                    text=text,
                    label=label,
                    title=(row.get("title") or "").strip(),
                )
            )
    return rows


def evaluate_records(
    records: list[EvalRecord],
    pipeline: ResearchBaselinePipeline,
    tokenizer: DistilBertTokenizerFast,
    model: DistilBertForSequenceClassification,
    device: str,
    retrieval_top_k: int,
    batch_size: int = 8,
) -> dict[str, object]:
    augmented_texts: list[str] = []
    retrieved_sources_rows: list[list[str]] = []

    for record in records:
        evidence = pipeline.retriever.search(record.text, top_k=retrieval_top_k)
        augmented_texts.append(_build_augmented_text(record.text, evidence))
        retrieved_sources_rows.append(sorted({chunk.url for chunk in evidence}))

    true_ids = np.array([LABEL_TO_ID[record.label] for record in records], dtype=np.int64)
    predicted_ids: list[int] = []
    probability_rows: list[list[float]] = []

    for start in range(0, len(augmented_texts), batch_size):
        batch_texts = augmented_texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        predicted_ids.extend(np.argmax(probs, axis=1).tolist())
        probability_rows.extend(probs.tolist())

    accuracy = accuracy_score(true_ids, predicted_ids)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_ids,
        predicted_ids,
        average="macro",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        true_ids,
        predicted_ids,
        labels=[0, 1],
        zero_division=0,
    )
    matrix = confusion_matrix(true_ids, predicted_ids, labels=[0, 1]).tolist()

    predictions: list[dict[str, object]] = []
    for record, predicted_id, prob, retrieved_sources in zip(records, predicted_ids, probability_rows, retrieved_sources_rows):
        predictions.append(
            {
                "record_id": record.record_id,
                "source_name": record.source_name,
                "dataset_family": record.dataset_family,
                "title": record.title,
                "true_label": record.label,
                "predicted_label": ID_TO_LABEL[int(predicted_id)],
                "p_not_misleading": prob[0],
                "p_misleading": prob[1],
                "retrieved_sources": " | ".join(retrieved_sources),
            }
        )

    summary = {
        "record_count": len(records),
        "class_breakdown": count_labels(records),
        "metrics": {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "labels": ["Not misleading", "Misleading"],
            "per_label": {
                "Not misleading": {
                    "precision": float(per_precision[0]),
                    "recall": float(per_recall[0]),
                    "f1": float(per_f1[0]),
                    "support": int(per_support[0]),
                },
                "Misleading": {
                    "precision": float(per_precision[1]),
                    "recall": float(per_recall[1]),
                    "f1": float(per_f1[1]),
                    "support": int(per_support[1]),
                },
            },
            "confusion_matrix": matrix,
        },
    }
    return {"summary": summary, "predictions": predictions}


def count_labels(records: list[EvalRecord]) -> dict[str, int]:
    counts = {"Not misleading": 0, "Misleading": 0}
    for record in records:
        counts[record.label] += 1
    return counts


def write_predictions_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
