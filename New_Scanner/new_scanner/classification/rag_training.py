from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

from ..evaluation.datasets import load_labeled_claims_csv
from ..pipeline import ResearchBaselinePipeline
from .datasets import ID_TO_LABEL, LABEL_TO_ID
from .training import ClassificationTorchDataset, _compute_metrics, _normalize_metrics


@dataclass(slots=True)
class RetrievalAugmentedRecord:
    record_id: str
    text: str
    label: int
    label_name: str
    original_text: str
    retrieved_sources: list[str]
    retrieved_chunk_ids: list[str]


def train_retrieval_augmented_distilbert(
    dataset_path: Path,
    output_root: Path,
    config_path: str | Path | None = None,
    pipeline: ResearchBaselinePipeline | None = None,
    base_model_name_or_path: str = "distilbert-base-uncased",
    limit: int | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    random_state: int = 42,
    top_k: int | None = None,
) -> dict[str, object]:
    records = build_retrieval_augmented_records(
        dataset_path=dataset_path,
        config_path=config_path,
        pipeline=pipeline,
        limit=limit,
        top_k=top_k,
    )
    splits = split_retrieval_augmented_records(records, random_state=random_state)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()

    tokenizer = DistilBertTokenizerFast.from_pretrained(base_model_name_or_path)
    model = DistilBertForSequenceClassification.from_pretrained(
        base_model_name_or_path,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.to(device)

    train_dataset = ClassificationTorchDataset(splits["train"], tokenizer, max_length=max_length)
    validation_dataset = ClassificationTorchDataset(splits["validation"], tokenizer, max_length=max_length)
    test_dataset = ClassificationTorchDataset(splits["test"], tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        no_cuda=(device != "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=_compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    evaluation = trainer.evaluate(test_dataset)
    predictions_output = trainer.predict(test_dataset)
    predicted_ids = np.argmax(predictions_output.predictions, axis=1)

    model_dir = run_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    _write_records(run_dir / "train_records.json", splits["train"])
    _write_records(run_dir / "validation_records.json", splits["validation"])
    _write_records(run_dir / "test_records.json", splits["test"])
    _write_predictions(run_dir / "test_predictions.json", splits["test"], predicted_ids)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "dataset_path": str(dataset_path),
        "record_count": len(records),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "config": {
            "base_model_name_or_path": base_model_name_or_path,
            "limit": limit,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "random_state": random_state,
            "device": device,
            "top_k": top_k,
        },
        "test_metrics": _normalize_metrics(evaluation),
        "model_dir": str(model_dir),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def evaluate_retrieval_augmented_classifier(
    model_dir: Path,
    dataset_path: Path,
    config_path: str | Path | None = None,
    pipeline: ResearchBaselinePipeline | None = None,
    limit: int | None = None,
    random_state: int = 42,
    top_k: int | None = None,
) -> dict[str, object]:
    records = build_retrieval_augmented_records(
        dataset_path=dataset_path,
        config_path=config_path,
        pipeline=pipeline,
        limit=limit,
        top_k=top_k,
    )
    splits = split_retrieval_augmented_records(records, random_state=random_state)
    device = _resolve_device()
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)

    test_dataset = ClassificationTorchDataset(splits["test"], tokenizer, max_length=512)
    training_args = TrainingArguments(
        output_dir=str(model_dir / "eval_tmp"),
        report_to="none",
        per_device_eval_batch_size=8,
        no_cuda=(device != "cuda"),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=_compute_metrics,
        tokenizer=tokenizer,
    )
    evaluation = trainer.evaluate(test_dataset)
    return {
        "dataset_path": str(dataset_path),
        "record_count": len(records),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "device": device,
        "test_metrics": _normalize_metrics(evaluation),
    }


def build_retrieval_augmented_records(
    dataset_path: Path,
    config_path: str | Path | None = None,
    pipeline: ResearchBaselinePipeline | None = None,
    limit: int | None = None,
    top_k: int | None = None,
) -> list[RetrievalAugmentedRecord]:
    labeled_claims = load_labeled_claims_csv(dataset_path, limit=limit)
    pipeline = pipeline or ResearchBaselinePipeline.from_config(config_path)
    pipeline.ensure_index()
    effective_top_k = top_k or pipeline.config.top_k_results

    records: list[RetrievalAugmentedRecord] = []
    for index, row in enumerate(labeled_claims, start=1):
        evidence = pipeline.retriever.search(row.text, top_k=effective_top_k)
        normalized_label = _normalize_binary_label(row.label)
        records.append(
            RetrievalAugmentedRecord(
                record_id=f"row_{index}",
                text=_build_augmented_text(row.text, evidence),
                label=LABEL_TO_ID[normalized_label],
                label_name=normalized_label,
                original_text=row.text,
                retrieved_sources=sorted({chunk.url for chunk in evidence}),
                retrieved_chunk_ids=[chunk.chunk_id for chunk in evidence],
            )
        )

    return records


def split_retrieval_augmented_records(
    records: list[RetrievalAugmentedRecord],
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, list[RetrievalAugmentedRecord]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    labels = [record.label for record in records]
    train_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    if validation_size == 0:
        return {"train": train_records, "validation": [], "test": test_records}

    adjusted_validation = validation_size / (1 - test_size)
    train_labels = [record.label for record in train_records]
    train_records, validation_records = train_test_split(
        train_records,
        test_size=adjusted_validation,
        random_state=random_state,
        stratify=train_labels,
    )
    return {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_augmented_text(claim_text: str, evidence: list[Any]) -> str:
    sections = ["[CLAIM]", claim_text.strip()]
    for index, chunk in enumerate(evidence, start=1):
        sections.extend(
            [
                "",
                f"[EVIDENCE {index}]",
                f"Title: {chunk.article_title}",
                f"Source: {chunk.url}",
                chunk.text.strip(),
            ]
        )
    return "\n".join(sections).strip()


def _write_records(path: Path, records: list[RetrievalAugmentedRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, ensure_ascii=False, indent=2)


def _write_predictions(path: Path, records: list[RetrievalAugmentedRecord], predicted_ids: np.ndarray) -> None:
    rows = []
    for record, predicted_id in zip(records, predicted_ids):
        rows.append(
            {
                "record_id": record.record_id,
                "original_text": record.original_text,
                "augmented_text": record.text,
                "true_label": record.label_name,
                "predicted_label": ID_TO_LABEL[int(predicted_id)],
                "retrieved_sources": record.retrieved_sources,
                "retrieved_chunk_ids": record.retrieved_chunk_ids,
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def _normalize_binary_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in {"not misleading", "supported", "true", "likely"}:
        return "Not misleading"
    if normalized in {"misleading", "contradicted", "false", "unlikely"}:
        return "Misleading"
    raise ValueError(
        "Unsupported label. Expected one of: "
        "'Not misleading', 'Misleading', 'Supported', 'Contradicted', 'True', 'False', 'Likely', or 'Unlikely'."
    )
