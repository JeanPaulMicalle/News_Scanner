from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from .datasets import ID_TO_LABEL, LABEL_TO_ID, ClassificationRecord, load_fakenewsnet_politifact_records, split_records


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClassificationTorchDataset(Dataset):
    def __init__(self, records: list[ClassificationRecord], tokenizer: DistilBertTokenizerFast, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoding = self.tokenizer(
            record.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(record.label, dtype=torch.long)
        return item


def train_distilbert_on_politifact(
    dataset_dir: Path,
    output_root: Path,
    include_title: bool = True,
    max_records: int | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    random_state: int = 42,
) -> dict[str, object]:
    records = load_fakenewsnet_politifact_records(
        root_dir=dataset_dir,
        include_title=include_title,
        max_records=max_records,
    )
    splits = split_records(records, random_state=random_state)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
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
        "dataset_dir": str(dataset_dir),
        "record_count": len(records),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "config": {
            "include_title": include_title,
            "max_records": max_records,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "random_state": random_state,
            "model_name": "distilbert-base-uncased",
            "device": device,
        },
        "test_metrics": _normalize_metrics(evaluation),
        "model_dir": str(model_dir),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def evaluate_saved_classifier(model_dir: Path, dataset_dir: Path, include_title: bool = True) -> dict[str, object]:
    records = load_fakenewsnet_politifact_records(root_dir=dataset_dir, include_title=include_title)
    splits = split_records(records)
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
        "dataset_dir": str(dataset_dir),
        "record_count": len(records),
        "device": device,
        "test_metrics": _normalize_metrics(evaluation),
    }


def compare_trained_vs_untrained_classifier(
    model_dir: Path,
    dataset_dir: Path,
    output_root: Path,
    include_title: bool = True,
) -> dict[str, Any]:
    records = load_fakenewsnet_politifact_records(root_dir=dataset_dir, include_title=include_title)
    splits = split_records(records)
    device = _resolve_device()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trained_tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    trained_model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    trained_model.to(device)

    untrained_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    untrained_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    untrained_model.to(device)

    test_dataset_for_trained = ClassificationTorchDataset(splits["test"], trained_tokenizer, max_length=512)
    test_dataset_for_untrained = ClassificationTorchDataset(splits["test"], untrained_tokenizer, max_length=512)

    trained_results = _run_classifier_evaluation(
        model=trained_model,
        tokenizer=trained_tokenizer,
        dataset=test_dataset_for_trained,
        records=splits["test"],
        device=device,
    )
    untrained_results = _run_classifier_evaluation(
        model=untrained_model,
        tokenizer=untrained_tokenizer,
        dataset=test_dataset_for_untrained,
        records=splits["test"],
        device=device,
    )

    comparison_summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "dataset_dir": str(dataset_dir),
        "record_count": len(records),
        "device": device,
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "trained_model_dir": str(model_dir),
        "trained": trained_results["metrics"],
        "untrained": untrained_results["metrics"],
    }

    with (run_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison_summary, handle, ensure_ascii=False, indent=2)
    with (run_dir / "trained_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(trained_results["predictions"], handle, ensure_ascii=False, indent=2)
    with (run_dir / "untrained_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(untrained_results["predictions"], handle, ensure_ascii=False, indent=2)

    return comparison_summary


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _run_classifier_evaluation(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    dataset: ClassificationTorchDataset,
    records: list[ClassificationRecord],
    device: str,
) -> dict[str, Any]:
    training_args = TrainingArguments(
        output_dir=str(Path.cwd() / "tmp_classifier_eval"),
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
    evaluation = trainer.evaluate(dataset)
    predictions_output = trainer.predict(dataset)
    predicted_ids = np.argmax(predictions_output.predictions, axis=1)
    prediction_rows = []
    for record, predicted_id in zip(records, predicted_ids):
        prediction_rows.append(
            {
                "record_id": record.record_id,
                "text": record.text,
                "true_label": record.label_name,
                "predicted_label": ID_TO_LABEL[int(predicted_id)],
                "source": record.source,
                "url": record.url,
                "title": record.title,
            }
        )
    return {
        "metrics": _normalize_metrics(evaluation),
        "predictions": prediction_rows,
    }


def _write_records(path: Path, records: list[ClassificationRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, ensure_ascii=False, indent=2)


def _write_predictions(path: Path, records: list[ClassificationRecord], predicted_ids: np.ndarray) -> None:
    rows = []
    for record, predicted_id in zip(records, predicted_ids):
        rows.append(
            {
                "record_id": record.record_id,
                "text": record.text,
                "true_label": record.label_name,
                "predicted_label": ID_TO_LABEL[int(predicted_id)],
                "source": record.source,
                "url": record.url,
                "title": record.title,
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def _normalize_metrics(metrics: dict[str, object]) -> dict[str, object]:
    normalized = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized
