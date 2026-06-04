from __future__ import annotations

import json
from pathlib import Path

from new_scanner.classification.rag_training import (
    evaluate_retrieval_augmented_classifier,
    train_retrieval_augmented_distilbert,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "dataset" / "multisource_human_benchmark" / "train.csv"
    config_path = root / "config_multisource_human_db.yaml"
    base_model_dir = root / "artifacts" / "classification" / "20260319_174042" / "model"
    output_root = root / "artifacts" / "multisource_human_rag_classification"

    summary = train_retrieval_augmented_distilbert(
        dataset_path=dataset_path,
        output_root=output_root,
        config_path=config_path,
        base_model_name_or_path=str(base_model_dir),
        epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        top_k=5,
    )

    eval_summary = evaluate_retrieval_augmented_classifier(
        model_dir=Path(summary["model_dir"]),
        dataset_path=dataset_path,
        config_path=config_path,
        top_k=5,
    )

    report = {
        "training_summary": summary,
        "evaluation_summary": eval_summary,
    }
    report_path = Path(summary["model_dir"]).parent / "post_train_evaluation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
