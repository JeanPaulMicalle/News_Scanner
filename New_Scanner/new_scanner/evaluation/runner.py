from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from ..models import Verdict
from ..pipeline import ResearchBaselinePipeline
from .datasets import load_labeled_claims_csv
from .metrics import compute_classification_metrics

EvaluationMode = Literal["rag", "llm_only"]


@dataclass(slots=True)
class ExperimentRunResult:
    run_id: str
    dataset_path: str
    output_dir: str
    row_count: int
    mode: str
    metrics: dict[str, object]


class ExperimentRunner:
    def __init__(self, pipeline: ResearchBaselinePipeline):
        self.pipeline = pipeline

    def run_csv_experiment(
        self,
        dataset_path: Path,
        mode: EvaluationMode = "rag",
        output_dir: Path | None = None,
        limit: int | None = None,
    ) -> ExperimentRunResult:
        records = load_labeled_claims_csv(dataset_path, limit=limit)
        if mode == "rag":
            self.pipeline.ensure_index()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = output_dir or (self.pipeline.config.artifacts_dir / "experiments" / run_id)
        base_dir.mkdir(parents=True, exist_ok=True)

        true_labels: list[str] = []
        predicted_labels: list[str] = []
        prediction_rows: list[dict[str, object]] = []

        for index, record in enumerate(records, start=1):
            verdict = self.pipeline.verify_claim(record.text, mode=mode)
            true_labels.append(record.label)
            predicted_labels.append(verdict.label)
            prediction_rows.append(self._build_prediction_row(index, record.text, record.label, verdict))

        metrics = compute_classification_metrics(true_labels, predicted_labels)
        self._write_outputs(
            output_dir=base_dir,
            run_id=run_id,
            dataset_path=dataset_path,
            row_count=len(records),
            mode=mode,
            metrics=metrics,
            prediction_rows=prediction_rows,
        )

        return ExperimentRunResult(
            run_id=run_id,
            dataset_path=str(dataset_path),
            output_dir=str(base_dir),
            row_count=len(records),
            mode=mode,
            metrics=metrics,
        )

    def run_comparison_experiment(
        self,
        dataset_path: Path,
        modes: list[EvaluationMode] | None = None,
        output_dir: Path | None = None,
        limit: int | None = None,
    ) -> dict[str, ExperimentRunResult]:
        selected_modes = modes or ["llm_only", "rag"]
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = output_dir or (self.pipeline.config.artifacts_dir / "comparisons" / run_id)
        base_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, ExperimentRunResult] = {}
        comparison_summary: dict[str, object] = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "dataset_path": str(dataset_path),
            "modes": {},
        }

        for mode in selected_modes:
            mode_dir = base_dir / mode
            result = self.run_csv_experiment(
                dataset_path=dataset_path,
                mode=mode,
                output_dir=mode_dir,
                limit=limit,
            )
            results[mode] = result
            comparison_summary["modes"][mode] = {
                "output_dir": result.output_dir,
                "row_count": result.row_count,
                "metrics": result.metrics,
            }

        with (base_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(comparison_summary, handle, ensure_ascii=False, indent=2)

        return results

    @staticmethod
    def _build_prediction_row(
        row_number: int,
        claim_text: str,
        true_label: str,
        verdict: Verdict,
    ) -> dict[str, object]:
        return {
            "row_number": row_number,
            "text": claim_text,
            "true_label": true_label,
            "predicted_label": verdict.label,
            "sentiment_label": verdict.sentiment_label,
            "sentiment_score": verdict.sentiment_score,
            "sources": verdict.sources,
            "explanation": verdict.explanation,
            "evidence": [asdict(chunk) for chunk in verdict.evidence],
        }

    def _write_outputs(
        self,
        output_dir: Path,
        run_id: str,
        dataset_path: Path,
        row_count: int,
        mode: str,
        metrics: dict[str, object],
        prediction_rows: list[dict[str, object]],
    ) -> None:
        summary = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "dataset_path": str(dataset_path),
            "row_count": row_count,
            "mode": mode,
            "config": {
                "embedding_model": self.pipeline.config.embedding_model,
                "sentiment_model": self.pipeline.config.sentiment_model,
                "ollama_model_name": self.pipeline.config.ollama_model_name,
                "top_k_results": self.pipeline.config.top_k_results,
                "chunk_size_words": self.pipeline.config.chunk_size_words,
                "chunk_overlap_words": self.pipeline.config.chunk_overlap_words,
            },
            "metrics": metrics,
        }

        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        with (output_dir / "predictions.json").open("w", encoding="utf-8") as handle:
            json.dump(prediction_rows, handle, ensure_ascii=False, indent=2)
