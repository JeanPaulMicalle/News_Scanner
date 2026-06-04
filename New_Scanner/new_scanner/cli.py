from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .charting import generate_comparison_chart
from .classification.rag_training import (
    evaluate_retrieval_augmented_classifier,
    train_retrieval_augmented_distilbert,
)
from .classification.training import (
    compare_trained_vs_untrained_classifier,
    evaluate_saved_classifier,
    train_distilbert_on_politifact,
)
from .evaluation.runner import ExperimentRunner
from .persona import PersonaCloneService
from .pipeline import ResearchBaselinePipeline


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    sanitized = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(sanitized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-ready baseline pipeline for the new scanner.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a config.yaml file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Build and save the retrieval index.")

    index_persona_parser = subparsers.add_parser(
        "index-persona",
        help="Build and save a persona-specific vector DB from a local corpus.",
    )
    index_persona_parser.add_argument("--subject", required=True, help="Persona subject, for example 'Donald Trump'.")
    index_persona_parser.add_argument("--corpus-dir", required=True, help="Directory containing persona corpus files.")
    index_persona_parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory where the persona vector DB artifacts should be saved.",
    )
    index_persona_parser.add_argument("--top-k", type=int, default=None)

    check_parser = subparsers.add_parser("check", help="Verify a claim against the indexed corpus.")
    check_parser.add_argument("--text", required=True, help="Claim text to verify.")
    check_parser.add_argument(
        "--mode",
        choices=["rag", "llm_only"],
        default="rag",
        help="Verification mode to use.",
    )

    check_persona_parser = subparsers.add_parser(
        "check-persona",
        help="Assess whether a statement or action matches a persona using retrieved evidence.",
    )
    check_persona_parser.add_argument("--subject", required=True, help="Persona subject, for example 'Donald Trump'.")
    check_persona_parser.add_argument("--text", required=True, help="Candidate statement or action to assess.")
    check_persona_parser.add_argument("--corpus-dir", required=True, help="Directory containing persona corpus files.")
    check_persona_parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing or receiving the persona vector DB artifacts.",
    )
    check_persona_parser.add_argument("--top-k", type=int, default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run the pipeline against a labeled CSV dataset.")
    evaluate_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a CSV file with 'text' and 'label' columns.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to store experiment results.",
    )
    evaluate_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick experiments.",
    )
    evaluate_parser.add_argument(
        "--mode",
        choices=["rag", "llm_only"],
        default="rag",
        help="Single evaluation mode to run.",
    )
    evaluate_parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Run both llm_only and rag and save a comparison summary.",
    )

    train_classifier_parser = subparsers.add_parser(
        "train-classifier",
        help="Train a DistilBERT classifier on FakeNewsNet PolitiFact.",
    )
    train_classifier_parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the FakeNewsNet PolitiFact directory.",
    )
    train_classifier_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to store classifier runs.",
    )
    train_classifier_parser.add_argument("--epochs", type=int, default=3)
    train_classifier_parser.add_argument("--batch-size", type=int, default=8)
    train_classifier_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_classifier_parser.add_argument("--max-records", type=int, default=None)

    evaluate_classifier_parser = subparsers.add_parser(
        "evaluate-classifier",
        help="Evaluate a saved DistilBERT classifier on FakeNewsNet PolitiFact.",
    )
    evaluate_classifier_parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the saved model directory.",
    )
    evaluate_classifier_parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the FakeNewsNet PolitiFact directory.",
    )

    compare_classifier_parser = subparsers.add_parser(
        "compare-classifier",
        help="Compare a trained DistilBERT classifier against an untrained DistilBERT baseline.",
    )
    compare_classifier_parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the trained model directory.",
    )
    compare_classifier_parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the FakeNewsNet PolitiFact directory.",
    )
    compare_classifier_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to store comparison outputs.",
    )

    train_rag_classifier_parser = subparsers.add_parser(
        "train-rag-classifier",
        help="Train a retrieval-augmented DistilBERT classifier on a labeled CSV using the vector DB.",
    )
    train_rag_classifier_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a CSV file with 'text' and 'label' columns.",
    )
    train_rag_classifier_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to store classifier runs.",
    )
    train_rag_classifier_parser.add_argument(
        "--base-model-dir",
        default="distilbert-base-uncased",
        help="Base Hugging Face model name or local trained model directory to continue from.",
    )
    train_rag_classifier_parser.add_argument("--epochs", type=int, default=3)
    train_rag_classifier_parser.add_argument("--batch-size", type=int, default=8)
    train_rag_classifier_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_rag_classifier_parser.add_argument("--limit", type=int, default=None)
    train_rag_classifier_parser.add_argument("--top-k", type=int, default=None)

    evaluate_rag_classifier_parser = subparsers.add_parser(
        "evaluate-rag-classifier",
        help="Evaluate a saved retrieval-augmented DistilBERT classifier on a labeled CSV.",
    )
    evaluate_rag_classifier_parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the saved model directory.",
    )
    evaluate_rag_classifier_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a CSV file with 'text' and 'label' columns.",
    )
    evaluate_rag_classifier_parser.add_argument("--limit", type=int, default=None)
    evaluate_rag_classifier_parser.add_argument("--top-k", type=int, default=None)

    train_persona_classifier_parser = subparsers.add_parser(
        "train-persona-classifier",
        help="Train a retrieval-augmented DistilBERT classifier against a persona-specific vector DB.",
    )
    train_persona_classifier_parser.add_argument("--subject", required=True, help="Persona subject, for example 'Donald Trump'.")
    train_persona_classifier_parser.add_argument("--corpus-dir", required=True, help="Directory containing persona corpus files.")
    train_persona_classifier_parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing or receiving the persona vector DB artifacts.",
    )
    train_persona_classifier_parser.add_argument("--dataset", required=True, help="Path to a CSV file with 'text' and 'label' columns.")
    train_persona_classifier_parser.add_argument("--output-dir", default=None, help="Optional directory to store classifier runs.")
    train_persona_classifier_parser.add_argument(
        "--base-model-dir",
        default="distilbert-base-uncased",
        help="Base Hugging Face model name or local trained model directory to continue from.",
    )
    train_persona_classifier_parser.add_argument("--epochs", type=int, default=3)
    train_persona_classifier_parser.add_argument("--batch-size", type=int, default=8)
    train_persona_classifier_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_persona_classifier_parser.add_argument("--limit", type=int, default=None)
    train_persona_classifier_parser.add_argument("--top-k", type=int, default=None)

    evaluate_persona_classifier_parser = subparsers.add_parser(
        "evaluate-persona-classifier",
        help="Evaluate a saved persona-specific retrieval-augmented DistilBERT classifier.",
    )
    evaluate_persona_classifier_parser.add_argument("--subject", required=True, help="Persona subject, for example 'Donald Trump'.")
    evaluate_persona_classifier_parser.add_argument("--corpus-dir", required=True, help="Directory containing persona corpus files.")
    evaluate_persona_classifier_parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing the persona vector DB artifacts.",
    )
    evaluate_persona_classifier_parser.add_argument("--model-dir", required=True, help="Path to the saved model directory.")
    evaluate_persona_classifier_parser.add_argument("--dataset", required=True, help="Path to a CSV file with 'text' and 'label' columns.")
    evaluate_persona_classifier_parser.add_argument("--limit", type=int, default=None)
    evaluate_persona_classifier_parser.add_argument("--top-k", type=int, default=None)

    generate_chart_parser = subparsers.add_parser(
        "generate-chart",
        help="Generate an SVG chart from a comparison-data JSON file.",
    )
    generate_chart_parser.add_argument(
        "--data-json",
        required=True,
        help="Path to a JSON file shaped like comparison_data.json.",
    )
    generate_chart_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to store generated charts.",
    )
    generate_chart_parser.add_argument(
        "--title",
        default="Trained vs Untrained DistilBERT",
        help="Chart title.",
    )
    generate_chart_parser.add_argument(
        "--subtitle",
        default="Comparison across supplied datasets",
        help="Chart subtitle.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = ResearchBaselinePipeline.from_config(args.config)

    if args.command == "index":
        chunk_count = pipeline.build_index()
        print(f"Indexed {chunk_count} chunks into {pipeline.config.index_path}")
        return

    if args.command == "index-persona":
        persona = PersonaCloneService(
            subject=args.subject,
            corpus_dir=Path(args.corpus_dir).resolve(),
            artifacts_dir=Path(args.artifacts_dir).resolve(),
            config_path=args.config,
            top_k=args.top_k,
        )
        chunk_count = persona.build_index()
        print(f"Subject: {args.subject}")
        print(f"Indexed {chunk_count} chunks into {persona.config.index_path}")
        return

    if args.command == "check":
        verdict = pipeline.verify_claim(args.text, mode=args.mode)
        print(f"Mode: {args.mode}")
        print(f"Label: {verdict.label}")
        print(f"Sentiment: {verdict.sentiment_label} ({verdict.sentiment_score:.4f})")
        print("\nExplanation:")
        _safe_print(verdict.explanation)
        print("\nSources:")
        for source in verdict.sources:
            print(f"- {source}")
        return

    if args.command == "check-persona":
        persona = PersonaCloneService(
            subject=args.subject,
            corpus_dir=Path(args.corpus_dir).resolve(),
            artifacts_dir=Path(args.artifacts_dir).resolve(),
            config_path=args.config,
            top_k=args.top_k,
        )
        verdict = persona.assess(args.text)
        print(f"Subject: {verdict.subject}")
        print(f"Label: {verdict.label}")
        print("\nExplanation:")
        _safe_print(verdict.explanation)
        print("\nSources:")
        for source in verdict.sources:
            print(f"- {source}")
        return

    if args.command == "evaluate":
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None
        runner = ExperimentRunner(pipeline)
        dataset_path = Path(args.dataset).resolve()

        if args.compare_baselines:
            results = runner.run_comparison_experiment(
                dataset_path=dataset_path,
                output_dir=output_dir,
                limit=args.limit,
            )
            print(f"Dataset: {dataset_path}")
            for mode, result in results.items():
                print(f"\nMode: {mode}")
                print(f"Run ID: {result.run_id}")
                print(f"Rows evaluated: {result.row_count}")
                print(f"Accuracy: {result.metrics['accuracy']:.4f}")
                print(f"Macro F1: {result.metrics['macro_f1']:.4f}")
                print(f"Saved to: {result.output_dir}")
            return

        result = runner.run_csv_experiment(
            dataset_path=dataset_path,
            mode=args.mode,
            output_dir=output_dir,
            limit=args.limit,
        )
        print(f"Run ID: {result.run_id}")
        print(f"Dataset: {result.dataset_path}")
        print(f"Mode: {result.mode}")
        print(f"Rows evaluated: {result.row_count}")
        print(f"Accuracy: {result.metrics['accuracy']:.4f}")
        print(f"Macro F1: {result.metrics['macro_f1']:.4f}")
        print(f"Saved to: {result.output_dir}")
        return

    if args.command == "train-classifier":
        output_root = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (pipeline.config.artifacts_dir / "classification")
        )
        summary = train_distilbert_on_politifact(
            dataset_dir=Path(args.dataset_dir).resolve(),
            output_root=output_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_records=args.max_records,
        )
        print(f"Run ID: {summary['run_id']}")
        print(f"Dataset: {summary['dataset_dir']}")
        print(f"Records: {summary['record_count']}")
        print(f"Model saved to: {summary['model_dir']}")
        print(f"Test metrics: {summary['test_metrics']}")
        return

    if args.command == "evaluate-classifier":
        summary = evaluate_saved_classifier(
            model_dir=Path(args.model_dir).resolve(),
            dataset_dir=Path(args.dataset_dir).resolve(),
        )
        print(f"Dataset: {summary['dataset_dir']}")
        print(f"Records: {summary['record_count']}")
        print(f"Test metrics: {summary['test_metrics']}")
        return

    if args.command == "compare-classifier":
        output_root = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (pipeline.config.artifacts_dir / "classification_comparisons")
        )
        summary = compare_trained_vs_untrained_classifier(
            model_dir=Path(args.model_dir).resolve(),
            dataset_dir=Path(args.dataset_dir).resolve(),
            output_root=output_root,
        )
        print(f"Run ID: {summary['run_id']}")
        print(f"Dataset: {summary['dataset_dir']}")
        print(f"Records: {summary['record_count']}")
        print(f"Device: {summary['device']}")
        print(f"Trained metrics: {summary['trained']}")
        print(f"Untrained metrics: {summary['untrained']}")
        print(f"Saved to: {output_root / summary['run_id']}")
        return

    if args.command == "train-rag-classifier":
        output_root = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (pipeline.config.artifacts_dir / "rag_classification")
        )
        summary = train_retrieval_augmented_distilbert(
            dataset_path=Path(args.dataset).resolve(),
            output_root=output_root,
            config_path=args.config,
            base_model_name_or_path=args.base_model_dir,
            limit=args.limit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            top_k=args.top_k,
        )
        print(f"Run ID: {summary['run_id']}")
        print(f"Dataset: {summary['dataset_path']}")
        print(f"Records: {summary['record_count']}")
        print(f"Model saved to: {summary['model_dir']}")
        print(f"Test metrics: {summary['test_metrics']}")
        return

    if args.command == "evaluate-rag-classifier":
        summary = evaluate_retrieval_augmented_classifier(
            model_dir=Path(args.model_dir).resolve(),
            dataset_path=Path(args.dataset).resolve(),
            config_path=args.config,
            limit=args.limit,
            top_k=args.top_k,
        )
        print(f"Dataset: {summary['dataset_path']}")
        print(f"Records: {summary['record_count']}")
        print(f"Split sizes: {summary['split_sizes']}")
        print(f"Test metrics: {summary['test_metrics']}")
        return

    if args.command == "train-persona-classifier":
        persona = PersonaCloneService(
            subject=args.subject,
            corpus_dir=Path(args.corpus_dir).resolve(),
            artifacts_dir=Path(args.artifacts_dir).resolve(),
            config_path=args.config,
            top_k=args.top_k,
        )
        output_root = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (pipeline.config.artifacts_dir / "persona_classification")
        )
        summary = persona.train_classifier(
            dataset_path=Path(args.dataset).resolve(),
            output_root=output_root,
            base_model_name_or_path=args.base_model_dir,
            limit=args.limit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            top_k=args.top_k,
        )
        print(f"Subject: {args.subject}")
        print(f"Run ID: {summary['run_id']}")
        print(f"Dataset: {summary['dataset_path']}")
        print(f"Records: {summary['record_count']}")
        print(f"Model saved to: {summary['model_dir']}")
        print(f"Test metrics: {summary['test_metrics']}")
        return

    if args.command == "evaluate-persona-classifier":
        persona = PersonaCloneService(
            subject=args.subject,
            corpus_dir=Path(args.corpus_dir).resolve(),
            artifacts_dir=Path(args.artifacts_dir).resolve(),
            config_path=args.config,
            top_k=args.top_k,
        )
        summary = persona.evaluate_classifier(
            model_dir=Path(args.model_dir).resolve(),
            dataset_path=Path(args.dataset).resolve(),
            limit=args.limit,
            top_k=args.top_k,
        )
        print(f"Subject: {args.subject}")
        print(f"Dataset: {summary['dataset_path']}")
        print(f"Records: {summary['record_count']}")
        print(f"Split sizes: {summary['split_sizes']}")
        print(f"Test metrics: {summary['test_metrics']}")
        return

    if args.command == "generate-chart":
        output_root = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (pipeline.config.artifacts_dir / "charts")
        )
        result = generate_comparison_chart(
            data_json_path=Path(args.data_json).resolve(),
            output_root=output_root,
            title=args.title,
            subtitle=args.subtitle,
        )
        print(f"Run ID: {result['run_id']}")
        print(f"Chart: {result['chart_path']}")
        print(f"Normalized data: {result['data_path']}")
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
