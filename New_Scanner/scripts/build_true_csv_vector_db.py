from __future__ import annotations

import csv
import json
import random
from dataclasses import replace
from pathlib import Path

from new_scanner.chunking import WordChunker
from new_scanner.config import load_config
from new_scanner.models import Article
from new_scanner.retrieval import EmbeddingService, FaissEvidenceRetriever


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "True.csv"
    holdout_path = root / "dataset" / "true_holdout_1000.csv"
    artifacts_dir = root / "artifacts" / "true_csv_vector_db"
    summary_path = artifacts_dir / "build_summary.json"
    random_seed = 42
    holdout_size = 1000

    rows = load_rows(csv_path)
    if len(rows) < holdout_size:
        raise RuntimeError(f"Need at least {holdout_size} valid rows, found {len(rows)}")

    holdout_rows, index_rows = split_rows(rows, holdout_size=holdout_size, random_seed=random_seed)
    write_holdout_csv(holdout_path, holdout_rows)

    articles = build_articles(index_rows)
    config = replace(load_config(root / "config.yaml"), artifacts_dir=artifacts_dir)
    chunker = WordChunker(
        chunk_size_words=config.chunk_size_words,
        chunk_overlap_words=config.chunk_overlap_words,
    )
    chunks = chunker.chunk_articles(articles)
    retriever = FaissEvidenceRetriever(config, EmbeddingService(config.embedding_model))
    retriever.build_and_save(chunks)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_csv": str(csv_path),
        "total_valid_rows": len(rows),
        "holdout_count": len(holdout_rows),
        "indexed_row_count": len(index_rows),
        "chunk_count": len(chunks),
        "holdout_csv": str(holdout_path),
        "artifacts_dir": str(artifacts_dir),
        "index_path": str(config.index_path),
        "metadata_path": str(config.metadata_path),
        "random_seed": random_seed,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "title": title,
                    "text": text,
                    "subject": (row.get("subject") or "").strip(),
                    "date": (row.get("date") or "").strip(),
                }
            )
    return rows


def split_rows(
    rows: list[dict[str, str]],
    holdout_size: int,
    random_seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(random_seed)
    holdout_indices = set(rng.sample(range(len(rows)), holdout_size))
    holdout_rows = [row for idx, row in enumerate(rows) if idx in holdout_indices]
    index_rows = [row for idx, row in enumerate(rows) if idx not in holdout_indices]
    return holdout_rows, index_rows


def write_holdout_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "text", "subject", "date"])
        writer.writeheader()
        writer.writerows(rows)


def build_articles(rows: list[dict[str, str]]) -> list[Article]:
    articles: list[Article] = []
    for idx, row in enumerate(rows, start=1):
        article_id = f"truecsv_{idx:06d}"
        title = row["title"] or article_id
        articles.append(
            Article(
                article_id=article_id,
                title=title,
                url=f"reuters-true://article/{article_id}",
                body=row["text"],
                source="Reuters",
                metadata={
                    "subject": row["subject"] or "Reuters",
                    "date": row["date"],
                    "dataset": "True.csv",
                },
            )
        )
    return articles


if __name__ == "__main__":
    main()
