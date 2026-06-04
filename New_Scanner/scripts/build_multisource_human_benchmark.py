from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

from new_scanner.chunking import WordChunker
from new_scanner.config import load_config
from new_scanner.models import Article
from new_scanner.retrieval import EmbeddingService, FaissEvidenceRetriever


@dataclass(slots=True)
class BenchmarkRecord:
    record_id: str
    source_name: str
    dataset_family: str
    text: str
    label: str
    title: str = ""
    description: str = ""
    date: str = ""
    subject: str = ""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    external_root = root.parent / "Fakenews-dataset" / "Dataset"
    dataset_dir = root / "dataset" / "multisource_human_benchmark"
    artifacts_dir = root / "artifacts" / "multisource_human_db"
    summary_path = artifacts_dir / "build_summary.json"
    random_seed = 42
    requested_cap_per_label = 1000
    train_fraction = 0.8

    source_groups = {
        "reuters": {
            "real": load_reuters_rows(root / "True.csv", label="Not misleading"),
            "fake": load_reuters_rows(root / "Fake.csv", label="Misleading"),
        },
        "gossipcop_human": {
            "real": load_json_records(external_root / "GossipCop++" / "HR.json", source_name="GossipCop"),
            "fake": load_json_records(external_root / "GossipCop++" / "HF.json", source_name="GossipCop"),
        },
        "politifact_human": {
            "real": load_json_records(external_root / "PolitiFact++" / "HR.json", source_name="PolitiFact"),
            "fake": load_json_records(external_root / "PolitiFact++" / "HF.json", source_name="PolitiFact"),
        },
    }

    rng = random.Random(random_seed)
    train_rows: list[BenchmarkRecord] = []
    eval_rows: list[BenchmarkRecord] = []
    eval_rows_by_source: dict[str, list[BenchmarkRecord]] = {}
    effective_caps: dict[str, int] = {}
    source_summaries: dict[str, object] = {}

    for source_key, groups in source_groups.items():
        effective_cap = min(requested_cap_per_label, len(groups["real"]), len(groups["fake"]))
        effective_caps[source_key] = effective_cap
        sampled_real = sample_records(groups["real"], effective_cap, rng)
        sampled_fake = sample_records(groups["fake"], effective_cap, rng)
        sampled = sampled_real + sampled_fake
        sampled_labels = [record.label for record in sampled]

        source_train, source_eval = train_test_split(
            sampled,
            train_size=train_fraction,
            random_state=random_seed,
            stratify=sampled_labels,
        )
        train_rows.extend(source_train)
        eval_rows.extend(source_eval)
        eval_rows_by_source[source_key] = list(source_eval)
        source_summaries[source_key] = {
            "available_real": len(groups["real"]),
            "available_fake": len(groups["fake"]),
            "effective_cap_per_label": effective_cap,
            "sampled_total": len(sampled),
            "train_count": len(source_train),
            "eval_count": len(source_eval),
        }

    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_csv(dataset_dir / "train.csv", train_rows)
    write_dataset_csv(dataset_dir / "eval_overall.csv", eval_rows)
    for source_key, rows in eval_rows_by_source.items():
        write_dataset_csv(dataset_dir / f"eval_{source_key}.csv", rows)

    real_train_rows = [row for row in train_rows if row.label == "Not misleading"]
    articles = build_articles(real_train_rows)
    config = replace(load_config(root / "config.yaml"), artifacts_dir=artifacts_dir)
    chunker = WordChunker(
        chunk_size_words=config.chunk_size_words,
        chunk_overlap_words=config.chunk_overlap_words,
    )
    chunks = chunker.chunk_articles(articles)
    retriever = FaissEvidenceRetriever(config, EmbeddingService(config.embedding_model))
    retriever.build_and_save(chunks)

    summary = {
        "created_at": datetime.now().isoformat(),
        "random_seed": random_seed,
        "requested_cap_per_label": requested_cap_per_label,
        "train_fraction": train_fraction,
        "dataset_dir": str(dataset_dir),
        "artifacts_dir": str(artifacts_dir),
        "train_count": len(train_rows),
        "eval_overall_count": len(eval_rows),
        "vector_db_real_article_count": len(real_train_rows),
        "vector_db_chunk_count": len(chunks),
        "source_summaries": source_summaries,
        "index_path": str(config.index_path),
        "metadata_path": str(config.metadata_path),
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_reuters_rows(csv_path: Path, label: str) -> list[BenchmarkRecord]:
    rows: list[BenchmarkRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            text = compose_text(
                title=(row.get("title") or "").strip(),
                description="",
                body=(row.get("text") or "").strip(),
            )
            if not text:
                continue
            rows.append(
                BenchmarkRecord(
                    record_id=f"reuters_{label.lower().replace(' ', '_')}_{index:06d}",
                    source_name="Reuters",
                    dataset_family="Reuters",
                    text=text,
                    label=label,
                    title=(row.get("title") or "").strip(),
                    subject=(row.get("subject") or "").strip(),
                    date=(row.get("date") or "").strip(),
                )
            )
    return rows


def load_json_records(json_path: Path, source_name: str) -> list[BenchmarkRecord]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    label = "Misleading" if json_path.stem.upper().endswith("F") else "Not misleading"
    dataset_family = json_path.parent.name
    rows: list[BenchmarkRecord] = []
    for index, item in enumerate(payload.values(), start=1):
        text = compose_text(
            title=(item.get("title") or "").strip(),
            description=(item.get("description") or "").strip(),
            body=(item.get("text") or "").strip(),
        )
        if not text:
            continue
        rows.append(
            BenchmarkRecord(
                record_id=str(item.get("id") or f"{json_path.stem.lower()}_{index:06d}"),
                source_name=source_name,
                dataset_family=dataset_family,
                text=text,
                label=label,
                title=(item.get("title") or "").strip(),
                description=(item.get("description") or "").strip(),
            )
        )
    return rows


def compose_text(title: str, description: str, body: str) -> str:
    parts: list[str] = []
    if title:
        parts.append(title)
    if description and description not in body:
        parts.append(description)
    if body:
        parts.append(body)
    return "\n\n".join(part for part in parts if part).strip()


def sample_records(rows: list[BenchmarkRecord], sample_size: int, rng: random.Random) -> list[BenchmarkRecord]:
    if len(rows) <= sample_size:
        sampled = list(rows)
    else:
        sampled = rng.sample(rows, sample_size)
    rng.shuffle(sampled)
    return sampled


def write_dataset_csv(path: Path, rows: list[BenchmarkRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "source_name",
                "dataset_family",
                "title",
                "description",
                "text",
                "label",
                "subject",
                "date",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "record_id": row.record_id,
                    "source_name": row.source_name,
                    "dataset_family": row.dataset_family,
                    "title": row.title,
                    "description": row.description,
                    "text": row.text,
                    "label": row.label,
                    "subject": row.subject,
                    "date": row.date,
                }
            )


def build_articles(rows: list[BenchmarkRecord]) -> list[Article]:
    articles: list[Article] = []
    for row in rows:
        title = row.title or row.record_id
        safe_source = row.source_name.lower().replace(" ", "_")
        articles.append(
            Article(
                article_id=row.record_id,
                title=title,
                url=f"{safe_source}://article/{row.record_id}",
                body=row.text,
                source=row.source_name,
                metadata={
                    "dataset_family": row.dataset_family,
                    "date": row.date,
                    "subject": row.subject,
                },
            )
        )
    return articles


if __name__ == "__main__":
    main()
