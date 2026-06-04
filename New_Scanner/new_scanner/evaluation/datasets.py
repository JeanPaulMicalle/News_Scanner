from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LabeledClaim:
    text: str
    label: str


def load_labeled_claims_csv(dataset_path: Path, limit: int | None = None) -> list[LabeledClaim]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: list[LabeledClaim] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Dataset CSV must contain 'text' and 'label' columns.")

        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                continue
            rows.append(LabeledClaim(text=text, label=label))
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise RuntimeError(f"No valid rows found in dataset: {dataset_path}")

    return rows
