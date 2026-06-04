from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split


LABEL_TO_ID = {
    "Not misleading": 0,
    "Misleading": 1,
}

ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(slots=True)
class ClassificationRecord:
    record_id: str
    text: str
    label: int
    label_name: str
    source: str
    url: str
    title: str


def load_fakenewsnet_politifact_records(
    root_dir: Path,
    include_title: bool = True,
    max_records: int | None = None,
) -> list[ClassificationRecord]:
    if not root_dir.exists():
        raise FileNotFoundError(f"FakeNewsNet root directory not found: {root_dir}")

    records: list[ClassificationRecord] = []
    for path in sorted(root_dir.rglob("news content.json")):
        label_name = _infer_label_from_path(path)
        if label_name is None:
            continue

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        title = _extract_first_non_empty(payload, ["title", "headline", "name"])
        body = _extract_first_non_empty(payload, ["text", "content", "article", "articleBody"])
        url = _extract_first_non_empty(payload, ["url"])
        source = _extract_source(payload, url)

        if not _is_valid_article_payload(title=title, body=body, url=url, source=source):
            continue

        text_parts = []
        if include_title and title:
            text_parts.append(title.strip())
        if body:
            text_parts.append(body.strip())
        combined_text = "\n\n".join(part for part in text_parts if part).strip()
        if not combined_text:
            continue

        story_id = path.parent.name
        records.append(
            ClassificationRecord(
                record_id=story_id,
                text=combined_text,
                label=LABEL_TO_ID[label_name],
                label_name=label_name,
                source=source,
                url=url,
                title=title,
            )
        )

        if max_records is not None and len(records) >= max_records:
            break

    if not records:
        raise RuntimeError(f"No valid FakeNewsNet PolitiFact records found in {root_dir}")

    return records


def split_records(
    records: list[ClassificationRecord],
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, list[ClassificationRecord]]:
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


def _infer_label_from_path(path: Path) -> str | None:
    lowered_parts = [part.lower() for part in path.parts]
    if "fake" in lowered_parts:
        return "Misleading"
    if "real" in lowered_parts:
        return "Not misleading"
    return None


def _extract_first_non_empty(payload: dict, keys: list[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_source(payload: dict, url: str) -> str:
    publisher = payload.get("source") or payload.get("publisher")
    if isinstance(publisher, dict):
        name = publisher.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    if isinstance(publisher, str) and publisher.strip():
        return publisher.strip()
    if url:
        try:
            from urllib.parse import urlparse

            return urlparse(url).netloc or "unknown"
        except Exception:
            return "unknown"
    return "unknown"


def _is_valid_article_payload(title: str, body: str, url: str, source: str) -> bool:
    normalized_title = (title or "").strip().lower()
    normalized_body = _normalize_whitespace((body or "").strip().lower())
    normalized_url = (url or "").strip().lower()
    normalized_source = (source or "").strip().lower()

    if len(normalized_body) < 200:
        return False

    blocked_title_patterns = [
        r"^facebook$",
        r"^sign up$",
        r"^login$",
        r"^page not found$",
        r"^404",
        r"^error$",
    ]
    if any(re.search(pattern, normalized_title) for pattern in blocked_title_patterns):
        return False

    blocked_url_fragments = [
        "facebook.com",
        "/signup",
        "/login",
    ]
    if any(fragment in normalized_url for fragment in blocked_url_fragments):
        return False

    blocked_source_fragments = [
        "facebook.com",
        "web.archive.org",
    ]
    if any(fragment in normalized_source for fragment in blocked_source_fragments):
        return False

    blocked_body_patterns = [
        r"\b404\b",
        r"page not found",
        r"sign up",
        r"log in",
        r"access denied",
        r"account suspended",
        r"domain for sale",
        r"this site can.t be reached",
        r"website is for sale",
        r"pasaran terlengkap dari agen togel terbaik",
    ]
    if any(re.search(pattern, normalized_body) for pattern in blocked_body_patterns):
        return False

    return True


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
