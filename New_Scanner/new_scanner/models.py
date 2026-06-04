from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Article:
    article_id: str
    title: str
    url: str
    body: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Claim:
    text: str


@dataclass(slots=True)
class EvidenceChunk:
    chunk_id: str
    article_id: str
    article_title: str
    source: str
    url: str
    text: str
    rank: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Verdict:
    label: str
    explanation: str
    sentiment_label: str
    sentiment_score: float
    sources: list[str]
    evidence: list[EvidenceChunk]


def chunk_to_dict(chunk: EvidenceChunk) -> dict[str, Any]:
    return asdict(chunk)


def chunk_from_dict(data: dict[str, Any]) -> EvidenceChunk:
    return EvidenceChunk(**data)
