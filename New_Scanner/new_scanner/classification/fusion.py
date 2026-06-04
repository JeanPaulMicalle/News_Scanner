from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LABELS = ("Not misleading", "Misleading")


@dataclass(slots=True)
class FusionDecision:
    article_probs: dict[str, float]
    evidence_probs: dict[str, float]
    fused_probs: dict[str, float]
    predicted_label: str
    article_weight: float
    evidence_weight: float
    evidence_strength: float
    rationale: str


def build_evidence_probabilities(
    evidence_label: str,
    evidence_count: int,
    source_count: int,
    base_strength: float = 0.68,
) -> tuple[dict[str, float], float]:
    if evidence_label not in LABELS:
        neutral = {label: 0.5 for label in LABELS}
        return neutral, 0.5

    strength = base_strength
    strength += min(source_count, 3) * 0.06
    strength += min(evidence_count, 5) * 0.025
    strength = max(0.5, min(strength, 0.9))

    if evidence_label == "Not misleading":
        return {
            "Not misleading": strength,
            "Misleading": 1.0 - strength,
        }, strength

    return {
        "Not misleading": 1.0 - strength,
        "Misleading": strength,
    }, strength


def fuse_article_and_evidence(
    article_probs: dict[str, float],
    evidence_label: str,
    evidence_count: int,
    source_count: int,
    article_weight: float = 0.35,
    evidence_weight: float = 0.65,
) -> FusionDecision:
    if article_weight <= 0 or evidence_weight <= 0:
        raise ValueError("article_weight and evidence_weight must be positive")

    total_weight = article_weight + evidence_weight
    normalized_article_weight = article_weight / total_weight
    normalized_evidence_weight = evidence_weight / total_weight

    safe_article_probs = {
        "Not misleading": float(article_probs.get("Not misleading", 0.5)),
        "Misleading": float(article_probs.get("Misleading", 0.5)),
    }
    article_total = safe_article_probs["Not misleading"] + safe_article_probs["Misleading"]
    if article_total <= 0:
        safe_article_probs = {"Not misleading": 0.5, "Misleading": 0.5}
    else:
        safe_article_probs = {
            label: value / article_total for label, value in safe_article_probs.items()
        }

    evidence_probs, evidence_strength = build_evidence_probabilities(
        evidence_label=evidence_label,
        evidence_count=evidence_count,
        source_count=source_count,
    )

    fused_probs = {
        label: (
            safe_article_probs[label] * normalized_article_weight
            + evidence_probs[label] * normalized_evidence_weight
        )
        for label in LABELS
    }
    fused_total = fused_probs["Not misleading"] + fused_probs["Misleading"]
    fused_probs = {label: value / fused_total for label, value in fused_probs.items()}
    predicted_label = max(fused_probs, key=fused_probs.get)

    rationale = (
        f"Article-only DistilBERT contributes {normalized_article_weight:.2f} of the final score. "
        f"Evidence-backed RAG contributes {normalized_evidence_weight:.2f}. "
        f"The evidence side was treated as hard evidence with strength {evidence_strength:.2f} "
        f"based on {source_count} unique sources and {evidence_count} retrieved chunks."
    )

    return FusionDecision(
        article_probs=safe_article_probs,
        evidence_probs=evidence_probs,
        fused_probs=fused_probs,
        predicted_label=predicted_label,
        article_weight=normalized_article_weight,
        evidence_weight=normalized_evidence_weight,
        evidence_strength=evidence_strength,
        rationale=rationale,
    )


def decision_to_dict(decision: FusionDecision) -> dict[str, Any]:
    return {
        "article_probs": decision.article_probs,
        "evidence_probs": decision.evidence_probs,
        "fused_probs": decision.fused_probs,
        "predicted_label": decision.predicted_label,
        "article_weight": decision.article_weight,
        "evidence_weight": decision.evidence_weight,
        "evidence_strength": decision.evidence_strength,
        "rationale": decision.rationale,
    }
