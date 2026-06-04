from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import requests
import torch
from transformers import pipeline

from .config import AppConfig
from .models import Claim, EvidenceChunk, Verdict


class SentimentService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pipeline = None
        self.device = 0 if torch.cuda.is_available() else -1

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device,
            )
        return self._pipeline

    def analyze(self, text: str) -> tuple[str, float]:
        result = self.pipeline(text[:512])[0]
        return result["label"], float(result["score"])


class BaseOllamaVerifier(ABC):
    def __init__(self, config: AppConfig, sentiment_service: SentimentService):
        self.config = config
        self.sentiment_service = sentiment_service

    @abstractmethod
    def verify(self, claim: Claim, evidence: list[EvidenceChunk]) -> Verdict:
        raise NotImplementedError

    def _query_ollama(self, prompt: str) -> str:
        response = requests.post(
            f"{self.config.ollama_url}/api/generate",
            json={"model": self.config.ollama_model_name, "prompt": prompt},
            timeout=120,
        )
        response.raise_for_status()

        lines = response.text.strip().splitlines()
        combined = ""
        for line in lines:
            payload = json.loads(line)
            combined += payload.get("response", "")
        return combined.strip()

    @staticmethod
    def _extract_label(response_text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.IGNORECASE | re.DOTALL).strip()
        normalized = cleaned.lower()
        if normalized.startswith("not misleading"):
            return "Not misleading"
        if normalized.startswith("misleading"):
            return "Misleading"
        for line in cleaned.splitlines():
            candidate = line.strip().lower()
            if candidate.startswith("not misleading"):
                return "Not misleading"
            if candidate.startswith("misleading"):
                return "Misleading"
        return "Unknown"


class OllamaVerifier(BaseOllamaVerifier):
    def __init__(self, config: AppConfig, sentiment_service: SentimentService):
        super().__init__(config, sentiment_service)

    def verify(self, claim: Claim, evidence: list[EvidenceChunk]) -> Verdict:
        sentiment_label, sentiment_score = self.sentiment_service.analyze(claim.text)
        prompt = self._build_prompt(claim, evidence, sentiment_label, sentiment_score)
        response_text = self._query_ollama(prompt)
        label = self._extract_label(response_text)
        return Verdict(
            label=label,
            explanation=response_text,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            sources=sorted({item.url for item in evidence}),
            evidence=evidence,
        )

    def _build_prompt(
        self,
        claim: Claim,
        evidence: list[EvidenceChunk],
        sentiment_label: str,
        sentiment_score: float,
    ) -> str:
        evidence_sections: list[str] = []
        for item in evidence:
            trimmed = item.text[: self.config.max_evidence_chars].strip()
            evidence_sections.append(
                f"[Rank {item.rank} | Score {item.score:.4f} | Source {item.url}]\n{trimmed}"
            )

        joined_evidence = "\n\n".join(evidence_sections)
        return (
            "You are a research-oriented news fact checker.\n"
            "Use the retrieved evidence to assess whether the claim is misleading.\n"
            "Start your answer with exactly one of these labels:\n"
            "Misleading\n"
            "Not misleading\n\n"
            f"Claim sentiment: {sentiment_label} ({sentiment_score:.4f})\n\n"
            "Retrieved evidence:\n"
            f"{joined_evidence}\n\n"
            f"Claim:\n{claim.text}\n\n"
            "After the label, provide a short explanation grounded in the evidence and mention the sources you relied on."
        )


class LlmOnlyVerifier(BaseOllamaVerifier):
    def __init__(self, config: AppConfig, sentiment_service: SentimentService):
        super().__init__(config, sentiment_service)

    def verify(self, claim: Claim, evidence: list[EvidenceChunk]) -> Verdict:
        sentiment_label, sentiment_score = self.sentiment_service.analyze(claim.text)
        prompt = self._build_prompt(claim, sentiment_label, sentiment_score)
        response_text = self._query_ollama(prompt)
        label = self._extract_label(response_text)
        return Verdict(
            label=label,
            explanation=response_text,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            sources=[],
            evidence=[],
        )

    @staticmethod
    def _build_prompt(claim: Claim, sentiment_label: str, sentiment_score: float) -> str:
        return (
            "You are a research-oriented news fact checker.\n"
            "Assess whether the claim is misleading using only your own reasoning and prior knowledge.\n"
            "Do not assume you have retrieved evidence.\n"
            "Start your answer with exactly one of these labels:\n"
            "Misleading\n"
            "Not misleading\n\n"
            f"Claim sentiment: {sentiment_label} ({sentiment_score:.4f})\n\n"
            f"Claim:\n{claim.text}\n\n"
            "After the label, provide a short explanation of your reasoning."
        )
