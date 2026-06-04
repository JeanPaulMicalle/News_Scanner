from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from .chunking import WordChunker
from .classification.rag_training import (
    evaluate_retrieval_augmented_classifier,
    train_retrieval_augmented_distilbert,
)
from .config import AppConfig, load_config, override_config
from .models import EvidenceChunk
from .repositories import CorpusArticleRepository
from .retrieval import EmbeddingService, FaissEvidenceRetriever

_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "they", "their", "them", "have", "will", "would",
    "your", "about", "into", "just", "more", "than", "what", "when", "where", "which", "while", "because",
    "there", "these", "those", "being", "been", "only", "really", "very", "like", "some", "most", "over",
    "under", "after", "before", "across", "every", "today", "yesterday", "always", "often", "keep", "keeps",
    "doing", "does", "done", "make", "made", "says", "said", "much", "many", "good", "great", "better",
    "best", "worse", "worst", "thing", "things", "people", "country", "years", "never", "again", "still",
    "also", "even", "then", "than", "here", "there", "each", "such", "using", "used", "doesn't", "dont",
    "can't", "could", "should", "into", "onto", "across", "are", "were", "from", "into", "over", "under",
    "have", "having", "after", "before", "while", "then", "than", "what", "when", "where", "there", "here",
    "just", "very", "some", "most", "only", "with", "without", "your", "ours", "ourselves", "himself",
    "herself", "itself", "because", "about", "around", "every", "always", "never", "again",
}

_LOW_SIGNAL_MATCHES = {
    "luck", "failure", "fail", "winning", "losing", "problem", "problems", "success", "successful",
    "strong", "weak", "hard", "easy", "deal", "deals", "work", "works", "working", "time", "times",
}

_DOMAIN_MISMATCH_HINTS = {
    "gaming", "gamer", "gameplay", "tabletop", "rpg", "dnd", "dice", "campaigns", "campaign",
    "anime", "manga", "pokemon", "minecraft", "discord", "reddit", "streamer", "fandom",
    "coding", "programming", "javascript", "python", "debugging", "server", "deployment",
    "therapy", "roommate", "semester", "professor", "homework", "assignment", "cosplay",
}

_FORMAL_INSTITUTIONAL_HINTS = {
    "effectively", "legislatures", "systematically", "explicit", "participation", "democracy",
    "democratic", "minority", "minorities", "institutional", "constitutional", "judicial",
    "voting", "rights", "racial", "bias", "partisanship", "mobilize", "equal", "overreach",
    "cherish", "upcoming", "election", "elections", "citizens", "participation", "pillar",
    "gerrymander", "districts", "legislative",
}

_TRUMP_STYLE_HINTS = {
    "great", "radical", "crooked", "fake", "border", "tariffs", "america", "christmas", "democrats",
    "republicans", "winning", "strong", "crime", "market", "gdp", "law", "enforcement", "security",
    "country", "bless", "wall", "witch", "hunt", "election", "fraud",
}


@dataclass(slots=True)
class PersonaVerdict:
    subject: str
    label: str
    explanation: str
    sources: list[str]
    evidence: list[EvidenceChunk]


class PersonaCloneService:
    def __init__(
        self,
        subject: str,
        corpus_dir: str | Path,
        artifacts_dir: str | Path,
        config_path: str | Path | None = None,
        top_k: int | None = None,
    ):
        self.subject = subject
        base_config = load_config(config_path)
        self.config = override_config(
            base_config,
            corpus_dir=corpus_dir,
            artifacts_dir=artifacts_dir,
            top_k_results=top_k,
        )
        self.repository = CorpusArticleRepository(self.config.corpus_dir)
        self.chunker = WordChunker(
            chunk_size_words=self.config.chunk_size_words,
            chunk_overlap_words=self.config.chunk_overlap_words,
        )
        self.embedding_service = EmbeddingService(self.config.embedding_model)
        self.retriever = FaissEvidenceRetriever(self.config, self.embedding_service)

    def build_index(self) -> int:
        articles = self.repository.load_articles()
        chunks = self.chunker.chunk_articles(articles)
        self.retriever.build_and_save(chunks)
        return len(chunks)

    def ensure_index(self) -> None:
        if self.retriever.index is not None and self.retriever.metadata:
            return

        if self.config.index_path.exists() and self.config.metadata_path.exists():
            self.retriever.load()
            return

        self.build_index()

    def assess(self, candidate_text: str) -> PersonaVerdict:
        self.ensure_index()
        evidence = self.retriever.search(candidate_text, top_k=self.config.top_k_results)
        if not evidence:
            return PersonaVerdict(
                subject=self.subject,
                label="Unclear",
                explanation=f"No retrieved evidence was available for {self.subject}.",
                sources=[],
                evidence=[],
            )

        heuristic = self._analyze_topic_alignment(candidate_text, evidence)
        if heuristic["obvious_topic_mismatch"]:
            explanation = (
                "Unlikely\n\n"
                f"The candidate text appears to be outside {self.subject}'s typical topic and rhetorical domain. "
                f"Meaningful topic overlap with the retrieved evidence was too low "
                f"(distinctive overlap tokens: {', '.join(heuristic['distinctive_overlap'][:8]) or 'none'}). "
                "Generic similarities like 'luck', 'failure', or 'problems' were not treated as enough evidence "
                "of persona consistency."
            )
            return PersonaVerdict(
                subject=self.subject,
                label="Unlikely",
                explanation=explanation,
                sources=sorted({item.url for item in evidence}),
                evidence=evidence,
            )

        if heuristic["obvious_style_mismatch"]:
            explanation = (
                "Unlikely\n\n"
                f"The candidate text appears rhetorically misaligned with {self.subject}'s usual style. "
                f"It leans toward formal/institutional language "
                f"({', '.join(heuristic['formal_markers'][:8]) or 'none'}) without enough matching persona-style markers "
                f"({', '.join(heuristic['style_overlap'][:8]) or 'none'}). "
                "So even though the topic may overlap, the voice and framing do not match strongly enough."
            )
            return PersonaVerdict(
                subject=self.subject,
                label="Unlikely",
                explanation=explanation,
                sources=sorted({item.url for item in evidence}),
                evidence=evidence,
            )

        response_text = self._query_ollama(self._build_prompt(candidate_text, evidence, heuristic))
        label = self._extract_label(response_text)
        return PersonaVerdict(
            subject=self.subject,
            label=label,
            explanation=response_text,
            sources=sorted({item.url for item in evidence}),
            evidence=evidence,
        )

    def train_classifier(
        self,
        dataset_path: Path,
        output_root: Path,
        base_model_name_or_path: str = "distilbert-base-uncased",
        limit: int | None = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        top_k: int | None = None,
    ) -> dict[str, object]:
        return train_retrieval_augmented_distilbert(
            dataset_path=dataset_path,
            output_root=output_root,
            pipeline=self._build_training_pipeline(top_k=top_k),
            base_model_name_or_path=base_model_name_or_path,
            limit=limit,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            top_k=top_k,
        )

    def evaluate_classifier(
        self,
        model_dir: Path,
        dataset_path: Path,
        limit: int | None = None,
        top_k: int | None = None,
    ) -> dict[str, object]:
        return evaluate_retrieval_augmented_classifier(
            model_dir=model_dir,
            dataset_path=dataset_path,
            pipeline=self._build_training_pipeline(top_k=top_k),
            limit=limit,
            top_k=top_k,
        )

    def _build_training_pipeline(self, top_k: int | None = None) -> "_PersonaTrainingPipeline":
        config = override_config(self.config, top_k_results=top_k)
        return _PersonaTrainingPipeline(config, self.repository, self.chunker, self.embedding_service)

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

    def _build_prompt(
        self,
        candidate_text: str,
        evidence: list[EvidenceChunk],
        heuristic: dict[str, object],
    ) -> str:
        evidence_sections: list[str] = []
        for item in evidence:
            trimmed = item.text[: self.config.max_evidence_chars].strip()
            evidence_sections.append(
                f"[Rank {item.rank} | Score {item.score:.4f} | Source {item.url}]\n{trimmed}"
            )

        joined_evidence = "\n\n".join(evidence_sections)
        heuristic_summary = (
            f"Distinctive overlap tokens: {', '.join(heuristic['distinctive_overlap'][:12]) or 'none'}\n"
            f"Low-signal overlap tokens: {', '.join(heuristic['low_signal_overlap'][:12]) or 'none'}\n"
            f"Top evidence keywords: {', '.join(heuristic['evidence_keywords'][:12]) or 'none'}\n"
            f"Formal/institutional markers in candidate: {', '.join(heuristic['formal_markers'][:12]) or 'none'}\n"
            f"Trump-style candidate markers: {', '.join(heuristic['candidate_style_markers'][:12]) or 'none'}\n"
            f"Trump-style evidence markers: {', '.join(heuristic['evidence_style_markers'][:12]) or 'none'}\n"
            f"Style overlap markers: {', '.join(heuristic['style_overlap'][:12]) or 'none'}\n"
            f"Obvious topic mismatch: {heuristic['obvious_topic_mismatch']}\n"
            f"Obvious style mismatch: {heuristic['obvious_style_mismatch']}"
        )
        return (
            f"You are evaluating a personality clone for {self.subject}.\n"
            f"Use only the retrieved evidence about {self.subject}'s prior statements, behavior, and framing.\n"
            "Judge whether the candidate statement or action is consistent with what this person would plausibly say, do, or support.\n"
            "This is a persona-consistency task, not a fact-checking task.\n"
            "You must care about topic fit, rhetoric, tone, and recurring interests.\n"
            "Do not mark something as Likely just because it shares generic ideas like luck, failure, strength, weakness, or problems.\n"
            "If the text is about a domain this person does not typically discuss, that should count strongly against Likely.\n"
            "If the text uses formal, institutional, academic, or civic-rights language that does not resemble this person's usual voice, that should also count strongly against Likely.\n"
            "Start your answer with exactly one of these labels:\n"
            "Likely\n"
            "Unlikely\n"
            "Unclear\n\n"
            "Heuristic alignment summary:\n"
            f"{heuristic_summary}\n\n"
            "Retrieved evidence:\n"
            f"{joined_evidence}\n\n"
            "Candidate statement or action:\n"
            f"{candidate_text.strip()}\n\n"
            "After the label, briefly explain which pieces of retrieved evidence most strongly support your judgment."
        )

    def _analyze_topic_alignment(
        self,
        candidate_text: str,
        evidence: list[EvidenceChunk],
    ) -> dict[str, object]:
        candidate_tokens = self._meaningful_tokens(candidate_text)
        evidence_tokens = self._meaningful_tokens(" ".join(chunk.text for chunk in evidence))
        candidate_set = set(candidate_tokens)
        evidence_set = set(evidence_tokens)

        raw_overlap = sorted(candidate_set & evidence_set)
        distinctive_overlap = [token for token in raw_overlap if token not in _LOW_SIGNAL_MATCHES]
        low_signal_overlap = [token for token in raw_overlap if token in _LOW_SIGNAL_MATCHES]
        candidate_unique = [token for token in sorted(candidate_set - evidence_set) if token not in _LOW_SIGNAL_MATCHES]
        evidence_keywords = self._top_keywords(evidence_tokens)
        mismatch_hints = [token for token in candidate_set if token in _DOMAIN_MISMATCH_HINTS and token not in evidence_set]
        formal_markers = [token for token in sorted(candidate_set) if token in _FORMAL_INSTITUTIONAL_HINTS]
        candidate_style_markers = [token for token in sorted(candidate_set) if token in _TRUMP_STYLE_HINTS]
        evidence_style_markers = [token for token in sorted(evidence_set) if token in _TRUMP_STYLE_HINTS]
        style_overlap = [token for token in candidate_style_markers if token in evidence_style_markers]

        normalized_overlap = len(distinctive_overlap) / max(1, min(len(candidate_set), 20))
        obvious_topic_mismatch = (
            (normalized_overlap < 0.08 and len(distinctive_overlap) <= 1 and len(candidate_unique) >= 6)
            or (len(mismatch_hints) >= 2 and len(distinctive_overlap) <= 2)
        )
        obvious_style_mismatch = (
            len(formal_markers) >= 4
            and len(style_overlap) <= 1
            and len(candidate_style_markers) <= 2
        )

        return {
            "candidate_keywords": self._top_keywords(candidate_tokens),
            "evidence_keywords": evidence_keywords,
            "distinctive_overlap": distinctive_overlap,
            "low_signal_overlap": low_signal_overlap,
            "candidate_unique": candidate_unique,
            "mismatch_hints": sorted(mismatch_hints),
            "formal_markers": formal_markers,
            "candidate_style_markers": candidate_style_markers,
            "evidence_style_markers": evidence_style_markers,
            "style_overlap": style_overlap,
            "normalized_overlap": normalized_overlap,
            "obvious_topic_mismatch": obvious_topic_mismatch,
            "obvious_style_mismatch": obvious_style_mismatch,
        }

    @staticmethod
    def _extract_label(response_text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.IGNORECASE | re.DOTALL).strip()
        normalized = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned).lower()
        if normalized.startswith("likely"):
            return "Likely"
        if normalized.startswith("unlikely"):
            return "Unlikely"
        if normalized.startswith("unclear"):
            return "Unclear"
        for line in cleaned.splitlines():
            candidate = re.sub(r"^[\W_]+|[\W_]+$", "", line.strip()).lower()
            if candidate.startswith("likely"):
                return "Likely"
            if candidate.startswith("unlikely"):
                return "Unlikely"
            if candidate.startswith("unclear"):
                return "Unclear"
        return "Unclear"

    @staticmethod
    def _meaningful_tokens(text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z'’-]{3,}", text.lower())
        return [token for token in tokens if token not in _STOPWORDS]

    @staticmethod
    def _top_keywords(tokens: list[str], limit: int = 12) -> list[str]:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return [
            token
            for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]


class _PersonaTrainingPipeline:
    def __init__(
        self,
        config: AppConfig,
        repository: CorpusArticleRepository,
        chunker: WordChunker,
        embedding_service: EmbeddingService,
    ):
        self.config = config
        self.repository = repository
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.retriever = FaissEvidenceRetriever(config, embedding_service)

    def ensure_index(self) -> None:
        if self.retriever.index is not None and self.retriever.metadata:
            return

        if self.config.index_path.exists() and self.config.metadata_path.exists():
            self.retriever.load()
            return

        articles = self.repository.load_articles()
        chunks = self.chunker.chunk_articles(articles)
        self.retriever.build_and_save(chunks)
