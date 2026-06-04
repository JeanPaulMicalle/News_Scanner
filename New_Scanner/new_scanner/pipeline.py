from __future__ import annotations

from .chunking import WordChunker
from .config import AppConfig, load_config
from .models import Claim, Verdict
from .repositories import JsonArticleRepository
from .retrieval import EmbeddingService, FaissEvidenceRetriever
from .verifier import LlmOnlyVerifier, OllamaVerifier, SentimentService


class ResearchBaselinePipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.repository = JsonArticleRepository(config.corpus_dir)
        self.chunker = WordChunker(
            chunk_size_words=config.chunk_size_words,
            chunk_overlap_words=config.chunk_overlap_words,
        )
        self.embedding_service = EmbeddingService(config.embedding_model)
        self.retriever = FaissEvidenceRetriever(config, self.embedding_service)
        self.sentiment_service = SentimentService(config.sentiment_model)
        self.rag_verifier = OllamaVerifier(config, self.sentiment_service)
        self.llm_only_verifier = LlmOnlyVerifier(config, self.sentiment_service)

    @classmethod
    def from_config(cls, config_path: str | None = None) -> "ResearchBaselinePipeline":
        return cls(load_config(config_path))

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

        articles = self.repository.load_articles()
        chunks = self.chunker.chunk_articles(articles)
        self.retriever.load_or_build(chunks)

    def verify_claim(self, claim_text: str, mode: str = "rag") -> Verdict:
        claim = Claim(text=claim_text)
        if mode == "llm_only":
            return self.llm_only_verifier.verify(claim, [])
        if mode == "rag":
            self.ensure_index()
            evidence = self.retriever.search(claim_text, top_k=self.config.top_k_results)
            return self.rag_verifier.verify(claim, evidence)
        raise ValueError(f"Unsupported verification mode: {mode}")
