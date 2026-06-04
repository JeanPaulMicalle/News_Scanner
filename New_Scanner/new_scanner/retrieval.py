from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import AppConfig
from .models import EvidenceChunk
from .vector_database.faiss_handler import (
    add_embeddings_to_index,
    create_faiss_index,
    load_faiss_index,
    save_faiss_index,
    search_faiss_index,
)
from .vector_database.metadata_store import load_chunk_metadata, save_chunk_metadata


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    local_files_only=True,
                )
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return np.asarray(embeddings, dtype="float32")


class FaissEvidenceRetriever:
    def __init__(self, config: AppConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self.index = None
        self.metadata: list[EvidenceChunk] = []

    def build_and_save(self, chunks: list[EvidenceChunk]) -> None:
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        embeddings = self.embedding_service.encode([chunk.text for chunk in chunks])
        dimension = embeddings.shape[1]
        index = create_faiss_index(dimension)
        add_embeddings_to_index(index, embeddings)

        save_faiss_index(index, str(self.config.index_path))
        save_chunk_metadata(chunks, self.config.metadata_path)

        self.index = index
        self.metadata = chunks

    def load(self) -> None:
        if not self.config.index_path.exists() or not self.config.metadata_path.exists():
            raise FileNotFoundError("Index artifacts do not exist yet. Run the index command first.")

        self.index = load_faiss_index(str(self.config.index_path))
        self.metadata = load_chunk_metadata(self.config.metadata_path)

    def load_or_build(self, chunks: list[EvidenceChunk]) -> None:
        if self.config.index_path.exists() and self.config.metadata_path.exists():
            self.load()
        else:
            self.build_and_save(chunks)

    def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        if self.index is None:
            raise RuntimeError("Retriever is not initialized. Load or build the index first.")

        query_embedding = self.embedding_service.encode([query])[0]
        distances, indices = search_faiss_index(self.index, query_embedding, top_k)

        results: list[EvidenceChunk] = []
        for rank, (distance, idx) in enumerate(zip(distances, indices), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue

            base = self.metadata[idx]
            results.append(
                EvidenceChunk(
                    chunk_id=base.chunk_id,
                    article_id=base.article_id,
                    article_title=base.article_title,
                    source=base.source,
                    url=base.url,
                    text=base.text,
                    rank=rank,
                    score=float(distance),
                    metadata=base.metadata,
                )
            )

        return results
