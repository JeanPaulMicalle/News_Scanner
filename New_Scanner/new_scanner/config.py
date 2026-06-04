from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(slots=True)
class AppConfig:
    embedding_model: str
    sentiment_model: str
    ollama_url: str
    ollama_model_name: str
    corpus_dir: Path
    artifacts_dir: Path
    index_file: str
    metadata_file: str
    top_k_results: int
    chunk_size_words: int
    chunk_overlap_words: int
    max_evidence_chars: int

    @property
    def index_path(self) -> Path:
        return self.artifacts_dir / self.index_file

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / self.metadata_file


def override_config(
    config: AppConfig,
    *,
    corpus_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    top_k_results: int | None = None,
) -> AppConfig:
    return replace(
        config,
        corpus_dir=Path(corpus_dir).resolve() if corpus_dir is not None else config.corpus_dir,
        artifacts_dir=Path(artifacts_dir).resolve() if artifacts_dir is not None else config.artifacts_dir,
        top_k_results=int(top_k_results) if top_k_results is not None else config.top_k_results,
    )


def load_config(config_path: str | Path | None = None) -> AppConfig:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        config_path = Path(config_path).resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    base_dir = config_path.parent
    return AppConfig(
        embedding_model=raw["embedding_model"],
        sentiment_model=raw["sentiment_model"],
        ollama_url=raw["ollama_url"],
        ollama_model_name=raw["ollama_model_name"],
        corpus_dir=(base_dir / raw["corpus_dir"]).resolve(),
        artifacts_dir=(base_dir / raw["artifacts_dir"]).resolve(),
        index_file=raw["index_file"],
        metadata_file=raw["metadata_file"],
        top_k_results=int(raw["top_k_results"]),
        chunk_size_words=int(raw["chunk_size_words"]),
        chunk_overlap_words=int(raw["chunk_overlap_words"]),
        max_evidence_chars=int(raw["max_evidence_chars"]),
    )
