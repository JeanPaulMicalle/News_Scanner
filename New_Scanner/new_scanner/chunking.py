from __future__ import annotations

from .models import Article, EvidenceChunk


class WordChunker:
    def __init__(self, chunk_size_words: int, chunk_overlap_words: int):
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if chunk_overlap_words < 0:
            raise ValueError("chunk_overlap_words cannot be negative")
        if chunk_overlap_words >= chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

    def chunk_articles(self, articles: list[Article]) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        for article in articles:
            chunks.extend(self.chunk_article(article))
        return chunks

    def chunk_article(self, article: Article) -> list[EvidenceChunk]:
        words = article.body.split()
        if not words:
            return []

        chunks: list[EvidenceChunk] = []
        step = self.chunk_size_words - self.chunk_overlap_words
        for start in range(0, len(words), step):
            end = start + self.chunk_size_words
            chunk_words = words[start:end]
            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words).strip()
            chunk_index = len(chunks)
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{article.article_id}::chunk::{chunk_index}",
                    article_id=article.article_id,
                    article_title=article.title,
                    source=article.source,
                    url=article.url,
                    text=chunk_text,
                    metadata={
                        "start_word": start,
                        "end_word": min(end, len(words)),
                    },
                )
            )

        return chunks
