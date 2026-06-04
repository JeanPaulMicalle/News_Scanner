from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import urlparse

from .models import Article


class CorpusArticleRepository:
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir

    def load_articles(self) -> list[Article]:
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")

        articles: list[Article] = []
        for path in sorted(self.corpus_dir.rglob("*")):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            if suffix == ".json":
                articles.extend(self._load_json_articles(path))
            elif suffix == ".txt":
                articles.extend(self._load_text_articles(path))
            elif suffix == ".csv":
                articles.extend(self._load_csv_articles(path))

        if not articles:
            raise RuntimeError(f"No valid corpus files found in {self.corpus_dir}")

        return articles

    def _load_json_articles(self, path: Path) -> list[Article]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        body = (data.get("articleBody") or data.get("body") or data.get("text") or "").strip()
        url = (data.get("url") or f"local://{path.relative_to(self.corpus_dir).as_posix()}").strip()
        if not body:
            return []

        title = data.get("headline") or data.get("title") or data.get("name") or path.stem
        publisher = data.get("publisher")
        if isinstance(publisher, dict):
            source = publisher.get("name", "unknown")
        else:
            source = urlparse(url).netloc or path.parent.name or "unknown"

        return [
            Article(
                article_id=path.stem,
                title=title.strip(),
                url=url,
                body=body,
                source=source,
                metadata={"path": str(path)},
            )
        ]

    def _load_text_articles(self, path: Path) -> list[Article]:
        body = path.read_text(encoding="utf-8").strip()
        if not body:
            return []

        return [
            Article(
                article_id=path.stem,
                title=path.stem.replace("_", " ").strip(),
                url=f"local://{path.relative_to(self.corpus_dir).as_posix()}",
                body=body,
                source=path.parent.name or "local_text",
                metadata={"path": str(path)},
            )
        ]

    def _load_csv_articles(self, path: Path) -> list[Article]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []

            text_key = self._select_field(
                reader.fieldnames,
                ["text", "content", "body", "articleBody", "tweet text", "tweet_text"],
            )
            if text_key is None:
                return []

            title_key = self._select_field(
                reader.fieldnames,
                ["title", "headline", "subject", "name"],
            )
            url_key = self._select_field(reader.fieldnames, ["url", "link", "tweet url", "tweet_url"])
            source_key = self._select_field(reader.fieldnames, ["source", "publisher", "id", "account", "username"])
            id_key = self._select_field(reader.fieldnames, ["article_id", "tweet id", "tweet_id"])
            time_key = self._select_field(reader.fieldnames, ["time", "created_at", "date", "timestamp"])

            articles: list[Article] = []
            for index, row in enumerate(reader, start=1):
                body = (row.get(text_key) or "").strip()
                if not body:
                    continue

                article_id = (row.get(id_key) or f"{path.stem}_{index}").strip() if id_key else f"{path.stem}_{index}"
                timestamp = (row.get(time_key) or "").strip() if time_key else ""
                if title_key:
                    title = (row.get(title_key) or article_id).strip()
                elif timestamp:
                    title = f"{path.stem} {timestamp}"
                else:
                    title = article_id
                url = (row.get(url_key) or f"csv://{path.name}#row-{index}").strip() if url_key else f"csv://{path.name}#row-{index}"
                source = (row.get(source_key) or path.stem).strip() if source_key else path.stem

                articles.append(
                    Article(
                        article_id=article_id,
                        title=title,
                        url=url,
                        body=body,
                        source=source,
                        metadata={"path": str(path), "row_number": index, "timestamp": timestamp},
                    )
                )

            return articles

    @staticmethod
    def _select_field(fieldnames: list[str], candidates: list[str]) -> str | None:
        normalized = {name.strip().lower(): name for name in fieldnames}
        for candidate in candidates:
            normalized_candidate = candidate.strip().lower()
            if normalized_candidate in normalized:
                return normalized[normalized_candidate]
        return None


class JsonArticleRepository(CorpusArticleRepository):
    pass
