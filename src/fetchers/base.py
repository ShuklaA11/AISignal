from __future__ import annotations

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

MAX_FETCH_RETRIES = 3


@dataclass
class RawArticle:
    """Normalized article from any source."""

    url: str
    title: str
    content: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    source_name: str = ""
    source_type: str = ""  # rss | api | scrape
    extra_metadata: dict = field(default_factory=dict)

    # Section taxonomy fields. Fetchers populate these so the section/quality
    # signal is set at ingestion time and flows through scoring + rendering.
    # See src/sections.py for valid section values.
    section: str | None = None
    audience_tags: list[str] = field(default_factory=list)
    quality_weight: float = 1.0

    @property
    def content_hash(self) -> str:
        """SHA256 of normalized title + URL for deduplication."""
        raw = f"{self.title.strip().lower()}|{self.url.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()


class BaseFetcher(ABC):
    """All fetchers implement this interface."""

    @abstractmethod
    async def fetch(self) -> list[RawArticle]:
        """Fetch new articles from the source."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        ...

    @property
    def source_type(self) -> str:
        return "api"

    async def safe_fetch(self) -> list[RawArticle]:
        """Fetch with retry and error handling — never crashes the pipeline."""
        last_error = None
        for attempt in range(1, MAX_FETCH_RETRIES + 1):
            try:
                articles = await self.fetch()
                logger.info(f"[{self.source_name}] Fetched {len(articles)} articles")
                return articles
            except Exception as e:
                last_error = e
                logger.warning(f"[{self.source_name}] Fetch attempt {attempt}/{MAX_FETCH_RETRIES} failed: {e}")
                if attempt < MAX_FETCH_RETRIES:
                    await asyncio.sleep(2 ** attempt)  # 2s, 4s backoff

        logger.error(f"[{self.source_name}] All {MAX_FETCH_RETRIES} fetch attempts failed. Last error: {last_error}")
        return []
