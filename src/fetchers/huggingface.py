from __future__ import annotations

import logging
from datetime import datetime

import httpx
from dateutil.parser import parse as parse_date

from src.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)

DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"


class HuggingFaceFetcher(BaseFetcher):
    """Fetches daily trending papers from HuggingFace's papers API."""

    def __init__(self, limit: int = 100):
        self.limit = limit

    @property
    def source_name(self) -> str:
        return "huggingface"

    async def fetch(self) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(DAILY_PAPERS_URL, params={"limit": self.limit})
            resp.raise_for_status()

        papers = resp.json()
        articles = []

        for paper in papers:
            paper_data = paper.get("paper", {})
            paper_id = paper_data.get("id", "")
            title = paper_data.get("title", "").strip()
            if not title or not paper_id:
                continue

            url = f"https://huggingface.co/papers/{paper_id}"

            published = None
            if paper_data.get("publishedAt"):
                try:
                    published = parse_date(paper_data["publishedAt"])
                except (ValueError, TypeError):
                    pass

            authors = paper_data.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:3]) if authors else None

            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    content=paper_data.get("summary", ""),
                    author=author_str,
                    published_at=published,
                    source_name="huggingface",
                    source_type="api",
                    extra_metadata={
                        "upvotes": paper.get("numUpvotes", 0),
                        "paper_id": paper_id,
                        "arxiv_id": paper_data.get("arxivId"),
                    },
                )
            )
        return articles
