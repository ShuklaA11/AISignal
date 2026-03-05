from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import arxiv

from src.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)


class ArxivFetcher(BaseFetcher):
    """Fetches recent papers from arXiv using the arxiv Python library."""

    def __init__(self, categories: list[str] | None = None, max_results: int = 100):
        self.categories = categories or ["cs.AI", "cs.CL", "cs.CV", "cs.LG"]
        self.max_results = max_results

    @property
    def source_name(self) -> str:
        return "arxiv"

    async def fetch(self) -> list[RawArticle]:
        return await asyncio.to_thread(self._fetch_sync)

    def _fetch_sync(self) -> list[RawArticle]:
        # Build date range: papers submitted in the last 3 days
        # arXiv batches submissions and has weekend gaps, so 3 days
        # ensures we catch everything even after weekends/holidays
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=3)).strftime("%Y%m%d")
        end = now.strftime("%Y%m%d")

        cat_query = " OR ".join(f"cat:{cat}" for cat in self.categories)
        query = f"({cat_query}) AND submittedDate:[{start}0000 TO {end}2359]"

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        articles = []
        for result in client.results(search):
            authors = [a.name for a in result.authors[:5]]
            categories = list(result.categories)

            articles.append(
                RawArticle(
                    url=result.entry_id,
                    title=result.title.strip(),
                    content=result.summary.strip(),
                    author=", ".join(authors),
                    published_at=result.published,
                    source_name="arxiv",
                    source_type="api",
                    extra_metadata={
                        "categories": categories,
                        "pdf_url": result.pdf_url,
                        "primary_category": result.primary_category,
                    },
                )
            )

        logger.info(f"[arxiv] Date range {start}-{end}, fetched {len(articles)} papers")
        return articles
