from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import praw

from src.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)


class RedditFetcher(BaseFetcher):
    """Fetches top posts from configured subreddits via the Reddit API (PRAW)."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        subreddits: list[str] | None = None,
        min_score: int = 10,
        limit: int = 25,
    ):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        self.subreddits = subreddits or ["MachineLearning", "artificial", "LocalLLaMA"]
        self.min_score = min_score
        self.limit = limit

    @property
    def source_name(self) -> str:
        return "reddit"

    async def fetch(self) -> list[RawArticle]:
        return await asyncio.to_thread(self._fetch_sync)

    def _fetch_sync(self) -> list[RawArticle]:
        multi = "+".join(self.subreddits)
        subreddit = self.reddit.subreddit(multi)

        articles = []
        for post in subreddit.hot(limit=self.limit):
            if post.score < self.min_score:
                continue

            url = post.url
            if post.is_self:
                url = f"https://reddit.com{post.permalink}"

            content = post.selftext[:2000] if post.selftext else ""

            articles.append(
                RawArticle(
                    url=url,
                    title=post.title.strip(),
                    content=content,
                    author=str(post.author) if post.author else None,
                    published_at=datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                    source_name=f"r/{post.subreddit.display_name}",
                    source_type="api",
                    extra_metadata={
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "subreddit": post.subreddit.display_name,
                        "permalink": post.permalink,
                    },
                )
            )
        return articles
