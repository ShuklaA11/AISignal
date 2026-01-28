from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import tweepy

from src.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)


class TwitterFetcher(BaseFetcher):
    """Fetches recent tweets matching an AI-related query via the Twitter API v2."""

    def __init__(self, bearer_token: str, query: str = "", max_results: int = 50):
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.query = query or '(AI OR "machine learning" OR LLM) lang:en -is:retweet'
        self.max_results = min(max_results, 100)  # API limit

    @property
    def source_name(self) -> str:
        return "twitter"

    async def fetch(self) -> list[RawArticle]:
        response = await asyncio.to_thread(
            self.client.search_recent_tweets,
            query=self.query,
            max_results=self.max_results,
            tweet_fields=["created_at", "public_metrics", "author_id"],
            user_fields=["username", "name"],
            expansions=["author_id"],
        )

        if not response.data:
            return []

        # Build user lookup
        users = {}
        if response.includes and "users" in response.includes:
            for user in response.includes["users"]:
                users[user.id] = user

        articles = []
        for tweet in response.data:
            metrics = tweet.public_metrics or {}
            engagement = metrics.get("like_count", 0) + metrics.get("retweet_count", 0)

            user = users.get(tweet.author_id)
            author = f"@{user.username}" if user else None

            articles.append(
                RawArticle(
                    url=f"https://x.com/i/status/{tweet.id}",
                    title=tweet.text[:120].strip(),
                    content=tweet.text,
                    author=author,
                    published_at=tweet.created_at,
                    source_name="twitter",
                    source_type="api",
                    extra_metadata={
                        "likes": metrics.get("like_count", 0),
                        "retweets": metrics.get("retweet_count", 0),
                        "engagement": engagement,
                        "tweet_id": str(tweet.id),
                    },
                )
            )
        return articles
