from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime

import feedparser
from dateutil.parser import parse as parse_date

from src.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)


# Feeds from dedicated AI labs/companies — all content is relevant by definition
_AI_NATIVE_FEEDS = {
    "openai_blog", "anthropic_blog", "deepmind_blog",
    "huggingface_blog", "meta_ai_blog",
}

# Keywords (lowercased) that signal AI/ML relevance in title or content
_AI_KEYWORDS = {
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "llm", "large language model", "gpt", "chatgpt",
    "openai", "anthropic", "claude", "gemini", "copilot", "midjourney",
    "stable diffusion", "transformer", "diffusion model", "generative",
    "nlp", "natural language", "computer vision", "reinforcement learning",
    "robotics", "autonomous", "self-driving", "deepfake", "ai safety",
    "alignment", "agi", "superintelligence", "foundation model",
    "fine-tuning", "fine tuning", "rag", "retrieval augmented",
    "embedding", "vector database", "prompt engineering", "inference",
    "training data", "gpu", "nvidia", "tensor", "pytorch", "tensorflow",
    "hugging face", "huggingface", "ollama", "llama", "mistral",
    "multimodal", "text-to-image", "text-to-video", "speech recognition",
    "ai agent", "agentic", "ai model", "ml model", "dataset",
}


def _is_ai_relevant(title: str, content: str) -> bool:
    """Check if an article is AI/ML-related by keyword matching."""
    text = (title + " " + content).lower()
    return any(kw in text for kw in _AI_KEYWORDS)


class RSSFetcher(BaseFetcher):
    """Fetches articles from an RSS/Atom feed URL using feedparser."""

    def __init__(self, name: str, feed_url: str):
        self._name = name
        self.feed_url = feed_url

    @property
    def source_name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return "rss"

    async def fetch(self) -> list[RawArticle]:
        # Use a browser-like user agent — some feeds (deepmind, venturebeat)
        # return 403 or empty responses with the default feedparser UA
        ua = "Mozilla/5.0 (compatible; AINewsletterBot/1.0)"
        feed = await asyncio.to_thread(feedparser.parse, self.feed_url, agent=ua)

        if feed.bozo and not feed.entries:
            logger.warning(f"[{self._name}] Feed parse error for {self.feed_url}: {feed.bozo_exception}")
            return []

        if not feed.entries:
            logger.warning(f"[{self._name}] Feed returned 0 entries from {self.feed_url}")
            return []

        articles = []
        for entry in feed.entries:
            published = None
            for date_field in ("published", "updated", "created"):
                raw = entry.get(date_field)
                if raw:
                    try:
                        published = parse_date(raw)
                        break
                    except (ValueError, TypeError):
                        continue

            url = entry.get("link", "").strip()
            title = html.unescape(entry.get("title", "")).strip()
            if not url or not title:
                continue

            content = entry.get("summary", "") or ""
            # Some feeds put full content in content[0]
            if hasattr(entry, "content") and entry.content:
                content = entry.content[0].get("value", content)

            tags = [t.get("term", "") for t in entry.get("tags", []) if t.get("term")]

            # Skip non-AI articles from general news feeds
            if self._name not in _AI_NATIVE_FEEDS and not _is_ai_relevant(title, content):
                logger.debug(f"[{self._name}] Skipped non-AI article: {title[:60]}")
                continue

            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    content=content,
                    author=entry.get("author"),
                    published_at=published,
                    source_name=self._name,
                    source_type="rss",
                    extra_metadata={"tags": tags},
                )
            )
        return articles
