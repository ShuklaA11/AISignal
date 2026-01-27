from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

from src.fetchers.base import BaseFetcher, RawArticle
from src.storage.models import utcnow

logger = logging.getLogger(__name__)

GITHUB_TRENDING_URL = "https://github.com/trending"
AI_KEYWORDS = {
    "ai", "ml", "llm", "machine-learning", "deep-learning", "neural",
    "transformer", "gpt", "diffusion", "nlp", "computer-vision",
    "reinforcement-learning", "language model", "embedding", "vector database",
    "rag", "inference", "fine-tun", "claude", "openai", "anthropic",
    "agent", "agentic", "chatbot", "generative", "stable-diffusion",
    "vision", "multimodal", "bert", "lora", "gguf", "ollama", "huggingface",
}


class GitHubTrendingFetcher(BaseFetcher):
    """Scrapes GitHub's trending page for AI/ML repositories."""

    @property
    def source_name(self) -> str:
        return "github"

    @property
    def source_type(self) -> str:
        return "scrape"

    async def fetch(self) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(GITHUB_TRENDING_URL, follow_redirects=True)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("article.Box-row")
        if not rows:
            logger.warning("[github] No trending rows found — GitHub may have changed their HTML structure")

        articles = []
        for row in rows:
            # Parse repo name
            h2 = row.select_one("h2 a")
            if not h2:
                continue
            repo_path = h2.get("href", "").strip("/")
            if not repo_path:
                continue

            # Append date fragment so the same repo trending on different
            # days produces a unique URL (avoids dedup killing repeat appearances)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            url = f"https://github.com/{repo_path}#trending-{today}"
            title = repo_path

            # Parse description
            p = row.select_one("p")
            description = p.get_text(strip=True) if p else ""

            # Filter for AI-related repos
            text_lower = f"{title} {description}".lower()
            if not any(kw in text_lower for kw in AI_KEYWORDS):
                continue

            # Parse stars today
            stars_today = ""
            spans = row.select("span.d-inline-block.float-sm-right")
            if spans:
                stars_today = spans[0].get_text(strip=True)

            # Parse language
            lang_span = row.select_one("[itemprop='programmingLanguage']")
            language = lang_span.get_text(strip=True) if lang_span else ""

            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    content=description,
                    published_at=utcnow(),
                    source_name="github",
                    source_type="scrape",
                    extra_metadata={
                        "stars_today": stars_today,
                        "language": language,
                    },
                )
            )
        return articles
