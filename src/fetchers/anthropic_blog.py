"""Anthropic blog fetcher — scrapes anthropic.com/news since no RSS feed exists."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

import httpx
from dateutil.parser import parse as parse_date

from src.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)

ANTHROPIC_NEWS_URL = "https://www.anthropic.com/news"


class AnthropicBlogFetcher(BaseFetcher):
    """Scrapes Anthropic's /news page for blog posts.

    Anthropic's site is a Next.js/Sanity CMS app that embeds all article
    metadata (title, slug, summary, publishedOn, subjects) inside a React
    Server Component payload in a <script> tag.  We extract that data
    rather than relying on an RSS feed (which Anthropic does not provide).
    """

    @property
    def source_name(self) -> str:
        return "anthropic_blog"

    @property
    def source_type(self) -> str:
        return "scrape"

    async def fetch(self) -> list[RawArticle]:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(
                ANTHROPIC_NEWS_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) aisignal/1.0",
                },
            )
            resp.raise_for_status()

        html = resp.text

        # Strategy 1: Find the large RSC script containing Sanity CMS article data.
        # We use multiple heuristics to be resilient to site changes.
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)
        rsc_payload = ""
        for script in scripts:
            if "_type" in script and "publishedOn" in script and "slug" in script:
                rsc_payload = script
                break

        if not rsc_payload:
            # Strategy 2: Fall back to HTML link scraping as a last resort
            logger.warning("[anthropic_blog] RSC payload not found, falling back to HTML link scraping")
            return self._fallback_html_scrape(html)

        # The RSC payload is: self.__next_f.push([1,"<json-escaped-string>"])
        # Strip the wrapper and JSON-parse the inner string to get readable text.
        inner = rsc_payload.replace("self.__next_f.push([1,", "", 1)
        if inner.endswith("])"):
            inner = inner[:-2]
        try:
            unescaped = json.loads(inner)
        except (json.JSONDecodeError, ValueError):
            logger.warning("[anthropic_blog] Failed to JSON-parse RSC payload")
            return []

        # Split on "_type":"post" boundaries to isolate each article chunk
        parts = re.split(r'"_type":"post"', unescaped)

        articles: list[RawArticle] = []
        seen_slugs: set[str] = set()

        for part in parts[1:]:  # first chunk is before any post
            slug_m = re.search(
                r'"slug":\{"_type":"slug","current":"([^"]+)"\}', part
            )
            title_m = re.search(r'"title":"([^"]+)"', part)
            if not slug_m or not title_m:
                continue

            slug = slug_m.group(1)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            title = title_m.group(1)
            url = f"https://www.anthropic.com/news/{slug}"

            # Summary
            summary_m = re.search(r'"summary":"([^"]+)"', part)
            summary = summary_m.group(1) if summary_m else ""

            # Published date (Sanity uses "publishedOn")
            published = None
            pub_m = re.search(r'"publishedOn":"([^"]+)"', part)
            if pub_m:
                try:
                    published = parse_date(pub_m.group(1))
                except (ValueError, TypeError):
                    pass

            # Tags / subjects
            tags = re.findall(r'"label":"([^"]+)"', part[:600])

            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    content=summary,
                    author="Anthropic",
                    published_at=published,
                    source_name="anthropic_blog",
                    source_type="scrape",
                    extra_metadata={"tags": tags},
                )
            )

        logger.info(f"[anthropic_blog] Scraped {len(articles)} articles from /news")
        return articles

    def _fallback_html_scrape(self, html: str) -> list[RawArticle]:
        """Fallback: extract article links from HTML when RSC parsing fails."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        articles: list[RawArticle] = []
        seen: set[str] = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if not href.startswith("/news/") or href == "/news/" or href == "/news":
                continue
            slug = href.replace("/news/", "").strip("/")
            if not slug or slug in seen:
                continue
            seen.add(slug)

            title = link.get_text(strip=True) or slug.replace("-", " ").title()
            url = f"https://www.anthropic.com/news/{slug}"

            articles.append(
                RawArticle(
                    url=url,
                    title=title,
                    content="",
                    author="Anthropic",
                    published_at=None,
                    source_name="anthropic_blog",
                    source_type="scrape",
                    extra_metadata={"fallback_scrape": True},
                )
            )

        logger.info(f"[anthropic_blog] Fallback scrape found {len(articles)} article links")
        return articles
