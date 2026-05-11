"""Curated GitHub releases fetcher.

Polls the GitHub Releases API for a configured list of AI/ML tooling repos
(vLLM, llama.cpp, transformers, etc.). Each release becomes a RawArticle so
users see "what shipped" signal in the Builder section of the digest.

Unauthenticated GitHub API allows 60 requests/hour, which covers ~15 repos
once per scheduled fetch comfortably. Set GITHUB_TOKEN in the environment
for the authenticated 5000/hr limit if you scale the list.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from dateutil.parser import parse as parse_date

from src.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)


GITHUB_API_BASE = "https://api.github.com"
DEFAULT_LOOKBACK_DAYS = 30
RELEASES_PER_REPO = 5  # only the most recent N releases per repo per fetch


class GitHubReleasesFetcher(BaseFetcher):
    """Fetches recent releases for a curated list of AI/ML repos.

    Tagged as a high-signal Builder source: a release is a stronger
    "this just shipped" signal than trending, with a real changelog.
    """

    section = "builder"
    audience_tags = ("industry", "enthusiast", "researcher")
    quality_weight = 1.5

    def __init__(
        self,
        repos: list[str],
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        releases_per_repo: int = RELEASES_PER_REPO,
        token: str | None = None,
    ):
        self.repos = [r for r in repos if "/" in r]
        self.lookback_days = lookback_days
        self.releases_per_repo = releases_per_repo
        self.token = token or os.environ.get("GITHUB_TOKEN", "")

    @property
    def source_name(self) -> str:
        return "github_releases"

    @property
    def source_type(self) -> str:
        return "api"

    async def fetch(self) -> list[RawArticle]:
        if not self.repos:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        articles: list[RawArticle] = []

        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            for repo in self.repos:
                articles.extend(
                    await self._fetch_repo(client, repo, cutoff)
                )

        logger.info(
            f"[github_releases] fetched {len(articles)} releases across {len(self.repos)} repos"
        )
        return articles

    async def _fetch_repo(
        self, client: httpx.AsyncClient, repo: str, cutoff: datetime
    ) -> list[RawArticle]:
        """Fetch recent releases for a single owner/name repo. Errors per repo
        are logged but don't kill the whole fetch — one missing repo shouldn't
        blackhole the others."""
        url = f"{GITHUB_API_BASE}/repos/{repo}/releases"
        try:
            resp = await client.get(url, params={"per_page": self.releases_per_repo})
            if resp.status_code == 404:
                logger.warning(f"[github_releases] {repo}: 404 (renamed or private?)")
                return []
            resp.raise_for_status()
            releases = resp.json()
        except Exception as e:
            logger.warning(f"[github_releases] {repo}: {e}")
            return []

        out: list[RawArticle] = []
        for r in releases:
            if r.get("draft") or r.get("prerelease"):
                continue
            published_raw = r.get("published_at") or r.get("created_at")
            if not published_raw:
                continue
            try:
                published = parse_date(published_raw)
            except (ValueError, TypeError):
                continue
            if published < cutoff:
                continue

            tag = r.get("tag_name") or ""
            name = r.get("name") or tag or "(untitled release)"
            title = f"{repo} {tag}: {name}" if tag and tag not in name else f"{repo}: {name}"

            body = (r.get("body") or "").strip()
            if not body:
                body = title

            out.append(
                RawArticle(
                    url=r.get("html_url") or f"https://github.com/{repo}/releases",
                    title=title,
                    content=body,
                    published_at=published,
                    source_name="github_releases",
                    source_type="api",
                    extra_metadata={
                        "repo": repo,
                        "tag": tag,
                    },
                )
            )
        return out
