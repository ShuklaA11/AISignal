from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

from sqlmodel import Session

from src.config import Settings, load_settings
from src.fetchers.anthropic_blog import AnthropicBlogFetcher
from src.fetchers.arxiv_fetcher import ArxivFetcher
from src.fetchers.base import BaseFetcher, RawArticle
from src.fetchers.github_trending import GitHubTrendingFetcher
from src.fetchers.huggingface import HuggingFaceFetcher
from src.fetchers.reddit import RedditFetcher
from src.fetchers.rss import RSSFetcher
from src.fetchers.twitter import TwitterFetcher
from src.storage.database import init_db, session_scope
from src.storage.models import Article, FetchRun, utcnow
from src.storage.queries import article_exists, article_exists_by_title, get_title_fingerprints, _normalize_title, _title_fingerprint

logger = logging.getLogger(__name__)

import re

# Precompiled regex patterns for _clean_title
_RE_LATEX_FORMAT = re.compile(r"\\(?:mathcal|mathrm|mathbb|mathbf|textbf|textit|text)\{([^}]*)\}")
_RE_LATEX_BOLD = re.compile(r"\\(?:boldsymbol|bm)\{([^}]*)\}")
_RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
_RE_SUPERSCRIPT = re.compile(r"\^(.)")
_RE_SUBSCRIPT = re.compile(r"_(.)")
_RE_DOLLAR_BLOCK = re.compile(r"\$([^$]+)\$")
_RE_WHITESPACE = re.compile(r"\s+")

_LATEX_REPLACEMENTS = {
    r"\star": "*", r"\ast": "*", r"\times": "\u00d7", r"\cdot": "\u00b7",
    r"\leq": "\u2264", r"\geq": "\u2265", r"\neq": "\u2260", r"\approx": "\u2248",
    r"\infty": "\u221e", r"\pi": "\u03c0", r"\alpha": "\u03b1", r"\beta": "\u03b2",
    r"\gamma": "\u03b3", r"\delta": "\u03b4", r"\epsilon": "\u03b5", r"\lambda": "\u03bb",
    r"\mu": "\u03bc", r"\sigma": "\u03c3", r"\theta": "\u03b8", r"\omega": "\u03c9",
    r"\rightarrow": "\u2192", r"\leftarrow": "\u2190", r"\Rightarrow": "\u21d2",
    r"\sim": "~", r"\ell": "\u2113",
}


def _clean_title(title: str) -> str:
    r"""Strip LaTeX math notation and clean up article titles.

    Converts things like '$Q^\star$' -> 'Q*', '$\mathcal{O}(n)$' -> 'O(n)', etc.
    """
    if not title:
        return title

    def _latex_to_text(match: re.Match) -> str:
        tex = match.group(1)
        tex = _RE_LATEX_FORMAT.sub(r"\1", tex)
        tex = _RE_LATEX_BOLD.sub(r"\1", tex)
        for cmd, repl in _LATEX_REPLACEMENTS.items():
            tex = tex.replace(cmd, repl)
        tex = _RE_LATEX_CMD.sub("", tex)
        tex = tex.replace("{", "").replace("}", "")
        tex = _RE_SUPERSCRIPT.sub(r"\1", tex)
        tex = _RE_SUBSCRIPT.sub(r"\1", tex)
        return tex.strip()

    title = _RE_DOLLAR_BLOCK.sub(_latex_to_text, title)
    title = _RE_WHITESPACE.sub(" ", title).strip()
    return title


def build_fetchers(settings: Settings) -> list[BaseFetcher]:
    """Create all configured fetchers."""
    fetchers: list[BaseFetcher] = []

    # RSS feeds (skip anthropic_blog — handled by dedicated scraper below)
    for feed in settings.rss_feeds:
        if feed.name == "anthropic_blog":
            continue
        fetchers.append(RSSFetcher(name=feed.name, feed_url=feed.url))

    # Anthropic blog (no RSS feed — scrape /news page)
    fetchers.append(AnthropicBlogFetcher())

    # HuggingFace daily papers
    fetchers.append(HuggingFaceFetcher())

    # arXiv
    fetchers.append(
        ArxivFetcher(
            categories=settings.arxiv_categories,
            max_results=settings.arxiv_max_results,
        )
    )

    # GitHub Trending (scrapes github.com/trending, filters for AI repos)
    fetchers.append(GitHubTrendingFetcher())

    # Reddit (requires API keys)
    if settings.reddit_client_id and settings.reddit_client_secret:
        fetchers.append(
            RedditFetcher(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
                subreddits=settings.reddit_subreddits,
                min_score=settings.reddit_min_score,
            )
        )

    # Twitter (requires bearer token)
    if settings.twitter_bearer_token:
        fetchers.append(
            TwitterFetcher(
                bearer_token=settings.twitter_bearer_token,
                query=settings.twitter_query,
                max_results=settings.twitter_max_results,
            )
        )

    return fetchers


def store_articles(session: Session, raw_articles: list[RawArticle], existing_fps: set[str] | None = None) -> int:
    """Deduplicate and store raw articles. Returns count of new articles."""
    if existing_fps is None:
        existing_fps = get_title_fingerprints(session)
    new_count = 0
    for raw in raw_articles:
        if article_exists(session, raw.url):
            continue
        if article_exists_by_title(raw.title, existing_fps):
            logger.debug(f"Skipping title-duplicate: {raw.title[:80]}")
            continue

        article = Article(
            url=raw.url,
            content_hash=raw.content_hash,
            title=_clean_title(raw.title),
            author=raw.author,
            source_name=raw.source_name,
            source_type=raw.source_type,
            original_content=raw.content,
            published_at=raw.published_at,
            fetched_at=utcnow(),
            status="raw",
        )
        article.extra_metadata = raw.extra_metadata
        session.add(article)
        new_count += 1

    session.commit()
    return new_count


FETCHER_TIMEOUT = 120  # seconds per individual fetcher


async def _timed_fetch(fetcher: BaseFetcher) -> tuple[BaseFetcher, list[RawArticle], int, str | None]:
    """Fetch from a single source, returning (fetcher, articles, duration_ms, error)."""
    t0 = time.monotonic()
    try:
        articles = await asyncio.wait_for(fetcher.safe_fetch(), timeout=FETCHER_TIMEOUT)
        duration_ms = int((time.monotonic() - t0) * 1000)
        return fetcher, articles, duration_ms, None
    except asyncio.TimeoutError:
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.error(f"[{fetcher.source_name}] Timed out after {FETCHER_TIMEOUT}s")
        return fetcher, [], duration_ms, f"timeout after {FETCHER_TIMEOUT}s"
    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.error(f"[{fetcher.source_name}] Unexpected error: {e}")
        return fetcher, [], duration_ms, str(e)


async def run_ingestion(settings: Settings | None = None) -> int:
    """Fetch from all sources, deduplicate, store. Returns new article count."""
    if settings is None:
        settings = load_settings()

    init_db(settings.database_url)
    fetchers = build_fetchers(settings)

    logger.info(f"Running ingestion with {len(fetchers)} fetchers...")

    # Fetch from all sources concurrently, timing each one
    timed_results = await asyncio.gather(*[_timed_fetch(f) for f in fetchers])

    all_articles: list[RawArticle] = []
    fetch_results: list[tuple[str, int, int, str | None]] = []

    for fetcher, articles, duration_ms, error in timed_results:
        count = len(articles)
        all_articles.extend(articles)
        fetch_results.append((fetcher.source_name, count, duration_ms, error))
        if error:
            logger.error(f"[{fetcher.source_name}] error after {duration_ms}ms: {error}")
        elif count == 0:
            logger.warning(f"[{fetcher.source_name}] returned 0 articles in {duration_ms}ms — check feed health")
        else:
            logger.info(f"[{fetcher.source_name}] {count} articles in {duration_ms}ms")

    logger.info(f"Total fetched: {len(all_articles)} articles")

    # Deduplicate by URL and title fingerprint before storing (O(n) via set lookups)
    seen_urls: set[str] = set()
    seen_fps: set[str] = set()
    unique_articles: list[RawArticle] = []
    title_dupes = 0

    for article in all_articles:
        if article.url in seen_urls:
            continue

        fp = _title_fingerprint(article.title)
        if fp in seen_fps:
            title_dupes += 1
            continue

        seen_urls.add(article.url)
        seen_fps.add(fp)
        unique_articles.append(article)

    logger.info(
        f"After dedup: {len(unique_articles)} unique articles "
        f"({title_dupes} title duplicates removed)"
    )

    with session_scope(settings.database_url) as session:
        # Load existing title fingerprints once for fast DB-level dedup
        existing_fps = get_title_fingerprints(session)

        # Count new articles per source before storing
        new_by_source: dict[str, int] = {}
        for raw in unique_articles:
            if not article_exists(session, raw.url) and not article_exists_by_title(raw.title, existing_fps):
                new_by_source[raw.source_name] = new_by_source.get(raw.source_name, 0) + 1

        new_count = store_articles(session, unique_articles, existing_fps)
        logger.info(f"Stored {new_count} new articles")

        # Record fetch run metrics per source
        now = utcnow()
        for source_name, fetched_count, duration_ms, error in fetch_results:
            new_for_source = new_by_source.get(source_name, 0)
            status = "error" if error else ("empty" if fetched_count == 0 else "ok")
            run = FetchRun(
                source_name=source_name,
                fetched_at=now,
                articles_fetched=fetched_count,
                articles_new=new_for_source,
                duration_ms=duration_ms,
                error=error,
                status=status,
            )
            session.add(run)
        session.commit()

        return new_count
