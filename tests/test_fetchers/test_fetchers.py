"""Tests for fetcher modules: base, RSS, arXiv, GitHub, HuggingFace, Anthropic blog.

Covers:
- BaseFetcher.safe_fetch() retry/backoff logic
- RawArticle.content_hash deduplication
- RSSFetcher AI-keyword filtering and date parsing
- ArxivFetcher query construction and result mapping
- GitHubTrendingFetcher HTML scraping and AI filtering
- HuggingFaceFetcher API response mapping
- AnthropicBlogFetcher RSC parsing and fallback scraping
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.fetchers.base import BaseFetcher, RawArticle, MAX_FETCH_RETRIES


# ---------------------------------------------------------------------------
# RawArticle
# ---------------------------------------------------------------------------

class TestRawArticle:
    def test_content_hash_deterministic(self):
        a = RawArticle(url="https://example.com/1", title="Hello World")
        b = RawArticle(url="https://example.com/1", title="Hello World")
        assert a.content_hash == b.content_hash

    def test_content_hash_case_insensitive(self):
        a = RawArticle(url="https://example.com/1", title="Hello World")
        b = RawArticle(url="https://EXAMPLE.com/1", title="hello world")
        assert a.content_hash == b.content_hash

    def test_content_hash_differs_for_different_titles(self):
        a = RawArticle(url="https://example.com/1", title="Article A")
        b = RawArticle(url="https://example.com/1", title="Article B")
        assert a.content_hash != b.content_hash

    def test_content_hash_differs_for_different_urls(self):
        a = RawArticle(url="https://example.com/1", title="Same Title")
        b = RawArticle(url="https://example.com/2", title="Same Title")
        assert a.content_hash != b.content_hash

    def test_content_hash_strips_whitespace(self):
        a = RawArticle(url="https://example.com/1", title="  Hello  ")
        b = RawArticle(url="  https://example.com/1  ", title="Hello")
        assert a.content_hash == b.content_hash


# ---------------------------------------------------------------------------
# BaseFetcher.safe_fetch()
# ---------------------------------------------------------------------------

class _FailingFetcher(BaseFetcher):
    """A fetcher that fails N times then succeeds."""

    def __init__(self, fail_count: int):
        self.fail_count = fail_count
        self.attempts = 0

    @property
    def source_name(self) -> str:
        return "test_failing"

    async def fetch(self):
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise ConnectionError(f"Attempt {self.attempts} failed")
        return [RawArticle(url="https://ok.com", title="Success")]


class _AlwaysFailFetcher(BaseFetcher):
    def __init__(self):
        self.attempts = 0

    @property
    def source_name(self) -> str:
        return "test_always_fail"

    async def fetch(self):
        self.attempts += 1
        raise RuntimeError("permanent failure")


class TestSafeFetch:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        fetcher = _FailingFetcher(fail_count=0)
        result = await fetcher.safe_fetch()
        assert len(result) == 1
        assert result[0].title == "Success"
        assert fetcher.attempts == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        fetcher = _FailingFetcher(fail_count=2)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetcher.safe_fetch()
        assert len(result) == 1
        assert fetcher.attempts == 3

    @pytest.mark.asyncio
    async def test_returns_empty_after_max_retries(self):
        fetcher = _AlwaysFailFetcher()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetcher.safe_fetch()
        assert result == []
        assert fetcher.attempts == MAX_FETCH_RETRIES

    @pytest.mark.asyncio
    async def test_backoff_delays_are_exponential(self):
        fetcher = _AlwaysFailFetcher()
        sleep_calls = []

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await fetcher.safe_fetch()

        # Should have called sleep for attempts 1 and 2 (not after the last attempt)
        assert sleep_calls == [2, 4]  # 2^1, 2^2


# ---------------------------------------------------------------------------
# RSSFetcher
# ---------------------------------------------------------------------------

class TestRSSFetcher:
    def _make_entry(self, title, summary="", link="https://example.com/1",
                    tags=None, published=None):
        entry = MagicMock()
        entry.get = lambda k, d="": {
            "title": title,
            "summary": summary,
            "link": link,
            "author": "Test Author",
        }.get(k, d)
        entry.content = []
        entry.tags = [{"term": t} for t in (tags or [])]

        if published:
            entry.published = published
        else:
            # Simulate missing date fields
            original_get = entry.get
            entry.get = lambda k, d="": published if k in ("published", "updated", "created") else original_get(k, d)

        return entry

    def _make_feed(self, entries, bozo=False, bozo_exception=None):
        feed = MagicMock()
        feed.entries = entries
        feed.bozo = bozo
        feed.bozo_exception = bozo_exception
        return feed

    @pytest.mark.asyncio
    async def test_filters_non_ai_articles_from_general_feeds(self):
        from src.fetchers.rss import RSSFetcher

        entries = [
            self._make_entry("New AI Model Breaks Records", "A large language model..."),
            self._make_entry("Stock Market Update", "The market rose 2% today"),
        ]
        feed = self._make_feed(entries)

        with patch("feedparser.parse", return_value=feed):
            fetcher = RSSFetcher("techcrunch", "https://feed.example.com")
            result = await fetcher.fetch()

        assert len(result) == 1
        assert "AI" in result[0].title

    @pytest.mark.asyncio
    async def test_ai_native_feeds_skip_keyword_filter(self):
        from src.fetchers.rss import RSSFetcher

        entries = [
            self._make_entry("Introducing Our New Research", "We are excited..."),
        ]
        feed = self._make_feed(entries)

        with patch("feedparser.parse", return_value=feed):
            fetcher = RSSFetcher("openai_blog", "https://feed.example.com")
            result = await fetcher.fetch()

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_skips_entries_without_url_or_title(self):
        from src.fetchers.rss import RSSFetcher

        entries = [
            self._make_entry("", "content", link="https://example.com/1"),  # empty title
            self._make_entry("Title", "content", link=""),  # empty link
        ]
        feed = self._make_feed(entries)

        with patch("feedparser.parse", return_value=feed):
            fetcher = RSSFetcher("openai_blog", "https://feed.example.com")
            result = await fetcher.fetch()

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_handles_bozo_feed_with_no_entries(self):
        from src.fetchers.rss import RSSFetcher

        feed = self._make_feed([], bozo=True, bozo_exception=Exception("malformed XML"))

        with patch("feedparser.parse", return_value=feed):
            fetcher = RSSFetcher("test_feed", "https://feed.example.com")
            result = await fetcher.fetch()

        assert result == []

    @pytest.mark.asyncio
    async def test_unescapes_html_in_title(self):
        from src.fetchers.rss import RSSFetcher

        entries = [
            self._make_entry("AI &amp; Machine Learning", "deep learning transformer"),
        ]
        feed = self._make_feed(entries)

        with patch("feedparser.parse", return_value=feed):
            fetcher = RSSFetcher("openai_blog", "https://feed.example.com")
            result = await fetcher.fetch()

        assert result[0].title == "AI & Machine Learning"

    @pytest.mark.asyncio
    async def test_source_type_is_rss(self):
        from src.fetchers.rss import RSSFetcher
        fetcher = RSSFetcher("test", "https://example.com")
        assert fetcher.source_type == "rss"


# ---------------------------------------------------------------------------
# ArxivFetcher
# ---------------------------------------------------------------------------

class TestArxivFetcher:
    @pytest.mark.asyncio
    async def test_maps_results_correctly(self):
        from src.fetchers.arxiv_fetcher import ArxivFetcher

        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2401.00001"
        mock_result.title = "  Test Paper Title  "
        mock_result.summary = "  Abstract text  "
        author_a = MagicMock()
        author_a.name = "Author A"
        author_b = MagicMock()
        author_b.name = "Author B"
        mock_result.authors = [author_a, author_b]
        mock_result.published = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_result.categories = ["cs.AI", "cs.CL"]
        mock_result.pdf_url = "http://arxiv.org/pdf/2401.00001"
        mock_result.primary_category = "cs.AI"

        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]

        with patch("arxiv.Client", return_value=mock_client), \
             patch("arxiv.Search") as mock_search:
            fetcher = ArxivFetcher(categories=["cs.AI"], max_results=10)
            result = await fetcher.fetch()

        assert len(result) == 1
        assert result[0].title == "Test Paper Title"
        assert result[0].content == "Abstract text"
        assert result[0].url == "http://arxiv.org/abs/2401.00001"
        assert result[0].source_name == "arxiv"
        assert result[0].extra_metadata["categories"] == ["cs.AI", "cs.CL"]

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_results(self):
        from src.fetchers.arxiv_fetcher import ArxivFetcher

        mock_client = MagicMock()
        mock_client.results.return_value = []

        with patch("arxiv.Client", return_value=mock_client), \
             patch("arxiv.Search"):
            fetcher = ArxivFetcher()
            result = await fetcher.fetch()

        assert result == []

    @pytest.mark.asyncio
    async def test_truncates_authors_to_5(self):
        from src.fetchers.arxiv_fetcher import ArxivFetcher

        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2401.00001"
        mock_result.title = "Paper"
        mock_result.summary = "Summary"
        authors = []
        for i in range(10):
            a = MagicMock()
            a.name = f"Author {i}"
            authors.append(a)
        mock_result.authors = authors
        mock_result.published = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_result.categories = ["cs.AI"]
        mock_result.pdf_url = "http://arxiv.org/pdf/2401.00001"
        mock_result.primary_category = "cs.AI"

        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]

        with patch("arxiv.Client", return_value=mock_client), \
             patch("arxiv.Search"):
            fetcher = ArxivFetcher()
            result = await fetcher.fetch()

        # Author string should have at most 5 names
        author_names = result[0].author.split(", ")
        assert len(author_names) <= 5


# ---------------------------------------------------------------------------
# GitHubTrendingFetcher
# ---------------------------------------------------------------------------

class TestGitHubTrendingFetcher:
    TRENDING_HTML = """
    <article class="Box-row">
        <h2><a href="/openai/gpt-toolkit">openai / gpt-toolkit</a></h2>
        <p>A toolkit for building LLM applications</p>
        <span itemprop="programmingLanguage">Python</span>
        <span class="d-inline-block float-sm-right">500 stars today</span>
    </article>
    <article class="Box-row">
        <h2><a href="/user/cooking-recipes">user / cooking-recipes</a></h2>
        <p>A collection of cooking recipes</p>
        <span itemprop="programmingLanguage">JavaScript</span>
    </article>
    """

    @pytest.mark.asyncio
    async def test_filters_for_ai_repos(self):
        from src.fetchers.github_trending import GitHubTrendingFetcher

        mock_resp = MagicMock()
        mock_resp.text = self.TRENDING_HTML
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = GitHubTrendingFetcher()
            result = await fetcher.fetch()

        # Only the gpt-toolkit repo should match AI keywords
        assert len(result) == 1
        assert "gpt-toolkit" in result[0].title

    @pytest.mark.asyncio
    async def test_extracts_language_and_stars(self):
        from src.fetchers.github_trending import GitHubTrendingFetcher

        mock_resp = MagicMock()
        mock_resp.text = self.TRENDING_HTML
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = GitHubTrendingFetcher()
            result = await fetcher.fetch()

        assert result[0].extra_metadata["language"] == "Python"
        assert result[0].extra_metadata["stars_today"] == "500 stars today"

    @pytest.mark.asyncio
    async def test_handles_empty_page(self):
        from src.fetchers.github_trending import GitHubTrendingFetcher

        mock_resp = MagicMock()
        mock_resp.text = "<html><body></body></html>"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = GitHubTrendingFetcher()
            result = await fetcher.fetch()

        assert result == []

    @pytest.mark.asyncio
    async def test_source_type_is_scrape(self):
        from src.fetchers.github_trending import GitHubTrendingFetcher
        assert GitHubTrendingFetcher().source_type == "scrape"


# ---------------------------------------------------------------------------
# HuggingFaceFetcher
# ---------------------------------------------------------------------------

class TestHuggingFaceFetcher:
    SAMPLE_RESPONSE = [
        {
            "paper": {
                "id": "2401.00001",
                "title": "Scaling LLMs",
                "summary": "We study scaling laws.",
                "publishedAt": "2024-01-15T00:00:00Z",
                "authors": [
                    {"name": "Alice"},
                    {"name": "Bob"},
                ],
                "arxivId": "2401.00001",
            },
            "numUpvotes": 42,
        },
        {
            "paper": {
                "id": "",
                "title": "Missing ID Paper",
                "summary": "No ID",
                "authors": [],
            },
            "numUpvotes": 0,
        },
    ]

    @pytest.mark.asyncio
    async def test_maps_papers_correctly(self):
        from src.fetchers.huggingface import HuggingFaceFetcher

        mock_resp = MagicMock()
        mock_resp.json.return_value = self.SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = HuggingFaceFetcher(limit=10)
            result = await fetcher.fetch()

        # Should skip the paper with empty ID
        assert len(result) == 1
        assert result[0].title == "Scaling LLMs"
        assert result[0].url == "https://huggingface.co/papers/2401.00001"
        assert result[0].extra_metadata["upvotes"] == 42
        assert result[0].author == "Alice, Bob"
        assert result[0].source_name == "huggingface"

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        from src.fetchers.huggingface import HuggingFaceFetcher

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = HuggingFaceFetcher()
            result = await fetcher.fetch()

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_invalid_date(self):
        from src.fetchers.huggingface import HuggingFaceFetcher

        bad_data = [{
            "paper": {
                "id": "123",
                "title": "Test",
                "summary": "Test",
                "publishedAt": "not-a-date",
                "authors": [],
            },
            "numUpvotes": 0,
        }]

        mock_resp = MagicMock()
        mock_resp.json.return_value = bad_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = HuggingFaceFetcher()
            result = await fetcher.fetch()

        assert len(result) == 1
        assert result[0].published_at is None


# ---------------------------------------------------------------------------
# AnthropicBlogFetcher
# ---------------------------------------------------------------------------

class TestAnthropicBlogFetcher:
    def _make_rsc_html(self, posts):
        """Build fake HTML with RSC payload containing posts."""
        parts = []
        for post in posts:
            part = (
                f'"_type":"post","slug":{{"_type":"slug","current":"{post["slug"]}"}}'
                f',"title":"{post["title"]}"'
                f',"summary":"{post.get("summary", "")}"'
                f',"publishedOn":"{post.get("date", "2024-01-01")}"'
                f',"label":"{post.get("tag", "Research")}"'
            )
            parts.append(part)

        inner = "PREAMBLE" + "".join(parts)
        # JSON-encode the inner string
        encoded = json.dumps(inner)
        script = f"self.__next_f.push([1,{encoded}])"
        return f'<html><head></head><body><script>{script}</script></body></html>'

    @pytest.mark.asyncio
    async def test_parses_rsc_payload(self):
        from src.fetchers.anthropic_blog import AnthropicBlogFetcher

        html = self._make_rsc_html([
            {"slug": "claude-4", "title": "Introducing Claude 4", "summary": "Our latest model."},
            {"slug": "safety-update", "title": "Safety Update", "summary": "New safety features."},
        ])

        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = AnthropicBlogFetcher()
            result = await fetcher.fetch()

        assert len(result) == 2
        assert result[0].title == "Introducing Claude 4"
        assert result[0].url == "https://www.anthropic.com/news/claude-4"
        assert result[0].author == "Anthropic"
        assert result[0].source_name == "anthropic_blog"

    @pytest.mark.asyncio
    async def test_deduplicates_by_slug(self):
        from src.fetchers.anthropic_blog import AnthropicBlogFetcher

        html = self._make_rsc_html([
            {"slug": "same-post", "title": "Same Post"},
            {"slug": "same-post", "title": "Same Post Duplicate"},
        ])

        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = AnthropicBlogFetcher()
            result = await fetcher.fetch()

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fallback_html_scrape_when_no_rsc(self):
        from src.fetchers.anthropic_blog import AnthropicBlogFetcher

        html = """
        <html><body>
            <a href="/news/claude-update">Claude Update</a>
            <a href="/news/safety">Safety</a>
            <a href="/about">About</a>
            <a href="/news/">News Index</a>
        </body></html>
        """

        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = AnthropicBlogFetcher()
            result = await fetcher.fetch()

        # Should find 2 valid article links (/news/claude-update and /news/safety)
        assert len(result) == 2
        slugs = {r.url.split("/")[-1] for r in result}
        assert "claude-update" in slugs
        assert "safety" in slugs
        assert result[0].extra_metadata.get("fallback_scrape") is True

    @pytest.mark.asyncio
    async def test_handles_malformed_rsc_json(self):
        from src.fetchers.anthropic_blog import AnthropicBlogFetcher

        html = '<html><script>self.__next_f.push([1,{not valid json}])</script></html>'

        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            fetcher = AnthropicBlogFetcher()
            result = await fetcher.fetch()

        # Should gracefully return empty, not crash
        assert result == []
