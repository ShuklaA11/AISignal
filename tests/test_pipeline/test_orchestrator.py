"""Tests for pipeline orchestrator: title cleaning, article storage/dedup, fetcher creation.

Covers:
- _clean_title() LaTeX stripping (dollar signs, mathcal, superscripts, greek letters, passthrough)
- store_articles() deduplication (URL dedup, title fingerprint dedup, new storage, empty input)
- build_fetchers() with mocked Settings
"""

import pytest
from sqlmodel import Session, SQLModel, create_engine
from unittest.mock import MagicMock, patch

from src.fetchers.base import RawArticle
from src.pipeline.orchestrator import _clean_title, build_fetchers, store_articles
from src.storage.models import Article, utcnow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """In-memory SQLite for testing."""
    eng = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def _make_raw(title="Test Article", url="https://example.com/1", source="rss_test"):
    return RawArticle(
        url=url,
        title=title,
        content="Some content.",
        source_name=source,
        source_type="rss",
    )


# ---------------------------------------------------------------------------
# _clean_title
# ---------------------------------------------------------------------------

class TestCleanTitle:
    def test_strips_dollar_signs(self):
        """$Q*$ -> Q*"""
        result = _clean_title("Learning $Q*$ Methods")
        assert result == "Learning Q* Methods"

    def test_mathcal(self):
        r"""$\mathcal{O}(n)$ -> O(n)"""
        result = _clean_title(r"$\mathcal{O}(n)$ Complexity")
        assert result == "O(n) Complexity"

    def test_superscripts(self):
        """$x^2$ -> x2"""
        result = _clean_title(r"Computing $x^2$ Fast")
        assert result == "Computing x2 Fast"

    def test_greek_letters(self):
        r"""\alpha -> alpha unicode"""
        result = _clean_title(r"$\alpha$-divergence")
        assert "\u03b1" in result
        assert "$" not in result

    def test_clean_passthrough(self):
        """Title without LaTeX passes through unchanged."""
        title = "Attention Is All You Need"
        assert _clean_title(title) == title

    def test_empty_string(self):
        assert _clean_title("") == ""

    def test_multiple_latex_blocks(self):
        result = _clean_title(r"From $\beta$-VAE to $\gamma$-Networks")
        assert "\u03b2" in result
        assert "\u03b3" in result
        assert "$" not in result


# ---------------------------------------------------------------------------
# store_articles
# ---------------------------------------------------------------------------

class TestStoreArticles:
    def test_stores_new_article(self, session):
        """A fresh article is stored and counted."""
        raw = [_make_raw()]
        count = store_articles(session, raw, existing_fps=set())
        assert count == 1
        stored = session.get(Article, 1)
        assert stored is not None
        assert stored.title == "Test Article"
        assert stored.status == "raw"

    def test_url_dedup(self, session):
        """Duplicate URL is skipped."""
        # Pre-populate an article with the same URL
        existing = Article(
            url="https://example.com/1",
            content_hash="abc",
            title="Old Article",
            source_name="rss_test",
            source_type="rss",
            fetched_at=utcnow(),
            status="raw",
        )
        session.add(existing)
        session.commit()

        raw = [_make_raw(title="New Title", url="https://example.com/1")]
        count = store_articles(session, raw, existing_fps=set())
        assert count == 0

    def test_title_fingerprint_dedup(self, session):
        """Article with matching title fingerprint is skipped."""
        from src.storage.queries import _title_fingerprint

        existing_title = "Test Article About AI Safety Research"
        fp = _title_fingerprint(existing_title)
        existing_fps = {fp}

        raw = [_make_raw(title=existing_title, url="https://unique-url.com/new")]
        count = store_articles(session, raw, existing_fps=existing_fps)
        assert count == 0

    def test_empty_input(self, session):
        """Empty list returns 0 and stores nothing."""
        count = store_articles(session, [], existing_fps=set())
        assert count == 0

    def test_mixed_new_and_duplicate(self, session):
        """Mix of new and duplicate articles: only new ones stored."""
        existing = Article(
            url="https://example.com/dup",
            content_hash="dup",
            title="Duplicate",
            source_name="rss_test",
            source_type="rss",
            fetched_at=utcnow(),
            status="raw",
        )
        session.add(existing)
        session.commit()

        raw = [
            _make_raw(title="Duplicate", url="https://example.com/dup"),
            _make_raw(title="Brand New Article", url="https://example.com/new"),
        ]
        count = store_articles(session, raw, existing_fps=set())
        assert count == 1

    def test_clean_title_applied(self, session):
        """LaTeX in titles is cleaned during storage."""
        raw = [_make_raw(title=r"$\mathcal{O}(n)$ Algo", url="https://example.com/latex")]
        store_articles(session, raw, existing_fps=set())
        stored = session.get(Article, 1)
        assert "$" not in stored.title
        assert "O(n)" in stored.title


# ---------------------------------------------------------------------------
# build_fetchers
# ---------------------------------------------------------------------------

class TestBuildFetchers:
    @patch("src.pipeline.orchestrator.RSSFetcher")
    @patch("src.pipeline.orchestrator.AnthropicBlogFetcher")
    @patch("src.pipeline.orchestrator.HuggingFaceFetcher")
    @patch("src.pipeline.orchestrator.ArxivFetcher")
    @patch("src.pipeline.orchestrator.GitHubTrendingFetcher")
    @patch("src.pipeline.orchestrator.RedditFetcher")
    @patch("src.pipeline.orchestrator.TwitterFetcher")
    def test_all_fetchers_created(
        self, mock_twitter, mock_reddit, mock_github, mock_arxiv,
        mock_hf, mock_anthropic, mock_rss,
    ):
        """With all credentials provided, all fetcher types are instantiated."""
        feed1 = MagicMock()
        feed1.name = "techcrunch"
        feed1.url = "https://tc.com/rss"
        feed2 = MagicMock()
        feed2.name = "anthropic_blog"
        feed2.url = "https://anthropic.com/rss"

        settings = MagicMock()
        settings.rss_feeds = [feed1, feed2]
        settings.arxiv_categories = ["cs.AI"]
        settings.arxiv_max_results = 10
        settings.reddit_client_id = "id"
        settings.reddit_client_secret = "secret"
        settings.reddit_user_agent = "bot"
        settings.reddit_subreddits = ["MachineLearning"]
        settings.reddit_min_score = 10
        settings.twitter_bearer_token = "token"
        settings.twitter_query = "AI"
        settings.twitter_max_results = 50

        fetchers = build_fetchers(settings)

        # RSS: feed1 only (anthropic_blog skipped)
        mock_rss.assert_called_once()
        # Dedicated fetchers
        mock_anthropic.assert_called_once()
        mock_hf.assert_called_once()
        mock_arxiv.assert_called_once()
        mock_github.assert_called_once()
        mock_reddit.assert_called_once()
        mock_twitter.assert_called_once()

    @patch("src.pipeline.orchestrator.RSSFetcher")
    @patch("src.pipeline.orchestrator.AnthropicBlogFetcher")
    @patch("src.pipeline.orchestrator.HuggingFaceFetcher")
    @patch("src.pipeline.orchestrator.ArxivFetcher")
    @patch("src.pipeline.orchestrator.GitHubTrendingFetcher")
    def test_optional_fetchers_skipped_without_creds(
        self, mock_github, mock_arxiv, mock_hf, mock_anthropic, mock_rss,
    ):
        """Reddit and Twitter fetchers are skipped when credentials are absent."""
        settings = MagicMock()
        settings.rss_feeds = []
        settings.arxiv_categories = ["cs.AI"]
        settings.arxiv_max_results = 10
        settings.reddit_client_id = ""
        settings.reddit_client_secret = ""
        settings.twitter_bearer_token = ""

        with patch("src.pipeline.orchestrator.RedditFetcher") as mock_reddit, \
             patch("src.pipeline.orchestrator.TwitterFetcher") as mock_twitter:
            fetchers = build_fetchers(settings)
            mock_reddit.assert_not_called()
            mock_twitter.assert_not_called()
