"""Tests for BaseFetcher's section auto-tagging via class attributes."""
import pytest

from src.fetchers.anthropic_blog import AnthropicBlogFetcher
from src.fetchers.arxiv_fetcher import ArxivFetcher
from src.fetchers.base import BaseFetcher, RawArticle
from src.fetchers.github_trending import GitHubTrendingFetcher
from src.fetchers.huggingface import HuggingFaceFetcher


class _StubFetcher(BaseFetcher):
    """Test double that returns whatever articles you hand it."""

    section = "research"
    audience_tags = ("researcher",)
    quality_weight = 1.5

    def __init__(self, articles: list[RawArticle]):
        self._articles = articles

    @property
    def source_name(self) -> str:
        return "stub"

    async def fetch(self) -> list[RawArticle]:
        return list(self._articles)


@pytest.mark.asyncio
async def test_auto_tags_untagged_articles():
    raw = RawArticle(url="u", title="t", source_name="stub", source_type="api")
    fetcher = _StubFetcher([raw])

    result = await fetcher.safe_fetch()

    assert result[0].section == "research"
    assert result[0].audience_tags == ["researcher"]
    assert result[0].quality_weight == 1.5


@pytest.mark.asyncio
async def test_per_article_section_overrides_default():
    """If a fetcher sets section explicitly on a RawArticle, auto-tag preserves it."""
    raw = RawArticle(
        url="u", title="t", source_name="stub", source_type="api",
        section="industry",
    )
    fetcher = _StubFetcher([raw])

    result = await fetcher.safe_fetch()

    assert result[0].section == "industry"  # not overwritten


@pytest.mark.asyncio
async def test_per_article_audience_overrides_default():
    raw = RawArticle(
        url="u", title="t", source_name="stub", source_type="api",
        audience_tags=["student"],
    )
    fetcher = _StubFetcher([raw])

    result = await fetcher.safe_fetch()

    assert result[0].audience_tags == ["student"]


@pytest.mark.asyncio
async def test_no_class_attrs_means_no_tagging():
    """A fetcher without class attrs leaves articles at neutral defaults."""

    class _Untagged(BaseFetcher):
        @property
        def source_name(self) -> str:
            return "untagged"

        async def fetch(self) -> list[RawArticle]:
            return [RawArticle(url="u", title="t", source_name="untagged", source_type="api")]

    result = await _Untagged().safe_fetch()
    assert result[0].section is None
    assert result[0].audience_tags == []
    assert result[0].quality_weight == 1.0


@pytest.mark.asyncio
async def test_audience_tags_returned_as_fresh_list_per_article():
    """Mutating one article's audience_tags must not bleed into another."""
    raw1 = RawArticle(url="u1", title="t1", source_name="stub", source_type="api")
    raw2 = RawArticle(url="u2", title="t2", source_name="stub", source_type="api")
    fetcher = _StubFetcher([raw1, raw2])

    result = await fetcher.safe_fetch()
    result[0].audience_tags.append("industry")

    assert result[1].audience_tags == ["researcher"]


# Sanity checks on the actual built-in fetchers' class attrs ----------------


def test_huggingface_tagged_research():
    assert HuggingFaceFetcher.section == "research"
    assert HuggingFaceFetcher.quality_weight > 1.0


def test_arxiv_tagged_research_downweighted():
    assert ArxivFetcher.section == "research"
    assert ArxivFetcher.quality_weight < 1.0  # downweighted vs HF Daily Papers


def test_github_trending_tagged_builder():
    assert GitHubTrendingFetcher.section == "builder"


def test_anthropic_blog_tagged_releases():
    assert AnthropicBlogFetcher.section == "releases"
