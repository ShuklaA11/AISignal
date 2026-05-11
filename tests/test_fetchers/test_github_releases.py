"""Tests for the curated GitHub releases fetcher."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.fetchers.github_releases import GitHubReleasesFetcher


def _release(
    tag: str = "v1.0",
    name: str | None = "Release 1.0",
    body: str = "Some changelog.",
    days_ago: float = 1,
    draft: bool = False,
    prerelease: bool = False,
    html_url: str = "https://github.com/foo/bar/releases/tag/v1.0",
) -> dict:
    return {
        "tag_name": tag,
        "name": name,
        "body": body,
        "published_at": (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(),
        "draft": draft,
        "prerelease": prerelease,
        "html_url": html_url,
    }


def _mock_client(repo_responses: dict[str, list[dict] | int]) -> MagicMock:
    """repo_responses maps repo string -> list of release dicts OR int status code."""

    async def get(url: str, params=None):
        repo = url.split("/repos/", 1)[1].rsplit("/releases", 1)[0]
        resp = MagicMock()
        resp.json = MagicMock(return_value=repo_responses.get(repo, []))
        val = repo_responses.get(repo, [])
        if isinstance(val, int):
            resp.status_code = val
            resp.raise_for_status = MagicMock(side_effect=Exception("HTTP error"))
            resp.json = MagicMock(return_value=[])
        else:
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=val)
        return resp

    client = MagicMock()
    client.get = AsyncMock(side_effect=get)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_returns_empty_with_no_repos():
    fetcher = GitHubReleasesFetcher(repos=[])
    assert await fetcher.fetch() == []


@pytest.mark.asyncio
async def test_drops_invalid_repo_format():
    """Repos must be 'owner/name'. Plain strings are silently dropped."""
    fetcher = GitHubReleasesFetcher(repos=["malformed-no-slash", "foo/bar"])
    assert fetcher.repos == ["foo/bar"]


@pytest.mark.asyncio
async def test_emits_articles_for_recent_releases(monkeypatch):
    responses = {
        "vllm-project/vllm": [_release(tag="v0.5", name="0.5 release", days_ago=2)],
        "ggml-org/llama.cpp": [_release(tag="b3000", name="build 3000", days_ago=5)],
    }
    client = _mock_client(responses)
    monkeypatch.setattr("src.fetchers.github_releases.httpx.AsyncClient", lambda **kw: client)

    fetcher = GitHubReleasesFetcher(repos=list(responses.keys()))
    articles = await fetcher.fetch()

    assert len(articles) == 2
    titles = [a.title for a in articles]
    assert any("vllm-project/vllm" in t for t in titles)
    assert any("ggml-org/llama.cpp" in t for t in titles)


@pytest.mark.asyncio
async def test_filters_drafts_and_prereleases(monkeypatch):
    responses = {
        "foo/bar": [
            _release(tag="v1", name="real", days_ago=1),
            _release(tag="v2-rc1", name="rc", days_ago=1, prerelease=True),
            _release(tag="v3", name="draft", days_ago=1, draft=True),
        ],
    }
    client = _mock_client(responses)
    monkeypatch.setattr("src.fetchers.github_releases.httpx.AsyncClient", lambda **kw: client)

    fetcher = GitHubReleasesFetcher(repos=["foo/bar"])
    articles = await fetcher.fetch()

    assert len(articles) == 1
    assert "real" in articles[0].title


@pytest.mark.asyncio
async def test_filters_old_releases_by_lookback(monkeypatch):
    responses = {
        "foo/bar": [
            _release(tag="v1", days_ago=2),    # within 7-day window
            _release(tag="v0", days_ago=20),   # outside
        ],
    }
    client = _mock_client(responses)
    monkeypatch.setattr("src.fetchers.github_releases.httpx.AsyncClient", lambda **kw: client)

    fetcher = GitHubReleasesFetcher(repos=["foo/bar"], lookback_days=7)
    articles = await fetcher.fetch()

    assert len(articles) == 1
    assert "v1" in articles[0].title


@pytest.mark.asyncio
async def test_404_for_one_repo_does_not_kill_others(monkeypatch):
    """A renamed or private repo shouldn't blackhole the rest."""
    responses = {
        "deleted/repo": 404,
        "live/repo": [_release(tag="v1", name="ok", days_ago=1)],
    }
    client = _mock_client(responses)
    monkeypatch.setattr("src.fetchers.github_releases.httpx.AsyncClient", lambda **kw: client)

    fetcher = GitHubReleasesFetcher(repos=list(responses.keys()))
    articles = await fetcher.fetch()

    assert len(articles) == 1
    assert "live/repo" in articles[0].title


def test_class_attrs_tag_section():
    assert GitHubReleasesFetcher.section == "builder"
    assert GitHubReleasesFetcher.quality_weight > 1.0


@pytest.mark.asyncio
async def test_auto_tagging_via_safe_fetch(monkeypatch):
    """Verify the base class auto-tagging applies to articles from this fetcher."""
    responses = {"foo/bar": [_release(tag="v1", days_ago=1)]}
    client = _mock_client(responses)
    monkeypatch.setattr("src.fetchers.github_releases.httpx.AsyncClient", lambda **kw: client)

    fetcher = GitHubReleasesFetcher(repos=["foo/bar"])
    articles = await fetcher.safe_fetch()

    assert articles[0].section == "builder"
    assert "industry" in articles[0].audience_tags
    assert articles[0].quality_weight == 1.5
