"""Tests for the Artificial Analysis provider's parser + provider contract."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.leaderboards.artificial_analysis import (
    ArtificialAnalysisProvider,
    METRIC_INTELLIGENCE,
    METRIC_PRICE,
    METRIC_SPEED,
    PROVIDER,
    build_snapshots,
    parse_models,
)


def _model_chunk(
    slug: str,
    name: str,
    org: str,
    intel: float | None,
    speed: float | None,
    price: float | None,
) -> str:
    """Build a synthetic AA-style escaped-JSON chunk for one model.

    Mirrors the real schema: model name/slug, model_creators block (with the
    org name), then intelligence_index + timescaleData.median_output_speed +
    price_1m_blended_0_3_1, ending with hosts_url which is the parser anchor.
    """
    def fmt(v):
        return "null" if v is None else str(v)

    return (
        f'{{\\"id\\":\\"xx\\","short_name":\\"{name}\\",\\"name\\":\\"{name}\\",'
        f'\\"model_url\\":\\"/models/{slug}\\",'
        f'\\"model_creators\\":{{\\"id\\":\\"yy\\",\\"name\\":\\"{org}\\",\\"slug\\":\\"{org.lower()}\\"}},'
        f'\\"intelligence_index\\":{fmt(intel)},'
        f'\\"timescaleData\\":{{\\"median_output_speed\\":{fmt(speed)}}},'
        f'\\"price_1m_blended_0_3_1\\":{fmt(price)},'
        f'\\"hosts_url\\":\\"/models/{slug}/providers\\"}}'
    )


@pytest.fixture
def synthetic_html() -> str:
    chunks = [
        _model_chunk("claude-opus-4-7", "Claude Opus 4.7", "Anthropic", 57.28, 92.0, 7.5),
        _model_chunk("gpt-5-5-xhigh", "GPT-5.5 (xhigh)", "OpenAI", 60.24, 56.65, 11.25),
        _model_chunk("gemini-3-1-pro", "Gemini 3.1 Pro Preview", "Google", 57.18, 170.08, 6.0),
        _model_chunk("gpt-oss-20b", "gpt-oss-20B (high)", "OpenAI", 24.47, 235.43, 0.0875),
        _model_chunk("kimi-k2-6", "Kimi K2.6", "Kimi", 53.9, 84.0, 2.4),
    ]
    return ",".join(chunks)


# -- Parser ----------------------------------------------------------------


def test_parses_synthetic_fixture(synthetic_html: str):
    rows = parse_models(synthetic_html)
    assert len(rows) == 5
    by_slug = {r.slug: r for r in rows}
    assert by_slug["claude-opus-4-7"].name == "Claude Opus 4.7"
    assert by_slug["claude-opus-4-7"].organization == "Anthropic"
    assert by_slug["claude-opus-4-7"].intelligence == 57.28
    assert by_slug["claude-opus-4-7"].speed == 92.0
    assert by_slug["claude-opus-4-7"].price == 7.5


def test_parser_handles_null_metrics():
    html = _model_chunk("unreleased", "Some Model", "Lab", None, None, None)
    rows = parse_models(html)
    assert len(rows) == 1
    assert rows[0].intelligence is None
    assert rows[0].speed is None
    assert rows[0].price is None


def test_parser_returns_empty_for_unrelated_html():
    assert parse_models("<html>no leaderboard data here</html>") == []


# -- build_snapshots -------------------------------------------------------


def test_build_snapshots_produces_three_metrics(synthetic_html: str):
    rows = parse_models(synthetic_html)
    snaps = build_snapshots(rows)
    metrics = {s.metric for s in snaps}
    assert metrics == {METRIC_INTELLIGENCE, METRIC_SPEED, METRIC_PRICE}
    assert all(s.provider == PROVIDER for s in snaps)


def test_intelligence_ranking_is_descending(synthetic_html: str):
    rows = parse_models(synthetic_html)
    intel = [s for s in build_snapshots(rows) if s.metric == METRIC_INTELLIGENCE][0]
    scores = [r.score for r in intel.rankings]
    assert scores == sorted(scores, reverse=True)
    # GPT-5.5 at 60.24 should be #1
    assert intel.rankings[0].model == "GPT-5.5 (xhigh)"


def test_price_ranking_is_ascending_cheaper_is_better(synthetic_html: str):
    rows = parse_models(synthetic_html)
    price = [s for s in build_snapshots(rows) if s.metric == METRIC_PRICE][0]
    scores = [r.score for r in price.rankings]
    assert scores == sorted(scores)  # ascending
    # gpt-oss-20b at $0.0875/Mtok should be #1
    assert "gpt-oss-20B" in price.rankings[0].model


def test_speed_ranking_excludes_null_or_zero(synthetic_html: str):
    rows = parse_models(synthetic_html)
    # Add a row with speed=None
    rows.append(type(rows[0])(
        slug="bench", name="B", organization="O", intelligence=10.0, speed=None, price=1.0
    ))
    speed = [s for s in build_snapshots(rows) if s.metric == METRIC_SPEED][0]
    assert all(r.score is not None and r.score > 0 for r in speed.rankings)
    assert "B" not in [r.model for r in speed.rankings]


def test_ranking_carries_slug_in_extras(synthetic_html: str):
    rows = parse_models(synthetic_html)
    snap = build_snapshots(rows)[0]
    assert snap.rankings[0].extras["slug"]


# -- Provider HTTP path (mocked) ------------------------------------------


@pytest.mark.asyncio
async def test_fetch_snapshot_hits_url_and_parses(monkeypatch, synthetic_html: str):
    resp = MagicMock()
    resp.text = synthetic_html
    resp.raise_for_status = MagicMock()

    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "src.leaderboards.artificial_analysis.httpx.AsyncClient",
        lambda **kw: client,
    )

    provider = ArtificialAnalysisProvider()
    snaps = list(await provider.fetch_snapshot())

    assert len(snaps) == 3
    client.get.assert_called_once()
    intel = [s for s in snaps if s.metric == METRIC_INTELLIGENCE][0]
    assert intel.rankings[0].model == "GPT-5.5 (xhigh)"


@pytest.mark.asyncio
async def test_fetch_snapshot_returns_empty_when_page_format_changed(monkeypatch):
    """If AA breaks our parser, return [] rather than crashing."""
    resp = MagicMock()
    resp.text = "<html>completely different layout</html>"
    resp.raise_for_status = MagicMock()
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "src.leaderboards.artificial_analysis.httpx.AsyncClient",
        lambda **kw: client,
    )

    snaps = list(await ArtificialAnalysisProvider().fetch_snapshot())
    assert snaps == []


def test_provider_name_is_stable():
    assert ArtificialAnalysisProvider().provider_name == PROVIDER
