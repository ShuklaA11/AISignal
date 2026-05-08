"""Tests for config schema extensions: tagged RSS feeds and section weights."""
import pytest
from pydantic import ValidationError

from src.config import RSSFeed, Settings
from src.sections import (
    ALL_SECTIONS,
    SECTION_BUILDER,
    SECTION_RESEARCH,
)


# -- RSSFeed schema -----------------------------------------------------------


@pytest.mark.unit
def test_rss_feed_minimal_fields_still_parse() -> None:
    """Backward compat: feeds without the new fields must still load."""
    feed = RSSFeed(name="x", url="https://example.com/feed.xml")
    assert feed.section is None
    assert feed.audience_tags == []
    assert feed.quality_weight == 1.0


@pytest.mark.unit
def test_rss_feed_accepts_all_new_fields() -> None:
    feed = RSSFeed(
        name="lilian_weng",
        url="https://lilianweng.github.io/feed.xml",
        section=SECTION_RESEARCH,
        audience_tags=["researcher", "industry"],
        quality_weight=2.0,
    )
    assert feed.section == SECTION_RESEARCH
    assert feed.audience_tags == ["researcher", "industry"]
    assert feed.quality_weight == 2.0


@pytest.mark.unit
def test_rss_feed_rejects_unknown_section() -> None:
    with pytest.raises(ValidationError):
        RSSFeed(name="x", url="u", section="frontier")


@pytest.mark.unit
@pytest.mark.parametrize("section", list(ALL_SECTIONS))
def test_rss_feed_accepts_every_known_section(section: str) -> None:
    feed = RSSFeed(name="x", url="u", section=section)
    assert feed.section == section


# -- Settings.get_section_weights --------------------------------------------


def _settings_with_weights(weights: dict) -> Settings:
    """Construct a Settings with explicit section_weights, bypassing yaml."""
    return Settings(secret_key="x" * 32, section_weights=weights)


@pytest.mark.unit
def test_section_weights_neutral_when_unconfigured() -> None:
    s = _settings_with_weights({})
    weights = s.get_section_weights("anything")
    assert weights == {section: 1.0 for section in ALL_SECTIONS}


@pytest.mark.unit
def test_section_weights_role_specific() -> None:
    s = _settings_with_weights({
        "industry": {"builder": 1.5, "releases": 1.3},
    })
    weights = s.get_section_weights("industry")
    assert weights["builder"] == 1.5
    assert weights["releases"] == 1.3
    # Unspecified sections fall back to neutral
    assert weights[SECTION_RESEARCH] == 1.0


@pytest.mark.unit
def test_section_weights_falls_back_to_default() -> None:
    s = _settings_with_weights({
        "default": {"research": 0.5},
    })
    weights = s.get_section_weights("never_configured_role")
    assert weights["research"] == 0.5
    assert weights[SECTION_BUILDER] == 1.0  # neutral fallback


@pytest.mark.unit
def test_section_weights_role_overrides_default() -> None:
    s = _settings_with_weights({
        "default": {"research": 0.5},
        "researcher": {"research": 2.0},
    })
    assert s.get_section_weights("researcher")["research"] == 2.0
    assert s.get_section_weights("other_role")["research"] == 0.5


@pytest.mark.unit
def test_section_weights_ignores_unknown_section_keys() -> None:
    s = _settings_with_weights({
        "default": {"research": 1.5, "frontier": 99.0},
    })
    weights = s.get_section_weights("default")
    assert weights["research"] == 1.5
    assert "frontier" not in weights


@pytest.mark.unit
def test_section_weights_returns_complete_section_map() -> None:
    s = _settings_with_weights({"default": {"research": 1.5}})
    weights = s.get_section_weights("default")
    assert set(weights.keys()) == set(ALL_SECTIONS)
