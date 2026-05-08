"""Tests for section-aware scoring (Phase 1.3a).

Verifies that section weights from config and per-article quality_weight
are folded into score_article_for_user without breaking existing factors.
"""
import json
from unittest.mock import patch

import pytest

from src.sections import SECTION_BUILDER, SECTION_RESEARCH
from src.storage.models import Article, User
from src.personalization.scorer import score_article_for_user


def _user(role: str = "enthusiast") -> User:
    return User(
        id=1,
        email="t@example.com",
        role=role,
        level="intermediate",
        topics_json="[]",
        source_preferences_json="{}",
    )


def _article(
    section: str | None = None,
    quality_weight: float = 1.0,
    base: float = 5.0,
) -> Article:
    return Article(
        id=1,
        url="https://example.com/a",
        content_hash="h",
        title="t",
        source_name="generic_rss",
        source_type="rss",
        category="opinion",
        base_importance_score=base,
        topics_json="[]",
        difficulty_level="intermediate",
        key_entities_json="[]",
        status="processed",
        section=section,
        quality_weight=quality_weight,
    )


def _patched_settings(weights: dict) -> "patch":
    """Patch load_settings() in the scorer to return a stub with given weights."""
    class StubSettings:
        section_weights = weights

        def get_section_weights(self, role: str) -> dict[str, float]:
            from src.sections import ALL_SECTIONS
            neutral = {s: 1.0 for s in ALL_SECTIONS}
            profile = self.section_weights.get(role) or self.section_weights.get("default") or {}
            return {**neutral, **{k: v for k, v in profile.items() if k in ALL_SECTIONS}}

    return patch("src.personalization.scorer.load_settings", return_value=StubSettings())


@pytest.mark.unit
def test_no_section_yields_neutral_section_factor() -> None:
    """Articles without a section get a 1.0 multiplier (no effect)."""
    with _patched_settings({"default": {SECTION_RESEARCH: 2.0}}):
        score_with = score_article_for_user(_article(section=SECTION_RESEARCH), _user())
        score_without = score_article_for_user(_article(section=None), _user())
    assert score_with > score_without


@pytest.mark.unit
def test_section_weight_boosts_score_for_matching_role() -> None:
    """A role's heavy section ranks higher than a neutral default."""
    with _patched_settings({"researcher": {SECTION_RESEARCH: 1.8}}):
        boosted = score_article_for_user(
            _article(section=SECTION_RESEARCH), _user(role="researcher"),
        )
    with _patched_settings({}):
        neutral = score_article_for_user(
            _article(section=SECTION_RESEARCH), _user(role="researcher"),
        )
    assert boosted > neutral


@pytest.mark.unit
def test_section_weight_suppresses_score_when_below_one() -> None:
    with _patched_settings({"industry": {SECTION_RESEARCH: 0.5}}):
        suppressed = score_article_for_user(
            _article(section=SECTION_RESEARCH), _user(role="industry"),
        )
    with _patched_settings({}):
        neutral = score_article_for_user(
            _article(section=SECTION_RESEARCH), _user(role="industry"),
        )
    assert suppressed < neutral


@pytest.mark.unit
def test_quality_weight_multiplies_into_score() -> None:
    with _patched_settings({}):
        high_quality = score_article_for_user(
            _article(section=SECTION_BUILDER, quality_weight=2.0), _user(),
        )
        baseline = score_article_for_user(
            _article(section=SECTION_BUILDER, quality_weight=1.0), _user(),
        )
        low_quality = score_article_for_user(
            _article(section=SECTION_BUILDER, quality_weight=0.5), _user(),
        )
    assert high_quality > baseline > low_quality


@pytest.mark.unit
def test_quality_weight_clamped_to_safe_range() -> None:
    """quality_weight is clamped to [0.5, 2.0] like other factors."""
    with _patched_settings({}):
        absurd_high = score_article_for_user(
            _article(quality_weight=99.0), _user(),
        )
        capped_high = score_article_for_user(
            _article(quality_weight=2.0), _user(),
        )
    assert absurd_high == capped_high


@pytest.mark.unit
def test_role_specific_weights_override_default() -> None:
    """Role-specific section weight beats the default profile."""
    weights = {
        "default": {SECTION_RESEARCH: 0.5},
        "researcher": {SECTION_RESEARCH: 2.0},
    }
    with _patched_settings(weights):
        researcher_score = score_article_for_user(
            _article(section=SECTION_RESEARCH), _user(role="researcher"),
        )
        other_score = score_article_for_user(
            _article(section=SECTION_RESEARCH), _user(role="anyone_else"),
        )
    assert researcher_score > other_score


@pytest.mark.unit
def test_score_does_not_regress_when_no_section_weights_configured() -> None:
    """With empty config, behavior matches prior scorer (within rounding)."""
    article = _article(section=None, quality_weight=1.0)
    user = _user()
    with _patched_settings({}):
        score_a = score_article_for_user(article, user)
    with _patched_settings({"default": {}}):
        score_b = score_article_for_user(article, user)
    assert score_a == score_b
