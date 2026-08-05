"""Scoring must not mutate the profile it scores against.

`scripts/eval_mind.py` memoises one `UserMLProfile` per distinct click history
and reuses it across every impression and every method that shares that history.
That is only sound while scoring is read-only. If a scorer ever starts writing
back into the profile, the cache would silently carry state between impressions
and every number downstream of it would be wrong, so the property is pinned
here rather than left to inspection.
"""

import copy
import json

import pytest

from src.personalization.scorer import (
    _compute_learned_score,
    score_article_for_user,
    score_article_for_user_ml,
)
from src.storage.models import Article, User, UserMLProfile


@pytest.fixture
def article() -> Article:
    return Article(
        id=1,
        url="mind://N1",
        content_hash="N1",
        title="A market moving story",
        source_name="mind",
        source_type="mind",
        category="finance",
        base_importance_score=None,
        topics_json=json.dumps(["markets"]),
        difficulty_level=None,
        key_entities_json=json.dumps(["Q1", "Q2"]),
        section=None,
    )


@pytest.fixture
def user() -> User:
    return User(
        id=1,
        email="u@mind.local",
        role="",
        level="",
        topics_json=json.dumps(["markets", "economy"]),
        source_preferences_json="{}",
    )


@pytest.fixture
def profile() -> UserMLProfile:
    trained = UserMLProfile(user_id=1)
    trained.category_weights = {"finance": 1.4, "sports": 0.7}
    trained.topic_weights = {"markets": 1.6, "economy": 1.1}
    trained.entity_weights = {"Q1": 1.3, "Q2": 0.9}
    trained.source_weights = {"mind": 1.2}
    trained.difficulty_weights = {"intermediate": 1.05}
    trained.total_clicks = 40
    trained.alpha = 0.4
    return trained


def _snapshot(profile: UserMLProfile) -> dict:
    """Everything the scorer could plausibly write to."""
    return {
        "category_weights": copy.deepcopy(profile.category_weights),
        "topic_weights": copy.deepcopy(profile.topic_weights),
        "entity_weights": copy.deepcopy(profile.entity_weights),
        "source_weights": copy.deepcopy(profile.source_weights),
        "difficulty_weights": copy.deepcopy(profile.difficulty_weights),
        "total_clicks": profile.total_clicks,
        "alpha": profile.alpha,
    }


def test_blended_scoring_leaves_the_profile_unchanged(article, user, profile):
    # Arrange
    before = _snapshot(profile)

    # Act
    score_article_for_user_ml(article, user, profile)

    # Assert
    assert _snapshot(profile) == before


def test_learned_scoring_leaves_the_profile_unchanged(article, user, profile):
    # Arrange
    before = _snapshot(profile)

    # Act
    _compute_learned_score(article, profile)

    # Assert
    assert _snapshot(profile) == before


def test_rule_scoring_leaves_the_profile_unchanged(article, user, profile):
    # Arrange
    before = _snapshot(profile)

    # Act
    score_article_for_user(article, user, ml_profile=profile)

    # Assert
    assert _snapshot(profile) == before


def test_repeated_scoring_of_a_shared_profile_is_stable(article, user, profile):
    """The cache reuses one profile across many impressions; scores must not drift."""
    # Act
    scores = [score_article_for_user_ml(article, user, profile) for _ in range(50)]

    # Assert
    assert len(set(scores)) == 1


def test_a_shared_profile_matches_a_freshly_built_one(article, user, profile):
    """A cache hit must be indistinguishable from a cache miss."""
    # Arrange
    fresh = copy.deepcopy(profile)

    # Act
    warmed = score_article_for_user_ml(article, user, profile)
    _ = score_article_for_user_ml(article, user, profile)
    uncached = score_article_for_user_ml(article, user, fresh)

    # Assert
    assert warmed == uncached
