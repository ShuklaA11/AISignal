"""The popularity baseline must not count clicks from the split it is scored on.

Validation is carved out of MINDsmall_train, so a popularity table built from
the whole train file has already seen the clicks it is being asked to predict.
Left unfixed this inflates popularity from 0.3103 nDCG@10 (dev, honest) to
0.3874 (validation, leaked) — above the rules baseline — and would have
contaminated the Phase 1 work on time-windowed popularity, which builds on the
same counts.
"""

from pathlib import Path

import pytest

from src.eval.mind_data import load_behaviors, train_click_counts
from src.eval.splits import (
    TEMPORAL_FIT,
    VALIDATION,
    VALIDATION_DAY,
    split_for_user,
)

BEHAVIORS = Path("data/mind/train/behaviors.tsv")

pytestmark = pytest.mark.skipif(
    not BEHAVIORS.exists(), reason="MIND data not present in this checkout"
)


def test_excluding_a_split_drops_that_split_s_clicks():
    # Act
    everything = train_click_counts(BEHAVIORS)
    without_validation = train_click_counts(BEHAVIORS, exclude_split=VALIDATION)

    # Assert
    assert sum(without_validation.values()) < sum(everything.values())
    for news_id, count in without_validation.items():
        assert count <= everything[news_id]


def test_no_validation_user_click_survives_the_exclusion():
    """The property that makes the held-out popularity number honest."""
    # Arrange
    from src.eval.mind_data import load_behaviors

    expected: dict[str, int] = {}
    for impression in load_behaviors(BEHAVIORS):
        if split_for_user(impression.user_id) == VALIDATION:
            continue
        for news_id, label in zip(impression.candidates, impression.labels):
            if label == 1:
                expected[news_id] = expected.get(news_id, 0) + 1

    # Act
    actual = train_click_counts(BEHAVIORS, exclude_split=VALIDATION)

    # Assert
    assert dict(actual) == expected


def test_default_behaviour_is_unchanged():
    """Scoring dev still counts every train click, as the published numbers did."""
    # Act
    default = train_click_counts(BEHAVIORS)
    explicit = train_click_counts(BEHAVIORS, exclude_split=None)

    # Assert
    assert default == explicit


def test_unknown_split_is_rejected():
    # Act / Assert
    with pytest.raises(ValueError, match="unknown split"):
        train_click_counts(BEHAVIORS, exclude_split="nonsense")


# ── Temporal restriction ─────────────────────────────────────────────


def test_restricting_to_earlier_days_drops_validation_day_clicks():
    """Popularity for a Nov 14 evaluation must come from Nov 9-13 only.

    Without this the counts are contemporaneous with the impressions they
    score, which is not the situation on dev — there the same table is a full
    day stale. Contemporaneous counts flatter every recency-flavoured feature.
    """
    # Act
    everything = train_click_counts(BEHAVIORS)
    before_only = train_click_counts(BEHAVIORS, restrict_to_period=TEMPORAL_FIT)

    # Assert
    assert sum(before_only.values()) < sum(everything.values())
    for news_id, count in before_only.items():
        assert count <= everything[news_id]


def test_restricted_counts_match_a_hand_rolled_count():
    # Arrange
    expected: dict[str, int] = {}
    for impression in load_behaviors(BEHAVIORS):
        if impression.timestamp.date() >= VALIDATION_DAY:
            continue
        for news_id, label in zip(impression.candidates, impression.labels):
            if label == 1:
                expected[news_id] = expected.get(news_id, 0) + 1

    # Act
    actual = train_click_counts(BEHAVIORS, restrict_to_period=TEMPORAL_FIT)

    # Assert
    assert dict(actual) == expected


def test_temporal_and_user_restrictions_compose():
    """Phase 2 needs both at once; neither may silently override the other."""
    # Act
    both = train_click_counts(
        BEHAVIORS, exclude_split=VALIDATION, restrict_to_period=TEMPORAL_FIT
    )
    temporal_only = train_click_counts(BEHAVIORS, restrict_to_period=TEMPORAL_FIT)

    # Assert
    assert sum(both.values()) < sum(temporal_only.values())
    for news_id, count in both.items():
        assert count <= temporal_only[news_id]


def test_unknown_period_is_rejected():
    # Act / Assert
    with pytest.raises(ValueError, match="unknown temporal split"):
        train_click_counts(BEHAVIORS, restrict_to_period="yesterday")
