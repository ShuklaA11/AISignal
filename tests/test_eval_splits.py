"""Tests for the user-level tri-split of MINDsmall_train."""

from dataclasses import dataclass

import pytest

from src.eval.splits import (
    EARLY_STOP,
    FIT,
    SPLIT_NAMES,
    VALIDATION,
    filter_impressions,
    is_validation_user,
    split_for_user,
    user_fraction,
)

SAMPLE_USERS = [f"U{index}" for index in range(20_000)]


@dataclass(frozen=True)
class FakeImpression:
    """Minimal stand-in exposing the only attribute the filter reads."""

    user_id: str


def test_every_user_lands_in_exactly_one_known_split():
    # Act
    assignments = {split_for_user(user) for user in SAMPLE_USERS}

    # Assert
    assert assignments <= set(SPLIT_NAMES)
    assert assignments == set(SPLIT_NAMES)


def test_assignment_is_deterministic_for_a_fixed_seed():
    # Act
    first = [split_for_user(user, seed=7) for user in SAMPLE_USERS[:100]]
    second = [split_for_user(user, seed=7) for user in SAMPLE_USERS[:100]]

    # Assert
    assert first == second


def test_changing_the_seed_reshuffles_the_assignment():
    # Act
    default_seed = [split_for_user(user, seed=1) for user in SAMPLE_USERS[:500]]
    other_seed = [split_for_user(user, seed=2) for user in SAMPLE_USERS[:500]]

    # Assert
    assert default_seed != other_seed


def test_split_proportions_are_close_to_the_configured_bounds():
    # Arrange
    expected = {FIT: 0.70, EARLY_STOP: 0.10, VALIDATION: 0.20}

    # Act
    counts = {name: 0 for name in SPLIT_NAMES}
    for user in SAMPLE_USERS:
        counts[split_for_user(user)] += 1

    # Assert
    for name, share in expected.items():
        assert counts[name] / len(SAMPLE_USERS) == pytest.approx(share, abs=0.02)


def test_user_fraction_stays_within_the_unit_interval():
    # Act
    fractions = [user_fraction(user) for user in SAMPLE_USERS[:1_000]]

    # Assert
    assert all(0.0 <= value < 1.0 for value in fractions)


def test_is_validation_user_agrees_with_split_for_user():
    # Act / Assert
    for user in SAMPLE_USERS[:1_000]:
        assert is_validation_user(user) == (split_for_user(user) == VALIDATION)


def test_a_validation_user_never_appears_in_a_training_split():
    """The property that makes the held-out measurement meaningful."""
    # Act
    validation_users = {u for u in SAMPLE_USERS if split_for_user(u) == VALIDATION}
    training_users = {u for u in SAMPLE_USERS if split_for_user(u) in (FIT, EARLY_STOP)}

    # Assert
    assert validation_users
    assert not (validation_users & training_users)


def test_empty_user_id_is_rejected():
    # Act / Assert
    with pytest.raises(ValueError, match="user_id"):
        split_for_user("")


def test_filter_impressions_keeps_only_the_requested_split():
    # Arrange
    impressions = [FakeImpression(user) for user in SAMPLE_USERS[:2_000]]

    # Act
    kept = list(filter_impressions(iter(impressions), VALIDATION))

    # Assert
    assert kept
    assert all(split_for_user(i.user_id) == VALIDATION for i in kept)


def test_filtering_every_split_partitions_the_input():
    # Arrange
    impressions = [FakeImpression(user) for user in SAMPLE_USERS[:2_000]]

    # Act
    counts = {
        name: len(list(filter_impressions(iter(impressions), name)))
        for name in SPLIT_NAMES
    }

    # Assert
    assert sum(counts.values()) == len(impressions)


def test_filter_impressions_rejects_an_unknown_split():
    # Act / Assert
    with pytest.raises(ValueError, match="unknown split"):
        list(filter_impressions(iter([]), "test"))
