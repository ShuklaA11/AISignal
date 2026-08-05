"""Tests for the paired-difference significance harness."""

import numpy as np
import pytest
from src.eval.paired_stats import PairedDelta, paired_delta


def test_identical_methods_produce_a_zero_delta_that_is_not_significant():
    # Arrange
    scores = np.random.RandomState(0).uniform(0, 1, size=500)

    # Act
    result = paired_delta(scores, scores, seed=1)

    # Assert
    assert result.mean == pytest.approx(0.0)
    assert result.ci_low <= 0.0 <= result.ci_high
    assert not result.is_significant


def test_constant_uplift_is_recovered_and_flagged_significant():
    # Arrange
    baseline = np.random.RandomState(0).uniform(0, 1, size=500)
    variant = baseline + 0.05

    # Act
    result = paired_delta(baseline, variant, seed=1)

    # Assert
    assert result.mean == pytest.approx(0.05, abs=1e-9)
    assert result.is_significant
    assert result.ci_low > 0.0


def test_uplift_buried_in_shared_noise_is_still_detected():
    """The whole point of pairing: a small effect under large shared variance.

    Both methods see the same per-impression difficulty term, which dwarfs the
    effect. Unpaired comparison cannot see through it; paired comparison can.
    """
    # Arrange
    rng = np.random.RandomState(7)
    difficulty = rng.uniform(0, 1, size=20_000)
    baseline = difficulty
    variant = difficulty + 0.003

    # Act
    paired = paired_delta(baseline, variant, seed=1)
    unpaired_standard_error = np.sqrt(
        baseline.var(ddof=1) / baseline.size + variant.var(ddof=1) / variant.size
    )

    # Assert
    assert paired.is_significant
    assert paired.standard_error < unpaired_standard_error / 10


def test_noise_only_difference_is_reported_as_not_significant():
    # Arrange
    rng = np.random.RandomState(3)
    baseline = rng.uniform(0, 1, size=5_000)
    variant = baseline + rng.normal(0, 0.1, size=5_000)

    # Act
    result = paired_delta(baseline, variant, seed=1)

    # Assert
    assert not result.is_significant


def test_confidence_interval_brackets_the_mean():
    # Arrange
    rng = np.random.RandomState(11)
    baseline = rng.uniform(0, 1, size=2_000)
    variant = baseline + rng.normal(0.02, 0.05, size=2_000)

    # Act
    result = paired_delta(baseline, variant, seed=1)

    # Assert
    assert result.ci_low < result.mean < result.ci_high


def test_a_wider_confidence_level_produces_a_wider_interval():
    # Arrange
    rng = np.random.RandomState(5)
    baseline = rng.uniform(0, 1, size=2_000)
    variant = baseline + rng.normal(0.01, 0.05, size=2_000)

    # Act
    narrow = paired_delta(baseline, variant, confidence=0.80, seed=1)
    wide = paired_delta(baseline, variant, confidence=0.99, seed=1)

    # Assert
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_result_is_reproducible_for_a_fixed_seed():
    # Arrange
    rng = np.random.RandomState(13)
    baseline = rng.uniform(0, 1, size=1_000)
    variant = baseline + rng.normal(0.01, 0.05, size=1_000)

    # Act
    first = paired_delta(baseline, variant, seed=99)
    second = paired_delta(baseline, variant, seed=99)

    # Assert
    assert first == second


def test_mismatched_lengths_are_rejected():
    # Arrange
    baseline = [0.1, 0.2, 0.3]
    variant = [0.1, 0.2]

    # Act / Assert
    with pytest.raises(ValueError, match="same length"):
        paired_delta(baseline, variant)


def test_empty_input_is_rejected():
    # Act / Assert
    with pytest.raises(ValueError, match="at least"):
        paired_delta([], [])


def test_accepts_plain_sequences_as_well_as_arrays():
    # Arrange
    baseline = [0.1, 0.2, 0.3, 0.4] * 50
    variant = [0.2, 0.3, 0.4, 0.5] * 50

    # Act
    result = paired_delta(baseline, variant, seed=1)

    # Assert
    assert isinstance(result, PairedDelta)
    assert result.n == 200
    assert result.mean == pytest.approx(0.1)
