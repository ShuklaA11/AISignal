"""Impression timestamps, and the temporal validation split built on them.

MINDsmall_train spans Nov 9-14 2019 and MINDsmall_dev is Nov 15 only, so dev's
defining property is that everything it is scored with was computed on a
disjoint earlier period. The validation split has to reproduce that, or any
recency- or popularity-flavoured feature is measured under contemporaneous
counts and looks better on validation than it will on dev.
"""

from datetime import date, datetime
from pathlib import Path

import pytest

from src.eval.mind_data import load_behaviors, parse_impression_time
from src.eval.splits import (
    TEMPORAL_FIT,
    TEMPORAL_VALIDATION,
    VALIDATION_DAY,
    filter_by_day,
    temporal_split_for,
)

BEHAVIORS = Path("data/mind/train/behaviors.tsv")

needs_data = pytest.mark.skipif(
    not BEHAVIORS.exists(), reason="MIND data not present in this checkout"
)


# ── Timestamp parsing ────────────────────────────────────────────────


def test_parses_the_am_pm_format_mind_ships():
    # Act
    parsed = parse_impression_time("11/11/2019 9:05:58 AM")

    # Assert
    assert parsed == datetime(2019, 11, 11, 9, 5, 58)


def test_parses_afternoon_times_as_twenty_four_hour():
    # Act
    parsed = parse_impression_time("11/12/2019 6:11:30 PM")

    # Assert
    assert parsed == datetime(2019, 11, 12, 18, 11, 30)


def test_parses_single_digit_days_and_months():
    # Act
    parsed = parse_impression_time("11/9/2019 6:11:24 AM")

    # Assert
    assert parsed == datetime(2019, 11, 9, 6, 11, 24)


def test_an_unparseable_timestamp_is_rejected_rather_than_defaulted():
    """Silently defaulting would put rows in the wrong split without a trace."""
    # Act / Assert
    with pytest.raises(ValueError, match="timestamp"):
        parse_impression_time("not a date")


# ── Temporal split ───────────────────────────────────────────────────


def test_the_last_train_day_is_the_validation_day():
    # Assert
    assert VALIDATION_DAY == date(2019, 11, 14)


def test_impressions_on_the_validation_day_are_held_out():
    # Act
    assigned = temporal_split_for(datetime(2019, 11, 14, 7, 1, 48))

    # Assert
    assert assigned == TEMPORAL_VALIDATION


def test_earlier_impressions_belong_to_the_fit_period():
    # Act / Assert
    for day in (9, 10, 11, 12, 13):
        assert temporal_split_for(datetime(2019, 11, day, 12, 0)) == TEMPORAL_FIT


def test_the_split_boundary_falls_at_midnight():
    # Act
    just_before = temporal_split_for(datetime(2019, 11, 13, 23, 59, 59))
    just_after = temporal_split_for(datetime(2019, 11, 14, 0, 0, 0))

    # Assert
    assert just_before == TEMPORAL_FIT
    assert just_after == TEMPORAL_VALIDATION


def test_dev_day_impressions_are_not_treated_as_fit_data():
    """Nov 15 is dev; it must never be mistaken for training material."""
    # Act
    assigned = temporal_split_for(datetime(2019, 11, 15, 8, 0))

    # Assert
    assert assigned != TEMPORAL_FIT


# ── Integration with the loader ──────────────────────────────────────


@needs_data
def test_loaded_impressions_carry_a_parsed_timestamp():
    # Act
    first = next(load_behaviors(BEHAVIORS))

    # Assert
    assert isinstance(first.timestamp, datetime)
    assert first.timestamp.year == 2019


@needs_data
def test_the_validation_day_holds_the_expected_impression_count():
    # Act
    validation = list(filter_by_day(load_behaviors(BEHAVIORS), TEMPORAL_VALIDATION))

    # Assert
    assert len(validation) == 30_270


@needs_data
def test_fit_and_validation_days_partition_the_train_file():
    # Act
    fit = sum(1 for _ in filter_by_day(load_behaviors(BEHAVIORS), TEMPORAL_FIT))
    validation = sum(
        1 for _ in filter_by_day(load_behaviors(BEHAVIORS), TEMPORAL_VALIDATION)
    )

    # Assert
    assert fit + validation == 156_965


@needs_data
def test_no_fit_impression_falls_on_or_after_the_validation_day():
    """The property that makes the temporal holdout honest."""
    # Act
    fit = filter_by_day(load_behaviors(BEHAVIORS), TEMPORAL_FIT)

    # Assert
    assert all(i.timestamp.date() < VALIDATION_DAY for i in fit)


def test_filter_by_day_rejects_an_unknown_period():
    # Act / Assert
    with pytest.raises(ValueError, match="unknown temporal split"):
        list(filter_by_day(iter([]), "someday"))
