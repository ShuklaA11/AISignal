"""Tests for build_movers — the glue between snapshots and the digest template."""
from datetime import datetime, timedelta

import pytest
from sqlmodel import Session, SQLModel, create_engine

from src.leaderboards import Ranking, Snapshot, persist_snapshot
from src.leaderboards.digest_helpers import (
    METRIC_LABELS,
    PROVIDER_LABELS,
    build_movers,
)


@pytest.fixture
def session() -> Session:
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        yield s


def _snap(provider, metric, rankings, when):
    """Helper to persist a snapshot directly with a specific captured_at."""
    return Snapshot(
        provider=provider,
        metric=metric,
        rankings=[Ranking(m, r, s, None) for m, r, s in rankings],
        captured_at=when,
    )


def test_empty_db_returns_no_groups(session: Session):
    assert build_movers(session) == []


def test_only_one_snapshot_returns_no_groups(session: Session):
    """No previous snapshot = no movement signal = no group."""
    persist_snapshot(
        session,
        _snap("artificial_analysis", "intelligence",
              [("A", 1, 60.0), ("B", 2, 55.0)], datetime(2026, 5, 11)),
    )
    assert build_movers(session) == []


def test_two_snapshots_produces_one_group(session: Session):
    persist_snapshot(
        session,
        _snap("artificial_analysis", "intelligence",
              [("A", 3, 50.0), ("B", 1, 60.0), ("C", 2, 55.0)],
              datetime(2026, 5, 4)),
    )
    persist_snapshot(
        session,
        _snap("artificial_analysis", "intelligence",
              [("A", 1, 60.0), ("B", 2, 55.0), ("C", 3, 50.0)],
              datetime(2026, 5, 11)),
    )
    groups = build_movers(session, top_n=3)
    assert len(groups) == 1
    g = groups[0]
    assert g["provider"] == PROVIDER_LABELS["artificial_analysis"]
    assert g["metric"] == METRIC_LABELS["intelligence"]
    # A climbed 3 -> 1
    a_mover = next(i for i in g["entries"] if i["model"] == "A")
    assert a_mover["rank_delta"] == -2
    assert a_mover["direction"] == "up"


def test_separate_metrics_get_separate_groups(session: Session):
    for metric in ("intelligence", "output_speed"):
        persist_snapshot(
            session,
            _snap("artificial_analysis", metric,
                  [("A", 2, 40.0), ("B", 1, 60.0)],
                  datetime(2026, 5, 4)),
        )
        persist_snapshot(
            session,
            _snap("artificial_analysis", metric,
                  [("A", 1, 60.0), ("B", 2, 40.0)],
                  datetime(2026, 5, 11)),
        )

    groups = build_movers(session)
    assert {g["metric"] for g in groups} == {
        METRIC_LABELS["intelligence"],
        METRIC_LABELS["output_speed"],
    }


def test_falls_back_to_most_recent_prior_if_no_week_old(session: Session):
    """If no snapshot is ≥7 days old, use whatever's immediately before."""
    persist_snapshot(
        session,
        _snap("artificial_analysis", "intelligence",
              [("A", 2, 50.0)], datetime(2026, 5, 10)),
    )
    persist_snapshot(
        session,
        _snap("artificial_analysis", "intelligence",
              [("A", 1, 55.0)], datetime(2026, 5, 11)),
    )
    groups = build_movers(session, days_back=7)
    assert len(groups) == 1
    assert groups[0]["entries"][0]["rank_delta"] == -1


def test_unknown_provider_metric_falls_through_with_raw_id(session: Session):
    """Provider/metric not in label map still renders, just with the raw id."""
    persist_snapshot(
        session,
        _snap("future_lmsys", "elo",
              [("A", 2, 1200.0)], datetime(2026, 5, 4)),
    )
    persist_snapshot(
        session,
        _snap("future_lmsys", "elo",
              [("A", 1, 1300.0)], datetime(2026, 5, 11)),
    )
    groups = build_movers(session)
    assert groups[0]["provider"] == "future_lmsys"
    assert groups[0]["metric"] == "elo"
