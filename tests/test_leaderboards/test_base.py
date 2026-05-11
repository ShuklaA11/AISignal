"""Tests for the leaderboard provider abstraction + movement logic."""
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from src.leaderboards import (
    LeaderboardProvider,
    Movement,
    Ranking,
    Snapshot,
    compute_movement,
    latest_snapshot,
    persist_snapshot,
    previous_snapshot,
    top_movers,
)
from src.storage.models import LeaderboardSnapshot


@pytest.fixture
def session() -> Session:
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        yield s


# -- Ranking / Snapshot dataclasses ---------------------------------------


def test_ranking_roundtrip_dict():
    r = Ranking(model="Claude Opus 4.7", rank=1, score=1284.0, organization="Anthropic")
    d = r.to_dict()
    r2 = Ranking.from_dict(d)
    assert r2 == r


def test_ranking_carries_extras():
    r = Ranking(model="GPT-6", rank=2, score=1271.0, extras={"votes": 12345})
    d = r.to_dict()
    assert d["extras"]["votes"] == 12345
    assert Ranking.from_dict(d).extras == {"votes": 12345}


# -- persist_snapshot + latest_snapshot -----------------------------------


def test_persist_snapshot_writes_full_ranking_list(session: Session):
    snap = Snapshot(
        provider="lmsys_arena",
        metric="elo",
        rankings=[
            Ranking("A", 1, 1300.0, "Anthropic"),
            Ranking("B", 2, 1280.0, "OpenAI"),
        ],
        source_url="https://lmsys.org/arena",
    )
    row = persist_snapshot(session, snap)
    assert row.id is not None
    assert row.provider == "lmsys_arena"
    assert len(row.rankings) == 2
    assert row.rankings[0]["model"] == "A"


def test_latest_snapshot_returns_most_recent(session: Session):
    base = datetime(2026, 5, 1, 10, 0, 0)
    for i, dt in enumerate([base, base + timedelta(days=1), base + timedelta(days=2)]):
        snap = Snapshot(
            provider="p1",
            metric="elo",
            rankings=[Ranking("A", i + 1, 1000.0 + i, None)],
            captured_at=dt,
        )
        persist_snapshot(session, snap)

    latest = latest_snapshot(session, "p1", "elo")
    assert latest is not None
    assert latest.captured_at == base + timedelta(days=2)


def test_previous_snapshot_returns_strictly_before(session: Session):
    base = datetime(2026, 5, 1, 10, 0, 0)
    for dt in [base, base + timedelta(days=1), base + timedelta(days=2)]:
        persist_snapshot(
            session,
            Snapshot(provider="p1", metric="elo", rankings=[], captured_at=dt),
        )
    middle = base + timedelta(days=1)
    prev = previous_snapshot(session, "p1", "elo", before=middle)
    assert prev.captured_at == base


# -- compute_movement -----------------------------------------------------


def _snap(rankings: list[tuple[str, int, float]]) -> Snapshot:
    return Snapshot(
        provider="p",
        metric="elo",
        rankings=[Ranking(m, r, s, None) for m, r, s in rankings],
    )


def test_movement_with_no_previous_returns_zero_deltas():
    cur = _snap([("A", 1, 1300.0), ("B", 2, 1280.0)])
    movements = compute_movement(cur, None)
    assert len(movements) == 2
    assert all(m.rank_delta == 0 and m.score_delta == 0 for m in movements)


def test_movement_computes_rank_delta_negative_for_climbing():
    prev = _snap([("A", 3, 1200.0), ("B", 1, 1310.0)])
    cur = _snap([("A", 1, 1310.0), ("B", 2, 1290.0)])
    by_model = {m.model: m for m in compute_movement(cur, prev)}
    # A climbed 3 -> 1, so delta is -2
    assert by_model["A"].rank_delta == -2
    # B fell 1 -> 2
    assert by_model["B"].rank_delta == 1


def test_movement_score_delta():
    prev = _snap([("A", 1, 1300.0)])
    cur = _snap([("A", 1, 1325.0)])
    m = compute_movement(cur, prev)[0]
    assert m.score_delta == pytest.approx(25.0)


def test_movement_new_entry_has_none_before():
    prev = _snap([("A", 1, 1300.0)])
    cur = _snap([("A", 1, 1310.0), ("B", 2, 1290.0)])
    by_model = {m.model: m for m in compute_movement(cur, prev)}
    assert by_model["B"].rank_before is None
    assert by_model["B"].rank_delta == 0  # not counted as movement


# -- top_movers -----------------------------------------------------------


def test_top_movers_ranks_by_abs_rank_delta():
    movements = [
        Movement("A", rank_delta=-3, score_delta=0, rank_now=1, score_now=0,
                 rank_before=4, score_before=0),
        Movement("B", rank_delta=1, score_delta=0, rank_now=2, score_now=0,
                 rank_before=1, score_before=0),
        Movement("C", rank_delta=-5, score_delta=0, rank_now=3, score_now=0,
                 rank_before=8, score_before=0),
        Movement("D", rank_delta=0, score_delta=0, rank_now=4, score_now=0,
                 rank_before=4, score_before=0),
    ]
    top = top_movers(movements, n=2)
    assert [m.model for m in top] == ["C", "A"]


def test_top_movers_excludes_new_entries_and_zero_movement():
    movements = [
        Movement("new", rank_delta=0, score_delta=0, rank_now=1, score_now=0,
                 rank_before=None, score_before=None),
        Movement("stale", rank_delta=0, score_delta=0, rank_now=2, score_now=0,
                 rank_before=2, score_before=0),
        Movement("real", rank_delta=-2, score_delta=0, rank_now=3, score_now=0,
                 rank_before=5, score_before=0),
    ]
    top = top_movers(movements, n=5)
    assert [m.model for m in top] == ["real"]


# -- LeaderboardProvider ABC ----------------------------------------------


@pytest.mark.asyncio
async def test_safe_fetch_swallows_exceptions():
    class Broken(LeaderboardProvider):
        @property
        def provider_name(self) -> str:
            return "broken"

        async def fetch_snapshot(self):
            raise RuntimeError("simulated outage")

    result = await Broken().safe_fetch()
    assert result == []


@pytest.mark.asyncio
async def test_safe_fetch_returns_snapshots_on_success():
    class Working(LeaderboardProvider):
        @property
        def provider_name(self) -> str:
            return "working"

        async def fetch_snapshot(self):
            return [Snapshot(provider="working", metric="elo", rankings=[])]

    result = await Working().safe_fetch()
    assert len(result) == 1
    assert result[0].metric == "elo"
