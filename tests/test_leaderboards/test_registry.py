"""Tests for the leaderboard provider registry + snapshot runner."""
from typing import Iterable

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from src.leaderboards import (
    LeaderboardProvider,
    Ranking,
    Snapshot,
    all_providers,
    run_all_snapshots,
)
from src.leaderboards import registry as registry_mod
from src.storage.models import LeaderboardSnapshot


@pytest.fixture
def session() -> Session:
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        yield s


# -- registry shape -------------------------------------------------------


def test_all_providers_returns_at_least_one():
    providers = all_providers()
    assert len(providers) >= 1
    assert all(isinstance(p, LeaderboardProvider) for p in providers)


def test_provider_names_are_unique():
    names = [p.provider_name for p in all_providers()]
    assert len(set(names)) == len(names)


# -- run_all_snapshots ----------------------------------------------------


class _FakeProvider(LeaderboardProvider):
    def __init__(self, name: str, snaps: list[Snapshot]):
        self._name = name
        self._snaps = snaps

    @property
    def provider_name(self) -> str:
        return self._name

    async def fetch_snapshot(self) -> Iterable[Snapshot]:
        return self._snaps


class _BrokenProvider(LeaderboardProvider):
    @property
    def provider_name(self) -> str:
        return "broken"

    async def fetch_snapshot(self) -> Iterable[Snapshot]:
        raise RuntimeError("simulated outage")


def _snap(name: str, metric: str) -> Snapshot:
    return Snapshot(
        provider=name,
        metric=metric,
        rankings=[Ranking("A", 1, 100.0, "Org")],
    )


@pytest.mark.asyncio
async def test_run_all_snapshots_persists_per_provider(session: Session, monkeypatch):
    fakes = [
        _FakeProvider("p1", [_snap("p1", "elo"), _snap("p1", "speed")]),
        _FakeProvider("p2", [_snap("p2", "elo")]),
    ]
    monkeypatch.setattr(registry_mod, "all_providers", lambda: fakes)

    written = await run_all_snapshots(session)
    assert written == 3

    rows = list(session.exec(select(LeaderboardSnapshot)).all())
    assert {r.provider for r in rows} == {"p1", "p2"}
    assert {(r.provider, r.metric) for r in rows} == {
        ("p1", "elo"), ("p1", "speed"), ("p2", "elo"),
    }


@pytest.mark.asyncio
async def test_run_all_snapshots_isolates_provider_failures(
    session: Session, monkeypatch
):
    """A broken provider must not prevent others from writing."""
    fakes = [
        _BrokenProvider(),
        _FakeProvider("good", [_snap("good", "elo")]),
    ]
    monkeypatch.setattr(registry_mod, "all_providers", lambda: fakes)

    written = await run_all_snapshots(session)
    assert written == 1  # only the good one
    rows = list(session.exec(select(LeaderboardSnapshot)).all())
    assert [r.provider for r in rows] == ["good"]


@pytest.mark.asyncio
async def test_run_all_snapshots_with_no_providers_returns_zero(
    session: Session, monkeypatch
):
    monkeypatch.setattr(registry_mod, "all_providers", lambda: [])
    assert await run_all_snapshots(session) == 0
