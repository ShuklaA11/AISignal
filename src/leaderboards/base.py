"""Leaderboard provider abstraction + movement computation.

Each provider (LMSYS Arena, Artificial Analysis, etc.) subclasses
LeaderboardProvider and returns a Snapshot containing a list of
Ranking entries. Snapshots persist to the leaderboard_snapshots
table; movement between two snapshots is computed on the fly.

The Ranking schema is intentionally narrow — model, rank, score,
organization — so heterogeneous providers (Elo, throughput, price)
share the same shape. Provider-specific extras land in extras dict.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from sqlmodel import Session, select

from src.storage.models import LeaderboardSnapshot, utcnow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ranking:
    """One model's position in a leaderboard at a point in time."""

    model: str
    rank: int
    score: float
    organization: str | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"model": self.model, "rank": self.rank, "score": self.score}
        if self.organization:
            d["organization"] = self.organization
        if self.extras:
            d["extras"] = dict(self.extras)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Ranking":
        return cls(
            model=d["model"],
            rank=int(d["rank"]),
            score=float(d["score"]),
            organization=d.get("organization"),
            extras=dict(d.get("extras") or {}),
        )


@dataclass
class Snapshot:
    """A complete leaderboard at one point in time for one provider/metric."""

    provider: str
    metric: str
    rankings: list[Ranking]
    captured_at: datetime = field(default_factory=utcnow)
    source_url: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Movement:
    """A model's rank/score delta between two snapshots."""

    model: str
    rank_delta: int  # negative = climbed (rank 3 -> 1 = -2)
    score_delta: float
    rank_now: int
    score_now: float
    rank_before: int | None  # None if model is newly listed
    score_before: float | None


class LeaderboardProvider(ABC):
    """All leaderboard providers implement this interface."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable identifier (e.g. 'lmsys_arena', 'artificial_analysis')."""
        ...

    @abstractmethod
    async def fetch_snapshot(self) -> Iterable[Snapshot]:
        """Return one Snapshot per metric this provider tracks.

        A provider may emit a single Snapshot (e.g. LMSYS Arena only
        reports Elo) or several (Artificial Analysis tracks speed +
        price + quality).
        """
        ...

    async def safe_fetch(self) -> list[Snapshot]:
        """Wrap fetch_snapshot with error handling. Never crashes the
        scheduled snapshot job."""
        try:
            snapshots = list(await self.fetch_snapshot())
            logger.info(
                f"[{self.provider_name}] fetched {len(snapshots)} snapshot(s)"
            )
            return snapshots
        except Exception as e:
            logger.warning(f"[{self.provider_name}] fetch failed: {e}")
            return []


def persist_snapshot(session: Session, snapshot: Snapshot) -> LeaderboardSnapshot:
    """Save a snapshot to the leaderboard_snapshots table."""
    row = LeaderboardSnapshot(
        provider=snapshot.provider,
        metric=snapshot.metric,
        captured_at=snapshot.captured_at,
        source_url=snapshot.source_url,
        notes=snapshot.notes,
    )
    row.rankings = [r.to_dict() for r in snapshot.rankings]
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def latest_snapshot(
    session: Session, provider: str, metric: str
) -> LeaderboardSnapshot | None:
    """Most recent persisted snapshot for one provider/metric."""
    stmt = (
        select(LeaderboardSnapshot)
        .where(LeaderboardSnapshot.provider == provider)
        .where(LeaderboardSnapshot.metric == metric)
        .order_by(LeaderboardSnapshot.captured_at.desc())
        .limit(1)
    )
    return session.exec(stmt).first()


def previous_snapshot(
    session: Session, provider: str, metric: str, before: datetime
) -> LeaderboardSnapshot | None:
    """Most recent snapshot strictly before a given timestamp."""
    stmt = (
        select(LeaderboardSnapshot)
        .where(LeaderboardSnapshot.provider == provider)
        .where(LeaderboardSnapshot.metric == metric)
        .where(LeaderboardSnapshot.captured_at < before)
        .order_by(LeaderboardSnapshot.captured_at.desc())
        .limit(1)
    )
    return session.exec(stmt).first()


def compute_movement(
    current: LeaderboardSnapshot | Snapshot,
    previous: LeaderboardSnapshot | Snapshot | None,
) -> list[Movement]:
    """Compute per-model rank/score deltas between two snapshots.

    Returns one Movement per model in `current`. Models not present in
    `previous` get rank_before=None and rank_delta computed as 0 (so
    they don't dominate the "biggest movers" sort by virtue of being
    new). If `previous` is None or empty, returns Movements where all
    deltas are 0.
    """
    cur_rankings = _coerce_rankings(current)
    prev_rankings = _coerce_rankings(previous) if previous else []
    prev_by_model = {r.model: r for r in prev_rankings}

    movements: list[Movement] = []
    for r in cur_rankings:
        prev = prev_by_model.get(r.model)
        if prev is None:
            movements.append(
                Movement(
                    model=r.model,
                    rank_delta=0,
                    score_delta=0.0,
                    rank_now=r.rank,
                    score_now=r.score,
                    rank_before=None,
                    score_before=None,
                )
            )
        else:
            movements.append(
                Movement(
                    model=r.model,
                    rank_delta=r.rank - prev.rank,
                    score_delta=r.score - prev.score,
                    rank_now=r.rank,
                    score_now=r.score,
                    rank_before=prev.rank,
                    score_before=prev.score,
                )
            )
    return movements


def _coerce_rankings(
    snapshot: LeaderboardSnapshot | Snapshot,
) -> list[Ranking]:
    if isinstance(snapshot, Snapshot):
        return list(snapshot.rankings)
    return [Ranking.from_dict(d) for d in snapshot.rankings]


def top_movers(movements: Iterable[Movement], n: int = 3) -> list[Movement]:
    """Pick the N models with the biggest absolute rank movement.

    New entries (rank_before is None) are excluded since they have no
    movement signal.
    """
    eligible = [m for m in movements if m.rank_before is not None and m.rank_delta != 0]
    eligible.sort(key=lambda m: abs(m.rank_delta), reverse=True)
    return eligible[:n]
