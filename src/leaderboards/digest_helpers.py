"""Build leaderboard "top movers" blocks for the daily digest.

Glue code between the persisted snapshots and the email template. Pulls
the latest snapshot per (provider, metric), finds the previous snapshot
≥7 days back, computes movements, picks the top N rank changes, and
formats them as dicts the Jinja template can render directly.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import TypedDict

from sqlmodel import Session, select

from src.leaderboards.base import (
    LeaderboardSnapshot,
    compute_movement,
    latest_snapshot,
    previous_snapshot,
    top_movers,
)

logger = logging.getLogger(__name__)


PROVIDER_LABELS: dict[str, str] = {
    "artificial_analysis": "Artificial Analysis",
}

METRIC_LABELS: dict[str, str] = {
    "intelligence": "Intelligence",
    "output_speed": "Output Speed",
    "price": "Price",
}


class MoverItem(TypedDict):
    model: str
    organization: str | None
    rank_now: int
    rank_delta: int
    score_now: float
    score_delta: float
    direction: str  # "up" | "down" — for arrow rendering


class MoverGroup(TypedDict):
    provider: str  # human label
    metric: str  # human label
    captured_at: str  # ISO date for "as of"
    entries: list[MoverItem]  # NOTE: not "items" — collides with dict.items() in Jinja


def _format_movement(provider_id: str, metric_id: str, current: LeaderboardSnapshot,
                     previous: LeaderboardSnapshot | None, top_n: int) -> MoverGroup | None:
    """Compute top movers for one snapshot pair, format as a template-ready dict.

    Returns None if there's no meaningful movement to show (e.g. no previous
    snapshot, or every model is brand-new in the current one).
    """
    movements = compute_movement(current, previous)
    top = top_movers(movements, n=top_n)
    if not top:
        return None
    return MoverGroup(
        provider=PROVIDER_LABELS.get(provider_id, provider_id),
        metric=METRIC_LABELS.get(metric_id, metric_id),
        captured_at=current.captured_at.date().isoformat(),
        entries=[
            MoverItem(
                model=m.model,
                organization=None,  # Movement doesn't carry org; cheap to add later
                rank_now=m.rank_now,
                rank_delta=m.rank_delta,
                score_now=round(m.score_now, 2),
                score_delta=round(m.score_delta, 2),
                direction="up" if m.rank_delta < 0 else "down",
            )
            for m in top
        ],
    )


def build_movers(
    session: Session, days_back: int = 7, top_n: int = 3
) -> list[MoverGroup]:
    """Build the list of mover groups for inclusion in a digest.

    One group per (provider, metric) pair that has a current snapshot and a
    previous one ≥days_back days older. Groups with no movement are dropped.
    """
    latest_per_pair = _latest_snapshot_per_pair(session)
    groups: list[MoverGroup] = []
    for snap in latest_per_pair:
        cutoff = snap.captured_at - timedelta(days=days_back)
        prev = previous_snapshot(session, snap.provider, snap.metric, before=cutoff)
        # If no snapshot that far back, fall back to whatever's just before
        # the current one — better some signal than none.
        if prev is None:
            prev = previous_snapshot(session, snap.provider, snap.metric,
                                     before=snap.captured_at)
        group = _format_movement(snap.provider, snap.metric, snap, prev, top_n)
        if group is not None:
            groups.append(group)
    return groups


def _latest_snapshot_per_pair(session: Session) -> list[LeaderboardSnapshot]:
    """One snapshot per (provider, metric), the most recent of each."""
    stmt = select(LeaderboardSnapshot).order_by(
        LeaderboardSnapshot.captured_at.desc()
    )
    seen: set[tuple[str, str]] = set()
    out: list[LeaderboardSnapshot] = []
    for row in session.exec(stmt).all():
        key = (row.provider, row.metric)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
