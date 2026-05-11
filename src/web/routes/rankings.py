"""Leaderboard rankings dashboard.

Renders the current state of every persisted provider/metric pair, plus
a weekly movers summary. Pure read endpoint — backed by the snapshots
the scheduler writes daily.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlmodel import select

from src.leaderboards import latest_snapshot
from src.leaderboards.digest_helpers import (
    METRIC_LABELS,
    PROVIDER_LABELS,
    build_movers,
)
from src.storage.database import session_scope
from src.storage.models import LeaderboardSnapshot
from src.web.template_engine import templates

router = APIRouter()


@dataclass
class RankingBoard:
    """One table on the rankings page: provider + metric + the top N entries."""

    provider_id: str
    provider_label: str
    metric_id: str
    metric_label: str
    captured_at: str
    rows: list[dict]  # {model, organization, rank, score}


def _board(snap: LeaderboardSnapshot, top_n: int = 10) -> RankingBoard:
    return RankingBoard(
        provider_id=snap.provider,
        provider_label=PROVIDER_LABELS.get(snap.provider, snap.provider),
        metric_id=snap.metric,
        metric_label=METRIC_LABELS.get(snap.metric, snap.metric),
        captured_at=snap.captured_at.date().isoformat(),
        rows=snap.rankings[:top_n],
    )


def _distinct_pairs(session) -> list[LeaderboardSnapshot]:
    """One snapshot per (provider, metric), most recent."""
    rows = session.exec(
        select(LeaderboardSnapshot).order_by(LeaderboardSnapshot.captured_at.desc())
    ).all()
    seen: set[tuple[str, str]] = set()
    out = []
    for r in rows:
        key = (r.provider, r.metric)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


@router.get(
    "/rankings",
    response_class=HTMLResponse,
    summary="Model rankings dashboard",
    description="Per-provider leaderboards with current top 10 and weekly movers.",
)
async def rankings_page(request: Request):
    with session_scope() as session:
        snaps = _distinct_pairs(session)
        boards = [_board(s) for s in snaps]
        movers = build_movers(session, days_back=7, top_n=5)

    return templates.TemplateResponse(
        "rankings.html",
        {
            "request": request,
            "boards": boards,
            "movers": movers,
            "has_data": bool(boards),
        },
    )
