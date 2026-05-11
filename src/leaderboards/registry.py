"""Registry + snapshot runner for leaderboard providers.

Keeps a single source of truth for which providers ship. The scheduler
imports run_all_snapshots which fans out to every registered provider,
persists snapshots, and never crashes the parent job.
"""
from __future__ import annotations

import logging

from sqlmodel import Session

from src.leaderboards.artificial_analysis import ArtificialAnalysisProvider
from src.leaderboards.base import LeaderboardProvider, persist_snapshot

logger = logging.getLogger(__name__)


def all_providers() -> list[LeaderboardProvider]:
    """Return one instance per registered provider.

    Adding a new provider = one entry here. Future providers (LMSYS Arena,
    LiveBench, SWE-bench, Aider polyglot) join this list once their fetch
    path is implemented and tested.
    """
    return [
        ArtificialAnalysisProvider(),
    ]


async def run_all_snapshots(session: Session) -> int:
    """Fetch + persist snapshots from every registered provider.

    Returns the count of snapshots written. Per-provider errors are
    logged but don't stop other providers from running.
    """
    written = 0
    for provider in all_providers():
        try:
            snapshots = await provider.safe_fetch()
            for snap in snapshots:
                persist_snapshot(session, snap)
                written += 1
            logger.info(
                f"[{provider.provider_name}] persisted {len(snapshots)} snapshot(s)"
            )
        except Exception as e:
            logger.warning(f"[{provider.provider_name}] persist failed: {e}")
    return written
