#!/usr/bin/env python3
"""Manual trigger: fetch + persist leaderboard snapshots from every provider."""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings
from src.leaderboards import run_all_snapshots
from src.storage.database import init_db, session_scope

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


async def main():
    settings = load_settings()
    init_db(settings.database_url)
    with session_scope(settings.database_url) as session:
        n = await run_all_snapshots(session)
    print(f"\nDone! Wrote {n} snapshot(s).")


if __name__ == "__main__":
    asyncio.run(main())
