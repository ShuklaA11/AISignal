"""Backfill metrics from historical impression data."""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.config import load_settings
from src.metrics.calculator import compute_daily_metrics
from src.storage.database import get_session, init_db
from src.storage.models import ScoringMetric  # noqa: F401 — register table
from src.storage.queries import get_active_users


def main():
    settings = load_settings()
    init_db(settings.database_url)
    session = get_session()
    try:
        users = get_active_users(session)
        today = date.today()
        count = 0
        for user in users:
            for days_ago in range(30, -1, -1):
                d = today - timedelta(days=days_ago)
                compute_daily_metrics(session, user.id, d)
                count += 1
        print(f"Done: computed {count} daily metric records for {len(users)} users")
    finally:
        session.close()


if __name__ == "__main__":
    main()
