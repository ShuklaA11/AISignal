"""Backfill embeddings for all processed articles."""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.config import load_settings
from src.embeddings.pipeline import run_embedding_generation  # noqa: E402 — registers ArticleEmbedding
from src.storage.database import get_session, init_db


async def main():
    settings = load_settings()
    init_db(settings.database_url)
    session = get_session()
    try:
        count = await run_embedding_generation(session, batch_size=20)
        print(f"Done: generated {count} embeddings")
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
