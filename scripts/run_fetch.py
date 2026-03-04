#!/usr/bin/env python3
"""Manual trigger: fetch articles from all configured sources."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.orchestrator import run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


async def main():
    new_count = await run_ingestion()
    print(f"\nDone! Stored {new_count} new articles.")


if __name__ == "__main__":
    asyncio.run(main())
