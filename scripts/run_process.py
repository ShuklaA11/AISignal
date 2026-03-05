#!/usr/bin/env python3
"""Manual trigger: process raw articles with LLM."""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.processor import run_processing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


async def main():
    processed_count = await run_processing()
    print(f"\nDone! Processed {processed_count} articles.")


if __name__ == "__main__":
    asyncio.run(main())
