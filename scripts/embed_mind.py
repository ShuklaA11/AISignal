#!/usr/bin/env python3
"""Build the article-embedding cache for the MIND benchmark evaluation.

Embeds the union of the MIND train and dev news files (~65k items) with the same
model the production pipeline uses (mxbai-embed-large via Ollama) and writes the
vectors to a single .npz. The run checkpoints periodically and resumes from an
existing cache, so an interrupted pass does not start over.

Usage:
    python scripts/embed_mind.py
    python scripts/embed_mind.py --limit 500        # quick smoke run
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.provider import EMBEDDING_DIM, generate_embeddings_batch
from src.eval.mind_data import load_news

DEFAULT_ROOT = Path("data/mind")
DEFAULT_OUTPUT = DEFAULT_ROOT / "embeddings.npz"
CHECKPOINT_EVERY = 5_000


def load_cache(path: Path) -> dict[str, np.ndarray]:
    """Load an existing embedding cache, or return an empty mapping."""
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        ids = data["news_ids"]
        vectors = data["embeddings"]
    return {str(news_id): vectors[i] for i, news_id in enumerate(ids)}


def save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    """Write the cache atomically so an interrupted save cannot corrupt it."""
    news_ids = sorted(cache)
    matrix = np.stack([cache[n] for n in news_ids]).astype(np.float32)
    # np.savez appends ".npz" unless the name already ends with it, so the temp
    # name must keep that suffix or the rename below targets the wrong file.
    temp_path = path.with_name(path.name + ".tmp.npz")
    np.savez(temp_path, news_ids=np.array(news_ids), embeddings=matrix)
    temp_path.replace(path)


async def embed_all(
    root: Path,
    output: Path,
    batch_size: int,
    limit: int | None,
) -> None:
    news = {**load_news(root / "train/news.tsv"), **load_news(root / "dev/news.tsv")}
    news_ids = sorted(news)
    if limit is not None:
        news_ids = news_ids[:limit]

    cache = load_cache(output)
    pending = [n for n in news_ids if n not in cache]
    print(f"{len(news_ids)} news items | {len(cache)} cached | {len(pending)} to embed")
    if not pending:
        print("Nothing to do.")
        return

    started = time.time()
    failures = 0
    since_checkpoint = 0

    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset : offset + batch_size]
        vectors = await generate_embeddings_batch(
            [news[n].embedding_text for n in chunk],
            batch_size=batch_size,
        )
        for news_id, vector in zip(chunk, vectors):
            if vector is None or vector.shape != (EMBEDDING_DIM,):
                failures += 1
                continue
            cache[news_id] = vector.astype(np.float32)

        since_checkpoint += len(chunk)
        done = offset + len(chunk)
        if since_checkpoint >= CHECKPOINT_EVERY:
            save_cache(output, cache)
            since_checkpoint = 0
            rate = done / (time.time() - started)
            remaining = (len(pending) - done) / rate / 60 if rate > 0 else 0
            print(
                f"  {done}/{len(pending)} embedded "
                f"({rate:.0f}/s, ~{remaining:.0f} min left, {failures} failed)"
            )

    save_cache(output, cache)
    elapsed = (time.time() - started) / 60
    print(f"Wrote {len(cache)} embeddings to {output} in {elapsed:.1f} min")
    if failures:
        print(
            f"WARNING: {failures} items failed to embed and are absent from the cache"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed MIND news for offline eval")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(embed_all(args.root, args.output, args.batch_size, args.limit))


if __name__ == "__main__":
    main()
