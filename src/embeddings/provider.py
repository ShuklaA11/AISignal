"""Embedding generation via Ollama through LiteLLM."""

import logging

import litellm
import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "ollama/mxbai-embed-large"
EMBEDDING_DIM = 1024


async def generate_embedding(text: str) -> np.ndarray | None:
    """Generate a single embedding vector for the given text."""
    try:
        response = await litellm.aembedding(
            model=EMBEDDING_MODEL,
            input=[text],
        )
        vector = response.data[0]["embedding"]
        return np.array(vector, dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


async def generate_embeddings_batch(
    texts: list[str], batch_size: int = 10
) -> list[np.ndarray | None]:
    """Generate embeddings for a batch of texts."""
    results: list[np.ndarray | None] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = await litellm.aembedding(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            for item in response.data:
                results.append(np.array(item["embedding"], dtype=np.float32))
        except Exception as e:
            logger.error(f"Batch embedding failed for batch {i // batch_size}: {e}")
            results.extend([None] * len(batch))
    return results
