"""Semantic similarity computation for recommendation."""

import logging

import numpy as np
from sqlmodel import Session, select

from src.storage.models import ReadArticle, SavedArticle

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_user_embedding(
    session: Session, user_id: int, embedding_lookup: dict[int, np.ndarray]
) -> np.ndarray | None:
    """Compute user embedding, preferring the learned model over weighted average.

    If a trained UserTower model exists with sufficient data, uses it for
    a learned embedding. Otherwise falls back to weighted average of
    engaged article vectors.
    """
    # Try learned model first
    try:
        from src.embeddings.user_model_store import load_user_model
        from src.embeddings.user_tower import build_user_features, compute_learned_user_embedding

        model = load_user_model(session, user_id)
        if model is not None:
            features = build_user_features(session, user_id, embedding_lookup)
            if features is not None:
                learned_emb = compute_learned_user_embedding(model, features)
                logger.debug(f"Using learned embedding for user {user_id}")
                return learned_emb
    except Exception as e:
        logger.warning(f"Learned embedding failed for user {user_id}, falling back: {e}")

    # Fallback: weighted average
    return _weighted_average_embedding(session, user_id, embedding_lookup)


def _weighted_average_embedding(
    session: Session, user_id: int, embedding_lookup: dict[int, np.ndarray]
) -> np.ndarray | None:
    """Compute a weighted average embedding from user's engaged articles.

    Weights: saved=2.0, clicked/read=1.0. Uses most recent 50 engaged articles.
    """
    saved_stmt = (
        select(SavedArticle.article_id)
        .where(SavedArticle.user_id == user_id)
        .order_by(SavedArticle.saved_at.desc())
        .limit(50)
    )
    saved_ids = set(session.exec(saved_stmt).all())

    read_stmt = (
        select(ReadArticle.article_id)
        .where(ReadArticle.user_id == user_id)
        .order_by(ReadArticle.read_at.desc())
        .limit(50)
    )
    read_ids = set(session.exec(read_stmt).all())

    all_engaged_ids = saved_ids | read_ids
    if not all_engaged_ids:
        return None

    vectors = []
    weights = []
    for article_id in all_engaged_ids:
        if article_id not in embedding_lookup:
            continue
        vectors.append(embedding_lookup[article_id])
        weights.append(2.0 if article_id in saved_ids else 1.0)

    if not vectors:
        return None

    weights_arr = np.array(weights, dtype=np.float32)
    stacked = np.stack(vectors)
    user_vec = (stacked * weights_arr[:, np.newaxis]).sum(axis=0) / weights_arr.sum()

    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec


def compute_embedding_factor(
    article_embedding: np.ndarray | None,
    user_embedding: np.ndarray | None,
) -> float:
    """Compute the multiplicative embedding factor for scoring.

    Maps cosine similarity [-1, 1] to factor [0.3, 2.0].  The wider range
    (previously [0.5, 1.5]) gives the semantic signal meaningful leverage in
    the multiplicative scoring chain.

    Formula: factor = 1.0 + sim, then clamped to [0.3, 2.0].
    - sim == 1.0  -> factor 2.0  (very similar)
    - sim == 0.0  -> factor 1.0  (neutral / orthogonal)
    - sim == -1.0 -> factor 0.3  (opposite / dissimilar, clamp kicks in at -0.7)

    Returns 1.0 (neutral) if either embedding is missing.
    """
    if article_embedding is None or user_embedding is None:
        return 1.0
    sim = cosine_similarity(article_embedding, user_embedding)
    factor = 1.0 + (sim * 1.0)
    return max(0.3, min(2.0, factor))
