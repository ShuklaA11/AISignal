"""Two-tower user embedding model with contrastive training.

Learns a per-user MLP that projects interaction features into the same
1024-dim embedding space as article vectors (mxbai-embed-large).
Trained nightly via cosine embedding loss on engaged vs skipped articles.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sqlmodel import Session, select, func

from src.embeddings.provider import EMBEDDING_DIM
from src.storage.models import ArticleEmbedding, FeedImpression, ReadArticle, SavedArticle

logger = logging.getLogger(__name__)

# Feature dimensions: 3 embedding pools × 32-dim projections + 32 engagement stats
POOL_DIM = 32
FEATURE_DIM = POOL_DIM * 4  # 128 total
MIN_TRAINING_SAMPLES = 10


class UserTower(nn.Module):
    """Small MLP: 128-dim user features → 1024-dim embedding space."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, EMBEDDING_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, dim=-1)


@dataclass
class TrainingResult:
    model: UserTower
    loss: float
    num_samples: int


def _random_projection_matrix(input_dim: int, output_dim: int) -> np.ndarray:
    """Fixed random projection for dimensionality reduction (seed per dims for stability)."""
    rng = np.random.RandomState(seed=input_dim * 1000 + output_dim)
    proj = rng.randn(input_dim, output_dim).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    return proj


# Precompute the projection matrix (1024 → 32)
_PROJ = _random_projection_matrix(EMBEDDING_DIM, POOL_DIM)


def _pool_embeddings(embeddings: list[np.ndarray], max_items: int = 20) -> np.ndarray:
    """Mean-pool up to max_items embeddings and project to POOL_DIM."""
    if not embeddings:
        return np.zeros(POOL_DIM, dtype=np.float32)
    subset = embeddings[:max_items]
    mean_vec = np.mean(subset, axis=0)
    return (mean_vec @ _PROJ).astype(np.float32)


def build_user_features(
    session: Session, user_id: int, embedding_lookup: dict[int, np.ndarray],
) -> np.ndarray | None:
    """Build a 128-dim feature vector from user engagement history.

    Features:
      [0:32]   - mean saved article embeddings (projected)
      [32:64]  - mean clicked article embeddings (projected)
      [64:96]  - mean skipped article embeddings (projected)
      [96:128] - engagement statistics (click rate, save rate, etc.)
    """
    # Saved articles
    saved_ids = set(session.exec(
        select(SavedArticle.article_id)
        .where(SavedArticle.user_id == user_id)
        .order_by(SavedArticle.saved_at.desc())
        .limit(50)
    ).all())

    # Read/clicked articles
    read_ids = set(session.exec(
        select(ReadArticle.article_id)
        .where(ReadArticle.user_id == user_id)
        .order_by(ReadArticle.read_at.desc())
        .limit(50)
    ).all())
    clicked_ids = read_ids - saved_ids  # clicked but not saved

    # Skipped: shown but not engaged
    skipped_stmt = (
        select(FeedImpression.article_id)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.clicked == False)  # noqa: E712
        .where(FeedImpression.saved == False)  # noqa: E712
        .where(FeedImpression.processed == True)  # noqa: E712
        .limit(100)
    )
    skipped_ids = set(session.exec(skipped_stmt).all())

    all_ids = saved_ids | clicked_ids | skipped_ids
    if len(all_ids) < MIN_TRAINING_SAMPLES:
        return None

    saved_embs = [embedding_lookup[aid] for aid in saved_ids if aid in embedding_lookup]
    clicked_embs = [embedding_lookup[aid] for aid in clicked_ids if aid in embedding_lookup]
    skipped_embs = [embedding_lookup[aid] for aid in skipped_ids if aid in embedding_lookup]

    # Pool each group to 32 dims
    saved_pool = _pool_embeddings(saved_embs)
    clicked_pool = _pool_embeddings(clicked_embs)
    skipped_pool = _pool_embeddings(skipped_embs)

    # Engagement stats (32-dim, zero-padded)
    total_imps = session.exec(
        select(func.count(FeedImpression.id))
        .where(FeedImpression.user_id == user_id)
    ).one() or 1
    total_clicks = len(read_ids)
    total_saves = len(saved_ids)
    total_skips = len(skipped_ids)

    stats = np.zeros(POOL_DIM, dtype=np.float32)
    stats[0] = total_clicks / max(total_imps, 1)  # CTR
    stats[1] = total_saves / max(total_imps, 1)    # save rate
    stats[2] = min(total_imps / 100, 1.0)          # interaction maturity
    stats[3] = total_skips / max(total_imps, 1)    # skip rate
    stats[4] = len(saved_embs) / max(len(saved_ids), 1)  # embedding coverage (saved)
    stats[5] = len(clicked_embs) / max(len(clicked_ids), 1)  # embedding coverage (clicked)
    # Diversity: std of saved embeddings projected
    if len(saved_embs) > 1:
        stacked = np.stack(saved_embs) @ _PROJ
        stats[6] = float(np.mean(np.std(stacked, axis=0)))
    # Remaining dims stay zero (room for future features)

    return np.concatenate([saved_pool, clicked_pool, skipped_pool, stats])


def _collect_training_pairs(
    session: Session, user_id: int, embedding_lookup: dict[int, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Collect (article_embedding, label) pairs for training.

    Returns (article_embeddings, user_features_repeated, labels)
    where label = +1 for engaged, -1 for skipped.
    Applies 3:1 negative sampling ratio.
    """
    saved_ids = set(session.exec(
        select(SavedArticle.article_id)
        .where(SavedArticle.user_id == user_id)
    ).all())
    read_ids = set(session.exec(
        select(ReadArticle.article_id)
        .where(ReadArticle.user_id == user_id)
    ).all())
    positive_ids = saved_ids | read_ids

    skipped_stmt = (
        select(FeedImpression.article_id)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.clicked == False)  # noqa: E712
        .where(FeedImpression.saved == False)  # noqa: E712
        .where(FeedImpression.processed == True)  # noqa: E712
    )
    negative_ids = set(session.exec(skipped_stmt).all()) - positive_ids

    # Filter to articles with embeddings
    pos_with_emb = [aid for aid in positive_ids if aid in embedding_lookup]
    neg_with_emb = [aid for aid in negative_ids if aid in embedding_lookup]

    if not pos_with_emb:
        return [], [], []

    article_embs = []
    labels = []

    for aid in pos_with_emb:
        article_embs.append(embedding_lookup[aid])
        labels.append(1)

        # Sample up to 3 negatives per positive
        n_neg = min(3, len(neg_with_emb))
        if n_neg > 0:
            sampled_neg = random.sample(neg_with_emb, n_neg)
            for nid in sampled_neg:
                article_embs.append(embedding_lookup[nid])
                labels.append(-1)

    return article_embs, labels, len(pos_with_emb)


def train_user_tower(
    session: Session, user_id: int, embedding_lookup: dict[int, np.ndarray],
    epochs: int = 50, lr: float = 1e-3,
) -> TrainingResult | None:
    """Train a UserTower model for a specific user.

    Returns TrainingResult with trained model, final loss, and sample count.
    Returns None if insufficient data.
    """
    user_features = build_user_features(session, user_id, embedding_lookup)
    if user_features is None:
        return None

    article_embs, labels, num_positives = _collect_training_pairs(
        session, user_id, embedding_lookup,
    )
    if num_positives < MIN_TRAINING_SAMPLES:
        return None

    # Convert to tensors
    device = torch.device("cpu")  # small model, CPU is fine
    features_t = torch.tensor(user_features, dtype=torch.float32, device=device).unsqueeze(0)
    article_t = torch.tensor(np.stack(article_embs), dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

    model = UserTower().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CosineEmbeddingLoss(margin=0.2)

    model.train()
    final_loss = 0.0

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Same user features for all pairs — expand to match batch
        user_emb = model(features_t)  # (1, 1024)
        user_emb_expanded = user_emb.expand(len(labels), -1)  # (N, 1024)

        loss = loss_fn(user_emb_expanded, article_t, labels_t)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    logger.info(
        f"[ML] User {user_id}: trained embedding model, "
        f"loss={final_loss:.4f}, samples={len(labels)} "
        f"({num_positives} pos, {len(labels) - num_positives} neg)"
    )

    return TrainingResult(model=model, loss=final_loss, num_samples=len(labels))


def compute_learned_user_embedding(
    model: UserTower, user_features: np.ndarray,
) -> np.ndarray:
    """Run inference through a trained UserTower to get a 1024-dim embedding."""
    model.eval()
    with torch.no_grad():
        features_t = torch.tensor(user_features, dtype=torch.float32).unsqueeze(0)
        embedding = model(features_t).squeeze(0).numpy()
    return embedding
