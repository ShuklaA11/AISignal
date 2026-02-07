"""Persistence for per-user trained embedding models."""

from __future__ import annotations

import io
import logging

import torch
from sqlmodel import Session, select

from src.embeddings.user_tower import UserTower
from src.storage.models import UserEmbeddingModel, utcnow

logger = logging.getLogger(__name__)


def save_user_model(
    session: Session, user_id: int, model: UserTower,
    loss: float, num_samples: int,
) -> None:
    """Serialize and store a trained UserTower model."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    blob = buf.getvalue()

    existing = session.exec(
        select(UserEmbeddingModel)
        .where(UserEmbeddingModel.user_id == user_id)
    ).first()

    if existing:
        existing.model_weights_blob = blob
        existing.training_loss = loss
        existing.num_training_samples = num_samples
        existing.trained_at = utcnow()
        session.add(existing)
    else:
        record = UserEmbeddingModel(
            user_id=user_id,
            model_weights_blob=blob,
            training_loss=loss,
            num_training_samples=num_samples,
        )
        session.add(record)

    session.commit()


def load_user_model(session: Session, user_id: int) -> UserTower | None:
    """Load a trained UserTower model from the database."""
    record = session.exec(
        select(UserEmbeddingModel)
        .where(UserEmbeddingModel.user_id == user_id)
    ).first()

    if record is None:
        return None

    model = UserTower()
    buf = io.BytesIO(record.model_weights_blob)
    model.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
    model.eval()
    return model
