"""Behavioral weight learning engine using exponential moving averages.

Learns per-user preferences from click/save/skip signals and updates
feature-level weights (source, category, topic, difficulty, entity).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

from sqlmodel import Session, select

from src.storage.models import Article, FeedImpression, ScoringMetric, UserMLProfile, utcnow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_on_click(session: Session, user_id: int, article_id: int, position: int | None = None) -> None:
    """Update ML profile when user clicks/reads an article.

    Args:
        session: Active database session.
        user_id: ID of the user who clicked.
        article_id: ID of the article that was clicked.
        position: Zero-based feed position where the article appeared.
            When provided, the raw signal is scaled by
            ``min(2.0, 1.0 + 0.1 * position)`` so that clicks on lower-ranked
            items carry a stronger learning signal (inverse propensity
            weighting).
    """
    profile = _get_or_create_profile(session, user_id)
    article = session.get(Article, article_id)
    if not article:
        return
    signal = 0.5
    if position is not None:
        signal *= min(2.0, 1.0 + 0.1 * position)
    _apply_signal(profile, article, signal=signal)
    profile.total_clicks += 1
    _update_alpha(profile)
    profile.updated_at = utcnow()
    session.add(profile)
    session.commit()


def update_on_save(session: Session, user_id: int, article_id: int, position: int | None = None) -> None:
    """Update ML profile when user saves an article.

    Args:
        session: Active database session.
        user_id: ID of the user who saved.
        article_id: ID of the article that was saved.
        position: Zero-based feed position where the article appeared.
            When provided, the raw signal is scaled by
            ``min(2.0, 1.0 + 0.1 * position)`` (inverse propensity weighting).
    """
    profile = _get_or_create_profile(session, user_id)
    article = session.get(Article, article_id)
    if not article:
        return
    signal = 1.0
    if position is not None:
        signal *= min(2.0, 1.0 + 0.1 * position)
    _apply_signal(profile, article, signal=signal)
    profile.total_saves += 1
    _update_alpha(profile)
    profile.updated_at = utcnow()
    session.add(profile)
    session.commit()


def update_on_like(session: Session, user_id: int, article_id: int, position: int | None = None) -> None:
    """Update ML profile when user explicitly likes an article (strongest positive).

    Args:
        session: Active database session.
        user_id: ID of the user who liked.
        article_id: ID of the article that was liked.
        position: Zero-based feed position where the article appeared.
            When provided, the raw signal is scaled by
            ``min(2.0, 1.0 + 0.1 * position)`` (inverse propensity weighting).
    """
    profile = _get_or_create_profile(session, user_id)
    article = session.get(Article, article_id)
    if not article:
        return
    signal = 1.5
    if position is not None:
        signal *= min(2.0, 1.0 + 0.1 * position)
    _apply_signal(profile, article, signal=signal)
    profile.total_clicks += 1
    _update_alpha(profile)
    profile.updated_at = utcnow()
    session.add(profile)
    session.commit()


def update_on_dislike(session: Session, user_id: int, article_id: int, position: int | None = None) -> None:
    """Update ML profile when user explicitly dislikes an article (strong negative).

    Args:
        session: Active database session.
        user_id: ID of the user who disliked.
        article_id: ID of the article that was disliked.
        position: Zero-based feed position where the article appeared.
            When provided, the raw signal is scaled by
            ``min(2.0, 1.0 + 0.1 * position)`` (inverse propensity weighting).
            Higher positions amplify the negative signal, reflecting that a
            deliberate dislike of a low-ranked item is a strong rejection.
    """
    profile = _get_or_create_profile(session, user_id)
    article = session.get(Article, article_id)
    if not article:
        return
    signal = -0.5
    if position is not None:
        signal *= min(2.0, 1.0 + 0.1 * position)
    _apply_signal(profile, article, signal=signal)
    _update_alpha(profile)
    profile.updated_at = utcnow()
    session.add(profile)
    session.commit()


def process_skips(session: Session, user_id: int) -> int:
    """Process shown-but-not-clicked impressions as weak negatives.

    Only considers impressions older than 6h that haven't been processed yet.
    Returns the number of skip signals applied.
    """
    cutoff = utcnow() - timedelta(hours=6)
    stmt = (
        select(FeedImpression)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.clicked == False)  # noqa: E712
        .where(FeedImpression.saved == False)  # noqa: E712
        .where(FeedImpression.processed == False)  # noqa: E712
        .where(FeedImpression.shown_at < cutoff)
    )
    skipped = list(session.exec(stmt).all())
    if not skipped:
        return 0

    profile = _get_or_create_profile(session, user_id)
    count = 0

    # Bulk-load all articles referenced by skipped impressions
    article_ids = list({imp.article_id for imp in skipped})
    articles_by_id = {a.id: a for a in session.exec(select(Article).where(Article.id.in_(article_ids))).all()}

    for imp in skipped:
        article = articles_by_id.get(imp.article_id)
        if article:
            # Higher feed positions are less likely to be seen organically,
            # so a skip there is a stronger rejection signal than at position 0.
            position_boost = min(2.0, 1.0 + 0.1 * imp.position)
            _apply_signal(profile, article, signal=-0.25 * position_boost)
            count += 1
        imp.processed = True
        session.add(imp)

    _update_alpha(profile)
    profile.updated_at = utcnow()
    session.add(profile)
    session.commit()

    logger.info(f"[ML] Processed {count} skip signals for user {user_id}")
    return count


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_or_create_profile(session: Session, user_id: int) -> UserMLProfile:
    """Fetch existing ML profile or create a new one."""
    from sqlalchemy.exc import IntegrityError

    stmt = select(UserMLProfile).where(UserMLProfile.user_id == user_id)
    profile = session.exec(stmt).first()
    if profile is None:
        try:
            profile = UserMLProfile(user_id=user_id)
            session.add(profile)
            session.flush()
        except IntegrityError:
            session.rollback()
            profile = session.exec(stmt).first()
    return profile


def _apply_signal(profile: UserMLProfile, article: Article, signal: float) -> None:
    """Apply a signal to all relevant feature weights via EMA.

    Also increments per-feature signal counts for confidence-aware decay.
    """
    lr = _get_learning_rate(profile)
    counts = profile.signal_counts

    # Source weight (normalize key so learned weights match rule-based scoring)
    from src.personalization.scorer import normalize_source_key
    source_w = profile.source_weights
    source_key = normalize_source_key(article.source_name)
    source_w[source_key] = _ema(source_w.get(source_key, 1.0), signal, lr)
    profile.source_weights = source_w
    sk = f"source:{source_key}"
    counts[sk] = counts.get(sk, 0) + 1

    # Category weight
    if article.category:
        cat_w = profile.category_weights
        cat_w[article.category] = _ema(cat_w.get(article.category, 1.0), signal, lr)
        profile.category_weights = cat_w
        ck = f"category:{article.category}"
        counts[ck] = counts.get(ck, 0) + 1

    # Topic weights (one update per topic on the article)
    article_topics = article.topics
    if article_topics:
        topic_w = profile.topic_weights
        for topic in article_topics:
            topic_w[topic] = _ema(topic_w.get(topic, 1.0), signal, lr)
            tk = f"topic:{topic}"
            counts[tk] = counts.get(tk, 0) + 1
        profile.topic_weights = topic_w

    # Difficulty weight
    if article.difficulty_level:
        diff_w = profile.difficulty_weights
        diff_w[article.difficulty_level] = _ema(
            diff_w.get(article.difficulty_level, 1.0), signal, lr
        )
        profile.difficulty_weights = diff_w
        dk = f"difficulty:{article.difficulty_level}"
        counts[dk] = counts.get(dk, 0) + 1

    # Entity weights (cap at top 5 to prevent noise)
    article_entities = article.key_entities[:5]
    if article_entities:
        ent_w = profile.entity_weights
        for entity in article_entities:
            ent_w[entity] = _ema(ent_w.get(entity, 1.0), signal, lr)
            ek = f"entity:{entity}"
            counts[ek] = counts.get(ek, 0) + 1
        profile.entity_weights = ent_w

    profile.signal_counts = counts


def _ema(old_weight: float, signal: float, lr: float) -> float:
    """Exponential moving average update, clamped to [0.1, 3.0].

    Signal mapping:
      +1.5 (like)   -> target 2.5 (strongest positive, explicit)
      +1.0 (save)   -> target 2.0 (double weight)
      +0.5 (click)  -> target 1.5
      -0.25 (skip)  -> target 0.75 (moderate decrease)
      -0.5 (dislike) -> target 0.5 (strong negative, explicit)
    """
    target = 1.0 + signal
    new = (1 - lr) * old_weight + lr * target
    return round(max(0.1, min(3.0, new)), 4)


def _get_base_learning_rate(profile: UserMLProfile) -> float:
    """Interaction-count-based learning rate tier."""
    total = profile.total_clicks + profile.total_saves
    if total < 10:
        return 0.3
    elif total < 50:
        return 0.15
    else:
        return 0.05


def _get_learning_rate(profile: UserMLProfile) -> float:
    """Adaptive learning rate: interaction tiers with metrics override.

    If nightly metrics adaptation has set a learning_rate_override,
    use it. Otherwise fall back to the interaction-count tiers.
    """
    if profile.learning_rate_override is not None:
        return profile.learning_rate_override
    return _get_base_learning_rate(profile)


def _update_alpha(profile: UserMLProfile) -> None:
    """Update the blending weight between rule-based and learned scores.

    alpha=1.0 → 100% rule-based (no interactions yet)
    alpha=0.3 → 70% learned (100+ interactions, never goes lower)

    This sets the interaction-based alpha. The nightly adapt_from_metrics()
    may further adjust alpha upward if the learned model underperforms.
    """
    total = profile.total_clicks + profile.total_saves
    import math
    profile.alpha = round(0.3 + 0.7 * math.exp(-total / 40), 4)


# ---------------------------------------------------------------------------
# Temporal decay (called nightly to prevent stale weights)
# ---------------------------------------------------------------------------

def decay_weights(session: Session, user_id: int) -> None:
    """Decay learned weights toward 1.0 with confidence-aware rate.

    High-confidence weights (many signals) decay slower than low-confidence
    ones, preserving well-established preferences while letting uncertain
    weights revert to neutral quickly.

    Decay rate formula:
        decay_rate = 0.95 + 0.04 * min(1.0, signal_count / 50)

    Examples:
        0 signals  -> 0.95 (standard decay, same as before)
        25 signals -> 0.97 (moderate decay)
        50+ signals -> 0.99 (barely decays, high confidence)

    Prevents stale preferences from dominating indefinitely.
    Called nightly before metrics computation.
    """
    profile = _get_or_create_profile(session, user_id)
    counts = profile.signal_counts

    # Map from weight dict attribute -> signal count key prefix
    attr_prefix_map = {
        "source_weights": "source",
        "category_weights": "category",
        "topic_weights": "topic",
        "difficulty_weights": "difficulty",
        "entity_weights": "entity",
    }

    for attr, prefix in attr_prefix_map.items():
        weights = getattr(profile, attr)
        if not weights:
            continue
        decayed = {}
        for k, v in weights.items():
            count_key = f"{prefix}:{k}"
            signal_count = counts.get(count_key, 0)
            decay_rate = 0.95 + 0.04 * min(1.0, signal_count / 50)
            decayed[k] = round(1.0 + (v - 1.0) * decay_rate, 4)
        setattr(profile, attr, decayed)

    profile.updated_at = utcnow()
    session.add(profile)
    session.commit()


# ---------------------------------------------------------------------------
# Metrics-driven adaptation (called nightly after metrics computation)
# ---------------------------------------------------------------------------

def adapt_from_metrics(session: Session, user_id: int) -> None:
    """Adjust alpha and learning rate based on recent performance metrics.

    Called nightly after compute_daily_metrics(). Uses personalization lift
    to gate alpha decay and nDCG trend to modulate learning rate.
    """
    profile = _get_or_create_profile(session, user_id)

    # Need enough interactions to have meaningful metrics
    total = profile.total_clicks + profile.total_saves
    if total < 5:
        return

    today = datetime.now(timezone.utc).date()
    week_ago = today - timedelta(days=7)

    recent_metrics = list(session.exec(
        select(ScoringMetric)
        .where(ScoringMetric.user_id == user_id)
        .where(ScoringMetric.metric_date >= week_ago)
        .order_by(ScoringMetric.metric_date.desc())
    ).all())

    # Filter out days with no impressions — they produce meaningless
    # lift=1.0 and ndcg=0.0 that would corrupt adaptation decisions.
    recent_metrics = [m for m in recent_metrics if m.total_impressions > 0]

    if len(recent_metrics) < 3:
        # Not enough history — keep interaction-based defaults
        return

    # --- Alpha adaptation via personalization lift ---
    last_3_lift = [m.personalization_lift for m in recent_metrics[:3]]
    avg_lift = sum(last_3_lift) / len(last_3_lift)

    # Compute the interaction-based alpha as baseline
    import math
    interaction_alpha = round(0.3 + 0.7 * math.exp(-total / 40), 4)

    if avg_lift < 1.0:
        # Learned model is hurting — retreat toward rules
        profile.alpha = round(min(1.0, interaction_alpha + 0.15), 4)
        logger.info(
            f"[ML] User {user_id}: lift={avg_lift:.3f} < 1.0, "
            f"alpha increased to {profile.alpha} (retreat to rules)"
        )
    else:
        # Learned model is helping — use the interaction-based alpha
        profile.alpha = interaction_alpha
        logger.info(
            f"[ML] User {user_id}: lift={avg_lift:.3f} >= 1.0, "
            f"alpha={profile.alpha} (learned model on track)"
        )

    # --- Learning rate adaptation via nDCG trend ---
    if len(recent_metrics) >= 7:
        recent_ndcg = [m.ndcg_at_10 for m in recent_metrics[:3]]
        prior_ndcg = [m.ndcg_at_10 for m in recent_metrics[3:7]]
    else:
        # Use what we have: split in half
        mid = len(recent_metrics) // 2
        recent_ndcg = [m.ndcg_at_10 for m in recent_metrics[:mid]]
        prior_ndcg = [m.ndcg_at_10 for m in recent_metrics[mid:]]

    if recent_ndcg and prior_ndcg:
        avg_recent = sum(recent_ndcg) / len(recent_ndcg)
        avg_prior = sum(prior_ndcg) / len(prior_ndcg)
        trend = avg_recent - avg_prior

        base_lr = _get_base_learning_rate(profile)

        if trend < -0.05:
            # nDCG declining — increase LR to escape local optimum
            adapted_lr = round(min(0.3, base_lr * 1.5), 4)
            logger.info(
                f"[ML] User {user_id}: nDCG trend={trend:+.3f} (declining), "
                f"LR bumped {base_lr} -> {adapted_lr}"
            )
        elif trend > 0.05:
            # nDCG improving — decrease LR to stabilize
            adapted_lr = round(max(0.02, base_lr * 0.7), 4)
            logger.info(
                f"[ML] User {user_id}: nDCG trend={trend:+.3f} (improving), "
                f"LR reduced {base_lr} -> {adapted_lr}"
            )
        else:
            # Flat — use base rate, clear any override
            adapted_lr = None
            logger.info(
                f"[ML] User {user_id}: nDCG trend={trend:+.3f} (stable), "
                f"LR={base_lr} (base)"
            )

        profile.learning_rate_override = adapted_lr
    else:
        profile.learning_rate_override = None

    profile.updated_at = utcnow()
    session.add(profile)
    session.commit()
