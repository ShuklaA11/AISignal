"""Evaluation metrics: CTR, nDCG, save rate, personalization lift."""

import math
import logging
from datetime import date, datetime, timedelta

from sqlalchemy import case
from sqlmodel import Session, select, func

from src.storage.models import FeedImpression, ScoringMetric, utcnow

logger = logging.getLogger(__name__)


def compute_ctr(
    session: Session, user_id: int,
    start: datetime | None = None, end: datetime | None = None,
    feed_view: str = "",
) -> float:
    """Click-through rate = clicks / impressions."""
    total_stmt = (
        select(func.count(FeedImpression.id))
        .where(FeedImpression.user_id == user_id)
    )
    clicked_stmt = (
        select(func.count(FeedImpression.id))
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.clicked == True)  # noqa: E712
    )
    if feed_view:
        total_stmt = total_stmt.where(FeedImpression.feed_view == feed_view)
        clicked_stmt = clicked_stmt.where(FeedImpression.feed_view == feed_view)
    if start:
        total_stmt = total_stmt.where(FeedImpression.shown_at >= start)
        clicked_stmt = clicked_stmt.where(FeedImpression.shown_at >= start)
    if end:
        total_stmt = total_stmt.where(FeedImpression.shown_at < end)
        clicked_stmt = clicked_stmt.where(FeedImpression.shown_at < end)

    total = session.exec(total_stmt).one() or 0
    clicked = session.exec(clicked_stmt).one() or 0
    return round(clicked / total, 4) if total else 0.0


def compute_save_rate(
    session: Session, user_id: int,
    start: datetime | None = None, end: datetime | None = None,
) -> float:
    """Save rate = saves / impressions."""
    total_stmt = (
        select(func.count(FeedImpression.id))
        .where(FeedImpression.user_id == user_id)
    )
    saved_stmt = (
        select(func.count(FeedImpression.id))
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.saved == True)  # noqa: E712
    )
    if start:
        total_stmt = total_stmt.where(FeedImpression.shown_at >= start)
        saved_stmt = saved_stmt.where(FeedImpression.shown_at >= start)
    if end:
        total_stmt = total_stmt.where(FeedImpression.shown_at < end)
        saved_stmt = saved_stmt.where(FeedImpression.shown_at < end)

    total = session.exec(total_stmt).one() or 0
    saved = session.exec(saved_stmt).one() or 0
    return round(saved / total, 4) if total else 0.0


def compute_ndcg_at_k(
    session: Session, user_id: int, k: int = 10,
    start: datetime | None = None, end: datetime | None = None,
) -> float:
    """nDCG@k — ranking quality. Relevance: saved=3, clicked=1, skipped=0.

    Evaluates each feed session (batch of impressions sharing the same
    shown_at timestamp) independently and returns the average nDCG across
    all sessions in the period.
    """
    stmt = (
        select(FeedImpression)
        .where(FeedImpression.user_id == user_id)
        .order_by(FeedImpression.shown_at.asc(), FeedImpression.position.asc())
    )
    if start:
        stmt = stmt.where(FeedImpression.shown_at >= start)
    if end:
        stmt = stmt.where(FeedImpression.shown_at < end)
    impressions = list(session.exec(stmt).all())

    if not impressions:
        return 0.0

    # Group impressions by session. record_impressions() sets the same
    # shown_at for an entire batch, but test helpers may produce slightly
    # different timestamps. Round to the nearest second to be robust.
    sessions: dict[datetime, list] = {}
    for imp in impressions:
        key = imp.shown_at.replace(microsecond=0)
        sessions.setdefault(key, []).append(imp)

    # Compute nDCG per session and average
    ndcg_values = []
    for session_imps in sessions.values():
        # Sort by position within this session
        session_imps.sort(key=lambda imp: imp.position)
        top_k = session_imps[:k]

        dcg = 0.0
        for i, imp in enumerate(top_k):
            rel = 3.0 if imp.saved else (1.0 if imp.clicked else 0.0)
            dcg += rel / math.log2(i + 2)

        # IDCG: best possible ranking from this session's impressions
        all_rels = sorted(
            [3.0 if imp.saved else (1.0 if imp.clicked else 0.0) for imp in session_imps],
            reverse=True,
        )
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(all_rels[:k]))

        if idcg > 0:
            ndcg_values.append(dcg / idcg)

    if not ndcg_values:
        return 0.0

    return round(sum(ndcg_values) / len(ndcg_values), 4)


def compute_position_ctr(session: Session) -> dict[int, float]:
    """Compute average CTR by position across all users (last 30 days).

    Returns dict mapping position -> baseline CTR. Positions with fewer
    than 10 impressions use the global average as fallback.

    Args:
        session: Active database session.

    Returns:
        A dict like ``{0: 0.45, 1: 0.35, 2: 0.28, ...}`` where each key is
        a feed position (0-based) and each value is the fraction of
        impressions at that position that resulted in a click.  Positions
        with fewer than 10 impressions are filled with the global average CTR
        so callers always get a sensible fallback.
    """
    cutoff = utcnow() - timedelta(days=30)

    # Aggregate per-position counts in a single query.
    # Use CASE to convert boolean to integer for reliable SUM across backends.
    click_as_int = case(
        (FeedImpression.clicked == True, 1),  # noqa: E712
        else_=0,
    )
    stmt = (
        select(
            FeedImpression.position,
            func.count(FeedImpression.id).label("total"),
            func.sum(click_as_int).label("clicks"),
        )
        .where(FeedImpression.shown_at >= cutoff)
        .group_by(FeedImpression.position)
    )
    rows = session.exec(stmt).all()

    # Compute global totals for the fallback rate.
    global_total: int = 0
    global_clicks: int = 0
    position_data: dict[int, tuple[int, int]] = {}  # position -> (total, clicks)

    for row in rows:
        pos = row.position
        total = row.total or 0
        clicks = int(row.clicks or 0)
        position_data[pos] = (total, clicks)
        global_total += total
        global_clicks += clicks

    global_avg = round(global_clicks / global_total, 4) if global_total else 0.0

    result: dict[int, float] = {}
    for pos, (total, clicks) in position_data.items():
        if total >= 10:
            result[pos] = round(clicks / total, 4)
        else:
            result[pos] = global_avg

    return result


def compute_daily_metrics(session: Session, user_id: int, target_date: date) -> ScoringMetric:
    """Compute all metrics for a user for a specific date and store them."""
    start = datetime.combine(target_date, datetime.min.time())
    end = start + timedelta(days=1)

    ctr = compute_ctr(session, user_id, start=start, end=end)
    save_rate = compute_save_rate(session, user_id, start=start, end=end)
    ndcg = compute_ndcg_at_k(session, user_id, k=10, start=start, end=end)

    # Personalization lift: for_you CTR / overall CTR
    # Compares personalized feed engagement against the baseline of all impressions.
    # Default to 1.0 (neutral) when there are no for_you impressions to compare.
    fy_count = session.exec(
        select(func.count(FeedImpression.id))
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.shown_at >= start)
        .where(FeedImpression.shown_at < end)
        .where(FeedImpression.feed_view == "for_you")
    ).one() or 0
    if fy_count > 0 and ctr > 0:
        for_you_ctr = compute_ctr(session, user_id, start=start, end=end, feed_view="for_you")
        lift = round(for_you_ctr / ctr, 4)
    else:
        lift = 1.0

    # Count totals for this day
    total_stmt = (
        select(func.count(FeedImpression.id))
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.shown_at >= start)
        .where(FeedImpression.shown_at < end)
    )
    total_imps = session.exec(total_stmt).one() or 0

    clicked_stmt = (
        select(func.count(FeedImpression.id))
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.shown_at >= start)
        .where(FeedImpression.shown_at < end)
        .where(FeedImpression.clicked == True)  # noqa: E712
    )
    total_clicks = session.exec(clicked_stmt).one() or 0

    saved_stmt = (
        select(func.count(FeedImpression.id))
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.shown_at >= start)
        .where(FeedImpression.shown_at < end)
        .where(FeedImpression.saved == True)  # noqa: E712
    )
    total_saves = session.exec(saved_stmt).one() or 0

    # Upsert
    existing = session.exec(
        select(ScoringMetric)
        .where(ScoringMetric.user_id == user_id)
        .where(ScoringMetric.metric_date == target_date)
    ).first()

    if existing:
        existing.ctr = ctr
        existing.save_rate = save_rate
        existing.ndcg_at_10 = ndcg
        existing.personalization_lift = lift
        existing.total_impressions = total_imps
        existing.total_clicks = total_clicks
        existing.total_saves = total_saves
        existing.computed_at = utcnow()
        session.add(existing)
        metric = existing
    else:
        metric = ScoringMetric(
            user_id=user_id,
            metric_date=target_date,
            ctr=ctr,
            save_rate=save_rate,
            ndcg_at_10=ndcg,
            personalization_lift=lift,
            total_impressions=total_imps,
            total_clicks=total_clicks,
            total_saves=total_saves,
        )
        session.add(metric)

    session.commit()
    return metric
