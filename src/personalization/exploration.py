"""Thompson sampling for explore/exploit balance in article ranking.

Injects principled exploration by sampling from a Beta distribution
per article. Articles with few impressions have high uncertainty and
occasionally get promoted, preventing filter bubbles.
"""

from __future__ import annotations

import logging
from random import betavariate

from sqlmodel import Session, select, func, case

from src.storage.models import Article, FeedImpression

logger = logging.getLogger(__name__)


def apply_thompson_exploration(
    articles: list[Article], user_id: int, session: Session,
) -> list[Article]:
    """Re-rank articles using Thompson sampling on per-article engagement.

    For each article, samples from Beta(1 + successes, 1 + failures) where
    successes = clicks + saves and failures = impressions - successes.
    The sample modulates the existing personalized score to inject exploration.

    Returns articles sorted by exploration-boosted score.
    """
    if not articles:
        return articles

    article_ids = [a.id for a in articles]

    # Batch query: aggregate impressions and successes per article for this user
    engaged = case(
        (FeedImpression.clicked == True, 1),  # noqa: E712
        (FeedImpression.saved == True, 1),  # noqa: E712
        else_=0,
    )
    stmt = (
        select(
            FeedImpression.article_id,
            func.count(FeedImpression.id).label("trials"),
            func.sum(engaged).label("successes"),
        )
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.article_id.in_(article_ids))
        .group_by(FeedImpression.article_id)
    )
    rows = session.exec(stmt).all()
    stats = {row[0]: (row[1], row[2] or 0) for row in rows}

    for article in articles:
        trials, successes = stats.get(article.id, (0, 0))
        # Beta(1 + successes, 1 + failures) — uniform prior
        alpha = 1 + successes
        beta = max(1, 1 + trials - successes)
        thompson_sample = betavariate(alpha, beta)

        # Modulate: multiplier in [0.5, 1.5]
        base_score = getattr(article, "_personalized_score", 1.0)
        article._exploration_score = base_score * (0.5 + thompson_sample)

    articles.sort(key=lambda a: a._exploration_score, reverse=True)
    return articles
