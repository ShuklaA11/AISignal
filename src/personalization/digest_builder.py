"""Build per-user digests with MMR dedup, source interleaving, and Thompson exploration."""

from __future__ import annotations

import logging
from datetime import date as date_type, datetime, time
from random import betavariate

import numpy as np
from sqlmodel import Session, case, func, select

from src.embeddings.similarity import (
    compute_embedding_factor, compute_user_embedding, cosine_similarity,
)
from src.personalization.scorer import score_article_for_user_ml
from src.storage.models import (
    Article, Digest, DigestArticle, FeedImpression, User, utcnow,
)
from src.storage.queries import get_article_embeddings, get_ml_profile

logger = logging.getLogger(__name__)

# Digest sizing per section
NEWS_ARTICLES = 3       # News & industry stories
RESEARCH_ARTICLES = 3   # Papers, repos, open-source
EXPLORE_ARTICLES = 3    # Thompson exploration picks

# Sources/categories that count as "research & repos"
RESEARCH_SOURCES = {"arxiv", "github", "huggingface"}
RESEARCH_CATEGORIES = {"research", "open_source"}

# Display order encoding:
#   0–99   = news section
#   100–199 = research/repos section
#   negative = explore section
NEWS_ORDER_OFFSET = 0
RESEARCH_ORDER_OFFSET = 100

# MMR redundancy threshold — articles more similar than this are skipped
REDUNDANCY_THRESHOLD = 0.85

# MMR trade-off: relevance vs diversity (1.0 = pure relevance, 0.0 = pure diversity)
MMR_LAMBDA = 0.7




def _mmr_select(
    scored_articles: list[tuple[Article, float]],
    embeddings: dict[int, np.ndarray],
    max_articles: int,
) -> list[tuple[Article, float]]:
    """Select articles using Maximal Marginal Relevance.

    Balances relevance (personalized score) with diversity (low similarity
    to already-selected articles). Articles without embeddings fall back
    to pure score ranking.
    """
    if not scored_articles:
        return []

    max_score = max(s for _, s in scored_articles) or 1.0
    candidates = [
        (article, score, score / max_score)
        for article, score in scored_articles
    ]

    selected: list[tuple[Article, float]] = []
    selected_embeddings: list[np.ndarray] = []
    remaining = list(candidates)

    while remaining and len(selected) < max_articles:
        best_idx = -1
        best_mmr = -float("inf")

        for i, (article, raw_score, norm_score) in enumerate(remaining):
            emb = embeddings.get(article.id)

            if not selected_embeddings or emb is None:
                mmr_score = norm_score
            else:
                max_sim = max(
                    cosine_similarity(emb, sel_emb)
                    for sel_emb in selected_embeddings
                )

                if max_sim > REDUNDANCY_THRESHOLD:
                    continue

                mmr_score = MMR_LAMBDA * norm_score - (1 - MMR_LAMBDA) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx == -1:
            break

        article, raw_score, _ = remaining.pop(best_idx)
        selected.append((article, raw_score))

        emb = embeddings.get(article.id)
        if emb is not None:
            selected_embeddings.append(emb)

    return selected


def _interleave_sources(articles: list[tuple[Article, float]]) -> list[tuple[Article, float]]:
    """Reorder articles to avoid consecutive items from the same source.

    Greedy round-robin: picks from the source with the most remaining articles
    that wasn't just used, while preserving relative score ordering within
    each source.
    """
    if len(articles) <= 2:
        return articles

    # Group by source, preserving score order within each group
    by_source: dict[str, list[tuple[Article, float]]] = {}
    for article, score in articles:
        src = article.source_name
        by_source.setdefault(src, []).append((article, score))

    result: list[tuple[Article, float]] = []
    last_source = None

    while any(by_source.values()):
        # Pick the source with the most remaining articles (but not the same as last)
        candidates = [
            (src, items) for src, items in by_source.items()
            if items and src != last_source
        ]
        if not candidates:
            # Only one source left — just drain it
            candidates = [(src, items) for src, items in by_source.items() if items]

        # Sort by queue length descending, then pick from longest queue
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        src, items = candidates[0]

        item = items.pop(0)
        result.append(item)
        last_source = src

        if not items:
            del by_source[src]

    return result


def _thompson_explore(
    session: Session,
    user_id: int,
    all_articles: list[Article],
    selected_ids: set[int],
    n: int = EXPLORE_ARTICLES,
) -> list[Article]:
    """Pick exploration articles using Thompson sampling on engagement stats.

    Selects articles NOT in the main digest that have high uncertainty
    (few impressions) or surprisingly high engagement. These are the
    "Other news you may be interested in" section.
    """
    candidates = [a for a in all_articles if a.id not in selected_ids]
    if not candidates:
        return []

    article_ids = [a.id for a in candidates]

    # Batch query engagement stats
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

    # Sample from Beta distribution for each candidate
    scored = []
    for article in candidates:
        trials, successes = stats.get(article.id, (0, 0))
        alpha = 1 + successes
        beta = max(1, 1 + trials - successes)
        thompson_score = betavariate(alpha, beta)
        scored.append((article, thompson_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [article for article, _ in scored[:n]]


def build_digest_for_user(
    session: Session,
    user: User,
    digest_date: date_type | None = None,
    max_articles: int | None = None,
    manual: bool = False,
) -> Digest:
    """Create a personalized, deduplicated digest for a user.

    Auto-curates from all processed articles — no manual approval needed.
    Uses MMR to remove redundant articles covering the same story.
    Interleaves sources to avoid monotony.
    Adaptive digest size based on user engagement level.
    """
    if digest_date is None:
        digest_date = utcnow().date()

    # Check if digest already exists
    existing = session.exec(
        select(Digest)
        .where(Digest.user_id == user.id)
        .where(Digest.digest_date == digest_date)
    ).first()

    if existing:
        if existing.status == "sent" and not manual:
            # Already sent today — nothing to do
            return existing
        else:
            # Rebuild: manual trigger OR unsent draft from earlier run
            old_links = session.exec(
                select(DigestArticle).where(DigestArticle.digest_id == existing.id)
            ).all()
            for link in old_links:
                session.delete(link)
            session.delete(existing)
            session.commit()
            logger.info(
                f"Cleared stale digest (status={existing.status}) for user_id={user.id} "
                f"to rebuild ({'manual' if manual else 'scheduled'})"
            )

    if max_articles is None:
        max_articles = NEWS_ARTICLES + RESEARCH_ARTICLES

    from datetime import timedelta

    if manual:
        # Manual trigger: always use last 24 hours
        lookback_start = utcnow() - timedelta(hours=24)
    else:
        # Scheduled: pull everything since the last *scheduled* digest was sent
        # (ignore manual digests so they don't shrink the lookback window)
        last_digest = session.exec(
            select(Digest)
            .where(Digest.user_id == user.id)
            .where(Digest.trigger == "scheduled")
            .where(Digest.status == "sent")
            .order_by(Digest.sent_at.desc())
        ).first()

        if last_digest and last_digest.sent_at:
            lookback_start = last_digest.sent_at
        elif last_digest:
            # Naive midnight UTC — consistent with utcnow() used everywhere
            lookback_start = datetime.combine(last_digest.digest_date, time())
        else:
            # No previous scheduled digest — pull from last 7 days
            lookback_start = datetime.combine(digest_date, time()) - timedelta(days=7)

    logger.info(
        f"Digest lookback for user_id={user.id} ({'manual' if manual else 'scheduled'}): "
        f"articles since {lookback_start.isoformat()}"
    )

    stmt = (
        select(Article)
        .where(Article.status.in_(["processed", "approved"]))
        .where(Article.fetched_at >= lookback_start)
    )
    articles = list(session.exec(stmt).all())

    if not articles:
        logger.info(f"No articles available for digest on {digest_date}")
        digest = Digest(
            user_id=user.id,
            digest_date=digest_date,
            status="draft",
            trigger="manual" if manual else "scheduled",
            subject_line=f"Your AI Signal — {digest_date.strftime('%B %d, %Y')}",
        )
        session.add(digest)
        session.commit()
        return digest

    # Score all articles for this user (ML-blended, matching feed scoring)
    article_ids = [a.id for a in articles]
    embeddings = get_article_embeddings(session, article_ids)
    ml_profile = get_ml_profile(session, user.id)
    user_embedding = compute_user_embedding(session, user.id, embeddings)

    scored = []
    for article in articles:
        emb_factor = compute_embedding_factor(
            embeddings.get(article.id), user_embedding,
        )
        score = score_article_for_user_ml(
            article, user, ml_profile, embedding_factor=emb_factor,
        )
        scored.append((article, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Split into news vs research/repos pools
    scored_news = []
    scored_research = []
    for article, score in scored:
        is_research = (
            article.source_name in RESEARCH_SOURCES
            or (article.category or "").lower() in RESEARCH_CATEGORIES
        )
        if is_research:
            scored_research.append((article, score))
        else:
            scored_news.append((article, score))

    # MMR-select from each pool independently
    news_articles = _mmr_select(scored_news, embeddings, NEWS_ARTICLES)
    research_articles = _mmr_select(scored_research, embeddings, RESEARCH_ARTICLES)

    # If one pool is short, fill from the other
    news_shortfall = NEWS_ARTICLES - len(news_articles)
    research_shortfall = RESEARCH_ARTICLES - len(research_articles)
    if news_shortfall > 0 and len(research_articles) > RESEARCH_ARTICLES:
        # Steal extras from research
        extras = research_articles[RESEARCH_ARTICLES:]
        news_articles.extend(extras[:news_shortfall])
        research_articles = research_articles[:RESEARCH_ARTICLES]
    elif research_shortfall > 0 and len(news_articles) > NEWS_ARTICLES:
        extras = news_articles[NEWS_ARTICLES:]
        research_articles.extend(extras[:research_shortfall])
        news_articles = news_articles[:NEWS_ARTICLES]

    # Interleave sources within each section
    news_articles = _interleave_sources(news_articles)
    research_articles = _interleave_sources(research_articles)

    # Create digest
    digest = Digest(
        user_id=user.id,
        digest_date=digest_date,
        status="draft",
        trigger="manual" if manual else "scheduled",
        subject_line=f"Your AI Signal — {digest_date.strftime('%B %d, %Y')}",
    )
    session.add(digest)
    session.flush()

    # Link news articles (display_order 0–99)
    for order, (article, score) in enumerate(news_articles):
        link = DigestArticle(
            digest_id=digest.id,
            article_id=article.id,
            personalized_score=score,
            display_order=NEWS_ORDER_OFFSET + order,
            approved=True,
        )
        session.add(link)

    # Link research/repos articles (display_order 100–199)
    for order, (article, score) in enumerate(research_articles):
        link = DigestArticle(
            digest_id=digest.id,
            article_id=article.id,
            personalized_score=score,
            display_order=RESEARCH_ORDER_OFFSET + order,
            approved=True,
        )
        session.add(link)

    # Thompson exploration picks (negative display_order)
    selected_ids = {a.id for a, _ in news_articles} | {a.id for a, _ in research_articles}
    explore = _thompson_explore(session, user.id, articles, selected_ids)
    for i, article in enumerate(explore):
        link = DigestArticle(
            digest_id=digest.id,
            article_id=article.id,
            personalized_score=0.0,
            display_order=-(i + 1),
            approved=True,
        )
        session.add(link)

    session.commit()
    logger.info(
        f"Built digest for user_id={user.id}: {len(news_articles)} news + "
        f"{len(research_articles)} research + {len(explore)} explore "
        f"/ {len(articles)} total"
    )
    return digest
