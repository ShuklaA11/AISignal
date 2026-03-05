"""Public feed route — no login required."""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from sqlmodel import select

from src.config import load_settings
from src.personalization.exploration import apply_thompson_exploration
from src.personalization.scorer import score_article_for_user_ml
from src.storage.database import session_scope
from src.storage.models import Article, ArticleSummary
from src.storage.queries import (
    get_ml_profile, get_read_article_ids, get_saved_article_ids,
    get_user_by_id, record_impressions,
)
from src.web.template_engine import templates

router = APIRouter()

# Map source_name values to display groups
SOURCE_GROUPS = {
    "RSS News": [
        "techcrunch_ai", "venturebeat_ai", "mit_tech_review_ai",
        "the_verge_ai", "openai_blog", "anthropic_blog",
        "deepmind_blog", "meta_ai_blog", "huggingface_blog",
    ],
    "Research & Open Source": ["arxiv", "huggingface", "github"],
}

GROUP_NAMES = list(SOURCE_GROUPS.keys())
DEFAULT_GROUP = "RSS News"
ARTICLES_PER_PAGE = 30


_DEFAULT_TOPICS = [
    "NLP", "Computer Vision", "Reinforcement Learning", "ML Theory",
    "AI Safety", "Multimodal", "Robotics", "AI Agents",
    "LLM APIs", "AI Infrastructure", "AI Startups", "Enterprise AI",
    "AI Regulation", "Fundraising", "Open Source Models", "AI Art",
    "AI Coding Tools", "AI Hardware", "Tutorials",
]


def _get_topics_from_config():
    """Load topic list from Settings (which reads config.yaml + env overrides)."""
    settings = load_settings()
    if settings.topics:
        topics = []
        for group_topics in settings.topics.values():
            topics.extend(group_topics)
        return topics
    return _DEFAULT_TOPICS


def _score_articles(session, articles, user):
    """Compute personalized scores + display_pct for a list of articles (does NOT re-sort)."""
    if not articles:
        return
    ml_profile = get_ml_profile(session, user.id)
    from src.embeddings.similarity import compute_embedding_factor, compute_user_embedding
    from src.storage.queries import get_article_embeddings
    article_ids = [a.id for a in articles]
    embedding_lookup = get_article_embeddings(session, article_ids)
    user_embedding = compute_user_embedding(session, user.id, embedding_lookup)

    for article in articles:
        emb_factor = compute_embedding_factor(
            embedding_lookup.get(article.id), user_embedding,
        )
        article._personalized_score = score_article_for_user_ml(
            article, user, ml_profile, embedding_factor=emb_factor,
        )

    # Use absolute scale: MAX_SCORE (20.0) is the theoretical ceiling.
    # Percentile-rank within the batch so the spread is meaningful,
    # but only when there's actual variance; fall back to absolute %.
    scores = sorted(set(a._personalized_score for a in articles))
    if len(scores) > 1:
        min_s, max_s = scores[0], scores[-1]
        spread = max_s - min_s
        for article in articles:
            # Rank-normalized within batch → 0-1, then map to 50-99
            rank_pct = (article._personalized_score - min_s) / spread
            article._display_pct = int(50 + rank_pct * 49)
    else:
        # All articles scored identically — use absolute scale
        from src.personalization.scorer import MAX_SCORE
        for article in articles:
            article._display_pct = int(min(99, (article._personalized_score / MAX_SCORE) * 100))


def _query_articles(session, group=None, levels=None, topics=None, sources=None, user=None, sort="for_you", read_ids=None):
    """Query up to 200 articles for a source group, with optional sidebar filters and personalization.

    sort="for_you" — score and rank by personalization (default for logged-in)
    sort="recent"  — keep chronological order
    Both compute match % when user is logged in.
    """
    group = group or DEFAULT_GROUP
    query_sources = SOURCE_GROUPS.get(group, SOURCE_GROUPS[DEFAULT_GROUP])

    # If individual source filters are active, intersect with group sources
    if sources:
        query_sources = [s for s in query_sources if s in sources]
        if not query_sources:
            return []

    stmt = (
        select(Article)
        .where(Article.status.in_(["processed", "approved", "sent"]))
        .where(Article.source_name.in_(query_sources))
    )

    if read_ids:
        stmt = stmt.where(Article.id.not_in(read_ids))

    if levels:
        stmt = stmt.where(Article.difficulty_level.in_(levels))

    stmt = stmt.order_by(Article.published_at.desc().nulls_last()).limit(200)
    articles = list(session.exec(stmt).all())

    # Topic filtering requires JSON field inspection — keep in Python
    if topics:
        topics_set = set(topics)
        articles = [
            a for a in articles
            if topics_set.intersection(set(a.topics or []))
        ]

    # Always compute match scores when logged in
    if user:
        _score_articles(session, articles, user)

        if sort == "for_you":
            # Re-rank by personalization score with Thompson sampling exploration
            articles.sort(key=lambda a: a._personalized_score, reverse=True)
            articles = apply_thompson_exploration(articles, user.id, session)

        # Record impressions for ML learning (all views, not just for_you)
        feed_view = sort or "recent"
        if articles:
            record_impressions(session, user.id, [a.id for a in articles], group, feed_view=feed_view)

    return articles


def _load_summaries(session, articles, role: str = "enthusiast") -> dict[int, str]:
    """Bulk-load the best summary for each article matching the user's role.

    Returns {article_id: summary_text}.
    """
    if not articles:
        return {}
    article_ids = [a.id for a in articles]
    stmt = (
        select(ArticleSummary)
        .where(ArticleSummary.article_id.in_(article_ids))
        .where(ArticleSummary.role == role)
    )
    rows = list(session.exec(stmt).all())
    return {r.article_id: r.summary_text for r in rows}


@router.get("/feed", response_class=HTMLResponse,
            summary="Article feed",
            description="Public feed page with source group tabs, personalized/chronological sort, and topic/level/source filter sidebar. No login required.")
async def feed_page(request: Request, group: Optional[str] = None, sort: Optional[str] = None, page: int = 1):
    """Public feed page with group tabs, sort toggle, and filter sidebar."""
    active_group = group or DEFAULT_GROUP
    if active_group not in SOURCE_GROUPS:
        active_group = DEFAULT_GROUP

    with session_scope() as session:
        user = None
        user_id = request.session.get("user_id")
        if user_id:
            user = get_user_by_id(session, user_id)

        # Default to "for_you" when logged in, "recent" when logged out
        active_sort = sort or ("for_you" if user else "recent")
        if active_sort not in ("for_you", "recent"):
            active_sort = "for_you" if user else "recent"

        read_ids = get_read_article_ids(session, user.id) if user else set()
        all_articles = _query_articles(
            session, group=active_group, user=user, sort=active_sort, read_ids=read_ids,
        )

        # Paginate
        page = max(1, page)
        total_count = len(all_articles)
        total_pages = max(1, (total_count + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE)
        page = min(page, total_pages)
        start = (page - 1) * ARTICLES_PER_PAGE
        articles = all_articles[start : start + ARTICLES_PER_PAGE]

        all_topics = _get_topics_from_config()
        saved_ids = get_saved_article_ids(session, user.id) if user else set()
        role = user.role if user else "enthusiast"
        summaries_map = _load_summaries(session, articles, role=role)

        return templates.TemplateResponse(
            "feed.html",
            {
                "request": request,
                "user": user,
                "articles": articles,
                "total_count": total_count,
                "all_topics": all_topics,
                "source_groups": SOURCE_GROUPS,
                "group_names": GROUP_NAMES,
                "active_group": active_group,
                "active_sort": active_sort,
                "saved_ids": saved_ids,
                "summaries_map": summaries_map,
                "page": page,
                "total_pages": total_pages,
            },
        )


@router.get("/api/feed/articles", response_class=HTMLResponse,
            summary="Filter feed articles",
            description="HTMX endpoint returning filtered article HTML partials. Supports group, sort, level, topic, and source filters with pagination.")
async def feed_articles_filter(
    request: Request,
    group: Optional[str] = Query(None),
    sort: Optional[str] = Query(None),
    level: Optional[list[str]] = Query(None),
    topic: Optional[list[str]] = Query(None),
    source: Optional[list[str]] = Query(None),
    page: int = Query(1),
):
    """HTMX endpoint: returns filtered article HTML partial."""
    active_group = group or DEFAULT_GROUP
    if active_group not in SOURCE_GROUPS:
        active_group = DEFAULT_GROUP

    with session_scope() as session:
        user = None
        user_id = request.session.get("user_id")
        if user_id:
            user = get_user_by_id(session, user_id)

        active_sort = sort or ("for_you" if user else "recent")
        read_ids = get_read_article_ids(session, user.id) if user else set()

        all_articles = _query_articles(
            session, group=active_group, levels=level, topics=topic, sources=source,
            user=user, sort=active_sort, read_ids=read_ids,
        )

        # Paginate
        page = max(1, page)
        total_count = len(all_articles)
        total_pages = max(1, (total_count + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE)
        page = min(page, total_pages)
        start = (page - 1) * ARTICLES_PER_PAGE
        articles = all_articles[start : start + ARTICLES_PER_PAGE]

        saved_ids = get_saved_article_ids(session, user.id) if user else set()
        role = user.role if user else "enthusiast"
        summaries_map = _load_summaries(session, articles, role=role)

        return templates.TemplateResponse(
            "partials/feed_articles.html",
            {
                "request": request,
                "user": user,
                "articles": articles,
                "total_count": total_count,
                "saved_ids": saved_ids,
                "active_sort": active_sort,
                "summaries_map": summaries_map,
                "page": page,
                "total_pages": total_pages,
            },
        )
