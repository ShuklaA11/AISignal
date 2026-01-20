import re
from datetime import date, datetime, timedelta
from typing import Optional

from sqlmodel import Session, select

from src.storage.models import (
    Article, ArticleSummary, Digest, DigestArticle, DigestClick,
    FeedImpression, FetchRun, ReadArticle, SavedArticle, User, UserMLProfile,
    utcnow,
)


def get_articles_by_status(session: Session, status: str, limit: int = 100) -> list[Article]:
    """Return articles with the given status, newest first."""
    stmt = (
        select(Article)
        .where(Article.status == status)
        .order_by(Article.fetched_at.desc())
        .limit(limit)
    )
    return list(session.exec(stmt).all())


def get_today_articles(session: Session, status: str | None = None) -> list[Article]:
    """Return today's articles, optionally filtered by status, sorted by importance."""
    today = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    stmt = select(Article).where(Article.fetched_at >= today)
    if status:
        stmt = stmt.where(Article.status == status)
    return list(session.exec(stmt.order_by(Article.base_importance_score.desc())).all())


def article_exists(session: Session, url: str) -> bool:
    """Check if an article with the given URL already exists in the database."""
    stmt = select(Article).where(Article.url == url)
    return session.exec(stmt).first() is not None


_RE_PUNCTUATION = re.compile(r"[^\w\s]")
_RE_WHITESPACE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Normalize a title for dedup: lowercase, strip punctuation/whitespace."""
    t = title.lower().strip()
    t = _RE_PUNCTUATION.sub("", t)   # remove punctuation
    t = _RE_WHITESPACE.sub(" ", t)   # collapse whitespace
    return t


def _title_fingerprint(title: str) -> str:
    """Create a short fingerprint from first 6 significant words.

    Catches duplicates like 'Foo Bar: A Study' vs 'Foo Bar — A Study in X'
    while being O(1) to compare via set lookup.
    """
    words = _normalize_title(title).split()
    # Use first 6 words (or all if shorter) — enough to uniquely identify most articles
    return " ".join(words[:6])


def get_title_fingerprints(session: Session, days: int = 7) -> set[str]:
    """Load all title fingerprints from recent articles. Call once per ingestion run."""
    cutoff = utcnow() - timedelta(days=days)
    stmt = select(Article.title).where(Article.fetched_at >= cutoff)
    titles = session.exec(stmt).all()
    return {_title_fingerprint(t) for t in titles if t}


def article_exists_by_title(title: str, existing_fingerprints: set[str]) -> bool:
    """Check if an article's title fingerprint matches any existing article.

    O(1) set lookup — no database query per call.
    """
    fp = _title_fingerprint(title)
    if not fp:
        return False
    return fp in existing_fingerprints


def get_or_create_summary(
    session: Session, article_id: int, role: str, level: str
) -> ArticleSummary | None:
    """Return an existing summary for the given article/role/level, or None."""
    stmt = (
        select(ArticleSummary)
        .where(ArticleSummary.article_id == article_id)
        .where(ArticleSummary.role == role)
        .where(ArticleSummary.level == level)
    )
    return session.exec(stmt).first()


def get_user_by_email(session: Session, email: str) -> User | None:
    """Look up a user by email address."""
    stmt = select(User).where(User.email == email)
    return session.exec(stmt).first()


def get_user_by_id(session: Session, user_id: int) -> User | None:
    """Look up a user by primary key."""
    stmt = select(User).where(User.id == user_id)
    return session.exec(stmt).first()


def get_active_users(session: Session) -> list[User]:
    """Return all users with active=True."""
    stmt = select(User).where(User.active == True)
    return list(session.exec(stmt).all())


def get_digest_for_user_date(session: Session, user_id: int, digest_date: date) -> Digest | None:
    """Return the digest for a user on a specific date, or None."""
    stmt = (
        select(Digest)
        .where(Digest.user_id == user_id)
        .where(Digest.digest_date == digest_date)
    )
    return session.exec(stmt).first()


def get_saved_article_ids(session: Session, user_id: int) -> set[int]:
    """Return set of article IDs saved by a user (for efficient lookup in feed)."""
    stmt = select(SavedArticle.article_id).where(SavedArticle.user_id == user_id)
    return set(session.exec(stmt).all())


def toggle_saved_article(session: Session, user_id: int, article_id: int) -> bool:
    """Save if not saved, unsave if saved. Returns new is_saved state."""
    from sqlalchemy.exc import IntegrityError

    stmt = (
        select(SavedArticle)
        .where(SavedArticle.user_id == user_id)
        .where(SavedArticle.article_id == article_id)
    )
    existing = session.exec(stmt).first()
    if existing:
        session.delete(existing)
        session.commit()
        return False
    else:
        try:
            saved = SavedArticle(user_id=user_id, article_id=article_id)
            session.add(saved)
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            return True  # Already saved by a concurrent request


def get_saved_articles_for_user(session: Session, user_id: int) -> list[Article]:
    """Return all saved articles for a user, newest saves first."""
    stmt = (
        select(Article)
        .join(SavedArticle, SavedArticle.article_id == Article.id)
        .where(SavedArticle.user_id == user_id)
        .order_by(SavedArticle.saved_at.desc())
    )
    return list(session.exec(stmt).all())


def get_read_article_ids(session: Session, user_id: int) -> set[int]:
    """Return set of article IDs the user has read (for efficient lookup in feed)."""
    stmt = select(ReadArticle.article_id).where(ReadArticle.user_id == user_id)
    return set(session.exec(stmt).all())


def mark_article_read(session: Session, user_id: int, article_id: int) -> None:
    """Record that a user read an article (idempotent)."""
    stmt = (
        select(ReadArticle)
        .where(ReadArticle.user_id == user_id)
        .where(ReadArticle.article_id == article_id)
    )
    if session.exec(stmt).first():
        return  # Already recorded
    read = ReadArticle(user_id=user_id, article_id=article_id)
    session.add(read)
    session.commit()


def get_read_articles_for_user(session: Session, user_id: int, limit: int = 50) -> list[Article]:
    """Return recently read articles for a user, newest reads first."""
    stmt = (
        select(Article)
        .join(ReadArticle, ReadArticle.article_id == Article.id)
        .where(ReadArticle.user_id == user_id)
        .order_by(ReadArticle.read_at.desc())
        .limit(limit)
    )
    return list(session.exec(stmt).all())


# ---------------------------------------------------------------------------
# ML / Impression queries
# ---------------------------------------------------------------------------

def get_or_create_ml_profile(session: Session, user_id: int) -> UserMLProfile:
    """Fetch existing ML profile or create a new one."""
    stmt = select(UserMLProfile).where(UserMLProfile.user_id == user_id)
    profile = session.exec(stmt).first()
    if profile is None:
        profile = UserMLProfile(user_id=user_id)
        session.add(profile)
        session.commit()
        session.refresh(profile)
    return profile


def get_ml_profile(session: Session, user_id: int) -> UserMLProfile | None:
    """Fetch ML profile for a user (returns None if no profile exists)."""
    stmt = select(UserMLProfile).where(UserMLProfile.user_id == user_id)
    return session.exec(stmt).first()


def record_impressions(
    session: Session, user_id: int, article_ids: list[int], feed_group: str,
    feed_view: str = "",
) -> None:
    """Batch-record that articles were shown to a user.

    Deduplicates: skips articles already impressed within the last hour.
    """
    cutoff = utcnow() - timedelta(hours=1)
    # Get recently impressed article IDs for this user
    stmt = (
        select(FeedImpression.article_id)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.shown_at > cutoff)
    )
    recent_ids = set(session.exec(stmt).all())

    now = utcnow()
    new_count = 0
    for position, article_id in enumerate(article_ids):
        if article_id in recent_ids:
            continue
        imp = FeedImpression(
            user_id=user_id,
            article_id=article_id,
            shown_at=now,
            position=position,
            feed_group=feed_group,
            feed_view=feed_view,
        )
        session.add(imp)
        new_count += 1

    # Update the ML profile impression counter
    if new_count:
        profile = session.exec(
            select(UserMLProfile).where(UserMLProfile.user_id == user_id)
        ).first()
        if profile:
            profile.total_impressions += new_count
            session.add(profile)

    session.commit()


def update_impression_clicked(session: Session, user_id: int, article_id: int) -> bool:
    """Mark the most recent impression of an article as clicked.

    Returns True if a matching impression was found and updated,
    False if the article was never shown in the feed (e.g. opened
    from read-history or a direct link).
    """
    stmt = (
        select(FeedImpression)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.article_id == article_id)
        .order_by(FeedImpression.shown_at.desc())
    )
    imp = session.exec(stmt).first()
    if imp:
        imp.clicked = True
        session.add(imp)
        session.commit()
        return True
    return False


def update_impression_saved(session: Session, user_id: int, article_id: int) -> bool:
    """Mark the most recent impression of an article as saved.

    Returns True if a matching impression was found and updated,
    False if the article was never shown in the feed.
    """
    stmt = (
        select(FeedImpression)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.article_id == article_id)
        .order_by(FeedImpression.shown_at.desc())
    )
    imp = session.exec(stmt).first()
    if imp:
        imp.saved = True
        session.add(imp)
        session.commit()
        return True
    return False


def get_impression_feedback(session: Session, user_id: int, article_id: int) -> tuple[bool, bool]:
    """Return (liked, disliked) state of the most recent impression."""
    stmt = (
        select(FeedImpression)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.article_id == article_id)
        .order_by(FeedImpression.shown_at.desc())
    )
    imp = session.exec(stmt).first()
    if imp:
        return bool(imp.liked), bool(imp.disliked)
    return False, False


def update_impression_liked(session: Session, user_id: int, article_id: int) -> bool:
    """Mark the most recent impression of an article as liked (clears disliked).

    Returns True if a matching impression was found, False otherwise.
    """
    stmt = (
        select(FeedImpression)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.article_id == article_id)
        .order_by(FeedImpression.shown_at.desc())
    )
    imp = session.exec(stmt).first()
    if imp:
        imp.liked = True
        imp.disliked = False
        session.add(imp)
        session.commit()
        return True
    return False


def update_impression_disliked(session: Session, user_id: int, article_id: int) -> bool:
    """Mark the most recent impression of an article as disliked (clears liked).

    Returns True if a matching impression was found, False otherwise.
    """
    stmt = (
        select(FeedImpression)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.article_id == article_id)
        .order_by(FeedImpression.shown_at.desc())
    )
    imp = session.exec(stmt).first()
    if imp:
        imp.disliked = True
        imp.liked = False
        session.add(imp)
        session.commit()
        return True
    return False


def update_impression_feedback_cleared(session: Session, user_id: int, article_id: int) -> None:
    """Clear both liked and disliked on the most recent impression."""
    stmt = (
        select(FeedImpression)
        .where(FeedImpression.user_id == user_id)
        .where(FeedImpression.article_id == article_id)
        .order_by(FeedImpression.shown_at.desc())
    )
    imp = session.exec(stmt).first()
    if imp:
        imp.liked = False
        imp.disliked = False
        session.add(imp)
        session.commit()


def get_article_embeddings(session: Session, article_ids: list[int]) -> dict:
    """Bulk-load embeddings for articles. Returns {article_id: numpy_vector}."""
    import numpy as np
    from src.storage.models import ArticleEmbedding
    if not article_ids:
        return {}
    stmt = select(ArticleEmbedding).where(ArticleEmbedding.article_id.in_(article_ids))
    rows = list(session.exec(stmt).all())
    return {
        r.article_id: np.frombuffer(r.embedding_blob, dtype=np.float32).copy()
        for r in rows
    }


def get_metrics_for_user(session: Session, user_id: int, days: int = 30) -> list:
    """Return daily metrics for the past N days."""
    from src.storage.models import ScoringMetric
    cutoff = utcnow() - timedelta(days=days)
    stmt = (
        select(ScoringMetric)
        .where(ScoringMetric.user_id == user_id)
        .where(ScoringMetric.computed_at >= cutoff)
        .order_by(ScoringMetric.metric_date.asc())
    )
    return list(session.exec(stmt).all())


def get_aggregate_daily_metrics(session: Session, days: int = 30) -> list[dict]:
    """Return aggregate daily metrics across all users for the past N days.

    Groups by metric_date, averages CTR/save_rate/nDCG/lift, sums totals.
    """
    from sqlmodel import func as sqlfunc
    from src.storage.models import ScoringMetric
    cutoff = utcnow() - timedelta(days=days)
    stmt = (
        select(
            ScoringMetric.metric_date,
            sqlfunc.avg(ScoringMetric.ctr).label("avg_ctr"),
            sqlfunc.avg(ScoringMetric.save_rate).label("avg_save_rate"),
            sqlfunc.avg(ScoringMetric.ndcg_at_10).label("avg_ndcg"),
            sqlfunc.avg(ScoringMetric.personalization_lift).label("avg_lift"),
            sqlfunc.sum(ScoringMetric.total_impressions).label("sum_impressions"),
            sqlfunc.sum(ScoringMetric.total_clicks).label("sum_clicks"),
            sqlfunc.sum(ScoringMetric.total_saves).label("sum_saves"),
            sqlfunc.count(ScoringMetric.user_id).label("user_count"),
        )
        .where(ScoringMetric.computed_at >= cutoff)
        .group_by(ScoringMetric.metric_date)
        .order_by(ScoringMetric.metric_date.asc())
    )
    rows = session.exec(stmt).all()
    return [
        {
            "metric_date": r[0],
            "avg_ctr": round(r[1] or 0, 4),
            "avg_save_rate": round(r[2] or 0, 4),
            "avg_ndcg": round(r[3] or 0, 4),
            "avg_lift": round(r[4] or 0, 4),
            "sum_impressions": r[5] or 0,
            "sum_clicks": r[6] or 0,
            "sum_saves": r[7] or 0,
            "user_count": r[8] or 0,
        }
        for r in rows
    ]


def get_fetch_health(session: Session, days: int = 14) -> dict:
    """Return fetch health data for the admin dashboard.

    Returns:
        {
            "runs": [FetchRun, ...],  -- most recent runs (last N days)
            "per_source": {source_name: {
                "total_runs": int,
                "ok_runs": int,
                "empty_runs": int,
                "error_runs": int,
                "total_fetched": int,
                "total_new": int,
                "avg_duration_ms": int,
                "last_run": datetime | None,
                "last_error": str | None,
            }},
            "daily": [{date, total_fetched, total_new, source_counts: {}}],
        }
    """
    from sqlmodel import func as sqlfunc

    cutoff = utcnow() - timedelta(days=days)

    # All runs in the window
    stmt = (
        select(FetchRun)
        .where(FetchRun.fetched_at >= cutoff)
        .order_by(FetchRun.fetched_at.desc())
    )
    runs = list(session.exec(stmt).all())

    # Per-source aggregates
    per_source: dict[str, dict] = {}
    for run in runs:
        src = run.source_name
        if src not in per_source:
            per_source[src] = {
                "total_runs": 0,
                "ok_runs": 0,
                "empty_runs": 0,
                "error_runs": 0,
                "total_fetched": 0,
                "total_new": 0,
                "total_duration_ms": 0,
                "last_run": None,
                "last_error": None,
            }
        s = per_source[src]
        s["total_runs"] += 1
        s[f"{run.status}_runs"] = s.get(f"{run.status}_runs", 0) + 1
        s["total_fetched"] += run.articles_fetched
        s["total_new"] += run.articles_new
        s["total_duration_ms"] += run.duration_ms
        if s["last_run"] is None or run.fetched_at > s["last_run"]:
            s["last_run"] = run.fetched_at
        if run.error:
            # Runs are ordered desc, so the first error we encounter is the
            # most recent one. Only store it once.
            if s["last_error"] is None:
                s["last_error"] = run.error

    # Compute avg_duration_ms
    for src, s in per_source.items():
        s["avg_duration_ms"] = s["total_duration_ms"] // s["total_runs"] if s["total_runs"] else 0
        del s["total_duration_ms"]

    # Daily totals
    daily_map: dict[date, dict] = {}
    for run in runs:
        d = run.fetched_at.date()
        if d not in daily_map:
            daily_map[d] = {"date": d, "total_fetched": 0, "total_new": 0, "source_counts": {}}
        daily_map[d]["total_fetched"] += run.articles_fetched
        daily_map[d]["total_new"] += run.articles_new
        sc = daily_map[d]["source_counts"]
        sc[run.source_name] = sc.get(run.source_name, 0) + run.articles_new

    daily = sorted(daily_map.values(), key=lambda x: x["date"])

    return {
        "runs": runs,
        "per_source": per_source,
        "daily": daily,
    }


# ── Token queries ──────────────────────────────────────────────────────

def create_token(
    session: Session, user_id: int, token_type: str, token_hash: str, expires_at: datetime,
) -> "Token":
    """Create and persist a new one-time-use token (email verification or password reset)."""
    from src.storage.models import Token
    token = Token(
        user_id=user_id,
        token_type=token_type,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    session.add(token)
    session.commit()
    session.refresh(token)
    return token


def get_token_by_hash(session: Session, token_hash: str, token_type: str):
    """Look up a valid (unused, unexpired) token by its hash and type."""
    from src.storage.models import Token
    stmt = (
        select(Token)
        .where(Token.token_hash == token_hash)
        .where(Token.token_type == token_type)
        .where(Token.expires_at > utcnow())
        .where(Token.used_at == None)  # noqa: E711
    )
    return session.exec(stmt).first()


def mark_token_used(session: Session, token) -> None:
    """Mark a token as consumed so it cannot be reused."""
    token.used_at = utcnow()
    session.add(token)
    session.commit()


def invalidate_user_tokens(session: Session, user_id: int, token_type: str) -> None:
    """Mark all unused tokens of a given type for a user as used."""
    from src.storage.models import Token
    stmt = (
        select(Token)
        .where(Token.user_id == user_id)
        .where(Token.token_type == token_type)
        .where(Token.used_at == None)  # noqa: E711
    )
    now = utcnow()
    for token in session.exec(stmt).all():
        token.used_at = now
        session.add(token)
    session.commit()


# ── Digest click tracking ──────────────────────────────────────────────

def record_digest_click(
    session: Session, user_id: int, article_id: int, digest_id: int, section: str = "main",
) -> DigestClick:
    """Record that a user clicked an article link in a digest email."""
    click = DigestClick(
        user_id=user_id,
        article_id=article_id,
        digest_id=digest_id,
        section=section,
    )
    session.add(click)
    session.commit()
    return click


def cleanup_expired_tokens(session: Session) -> int:
    """Delete expired and used tokens. Returns the number of tokens removed."""
    from sqlalchemy import delete as sa_delete
    from src.storage.models import Token
    now = utcnow()
    stmt = sa_delete(Token).where(
        (Token.expires_at <= now) | (Token.used_at != None)  # noqa: E711
    )
    result = session.exec(stmt)
    session.commit()
    return result.rowcount
