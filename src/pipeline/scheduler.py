"""Simple asyncio-based scheduler for pipeline jobs.

Replaces APScheduler which silently dies inside uvicorn's event loop.
Uses plain asyncio.create_task with sleep loops — reliable and transparent.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, time, timedelta, timezone

from src.config import Settings

logger = logging.getLogger(__name__)


def _log(msg: str, level: str = "info"):
    """Log scheduler events via the Python logger."""
    getattr(logger, level)(f"[Scheduler] {msg}")


class SimpleScheduler:
    """Lightweight asyncio scheduler that never silently dies."""

    def __init__(self):
        self._tasks: list[asyncio.Task] = []
        self._pending: list = []  # (factory_func,) — deferred until start()
        self._running = False

    def start(self):
        """Create all tasks on the *running* event loop, then mark as running."""
        self._running = True
        loop = asyncio.get_event_loop()
        for factory in self._pending:
            task = loop.create_task(factory())
            self._tasks.append(task)
        self._pending.clear()
        _log(f"started {len(self._tasks)} task loops")

    def shutdown(self):
        self._running = False
        for task in self._tasks:
            task.cancel()

    def add_interval_job(self, coro_func, seconds: int, name: str, kwargs: dict | None = None, run_now: bool = False, initial_delay: int = 0):
        """Schedule a coroutine to run at a fixed interval.

        If run_now=True, runs once immediately, then every `seconds`.
        If initial_delay>0, waits that many seconds before the first run,
        then every `seconds` after that.
        """
        scheduler = self

        async def _loop():
            if initial_delay > 0:
                await asyncio.sleep(initial_delay)
            elif not run_now:
                await asyncio.sleep(seconds)

            while scheduler._running:
                await _safe_run(coro_func, name, kwargs)
                await asyncio.sleep(seconds)

        self._pending.append(_loop)

    def add_daily_job(self, coro_func, hour: int, minute: int, name: str, kwargs: dict | None = None):
        """Schedule a coroutine to run daily at a specific local time."""
        scheduler = self

        async def _loop():
            while scheduler._running:
                now = datetime.now()
                target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                wait_seconds = (target - now).total_seconds()
                _log(f"{name}: next run at {target.strftime('%H:%M')} local ({wait_seconds / 3600:.1f}h from now)")
                await asyncio.sleep(wait_seconds)
                if scheduler._running:
                    await _safe_run(coro_func, name, kwargs)

        self._pending.append(_loop)


async def _safe_run(coro_func, name: str, kwargs: dict | None = None):
    """Run a job coroutine with full error protection."""
    try:
        _log(f"{name}: starting")
        await coro_func(**(kwargs or {}))
        _log(f"{name}: completed")
    except asyncio.CancelledError:
        _log(f"{name}: cancelled")
        raise
    except Exception as e:
        _log(f"{name}: FAILED — {e}\n{traceback.format_exc()}", "error")


def setup_scheduler(settings: Settings) -> SimpleScheduler:
    scheduler = SimpleScheduler()

    # Heartbeat every 15 min
    scheduler.add_interval_job(
        _heartbeat, seconds=900, name="heartbeat", run_now=True,
    )

    # Fetch all sources every 2 hours, immediately on startup
    scheduler.add_interval_job(
        _run_fetch, seconds=7200, name="fetch",
        kwargs={"settings": settings}, run_now=True,
    )

    # LLM processing every 30 min, staggered 5 min after startup
    scheduler.add_interval_job(
        _run_process, seconds=1800, name="process",
        kwargs={"settings": settings}, run_now=False, initial_delay=300,
    )

    # Nightly jobs
    scheduler.add_daily_job(_run_skip_processing, hour=2, minute=0, name="skip_processing", kwargs={"settings": settings})
    scheduler.add_daily_job(_run_weight_decay, hour=2, minute=30, name="weight_decay", kwargs={"settings": settings})

    # Midday skip processing (second daily pass to catch morning reading sessions faster)
    scheduler.add_daily_job(_run_skip_processing, hour=14, minute=0, name="skip_processing_midday", kwargs={"settings": settings})
    scheduler.add_daily_job(_run_metrics_computation, hour=3, minute=0, name="metrics", kwargs={"settings": settings})
    scheduler.add_daily_job(_run_metrics_adaptation, hour=3, minute=15, name="metrics_adaptation", kwargs={"settings": settings})
    scheduler.add_daily_job(_run_user_model_training, hour=3, minute=30, name="model_training", kwargs={"settings": settings})
    scheduler.add_daily_job(_run_token_cleanup, hour=4, minute=0, name="token_cleanup")

    # Daily digest send
    scheduler.add_daily_job(
        _run_send_digests,
        hour=settings.schedule.auto_send_hour,
        minute=settings.schedule.auto_send_minute,
        name="digest_send",
        kwargs={"settings": settings},
    )

    _log(
        f"Configured: fetch every 2h, process every 30m, "
        f"digest at {settings.schedule.auto_send_hour:02d}:{settings.schedule.auto_send_minute:02d}"
    )
    return scheduler


async def _heartbeat():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _log(f"heartbeat: alive at {now} UTC")


async def _run_fetch(settings: Settings):
    from src.pipeline.orchestrator import run_ingestion
    count = await run_ingestion(settings)
    _log(f"fetch: {count} new articles")


async def _run_process(settings: Settings):
    from src.pipeline.processor import run_processing
    count = await run_processing(settings)
    _log(f"process: {count} articles processed")

    # Generate embeddings
    from src.embeddings.pipeline import run_embedding_generation
    from src.storage.database import session_scope
    with session_scope() as session:
        emb_count = await run_embedding_generation(session)
        _log(f"embeddings: {emb_count} generated")


async def _run_skip_processing(settings: Settings):
    from src.personalization.learner import process_skips
    from src.storage.database import session_scope
    from src.storage.queries import get_active_users
    with session_scope() as session:
        users = get_active_users(session)
        total = 0
        for u in users:
            try:
                total += process_skips(session, u.id)
            except Exception as e:
                _log(f"skip_processing: failed for user_id={u.id}: {e}", "error")
        _log(f"skip_processing: {total} signals across {len(users)} users")


async def _run_weight_decay(settings: Settings):
    from src.personalization.learner import decay_weights
    from src.storage.database import session_scope
    from src.storage.queries import get_active_users
    with session_scope() as session:
        users = get_active_users(session)
        for user in users:
            try:
                decay_weights(session, user.id)
            except Exception as e:
                _log(f"weight_decay: failed for user_id={user.id}: {e}", "error")
        _log(f"weight_decay: decayed weights for {len(users)} users")


async def _run_metrics_computation(settings: Settings):
    from src.metrics.calculator import compute_daily_metrics
    from src.storage.database import session_scope
    from src.storage.queries import get_active_users
    with session_scope() as session:
        users = get_active_users(session)
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        for user in users:
            try:
                compute_daily_metrics(session, user.id, yesterday)
            except Exception as e:
                _log(f"metrics: failed for user_id={user.id}: {e}", "error")
        _log(f"metrics: computed for {len(users)} users")


async def _run_metrics_adaptation(settings: Settings):
    from src.personalization.learner import adapt_from_metrics
    from src.storage.database import session_scope
    from src.storage.queries import get_active_users
    with session_scope() as session:
        users = get_active_users(session)
        for user in users:
            try:
                adapt_from_metrics(session, user.id)
            except Exception as e:
                _log(f"metrics_adaptation: failed for user_id={user.id}: {e}", "error")
        _log(f"metrics_adaptation: adapted for {len(users)} users")


async def _run_user_model_training(settings: Settings):
    from src.embeddings.user_model_store import save_user_model
    from src.embeddings.user_tower import train_user_tower
    from src.storage.database import session_scope
    from src.storage.queries import get_active_users
    from sqlmodel import select
    from src.storage.models import ArticleEmbedding
    import numpy as np
    with session_scope() as session:
        users = get_active_users(session)
        all_emb_rows = list(session.exec(select(ArticleEmbedding)).all())
        embedding_lookup = {
            r.article_id: np.frombuffer(r.embedding_blob, dtype=np.float32).copy()
            for r in all_emb_rows
        }
        trained = 0
        for user in users:
            result = train_user_tower(session, user.id, embedding_lookup)
            if result is not None:
                save_user_model(session, user.id, result.model, result.loss, result.num_samples)
                trained += 1
        _log(f"model_training: trained {trained}/{len(users)} users")


async def _run_token_cleanup():
    from src.storage.database import session_scope
    from src.storage.queries import cleanup_expired_tokens
    with session_scope() as session:
        count = cleanup_expired_tokens(session)
        if count:
            _log(f"token_cleanup: cleaned {count} expired tokens")


def _build_and_send_one(session, sender, user, manual: bool = False) -> str:
    """Build and send a digest for a single user. Returns 'sent', 'failed', or 'skipped'."""
    from sqlmodel import select

    from src.personalization.digest_builder import build_digest_for_user
    from src.storage.models import ArticleSummary, DigestArticle

    digest = build_digest_for_user(session, user, manual=manual)

    stmt = (
        select(DigestArticle)
        .where(DigestArticle.digest_id == digest.id)
        .order_by(DigestArticle.display_order)
    )
    links = list(session.exec(stmt).all())

    if not links:
        return "skipped"

    # Filter out links whose articles may have been deleted
    links = [l for l in links if l.article is not None]

    # Split into 3 sections by display_order encoding:
    #   0–99   = news
    #   100–199 = research/repos
    #   negative = explore
    news_links = sorted(
        [l for l in links if 0 <= l.display_order < 100],
        key=lambda l: l.display_order,
    )
    research_links = sorted(
        [l for l in links if 100 <= l.display_order < 200],
        key=lambda l: l.display_order,
    )
    explore_links = sorted(
        [l for l in links if l.display_order < 0],
        key=lambda l: l.display_order,
        reverse=True,  # so -1 comes first
    )

    # Bulk-load summaries for all articles in this digest
    all_article_ids = [l.article.id for l in links]
    summary_stmt = (
        select(ArticleSummary)
        .where(ArticleSummary.article_id.in_(all_article_ids))
        .where(ArticleSummary.role == user.role)
    )
    summary_map = {s.article_id: s.summary_text for s in session.exec(summary_stmt).all()}

    def _link_to_dict(link):
        article = link.article
        return {
            "id": article.id,
            "title": article.title,
            "url": article.url,
            "source_name": article.source_name,
            "summary": summary_map.get(article.id),
            "topics": article.topics,
        }

    news_data = [_link_to_dict(l) for l in news_links]
    research_data = [_link_to_dict(l) for l in research_links]
    explore_data = [_link_to_dict(l) for l in explore_links]

    html = sender.render_digest(
        digest, news_data, user,
        research_articles=research_data,
        explore_articles=explore_data,
    )
    success = sender.send(
        to_email=user.email,
        subject=digest.subject_line or "Your AI Signal",
        html_body=html,
    )

    if success:
        digest.status = "sent"
        digest.sent_at = datetime.now(timezone.utc)
        session.add(digest)
        session.commit()
        return "sent"
    return "failed"


async def _run_send_digests(settings: Settings):
    """Build and send personalized digests to all active users."""
    from src.email_delivery.sender import EmailSender
    from src.storage.database import session_scope
    from src.storage.queries import get_active_users

    sender = EmailSender(settings)

    with session_scope() as session:
        users = get_active_users(session)
        if not users:
            _log("digest_send: no active users")
            return

        sent = failed = 0
        for user in users:
            try:
                result = _build_and_send_one(session, sender, user)
                if result == "sent":
                    sent += 1
                elif result == "failed":
                    failed += 1
            except Exception as e:
                _log(f"digest_send: failed for user_id={user.id}: {e}", "error")
                failed += 1

        _log(f"digest_send: {sent} sent, {failed} failed")


async def send_all_digests(settings: Settings) -> dict:
    """Public entry point for manual digest trigger (from API endpoint)."""
    from src.email_delivery.sender import EmailSender
    from src.storage.database import session_scope
    from src.storage.queries import get_active_users

    sender = EmailSender(settings)
    sent = failed = skipped = 0

    with session_scope() as session:
        users = get_active_users(session)
        for user in users:
            try:
                result = _build_and_send_one(session, sender, user, manual=True)
                if result == "sent":
                    sent += 1
                elif result == "failed":
                    failed += 1
                elif result == "skipped":
                    skipped += 1
            except Exception as e:
                logger.error(f"Digest failed for user_id={user.id}: {e}")
                failed += 1

    return {"sent": sent, "failed": failed, "skipped": skipped, "total": sent + failed + skipped}
