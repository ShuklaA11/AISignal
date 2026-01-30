"""LLM processing pipeline: summarize, categorize, rank articles."""

from __future__ import annotations

import asyncio
import json
import logging

from sqlmodel import Session

from src.config import Settings, load_settings
from src.llm.provider import LLMProvider
from src.llm.summarizer import process_articles_batch
from src.storage.database import init_db, session_scope
from src.storage.models import Article, ArticleSummary
from src.storage.queries import get_articles_by_status

logger = logging.getLogger(__name__)

MAX_BATCH_RETRIES = 3

ROLE_SUMMARY_MAP = {
    "student": "summary_student",
    "industry": "summary_industry",
    "enthusiast": "summary_enthusiast",
}


def _validate_topics(raw_topics: list) -> list[str]:
    """Filter topics to allowed list, fallback to ['General AI'] if empty."""
    from src.llm.prompts import ALL_TOPICS
    allowed = set(ALL_TOPICS)
    valid = [t for t in raw_topics if t in allowed]
    return valid if valid else ["General AI"]


def _apply_result_to_article(article: Article, result: dict) -> None:
    """Apply LLM processing results to an article."""
    article.category = result.get("category")
    article.base_importance_score = result.get("base_importance_score")
    article.difficulty_level = result.get("difficulty_level")
    article.topics_json = json.dumps(_validate_topics(result.get("topics", [])))
    article.key_entities_json = json.dumps(result.get("key_entities", []))
    article.status = "processed"


def _store_summaries(session: Session, article: Article, result: dict) -> int:
    """Store the 3 role-based summaries at intermediate level (skip if already exist).

    Returns number of summaries created.
    """
    from sqlmodel import select
    existing = session.exec(
        select(ArticleSummary.role)
        .where(ArticleSummary.article_id == article.id)
    ).all()
    existing_roles = set(existing)
    created = 0

    for role, key in ROLE_SUMMARY_MAP.items():
        if role in existing_roles:
            created += 1  # Already exists, count it
            continue
        summary_text = result.get(key, "")
        if not summary_text:
            logger.warning(f"Empty {role} summary for article {article.id}: {article.title[:50]}")
            continue
        summary = ArticleSummary(
            article_id=article.id,
            role=role,
            level="intermediate",
            summary_text=summary_text,
        )
        session.add(summary)
        created += 1

    return created


async def run_processing(settings: Settings | None = None, batch_size: int | None = None) -> int:
    """Process all raw articles with LLM. Returns count of processed articles."""
    if settings is None:
        settings = load_settings()

    init_db(settings.database_url)
    llm = LLMProvider(settings)
    batch_size = batch_size or settings.llm.batch_size

    with session_scope(settings.database_url) as session:
        raw_articles = get_articles_by_status(session, "raw", limit=500)
        if not raw_articles:
            logger.info("No raw articles to process")
            return 0

        logger.info(f"Processing {len(raw_articles)} articles in batches of {batch_size}...")
        processed_count = 0

        for i in range(0, len(raw_articles), batch_size):
            batch = raw_articles[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} articles)...")

            results = None
            for attempt in range(1, MAX_BATCH_RETRIES + 1):
                try:
                    results = await process_articles_batch(llm, batch)
                    break
                except Exception as e:
                    logger.error(f"Batch {i // batch_size + 1} attempt {attempt}/{MAX_BATCH_RETRIES} failed: {e}")
                    if attempt < MAX_BATCH_RETRIES:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s

            if results is None:
                titles = [a.title[:50] for a in batch]
                logger.error(f"Batch permanently failed after {MAX_BATCH_RETRIES} retries. "
                             f"Articles remain as 'raw' for next run: {titles}")
                continue

            # Map results back to articles by index
            results_by_index = {r.get("index", -1): r for r in results}

            for idx, article in enumerate(batch):
                result = results_by_index.get(idx)
                if not result:
                    logger.warning(f"No result for article {idx}: {article.title[:50]}")
                    continue

                _apply_result_to_article(article, result)
                session.add(article)
                session.flush()  # Ensure article.id is set
                summary_count = _store_summaries(session, article, result)
                if summary_count == 0:
                    logger.warning(f"No summaries created for article {article.id}: {article.title[:50]}. "
                                   f"Reverting to 'raw' status.")
                    article.status = "raw"
                    session.add(article)
                else:
                    processed_count += 1

            session.commit()

        logger.info(f"Processed {processed_count} articles")
        return processed_count
