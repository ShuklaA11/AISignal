"""Pipeline step: generate embeddings for processed articles."""

import logging

import numpy as np
from sqlmodel import Session, select

from src.embeddings.provider import generate_embeddings_batch
from src.storage.models import Article, ArticleEmbedding, ArticleSummary

logger = logging.getLogger(__name__)


async def run_embedding_generation(session: Session, batch_size: int = 20) -> int:
    """Generate embeddings for all processed articles that lack them.

    Split into three phases so the session isn't held open during the API call:
    1. Query phase: load articles and build text inputs
    2. API call phase: generate embeddings (no session usage)
    3. Store phase: persist results
    """
    # Phase 1: Query — collect article data and build texts
    existing_ids_stmt = select(ArticleEmbedding.article_id)
    existing_ids = set(session.exec(existing_ids_stmt).all())

    articles_stmt = (
        select(Article)
        .where(Article.status.in_(["processed", "approved", "sent"]))
    )
    all_articles = list(session.exec(articles_stmt).all())
    articles = [a for a in all_articles if a.id not in existing_ids]

    if not articles:
        logger.info("No articles need embedding generation")
        return 0

    # Bulk-load enthusiast summaries with IN query
    article_ids = [a.id for a in articles]
    summary_stmt = (
        select(ArticleSummary)
        .where(ArticleSummary.article_id.in_(article_ids))
        .where(ArticleSummary.role == "enthusiast")
    )
    summary_map = {s.article_id: s.summary_text for s in session.exec(summary_stmt).all()}

    texts = []
    for article in articles:
        summary = summary_map.get(article.id)
        if summary:
            text = f"{article.title} | {summary}"
        else:
            content = (article.original_content or "")[:500]
            text = f"{article.title} | {content}"
        texts.append(text)

    # Collect article metadata needed for store phase
    article_meta = [(a.id,) for a in articles]

    logger.info(f"Generating embeddings for {len(articles)} articles...")

    # Phase 2: API call — no session usage
    try:
        embeddings = await generate_embeddings_batch(texts, batch_size=batch_size)
    except Exception as e:
        logger.error(f"Embedding generation failed (provider may be down): {e}. "
                     f"{len(articles)} articles will be retried on next run.")
        return 0

    # Phase 3: Store — persist results
    count = 0
    for (article_id,), embedding in zip(article_meta, embeddings):
        if embedding is not None:
            record = ArticleEmbedding(
                article_id=article_id,
                embedding_blob=embedding.tobytes(),
                embedding_dim=len(embedding),
                model_name="mxbai-embed-large",
            )
            session.add(record)
            count += 1

    session.commit()
    logger.info(f"Generated {count}/{len(articles)} embeddings")
    if count < len(articles):
        logger.warning(f"{len(articles) - count} articles failed embedding generation")
    return count
