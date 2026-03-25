"""Batch article summarization, categorization, and ranking."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.llm.prompts import (
    ALL_TOPICS,
    BATCH_PROCESS_PROMPT,
    CATEGORIES,
    DIFFICULTY_LEVELS,
    LEVEL_ADAPT_ADVANCED_PROMPT,
    LEVEL_ADAPT_BEGINNER_PROMPT,
    SYSTEM_PROMPT,
)
from src.storage.models import Article

logger = logging.getLogger(__name__)


def _format_article_for_prompt(index: int, article: Article) -> str:
    content = (article.original_content or "")[:1000]  # Truncate long content
    return f"[{index}] Title: {article.title}\nSource: {article.source_name}\nContent: {content}\n"


def build_batch_prompt(articles: list[Article]) -> str:
    """Build the LLM prompt for batch article processing (summarize, categorize, tag)."""
    articles_text = "\n---\n".join(
        _format_article_for_prompt(i, a) for i, a in enumerate(articles)
    )
    return BATCH_PROCESS_PROMPT.format(
        count=len(articles),
        categories=", ".join(CATEGORIES),
        topics=", ".join(ALL_TOPICS),
        difficulty_levels=", ".join(DIFFICULTY_LEVELS),
        articles=articles_text,
    )


def parse_batch_response(raw_response: str, expected_count: int) -> list[dict[str, Any]]:
    """Parse the LLM JSON response, with fallback for partial results."""
    # Strip markdown code fences if present
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM response as JSON: {text[:200]}...")
        return []

    if isinstance(results, dict):
        # Some models (e.g. Ollama with format="json") return a single object
        # instead of an array for batch-of-1 or when constrained to JSON output.
        # Wrap it so the rest of the pipeline works.
        results = [results]

    if not isinstance(results, list):
        logger.error("LLM response is not a JSON array or object")
        return []

    return results


async def process_articles_batch(
    llm, articles: list[Article]
) -> list[dict[str, Any]]:
    """Send a batch of articles to the LLM for processing."""
    prompt = build_batch_prompt(articles)
    # Allow more tokens for batch responses
    max_tokens = 500 * len(articles)
    raw_response = await llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens)
    return parse_batch_response(raw_response, len(articles))


async def adapt_summary_to_level(
    llm, summary: str, level: str
) -> str:
    """Adapt an intermediate-level summary to beginner or advanced level."""
    if level == "intermediate":
        return summary

    if level == "beginner":
        prompt = LEVEL_ADAPT_BEGINNER_PROMPT.format(summary=summary)
    elif level == "advanced":
        prompt = LEVEL_ADAPT_ADVANCED_PROMPT.format(summary=summary)
    else:
        return summary

    return await llm.generate(prompt, max_tokens=200)
