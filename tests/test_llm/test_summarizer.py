"""Tests for LLM summarizer: batch prompt construction, JSON parsing, level adaptation.

Covers:
- build_batch_prompt() formatting
- parse_batch_response() with valid JSON, malformed JSON, markdown fences, non-array
- process_articles_batch() integration with mocked LLM
- adapt_summary_to_level() routing
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.summarizer import (
    build_batch_prompt,
    parse_batch_response,
    process_articles_batch,
    adapt_summary_to_level,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article(index=0, title="Test Article", source="techcrunch", content="Some content"):
    article = MagicMock()
    article.id = index
    article.title = title
    article.source_name = source
    article.original_content = content
    return article


# ---------------------------------------------------------------------------
# build_batch_prompt
# ---------------------------------------------------------------------------

class TestBuildBatchPrompt:
    def test_includes_all_articles(self):
        articles = [_make_article(i, f"Article {i}") for i in range(3)]
        prompt = build_batch_prompt(articles)
        assert "Article 0" in prompt
        assert "Article 1" in prompt
        assert "Article 2" in prompt

    def test_includes_article_index(self):
        articles = [_make_article(0, "First"), _make_article(1, "Second")]
        prompt = build_batch_prompt(articles)
        assert "[0]" in prompt
        assert "[1]" in prompt

    def test_truncates_long_content(self):
        long_content = "x" * 5000
        articles = [_make_article(0, content=long_content)]
        prompt = build_batch_prompt(articles)
        # Content should be truncated to 1000 chars
        assert "x" * 1001 not in prompt

    def test_includes_count(self):
        articles = [_make_article(i) for i in range(5)]
        prompt = build_batch_prompt(articles)
        assert "5 articles" in prompt

    def test_includes_categories_and_topics(self):
        articles = [_make_article()]
        prompt = build_batch_prompt(articles)
        assert "research" in prompt
        assert "NLP" in prompt


# ---------------------------------------------------------------------------
# parse_batch_response
# ---------------------------------------------------------------------------

class TestParseBatchResponse:
    def test_valid_json_array(self):
        data = [
            {"index": 0, "category": "research", "summary_student": "A study"},
            {"index": 1, "category": "product", "summary_student": "A product"},
        ]
        result = parse_batch_response(json.dumps(data), expected_count=2)
        assert len(result) == 2
        assert result[0]["category"] == "research"

    def test_strips_markdown_code_fences(self):
        data = [{"index": 0, "category": "research"}]
        raw = f"```json\n{json.dumps(data)}\n```"
        result = parse_batch_response(raw, expected_count=1)
        assert len(result) == 1

    def test_strips_code_fence_without_language(self):
        data = [{"index": 0}]
        raw = f"```\n{json.dumps(data)}\n```"
        result = parse_batch_response(raw, expected_count=1)
        assert len(result) == 1

    def test_returns_empty_for_invalid_json(self):
        result = parse_batch_response("this is not json at all", expected_count=1)
        assert result == []

    def test_returns_empty_for_non_array_json(self):
        result = parse_batch_response('{"key": "value"}', expected_count=1)
        assert result == []

    def test_returns_empty_for_empty_string(self):
        result = parse_batch_response("", expected_count=0)
        assert result == []

    def test_handles_partial_results(self):
        """Parser should return whatever valid entries exist, even if fewer than expected."""
        data = [{"index": 0, "category": "research"}]
        result = parse_batch_response(json.dumps(data), expected_count=5)
        assert len(result) == 1

    def test_handles_whitespace_padding(self):
        data = [{"index": 0}]
        raw = f"  \n  {json.dumps(data)}  \n  "
        result = parse_batch_response(raw, expected_count=1)
        assert len(result) == 1

    def test_handles_nested_code_fence(self):
        """Markdown fence with ```json ... ``` wrapping."""
        data = [{"index": 0, "topics": ["NLP"]}]
        raw = "```json\n" + json.dumps(data, indent=2) + "\n```"
        result = parse_batch_response(raw, expected_count=1)
        assert result[0]["topics"] == ["NLP"]


# ---------------------------------------------------------------------------
# process_articles_batch
# ---------------------------------------------------------------------------

class TestProcessArticlesBatch:
    @pytest.mark.asyncio
    async def test_sends_prompt_and_parses_response(self):
        mock_llm = AsyncMock()
        data = [{"index": 0, "category": "research", "summary_student": "Good paper"}]
        mock_llm.generate.return_value = json.dumps(data)

        articles = [_make_article(0)]
        result = await process_articles_batch(mock_llm, articles)

        assert len(result) == 1
        assert result[0]["summary_student"] == "Good paper"
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculates_max_tokens_from_batch_size(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "[]"

        articles = [_make_article(i) for i in range(5)]
        await process_articles_batch(mock_llm, articles)

        # max_tokens should be 500 * len(articles) = 2500
        call_kwargs = mock_llm.generate.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 2500 or call_kwargs[1].get("max_tokens") == 2500

    @pytest.mark.asyncio
    async def test_handles_llm_returning_garbage(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "I cannot process these articles."

        articles = [_make_article(0)]
        result = await process_articles_batch(mock_llm, articles)

        assert result == []


# ---------------------------------------------------------------------------
# adapt_summary_to_level
# ---------------------------------------------------------------------------

class TestAdaptSummaryToLevel:
    @pytest.mark.asyncio
    async def test_intermediate_returns_as_is(self):
        mock_llm = AsyncMock()
        result = await adapt_summary_to_level(mock_llm, "Original summary", "intermediate")
        assert result == "Original summary"
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_beginner_calls_llm(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Simplified version"

        result = await adapt_summary_to_level(mock_llm, "Technical summary", "beginner")
        assert result == "Simplified version"
        # Verify the prompt contains the original summary
        call_args = mock_llm.generate.call_args[0][0]
        assert "Technical summary" in call_args

    @pytest.mark.asyncio
    async def test_advanced_calls_llm(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "Condensed version"

        result = await adapt_summary_to_level(mock_llm, "Regular summary", "advanced")
        assert result == "Condensed version"

    @pytest.mark.asyncio
    async def test_unknown_level_returns_as_is(self):
        mock_llm = AsyncMock()
        result = await adapt_summary_to_level(mock_llm, "Original", "expert")
        assert result == "Original"
        mock_llm.generate.assert_not_called()
