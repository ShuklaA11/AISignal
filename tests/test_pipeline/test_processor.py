"""Tests for pipeline processor: topic validation, result application, summary storage, processing run.

Covers:
- _validate_topics() valid/invalid/empty fallback
- _apply_result_to_article() field mapping
- _store_summaries() creates 3 role summaries, skips existing, handles empty
- run_processing() end-to-end with mocked LLM
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

from src.pipeline.processor import (
    _apply_result_to_article,
    _store_summaries,
    _validate_topics,
    run_processing,
)
from src.storage.models import Article, ArticleSummary, utcnow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """In-memory SQLite for testing."""
    eng = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def _make_article(session, idx=1, status="raw"):
    """Create and persist an Article in the DB."""
    article = Article(
        url=f"https://example.com/{idx}",
        content_hash=f"hash{idx}",
        title=f"Test Article {idx}",
        source_name="rss_test",
        source_type="rss",
        original_content="Some content for processing.",
        fetched_at=utcnow(),
        status=status,
    )
    session.add(article)
    session.commit()
    session.refresh(article)
    return article


def _make_llm_result(index=0):
    """Return a well-formed LLM result dict."""
    return {
        "index": index,
        "category": "Research",
        "base_importance_score": 0.85,
        "difficulty_level": "intermediate",
        "topics": ["NLP", "AI Safety"],
        "key_entities": ["GPT-5", "Anthropic"],
        "summary_student": "Student summary text.",
        "summary_industry": "Industry summary text.",
        "summary_enthusiast": "Enthusiast summary text.",
    }


# ---------------------------------------------------------------------------
# _validate_topics
# ---------------------------------------------------------------------------

class TestValidateTopics:
    def test_valid_topics_pass_through(self):
        result = _validate_topics(["NLP", "AI Safety", "Robotics"])
        assert result == ["NLP", "AI Safety", "Robotics"]

    def test_invalid_topics_filtered(self):
        result = _validate_topics(["NLP", "Fake Topic", "AI Safety"])
        assert "Fake Topic" not in result
        assert "NLP" in result
        assert "AI Safety" in result

    def test_all_invalid_falls_back(self):
        result = _validate_topics(["Not A Topic", "Also Fake"])
        assert result == ["General AI"]

    def test_empty_list_falls_back(self):
        result = _validate_topics([])
        assert result == ["General AI"]

    def test_mixed_valid_invalid(self):
        result = _validate_topics(["Computer Vision", "Bogus"])
        assert result == ["Computer Vision"]


# ---------------------------------------------------------------------------
# _apply_result_to_article
# ---------------------------------------------------------------------------

class TestApplyResultToArticle:
    def test_applies_all_fields(self, session):
        article = _make_article(session)
        result = _make_llm_result()
        _apply_result_to_article(article, result)

        assert article.category == "Research"
        assert article.base_importance_score == 0.85
        assert article.difficulty_level == "intermediate"
        assert article.status == "processed"
        assert "NLP" in article.topics
        assert "AI Safety" in article.topics
        assert "GPT-5" in article.key_entities

    def test_topics_validated(self, session):
        article = _make_article(session)
        result = _make_llm_result()
        result["topics"] = ["Bogus", "NLP"]
        _apply_result_to_article(article, result)
        topics = json.loads(article.topics_json)
        assert "NLP" in topics
        assert "Bogus" not in topics

    def test_missing_fields_set_to_none(self, session):
        article = _make_article(session)
        _apply_result_to_article(article, {})
        assert article.category is None
        assert article.base_importance_score is None
        assert article.status == "processed"


# ---------------------------------------------------------------------------
# _store_summaries
# ---------------------------------------------------------------------------

class TestStoreSummaries:
    def test_stores_three_role_summaries(self, session):
        article = _make_article(session)
        result = _make_llm_result()
        count = _store_summaries(session, article, result)
        session.commit()
        assert count == 3

        from sqlmodel import select
        summaries = session.exec(
            select(ArticleSummary).where(ArticleSummary.article_id == article.id)
        ).all()
        assert len(summaries) == 3
        roles = {s.role for s in summaries}
        assert roles == {"student", "industry", "enthusiast"}
        for s in summaries:
            assert s.level == "intermediate"

    def test_skips_existing_summaries(self, session):
        article = _make_article(session)
        # Pre-populate a student summary
        existing = ArticleSummary(
            article_id=article.id,
            role="student",
            level="intermediate",
            summary_text="Existing student summary.",
        )
        session.add(existing)
        session.commit()

        result = _make_llm_result()
        count = _store_summaries(session, article, result)
        session.commit()
        # 1 existing (counted) + 2 new = 3
        assert count == 3

        from sqlmodel import select
        summaries = session.exec(
            select(ArticleSummary).where(ArticleSummary.article_id == article.id)
        ).all()
        assert len(summaries) == 3

    def test_handles_empty_summaries(self, session):
        article = _make_article(session)
        result = _make_llm_result()
        result["summary_student"] = ""
        result["summary_industry"] = ""
        result["summary_enthusiast"] = ""
        count = _store_summaries(session, article, result)
        session.commit()
        # All empty -> nothing created
        assert count == 0

    def test_partial_empty_summaries(self, session):
        article = _make_article(session)
        result = _make_llm_result()
        result["summary_student"] = ""  # empty
        count = _store_summaries(session, article, result)
        session.commit()
        # industry + enthusiast created = 2
        assert count == 2


# ---------------------------------------------------------------------------
# run_processing
# ---------------------------------------------------------------------------

class TestRunProcessing:
    @pytest.mark.asyncio
    async def test_processes_raw_articles(self, engine):
        """Raw articles are processed to 'processed' status via mocked LLM."""
        # Set up DB with raw articles
        with Session(engine) as session:
            for i in range(3):
                _make_article(session, idx=i, status="raw")

        # Mock LLM to return valid results
        llm_results = [_make_llm_result(i) for i in range(3)]

        mock_settings = MagicMock()
        mock_settings.database_url = "sqlite://"
        mock_settings.llm.batch_size = 10

        with patch("src.pipeline.processor.init_db") as mock_init_db, \
             patch("src.pipeline.processor.LLMProvider") as mock_llm_cls, \
             patch("src.pipeline.processor.session_scope") as mock_scope, \
             patch("src.pipeline.processor.process_articles_batch", new_callable=AsyncMock) as mock_batch:

            mock_batch.return_value = llm_results

            # Wire session_scope to use our engine
            from contextlib import contextmanager

            @contextmanager
            def fake_scope(url):
                with Session(engine) as s:
                    yield s

            mock_scope.side_effect = fake_scope

            count = await run_processing(settings=mock_settings, batch_size=10)

        assert count == 3

        # Verify articles are now "processed"
        with Session(engine) as session:
            from sqlmodel import select
            articles = session.exec(select(Article)).all()
            for a in articles:
                assert a.status == "processed"

    @pytest.mark.asyncio
    async def test_no_raw_articles_returns_zero(self, engine):
        """When there are no raw articles, returns 0 immediately."""
        mock_settings = MagicMock()
        mock_settings.database_url = "sqlite://"

        with patch("src.pipeline.processor.init_db"), \
             patch("src.pipeline.processor.LLMProvider"), \
             patch("src.pipeline.processor.session_scope") as mock_scope:

            from contextlib import contextmanager

            @contextmanager
            def fake_scope(url):
                with Session(engine) as s:
                    yield s

            mock_scope.side_effect = fake_scope

            count = await run_processing(settings=mock_settings, batch_size=10)

        assert count == 0

    @pytest.mark.asyncio
    async def test_failed_batch_leaves_articles_raw(self, engine):
        """When LLM batch fails all retries, articles stay as 'raw'."""
        with Session(engine) as session:
            _make_article(session, idx=1, status="raw")

        mock_settings = MagicMock()
        mock_settings.database_url = "sqlite://"
        mock_settings.llm.batch_size = 10

        with patch("src.pipeline.processor.init_db"), \
             patch("src.pipeline.processor.LLMProvider"), \
             patch("src.pipeline.processor.session_scope") as mock_scope, \
             patch("src.pipeline.processor.process_articles_batch", new_callable=AsyncMock) as mock_batch, \
             patch("src.pipeline.processor.asyncio.sleep", new_callable=AsyncMock):

            mock_batch.side_effect = Exception("LLM unavailable")

            from contextlib import contextmanager

            @contextmanager
            def fake_scope(url):
                with Session(engine) as s:
                    yield s

            mock_scope.side_effect = fake_scope

            count = await run_processing(settings=mock_settings, batch_size=10)

        assert count == 0

        # Verify article still raw
        with Session(engine) as session:
            from sqlmodel import select
            article = session.exec(select(Article)).first()
            assert article.status == "raw"
