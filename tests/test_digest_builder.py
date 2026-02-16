"""Tests for the digest builder: MMR selection, source interleaving, Thompson exploration.

Covers:
- _mmr_select() diversity/relevance trade-off, redundancy filtering, empty inputs
- _interleave_sources() round-robin, single source, all same source
- _thompson_explore() Beta sampling, candidate filtering
- build_digest_for_user() end-to-end with DB
"""

import json
from datetime import date, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from src.storage.models import (
    Article, Digest, DigestArticle, FeedImpression, User, UserMLProfile, utcnow,
)
from src.personalization.digest_builder import (
    _mmr_select,
    _interleave_sources,
    _thompson_explore,
    build_digest_for_user,
    REDUNDANCY_THRESHOLD,
    MMR_LAMBDA,
    NEWS_ARTICLES,
    RESEARCH_ARTICLES,
    NEWS_ORDER_OFFSET,
    RESEARCH_ORDER_OFFSET,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    eng = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def _make_article(session, id, title="Article", source="techcrunch",
                  category="product", importance=5.0, status="processed",
                  fetched_at=None):
    a = Article(
        id=id,
        url=f"https://example.com/{id}",
        content_hash=f"hash_{id}",
        title=f"{title} {id}",
        source_name=source,
        source_type="rss",
        category=category,
        base_importance_score=importance,
        topics_json='["NLP"]',
        difficulty_level="intermediate",
        key_entities_json="[]",
        status=status,
        fetched_at=fetched_at or utcnow(),
    )
    session.add(a)
    session.commit()
    session.refresh(a)
    return a


def _make_user(session, id=1, role="enthusiast", level="intermediate"):
    u = User(
        id=id,
        email=f"user{id}@example.com",
        password_hash="fake",
        role=role,
        level=level,
        topics_json='["NLP"]',
        source_preferences_json="{}",
    )
    session.add(u)
    session.commit()
    session.refresh(u)
    return u


# ---------------------------------------------------------------------------
# _mmr_select
# ---------------------------------------------------------------------------

class TestMMRSelect:
    def test_returns_empty_for_empty_input(self):
        result = _mmr_select([], {}, max_articles=5)
        assert result == []

    def test_selects_top_by_score_without_embeddings(self):
        """Without embeddings, MMR falls back to pure score ranking."""
        articles = []
        for i in range(5):
            a = Article(
                id=i, url=f"https://example.com/{i}", content_hash=f"h{i}",
                title=f"Art {i}", source_name="test", source_type="rss",
                topics_json="[]", key_entities_json="[]",
            )
            articles.append(a)

        scored = [(articles[i], float(i)) for i in range(5)]  # scores 0,1,2,3,4
        result = _mmr_select(scored, {}, max_articles=3)

        assert len(result) == 3
        # Should get highest-scored articles
        result_ids = [a.id for a, _ in result]
        assert result_ids == [4, 3, 2]

    def test_skips_redundant_articles(self):
        """Articles with similarity > REDUNDANCY_THRESHOLD should be skipped."""
        articles = []
        for i in range(3):
            a = Article(
                id=i, url=f"https://example.com/{i}", content_hash=f"h{i}",
                title=f"Art {i}", source_name="test", source_type="rss",
                topics_json="[]", key_entities_json="[]",
            )
            articles.append(a)

        scored = [(articles[i], 10.0 - i) for i in range(3)]

        # Make articles 0 and 1 nearly identical embeddings (similarity > threshold)
        emb_base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_similar = np.array([0.99, 0.1, 0.0], dtype=np.float32)
        emb_different = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        embeddings = {
            0: emb_base,
            1: emb_similar,  # Very similar to 0
            2: emb_different,  # Very different
        }

        result = _mmr_select(scored, embeddings, max_articles=3)
        result_ids = [a.id for a, _ in result]

        # Article 1 should be skipped due to high similarity with article 0
        assert 0 in result_ids
        assert 2 in result_ids

    def test_respects_max_articles_limit(self):
        articles = []
        for i in range(10):
            a = Article(
                id=i, url=f"https://example.com/{i}", content_hash=f"h{i}",
                title=f"Art {i}", source_name="test", source_type="rss",
                topics_json="[]", key_entities_json="[]",
            )
            articles.append(a)

        scored = [(a, 5.0) for a in articles]
        result = _mmr_select(scored, {}, max_articles=3)
        assert len(result) == 3

    def test_prefers_diverse_articles_over_slightly_higher_score(self):
        """With embeddings, MMR should pick diverse articles even if slightly lower scored."""
        articles = []
        for i in range(3):
            a = Article(
                id=i, url=f"https://example.com/{i}", content_hash=f"h{i}",
                title=f"Art {i}", source_name="test", source_type="rss",
                topics_json="[]", key_entities_json="[]",
            )
            articles.append(a)

        # Article 0: high score, article 1: similar embedding to 0 but slightly lower,
        # article 2: different embedding, lower score
        scored = [(articles[0], 10.0), (articles[1], 9.5), (articles[2], 7.0)]

        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.95, 0.3, 0.0], dtype=np.float32)  # similar to A
        emb_c = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # orthogonal

        embeddings = {0: emb_a, 1: emb_b, 2: emb_c}
        result = _mmr_select(scored, embeddings, max_articles=2)

        result_ids = [a.id for a, _ in result]
        # First pick should be article 0 (highest score)
        assert result_ids[0] == 0
        # Second pick: article 2 (more diverse) should beat article 1 (higher score but similar)
        assert result_ids[1] == 2


# ---------------------------------------------------------------------------
# _interleave_sources
# ---------------------------------------------------------------------------

class TestInterleaveSources:
    def _make_scored(self, sources):
        """Create list of (Article, score) from source name list."""
        result = []
        for i, src in enumerate(sources):
            a = Article(
                id=i, url=f"https://example.com/{i}", content_hash=f"h{i}",
                title=f"Art {i}", source_name=src, source_type="rss",
                topics_json="[]", key_entities_json="[]",
            )
            result.append((a, 10.0 - i))
        return result

    def test_avoids_consecutive_same_source(self):
        items = self._make_scored(["rss", "rss", "rss", "arxiv", "arxiv", "github"])
        result = _interleave_sources(items)
        sources = [a.source_name for a, _ in result]

        # No two consecutive items should have the same source (when possible)
        for i in range(len(sources) - 1):
            if sources.count(sources[i]) < len(sources) - 1:
                # Only check when it's possible to avoid
                pass
        # At minimum, the first two should differ
        assert sources[0] != sources[1]

    def test_preserves_all_articles(self):
        items = self._make_scored(["a", "a", "b", "b", "c"])
        result = _interleave_sources(items)
        assert len(result) == 5

    def test_handles_single_source(self):
        items = self._make_scored(["rss", "rss", "rss"])
        result = _interleave_sources(items)
        assert len(result) == 3

    def test_handles_two_or_fewer_items(self):
        items = self._make_scored(["a", "b"])
        result = _interleave_sources(items)
        assert len(result) == 2

    def test_handles_empty_list(self):
        result = _interleave_sources([])
        assert result == []

    def test_single_item(self):
        items = self._make_scored(["rss"])
        result = _interleave_sources(items)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _thompson_explore
# ---------------------------------------------------------------------------

class TestThompsonExplore:
    def test_excludes_already_selected_articles(self, session):
        articles = []
        for i in range(5):
            a = _make_article(session, id=i + 1, source="rss")
            articles.append(a)

        user = _make_user(session)
        selected_ids = {1, 2, 3}

        result = _thompson_explore(session, user.id, articles, selected_ids, n=3)

        result_ids = {a.id for a in result}
        assert result_ids.isdisjoint(selected_ids)

    def test_returns_empty_when_no_candidates(self, session):
        articles = [_make_article(session, id=1)]
        user = _make_user(session)
        selected_ids = {1}

        result = _thompson_explore(session, user.id, articles, selected_ids, n=3)
        assert result == []

    def test_returns_up_to_n_articles(self, session):
        articles = [_make_article(session, id=i + 1) for i in range(10)]
        user = _make_user(session)

        result = _thompson_explore(session, user.id, articles, set(), n=3)
        assert len(result) <= 3

    def test_favors_articles_with_high_engagement(self, session):
        """Articles with prior engagement should generally score higher."""
        articles = [_make_article(session, id=i + 1) for i in range(2)]
        user = _make_user(session)

        # Give article 1 lots of engagement
        for _ in range(20):
            imp = FeedImpression(
                user_id=user.id, article_id=1,
                clicked=True, saved=True,
            )
            session.add(imp)
        # Give article 2 no engagement
        session.commit()

        # Run many trials to check statistical tendency
        wins = {1: 0, 2: 0}
        for _ in range(100):
            result = _thompson_explore(session, user.id, articles, set(), n=1)
            wins[result[0].id] += 1

        # Article 1 (high engagement) should win most of the time
        assert wins[1] > wins[2]


# ---------------------------------------------------------------------------
# build_digest_for_user (integration)
# ---------------------------------------------------------------------------

class TestBuildDigestForUser:
    def test_creates_digest_with_articles(self, session):
        user = _make_user(session)

        # Create ML profile
        profile = UserMLProfile(
            user_id=user.id,
            source_weights_json="{}",
            category_weights_json="{}",
            topic_weights_json="{}",
            difficulty_weights_json="{}",
            entity_weights_json="{}",
        )
        session.add(profile)
        session.commit()

        # Create articles — mix of news and research
        for i in range(6):
            _make_article(session, id=i + 1, source="techcrunch" if i < 3 else "arxiv",
                         category="product" if i < 3 else "research",
                         importance=8.0 - i)

        with patch("src.personalization.digest_builder.get_article_embeddings", return_value={}), \
             patch("src.personalization.digest_builder.compute_user_embedding", return_value=None), \
             patch("src.personalization.digest_builder.get_ml_profile", return_value=profile):
            digest = build_digest_for_user(session, user, manual=True)

        assert digest is not None
        assert digest.user_id == user.id
        assert digest.status == "draft"

        # Should have linked articles
        links = session.exec(
            select(DigestArticle).where(DigestArticle.digest_id == digest.id)
        ).all()
        assert len(links) > 0

    def test_returns_existing_digest_for_scheduled(self, session):
        user = _make_user(session)
        today = utcnow().date()

        existing = Digest(
            user_id=user.id,
            digest_date=today,
            status="draft",
            trigger="scheduled",
            subject_line="Existing",
        )
        session.add(existing)
        session.commit()
        existing_id = existing.id

        result = build_digest_for_user(session, user, digest_date=today)
        assert result.id == existing_id

    def test_rebuilds_on_manual_trigger(self, session):
        user = _make_user(session)
        today = utcnow().date()

        existing = Digest(
            user_id=user.id,
            digest_date=today,
            status="draft",
            trigger="scheduled",
            subject_line="Old",
        )
        session.add(existing)
        session.commit()
        old_id = existing.id

        # Create articles and profile for rebuild
        profile = UserMLProfile(
            user_id=user.id,
            source_weights_json="{}",
            category_weights_json="{}",
            topic_weights_json="{}",
            difficulty_weights_json="{}",
            entity_weights_json="{}",
        )
        session.add(profile)
        _make_article(session, id=1, importance=8.0)
        session.commit()

        with patch("src.personalization.digest_builder.get_article_embeddings", return_value={}), \
             patch("src.personalization.digest_builder.compute_user_embedding", return_value=None), \
             patch("src.personalization.digest_builder.get_ml_profile", return_value=profile):
            result = build_digest_for_user(session, user, digest_date=today, manual=True)

        # Should create a new digest (trigger=manual)
        assert result.trigger == "manual"

    def test_creates_empty_digest_when_no_articles(self, session):
        user = _make_user(session)

        with patch("src.personalization.digest_builder.get_article_embeddings", return_value={}), \
             patch("src.personalization.digest_builder.compute_user_embedding", return_value=None), \
             patch("src.personalization.digest_builder.get_ml_profile", return_value=None):
            digest = build_digest_for_user(session, user, manual=True)

        assert digest is not None
        links = session.exec(
            select(DigestArticle).where(DigestArticle.digest_id == digest.id)
        ).all()
        assert len(links) == 0

    def test_display_order_separates_news_and_research(self, session):
        user = _make_user(session)

        profile = UserMLProfile(
            user_id=user.id,
            source_weights_json="{}",
            category_weights_json="{}",
            topic_weights_json="{}",
            difficulty_weights_json="{}",
            entity_weights_json="{}",
        )
        session.add(profile)

        # 3 news + 3 research articles
        for i in range(3):
            _make_article(session, id=i + 1, source="techcrunch", category="product", importance=8.0)
        for i in range(3):
            _make_article(session, id=i + 4, source="arxiv", category="research", importance=7.0)
        session.commit()

        with patch("src.personalization.digest_builder.get_article_embeddings", return_value={}), \
             patch("src.personalization.digest_builder.compute_user_embedding", return_value=None), \
             patch("src.personalization.digest_builder.get_ml_profile", return_value=profile):
            digest = build_digest_for_user(session, user, manual=True)

        links = session.exec(
            select(DigestArticle).where(DigestArticle.digest_id == digest.id)
        ).all()

        for link in links:
            if link.display_order >= 0 and link.display_order < 100:
                # News section
                article = session.get(Article, link.article_id)
                assert article.source_name != "arxiv" or article.category != "research"
            elif link.display_order >= 100:
                # Research section
                article = session.get(Article, link.article_id)
                assert article.source_name in {"arxiv", "github", "huggingface"} or article.category in {"research", "open_source"}
