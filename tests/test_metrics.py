"""Tests for metrics calculator: CTR, save rate, nDCG@k, personalization lift.

Covers:
- compute_ctr() with zero impressions, filtered by date/view
- compute_save_rate() edge cases
- compute_ndcg_at_k() ranking quality, all-zero relevance, perfect ranking, division by zero
- compute_daily_metrics() aggregation and upsert behavior
"""

import math
from datetime import date, datetime, timedelta

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from src.storage.models import FeedImpression, ScoringMetric, User, utcnow
from src.metrics.calculator import (
    compute_ctr,
    compute_save_rate,
    compute_ndcg_at_k,
    compute_daily_metrics,
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


def _make_user(session, id=1):
    user = User(
        id=id,
        email=f"user{id}@test.com",
        password_hash="fake",
        role="enthusiast",
        level="intermediate",
        topics_json="[]",
        source_preferences_json="{}",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _make_impression(session, user_id, article_id, clicked=False, saved=False,
                     position=0, shown_at=None, feed_view="all"):
    imp = FeedImpression(
        user_id=user_id,
        article_id=article_id,
        clicked=clicked,
        saved=saved,
        position=position,
        shown_at=shown_at or utcnow(),
        feed_view=feed_view,
    )
    session.add(imp)
    session.commit()
    return imp


# ---------------------------------------------------------------------------
# compute_ctr
# ---------------------------------------------------------------------------

class TestComputeCTR:
    def test_zero_impressions_returns_zero(self, session):
        user = _make_user(session)
        result = compute_ctr(session, user.id)
        assert result == 0.0

    def test_basic_ctr_calculation(self, session):
        user = _make_user(session)
        # 2 clicks out of 10 impressions = 0.2
        for i in range(10):
            _make_impression(session, user.id, article_id=i + 1,
                           clicked=(i < 2))

        result = compute_ctr(session, user.id)
        assert result == 0.2

    def test_all_clicked(self, session):
        user = _make_user(session)
        for i in range(5):
            _make_impression(session, user.id, article_id=i + 1, clicked=True)

        result = compute_ctr(session, user.id)
        assert result == 1.0

    def test_none_clicked(self, session):
        user = _make_user(session)
        for i in range(5):
            _make_impression(session, user.id, article_id=i + 1, clicked=False)

        result = compute_ctr(session, user.id)
        assert result == 0.0

    def test_filters_by_date_range(self, session):
        user = _make_user(session)
        now = utcnow()
        yesterday = now - timedelta(days=1)
        two_days_ago = now - timedelta(days=2)

        # Old impression (should be excluded)
        _make_impression(session, user.id, article_id=1, clicked=True,
                        shown_at=two_days_ago)
        # Recent impression (should be included)
        _make_impression(session, user.id, article_id=2, clicked=False,
                        shown_at=now)

        result = compute_ctr(session, user.id, start=yesterday, end=now + timedelta(hours=1))
        assert result == 0.0  # Only the non-clicked one is in range

    def test_filters_by_feed_view(self, session):
        user = _make_user(session)
        _make_impression(session, user.id, article_id=1, clicked=True, feed_view="for_you")
        _make_impression(session, user.id, article_id=2, clicked=False, feed_view="all")

        result = compute_ctr(session, user.id, feed_view="for_you")
        assert result == 1.0  # Only the for_you impression counts


# ---------------------------------------------------------------------------
# compute_save_rate
# ---------------------------------------------------------------------------

class TestComputeSaveRate:
    def test_zero_impressions_returns_zero(self, session):
        user = _make_user(session)
        result = compute_save_rate(session, user.id)
        assert result == 0.0

    def test_basic_save_rate(self, session):
        user = _make_user(session)
        # 3 saves out of 10
        for i in range(10):
            _make_impression(session, user.id, article_id=i + 1,
                           saved=(i < 3))

        result = compute_save_rate(session, user.id)
        assert result == 0.3

    def test_filters_by_date_range(self, session):
        user = _make_user(session)
        now = utcnow()
        old = now - timedelta(days=5)

        _make_impression(session, user.id, article_id=1, saved=True, shown_at=old)
        _make_impression(session, user.id, article_id=2, saved=False, shown_at=now)

        result = compute_save_rate(session, user.id,
                                  start=now - timedelta(hours=1),
                                  end=now + timedelta(hours=1))
        assert result == 0.0


# ---------------------------------------------------------------------------
# compute_ndcg_at_k
# ---------------------------------------------------------------------------

class TestComputeNDCG:
    def test_zero_impressions_returns_zero(self, session):
        user = _make_user(session)
        result = compute_ndcg_at_k(session, user.id, k=10)
        assert result == 0.0

    def test_perfect_ranking(self, session):
        """If the best articles are ranked first, nDCG should be 1.0."""
        user = _make_user(session)
        # Position 0: saved (rel=3), Position 1: clicked (rel=1), Position 2: skipped (rel=0)
        _make_impression(session, user.id, article_id=1, saved=True, position=0)
        _make_impression(session, user.id, article_id=2, clicked=True, position=1)
        _make_impression(session, user.id, article_id=3, position=2)

        result = compute_ndcg_at_k(session, user.id, k=3)
        assert result == 1.0

    def test_worst_ranking(self, session):
        """If the best article is ranked last, nDCG should be < 1.0."""
        user = _make_user(session)
        # Position 0: skipped, Position 1: skipped, Position 2: saved
        _make_impression(session, user.id, article_id=1, position=0)
        _make_impression(session, user.id, article_id=2, position=1)
        _make_impression(session, user.id, article_id=3, saved=True, position=2)

        result = compute_ndcg_at_k(session, user.id, k=3)
        assert result < 1.0
        assert result > 0.0

    def test_all_skipped_returns_zero(self, session):
        """If no articles are clicked or saved, nDCG should be 0.0."""
        user = _make_user(session)
        for i in range(5):
            _make_impression(session, user.id, article_id=i + 1, position=i)

        result = compute_ndcg_at_k(session, user.id, k=5)
        assert result == 0.0

    def test_all_saved_returns_one(self, session):
        """If all articles are saved and ranking doesn't matter, nDCG should be 1.0."""
        user = _make_user(session)
        for i in range(3):
            _make_impression(session, user.id, article_id=i + 1, saved=True, position=i)

        result = compute_ndcg_at_k(session, user.id, k=3)
        assert result == 1.0

    def test_k_larger_than_impressions(self, session):
        """Should handle k > number of impressions gracefully."""
        user = _make_user(session)
        _make_impression(session, user.id, article_id=1, clicked=True, position=0)

        result = compute_ndcg_at_k(session, user.id, k=100)
        assert result == 1.0  # Only 1 impression, it's clicked, so it's perfect

    def test_saved_counts_more_than_clicked(self, session):
        """Verify relevance weighting: saved=3 > clicked=1."""
        user = _make_user(session)
        # Scenario: saved at position 0 vs clicked at position 0
        _make_impression(session, user.id, article_id=1, saved=True, position=0)
        _make_impression(session, user.id, article_id=2, clicked=True, position=1)

        result = compute_ndcg_at_k(session, user.id, k=2)
        # DCG: 3/log2(2) + 1/log2(3) = 3.0 + 0.6309 = 3.6309
        # IDCG: same since 3 > 1 (already optimal order)
        assert result == 1.0


# ---------------------------------------------------------------------------
# compute_daily_metrics (integration)
# ---------------------------------------------------------------------------

class TestComputeDailyMetrics:
    def test_creates_metric_for_day(self, session):
        user = _make_user(session)
        today = utcnow().date()
        now = utcnow()

        # Create some impressions today
        for i in range(10):
            _make_impression(session, user.id, article_id=i + 1,
                           clicked=(i < 3), saved=(i < 1),
                           position=i, shown_at=now)

        metric = compute_daily_metrics(session, user.id, today)

        assert metric.user_id == user.id
        assert metric.metric_date == today
        assert metric.ctr == 0.3
        assert metric.save_rate == 0.1
        assert metric.total_impressions == 10
        assert metric.total_clicks == 3
        assert metric.total_saves == 1

    def test_upserts_existing_metric(self, session):
        user = _make_user(session)
        today = utcnow().date()
        now = utcnow()

        # Initial impressions
        _make_impression(session, user.id, article_id=1, clicked=True,
                        position=0, shown_at=now)

        metric1 = compute_daily_metrics(session, user.id, today)
        assert metric1.ctr == 1.0

        # Add more impressions
        _make_impression(session, user.id, article_id=2, clicked=False,
                        position=1, shown_at=now)

        metric2 = compute_daily_metrics(session, user.id, today)
        assert metric2.ctr == 0.5

        # Should be the same row (upsert)
        assert metric2.id == metric1.id

    def test_no_impressions_returns_zeroed_metric(self, session):
        user = _make_user(session)
        today = utcnow().date()

        metric = compute_daily_metrics(session, user.id, today)

        assert metric.ctr == 0.0
        assert metric.save_rate == 0.0
        assert metric.ndcg_at_10 == 0.0
        assert metric.personalization_lift == 1.0
        assert metric.total_impressions == 0

    def test_personalization_lift_with_for_you_view(self, session):
        user = _make_user(session)
        today = utcnow().date()
        now = utcnow()

        # 'for_you' feed: 2/2 clicked = 1.0 CTR
        _make_impression(session, user.id, article_id=1, clicked=True,
                        feed_view="for_you", shown_at=now)
        _make_impression(session, user.id, article_id=2, clicked=True,
                        feed_view="for_you", shown_at=now)
        # 'all' feed: 0/2 clicked = 0.0 CTR
        _make_impression(session, user.id, article_id=3, clicked=False,
                        feed_view="all", shown_at=now)
        _make_impression(session, user.id, article_id=4, clicked=False,
                        feed_view="all", shown_at=now)

        metric = compute_daily_metrics(session, user.id, today)

        # Overall CTR: 2/4 = 0.5
        # For-you CTR: 2/2 = 1.0
        # Lift: 1.0 / 0.5 = 2.0
        assert metric.personalization_lift == 2.0

    def test_personalization_lift_neutral_when_no_for_you(self, session):
        user = _make_user(session)
        today = utcnow().date()
        now = utcnow()

        _make_impression(session, user.id, article_id=1, clicked=True,
                        feed_view="all", shown_at=now)

        metric = compute_daily_metrics(session, user.id, today)
        assert metric.personalization_lift == 1.0  # Default neutral
