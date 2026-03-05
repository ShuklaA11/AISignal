"""Tests for storage query functions.

Covers:
- User lookup by email/id
- Article existence checks (by URL and title fingerprint)
- Title normalization and fingerprinting
- Saved/read article toggling (idempotent)
- Feed impression recording and deduplication
- Impression feedback state changes (liked/disliked mutual exclusion)
- ML profile get-or-create
- Expired token cleanup
"""

import json
from datetime import datetime, timedelta

import pytest
from sqlmodel import Session, SQLModel, create_engine

from src.storage.models import (
    Article, FeedImpression, Token, User, UserMLProfile, utcnow,
)
from src.storage.queries import (
    _normalize_title,
    _title_fingerprint,
    article_exists,
    article_exists_by_title,
    cleanup_expired_tokens,
    get_or_create_ml_profile,
    get_user_by_email,
    get_user_by_id,
    mark_article_read,
    record_impressions,
    toggle_saved_article,
    update_impression_clicked,
    update_impression_disliked,
    update_impression_liked,
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


def _make_user(session, id=1, email="user@test.com"):
    user = User(
        id=id,
        email=email,
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


def _make_article(session, id=1, url="https://example.com/1", title="Test Article"):
    article = Article(
        id=id,
        url=url,
        content_hash="abc123",
        title=title,
        source_name="rss",
        source_type="rss",
        status="processed",
    )
    session.add(article)
    session.commit()
    session.refresh(article)
    return article


# ===========================================================================
# 1. get_user_by_email / get_user_by_id
# ===========================================================================

class TestUserLookup:
    def test_get_user_by_email_found(self, session):
        user = _make_user(session, email="alice@test.com")
        result = get_user_by_email(session, "alice@test.com")
        assert result is not None
        assert result.id == user.id

    def test_get_user_by_email_not_found(self, session):
        result = get_user_by_email(session, "nobody@test.com")
        assert result is None

    def test_get_user_by_id_found(self, session):
        user = _make_user(session)
        result = get_user_by_id(session, user.id)
        assert result is not None
        assert result.email == user.email

    def test_get_user_by_id_not_found(self, session):
        result = get_user_by_id(session, 9999)
        assert result is None


# ===========================================================================
# 2. article_exists
# ===========================================================================

class TestArticleExists:
    def test_existing_url(self, session):
        _make_article(session, url="https://example.com/exists")
        assert article_exists(session, "https://example.com/exists") is True

    def test_nonexistent_url(self, session):
        assert article_exists(session, "https://example.com/nope") is False


# ===========================================================================
# 3. Title normalization and fingerprinting
# ===========================================================================

class TestTitleNormalization:
    def test_normalize_lowercases(self):
        assert _normalize_title("Hello World") == "hello world"

    def test_normalize_strips_punctuation(self):
        assert _normalize_title("Hello, World!") == "hello world"

    def test_normalize_collapses_whitespace(self):
        assert _normalize_title("  hello   world  ") == "hello world"

    def test_normalize_combined(self):
        # The em dash (—) is stripped as punctuation, whitespace collapsed
        assert _normalize_title("  Foo:  Bar — Baz!  ") == "foo bar baz"

    def test_fingerprint_uses_first_6_words(self):
        title = "one two three four five six seven eight"
        fp = _title_fingerprint(title)
        assert fp == "one two three four five six"

    def test_fingerprint_short_title(self):
        fp = _title_fingerprint("Short Title")
        assert fp == "short title"

    def test_fingerprint_catches_punctuation_variants(self):
        fp1 = _title_fingerprint("Foo Bar: A Study")
        fp2 = _title_fingerprint("Foo Bar — A Study in X")
        # "Foo Bar: A Study" -> "foo bar a study" (4 words, all used)
        assert fp1 == "foo bar a study"
        # "Foo Bar — A Study in X" -> "foo bar a study in x" (6 words)
        assert fp2 == "foo bar a study in x"

    def test_fingerprint_matches_same_prefix(self):
        """Titles sharing the same first 6 words produce the same fingerprint."""
        fp1 = _title_fingerprint("Alpha Beta Gamma Delta Epsilon Zeta Extra Words")
        fp2 = _title_fingerprint("Alpha Beta Gamma Delta Epsilon Zeta Different Ending")
        assert fp1 == fp2

    def test_article_exists_by_title_match(self):
        existing = {"hello world this is a test"}
        assert article_exists_by_title("Hello World This Is A Test!", existing) is True

    def test_article_exists_by_title_no_match(self):
        existing = {"hello world this is a"}
        assert article_exists_by_title("Something Completely Different", existing) is False

    def test_article_exists_by_title_empty_title(self):
        existing = {"hello world"}
        assert article_exists_by_title("", existing) is False


# ===========================================================================
# 4. toggle_saved_article
# ===========================================================================

class TestToggleSavedArticle:
    def test_save_unsave_resave(self, session):
        user = _make_user(session)
        article = _make_article(session)

        # Save
        result = toggle_saved_article(session, user.id, article.id)
        assert result is True

        # Unsave
        result = toggle_saved_article(session, user.id, article.id)
        assert result is False

        # Re-save
        result = toggle_saved_article(session, user.id, article.id)
        assert result is True


# ===========================================================================
# 5. mark_article_read (idempotent)
# ===========================================================================

class TestMarkArticleRead:
    def test_mark_read_creates_record(self, session):
        user = _make_user(session)
        article = _make_article(session)

        mark_article_read(session, user.id, article.id)

        from src.storage.models import ReadArticle
        from sqlmodel import select
        stmt = select(ReadArticle).where(
            ReadArticle.user_id == user.id,
            ReadArticle.article_id == article.id,
        )
        assert session.exec(stmt).first() is not None

    def test_mark_read_idempotent(self, session):
        user = _make_user(session)
        article = _make_article(session)

        mark_article_read(session, user.id, article.id)
        mark_article_read(session, user.id, article.id)

        from src.storage.models import ReadArticle
        from sqlmodel import select
        stmt = select(ReadArticle).where(
            ReadArticle.user_id == user.id,
            ReadArticle.article_id == article.id,
        )
        results = list(session.exec(stmt).all())
        assert len(results) == 1


# ===========================================================================
# 6. record_impressions
# ===========================================================================

class TestRecordImpressions:
    def test_bulk_insert(self, session):
        user = _make_user(session)
        for i in range(1, 4):
            _make_article(session, id=i, url=f"https://example.com/{i}")

        # Need ML profile for counter update
        profile = UserMLProfile(user_id=user.id)
        session.add(profile)
        session.commit()

        record_impressions(session, user.id, [1, 2, 3], feed_group="today")

        from sqlmodel import select
        stmt = select(FeedImpression).where(FeedImpression.user_id == user.id)
        impressions = list(session.exec(stmt).all())
        assert len(impressions) == 3

    def test_dedup_within_one_hour(self, session):
        user = _make_user(session)
        _make_article(session, id=1)

        profile = UserMLProfile(user_id=user.id)
        session.add(profile)
        session.commit()

        record_impressions(session, user.id, [1], feed_group="today")
        record_impressions(session, user.id, [1], feed_group="today")

        from sqlmodel import select
        stmt = select(FeedImpression).where(FeedImpression.user_id == user.id)
        impressions = list(session.exec(stmt).all())
        assert len(impressions) == 1

    def test_allows_after_one_hour(self, session):
        user = _make_user(session)
        _make_article(session, id=1)

        profile = UserMLProfile(user_id=user.id)
        session.add(profile)
        session.commit()

        # Insert an old impression (>1 hour ago)
        old_imp = FeedImpression(
            user_id=user.id,
            article_id=1,
            shown_at=utcnow() - timedelta(hours=2),
            position=0,
            feed_group="today",
        )
        session.add(old_imp)
        session.commit()

        # Should allow a new impression for the same article
        record_impressions(session, user.id, [1], feed_group="today")

        from sqlmodel import select
        stmt = select(FeedImpression).where(FeedImpression.user_id == user.id)
        impressions = list(session.exec(stmt).all())
        assert len(impressions) == 2

    def test_increments_ml_profile_counter(self, session):
        user = _make_user(session)
        for i in range(1, 4):
            _make_article(session, id=i, url=f"https://example.com/{i}")

        profile = UserMLProfile(user_id=user.id)
        session.add(profile)
        session.commit()

        record_impressions(session, user.id, [1, 2, 3], feed_group="today")

        session.refresh(profile)
        assert profile.total_impressions == 3


# ===========================================================================
# 7. Impression feedback: clicked / liked / disliked
# ===========================================================================

class TestImpressionFeedback:
    def _setup_impression(self, session):
        user = _make_user(session)
        article = _make_article(session)
        imp = FeedImpression(
            user_id=user.id,
            article_id=article.id,
            shown_at=utcnow(),
            position=0,
            feed_group="today",
        )
        session.add(imp)
        session.commit()
        return user, article, imp

    def test_update_clicked(self, session):
        user, article, imp = self._setup_impression(session)
        update_impression_clicked(session, user.id, article.id)
        session.refresh(imp)
        assert imp.clicked is True

    def test_liked_clears_disliked(self, session):
        user, article, imp = self._setup_impression(session)

        # First dislike
        update_impression_disliked(session, user.id, article.id)
        session.refresh(imp)
        assert imp.disliked is True
        assert imp.liked is False

        # Then like (should clear disliked)
        update_impression_liked(session, user.id, article.id)
        session.refresh(imp)
        assert imp.liked is True
        assert imp.disliked is False

    def test_disliked_clears_liked(self, session):
        user, article, imp = self._setup_impression(session)

        update_impression_liked(session, user.id, article.id)
        session.refresh(imp)
        assert imp.liked is True

        update_impression_disliked(session, user.id, article.id)
        session.refresh(imp)
        assert imp.disliked is True
        assert imp.liked is False

    def test_no_impression_is_noop(self, session):
        """Updating feedback on a nonexistent impression does not raise."""
        _make_user(session)
        update_impression_clicked(session, 1, 9999)  # No crash


# ===========================================================================
# 8. get_or_create_ml_profile
# ===========================================================================

class TestGetOrCreateMLProfile:
    def test_creates_if_missing(self, session):
        user = _make_user(session)
        profile = get_or_create_ml_profile(session, user.id)
        assert profile is not None
        assert profile.user_id == user.id
        assert profile.total_impressions == 0

    def test_returns_existing(self, session):
        user = _make_user(session)
        profile1 = get_or_create_ml_profile(session, user.id)
        profile1.total_clicks = 42
        session.add(profile1)
        session.commit()

        profile2 = get_or_create_ml_profile(session, user.id)
        assert profile2.id == profile1.id
        assert profile2.total_clicks == 42


# ===========================================================================
# 9. cleanup_expired_tokens
# ===========================================================================

class TestCleanupExpiredTokens:
    def test_deletes_expired_tokens(self, session):
        user = _make_user(session)
        # Expired token
        t1 = Token(
            user_id=user.id,
            token_hash="expired_hash",
            token_type="email_verification",
            expires_at=utcnow() - timedelta(hours=1),
        )
        # Valid token
        t2 = Token(
            user_id=user.id,
            token_hash="valid_hash",
            token_type="email_verification",
            expires_at=utcnow() + timedelta(hours=24),
        )
        session.add(t1)
        session.add(t2)
        session.commit()

        removed = cleanup_expired_tokens(session)
        assert removed == 1

        from sqlmodel import select
        remaining = list(session.exec(select(Token)).all())
        assert len(remaining) == 1
        assert remaining[0].token_hash == "valid_hash"

    def test_deletes_used_tokens(self, session):
        user = _make_user(session)
        t = Token(
            user_id=user.id,
            token_hash="used_hash",
            token_type="password_reset",
            expires_at=utcnow() + timedelta(hours=24),
            used_at=utcnow(),
        )
        session.add(t)
        session.commit()

        removed = cleanup_expired_tokens(session)
        assert removed == 1

    def test_keeps_valid_unused_tokens(self, session):
        user = _make_user(session)
        t = Token(
            user_id=user.id,
            token_hash="fresh_hash",
            token_type="email_verification",
            expires_at=utcnow() + timedelta(hours=24),
        )
        session.add(t)
        session.commit()

        removed = cleanup_expired_tokens(session)
        assert removed == 0
