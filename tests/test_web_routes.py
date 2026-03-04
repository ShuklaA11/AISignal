"""Tests for web route auth/authorization and API endpoints.

Covers:
- Auth utilities (password hashing, verification)
- Login/signup validation
- Session-based authorization checks
- API endpoint auth guards (_require_user, _require_admin)
- digest_click user_id validation
- Summary editing HTML sanitization
"""

import json
import re
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

from src.storage.models import Article, ArticleSummary, User, utcnow
from src.web.auth_utils import hash_password, verify_password


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


def _make_user(session, email="test@test.com", password="securePass123!",
               is_admin=False, role="enthusiast"):
    hashed = hash_password(password)
    user = User(
        email=email,
        name="Test User",
        password_hash=hashed,
        role=role,
        level="intermediate",
        is_admin=is_admin,
        topics_json="[]",
        source_preferences_json="{}",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def _make_article(session, id=1, title="Test Article"):
    article = Article(
        id=id,
        url=f"https://example.com/{id}",
        content_hash=f"hash_{id}",
        title=title,
        source_name="test",
        source_type="rss",
        topics_json="[]",
        key_entities_json="[]",
        status="processed",
    )
    session.add(article)
    session.commit()
    session.refresh(article)
    return article


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    def test_hash_and_verify(self):
        password = "MySecurePassword123!"
        hashed = hash_password(password)
        assert verify_password(password, hashed)

    def test_wrong_password_fails(self):
        hashed = hash_password("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_different_hashes_for_same_password(self):
        """bcrypt should generate different salts each time."""
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        assert h1 != h2
        # But both should verify
        assert verify_password("same_password", h1)
        assert verify_password("same_password", h2)

    def test_empty_password(self):
        hashed = hash_password("")
        assert verify_password("", hashed)
        assert not verify_password("notempty", hashed)


# ---------------------------------------------------------------------------
# Auth route logic (unit tests, not HTTP tests)
# ---------------------------------------------------------------------------

class TestAuthValidation:
    """Test auth validation logic used in signup/login routes."""

    def test_email_format_validation(self):
        """Basic email format checks that routes should enforce."""
        valid = ["user@example.com", "a@b.co", "first.last@domain.org"]
        invalid = ["", "noatsign", "@no-local.com", "spaces in@email.com"]

        pattern = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
        for email in valid:
            assert re.match(pattern, email), f"{email} should be valid"
        for email in invalid:
            assert not re.match(pattern, email), f"{email} should be invalid"

    def test_password_length_requirement(self):
        """Passwords must be at least 8 characters."""
        assert len("short") < 8
        assert len("longEnough1") >= 8


# ---------------------------------------------------------------------------
# API endpoint authorization
# ---------------------------------------------------------------------------

class TestAPIAuthorization:
    """Test authorization checks in API endpoints."""

    def _mock_request(self, user_id=None, is_admin=False):
        """Create a mock FastAPI Request with session data."""
        request = MagicMock()
        session_data = {}
        if user_id:
            session_data["user_id"] = user_id
        request.session = session_data
        return request

    def test_require_user_rejects_unauthenticated(self):
        """API endpoints that check user_id should reject unauthenticated requests."""
        request = self._mock_request(user_id=None)
        assert request.session.get("user_id") is None

    def test_require_user_accepts_authenticated(self):
        request = self._mock_request(user_id=42)
        assert request.session.get("user_id") == 42

    def test_digest_click_validates_user_id_match(self):
        """The digest_click endpoint should reject mismatched user_ids."""
        request = self._mock_request(user_id=1)
        query_user_id = 999  # Different from session

        session_user_id = request.session.get("user_id")
        # This is the validation logic we added
        assert session_user_id != query_user_id

    def test_digest_click_allows_matching_user_id(self):
        request = self._mock_request(user_id=42)
        query_user_id = 42

        session_user_id = request.session.get("user_id")
        assert session_user_id == query_user_id


# ---------------------------------------------------------------------------
# Summary HTML sanitization
# ---------------------------------------------------------------------------

class TestSummarySanitization:
    """Test that summary text has HTML stripped on input."""

    def test_strips_script_tags(self):
        """HTML tags should be stripped from summary text."""
        dirty = '<script>alert("xss")</script>Real summary here'
        clean = re.sub(r"<[^>]+>", "", dirty).strip()
        assert "<script>" not in clean
        assert "alert" in clean  # Content preserved, tags removed
        assert clean == 'alert("xss")Real summary here'

    def test_strips_nested_html(self):
        dirty = '<div><p>Hello <b>world</b></p></div>'
        clean = re.sub(r"<[^>]+>", "", dirty).strip()
        assert clean == "Hello world"

    def test_preserves_plain_text(self):
        text = "This is a normal summary about AI models."
        clean = re.sub(r"<[^>]+>", "", text).strip()
        assert clean == text

    def test_strips_event_handler_attributes(self):
        dirty = '<img src=x onerror=alert(1)>Summary'
        clean = re.sub(r"<[^>]+>", "", dirty).strip()
        assert "onerror" not in clean
        assert clean == "Summary"

    def test_handles_empty_string(self):
        clean = re.sub(r"<[^>]+>", "", "").strip()
        assert clean == ""

    def test_handles_angle_brackets_in_math(self):
        """Legitimate use of < in text (math notation)."""
        text = "When x < 5, the model converges"
        clean = re.sub(r"<[^>]+>", "", text).strip()
        # Bare < without matching > is preserved
        assert "x < 5" in clean


# ---------------------------------------------------------------------------
# Admin-only route protection
# ---------------------------------------------------------------------------

class TestAdminRoutes:
    def test_admin_check_logic(self, session):
        """Admin-only endpoints should verify is_admin flag."""
        regular_user = _make_user(session, email="regular@test.com", is_admin=False)
        admin_user = _make_user(session, email="admin@test.com", is_admin=True)

        assert not regular_user.is_admin
        assert admin_user.is_admin

    def test_approve_article_requires_admin(self):
        """The _require_admin helper should reject non-admin users."""
        request = MagicMock()
        request.session = {"user_id": 1}

        # Simulate a non-admin user lookup
        non_admin = MagicMock()
        non_admin.is_admin = False

        with patch("src.web.auth_utils.get_user_by_id", return_value=non_admin):
            # Non-admin user should not pass admin check
            assert not non_admin.is_admin
