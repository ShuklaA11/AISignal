"""Tests for signed digest-click and unsubscribe tokens."""

import pytest

from src.web.digest_token import (
    sign_digest_click,
    sign_unsubscribe,
    verify_digest_click,
    verify_unsubscribe,
)

SECRET = "test-secret-key-for-tokens"


# ---------------------------------------------------------------------------
# sign_digest_click / verify_digest_click round-trip
# ---------------------------------------------------------------------------

def test_digest_click_round_trip():
    """Sign then verify returns the original payload."""
    token = sign_digest_click(SECRET, user_id=1, article_id=42, digest_id=7, section="research")
    result = verify_digest_click(SECRET, token)

    assert result is not None
    assert result["user_id"] == 1
    assert result["article_id"] == 42
    assert result["digest_id"] == 7
    assert result["section"] == "research"


def test_digest_click_default_section():
    """Section defaults to 'main' when not specified."""
    token = sign_digest_click(SECRET, user_id=1, article_id=2, digest_id=3)
    result = verify_digest_click(SECRET, token)

    assert result is not None
    assert result["section"] == "main"


# ---------------------------------------------------------------------------
# sign_unsubscribe / verify_unsubscribe round-trip
# ---------------------------------------------------------------------------

def test_unsubscribe_round_trip():
    """Sign then verify returns user_id and email."""
    token = sign_unsubscribe(SECRET, user_id=5, email="alice@example.com")
    result = verify_unsubscribe(SECRET, token)

    assert result is not None
    assert result["user_id"] == 5
    assert result["email"] == "alice@example.com"


# ---------------------------------------------------------------------------
# Invalid / tampered tokens
# ---------------------------------------------------------------------------

def test_digest_click_tampered_token_returns_none():
    """A tampered token string should return None."""
    token = sign_digest_click(SECRET, user_id=1, article_id=2, digest_id=3)
    tampered = token[:-4] + "XXXX"
    assert verify_digest_click(SECRET, tampered) is None


def test_unsubscribe_tampered_token_returns_none():
    """A tampered unsubscribe token should return None."""
    token = sign_unsubscribe(SECRET, user_id=1, email="test@test.com")
    tampered = token[:-4] + "XXXX"
    assert verify_unsubscribe(SECRET, tampered) is None


def test_completely_garbage_token_returns_none():
    """Totally invalid strings return None."""
    assert verify_digest_click(SECRET, "not-a-valid-token") is None
    assert verify_unsubscribe(SECRET, "garbage") is None


def test_empty_token_returns_none():
    """Empty string tokens return None."""
    assert verify_digest_click(SECRET, "") is None
    assert verify_unsubscribe(SECRET, "") is None


# ---------------------------------------------------------------------------
# Wrong secret returns None
# ---------------------------------------------------------------------------

def test_digest_click_wrong_secret_returns_none():
    """Verifying with a different secret key returns None."""
    token = sign_digest_click(SECRET, user_id=1, article_id=2, digest_id=3)
    assert verify_digest_click("wrong-secret", token) is None


def test_unsubscribe_wrong_secret_returns_none():
    """Verifying with a different secret key returns None."""
    token = sign_unsubscribe(SECRET, user_id=1, email="a@b.com")
    assert verify_unsubscribe("wrong-secret", token) is None


# ---------------------------------------------------------------------------
# Cross-type tokens are rejected
# ---------------------------------------------------------------------------

def test_digest_click_token_rejected_by_unsubscribe_verify():
    """A digest-click token should fail unsubscribe verification (different salt)."""
    token = sign_digest_click(SECRET, user_id=1, article_id=2, digest_id=3)
    assert verify_unsubscribe(SECRET, token) is None


def test_unsubscribe_token_rejected_by_digest_click_verify():
    """An unsubscribe token should fail digest-click verification (different salt)."""
    token = sign_unsubscribe(SECRET, user_id=1, email="a@b.com")
    assert verify_digest_click(SECRET, token) is None
