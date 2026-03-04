"""Tests for the in-memory rate limiter."""

import time
from unittest.mock import patch

import pytest

from src.web.rate_limit import RateLimiter


# ---------------------------------------------------------------------------
# is_rate_limited
# ---------------------------------------------------------------------------

def test_is_rate_limited_false_under_limit():
    """Returns False when attempts are below max_attempts."""
    rl = RateLimiter(max_attempts=3, window_seconds=60)
    rl.record_attempt("192.168.1.1")
    rl.record_attempt("192.168.1.1")
    assert rl.is_rate_limited("192.168.1.1") is False


def test_is_rate_limited_true_at_limit():
    """Returns True once max_attempts is reached."""
    rl = RateLimiter(max_attempts=3, window_seconds=60)
    for _ in range(3):
        rl.record_attempt("192.168.1.1")
    assert rl.is_rate_limited("192.168.1.1") is True


def test_is_rate_limited_false_for_unknown_key():
    """A key with no recorded attempts is never limited."""
    rl = RateLimiter(max_attempts=3, window_seconds=60)
    assert rl.is_rate_limited("unknown") is False


# ---------------------------------------------------------------------------
# record_attempt
# ---------------------------------------------------------------------------

def test_record_attempt_increments_count():
    """Each record_attempt call adds one entry."""
    rl = RateLimiter(max_attempts=5, window_seconds=60)
    rl.record_attempt("k")
    assert len(rl._attempts["k"]) == 1
    rl.record_attempt("k")
    assert len(rl._attempts["k"]) == 2


def test_record_attempt_different_keys_are_independent():
    """Attempts for separate keys don't interfere."""
    rl = RateLimiter(max_attempts=2, window_seconds=60)
    rl.record_attempt("a")
    rl.record_attempt("a")
    rl.record_attempt("b")
    assert rl.is_rate_limited("a") is True
    assert rl.is_rate_limited("b") is False


# ---------------------------------------------------------------------------
# remaining_seconds
# ---------------------------------------------------------------------------

def test_remaining_seconds_zero_when_no_attempts():
    """No attempts -> 0 remaining seconds."""
    rl = RateLimiter(max_attempts=5, window_seconds=60)
    assert rl.remaining_seconds("nobody") == 0


def test_remaining_seconds_returns_correct_wait():
    """Should return roughly window_seconds right after the first attempt."""
    rl = RateLimiter(max_attempts=5, window_seconds=100)
    rl.record_attempt("x")

    remaining = rl.remaining_seconds("x")
    # Should be close to 100 (window) since the attempt just happened
    assert 98 <= remaining <= 100


# ---------------------------------------------------------------------------
# Window expiry: old attempts don't count
# ---------------------------------------------------------------------------

def test_old_attempts_expire_from_window():
    """Attempts older than the window are ignored."""
    rl = RateLimiter(max_attempts=2, window_seconds=10)

    # Inject an old timestamp (20 seconds ago)
    old_time = time.monotonic() - 20
    rl._attempts["ip"] = [old_time, old_time]

    # Those old attempts shouldn't count
    assert rl.is_rate_limited("ip") is False


def test_remaining_seconds_after_expiry():
    """After all attempts expire, remaining_seconds returns 0."""
    rl = RateLimiter(max_attempts=2, window_seconds=5)

    old_time = time.monotonic() - 10
    rl._attempts["ip"] = [old_time]

    assert rl.remaining_seconds("ip") == 0


# ---------------------------------------------------------------------------
# _cleanup_all
# ---------------------------------------------------------------------------

def test_cleanup_all_removes_expired_entries():
    """_cleanup_all removes keys whose timestamps are all expired."""
    rl = RateLimiter(max_attempts=5, window_seconds=10)

    now = time.monotonic()
    # Expired key
    rl._attempts["expired"] = [now - 20, now - 15]
    # Active key
    rl._attempts["active"] = [now - 1]

    rl._cleanup_all()

    assert "expired" not in rl._attempts
    assert "active" in rl._attempts


def test_cleanup_all_keeps_partially_valid_entries():
    """If a key has at least one non-expired attempt, keep it."""
    rl = RateLimiter(max_attempts=5, window_seconds=10)

    now = time.monotonic()
    rl._attempts["mixed"] = [now - 20, now - 1]  # one expired, one fresh

    rl._cleanup_all()

    # Key retained because one attempt is still within the window
    assert "mixed" in rl._attempts


def test_record_attempt_triggers_cleanup_at_max_keys():
    """When _MAX_KEYS is exceeded, _cleanup_all is called."""
    rl = RateLimiter(max_attempts=5, window_seconds=10)

    old_time = time.monotonic() - 20
    # Fill with expired keys just under the threshold
    with patch("src.web.rate_limit._MAX_KEYS", 3):
        rl._attempts["a"] = [old_time]
        rl._attempts["b"] = [old_time]
        rl._attempts["c"] = [old_time]

        # This 4th key triggers cleanup
        rl.record_attempt("d")

    # Expired keys should have been cleaned up
    assert "a" not in rl._attempts
    assert "b" not in rl._attempts
    assert "c" not in rl._attempts
    assert "d" in rl._attempts
