"""Simple in-memory rate limiter for auth endpoints."""

import time
from collections import defaultdict

# Maximum distinct keys before triggering a full cleanup sweep
_MAX_KEYS = 10_000


class RateLimiter:
    """Token-bucket style rate limiter keyed by IP address."""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def _cleanup(self, key: str) -> None:
        cutoff = time.monotonic() - self.window_seconds
        self._attempts[key] = [t for t in self._attempts[key] if t > cutoff]
        if not self._attempts[key]:
            del self._attempts[key]

    def _cleanup_all(self) -> None:
        """Remove all expired entries to bound memory usage."""
        cutoff = time.monotonic() - self.window_seconds
        expired_keys = [
            k for k, v in self._attempts.items()
            if not any(t > cutoff for t in v)
        ]
        for k in expired_keys:
            del self._attempts[k]

    def is_rate_limited(self, key: str) -> bool:
        """Check if the key has exceeded the rate limit."""
        self._cleanup(key)
        return len(self._attempts.get(key, [])) >= self.max_attempts

    def record_attempt(self, key: str) -> None:
        """Record an attempt for the given key."""
        self._attempts[key].append(time.monotonic())
        if len(self._attempts) > _MAX_KEYS:
            self._cleanup_all()

    def remaining_seconds(self, key: str) -> int:
        """Seconds until the oldest attempt expires from the window."""
        self._cleanup(key)
        attempts = self._attempts.get(key, [])
        if not attempts:
            return 0
        oldest = min(attempts)
        return max(0, int(self.window_seconds - (time.monotonic() - oldest)))


# Shared instance: 5 login attempts per 5-minute window
login_limiter = RateLimiter(max_attempts=5, window_seconds=300)

# slowapi limiter for global request rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
