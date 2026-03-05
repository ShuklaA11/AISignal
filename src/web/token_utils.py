"""Token generation and validation for email verification and password reset."""

import hashlib
import secrets
from datetime import timedelta
from typing import Optional

from src.storage.database import session_scope
from src.storage.models import utcnow
from src.storage.queries import (
    create_token, get_token_by_hash, invalidate_user_tokens, mark_token_used,
)

EMAIL_VERIFICATION_EXPIRY = timedelta(hours=24)
PASSWORD_RESET_EXPIRY = timedelta(hours=1)


def generate_token() -> str:
    """Generate a cryptographically secure URL-safe token (32 bytes).

    Returns:
        A URL-safe base64-encoded random string.
    """
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Compute the SHA-256 hex digest of a token for secure storage.

    Args:
        token: The raw token string.

    Returns:
        The hex-encoded SHA-256 hash of the token.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def create_verification_token(user_id: int) -> str:
    """Create an email verification token for a user.

    Invalidates any existing verification tokens for the user, generates a
    new token, and stores its SHA-256 hash in the database. The raw
    (unhashed) token is returned for inclusion in the verification URL.

    Args:
        user_id: The ID of the user to create a token for.

    Returns:
        The raw URL-safe token string (valid for 24 hours).
    """
    with session_scope() as session:
        invalidate_user_tokens(session, user_id, "email_verification")
        raw = generate_token()
        create_token(session, user_id, "email_verification", hash_token(raw),
                     utcnow() + EMAIL_VERIFICATION_EXPIRY)
        return raw


def create_reset_token(user_id: int) -> str:
    """Create a password reset token for a user.

    Invalidates any existing reset tokens for the user, generates a new
    token, and stores its SHA-256 hash in the database. The raw (unhashed)
    token is returned for inclusion in the reset URL.

    Args:
        user_id: The ID of the user to create a token for.

    Returns:
        The raw URL-safe token string (valid for 1 hour).
    """
    with session_scope() as session:
        invalidate_user_tokens(session, user_id, "password_reset")
        raw = generate_token()
        create_token(session, user_id, "password_reset", hash_token(raw),
                     utcnow() + PASSWORD_RESET_EXPIRY)
        return raw


def verify_token(raw_token: str, token_type: str) -> Optional[int]:
    """Return user_id if token is valid, None otherwise. Does not consume."""
    with session_scope() as session:
        token = get_token_by_hash(session, hash_token(raw_token), token_type)
        return token.user_id if token else None


def consume_token(raw_token: str, token_type: str) -> Optional[int]:
    """Verify and mark token as used. Returns user_id or None."""
    with session_scope() as session:
        token = get_token_by_hash(session, hash_token(raw_token), token_type)
        if token:
            user_id = token.user_id
            mark_token_used(session, token)
            return user_id
        return None
