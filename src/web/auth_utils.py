"""Authentication utilities and FastAPI dependencies."""

import bcrypt
from fastapi import Request

from src.storage.database import get_session
from src.storage.queries import get_user_by_id


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt with a random salt.

    Args:
        password: The plaintext password to hash.

    Returns:
        The bcrypt hash as a UTF-8 string.
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash.

    Args:
        password: The plaintext password to check.
        hashed: The bcrypt hash to compare against.

    Returns:
        True if the password matches the hash, False otherwise.
    """
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# FastAPI Dependencies (use with Depends())
# ---------------------------------------------------------------------------

def require_login(request: Request) -> tuple:
    """Dependency: returns (user, db_session). Raises _LoginRequired on failure.

    The caller is responsible for closing the session (use try/finally).
    Validates session_version to detect invalidated sessions (e.g. after password change).
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise _LoginRequired()
    session = get_session()
    user = get_user_by_id(session, user_id)
    if not user:
        session.close()
        raise _LoginRequired()
    # Invalidate session if password was changed on another device
    session_ver = request.session.get("session_version", 0)
    if session_ver != getattr(user, "session_version", 0):
        session.close()
        request.session.clear()
        raise _LoginRequired()
    return user, session


def require_admin(request: Request) -> tuple:
    """Dependency: returns (user, db_session). Raises _AdminRequired if not admin.

    The caller is responsible for closing the session (use try/finally).
    """
    user, session = require_login(request)
    if not getattr(user, "is_admin", False):
        session.close()
        raise _AdminRequired()
    return user, session


def require_user_id(request: Request) -> int:
    """Dependency for HTMX API endpoints: returns user_id or raises _LoginRequired."""
    user_id = request.session.get("user_id")
    if not user_id:
        raise _LoginRequired()
    # Validate session_version to detect invalidated sessions
    session = get_session()
    try:
        user = get_user_by_id(session, user_id)
        if not user:
            raise _LoginRequired()
        session_ver = request.session.get("session_version", 0)
        if session_ver != getattr(user, "session_version", 0):
            request.session.clear()
            raise _LoginRequired()
    finally:
        session.close()
    return user_id


class _LoginRequired(Exception):
    """Raised when login is required."""
    pass


class _AdminRequired(Exception):
    """Raised when admin access is required."""
    pass
