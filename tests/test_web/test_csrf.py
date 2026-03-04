"""Unit tests for CSRF middleware — no HTTP server required."""

import secrets
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.web.csrf import (
    CSRF_FIELD_NAME,
    CSRF_HEADER_NAME,
    CSRF_TOKEN_KEY,
    CSRFMiddleware,
    _get_or_create_token,
)


# ---------------------------------------------------------------------------
# _get_or_create_token
# ---------------------------------------------------------------------------

def test_get_or_create_token_creates_on_first_call():
    """First call stores a new token in the session and returns it."""
    session: dict = {}
    request = MagicMock()
    request.session = session

    token = _get_or_create_token(request)

    assert token
    assert isinstance(token, str)
    assert len(token) > 20  # url-safe base64 of 32 bytes
    assert session[CSRF_TOKEN_KEY] == token


def test_get_or_create_token_returns_same_on_second_call():
    """Subsequent calls return the same token without regenerating."""
    session: dict = {}
    request = MagicMock()
    request.session = session

    first = _get_or_create_token(request)
    second = _get_or_create_token(request)

    assert first == second


def test_get_or_create_token_preserves_existing():
    """If a token already exists in the session it is returned as-is."""
    existing = "my-existing-token"
    session = {CSRF_TOKEN_KEY: existing}
    request = MagicMock()
    request.session = session

    assert _get_or_create_token(request) == existing


# ---------------------------------------------------------------------------
# CSRFMiddleware — safe methods bypass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safe_methods_bypass_csrf():
    """GET, HEAD, OPTIONS, TRACE should pass through without checking tokens."""
    middleware = CSRFMiddleware(app=MagicMock())

    for method in ("GET", "HEAD", "OPTIONS", "TRACE"):
        request = MagicMock()
        request.method = method
        request.session = {}

        sentinel = MagicMock(name="downstream_response")
        call_next = AsyncMock(return_value=sentinel)

        response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert response is sentinel


# ---------------------------------------------------------------------------
# CSRFMiddleware — POST without token returns 403
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_without_session_token_returns_403():
    """POST when no token exists in the session at all -> 403."""
    middleware = CSRFMiddleware(app=MagicMock())

    request = MagicMock()
    request.method = "POST"
    request.session = {}  # no CSRF token stored

    call_next = AsyncMock()
    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 403
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_with_wrong_header_token_returns_403():
    """POST with an incorrect CSRF header token -> 403."""
    middleware = CSRFMiddleware(app=MagicMock())
    real_token = secrets.token_urlsafe(32)

    request = MagicMock()
    request.method = "POST"
    request.session = {CSRF_TOKEN_KEY: real_token}
    request.headers = {CSRF_HEADER_NAME: "wrong-token"}

    call_next = AsyncMock()
    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 403
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_with_correct_header_token_passes():
    """POST with matching CSRF header token -> passes to downstream."""
    middleware = CSRFMiddleware(app=MagicMock())
    real_token = secrets.token_urlsafe(32)

    request = MagicMock()
    request.method = "POST"
    request.session = {CSRF_TOKEN_KEY: real_token}
    request.headers = {CSRF_HEADER_NAME: real_token}

    sentinel = MagicMock(name="downstream_response")
    call_next = AsyncMock(return_value=sentinel)

    response = await middleware.dispatch(request, call_next)

    assert response is sentinel
    call_next.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_post_with_correct_form_token_passes():
    """POST with matching CSRF token in form body -> passes to downstream."""
    middleware = CSRFMiddleware(app=MagicMock())
    real_token = secrets.token_urlsafe(32)

    body_bytes = f"{CSRF_FIELD_NAME}={real_token}".encode("utf-8")

    headers_data = {
        CSRF_HEADER_NAME: None,
        "content-type": "application/x-www-form-urlencoded",
    }
    request = MagicMock()
    request.method = "POST"
    request.session = {CSRF_TOKEN_KEY: real_token}
    request.headers = MagicMock()
    request.headers.get = lambda key, default="": headers_data.get(key, default)
    request.body = AsyncMock(return_value=body_bytes)

    sentinel = MagicMock(name="downstream_response")
    call_next = AsyncMock(return_value=sentinel)

    response = await middleware.dispatch(request, call_next)

    assert response is sentinel
