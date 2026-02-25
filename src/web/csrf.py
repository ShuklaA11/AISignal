"""CSRF protection middleware using session-bound tokens."""

import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
CSRF_TOKEN_KEY = "_csrf_token"
CSRF_FIELD_NAME = "csrf_token"
CSRF_HEADER_NAME = "x-csrf-token"


def _get_or_create_token(request: Request) -> str:
    """Return the CSRF token from the session, creating one if needed."""
    token = request.session.get(CSRF_TOKEN_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        request.session[CSRF_TOKEN_KEY] = token
    return token


class CSRFMiddleware(BaseHTTPMiddleware):
    """Validates CSRF tokens on state-changing requests (POST, PUT, DELETE).

    Tokens are stored in the session and submitted via form field or X-CSRF-Token header.
    Safe methods (GET, HEAD, OPTIONS, TRACE) are exempt.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method in _SAFE_METHODS:
            return await call_next(request)

        expected = request.session.get(CSRF_TOKEN_KEY)
        if not expected:
            return Response("CSRF token missing from session", status_code=403)

        # Check header first (HTMX requests)
        submitted = request.headers.get(CSRF_HEADER_NAME)
        if not submitted:
            # For form submissions, read the raw body and cache it so
            # downstream handlers (FastAPI Form(...)) can re-read it.
            body = await request.body()  # this caches on request._body
            # Parse form data manually to extract the token without
            # consuming the body stream.
            content_type = request.headers.get("content-type", "")
            if "application/x-www-form-urlencoded" in content_type:
                from urllib.parse import parse_qs
                parsed = parse_qs(body.decode("utf-8"))
                values = parsed.get(CSRF_FIELD_NAME, [])
                submitted = values[0] if values else None
            elif "multipart/form-data" in content_type:
                # Fall back to starlette's form parser for multipart
                form = await request.form()
                submitted = form.get(CSRF_FIELD_NAME)

        if not submitted or not secrets.compare_digest(submitted, expected):
            return Response("CSRF token invalid", status_code=403)

        return await call_next(request)
