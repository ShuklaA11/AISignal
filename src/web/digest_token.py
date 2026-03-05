"""Signed tokens for digest email click tracking.

Uses itsdangerous to produce tamper-proof tokens that embed user_id,
article_id, digest_id, and section — so none of these appear as
plain query parameters in email links.
"""

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer


def _get_serializer(secret_key: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret_key, salt="digest-click")


# 30 days in seconds
TOKEN_MAX_AGE = 86400 * 30


def sign_digest_click(
    secret_key: str,
    user_id: int,
    article_id: int,
    digest_id: int,
    section: str = "main",
) -> str:
    """Create a signed token encoding the click parameters."""
    s = _get_serializer(secret_key)
    return s.dumps({
        "u": user_id,
        "a": article_id,
        "d": digest_id,
        "s": section,
    })


def sign_unsubscribe(secret_key: str, user_id: int, email: str) -> str:
    """Create a signed token for the unsubscribe link."""
    s = URLSafeTimedSerializer(secret_key, salt="unsubscribe")
    return s.dumps({"u": user_id, "e": email})


def verify_unsubscribe(secret_key: str, token: str) -> dict | None:
    """Verify an unsubscribe token. Returns dict with user_id and email, or None."""
    s = URLSafeTimedSerializer(secret_key, salt="unsubscribe")
    try:
        data = s.loads(token, max_age=TOKEN_MAX_AGE)
        return {"user_id": data["u"], "email": data["e"]}
    except (BadSignature, SignatureExpired, KeyError):
        return None


def verify_digest_click(secret_key: str, token: str) -> dict | None:
    """Verify and decode a digest click token.

    Returns dict with keys user_id, article_id, digest_id, section
    or None if the token is invalid/tampered.
    """
    s = _get_serializer(secret_key)
    try:
        data = s.loads(token, max_age=TOKEN_MAX_AGE)
        return {
            "user_id": data["u"],
            "article_id": data["a"],
            "digest_id": data["d"],
            "section": data.get("s", "main"),
        }
    except (BadSignature, SignatureExpired, KeyError):
        return None
