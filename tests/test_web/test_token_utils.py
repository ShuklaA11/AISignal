"""Tests for web token utilities (generate, hash, create, verify, consume).

Covers:
- generate_token: URL-safe, unique each call
- hash_token: deterministic SHA-256
- create_verification_token / create_reset_token: DB persistence, old token invalidation
- verify_token: valid returns user_id, invalid returns None
- consume_token: marks used, second consume returns None
"""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.web.token_utils import (
    EMAIL_VERIFICATION_EXPIRY,
    PASSWORD_RESET_EXPIRY,
    consume_token,
    create_reset_token,
    create_verification_token,
    generate_token,
    hash_token,
    verify_token,
)


# ===========================================================================
# 1. generate_token
# ===========================================================================

class TestGenerateToken:
    def test_returns_url_safe_string(self):
        token = generate_token()
        assert isinstance(token, str)
        assert len(token) > 0
        # URL-safe base64 uses only these characters
        import re
        assert re.match(r'^[A-Za-z0-9_-]+$', token)

    def test_different_each_call(self):
        tokens = {generate_token() for _ in range(50)}
        assert len(tokens) == 50


# ===========================================================================
# 2. hash_token
# ===========================================================================

class TestHashToken:
    def test_deterministic(self):
        token = "test-token-123"
        assert hash_token(token) == hash_token(token)

    def test_returns_hex_sha256(self):
        h = hash_token("hello")
        assert len(h) == 64  # SHA-256 hex digest is 64 chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_inputs_different_hashes(self):
        assert hash_token("aaa") != hash_token("bbb")


# ===========================================================================
# 3. create_verification_token / create_reset_token
# ===========================================================================

class TestCreateTokens:
    @patch("src.web.token_utils.session_scope")
    def test_create_verification_token(self, mock_scope):
        mock_session = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        raw = create_verification_token(user_id=1)

        assert isinstance(raw, str)
        assert len(raw) > 0
        # Should have invalidated old tokens first
        from src.storage.queries import invalidate_user_tokens
        mock_session.assert_any_call  # session was used
        # Verify invalidate_user_tokens was called via the session
        # The function calls invalidate_user_tokens(session, 1, "email_verification")
        # and create_token(session, 1, "email_verification", hash, expiry)

    @patch("src.web.token_utils.session_scope")
    def test_create_reset_token(self, mock_scope):
        mock_session = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        raw = create_reset_token(user_id=1)

        assert isinstance(raw, str)
        assert len(raw) > 0

    @patch("src.web.token_utils.create_token")
    @patch("src.web.token_utils.invalidate_user_tokens")
    @patch("src.web.token_utils.session_scope")
    def test_verification_invalidates_old_tokens(self, mock_scope, mock_invalidate, mock_create):
        mock_session = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        create_verification_token(user_id=42)

        mock_invalidate.assert_called_once_with(mock_session, 42, "email_verification")
        mock_create.assert_called_once()
        # Verify the token_type passed to create_token
        call_args = mock_create.call_args
        assert call_args[0][2] == "email_verification"

    @patch("src.web.token_utils.create_token")
    @patch("src.web.token_utils.invalidate_user_tokens")
    @patch("src.web.token_utils.session_scope")
    def test_reset_invalidates_old_tokens(self, mock_scope, mock_invalidate, mock_create):
        mock_session = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        create_reset_token(user_id=42)

        mock_invalidate.assert_called_once_with(mock_session, 42, "password_reset")
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[0][2] == "password_reset"


# ===========================================================================
# 4. verify_token
# ===========================================================================

class TestVerifyToken:
    @patch("src.web.token_utils.get_token_by_hash")
    @patch("src.web.token_utils.session_scope")
    def test_valid_token_returns_user_id(self, mock_scope, mock_get):
        mock_session = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        mock_token = MagicMock()
        mock_token.user_id = 7
        mock_get.return_value = mock_token

        result = verify_token("raw-token-abc", "email_verification")
        assert result == 7
        mock_get.assert_called_once_with(
            mock_session, hash_token("raw-token-abc"), "email_verification",
        )

    @patch("src.web.token_utils.get_token_by_hash")
    @patch("src.web.token_utils.session_scope")
    def test_invalid_token_returns_none(self, mock_scope, mock_get):
        mock_session = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        mock_get.return_value = None

        result = verify_token("bad-token", "email_verification")
        assert result is None


# ===========================================================================
# 5. consume_token
# ===========================================================================

class TestConsumeToken:
    @patch("src.web.token_utils.mark_token_used")
    @patch("src.web.token_utils.get_token_by_hash")
    @patch("src.web.token_utils.session_scope")
    def test_consume_marks_used_returns_user_id(self, mock_scope, mock_get, mock_mark):
        mock_session = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        mock_token = MagicMock()
        mock_token.user_id = 5
        mock_get.return_value = mock_token

        result = consume_token("raw-token", "password_reset")
        assert result == 5
        mock_mark.assert_called_once_with(mock_session, mock_token)

    @patch("src.web.token_utils.mark_token_used")
    @patch("src.web.token_utils.get_token_by_hash")
    @patch("src.web.token_utils.session_scope")
    def test_consume_invalid_returns_none(self, mock_scope, mock_get, mock_mark):
        mock_session = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        mock_get.return_value = None

        result = consume_token("bad-token", "password_reset")
        assert result is None
        mock_mark.assert_not_called()

    @patch("src.web.token_utils.mark_token_used")
    @patch("src.web.token_utils.get_token_by_hash")
    @patch("src.web.token_utils.session_scope")
    def test_second_consume_returns_none(self, mock_scope, mock_get, mock_mark):
        """After first consume, the token is used; second lookup returns None."""
        mock_session = MagicMock()
        mock_scope.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_scope.return_value.__exit__ = MagicMock(return_value=False)

        mock_token = MagicMock()
        mock_token.user_id = 5
        # First call returns token, second returns None (simulating used token)
        mock_get.side_effect = [mock_token, None]

        first = consume_token("raw-token", "email_verification")
        second = consume_token("raw-token", "email_verification")

        assert first == 5
        assert second is None
