"""Tests for the EmailSender: console delivery, digest rendering, and helper methods."""

import logging
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.email_delivery.sender import EmailSender


# ---------------------------------------------------------------------------
# Helpers: build a mock Settings and fake domain objects
# ---------------------------------------------------------------------------

def _make_settings(provider: str = "console") -> MagicMock:
    email_settings = MagicMock()
    email_settings.provider = provider
    email_settings.from_address = "AI Digest <digest@test.com>"
    email_settings.smtp_host = "localhost"
    email_settings.smtp_port = 587
    email_settings.smtp_username = "user"
    email_settings.smtp_password = "pass"

    settings = MagicMock()
    settings.email = email_settings
    settings.base_url = "https://example.com"
    settings.secret_key = "test-secret-key"
    settings.resend_api_key = "re_test_key"
    return settings


def _make_user(user_id: int = 1, email: str = "alice@test.com", name: str = "Alice") -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.email = email
    user.name = name
    return user


def _make_digest(digest_id: int = 10, digest_date: date | None = None) -> MagicMock:
    d = MagicMock()
    d.id = digest_id
    d.digest_date = digest_date or date(2025, 3, 15)
    return d


def _make_article_dict(article_id: int = 1, title: str = "Test Article") -> dict:
    return {
        "id": article_id,
        "title": title,
        "source": "rss",
        "url": f"https://example.com/article/{article_id}",
        "summary": "A test summary of the article.",
        "importance_score": 0.85,
        "category": "llm",
    }


# ---------------------------------------------------------------------------
# _send_console
# ---------------------------------------------------------------------------

def test_send_console_returns_true(caplog):
    """Console provider always returns True."""
    sender = EmailSender(_make_settings("console"))
    result = sender._send_console("user@test.com", "Subject", "<html><body>Hello</body></html>")
    assert result is True


def test_send_console_logs_email(caplog):
    """Console provider logs recipient and subject."""
    sender = EmailSender(_make_settings("console"))
    with caplog.at_level(logging.INFO):
        sender._send_console("user@test.com", "My Subject", '<a href="http://link.com">Link</a>')

    combined = " ".join(caplog.text.split())
    assert "CONSOLE EMAIL" in combined
    assert "My Subject" in combined


def test_send_console_extracts_urls(caplog):
    """Console provider extracts and logs href URLs from the HTML body."""
    sender = EmailSender(_make_settings("console"))
    html = '<a href="http://one.com">1</a> <a href="http://two.com">2</a>'
    with caplog.at_level(logging.INFO):
        sender._send_console("u@t.com", "S", html)

    assert "http://one.com" in caplog.text
    assert "http://two.com" in caplog.text


# ---------------------------------------------------------------------------
# render_digest
# ---------------------------------------------------------------------------

def test_render_digest_contains_article_title():
    """Rendered HTML includes the article title."""
    sender = EmailSender(_make_settings())
    user = _make_user()
    digest = _make_digest()
    articles = [_make_article_dict(title="Transformers Are All You Need")]

    html = sender.render_digest(digest, articles, user)

    assert "Transformers Are All You Need" in html


def test_render_digest_contains_click_url():
    """Rendered HTML includes signed click tracking URLs."""
    sender = EmailSender(_make_settings())
    user = _make_user()
    digest = _make_digest()
    articles = [_make_article_dict()]

    html = sender.render_digest(digest, articles, user)

    assert "/api/digest/click?t=" in html


def test_render_digest_contains_unsubscribe_link():
    """Rendered HTML includes an unsubscribe link."""
    sender = EmailSender(_make_settings())
    user = _make_user()
    digest = _make_digest()
    articles = [_make_article_dict()]

    html = sender.render_digest(digest, articles, user)

    assert "/unsubscribe?t=" in html


def test_render_digest_with_research_and_explore():
    """research_articles and explore_articles are passed through to the template."""
    sender = EmailSender(_make_settings())
    user = _make_user()
    digest = _make_digest()
    main = [_make_article_dict(article_id=1, title="Main Article")]
    research = [_make_article_dict(article_id=2, title="Research Paper")]
    explore = [_make_article_dict(article_id=3, title="Explore Item")]

    html = sender.render_digest(digest, main, user, research_articles=research, explore_articles=explore)

    assert "Main Article" in html
    assert "Research Paper" in html
    assert "Explore Item" in html


def _sectioned_article(article_id: int, title: str, section: str | None) -> dict:
    return {
        **_make_article_dict(article_id=article_id, title=title),
        "section": section,
        "source_name": "test_source",
        "topics": [],
    }


def test_render_digest_groups_articles_into_section_blocks():
    """Articles tagged with sections render under their respective section labels."""
    sender = EmailSender(_make_settings())
    user = _make_user()
    digest = _make_digest()
    articles = [
        _sectioned_article(1, "Builder Article", "builder"),
        _sectioned_article(2, "Industry Article", "industry"),
        _sectioned_article(3, "Research Article", "research"),
    ]

    html = sender.render_digest(digest, articles, user)

    # All titles render
    assert "Builder Article" in html
    assert "Industry Article" in html
    assert "Research Article" in html
    # Section labels are present (case-insensitive contains check on Title casing)
    assert "Research" in html
    assert "Builder" in html
    assert "Industry" in html


def test_render_digest_un_sectioned_articles_render_in_other_bucket():
    """Articles without a section value still appear in the rendered HTML."""
    sender = EmailSender(_make_settings())
    user = _make_user()
    digest = _make_digest()
    articles = [_sectioned_article(1, "Untagged Article", None)]

    html = sender.render_digest(digest, articles, user)

    assert "Untagged Article" in html
    assert "More" in html  # the "other" bucket label


def test_render_digest_section_order_follows_taxonomy():
    """Section blocks render in the canonical taxonomy order."""
    sender = EmailSender(_make_settings())
    user = _make_user()
    digest = _make_digest()
    # Provide sections out of taxonomy order to verify reordering
    articles = [
        _sectioned_article(1, "Industry First", "industry"),
        _sectioned_article(2, "Research First", "research"),
    ]

    html = sender.render_digest(digest, articles, user)

    research_idx = html.index("Research First")
    industry_idx = html.index("Industry First")
    # Research section comes before Industry in ALL_SECTIONS order
    assert research_idx < industry_idx


def test_render_digest_merges_articles_and_research_articles_by_section():
    """Articles passed via the legacy research_articles param are also section-grouped."""
    sender = EmailSender(_make_settings())
    user = _make_user()
    digest = _make_digest()
    main = [_sectioned_article(1, "Main Item", "industry")]
    research = [_sectioned_article(2, "Paper", "research")]

    html = sender.render_digest(digest, main, user, research_articles=research)

    research_idx = html.index("Paper")
    industry_idx = html.index("Main Item")
    assert research_idx < industry_idx


# ---------------------------------------------------------------------------
# send() with console provider
# ---------------------------------------------------------------------------

def test_send_with_console_provider_succeeds():
    """send() with console provider always returns True."""
    sender = EmailSender(_make_settings("console"))
    result = sender.send("user@test.com", "Hi", "<p>Hello</p>")
    assert result is True


# ---------------------------------------------------------------------------
# send_verification_email
# ---------------------------------------------------------------------------

def test_send_verification_email_calls_send_with_correct_subject():
    """send_verification_email renders the template and calls send with the right subject."""
    sender = EmailSender(_make_settings("console"))
    user = _make_user()

    with patch.object(sender, "send", return_value=True) as mock_send:
        result = sender.send_verification_email(user, "https://example.com/verify?token=abc")

    assert result is True
    mock_send.assert_called_once()
    args = mock_send.call_args
    assert args[0][0] == user.email  # to_email
    assert "Verify your email" in args[0][1]  # subject
    assert isinstance(args[0][2], str)  # html_body


# ---------------------------------------------------------------------------
# send_password_reset_email
# ---------------------------------------------------------------------------

def test_send_password_reset_email_calls_send_with_correct_subject():
    """send_password_reset_email renders the template and calls send with the right subject."""
    sender = EmailSender(_make_settings("console"))
    user = _make_user()

    with patch.object(sender, "send", return_value=True) as mock_send:
        result = sender.send_password_reset_email(user, "https://example.com/reset?token=xyz")

    assert result is True
    mock_send.assert_called_once()
    args = mock_send.call_args
    assert args[0][0] == user.email
    assert "Reset your password" in args[0][1]


# ---------------------------------------------------------------------------
# send() retry logic with failing providers
# ---------------------------------------------------------------------------

def test_send_retries_on_failure_then_succeeds():
    """send() retries up to MAX_SEND_RETRIES; succeeds when a retry works."""
    sender = EmailSender(_make_settings("resend"))

    with patch.object(sender, "_send_resend", side_effect=[False, True]) as mock_resend, \
         patch("src.email_delivery.sender.time.sleep"):
        result = sender.send("u@t.com", "Sub", "<p>Hi</p>")

    assert result is True
    assert mock_resend.call_count == 2


def test_send_returns_false_after_all_retries_exhausted():
    """send() returns False after all retries fail."""
    sender = EmailSender(_make_settings("resend"))

    with patch.object(sender, "_send_resend", return_value=False) as mock_resend, \
         patch("src.email_delivery.sender.time.sleep"):
        result = sender.send("u@t.com", "Sub", "<p>Hi</p>")

    assert result is False
    assert mock_resend.call_count == 3  # MAX_SEND_RETRIES
