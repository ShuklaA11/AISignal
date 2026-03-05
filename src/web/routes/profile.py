import json
import logging
import re

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.config import load_settings
from src.storage.database import session_scope
from src.storage.queries import (
    get_read_articles_for_user, get_saved_article_ids, get_saved_articles_for_user,
    get_user_by_email,
)
from src.web.auth_utils import hash_password, require_login, verify_password
from src.web.rate_limit import RateLimiter
from src.web.template_engine import templates

logger = logging.getLogger(__name__)

router = APIRouter()

profile_sensitive_limiter = RateLimiter(max_attempts=5, window_seconds=300)

VALID_ROLES = {"student", "industry", "enthusiast"}
VALID_LEVELS = {"beginner", "intermediate", "advanced"}


@router.get("/profile", response_class=HTMLResponse,
            summary="User profile",
            description="View and manage user preferences: role, level, topics, source weights, saved articles, and reading history.")
async def profile_page(request: Request, auth: tuple = Depends(require_login)):
    user, session = auth
    try:
        settings = load_settings()
        all_topics = []
        for group_topics in settings.topics.values():
            all_topics.extend(group_topics)

        saved_articles = get_saved_articles_for_user(session, user.id)
        saved_ids = get_saved_article_ids(session, user.id)
        read_articles = get_read_articles_for_user(session, user.id)

        return templates.TemplateResponse(
            "profile.html",
            {
                "request": request,
                "user": user,
                "user_topics": user.topics,
                "all_topics": all_topics,
                "topic_groups": settings.topics,
                "source_preferences": user.source_preferences,
                "saved_articles": saved_articles,
                "saved_ids": saved_ids,
                "read_articles": read_articles,
            },
        )
    finally:
        session.close()


@router.post("/profile",
             summary="Update profile",
             description="Update user preferences (role, level, name, topics, and source weights).")
async def profile_update(request: Request, auth: tuple = Depends(require_login)):
    user, session = auth
    try:
        form = await request.form()

        if "role" in form and form["role"] in VALID_ROLES:
            user.role = form["role"]
        if "level" in form and form["level"] in VALID_LEVELS:
            user.level = form["level"]
        if "name" in form:
            user.name = str(form["name"]).strip()[:100]

        topics = form.getlist("topics")
        if topics:
            settings = load_settings()
            valid_topics = {t for group in settings.topics.values() for t in group}
            topics = [t for t in topics if t in valid_topics]
            user.topics_json = json.dumps(topics)

        # Source weights
        weights = {}
        for key, val in form.items():
            if key.startswith("weight_"):
                source_name = key.replace("weight_", "")
                try:
                    weight = int(val)
                except (ValueError, TypeError):
                    weight = 5
                weights[source_name] = max(0, min(10, weight))
        if weights:
            user.source_preferences_json = json.dumps(weights)

        session.add(user)
        session.commit()

        return RedirectResponse(url="/profile", status_code=302)
    finally:
        session.close()


# ── Password change ───────────────────────────────────────────────────

@router.post("/profile/password",
             summary="Change password",
             description="Change the logged-in user's password. Requires current password for verification.")
async def change_password(
    request: Request,
    auth: tuple = Depends(require_login),
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
):
    user, session = auth
    try:
        client_ip = request.client.host if request.client else "unknown"
        if profile_sensitive_limiter.is_rate_limited(client_ip):
            wait = profile_sensitive_limiter.remaining_seconds(client_ip)
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error=f"Too many attempts. Try again in {wait} seconds."),
                status_code=429,
            )
        profile_sensitive_limiter.record_attempt(client_ip)

        if not verify_password(current_password, user.password_hash):
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error="Current password is incorrect."),
            )

        if new_password != confirm_password:
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error="New passwords do not match."),
            )

        if len(new_password) < 8:
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error="Password must be at least 8 characters."),
            )
        if not re.search(r"[A-Za-z]", new_password) or not re.search(r"[0-9]", new_password):
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error="Password must contain at least one letter and one digit."),
            )

        user.password_hash = hash_password(new_password)
        user.session_version += 1
        session.add(user)
        session.commit()

        # Update current session to match new version
        request.session["session_version"] = user.session_version

        return templates.TemplateResponse(
            "profile.html",
            _profile_context(request, user, session, success="Password changed successfully."),
        )
    finally:
        session.close()


# ── Email change ──────────────────────────────────────────────────────

@router.post("/profile/email",
             summary="Change email",
             description="Change the logged-in user's email. Requires password for verification. Sends verification to new email.")
async def change_email(
    request: Request,
    auth: tuple = Depends(require_login),
    new_email: str = Form(...),
    password: str = Form(...),
):
    user, session = auth
    try:
        client_ip = request.client.host if request.client else "unknown"
        if profile_sensitive_limiter.is_rate_limited(client_ip):
            wait = profile_sensitive_limiter.remaining_seconds(client_ip)
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error=f"Too many attempts. Try again in {wait} seconds."),
                status_code=429,
            )
        profile_sensitive_limiter.record_attempt(client_ip)

        if not verify_password(password, user.password_hash):
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error="Password is incorrect."),
            )

        new_email = new_email.strip().lower()
        if not re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', new_email):
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error="Please enter a valid email address."),
            )

        if new_email == user.email:
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error="That is already your email address."),
            )

        existing = get_user_by_email(session, new_email)
        if existing:
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error="That email is already in use."),
            )

        user.email = new_email
        user.email_verified = False
        user.email_verified_at = None
        session.add(user)
        session.commit()

        # Send verification to new email
        try:
            from src.email_delivery.sender import EmailSender
            from src.web.token_utils import create_verification_token
            settings = load_settings()
            token = create_verification_token(user.id)
            base_url = settings.base_url.rstrip("/") if settings.base_url else str(request.base_url).rstrip("/")
            sender = EmailSender(settings)
            sender.send_verification_email(user, f"{base_url}/verify-email?token={token}")
        except Exception:
            logger.warning("Failed to send verification email after email change")

        return templates.TemplateResponse(
            "profile.html",
            _profile_context(request, user, session, success="Email updated. Please check your new email for a verification link."),
        )
    finally:
        session.close()


# ── Account deletion ──────────────────────────────────────────────────

@router.post("/profile/delete",
             summary="Delete account",
             description="Permanently delete the user's account and all associated data.")
async def delete_account(
    request: Request,
    auth: tuple = Depends(require_login),
    password: str = Form(...),
):
    user, session = auth
    try:
        client_ip = request.client.host if request.client else "unknown"
        if profile_sensitive_limiter.is_rate_limited(client_ip):
            wait = profile_sensitive_limiter.remaining_seconds(client_ip)
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error=f"Too many attempts. Try again in {wait} seconds."),
                status_code=429,
            )
        profile_sensitive_limiter.record_attempt(client_ip)

        if not verify_password(password, user.password_hash):
            return templates.TemplateResponse(
                "profile.html",
                _profile_context(request, user, session, error="Password is incorrect. Account not deleted."),
            )

        user_id = user.id
    finally:
        session.close()

    # Perform all deletes atomically in a single transaction
    from sqlmodel import delete as sa_delete, select
    from src.storage.models import (
        DigestArticle, DigestClick, Digest, FeedImpression,
        ReadArticle, SavedArticle, Token, User, UserMLProfile, UserEmbeddingModel,
        ScoringMetric,
    )

    with session_scope() as session:
        user = session.get(User, user_id)
        if not user:
            request.session.clear()
            return RedirectResponse(url="/", status_code=302)

        # Delete digest articles via digests
        digest_ids = [d.id for d in user.digests]
        if digest_ids:
            session.exec(sa_delete(DigestArticle).where(DigestArticle.digest_id.in_(digest_ids)))
            session.exec(sa_delete(DigestClick).where(DigestClick.digest_id.in_(digest_ids)))
        session.flush()

        session.exec(sa_delete(Digest).where(Digest.user_id == user_id))
        session.exec(sa_delete(FeedImpression).where(FeedImpression.user_id == user_id))
        session.exec(sa_delete(ReadArticle).where(ReadArticle.user_id == user_id))
        session.exec(sa_delete(SavedArticle).where(SavedArticle.user_id == user_id))
        session.exec(sa_delete(Token).where(Token.user_id == user_id))
        session.exec(sa_delete(ScoringMetric).where(ScoringMetric.user_id == user_id))
        session.exec(sa_delete(DigestClick).where(DigestClick.user_id == user_id))
        session.flush()

        # Delete ML profile and embedding model
        if user.ml_profile:
            session.delete(user.ml_profile)
        emb_model = session.exec(
            select(UserEmbeddingModel).where(UserEmbeddingModel.user_id == user_id)
        ).first()
        if emb_model:
            session.delete(emb_model)
        session.flush()

        session.delete(user)
        # session_scope auto-commits on exit

    request.session.clear()
    logger.info(f"User {user_id} deleted their account")
    return RedirectResponse(url="/", status_code=302)


# ── Helpers ───────────────────────────────────────────────────────────

def _profile_context(request, user, session, error=None, success=None):
    """Build template context dict for profile page with optional messages."""
    settings = load_settings()
    all_topics = []
    for group_topics in settings.topics.values():
        all_topics.extend(group_topics)

    ctx = {
        "request": request,
        "user": user,
        "user_topics": user.topics,
        "all_topics": all_topics,
        "topic_groups": settings.topics,
        "source_preferences": user.source_preferences,
        "saved_articles": get_saved_articles_for_user(session, user.id),
        "saved_ids": get_saved_article_ids(session, user.id),
        "read_articles": get_read_articles_for_user(session, user.id),
    }
    if error:
        ctx["error"] = error
    if success:
        ctx["success"] = success
    return ctx
