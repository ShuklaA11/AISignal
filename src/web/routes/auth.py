import logging
import re
from urllib.parse import urlparse

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.config import load_settings
from src.email_delivery.sender import EmailSender
from src.storage.database import session_scope
from src.storage.models import User, utcnow
from src.storage.queries import get_user_by_email, get_user_by_id
from src.utils import mask_email as _mask_email
from src.web.auth_utils import hash_password, verify_password
from src.web.rate_limit import RateLimiter, login_limiter
from src.web.template_engine import templates
from src.web.digest_token import verify_unsubscribe
from src.web.token_utils import consume_token, create_reset_token, create_verification_token, verify_token

logger = logging.getLogger(__name__)
router = APIRouter()

signup_limiter = RateLimiter(max_attempts=5, window_seconds=300)
password_reset_limiter = RateLimiter(max_attempts=3, window_seconds=300)
verification_resend_limiter = RateLimiter(max_attempts=3, window_seconds=300)


def _validate_email(email: str) -> str | None:
    """Return an error message if email format is invalid, or None if it's OK."""
    if not re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email):
        return "Please enter a valid email address."
    return None


def _validate_password(password: str) -> str | None:
    """Return an error message if password is too weak, or None if it's OK."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", password):
        return "Password must contain at least one letter."
    if not re.search(r"[0-9]", password):
        return "Password must contain at least one digit."
    return None



def _safe_referer(request: Request, fallback: str = "/feed") -> str:
    """Return the Referer header if it's a same-origin path, otherwise fallback."""
    referer = request.headers.get("referer", fallback)
    parsed = urlparse(referer)
    # Only allow relative paths or same-host URLs
    if parsed.netloc and parsed.netloc != request.url.netloc:
        return fallback
    # Return just the path (strip any external scheme/host)
    return parsed.path or fallback


def _build_base_url(request: Request) -> str:
    settings = load_settings()
    if settings.base_url:
        return settings.base_url.rstrip("/")
    return str(request.base_url).rstrip("/")


@router.get("/signup", response_class=HTMLResponse,
            summary="Signup page",
            description="Render the signup form. Optionally prefills the email field.")
async def signup_page(request: Request, email: str = ""):
    return templates.TemplateResponse("auth/signup.html", {"request": request, "prefill_email": email})


@router.post("/signup",
             summary="Submit signup",
             description="Create a new user account, send a verification email, and redirect to onboarding.")
async def signup_submit(
    request: Request,
    email: str = Form(...),
    name: str = Form(""),
    password: str = Form(...),
):
    client_ip = request.client.host if request.client else "unknown"
    if signup_limiter.is_rate_limited(client_ip):
        wait = signup_limiter.remaining_seconds(client_ip)
        return templates.TemplateResponse(
            "auth/signup.html",
            {"request": request, "error": f"Too many signup attempts. Try again in {wait} seconds.",
             "prefill_email": email},
            status_code=429,
        )
    signup_limiter.record_attempt(client_ip)

    # Validate email format before touching the database
    email_error = _validate_email(email)
    if email_error:
        return templates.TemplateResponse(
            "auth/signup.html",
            {"request": request, "error": email_error, "prefill_email": email},
        )

    with session_scope() as session:
        pw_error = _validate_password(password)
        if pw_error:
            return templates.TemplateResponse(
                "auth/signup.html",
                {"request": request, "error": pw_error, "prefill_email": email},
            )

        existing = get_user_by_email(session, email)
        if existing:
            return templates.TemplateResponse(
                "auth/signup.html",
                {"request": request, "error": "Email already registered"},
            )

        name = name.strip()[:100]

        user = User(
            email=email,
            name=name,
            password_hash=hash_password(password),
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        # Send verification email
        try:
            token = create_verification_token(user.id)
            base_url = _build_base_url(request)
            sender = EmailSender(load_settings())
            sender.send_verification_email(user, f"{base_url}/verify-email?token={token}")
        except Exception as e:
            logger.warning(f"Failed to send verification email to {_mask_email(email)}")

        request.session["user_id"] = user.id
        request.session["session_version"] = 0
        return RedirectResponse(url="/onboarding/role", status_code=302)


@router.get("/login", response_class=HTMLResponse,
            summary="Login page",
            description="Render the login form.")
async def login_page(request: Request):
    return templates.TemplateResponse("auth/login.html", {"request": request})


@router.post("/login",
             summary="Submit login",
             description="Authenticate user with email and password. Rate-limited per IP.")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    client_ip = request.client.host if request.client else "unknown"

    if login_limiter.is_rate_limited(client_ip):
        wait = login_limiter.remaining_seconds(client_ip)
        return templates.TemplateResponse(
            "auth/login.html",
            {"request": request, "error": f"Too many login attempts. Try again in {wait} seconds."},
            status_code=429,
        )

    with session_scope() as session:
        user = get_user_by_email(session, email)
        if not user or not verify_password(password, user.password_hash):
            login_limiter.record_attempt(client_ip)
            return templates.TemplateResponse(
                "auth/login.html",
                {"request": request, "error": "Invalid email or password"},
            )

        request.session["user_id"] = user.id
        request.session["session_version"] = getattr(user, "session_version", 0)
        return RedirectResponse(url="/feed", status_code=302)


@router.post("/logout",
             summary="Logout",
             description="Clear the user session and redirect to the landing page.")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)


# ── Email verification ─────────────────────────────────────────────────

@router.get("/verify-email",
            summary="Verify email",
            description="Consume a one-time email verification token and mark the user's email as verified.")
async def verify_email_handler(request: Request, token: str = ""):
    if not token:
        return templates.TemplateResponse(
            "auth/login.html", {"request": request, "error": "Invalid verification link."},
        )

    user_id = consume_token(token, "email_verification")
    if not user_id:
        return templates.TemplateResponse(
            "auth/login.html", {"request": request, "error": "Verification link is invalid or expired."},
        )

    with session_scope() as session:
        user = get_user_by_id(session, user_id)
        if user:
            user.email_verified = True
            user.email_verified_at = utcnow()
            session.add(user)
            session.commit()
            return templates.TemplateResponse("auth/verify_success.html", {"request": request})

    return templates.TemplateResponse(
        "auth/login.html", {"request": request, "error": "Verification failed."},
    )


@router.post("/resend-verification",
             summary="Resend verification email",
             description="Send a new verification email to the logged-in user. Rate-limited per IP.")
async def resend_verification(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    client_ip = request.client.host if request.client else "unknown"
    if verification_resend_limiter.is_rate_limited(client_ip):
        return RedirectResponse(url=_safe_referer(request), status_code=302)

    verification_resend_limiter.record_attempt(client_ip)

    with session_scope() as session:
        user = get_user_by_id(session, user_id)
        if not user or user.email_verified:
            return RedirectResponse(url="/feed", status_code=302)

        try:
            token = create_verification_token(user.id)
            base_url = _build_base_url(request)
            sender = EmailSender(load_settings())
            sender.send_verification_email(user, f"{base_url}/verify-email?token={token}")
        except Exception as e:
            logger.warning("Failed to resend verification email")

        return RedirectResponse(url=_safe_referer(request), status_code=302)


# ── Password reset ─────────────────────────────────────────────────────

@router.get("/forgot-password", response_class=HTMLResponse,
            summary="Forgot password page",
            description="Render the forgot-password form.")
async def forgot_password_page(request: Request):
    return templates.TemplateResponse("auth/forgot_password.html", {"request": request})


@router.post("/forgot-password",
             summary="Submit forgot password",
             description="Send a password reset email if the account exists. Always shows a generic success message to avoid leaking account existence.")
async def forgot_password_submit(request: Request, email: str = Form(...)):
    client_ip = request.client.host if request.client else "unknown"

    if password_reset_limiter.is_rate_limited(client_ip):
        wait = password_reset_limiter.remaining_seconds(client_ip)
        return templates.TemplateResponse(
            "auth/forgot_password.html",
            {"request": request, "email": email,
             "error": f"Too many reset attempts. Try again in {wait} seconds."},
            status_code=429,
        )

    password_reset_limiter.record_attempt(client_ip)

    with session_scope() as session:
        user = get_user_by_email(session, email)
        if user:
            try:
                token = create_reset_token(user.id)
                base_url = _build_base_url(request)
                sender = EmailSender(load_settings())
                sender.send_password_reset_email(user, f"{base_url}/reset-password?token={token}")
            except Exception as e:
                logger.warning("Failed to send reset email")

    # Always show generic success to avoid leaking account existence
    return templates.TemplateResponse(
        "auth/forgot_password.html",
        {"request": request, "success": "If an account exists with that email, we've sent a password reset link."},
    )


@router.get("/reset-password", response_class=HTMLResponse,
            summary="Reset password page",
            description="Render the password reset form after validating the reset token.")
async def reset_password_page(request: Request, token: str = ""):
    if not token:
        return templates.TemplateResponse(
            "auth/login.html", {"request": request, "error": "Invalid reset link."},
        )

    user_id = verify_token(token, "password_reset")
    if not user_id:
        return templates.TemplateResponse(
            "auth/login.html", {"request": request, "error": "Reset link is invalid or expired."},
        )

    return templates.TemplateResponse(
        "auth/reset_password.html", {"request": request, "token": token},
    )


@router.post("/reset-password",
             summary="Submit password reset",
             description="Consume the reset token and set a new password. Invalidates existing sessions.")
async def reset_password_submit(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    if password != confirm_password:
        return templates.TemplateResponse(
            "auth/reset_password.html",
            {"request": request, "token": token, "error": "Passwords do not match."},
        )

    pw_error = _validate_password(password)
    if pw_error:
        return templates.TemplateResponse(
            "auth/reset_password.html",
            {"request": request, "token": token, "error": pw_error},
        )

    user_id = consume_token(token, "password_reset")
    if not user_id:
        return templates.TemplateResponse(
            "auth/login.html",
            {"request": request, "error": "Reset link is invalid or expired."},
        )

    with session_scope() as session:
        user = get_user_by_id(session, user_id)
        if user:
            user.password_hash = hash_password(password)
            user.session_version = getattr(user, "session_version", 0) + 1
            session.add(user)
            session.commit()
            # Clear old session and start a fresh one with the new version
            request.session.clear()
            request.session["user_id"] = user.id
            request.session["session_version"] = user.session_version
            return RedirectResponse(url="/feed", status_code=302)

    return templates.TemplateResponse(
        "auth/login.html", {"request": request, "error": "Password reset failed."},
    )


# -- Unsubscribe (CAN-SPAM compliance) -------------------------------------

@router.get("/unsubscribe",
            summary="Unsubscribe from emails",
            description="Process a signed unsubscribe link from a digest email. Deactivates the user account.")
async def unsubscribe_handler(request: Request, t: str = ""):
    if not t:
        return templates.TemplateResponse(
            "auth/login.html",
            {"request": request, "error": "Invalid unsubscribe link."},
        )

    settings = load_settings()
    payload = verify_unsubscribe(settings.secret_key, t)
    if not payload:
        return templates.TemplateResponse(
            "auth/login.html",
            {"request": request, "error": "Unsubscribe link is invalid or expired."},
        )

    with session_scope() as session:
        user = get_user_by_id(session, payload["user_id"])
        if user and user.email == payload["email"]:
            user.active = False
            session.add(user)
            session.commit()
            logger.info(f"User {user.id} unsubscribed via email link")
            return HTMLResponse(
                content="<html><body><h1>Unsubscribed</h1>"
                "<p>You have been unsubscribed and will no longer receive digest emails.</p>"
                '<p><a href="/login">Log in</a> to re-subscribe.</p></body></html>'
            )

    return templates.TemplateResponse(
        "auth/login.html",
        {"request": request, "error": "Unsubscribe failed."},
    )
