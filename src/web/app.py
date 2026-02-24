import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse as StarletteRedirect

from markupsafe import Markup

from src.config import load_settings
from src.logging_config import setup_logging
from src.storage.database import init_db
from src.web.auth_utils import _AdminRequired, _LoginRequired
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.web.csrf import CSRFMiddleware, _get_or_create_token
from src.web.rate_limit import limiter
from src.web.template_engine import templates

setup_logging()
settings = load_settings()

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(settings.database_url)

    # Re-run logging setup AFTER uvicorn has finished its own logging
    # configuration. This is the last chance to attach our file handler
    # before any scheduled jobs fire.
    setup_logging()

    # Start scheduler
    from src.pipeline.scheduler import setup_scheduler
    scheduler = setup_scheduler(settings)
    scheduler.start()

    yield

    scheduler.shutdown()


app = FastAPI(title="AI Newsletter", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

SESSION_ABSOLUTE_TIMEOUT = 86400 * 30  # 30 days absolute max

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://unpkg.com https://cdn.jsdelivr.net https://code.iconify.design; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https://api.iconify.design"
        )
        if _https_only:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response


class AbsoluteSessionTimeoutMiddleware(BaseHTTPMiddleware):
    """Enforce an absolute session lifetime via a created_at timestamp."""

    async def dispatch(self, request: Request, call_next):
        created_at = request.session.get("created_at")
        if created_at and (time.time() - created_at) > SESSION_ABSOLUTE_TIMEOUT:
            request.session.clear()
        elif request.session.get("user_id") and not created_at:
            request.session["created_at"] = time.time()
        return await call_next(request)


app.add_middleware(CSRFMiddleware)
app.add_middleware(AbsoluteSessionTimeoutMiddleware)

_https_only = not settings.base_url.startswith("http://")
if not _https_only:
    logger.warning("Session cookie https_only is False — cookies will be sent over plain HTTP. "
                    "Set base_url to an https:// URL for production use.")

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    max_age=86400 * 7,  # 7 days
    same_site="lax",
    https_only=_https_only,
)

# SecurityHeadersMiddleware is added last so it is the outermost middleware
# (Starlette processes middleware in reverse order of add_middleware calls).
app.add_middleware(SecurityHeadersMiddleware)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def csrf_input(request: Request) -> Markup:
    """Generate a hidden input with the CSRF token for use in templates."""
    token = _get_or_create_token(request)
    return Markup('<input type="hidden" name="csrf_token" value="{}">').format(token)


def csrf_token(request: Request) -> str:
    """Return the raw CSRF token value (for HTMX headers)."""
    return _get_or_create_token(request)

templates.env.globals["csrf_token"] = csrf_token
templates.env.globals["csrf_input"] = csrf_input

# Import and include routers
from src.web.routes.api import router as api_router
from src.web.routes.auth import router as auth_router
from src.web.routes.onboarding import router as onboarding_router
from src.web.routes.profile import router as profile_router
from src.web.routes.feed import router as feed_router
from src.web.routes.review import router as review_router
from src.web.routes.analytics import router as analytics_router

app.include_router(auth_router)
app.include_router(onboarding_router)
app.include_router(feed_router)
app.include_router(review_router)
app.include_router(profile_router)
app.include_router(analytics_router)
app.include_router(api_router, prefix="/api")


@app.exception_handler(_LoginRequired)
async def login_required_handler(request: Request, exc: _LoginRequired):
    # HTMX API requests get 401; page requests get a redirect
    if request.url.path.startswith("/api/"):
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content="", status_code=401)
    return StarletteRedirect(url="/login", status_code=302)


@app.exception_handler(_AdminRequired)
async def admin_required_handler(request: Request, exc: _AdminRequired):
    if request.url.path.startswith("/api/"):
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content="Forbidden", status_code=403)
    return StarletteRedirect(url="/feed", status_code=302)


@app.get("/health",
         summary="Health check",
         description="Returns the application health status. Verifies database connectivity.")
async def health_check():
    """Health check endpoint for monitoring."""
    import logging as _logging
    from sqlalchemy import text
    from src.storage.database import session_scope
    try:
        with session_scope() as session:
            session.exec(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        _logging.getLogger(__name__).error(f"Health check failed: {e}")
        return {"status": "degraded"}


@app.get("/", response_class=HTMLResponse,
         summary="Landing page",
         description="Render the landing page. Shows user info if logged in.")
async def root(request: Request):
    user_id = request.session.get("user_id")
    user = None
    if user_id:
        from src.storage.database import session_scope
        from src.storage.queries import get_user_by_id
        with session_scope() as session:
            user = get_user_by_id(session, user_id)
    return templates.TemplateResponse("landing.html", {"request": request, "user": user})
