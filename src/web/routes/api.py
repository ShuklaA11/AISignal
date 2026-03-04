"""HTMX API endpoints for interactive actions."""

import logging
import re
from html import escape
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from src.personalization.learner import (
    update_on_click, update_on_dislike, update_on_like, update_on_save,
)
from src.storage.database import session_scope
from src.storage.models import Article, ArticleSummary
from src.storage.queries import (
    get_impression_feedback, mark_article_read,
    record_digest_click, toggle_saved_article, update_impression_clicked,
    update_impression_disliked, update_impression_feedback_cleared,
    update_impression_liked, update_impression_saved,
)
from src.config import load_settings
from src.web.auth_utils import require_admin, require_user_id
from src.web.digest_token import verify_digest_click
from src.web.rate_limit import limiter
from src.web.template_engine import templates

router = APIRouter()


@router.put("/articles/{article_id}/approve", response_class=HTMLResponse,
            summary="Approve article",
            description="Mark an article as approved for inclusion in digests. Admin only.")
async def approve_article(request: Request, article_id: int,
                          auth: tuple = Depends(require_admin)):
    user, session = auth
    try:
        article = session.get(Article, article_id)
        if article:
            article.status = "approved"
            session.add(article)
            session.commit()
            session.refresh(article)
        return templates.TemplateResponse(
            "partials/article_card_review.html",
            {"request": request, "article": article, "summaries": _get_summaries(session, article_id)},
        )
    finally:
        session.close()


@router.put("/articles/{article_id}/reject", response_class=HTMLResponse,
            summary="Reject article",
            description="Mark an article as rejected to exclude it from digests. Admin only.")
async def reject_article(request: Request, article_id: int,
                         auth: tuple = Depends(require_admin)):
    user, session = auth
    try:
        article = session.get(Article, article_id)
        if article:
            article.status = "rejected"
            session.add(article)
            session.commit()
            session.refresh(article)
        return templates.TemplateResponse(
            "partials/article_card_review.html",
            {"request": request, "article": article, "summaries": _get_summaries(session, article_id)},
        )
    finally:
        session.close()


@router.put("/articles/{article_id}/summary", response_class=HTMLResponse,
            summary="Update article summary",
            description="Edit the summary text for a specific role variant of an article. Admin only.")
async def update_summary(
    request: Request,
    article_id: int,
    role: str = Form(...),
    summary_text: str = Form(...),
    auth: tuple = Depends(require_admin),
):
    _, session = auth
    try:
        stmt = (
            select(ArticleSummary)
            .where(ArticleSummary.article_id == article_id)
            .where(ArticleSummary.role == role)
        )
        summary = session.exec(stmt).first()
        if summary:
            clean_text = re.sub(r"<[^>]+>", "", summary_text).strip()
            summary.summary_text = clean_text
            session.add(summary)
            session.commit()
        else:
            clean_text = re.sub(r"<[^>]+>", "", summary_text).strip()
        return HTMLResponse(content=escape(clean_text))
    finally:
        session.close()


@router.post("/articles/{article_id}/toggle-save", response_class=HTMLResponse,
             summary="Toggle saved article",
             description="Save or unsave an article for the current user. Updates ML profile on save.")
async def toggle_save_article(request: Request, article_id: int,
                              user_id: int = Depends(require_user_id)):
    with session_scope() as session:
        is_saved = toggle_saved_article(session, user_id, article_id)
        # Only update ML profile if there's a feed impression to match.
        # Saves from read-history or direct links aren't feed engagement signals.
        if is_saved:
            had_impression = update_impression_saved(session, user_id, article_id)
            if had_impression:
                update_on_save(session, user_id, article_id)
        # Return just the updated button (values are controlled, not user input)
        icon = "lucide:bookmark-check" if is_saved else "lucide:bookmark"
        color = "text-coral" if is_saved else "text-forest/20 hover:text-coral"
        title = "Unsave" if is_saved else "Save"
        return HTMLResponse(content=f"""
            <button hx-post="/api/articles/{escape(str(article_id))}/toggle-save"
                    hx-target="#save-btn-{escape(str(article_id))}"
                    hx-swap="innerHTML"
                    class="{escape(color)} transition flex-shrink-0"
                    title="{escape(title)} article">
                <iconify-icon icon="{escape(icon)}" class="text-lg"></iconify-icon>
            </button>
        """)


@router.post("/articles/{article_id}/mark-read",
             summary="Mark article as read",
             description="Record that the user read an article. Updates ML profile and impression tracking.")
async def mark_read(request: Request, article_id: int,
                    user_id: int = Depends(require_user_id)):
    with session_scope() as session:
        mark_article_read(session, user_id, article_id)
        # Only update ML profile if there's a feed impression to match.
        # Reads from read-history, saved articles, or direct links aren't
        # feed engagement signals and shouldn't train the learned model.
        had_impression = update_impression_clicked(session, user_id, article_id)
        if had_impression:
            update_on_click(session, user_id, article_id)
        return HTMLResponse(content="", status_code=204)


def _feedback_buttons_html(article_id: int, liked: bool, disliked: bool, toast: str = "") -> str:
    """Return the thumbs up/down button pair HTML with current state and optional toast."""
    aid = escape(str(article_id))
    up_color = "text-coral" if liked else "text-forest/20 hover:text-coral"
    down_color = "text-coral" if disliked else "text-forest/20 hover:text-coral"
    toast_html = ""
    if toast:
        toast_html = f"""
        <div class="absolute -top-8 left-1/2 -translate-x-1/2 whitespace-nowrap
                    bg-forest text-white text-xs px-2.5 py-1 rounded shadow-lg
                    animate-fade-in-out pointer-events-none z-10">
            {escape(toast)}
        </div>
        <style>
            @keyframes fadeInOut {{
                0% {{ opacity: 0; transform: translate(-50%, 4px); }}
                15% {{ opacity: 1; transform: translate(-50%, 0); }}
                70% {{ opacity: 1; transform: translate(-50%, 0); }}
                100% {{ opacity: 0; transform: translate(-50%, -4px); }}
            }}
            .animate-fade-in-out {{
                animation: fadeInOut 2s ease-in-out forwards;
            }}
        </style>
        """
    return f"""
        <div class="flex gap-1.5 relative">
            {toast_html}
            <button hx-post="/api/articles/{aid}/like"
                    hx-target="#feedback-btn-{aid}"
                    hx-swap="innerHTML"
                    class="{escape(up_color)} transition flex-shrink-0"
                    title="More like this">
                <iconify-icon icon="lucide:thumbs-up" class="text-base"></iconify-icon>
            </button>
            <button hx-post="/api/articles/{aid}/dislike"
                    hx-target="#feedback-btn-{aid}"
                    hx-swap="innerHTML"
                    class="{escape(down_color)} transition flex-shrink-0"
                    title="Less like this">
                <iconify-icon icon="lucide:thumbs-down" class="text-base"></iconify-icon>
            </button>
        </div>
    """


@router.post("/articles/{article_id}/like", response_class=HTMLResponse,
             summary="Like article",
             description="Toggle a like on an article. Trains the ML profile to show more similar content.")
async def like_article(request: Request, article_id: int,
                       user_id: int = Depends(require_user_id)):
    with session_scope() as session:
        was_liked, _ = get_impression_feedback(session, user_id, article_id)
        if was_liked:
            # Toggle off — undo the like
            update_impression_feedback_cleared(session, user_id, article_id)
            return HTMLResponse(content=_feedback_buttons_html(
                article_id, liked=False, disliked=False))
        # Apply like — only train ML if there's a feed impression
        had_impression = update_impression_liked(session, user_id, article_id)
        if had_impression:
            update_on_like(session, user_id, article_id)
        return HTMLResponse(content=_feedback_buttons_html(
            article_id, liked=True, disliked=False,
            toast="Showing more like this"))


@router.post("/articles/{article_id}/dislike", response_class=HTMLResponse,
             summary="Dislike article",
             description="Toggle a dislike on an article. Trains the ML profile to show less similar content.")
async def dislike_article(request: Request, article_id: int,
                          user_id: int = Depends(require_user_id)):
    with session_scope() as session:
        _, was_disliked = get_impression_feedback(session, user_id, article_id)
        if was_disliked:
            # Toggle off — undo the dislike
            update_impression_feedback_cleared(session, user_id, article_id)
            return HTMLResponse(content=_feedback_buttons_html(
                article_id, liked=False, disliked=False))
        # Apply dislike — only train ML if there's a feed impression
        had_impression = update_impression_disliked(session, user_id, article_id)
        if had_impression:
            update_on_dislike(session, user_id, article_id)
        return HTMLResponse(content=_feedback_buttons_html(
            article_id, liked=False, disliked=True,
            toast="Showing less like this"))


@router.get("/digest/click",
            summary="Track digest click",
            description="Track an email digest link click via signed token, update ML profile, then redirect to the article URL.")
async def digest_click(
    request: Request,
    t: str = "",
    section: str = "main",
):
    """Track an email digest link click, update ML profile, then redirect to the article."""
    settings = load_settings()

    if not t:
        return RedirectResponse(url="/feed", status_code=302)

    payload = verify_digest_click(settings.secret_key, t)
    if not payload:
        return RedirectResponse(url="/feed", status_code=302)

    click_user_id = payload["user_id"]
    click_article_id = payload["article_id"]
    click_digest_id = payload["digest_id"]
    click_section = payload["section"]

    with session_scope() as session:
        article = None
        try:
            article = session.get(Article, click_article_id)
            if not article:
                return RedirectResponse(url="/feed", status_code=302)

            # Validate URL scheme to prevent open redirects via poisoned article URLs
            parsed = urlparse(article.url)
            if parsed.scheme not in ("http", "https"):
                return RedirectResponse(url="/feed", status_code=302)

            # Record the click
            record_digest_click(session, click_user_id, click_article_id, click_digest_id, click_section)

            # Also feed into ML learner (same as a feed click)
            mark_article_read(session, click_user_id, click_article_id)
            update_on_click(session, click_user_id, click_article_id)

            # Redirect to the actual article
            return RedirectResponse(url=article.url, status_code=302)
        except Exception:
            # If anything fails, still redirect to the article URL or feed
            if article and urlparse(article.url).scheme in ("http", "https"):
                return RedirectResponse(url=article.url, status_code=302)
            return RedirectResponse(url="/feed", status_code=302)


@router.post("/digests/send", response_class=HTMLResponse,
             summary="Send digests",
             description="Build and send personalized digests to all active users. Admin only.")
@limiter.limit("5/minute")
async def send_digests(request: Request, auth: tuple = Depends(require_admin)):
    """Build and send personalized digests to all active users."""
    _, session = auth
    session.close()

    from src.config import load_settings
    from src.pipeline.scheduler import send_all_digests
    settings = load_settings()
    result = await send_all_digests(settings)

    return HTMLResponse(content=f"""
        <div class="px-4 py-2.5 rounded text-sm font-heading font-semibold
                    bg-mint/20 text-forest border border-mint">
            Sent {result['sent']} digest(s)
            {f", {result['failed']} failed" if result['failed'] else ""}
            {f", {result['skipped']} skipped (empty)" if result['skipped'] else ""}
        </div>
    """)


def _get_summaries(session, article_id: int) -> dict[str, str]:
    stmt = (
        select(ArticleSummary)
        .where(ArticleSummary.article_id == article_id)
        .where(ArticleSummary.role.in_(["student", "industry", "enthusiast"]))
    )
    return {s.role: s.summary_text for s in session.exec(stmt).all()}
