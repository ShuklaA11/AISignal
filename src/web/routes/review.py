from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from src.storage.models import Article, ArticleSummary, utcnow
from src.web.auth_utils import require_admin
from src.web.template_engine import templates

router = APIRouter()


@router.get("/review", response_class=HTMLResponse,
            summary="Review dashboard",
            description="Admin dashboard to review, approve, or reject today's processed articles with all summary variants.")
@router.get("/review/{date_str}", response_class=HTMLResponse,
            summary="Review dashboard by date",
            description="Admin dashboard for a specific date's articles.")
async def review_page(request: Request, date_str: str | None = None,
                      auth: tuple = Depends(require_admin)):
    user, session = auth
    try:
        today = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        stmt = (
            select(Article)
            .where(Article.status.in_(["processed", "approved"]))
            .where(Article.fetched_at >= today)
            .order_by(Article.base_importance_score.desc())
            .limit(100)
        )
        articles = list(session.exec(stmt).all())

        # Bulk-load all summaries for these articles in a single query
        article_ids = [a.id for a in articles]
        if article_ids:
            s_stmt = (
                select(ArticleSummary)
                .where(ArticleSummary.article_id.in_(article_ids))
            )
            all_summaries = list(session.exec(s_stmt).all())
        else:
            all_summaries = []

        # Group summaries by article_id
        summary_lookup: dict[int, dict[str, str]] = {}
        for s in all_summaries:
            summary_lookup.setdefault(s.article_id, {})[s.role] = s.summary_text

        articles_with_data = []
        for article in articles:
            articles_with_data.append({
                "article": article,
                "summaries": summary_lookup.get(article.id, {}),
                "topics": article.topics,
            })

        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "user": user,
                "articles": articles_with_data,
                "date": today.strftime("%B %d, %Y"),
            },
        )
    finally:
        session.close()
