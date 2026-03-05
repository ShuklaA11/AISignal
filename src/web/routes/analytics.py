"""Analytics dashboard — evaluation metrics for personalization."""

from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select, func as sqlfunc

from src.metrics.calculator import compute_daily_metrics
from src.storage.models import Article, ArticleEmbedding, FeedImpression, UserMLProfile
from src.storage.queries import (
    get_active_users, get_aggregate_daily_metrics, get_fetch_health,
    get_metrics_for_user, get_ml_profile,
)
from src.web.auth_utils import require_login
from src.web.template_engine import templates

router = APIRouter()


@router.get("/analytics", response_class=HTMLResponse,
            summary="Analytics dashboard",
            description="View personalization metrics (CTR, nDCG@10, save rate, lift). Admins see aggregate stats, per-user profiles, system health, and fetch monitoring.")
async def analytics_page(request: Request, tab: str = Query("personal"),
                         auth: tuple = Depends(require_login)):
    user, session = auth
    try:
        # Recompute today's metrics so the dashboard is always fresh
        compute_daily_metrics(session, user.id, datetime.now(timezone.utc).date())

        # Personal metrics (always computed)
        metrics = get_metrics_for_user(session, user.id, days=30)
        ml_profile = get_ml_profile(session, user.id)

        total_impressions = sum(m.total_impressions for m in metrics)
        total_clicks = sum(m.total_clicks for m in metrics)
        total_saves = sum(m.total_saves for m in metrics)

        latest = metrics[-1] if metrics else None

        ctr_data = [m.ctr for m in metrics]
        save_rate_data = [m.save_rate for m in metrics]
        ndcg_data = [m.ndcg_at_10 for m in metrics]
        lift_data = [m.personalization_lift for m in metrics]

        ctx = {
            "request": request,
            "user": user,
            "active_tab": tab if user.is_admin else "personal",
            "metrics": metrics,
            "ml_profile": ml_profile,
            "total_impressions": total_impressions,
            "total_clicks": total_clicks,
            "total_saves": total_saves,
            "latest": latest,
            "ctr_data": ctr_data,
            "save_rate_data": save_rate_data,
            "ndcg_data": ndcg_data,
            "lift_data": lift_data,
        }

        # Admin aggregate data
        if user.is_admin:
            agg = get_aggregate_daily_metrics(session, days=30)
            all_users = get_active_users(session)

            # Per-user ML profile summaries
            user_profiles = []
            for u in all_users:
                profile = get_ml_profile(session, u.id)
                u_metrics = get_metrics_for_user(session, u.id, days=30)
                u_imps = sum(m.total_impressions for m in u_metrics)
                u_clicks = sum(m.total_clicks for m in u_metrics)
                u_saves = sum(m.total_saves for m in u_metrics)
                user_profiles.append({
                    "user": u,
                    "profile": profile,
                    "total_impressions": u_imps,
                    "total_clicks": u_clicks,
                    "total_saves": u_saves,
                    "ctr": round(u_clicks / u_imps, 4) if u_imps else 0,
                })

            # System-level stats
            total_articles = session.exec(
                select(sqlfunc.count(Article.id))
            ).one() or 0
            processed_articles = session.exec(
                select(sqlfunc.count(Article.id))
                .where(Article.status.in_(["processed", "approved", "sent"]))
            ).one() or 0
            total_embeddings = session.exec(
                select(sqlfunc.count(ArticleEmbedding.id))
            ).one() or 0
            total_all_impressions = session.exec(
                select(sqlfunc.count(FeedImpression.id))
            ).one() or 0

            # Aggregate sparklines
            agg_ctr_data = [d["avg_ctr"] for d in agg]
            agg_save_data = [d["avg_save_rate"] for d in agg]
            agg_ndcg_data = [d["avg_ndcg"] for d in agg]
            agg_lift_data = [d["avg_lift"] for d in agg]

            # Aggregate totals
            agg_impressions = sum(d["sum_impressions"] for d in agg)
            agg_clicks = sum(d["sum_clicks"] for d in agg)
            agg_saves = sum(d["sum_saves"] for d in agg)
            agg_latest = agg[-1] if agg else None

            # Alpha distribution
            all_profiles = list(session.exec(select(UserMLProfile)).all())
            alpha_values = [p.alpha for p in all_profiles]
            avg_alpha = round(sum(alpha_values) / len(alpha_values), 2) if alpha_values else 1.0
            learning_count = sum(1 for a in alpha_values if a > 0.7)
            blending_count = sum(1 for a in alpha_values if 0.4 < a <= 0.7)
            ml_dominant_count = sum(1 for a in alpha_values if a <= 0.4)

            # Fetch health data
            fetch_health = get_fetch_health(session, days=14)

            ctx.update({
                "agg_metrics": agg,
                "agg_latest": agg_latest,
                "agg_ctr_data": agg_ctr_data,
                "agg_save_data": agg_save_data,
                "agg_ndcg_data": agg_ndcg_data,
                "agg_lift_data": agg_lift_data,
                "agg_impressions": agg_impressions,
                "agg_clicks": agg_clicks,
                "agg_saves": agg_saves,
                "user_profiles": user_profiles,
                "total_users": len(all_users),
                "total_articles": total_articles,
                "processed_articles": processed_articles,
                "total_embeddings": total_embeddings,
                "total_all_impressions": total_all_impressions,
                "avg_alpha": avg_alpha,
                "learning_count": learning_count,
                "blending_count": blending_count,
                "ml_dominant_count": ml_dominant_count,
                "fetch_health": fetch_health,
            })

        return templates.TemplateResponse("analytics.html", ctx)
    finally:
        session.close()
