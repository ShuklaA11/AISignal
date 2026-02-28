import json

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.config import load_settings
from src.web.auth_utils import require_login
from src.web.template_engine import templates

router = APIRouter(prefix="/onboarding")

VALID_ROLES = {"student", "industry", "enthusiast"}
VALID_LEVELS = {"beginner", "intermediate", "advanced"}


@router.get("/role", response_class=HTMLResponse,
            summary="Role selection page",
            description="Render the role selection step (student, industry, enthusiast).")
async def role_page(request: Request, auth: tuple = Depends(require_login)):
    user, session = auth
    session.close()
    return templates.TemplateResponse("onboarding/role.html", {"request": request, "user": user})


@router.post("/role",
             summary="Submit role selection",
             description="Save the user's chosen role and proceed to level selection.")
async def role_submit(request: Request, role: str = Form(...), auth: tuple = Depends(require_login)):
    user, session = auth
    try:
        if role not in VALID_ROLES:
            return RedirectResponse(url="/onboarding/role", status_code=302)
        user.role = role
        session.add(user)
        session.commit()
        return RedirectResponse(url="/onboarding/level", status_code=302)
    finally:
        session.close()


@router.get("/level", response_class=HTMLResponse,
            summary="Level selection page",
            description="Render the difficulty level selection step (beginner, intermediate, advanced).")
async def level_page(request: Request, auth: tuple = Depends(require_login)):
    user, session = auth
    session.close()
    return templates.TemplateResponse("onboarding/level.html", {"request": request, "user": user})


@router.post("/level",
             summary="Submit level selection",
             description="Save the user's chosen difficulty level and proceed to topic selection.")
async def level_submit(request: Request, level: str = Form(...), auth: tuple = Depends(require_login)):
    user, session = auth
    try:
        if level not in VALID_LEVELS:
            return RedirectResponse(url="/onboarding/level", status_code=302)
        user.level = level
        session.add(user)
        session.commit()
        return RedirectResponse(url="/onboarding/topics", status_code=302)
    finally:
        session.close()


@router.get("/topics", response_class=HTMLResponse,
            summary="Topics selection page",
            description="Render the topic selection step with role-specific defaults.")
async def topics_page(request: Request, auth: tuple = Depends(require_login)):
    user, session = auth
    session.close()
    settings = load_settings()
    all_topics = []
    for group_topics in settings.topics.values():
        all_topics.extend(group_topics)

    # Get defaults for user's role
    role_defaults = settings.role_defaults.get(user.role, {})
    default_topics = role_defaults.get("topics", [])

    return templates.TemplateResponse(
        "onboarding/topics.html",
        {
            "request": request,
            "user": user,
            "all_topics": all_topics,
            "topic_groups": settings.topics,
            "default_topics": default_topics,
        },
    )


@router.post("/topics",
             summary="Submit topic selection",
             description="Save the user's selected topics and proceed to source weighting.")
async def topics_submit(request: Request, auth: tuple = Depends(require_login)):
    user, session = auth
    try:
        form = await request.form()
        selected = form.getlist("topics")
        settings = load_settings()
        valid_topics = {t for group in settings.topics.values() for t in group}
        selected = [t for t in selected if t in valid_topics]
        user.topics_json = json.dumps(selected)
        session.add(user)
        session.commit()
        return RedirectResponse(url="/onboarding/sources", status_code=302)
    finally:
        session.close()


@router.get("/sources", response_class=HTMLResponse,
            summary="Source weighting page",
            description="Render the source preference weighting step (0-10 per source).")
async def sources_page(request: Request, auth: tuple = Depends(require_login)):
    user, session = auth
    session.close()
    settings = load_settings()
    role_defaults = settings.role_defaults.get(user.role, {})
    default_weights = role_defaults.get("source_weights", {})

    sources = [
        {"name": "rss", "label": "News & Blogs", "description": "TechCrunch, VentureBeat, MIT Tech Review, etc."},
        {"name": "arxiv", "label": "arXiv Papers", "description": "Latest research papers from cs.AI, cs.CL, cs.CV, cs.LG"},
        {"name": "huggingface", "label": "HuggingFace Papers", "description": "Daily trending papers on HuggingFace"},
        {"name": "reddit", "label": "Reddit", "description": "r/MachineLearning, r/artificial, r/LocalLLaMA"},
        {"name": "twitter", "label": "Twitter/X", "description": "AI discussions and announcements"},
        {"name": "github", "label": "GitHub Trending", "description": "Trending AI/ML repositories"},
    ]

    return templates.TemplateResponse(
        "onboarding/sources.html",
        {
            "request": request,
            "user": user,
            "sources": sources,
            "default_weights": default_weights,
        },
    )


@router.post("/sources",
             summary="Submit source weights",
             description="Save source preference weights and complete onboarding. Redirects to the feed.")
async def sources_submit(request: Request, auth: tuple = Depends(require_login)):
    user, session = auth
    try:
        form = await request.form()
        weights = {}
        for key, val in form.items():
            if key.startswith("weight_"):
                source_name = key.replace("weight_", "")
                try:
                    weight = int(val)
                except (ValueError, TypeError):
                    weight = 5  # Default weight on invalid input
                weights[source_name] = max(0, min(10, weight))  # Clamp to valid range
        user.source_preferences_json = json.dumps(weights)
        session.add(user)
        session.commit()
        return RedirectResponse(url="/feed", status_code=302)
    finally:
        session.close()
