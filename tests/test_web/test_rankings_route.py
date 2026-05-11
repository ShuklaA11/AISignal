"""Tests for the /rankings route."""
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine

from src.leaderboards import Ranking, Snapshot, persist_snapshot


@pytest.fixture
def app_with_db(monkeypatch, tmp_path):
    """Boot the FastAPI app pointed at a fresh sqlite file."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("NEWSLETTER_SECRET_KEY", "x" * 32)
    monkeypatch.setenv("NEWSLETTER_DATABASE_URL", db_url)

    # Reset cached settings so the env vars take effect
    import src.config
    src.config._cached_settings = None

    # Reset DB engine cache
    import src.storage.database as db_mod
    db_mod._engine = None

    # Initialize schema directly via create_all (skip alembic for tests)
    eng = create_engine(db_url)
    SQLModel.metadata.create_all(eng)

    from src.web.app import app
    client = TestClient(app)
    return client, eng


def test_rankings_page_empty_state(app_with_db):
    """With no snapshots, the page renders an empty-state message."""
    client, _ = app_with_db
    resp = client.get("/rankings")
    assert resp.status_code == 200
    assert "No snapshots yet" in resp.text


def test_rankings_page_renders_board(app_with_db):
    client, eng = app_with_db
    with Session(eng) as session:
        persist_snapshot(
            session,
            Snapshot(
                provider="artificial_analysis",
                metric="intelligence",
                rankings=[
                    Ranking("Claude Opus 4.7", 1, 57.28, "Anthropic"),
                    Ranking("GPT-5.5 (xhigh)", 2, 60.24, "OpenAI"),
                ],
                captured_at=datetime(2026, 5, 11),
            ),
        )

    resp = client.get("/rankings")
    assert resp.status_code == 200
    assert "Claude Opus 4.7" in resp.text
    assert "Anthropic" in resp.text
    assert "Artificial Analysis" in resp.text  # provider label
    assert "Intelligence" in resp.text  # metric label


def test_rankings_page_renders_movers(app_with_db):
    client, eng = app_with_db
    with Session(eng) as session:
        # previous snapshot
        persist_snapshot(
            session,
            Snapshot(
                provider="artificial_analysis",
                metric="intelligence",
                rankings=[
                    Ranking("Claude Opus 4.7", 3, 55.0, "Anthropic"),
                    Ranking("GPT-5.5", 1, 61.0, "OpenAI"),
                ],
                captured_at=datetime(2026, 5, 4),
            ),
        )
        # current snapshot — Claude climbed
        persist_snapshot(
            session,
            Snapshot(
                provider="artificial_analysis",
                metric="intelligence",
                rankings=[
                    Ranking("Claude Opus 4.7", 1, 57.28, "Anthropic"),
                    Ranking("GPT-5.5", 2, 60.24, "OpenAI"),
                ],
                captured_at=datetime(2026, 5, 11),
            ),
        )

    resp = client.get("/rankings")
    assert resp.status_code == 200
    assert "Top movers" in resp.text
    # Claude moved from 3 to 1 (-2 rank delta)
    assert "Claude Opus 4.7" in resp.text


def test_rankings_in_nav():
    """The nav bar links to /rankings."""
    from src.web.app import app
    client = TestClient(app)
    resp = client.get("/")
    assert "/rankings" in resp.text
