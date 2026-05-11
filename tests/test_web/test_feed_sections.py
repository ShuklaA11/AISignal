"""Tests for section filter on /feed."""
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from src.sections import SECTION_BUILDER, SECTION_RESEARCH
from src.storage.models import Article


@pytest.fixture
def app_with_articles(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("NEWSLETTER_SECRET_KEY", "x" * 32)
    monkeypatch.setenv("NEWSLETTER_DATABASE_URL", db_url)

    import src.config
    src.config._cached_settings = None
    import src.storage.database as db_mod
    db_mod._engine = None

    eng = create_engine(db_url)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as session:
        for i, section in enumerate([SECTION_RESEARCH, SECTION_BUILDER, SECTION_RESEARCH]):
            session.add(Article(
                url=f"https://example.com/{i}",
                content_hash=f"h{i}",
                title=f"Article {section} #{i}",
                source_name="openai_blog",
                source_type="rss",
                status="processed",
                section=section,
            ))
        session.commit()

    from src.web.app import app
    return TestClient(app)


def test_feed_renders_section_tabs(app_with_articles):
    resp = app_with_articles.get("/feed")
    assert resp.status_code == 200
    # Tab labels render
    assert "Research" in resp.text
    assert "Releases" in resp.text
    assert "Builder" in resp.text


def test_feed_with_section_param_filters_articles(app_with_articles):
    resp = app_with_articles.get("/feed?section=research")
    assert resp.status_code == 200
    # The Research article titles are present; Builder ones aren't
    assert "Article research" in resp.text
    assert "Article builder" not in resp.text


def test_feed_with_invalid_section_falls_back_to_all(app_with_articles):
    resp = app_with_articles.get("/feed?section=bogus")
    assert resp.status_code == 200
    # All articles render
    assert "Article research" in resp.text
    assert "Article builder" in resp.text


def test_feed_section_all_explicit(app_with_articles):
    resp = app_with_articles.get("/feed?section=all")
    assert resp.status_code == 200
    assert "Article research" in resp.text
    assert "Article builder" in resp.text
