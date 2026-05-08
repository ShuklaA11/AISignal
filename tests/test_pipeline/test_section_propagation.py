"""Tests that section/audience_tags/quality_weight flow from RawArticle to Article."""
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from src.fetchers.base import RawArticle
from src.pipeline.orchestrator import store_articles
from src.sections import SECTION_BUILDER, SECTION_RESEARCH
from src.storage.models import Article


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.mark.unit
def test_raw_article_defaults_for_section_fields() -> None:
    raw = RawArticle(url="u", title="t")
    assert raw.section is None
    assert raw.audience_tags == []
    assert raw.quality_weight == 1.0


@pytest.mark.unit
def test_store_articles_persists_section_metadata(session: Session) -> None:
    raw_articles = [
        RawArticle(
            url="https://example.com/paper",
            title="Some Paper",
            source_name="hf_daily_papers",
            source_type="api",
            section=SECTION_RESEARCH,
            audience_tags=["researcher", "industry"],
            quality_weight=1.7,
        ),
    ]
    new_count = store_articles(session, raw_articles)
    assert new_count == 1

    fetched = session.exec(select(Article).where(Article.url == "https://example.com/paper")).one()
    assert fetched.section == SECTION_RESEARCH
    assert fetched.audience_tags == ["researcher", "industry"]
    assert fetched.quality_weight == 1.7


@pytest.mark.unit
def test_store_articles_handles_untagged_raw(session: Session) -> None:
    """Backward compat: RawArticle without section fields stores neutral defaults."""
    raw_articles = [
        RawArticle(
            url="https://example.com/untagged",
            title="Untagged",
            source_name="legacy_rss",
            source_type="rss",
        ),
    ]
    store_articles(session, raw_articles)

    fetched = session.exec(select(Article).where(Article.url == "https://example.com/untagged")).one()
    assert fetched.section is None
    assert fetched.audience_tags == []
    assert fetched.quality_weight == 1.0


@pytest.mark.unit
def test_store_articles_preserves_section_per_article(session: Session) -> None:
    raw_articles = [
        RawArticle(url="u1", title="t1", source_name="s", source_type="api",
                   section=SECTION_RESEARCH, quality_weight=2.0),
        RawArticle(url="u2", title="t2", source_name="s", source_type="api",
                   section=SECTION_BUILDER, quality_weight=0.8),
    ]
    store_articles(session, raw_articles)

    a1 = session.exec(select(Article).where(Article.url == "u1")).one()
    a2 = session.exec(select(Article).where(Article.url == "u2")).one()
    assert a1.section == SECTION_RESEARCH
    assert a1.quality_weight == 2.0
    assert a2.section == SECTION_BUILDER
    assert a2.quality_weight == 0.8
