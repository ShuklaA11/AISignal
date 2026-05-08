"""Tests for section/audience_tags/quality_weight fields on Article and Source."""

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from src.sections import SECTION_BUILDER, SECTION_RESEARCH
from src.storage.models import Article, Source


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.mark.unit
def test_article_defaults_for_new_fields(session: Session) -> None:
    article = Article(
        url="https://example.com/a",
        content_hash="h1",
        title="t",
        source_name="lilian_weng",
        source_type="rss",
    )
    session.add(article)
    session.commit()
    session.refresh(article)

    assert article.section is None
    assert article.audience_tags == []
    assert article.quality_weight == 1.0


@pytest.mark.unit
def test_article_persists_section_audience_quality(session: Session) -> None:
    article = Article(
        url="https://example.com/b",
        content_hash="h2",
        title="t",
        source_name="simon_willison",
        source_type="rss",
        section=SECTION_BUILDER,
        quality_weight=1.5,
    )
    article.audience_tags = ["industry", "researcher"]
    session.add(article)
    session.commit()

    fetched = session.exec(
        select(Article).where(Article.url == "https://example.com/b")
    ).one()
    assert fetched.section == SECTION_BUILDER
    assert fetched.audience_tags == ["industry", "researcher"]
    assert fetched.quality_weight == 1.5


@pytest.mark.unit
def test_source_persists_section_audience_quality(session: Session) -> None:
    source = Source(
        name="lilian_weng",
        source_type="rss",
        url="https://lilianweng.github.io/feed.xml",
        section=SECTION_RESEARCH,
        quality_weight=2.0,
    )
    source.audience_tags = ["researcher"]
    session.add(source)
    session.commit()

    fetched = session.exec(select(Source).where(Source.name == "lilian_weng")).one()
    assert fetched.section == SECTION_RESEARCH
    assert fetched.audience_tags == ["researcher"]
    assert fetched.quality_weight == 2.0


@pytest.mark.unit
def test_article_can_filter_by_section(session: Session) -> None:
    session.add_all(
        [
            Article(
                url="u1",
                content_hash="c1",
                title="t1",
                source_name="s",
                source_type="rss",
                section=SECTION_RESEARCH,
            ),
            Article(
                url="u2",
                content_hash="c2",
                title="t2",
                source_name="s",
                source_type="rss",
                section=SECTION_BUILDER,
            ),
            Article(
                url="u3",
                content_hash="c3",
                title="t3",
                source_name="s",
                source_type="rss",
                section=SECTION_RESEARCH,
            ),
        ]
    )
    session.commit()

    research_articles = session.exec(
        select(Article).where(Article.section == SECTION_RESEARCH)
    ).all()
    assert len(research_articles) == 2
