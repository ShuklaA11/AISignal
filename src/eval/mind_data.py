"""Loaders that map the MIND news-recommendation dataset onto this project's models.

MIND (https://msnews.github.io/) ships two TSV files per split:

  news.tsv       news_id, category, subcategory, title, abstract, url,
                 title_entities, abstract_entities
  behaviors.tsv  impression_id, user_id, time, click_history, impressions

`impressions` is the candidate slate actually shown to the user, each entry
suffixed `-1` (clicked) or `-0` (skipped). Ranking that slate per impression and
averaging is the protocol every published MIND baseline uses, so it is what we
reproduce here.

Feature coverage is partial by necessity. MIND carries category, subcategory,
title/abstract text, and Wikidata entities; it carries no publication source, no
difficulty grading, no editorial importance score, and no user role or expertise
level. Those four signals of the production scorer are therefore inert on this
benchmark and are left at their neutral defaults rather than being faked.
"""

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from src.storage.models import Article, User

# Ranking-neutral defaults for the user attributes MIND does not provide. An
# unknown role/level misses every lookup in the scorer's weight tables, which
# collapses those factors to 1.0 for every candidate.
UNKNOWN_ROLE = ""
UNKNOWN_LEVEL = ""

# The scorer reads at most five entities per article.
MAX_ENTITIES = 5

# behaviors.tsv stamps each impression like "11/9/2019 6:11:24 AM" — a 12-hour
# clock with no zero padding on the month, day or hour.
IMPRESSION_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def parse_impression_time(raw: str) -> datetime:
    """Parse a behaviors.tsv timestamp.

    Raises rather than defaulting: the temporal validation split is defined by
    these values, so a row with an unreadable timestamp would be assigned to
    the wrong period silently and quietly invalidate the holdout.
    """
    try:
        return datetime.strptime(raw.strip(), IMPRESSION_TIME_FORMAT)
    except ValueError as error:
        raise ValueError(f"unparseable impression timestamp {raw!r}") from error


@dataclass(frozen=True)
class MindNews:
    """A single news item from news.tsv."""

    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    entities: tuple[str, ...]

    @property
    def embedding_text(self) -> str:
        """Text handed to the embedding model."""
        return f"{self.title}. {self.abstract}".strip()


@dataclass(frozen=True)
class MindImpression:
    """One impression: a user, their prior clicks, and the slate they saw."""

    impression_id: int
    user_id: str
    timestamp: datetime
    history: tuple[str, ...]
    candidates: tuple[str, ...]
    labels: tuple[int, ...]

    @property
    def num_clicks(self) -> int:
        return sum(self.labels)


def _parse_entities(raw: str) -> list[str]:
    """Pull Wikidata IDs out of a MIND entity JSON column.

    Wikidata IDs are used rather than surface labels so that "Prince Philip" and
    "Prince Philip, Duke of Edinburgh" collapse to one weight in the learner.
    """
    if not raw or raw == "[]":
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    entities = []
    for item in parsed:
        key = item.get("WikidataId") or item.get("Label")
        if key:
            entities.append(key)
    return entities


def load_news(path: Path) -> dict[str, MindNews]:
    """Parse a news.tsv into news_id -> MindNews."""
    news: dict[str, MindNews] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            news_id, category, subcategory, title, abstract = parts[:5]
            title_entities = _parse_entities(parts[6])
            abstract_entities = _parse_entities(parts[7])

            # Title entities first: the scorer truncates to the first five, and
            # title mentions are the more reliable topical signal.
            ordered = list(dict.fromkeys(title_entities + abstract_entities))

            news[news_id] = MindNews(
                news_id=news_id,
                category=category,
                subcategory=subcategory,
                title=title,
                abstract=abstract,
                entities=tuple(ordered[:MAX_ENTITIES]),
            )
    return news


def load_behaviors(path: Path, limit: Optional[int] = None) -> Iterator[MindImpression]:
    """Stream a behaviors.tsv, yielding one MindImpression per row.

    Rows whose impression column is empty are skipped; rows with an empty click
    history are kept, since cold-start users are part of the benchmark.
    """
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if limit is not None and row_index >= limit:
                return
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            impression_id, user_id, time_raw, history_raw, impressions_raw = parts[:5]
            if not impressions_raw.strip():
                continue

            candidates: list[str] = []
            labels: list[int] = []
            for token in impressions_raw.split():
                news_id, _, label = token.rpartition("-")
                if not news_id:
                    continue
                candidates.append(news_id)
                labels.append(1 if label == "1" else 0)

            if not candidates:
                continue

            yield MindImpression(
                impression_id=int(impression_id)
                if impression_id.isdigit()
                else row_index,
                user_id=user_id,
                timestamp=parse_impression_time(time_raw),
                history=tuple(history_raw.split()) if history_raw.strip() else (),
                candidates=tuple(candidates),
                labels=tuple(labels),
            )


def news_to_article(news: MindNews, article_id: int) -> Article:
    """Adapt a MindNews record to the Article shape the scorer expects.

    Mapping:
      category    -> Article.category      (MIND's top-level category)
      subcategory -> Article.topics        (the closest analogue to a topic tag)
      entities    -> Article.key_entities

    Left unset because MIND does not carry them: source_name (no publisher
    field; every URL is an MSN redirect), difficulty_level, and
    base_importance_score. Section stays None so the scorer skips its config
    lookup, and quality_weight stays at its neutral 1.0.
    """
    return Article(
        id=article_id,
        url=f"mind://{news.news_id}",
        content_hash=news.news_id,
        title=news.title,
        source_name="mind",
        source_type="mind",
        category=news.category,
        base_importance_score=None,
        topics_json=json.dumps([news.subcategory]),
        difficulty_level=None,
        key_entities_json=json.dumps(list(news.entities)),
        section=None,
    )


def build_user(
    user_id: str,
    history: tuple[str, ...],
    news: dict[str, MindNews],
    max_topics: int = 5,
) -> User:
    """Construct a User whose declared topics are inferred from click history.

    MIND has no signup questionnaire, so the analogue of "topics the user picked"
    is the subcategories they clicked most often. Role, level, and source
    preferences are left unset — see UNKNOWN_ROLE.
    """
    counts: Counter[str] = Counter()
    for news_id in history:
        item = news.get(news_id)
        if item is not None:
            counts[item.subcategory] += 1
    top_topics = [topic for topic, _ in counts.most_common(max_topics)]

    return User(
        id=1,
        email=f"{user_id}@mind.local",
        role=UNKNOWN_ROLE,
        level=UNKNOWN_LEVEL,
        topics_json=json.dumps(top_topics),
        source_preferences_json="{}",
    )


def train_click_counts(
    behaviors_path: Path,
    exclude_split: Optional[str] = None,
    restrict_to_period: Optional[str] = None,
) -> Counter:
    """Count clicks per news item across a split — the popularity baseline.

    Computed from the training split only; using dev clicks here would leak the
    labels being predicted.

    Two independent restrictions, each closing a different leak:

    `exclude_split` drops every impression belonging to that user bucket. It is
    required whenever the evaluation itself runs on a bucket of the train file:
    the held-out validation users are *inside* MINDsmall_train, so counting
    their clicks here would hand the popularity baseline the very labels it is
    being scored against.

    `restrict_to_period` keeps only the impressions on one side of the temporal
    holdout. It is required whenever the evaluation runs on the last train day,
    so that popularity is a day stale exactly as it is on dev. Without it the
    counts are contemporaneous with the slates they rank, and every
    recency-flavoured feature is overrated.

    Both unset is correct only when scoring dev.
    """
    # Imported here rather than at module scope because splits.py describes the
    # experimental protocol, while this module is the dataset adapter.
    from src.eval.splits import (
        SPLIT_NAMES,
        TEMPORAL_SPLIT_NAMES,
        split_for_user,
        temporal_split_for,
    )

    if exclude_split is not None and exclude_split not in SPLIT_NAMES:
        raise ValueError(
            f"unknown split {exclude_split!r}; expected one of {SPLIT_NAMES}"
        )
    if restrict_to_period is not None and restrict_to_period not in (
        TEMPORAL_SPLIT_NAMES
    ):
        raise ValueError(
            f"unknown temporal split {restrict_to_period!r}; "
            f"expected one of {TEMPORAL_SPLIT_NAMES}"
        )

    counts: Counter = Counter()
    for impression in load_behaviors(behaviors_path):
        if exclude_split is not None and (
            split_for_user(impression.user_id) == exclude_split
        ):
            continue
        if restrict_to_period is not None and (
            temporal_split_for(impression.timestamp) != restrict_to_period
        ):
            continue
        for news_id, label in zip(impression.candidates, impression.labels):
            if label == 1:
                counts[news_id] += 1
    return counts
