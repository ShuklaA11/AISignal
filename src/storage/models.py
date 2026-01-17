import json
from datetime import date as date_type
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
    """Return current UTC time as a naive datetime (for SQLite compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(SQLModel, table=True):
    """A registered user with personalization preferences and auth credentials."""
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: str = ""
    password_hash: str = ""
    role: str = Field(default="enthusiast")  # student | industry | enthusiast
    level: str = Field(default="intermediate")  # beginner | intermediate | advanced
    topics_json: str = Field(default="[]")
    source_preferences_json: str = Field(default="{}")
    is_admin: bool = Field(default=False)
    active: bool = Field(default=True)
    email_verified: bool = Field(default=False)
    email_verified_at: Optional[datetime] = None
    session_version: int = Field(default=0)
    created_at: datetime = Field(default_factory=utcnow)

    digests: List["Digest"] = Relationship(back_populates="user")
    saved_articles: List["SavedArticle"] = Relationship(back_populates="user")
    read_articles: List["ReadArticle"] = Relationship(back_populates="user")
    ml_profile: Optional["UserMLProfile"] = Relationship(back_populates="user")

    @property
    def topics(self) -> list[str]:
        return json.loads(self.topics_json)

    @topics.setter
    def topics(self, value: list[str]) -> None:
        self.topics_json = json.dumps(value)

    @property
    def source_preferences(self) -> dict[str, int]:
        return json.loads(self.source_preferences_json)

    @source_preferences.setter
    def source_preferences(self, value: dict[str, int]) -> None:
        self.source_preferences_json = json.dumps(value)


class Article(SQLModel, table=True):
    """A fetched article with metadata, LLM-assigned topics, and importance score."""
    __tablename__ = "articles"

    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(unique=True, index=True)
    content_hash: str = Field(index=True)
    title: str
    author: Optional[str] = None
    source_name: str = Field(index=True)
    source_type: str  # rss | api | scrape
    original_content: Optional[str] = None
    published_at: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=utcnow)

    category: Optional[str] = None
    base_importance_score: Optional[float] = None
    topics_json: str = Field(default="[]")
    difficulty_level: Optional[str] = None
    key_entities_json: str = Field(default="[]")
    status: str = Field(default="raw", index=True)
    extra_metadata_json: str = Field(default="{}")

    summaries: List["ArticleSummary"] = Relationship(back_populates="article")
    digest_links: List["DigestArticle"] = Relationship(back_populates="article")

    @property
    def topics(self) -> list[str]:
        return json.loads(self.topics_json)

    @topics.setter
    def topics(self, value: list[str]) -> None:
        self.topics_json = json.dumps(value)

    @property
    def key_entities(self) -> list[str]:
        return json.loads(self.key_entities_json)

    @key_entities.setter
    def key_entities(self, value: list[str]) -> None:
        self.key_entities_json = json.dumps(value)

    @property
    def extra_metadata(self) -> dict:
        return json.loads(self.extra_metadata_json)

    @extra_metadata.setter
    def extra_metadata(self, value: dict) -> None:
        self.extra_metadata_json = json.dumps(value)


class ArticleSummary(SQLModel, table=True):
    """LLM-generated summary variant for a specific role and difficulty level."""
    __tablename__ = "article_summaries"

    id: Optional[int] = Field(default=None, primary_key=True)
    article_id: int = Field(foreign_key="articles.id", index=True)
    role: str  # student | industry | enthusiast
    level: str  # beginner | intermediate | advanced
    summary_text: str

    article: Article = Relationship(back_populates="summaries")


class ArticleEmbedding(SQLModel, table=True):
    """Stored embedding vector for an article, used for semantic similarity scoring."""
    __tablename__ = "article_embeddings"

    id: Optional[int] = Field(default=None, primary_key=True)
    article_id: int = Field(foreign_key="articles.id", unique=True, index=True)
    embedding_blob: bytes  # numpy float32 array as .tobytes()
    embedding_dim: int = 1024
    model_name: str = "mxbai-embed-large"
    created_at: datetime = Field(default_factory=utcnow)


class UserEmbeddingModel(SQLModel, table=True):
    """Stores per-user trained embedding model weights."""
    __tablename__ = "user_embedding_models"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", unique=True, index=True)
    model_weights_blob: bytes
    training_loss: float = 0.0
    num_training_samples: int = 0
    trained_at: datetime = Field(default_factory=utcnow)


class DigestArticle(SQLModel, table=True):
    """Join table linking articles to digests with personalized score and display order."""
    __tablename__ = "digest_articles"

    id: Optional[int] = Field(default=None, primary_key=True)
    digest_id: int = Field(foreign_key="digests.id", index=True)
    article_id: int = Field(foreign_key="articles.id", index=True)
    personalized_score: float = 0.0
    display_order: int = 0
    approved: bool = Field(default=True)

    digest: "Digest" = Relationship(back_populates="article_links")
    article: Article = Relationship(back_populates="digest_links")


class Digest(SQLModel, table=True):
    """A personalized email digest sent to a user on a specific date."""
    __tablename__ = "digests"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    digest_date: date_type = Field(index=True)
    status: str = Field(default="draft")
    trigger: str = Field(default="scheduled")  # "scheduled" or "manual"
    subject_line: Optional[str] = None
    sent_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)

    user: User = Relationship(back_populates="digests")
    article_links: List[DigestArticle] = Relationship(back_populates="digest")


class SavedArticle(SQLModel, table=True):
    """Record of a user saving an article for later reading."""
    __tablename__ = "saved_articles"
    __table_args__ = (UniqueConstraint("user_id", "article_id"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    article_id: int = Field(foreign_key="articles.id", index=True)
    saved_at: datetime = Field(default_factory=utcnow)

    user: "User" = Relationship(back_populates="saved_articles")
    article: Article = Relationship()


class ReadArticle(SQLModel, table=True):
    """Record of a user reading (clicking through to) an article."""
    __tablename__ = "read_articles"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    article_id: int = Field(foreign_key="articles.id", index=True)
    read_at: datetime = Field(default_factory=utcnow)

    user: "User" = Relationship(back_populates="read_articles")
    article: Article = Relationship()


class FeedImpression(SQLModel, table=True):
    """Tracks each time an article is shown to a user in the feed, with engagement signals."""
    __tablename__ = "feed_impressions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    article_id: int = Field(foreign_key="articles.id", index=True)
    shown_at: datetime = Field(default_factory=utcnow)
    position: int = 0
    clicked: bool = Field(default=False)
    saved: bool = Field(default=False)
    liked: bool = Field(default=False)
    disliked: bool = Field(default=False)
    processed: bool = Field(default=False)
    feed_group: str = ""
    feed_view: str = ""  # "for_you" or "all"


class UserMLProfile(SQLModel, table=True):
    """Learned per-user weights for personalized article scoring, updated via online learning."""
    __tablename__ = "user_ml_profiles"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", unique=True, index=True)

    # Learned weights (JSON dicts: feature_value -> weight_float)
    source_weights_json: str = Field(default="{}")
    category_weights_json: str = Field(default="{}")
    topic_weights_json: str = Field(default="{}")
    difficulty_weights_json: str = Field(default="{}")
    entity_weights_json: str = Field(default="{}")

    # Per-feature signal counts for confidence-aware decay
    signal_counts_json: str = Field(default="{}")

    # Counters
    total_impressions: int = Field(default=0)
    total_clicks: int = Field(default=0)
    total_saves: int = Field(default=0)
    alpha: float = Field(default=1.0)  # 1.0 = 100% rule-based, 0.3 = 70% learned
    learning_rate_override: Optional[float] = Field(default=None)  # Set by metrics adaptation; None = use interaction-based LR
    updated_at: datetime = Field(default_factory=utcnow)

    user: "User" = Relationship(back_populates="ml_profile")

    @property
    def source_weights(self) -> dict[str, float]:
        return json.loads(self.source_weights_json)

    @source_weights.setter
    def source_weights(self, value: dict[str, float]) -> None:
        self.source_weights_json = json.dumps(value)

    @property
    def category_weights(self) -> dict[str, float]:
        return json.loads(self.category_weights_json)

    @category_weights.setter
    def category_weights(self, value: dict[str, float]) -> None:
        self.category_weights_json = json.dumps(value)

    @property
    def topic_weights(self) -> dict[str, float]:
        return json.loads(self.topic_weights_json)

    @topic_weights.setter
    def topic_weights(self, value: dict[str, float]) -> None:
        self.topic_weights_json = json.dumps(value)

    @property
    def difficulty_weights(self) -> dict[str, float]:
        return json.loads(self.difficulty_weights_json)

    @difficulty_weights.setter
    def difficulty_weights(self, value: dict[str, float]) -> None:
        self.difficulty_weights_json = json.dumps(value)

    @property
    def entity_weights(self) -> dict[str, float]:
        return json.loads(self.entity_weights_json)

    @entity_weights.setter
    def entity_weights(self, value: dict[str, float]) -> None:
        self.entity_weights_json = json.dumps(value)

    @property
    def signal_counts(self) -> dict[str, int]:
        return json.loads(self.signal_counts_json)

    @signal_counts.setter
    def signal_counts(self, value: dict[str, int]) -> None:
        self.signal_counts_json = json.dumps(value)


class FetchRun(SQLModel, table=True):
    """Per-source metrics from each fetch run."""
    __tablename__ = "fetch_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    source_name: str = Field(index=True)
    fetched_at: datetime = Field(default_factory=utcnow, index=True)
    articles_fetched: int = 0
    articles_new: int = 0
    duration_ms: int = 0
    error: Optional[str] = None
    status: str = "ok"  # ok | error | empty


class Source(SQLModel, table=True):
    """Configured data source with fetch schedule and connection settings."""
    __tablename__ = "sources"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    source_type: str
    url: Optional[str] = None
    config_json: str = Field(default="{}")
    enabled: bool = Field(default=True)
    last_fetched_at: Optional[datetime] = None
    fetch_interval_minutes: int = Field(default=360)


class ScoringMetric(SQLModel, table=True):
    """Daily computed evaluation metrics (CTR, nDCG, save rate) for a user."""
    __tablename__ = "scoring_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    metric_date: date_type = Field(index=True)
    ctr: float = 0.0
    save_rate: float = 0.0
    ndcg_at_10: float = 0.0
    personalization_lift: float = 1.0
    total_impressions: int = 0
    total_clicks: int = 0
    total_saves: int = 0
    computed_at: datetime = Field(default_factory=utcnow)


class DigestClick(SQLModel, table=True):
    """Tracks clicks on article links within email digests."""
    __tablename__ = "digest_clicks"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    article_id: int = Field(foreign_key="articles.id", index=True)
    digest_id: int = Field(foreign_key="digests.id", index=True)
    clicked_at: datetime = Field(default_factory=utcnow)
    section: str = Field(default="main")  # "main" or "explore"


class Token(SQLModel, table=True):
    """One-time-use token for email verification and password reset."""
    __tablename__ = "tokens"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    token_hash: str = Field(unique=True, index=True)
    token_type: str = Field(index=True)  # email_verification | password_reset
    expires_at: datetime = Field(index=True)
    created_at: datetime = Field(default_factory=utcnow)
    used_at: Optional[datetime] = None
