from __future__ import annotations

import os
import warnings
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


def _load_yaml() -> dict:
    config_path = CONFIG_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _env_overrides(prefix: str, fields: set[str]) -> dict:
    """Collect env-var overrides for a nested model.

    Looks for ``NEWSLETTER_{PREFIX}__{FIELD}`` env vars and returns a
    dict of matching field→value pairs to merge on top of YAML config.
    """
    result: dict[str, str] = {}
    for field in fields:
        key = f"NEWSLETTER_{prefix}__{field}".upper()
        val = os.environ.get(key)
        if val is not None:
            result[field] = val
    return result


_yaml_config = _load_yaml()


class LLMFallback(BaseModel):
    provider: str
    model: str


class LLMSettings(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    fallbacks: list[LLMFallback] = Field(default_factory=list)
    max_tokens_per_article: int = 500
    max_daily_cost_usd: float = 2.0
    temperature: float = 0.3
    batch_size: int = 10
    concurrent_requests: int = 3


class EmailSettings(BaseModel):
    provider: str = "resend"
    from_address: str = "AI Digest <digest@yourdomain.com>"
    resend_api_key: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""  # Gmail App Password


class ScheduleSettings(BaseModel):
    fetch_hour: int = 6
    fetch_minute: int = 0
    process_hour: int = 6
    process_minute: int = 30
    auto_send_hour: int = 9   # 9:00 AM CST (CronTrigger uses system timezone)
    auto_send_minute: int = 0


class RSSFeed(BaseModel):
    name: str
    url: str


class Settings(BaseSettings):
    model_config = {"env_prefix": "NEWSLETTER_"}

    secret_key: str = ""
    base_url: str = ""  # Set via NEWSLETTER_BASE_URL for production email links
    database_url: str = Field(
        default_factory=lambda: _yaml_config.get("database", {}).get(
            "url", f"sqlite:///{DATA_DIR / 'newsletter.db'}"
        )
    )

    llm: LLMSettings = Field(default_factory=lambda: LLMSettings(
        **{**_yaml_config.get("llm", {}), **_env_overrides("LLM", set(LLMSettings.model_fields))}
    ))
    email: EmailSettings = Field(default_factory=lambda: EmailSettings(
        **{**_yaml_config.get("email", {}), **_env_overrides("EMAIL", set(EmailSettings.model_fields))}
    ))
    schedule: ScheduleSettings = Field(default_factory=lambda: ScheduleSettings(
        **{**_yaml_config.get("schedule", {}), **_env_overrides("SCHEDULE", set(ScheduleSettings.model_fields))}
    ))

    rss_feeds: list[RSSFeed] = Field(
        default_factory=lambda: [RSSFeed(**f) for f in _yaml_config.get("rss_feeds", [])]
    )

    arxiv_categories: list[str] = Field(
        default_factory=lambda: _yaml_config.get("arxiv", {}).get("categories", ["cs.AI", "cs.CL", "cs.CV", "cs.LG"])
    )
    arxiv_max_results: int = Field(
        default_factory=lambda: _yaml_config.get("arxiv", {}).get("max_results", 50)
    )

    reddit_subreddits: list[str] = Field(
        default_factory=lambda: _yaml_config.get("reddit", {}).get("subreddits", [])
    )
    reddit_min_score: int = Field(
        default_factory=lambda: _yaml_config.get("reddit", {}).get("min_score", 10)
    )

    twitter_query: str = Field(
        default_factory=lambda: _yaml_config.get("twitter", {}).get("query", "")
    )
    twitter_max_results: int = Field(
        default_factory=lambda: _yaml_config.get("twitter", {}).get("max_results", 50)
    )

    topics: dict[str, list[str]] = Field(
        default_factory=lambda: _yaml_config.get("topics", {})
    )

    role_defaults: dict[str, dict] = Field(
        default_factory=lambda: _yaml_config.get("role_defaults", {})
    )

    # API keys loaded from environment / .env
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    twitter_bearer_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "ai-newsletter/1.0"
    resend_api_key: str = ""


_cached_settings: Settings | None = None


def load_settings() -> Settings:
    """Load settings with .env file support. Cached after first call."""
    global _cached_settings
    if _cached_settings is not None:
        return _cached_settings

    from dotenv import load_dotenv

    env_path = CONFIG_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    s = Settings()

    _weak_keys = {"", "change-me-to-a-random-string", "changeme", "secret", "test"}
    if not s.secret_key or s.secret_key in _weak_keys or len(s.secret_key) < 16:
        raise RuntimeError(
            "NEWSLETTER_SECRET_KEY is missing, too short (min 16 chars), or a known placeholder. "
            "This is required for session security, signed tokens, and password reset links. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    # Validate LLM API keys based on provider
    provider = s.llm.provider.lower()
    if provider == "anthropic" and not s.anthropic_api_key:
        warnings.warn(
            "LLM provider is 'anthropic' but NEWSLETTER_ANTHROPIC_API_KEY is not set. "
            "LLM calls will fail at runtime.",
            stacklevel=2,
        )
    elif provider == "openai" and not s.openai_api_key:
        warnings.warn(
            "LLM provider is 'openai' but NEWSLETTER_OPENAI_API_KEY is not set. "
            "LLM calls will fail at runtime.",
            stacklevel=2,
        )

    _cached_settings = s
    return s
