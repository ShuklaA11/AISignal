"""LLMProvider must send the key belonging to the active provider.

The original code picked `anthropic_api_key or openai_api_key`, i.e. by
precedence rather than by provider. With both keys configured — which is the
normal state while migrating between providers — an sk-ant-... key was sent
to OpenAI and rejected as invalid.
"""

from unittest.mock import MagicMock

from src.llm.provider import LLMProvider

ANTHROPIC_KEY = "sk-ant-test-key"
OPENAI_KEY = "sk-proj-test-key"


def _settings(provider: str, model: str = "some-model", **overrides) -> MagicMock:
    """Settings with BOTH keys populated — the migration state that broke."""
    s = MagicMock()
    s.llm.provider = provider
    s.llm.model = model
    s.llm.fallbacks = []
    s.llm.temperature = 0.3
    s.llm.max_tokens_per_article = 500
    s.anthropic_api_key = overrides.get("anthropic_api_key", ANTHROPIC_KEY)
    s.openai_api_key = overrides.get("openai_api_key", OPENAI_KEY)
    return s


def test_openai_provider_uses_openai_key():
    provider = LLMProvider(_settings("openai", "gpt-5.4-mini"))
    assert provider.api_key == OPENAI_KEY


def test_anthropic_provider_uses_anthropic_key():
    provider = LLMProvider(_settings("anthropic", "claude-haiku-4-5-20251001"))
    assert provider.api_key == ANTHROPIC_KEY


def test_ollama_provider_uses_no_key():
    """Local models need no credential, and must not be handed a cloud key."""
    provider = LLMProvider(_settings("ollama", "qwen3.5:4b"))
    assert provider.api_key is None


def test_unknown_provider_uses_no_key():
    provider = LLMProvider(_settings("mystery-llm"))
    assert provider.api_key is None


def test_missing_key_for_selected_provider_is_none():
    """An absent key must not silently fall back to another provider's key."""
    provider = LLMProvider(_settings("openai", openai_api_key=""))
    assert provider.api_key is None


def test_override_model_reselects_the_key():
    """Runtime switching must move the credential too, not just the model id."""
    provider = LLMProvider(_settings("anthropic", "claude-haiku-4-5-20251001"))
    assert provider.api_key == ANTHROPIC_KEY

    provider.override_model("openai", "gpt-5.4-mini")

    assert provider.model == "openai/gpt-5.4-mini"
    assert provider.api_key == OPENAI_KEY


def test_override_model_to_ollama_clears_the_key():
    provider = LLMProvider(_settings("openai", "gpt-5.4-mini"))

    provider.override_model("ollama", "qwen3.5:4b")

    assert provider.api_key is None
