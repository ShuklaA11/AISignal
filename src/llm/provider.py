from __future__ import annotations

import asyncio
import logging

import httpx
import litellm

from src.config import Settings

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# Timeout per LLM call (seconds). Qwen 4B should respond in <60s for a 5-article batch.
# 120s gives headroom without blocking the pipeline for 5 minutes on a hung connection.
DEFAULT_TIMEOUT = 120

# Ollama health-check endpoint
OLLAMA_BASE_URL = "http://localhost:11434"


class LLMProvider:
    """LiteLLM-based LLM client with provider fallback and timeout handling.

    Supports runtime model switching via ``override_model(provider, model)``.
    For Ollama models, automatically passes ``format="json"`` so the response
    is guaranteed valid JSON (no markdown fences, no trailing text).
    """

    def __init__(self, settings: Settings):
        self._provider = settings.llm.provider
        self._model_name = settings.llm.model
        self.model = f"{self._provider}/{self._model_name}"
        self.fallbacks = [
            f"{fb.provider}/{fb.model}" for fb in settings.llm.fallbacks
        ]
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens_per_article
        self.api_key = settings.anthropic_api_key or settings.openai_api_key or None

    # ------------------------------------------------------------------
    # Runtime model switching
    # ------------------------------------------------------------------
    def override_model(self, provider: str, model: str) -> None:
        """Switch the active model at runtime (e.g. from an admin endpoint)."""
        self._provider = provider
        self._model_name = model
        self.model = f"{provider}/{model}"
        logger.info(f"LLM model switched to {self.model}")

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------
    # Health check (Ollama-specific)
    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        """Return True if the configured LLM backend is reachable.

        For Ollama: pings the /api/tags endpoint with a 5-second timeout.
        For cloud providers: always returns True (LiteLLM handles retries).
        """
        if self._provider != "ollama":
            return True
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    async def generate(self, prompt: str, system: str = "", max_tokens: int | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build kwargs — Ollama models get think=False to disable thinking-mode
        # models (e.g. Qwen 3.5) which otherwise put all output in a hidden
        # "thinking" field.  We intentionally omit format="json" because it
        # forces Ollama to stop after a single JSON object, breaking batch
        # prompts that expect a JSON *array*.  parse_batch_response() already
        # handles markdown fences and raw JSON.
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "fallbacks": self.fallbacks,
            "api_key": self.api_key,
        }
        if self._provider == "ollama":
            kwargs["think"] = False

        try:
            response = await asyncio.wait_for(
                litellm.acompletion(**kwargs),
                timeout=DEFAULT_TIMEOUT,
            )
            return response.choices[0].message.content or ""
        except asyncio.TimeoutError:
            logger.error(f"LLM generation timed out after {DEFAULT_TIMEOUT}s for model {self.model}")
            raise
        except Exception as e:
            logger.error(f"LLM generation failed for {self.model}: {e}")
            raise
