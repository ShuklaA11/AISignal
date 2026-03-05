from __future__ import annotations

import asyncio
import logging

import litellm

from src.config import Settings

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LLMProvider:
    """LiteLLM-based LLM client with provider fallback and timeout handling."""

    def __init__(self, settings: Settings):
        self.model = f"{settings.llm.provider}/{settings.llm.model}"
        self.fallbacks = [
            f"{fb.provider}/{fb.model}" for fb in settings.llm.fallbacks
        ]
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens_per_article

        self.api_key = settings.anthropic_api_key or settings.openai_api_key or None

    async def generate(self, prompt: str, system: str = "", max_tokens: int | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    fallbacks=self.fallbacks,
                    api_key=self.api_key,
                ),
                timeout=300,
            )
            return response.choices[0].message.content or ""
        except asyncio.TimeoutError:
            logger.error(f"LLM generation timed out after 300s for model {self.model}")
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
