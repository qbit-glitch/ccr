"""OpenAI-compatible model client — works with vLLM, OpenRouter, etc."""

from __future__ import annotations

import logging
from typing import Any

import openai
from openai import OpenAI

from ccr.core.exceptions import (
    ModelAuthError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from ccr.core.types import TokenUsage, SessionUsage
from ccr.models.base import BaseLMClient
from ccr.models.retry import RetryConfig, retry_with_backoff
from ccr.utils.costs import calculate_cost

logger = logging.getLogger(__name__)

VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAICompatClient(BaseLMClient):
    """Client for OpenAI-compatible APIs (vLLM, OpenRouter, OpenAI, etc.)."""

    def __init__(
        self,
        model_name: str,
        base_url: str = VLLM_DEFAULT_BASE_URL,
        api_key: str | None = None,
        max_tokens: int = 8192,
        timeout: float = 300.0,
        retry_config: RetryConfig | None = None,
    ):
        super().__init__(model_name, timeout, retry_config=retry_config)
        self.max_tokens = max_tokens
        self.base_url = base_url

        # vLLM doesn't need an API key
        effective_key = api_key or "not-needed"

        self._client = OpenAI(
            api_key=effective_key,
            base_url=base_url,
            timeout=timeout,
        )
        self._usage = SessionUsage()
        self._last_usage = TokenUsage()

    def completion(self, messages: list[dict] | str, **kwargs) -> str:
        return retry_with_backoff(
            self._completion_inner, messages, config=self.retry_config, **kwargs
        )

    def _completion_inner(self, messages: list[dict] | str, **kwargs) -> str:
        """Single-attempt completion (retry wrapper calls this)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        model = kwargs.pop("model", self.model_name)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
        except openai.AuthenticationError as e:
            raise ModelAuthError(str(e), model=model) from e
        except openai.RateLimitError as e:
            raise ModelRateLimitError(str(e), model=model) from e
        except openai.APITimeoutError as e:
            raise ModelTimeoutError(str(e), model=model) from e
        except openai.APIConnectionError as e:
            raise ModelConnectionError(str(e), model=model) from e
        except openai.APIStatusError as e:
            raise ModelError(str(e), model=model) from e

        choice = response.choices[0]
        text = choice.message.content or ""

        # Track usage
        inp = response.usage.prompt_tokens if response.usage else 0
        out = response.usage.completion_tokens if response.usage else 0
        cost = calculate_cost(model, inp, out)
        self._last_usage = TokenUsage(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            cost_usd=cost,
        )
        self._usage.record(model, inp, out, cost)

        return text

    async def acompletion(self, messages: list[dict] | str, **kwargs) -> str:
        return self.completion(messages, **kwargs)

    def get_usage_summary(self) -> SessionUsage:
        return self._usage

    def get_last_usage(self) -> TokenUsage:
        return self._last_usage

    def check_connectivity(self) -> tuple[bool, str]:
        """Check if the OpenAI-compatible endpoint is reachable."""
        try:
            import httpx
            # vLLM/OpenAI-compat servers expose GET /v1/models
            base = self.base_url.rstrip("/")
            resp = httpx.get(f"{base}/models", timeout=5.0)
            if resp.status_code == 200:
                return True, "connected"
            else:
                return True, f"reachable (HTTP {resp.status_code})"
        except Exception as e:
            return False, f"unreachable: {e}"
