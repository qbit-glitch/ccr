"""Anthropic/Claude model client."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

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


class ClaudeClient(BaseLMClient):
    """Client for Anthropic Claude models."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-sonnet-4-5-20250514",
        max_tokens: int = 16384,
        base_url: str | None = None,
        timeout: float = 300.0,
        retry_config: RetryConfig | None = None,
    ):
        super().__init__(model_name, timeout, retry_config=retry_config)
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=api_key,
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

        system = kwargs.pop("system", None)
        model = kwargs.pop("model", self.model_name)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        # Extract system from messages if present
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", system)
            else:
                filtered_messages.append(msg)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": filtered_messages,
        }
        if system:
            create_kwargs["system"] = system

        try:
            response = self._client.messages.create(**create_kwargs)
        except anthropic.AuthenticationError as e:
            raise ModelAuthError(str(e), model=model) from e
        except anthropic.RateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after_str = e.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass
            raise ModelRateLimitError(str(e), model=model, retry_after=retry_after) from e
        except anthropic.APITimeoutError as e:
            raise ModelTimeoutError(str(e), model=model) from e
        except anthropic.APIConnectionError as e:
            raise ModelConnectionError(str(e), model=model) from e
        except anthropic.APIStatusError as e:
            raise ModelError(str(e), model=model) from e

        # Track usage
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        cost = calculate_cost(model, inp, out)
        self._last_usage = TokenUsage(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            cost_usd=cost,
        )
        self._usage.record(model, inp, out, cost)

        return response.content[0].text

    async def acompletion(self, messages: list[dict] | str, **kwargs) -> str:
        # For now, delegate to sync — can be made truly async later
        return self.completion(messages, **kwargs)

    def get_usage_summary(self) -> SessionUsage:
        return self._usage

    def get_last_usage(self) -> TokenUsage:
        return self._last_usage

    def check_connectivity(self) -> tuple[bool, str]:
        """Check if Anthropic API is reachable by counting available models."""
        try:
            import httpx
            base = str(self._client.base_url).rstrip("/")
            resp = httpx.get(
                f"{base}/v1/models",
                headers={
                    "x-api-key": self._client.api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                return True, "connected"
            elif resp.status_code == 401:
                return False, "invalid API key"
            else:
                return True, f"reachable (HTTP {resp.status_code})"
        except Exception as e:
            return False, f"unreachable: {e}"

    def forward_raw(self, body: bytes) -> bytes:
        """Forward a raw Anthropic API request body and return raw response bytes.

        Used for passthrough mode when CCR cannot process the request.
        """
        import httpx

        resp = httpx.post(
            f"{self._client.base_url}/v1/messages",
            content=body,
            headers={
                "x-api-key": self._client.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=self.timeout,
        )
        return resp.content
