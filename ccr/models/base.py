"""Base LM client interface — adapted from RLM clients/base_lm.py."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ccr.core.types import TokenUsage, SessionUsage
from ccr.models.retry import RetryConfig


class BaseLMClient(ABC):
    """Abstract base for all model clients."""

    def __init__(
        self,
        model_name: str,
        timeout: float = 300.0,
        retry_config: RetryConfig | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()

    @abstractmethod
    def completion(self, messages: list[dict] | str, **kwargs) -> str:
        """Synchronous completion. Returns response text."""

    @abstractmethod
    async def acompletion(self, messages: list[dict] | str, **kwargs) -> str:
        """Async completion. Returns response text."""

    @abstractmethod
    def get_usage_summary(self) -> SessionUsage:
        """Aggregate usage across all calls."""

    @abstractmethod
    def get_last_usage(self) -> TokenUsage:
        """Usage from the most recent call."""

    def check_connectivity(self) -> tuple[bool, str]:
        """Check if the model endpoint is reachable.

        Returns:
            Tuple of (is_reachable, detail_message).
        """
        return False, "not implemented"
