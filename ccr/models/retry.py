"""Retry logic with exponential backoff for LLM API calls.

Retries on transient errors (timeout, rate limit, connection).
Never retries on auth errors or unknown failures.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from ccr.core.exceptions import (
    ModelAuthError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exception types that are safe to retry
RETRYABLE_ERRORS = (ModelTimeoutError, ModelRateLimitError, ModelConnectionError)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # cap on backoff
    exponential_base: float = 2.0
    jitter: bool = True  # add randomness to prevent thundering herd


def retry_with_backoff(
    fn: Callable[..., T],
    *args,
    config: RetryConfig | None = None,
    **kwargs,
) -> T:
    """Call fn with retry + exponential backoff on transient errors.

    Args:
        fn: The function to call.
        *args: Positional arguments for fn.
        config: Retry configuration. Uses defaults if None.
        **kwargs: Keyword arguments for fn.

    Returns:
        The return value of fn.

    Raises:
        ModelAuthError: Immediately, never retried.
        ModelError: After all retries exhausted.
    """
    cfg = config or RetryConfig()
    last_error: ModelError | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ModelAuthError:
            raise  # Never retry auth errors
        except RETRYABLE_ERRORS as e:
            last_error = e

            if attempt >= cfg.max_retries:
                fn_name = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
                logger.error(
                    f"All {cfg.max_retries} retries exhausted for {fn_name}: {e}"
                )
                raise

            # Calculate delay
            delay = _calculate_delay(e, attempt, cfg)

            fn_name = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
            logger.warning(
                f"Retry {attempt + 1}/{cfg.max_retries} for {fn_name} "
                f"after {type(e).__name__} (waiting {delay:.1f}s): {e}"
            )
            time.sleep(delay)
        except ModelError:
            # Non-retryable model errors (e.g. 400 bad request)
            raise

    # Should never reach here, but just in case
    raise last_error  # type: ignore[misc]


def _calculate_delay(error: ModelError, attempt: int, cfg: RetryConfig) -> float:
    """Calculate backoff delay, respecting rate limit retry-after headers."""
    # If rate limited with a retry-after header, use that
    if isinstance(error, ModelRateLimitError) and error.retry_after:
        delay = error.retry_after
    else:
        # Exponential backoff: base * exponential_base^attempt
        delay = cfg.base_delay * (cfg.exponential_base ** attempt)

    # Cap at max_delay
    delay = min(delay, cfg.max_delay)

    # Add jitter (0.5x to 1.5x) to prevent thundering herd
    if cfg.jitter:
        delay *= 0.5 + random.random()

    return delay
