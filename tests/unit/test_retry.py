"""Tests for retry logic with exponential backoff."""

import time
from unittest.mock import MagicMock, patch

import pytest

from ccr.core.exceptions import (
    ModelAuthError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from ccr.models.retry import (
    RETRYABLE_ERRORS,
    RetryConfig,
    _calculate_delay,
    retry_with_backoff,
)


# --- RetryConfig tests ---


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.exponential_base == 2.0
        assert cfg.jitter is True

    def test_custom(self):
        cfg = RetryConfig(max_retries=5, base_delay=0.5, max_delay=30.0, jitter=False)
        assert cfg.max_retries == 5
        assert cfg.base_delay == 0.5
        assert cfg.max_delay == 30.0
        assert cfg.jitter is False


# --- Delay calculation tests ---


class TestCalculateDelay:
    def test_exponential_backoff_no_jitter(self):
        cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False, max_delay=60.0)
        err = ModelTimeoutError("timeout")
        assert _calculate_delay(err, 0, cfg) == 1.0   # 1 * 2^0
        assert _calculate_delay(err, 1, cfg) == 2.0   # 1 * 2^1
        assert _calculate_delay(err, 2, cfg) == 4.0   # 1 * 2^2
        assert _calculate_delay(err, 3, cfg) == 8.0   # 1 * 2^3

    def test_max_delay_cap(self):
        cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False, max_delay=5.0)
        err = ModelTimeoutError("timeout")
        assert _calculate_delay(err, 10, cfg) == 5.0  # capped at max_delay

    def test_rate_limit_retry_after_honored(self):
        cfg = RetryConfig(base_delay=1.0, jitter=False, max_delay=60.0)
        err = ModelRateLimitError("429", retry_after=30.0)
        assert _calculate_delay(err, 0, cfg) == 30.0  # uses retry-after

    def test_rate_limit_retry_after_capped(self):
        cfg = RetryConfig(base_delay=1.0, jitter=False, max_delay=10.0)
        err = ModelRateLimitError("429", retry_after=30.0)
        assert _calculate_delay(err, 0, cfg) == 10.0  # capped at max_delay

    def test_jitter_adds_randomness(self):
        cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=True, max_delay=60.0)
        err = ModelTimeoutError("timeout")
        delays = {_calculate_delay(err, 1, cfg) for _ in range(50)}
        # With jitter, delays should vary (0.5x to 1.5x of 2.0 = 1.0 to 3.0)
        assert len(delays) > 1, "Jitter should produce different delays"
        assert all(0.5 <= d <= 4.0 for d in delays), f"Unexpected delays: {delays}"


# --- retry_with_backoff tests ---


class TestRetryWithBackoff:
    def test_success_on_first_attempt(self):
        fn = MagicMock(return_value="ok")
        result = retry_with_backoff(fn, config=RetryConfig())
        assert result == "ok"
        assert fn.call_count == 1

    def test_success_after_transient_failures(self):
        """Recovers after timeout then succeeds."""
        fn = MagicMock(
            side_effect=[
                ModelTimeoutError("timeout 1"),
                ModelTimeoutError("timeout 2"),
                "success",
            ]
        )
        cfg = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = retry_with_backoff(fn, config=cfg)
        assert result == "success"
        assert fn.call_count == 3

    def test_auth_error_never_retried(self):
        """Auth errors propagate immediately."""
        fn = MagicMock(side_effect=ModelAuthError("bad key"))
        cfg = RetryConfig(max_retries=3, base_delay=0.01)

        with pytest.raises(ModelAuthError):
            retry_with_backoff(fn, config=cfg)

        assert fn.call_count == 1  # No retries

    def test_exhausted_retries_raises_last_error(self):
        """After max retries, the last error is raised."""
        fn = MagicMock(side_effect=ModelConnectionError("refused"))
        cfg = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)

        with pytest.raises(ModelConnectionError):
            retry_with_backoff(fn, config=cfg)

        assert fn.call_count == 3  # initial + 2 retries

    def test_non_retryable_model_error_not_retried(self):
        """Generic ModelError (e.g. 400) is not retried."""
        fn = MagicMock(side_effect=ModelError("bad request", model="test"))
        cfg = RetryConfig(max_retries=3, base_delay=0.01)

        with pytest.raises(ModelError):
            retry_with_backoff(fn, config=cfg)

        assert fn.call_count == 1

    def test_rate_limit_retried(self):
        fn = MagicMock(
            side_effect=[
                ModelRateLimitError("429", retry_after=0.01),
                "ok",
            ]
        )
        cfg = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = retry_with_backoff(fn, config=cfg)
        assert result == "ok"
        assert fn.call_count == 2

    def test_connection_error_retried(self):
        fn = MagicMock(
            side_effect=[
                ModelConnectionError("refused"),
                "connected",
            ]
        )
        cfg = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = retry_with_backoff(fn, config=cfg)
        assert result == "connected"

    def test_mixed_transient_errors(self):
        """Different transient error types across retries."""
        fn = MagicMock(
            side_effect=[
                ModelTimeoutError("timeout"),
                ModelConnectionError("refused"),
                ModelRateLimitError("429"),
                "finally ok",
            ]
        )
        cfg = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = retry_with_backoff(fn, config=cfg)
        assert result == "finally ok"
        assert fn.call_count == 4

    def test_zero_retries_means_single_attempt(self):
        fn = MagicMock(side_effect=ModelTimeoutError("timeout"))
        cfg = RetryConfig(max_retries=0, base_delay=0.01)

        with pytest.raises(ModelTimeoutError):
            retry_with_backoff(fn, config=cfg)

        assert fn.call_count == 1

    def test_passes_args_and_kwargs(self):
        fn = MagicMock(return_value="ok")
        retry_with_backoff(fn, "arg1", "arg2", config=RetryConfig(), key="val")
        fn.assert_called_once_with("arg1", "arg2", key="val")

    def test_actual_delay_occurs(self):
        """Verify actual time passes between retries."""
        fn = MagicMock(
            side_effect=[ModelTimeoutError("timeout"), "ok"]
        )
        cfg = RetryConfig(max_retries=1, base_delay=0.05, jitter=False)

        start = time.monotonic()
        retry_with_backoff(fn, config=cfg)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.04, f"Expected >=0.04s delay, got {elapsed:.3f}s"


# --- Integration: retry in model clients ---


class TestClaudeClientRetry:
    def test_client_uses_retry_config(self):
        """ClaudeClient passes retry_config to base class."""
        from ccr.models.anthropic_client import ClaudeClient

        cfg = RetryConfig(max_retries=5)
        client = ClaudeClient(api_key="key", retry_config=cfg)
        assert client.retry_config.max_retries == 5

    def test_client_default_retry(self):
        """ClaudeClient gets default RetryConfig if none provided."""
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(api_key="key")
        assert client.retry_config.max_retries == 3

    def test_client_retries_on_timeout(self):
        """Full integration: client retries timeout then succeeds."""
        import anthropic
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(
            api_key="key",
            model_name="test",
            retry_config=RetryConfig(max_retries=2, base_delay=0.01, jitter=False),
        )

        mock_response = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.content = [MagicMock(text="hello")]

        client._client.messages.create = MagicMock(
            side_effect=[
                anthropic.APITimeoutError(request=MagicMock()),
                mock_response,
            ]
        )

        result = client.completion("hi")
        assert result == "hello"
        assert client._client.messages.create.call_count == 2


class TestOpenAIClientRetry:
    def test_client_uses_retry_config(self):
        from ccr.models.openai_compat import OpenAICompatClient

        cfg = RetryConfig(max_retries=5)
        client = OpenAICompatClient(model_name="test", retry_config=cfg)
        assert client.retry_config.max_retries == 5

    def test_client_retries_on_connection_error(self):
        """Full integration: client retries connection error then succeeds."""
        import openai
        from ccr.models.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(
            model_name="test",
            api_key="key",
            retry_config=RetryConfig(max_retries=2, base_delay=0.01, jitter=False),
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        client._client.chat.completions.create = MagicMock(
            side_effect=[
                openai.APIConnectionError(request=MagicMock()),
                mock_response,
            ]
        )

        result = client.completion("hi")
        assert result == "hello"
        assert client._client.chat.completions.create.call_count == 2

    def test_no_retry_disabled(self):
        """RetryConfig(max_retries=0) means no retries."""
        import openai
        from ccr.models.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(
            model_name="test",
            api_key="key",
            retry_config=RetryConfig(max_retries=0),
        )

        client._client.chat.completions.create = MagicMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )

        with pytest.raises(ModelTimeoutError):
            client.completion("hi")

        assert client._client.chat.completions.create.call_count == 1


# --- RETRYABLE_ERRORS tuple ---


class TestRetryableErrors:
    def test_retryable_errors_are_correct(self):
        assert ModelTimeoutError in RETRYABLE_ERRORS
        assert ModelRateLimitError in RETRYABLE_ERRORS
        assert ModelConnectionError in RETRYABLE_ERRORS

    def test_non_retryable_errors(self):
        assert ModelAuthError not in RETRYABLE_ERRORS
        assert ModelError not in RETRYABLE_ERRORS
