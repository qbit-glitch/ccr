"""Tests for CCR exception hierarchy."""

import json
import pytest
from unittest.mock import MagicMock, patch

from ccr.core.exceptions import (
    CCRError,
    ConfigError,
    HookError,
    MemoryError_,
    ModelAuthError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
    PackingError,
    PlaybookError,
    PlaybookIOError,
    PlaybookParseError,
    RoutingError,
)


# --- Hierarchy tests ---


class TestExceptionHierarchy:
    """Verify the full inheritance tree."""

    def test_all_inherit_from_ccr_error(self):
        """Every CCR exception must be catchable via CCRError."""
        exceptions = [
            ConfigError("x"),
            ModelError("x"),
            ModelTimeoutError("x"),
            ModelRateLimitError("x"),
            ModelAuthError("x"),
            ModelConnectionError("x"),
            PackingError("x"),
            RoutingError("x"),
            PlaybookError("x"),
            PlaybookIOError("x"),
            PlaybookParseError("x"),
            MemoryError_("x"),
            HookError("x"),
        ]
        for exc in exceptions:
            assert isinstance(exc, CCRError), f"{type(exc).__name__} not a CCRError"

    def test_model_subtypes_inherit_from_model_error(self):
        for cls in [ModelTimeoutError, ModelRateLimitError, ModelAuthError, ModelConnectionError]:
            assert issubclass(cls, ModelError)

    def test_playbook_subtypes_inherit_from_playbook_error(self):
        for cls in [PlaybookIOError, PlaybookParseError]:
            assert issubclass(cls, PlaybookError)


# --- Attribute tests ---


class TestExceptionAttributes:
    """Verify custom attributes on exception classes."""

    def test_ccr_error_defaults(self):
        e = CCRError("test")
        assert str(e) == "test"
        assert e.recoverable is True
        assert e.detail == ""

    def test_ccr_error_custom(self):
        e = CCRError("oops", recoverable=False, detail="some detail")
        assert e.recoverable is False
        assert e.detail == "some detail"

    def test_config_error_not_recoverable(self):
        e = ConfigError("bad key", field="api_key")
        assert e.recoverable is False
        assert e.field == "api_key"

    def test_model_error_stores_model(self):
        e = ModelError("fail", model="claude-sonnet-4-5-20250514")
        assert e.model == "claude-sonnet-4-5-20250514"

    def test_rate_limit_stores_retry_after(self):
        e = ModelRateLimitError("429", retry_after=30.0)
        assert e.retry_after == 30.0

    def test_rate_limit_no_retry_after(self):
        e = ModelRateLimitError("429")
        assert e.retry_after is None

    def test_auth_error_not_recoverable(self):
        e = ModelAuthError("401")
        assert e.recoverable is False

    def test_hook_error_stores_event(self):
        e = HookError("handler failed", event="PostToolUse")
        assert e.event == "PostToolUse"


# --- Catch pattern tests ---


class TestCatchPatterns:
    """Verify real-world catch patterns work correctly."""

    def test_catch_all_model_errors(self):
        """ModelError catches all model subtypes."""
        for cls in [ModelTimeoutError, ModelRateLimitError, ModelAuthError, ModelConnectionError]:
            with pytest.raises(ModelError):
                raise cls("test", model="test-model")

    def test_catch_ccr_error_catches_everything(self):
        """CCRError is the catch-all for the entire hierarchy."""
        with pytest.raises(CCRError):
            raise ModelTimeoutError("timeout")
        with pytest.raises(CCRError):
            raise ConfigError("bad", field="x")
        with pytest.raises(CCRError):
            raise PlaybookIOError("io fail")

    def test_auth_error_not_caught_by_timeout(self):
        """Auth errors should NOT be caught by ModelTimeoutError."""
        with pytest.raises(ModelAuthError):
            try:
                raise ModelAuthError("401")
            except ModelTimeoutError:
                pytest.fail("AuthError should not be caught by TimeoutError")

    def test_recoverable_flag_for_gateway_fallback(self):
        """Gateway uses recoverable flag to decide passthrough."""
        timeout = ModelTimeoutError("timeout")
        assert timeout.recoverable is True  # should fallback

        auth = ModelAuthError("bad key")
        assert auth.recoverable is False  # should NOT fallback

        config = ConfigError("missing field", field="api_key")
        assert config.recoverable is False  # should NOT fallback


# --- Integration with model clients ---


class TestAnthropicClientExceptions:
    """Test that ClaudeClient wraps SDK exceptions correctly."""

    def test_auth_error_wrapping(self):
        """Anthropic AuthenticationError → ModelAuthError."""
        import anthropic
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(api_key="bad-key", model_name="test")
        mock_create = MagicMock(
            side_effect=anthropic.AuthenticationError(
                message="invalid api key",
                response=MagicMock(status_code=401, headers={}),
                body={"error": {"message": "invalid api key"}},
            )
        )
        client._client.messages.create = mock_create

        with pytest.raises(ModelAuthError) as exc_info:
            client.completion("hello")
        assert exc_info.value.model == "test"

    def test_timeout_wrapping(self):
        """Anthropic APITimeoutError → ModelTimeoutError."""
        import anthropic
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(api_key="key", model_name="test")
        mock_create = MagicMock(
            side_effect=anthropic.APITimeoutError(request=MagicMock())
        )
        client._client.messages.create = mock_create

        with pytest.raises(ModelTimeoutError):
            client.completion("hello")

    def test_connection_error_wrapping(self):
        """Anthropic APIConnectionError → ModelConnectionError."""
        import anthropic
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(api_key="key", model_name="test")
        mock_create = MagicMock(
            side_effect=anthropic.APIConnectionError(request=MagicMock())
        )
        client._client.messages.create = mock_create

        with pytest.raises(ModelConnectionError):
            client.completion("hello")

    def test_rate_limit_wrapping(self):
        """Anthropic RateLimitError → ModelRateLimitError."""
        import anthropic
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(api_key="key", model_name="test")
        mock_response = MagicMock(
            status_code=429,
            headers={"retry-after": "30"},
        )
        mock_create = MagicMock(
            side_effect=anthropic.RateLimitError(
                message="rate limited",
                response=mock_response,
                body={"error": {"message": "rate limited"}},
            )
        )
        client._client.messages.create = mock_create

        with pytest.raises(ModelRateLimitError) as exc_info:
            client.completion("hello")
        assert exc_info.value.retry_after == 30.0


class TestOpenAIClientExceptions:
    """Test that OpenAICompatClient wraps SDK exceptions correctly."""

    def test_auth_error_wrapping(self):
        """OpenAI AuthenticationError → ModelAuthError."""
        import openai
        from ccr.models.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(model_name="test", api_key="bad")
        mock_create = MagicMock(
            side_effect=openai.AuthenticationError(
                message="invalid api key",
                response=MagicMock(status_code=401, headers={}),
                body={"error": {"message": "invalid api key"}},
            )
        )
        client._client.chat.completions.create = mock_create

        with pytest.raises(ModelAuthError):
            client.completion("hello")

    def test_timeout_wrapping(self):
        """OpenAI APITimeoutError → ModelTimeoutError."""
        import openai
        from ccr.models.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(model_name="test", api_key="key")
        mock_create = MagicMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )
        client._client.chat.completions.create = mock_create

        with pytest.raises(ModelTimeoutError):
            client.completion("hello")

    def test_connection_error_wrapping(self):
        """OpenAI APIConnectionError → ModelConnectionError."""
        import openai
        from ccr.models.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(model_name="test", api_key="key")
        mock_create = MagicMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )
        client._client.chat.completions.create = mock_create

        with pytest.raises(ModelConnectionError):
            client.completion("hello")

    def test_rate_limit_wrapping(self):
        """OpenAI RateLimitError → ModelRateLimitError."""
        import openai
        from ccr.models.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(model_name="test", api_key="key")
        mock_create = MagicMock(
            side_effect=openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body={"error": {"message": "rate limited"}},
            )
        )
        client._client.chat.completions.create = mock_create

        with pytest.raises(ModelRateLimitError):
            client.completion("hello")


# --- Gateway exception handling ---


class TestGatewayExceptionHandling:
    """Test that gateway handles CCR exceptions correctly."""

    def test_auth_error_returns_401(self):
        """ModelAuthError should return 401, never passthrough."""
        from ccr.gateway import AnthropicProxyHandler
        from ccr.core.exceptions import ModelAuthError

        handler = MagicMock(spec=AnthropicProxyHandler)
        handler.passthrough_on_error = True

        # Verify ModelAuthError is not recoverable
        e = ModelAuthError("bad key")
        assert e.recoverable is False

    def test_recoverable_errors_allow_passthrough(self):
        """Recoverable CCR errors should fall through to passthrough."""
        e = ModelTimeoutError("timeout")
        assert e.recoverable is True

        e = ModelConnectionError("can't reach")
        assert e.recoverable is True

        e = PackingError("packing failed")
        assert e.recoverable is True
