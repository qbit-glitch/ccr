"""Tests for health check endpoint and connectivity checks."""

import json
import time
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from ccr.core.exceptions import ModelConnectionError


# --- Model connectivity checks ---


class TestOpenAICompatConnectivity:
    def test_connected(self):
        from ccr.models.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(model_name="test", api_key="key")
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp):
            ok, detail = client.check_connectivity()
        assert ok is True
        assert detail == "connected"

    def test_non_200_still_reachable(self):
        from ccr.models.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(model_name="test", api_key="key")
        mock_resp = MagicMock(status_code=404)
        with patch("httpx.get", return_value=mock_resp):
            ok, detail = client.check_connectivity()
        assert ok is True
        assert "404" in detail

    def test_unreachable(self):
        from ccr.models.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(model_name="test", api_key="key")
        with patch("httpx.get", side_effect=ConnectionError("refused")):
            ok, detail = client.check_connectivity()
        assert ok is False
        assert "unreachable" in detail

    def test_timeout(self):
        from ccr.models.openai_compat import OpenAICompatClient
        import httpx

        client = OpenAICompatClient(model_name="test", api_key="key")
        with patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
            ok, detail = client.check_connectivity()
        assert ok is False
        assert "unreachable" in detail


class TestClaudeClientConnectivity:
    def test_connected(self):
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(api_key="sk-ant-test", model_name="test")
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp):
            ok, detail = client.check_connectivity()
        assert ok is True
        assert detail == "connected"

    def test_invalid_api_key(self):
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(api_key="bad-key", model_name="test")
        mock_resp = MagicMock(status_code=401)
        with patch("httpx.get", return_value=mock_resp):
            ok, detail = client.check_connectivity()
        assert ok is False
        assert "invalid API key" in detail

    def test_unreachable(self):
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(api_key="sk-ant-test", model_name="test")
        with patch("httpx.get", side_effect=ConnectionError("refused")):
            ok, detail = client.check_connectivity()
        assert ok is False
        assert "unreachable" in detail


class TestBaseClientConnectivity:
    def test_default_not_implemented(self):
        """Base class returns False with 'not implemented'."""
        from ccr.models.base import BaseLMClient

        # Create a minimal concrete subclass
        class MinimalClient(BaseLMClient):
            def completion(self, messages, **kwargs): return ""
            async def acompletion(self, messages, **kwargs): return ""
            def get_usage_summary(self): return None
            def get_last_usage(self): return None

        client = MinimalClient("test")
        ok, detail = client.check_connectivity()
        assert ok is False
        assert "not implemented" in detail


# --- Gateway health endpoint ---


class TestHealthEndpoint:
    """Test /health endpoint."""

    def _make_handler(self):
        """Create a mock handler with the right methods."""
        from ccr.gateway import AnthropicProxyHandler
        handler = MagicMock(spec=AnthropicProxyHandler)
        handler._send_json = MagicMock()
        # Bind the real methods
        handler._handle_health = AnthropicProxyHandler._handle_health.__get__(handler)
        handler._handle_ready = AnthropicProxyHandler._handle_ready.__get__(handler)
        return handler

    def test_health_returns_200(self):
        import ccr.gateway as gw
        gw._gateway_start_time = time.time() - 60  # 60s ago

        handler = self._make_handler()
        handler._handle_health()

        handler._send_json.assert_called_once()
        status, data = handler._send_json.call_args[0]
        assert status == 200
        assert data["status"] == "ok"
        assert data["uptime_seconds"] >= 59

    def test_health_zero_uptime_before_start(self):
        import ccr.gateway as gw
        gw._gateway_start_time = 0

        handler = self._make_handler()
        handler._handle_health()

        status, data = handler._send_json.call_args[0]
        assert status == 200
        assert data["uptime_seconds"] == 0


class TestReadyEndpoint:
    """Test /ready endpoint."""

    def _make_handler(self, engine_initialized=True, sub_ok=True, claude_ok=True):
        from ccr.gateway import AnthropicProxyHandler

        handler = MagicMock(spec=AnthropicProxyHandler)
        handler._send_json = MagicMock()
        handler._handle_ready = AnthropicProxyHandler._handle_ready.__get__(handler)

        # Mock engine
        engine = MagicMock()
        engine._initialized = engine_initialized

        # Mock sub client
        sub_client = MagicMock()
        sub_client.check_connectivity.return_value = (
            sub_ok, "connected" if sub_ok else "unreachable"
        )
        engine._sub_client = sub_client

        # Mock claude client
        claude_client = MagicMock()
        claude_client.check_connectivity.return_value = (
            claude_ok, "connected" if claude_ok else "invalid API key"
        )
        engine._claude_client = claude_client

        # Mock cost tracker
        engine.cost_tracker = MagicMock()
        engine.cost_tracker.get_report.return_value = {
            "total_calls": 42,
            "total_cost_usd": 1.23,
        }

        handler.engine = engine
        return handler

    def test_all_healthy_returns_200(self):
        handler = self._make_handler()
        handler._handle_ready()

        status, data = handler._send_json.call_args[0]
        assert status == 200
        assert data["status"] == "ready"
        assert data["checks"]["engine"]["ok"] is True
        assert data["checks"]["sub_model"]["ok"] is True
        assert data["checks"]["claude"]["ok"] is True
        assert data["checks"]["usage"]["total_calls"] == 42

    def test_sub_model_down_returns_503(self):
        handler = self._make_handler(sub_ok=False)
        handler._handle_ready()

        status, data = handler._send_json.call_args[0]
        assert status == 503
        assert data["status"] == "not_ready"
        assert data["checks"]["sub_model"]["ok"] is False
        assert "unreachable" in data["checks"]["sub_model"]["detail"]

    def test_claude_down_returns_503(self):
        handler = self._make_handler(claude_ok=False)
        handler._handle_ready()

        status, data = handler._send_json.call_args[0]
        assert status == 503
        assert data["checks"]["claude"]["ok"] is False

    def test_engine_not_initialized_returns_503(self):
        handler = self._make_handler(engine_initialized=False)
        handler._handle_ready()

        status, data = handler._send_json.call_args[0]
        assert status == 503
        assert data["checks"]["engine"]["ok"] is False

    def test_no_clients_returns_503(self):
        """Before initialization, clients are None."""
        from ccr.gateway import AnthropicProxyHandler

        handler = MagicMock(spec=AnthropicProxyHandler)
        handler._send_json = MagicMock()
        handler._handle_ready = AnthropicProxyHandler._handle_ready.__get__(handler)

        engine = MagicMock()
        engine._initialized = False
        engine._sub_client = None
        engine._claude_client = None
        engine.cost_tracker = None
        handler.engine = engine

        handler._handle_ready()
        status, data = handler._send_json.call_args[0]
        assert status == 503
        assert data["checks"]["sub_model"]["ok"] is False
        assert data["checks"]["claude"]["ok"] is False


class TestGatewayRouting:
    """Test that GET /health and /ready are routed correctly."""

    def test_get_health_dispatches(self):
        from ccr.gateway import AnthropicProxyHandler

        handler = MagicMock(spec=AnthropicProxyHandler)
        handler.path = "/health"
        handler._handle_health = MagicMock()
        handler._handle_ready = MagicMock()
        handler._passthrough = MagicMock()

        # Call the real do_GET
        AnthropicProxyHandler.do_GET(handler)
        handler._handle_health.assert_called_once()
        handler._handle_ready.assert_not_called()
        handler._passthrough.assert_not_called()

    def test_get_ready_dispatches(self):
        from ccr.gateway import AnthropicProxyHandler

        handler = MagicMock(spec=AnthropicProxyHandler)
        handler.path = "/ready"
        handler._handle_health = MagicMock()
        handler._handle_ready = MagicMock()
        handler._passthrough = MagicMock()

        AnthropicProxyHandler.do_GET(handler)
        handler._handle_ready.assert_called_once()

    def test_other_get_passes_through(self):
        from ccr.gateway import AnthropicProxyHandler

        handler = MagicMock(spec=AnthropicProxyHandler)
        handler.path = "/v1/models"
        handler._handle_health = MagicMock()
        handler._handle_ready = MagicMock()
        handler._passthrough = MagicMock()

        AnthropicProxyHandler.do_GET(handler)
        handler._passthrough.assert_called_once_with("/v1/models", b"", method="GET")
        handler._handle_health.assert_not_called()
