"""CCR Gateway — HTTP proxy that intercepts Claude Code API calls.

Sits between Claude Code and api.anthropic.com.
Usage: set ANTHROPIC_BASE_URL=http://127.0.0.1:7447 before starting Claude Code.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

from ccr.core.engine import CCREngine
from ccr.core.exceptions import CCRError, ModelAuthError
from ccr.core.types import CCREngineConfig, CCRResponse, TokenUsage
from ccr.utils.parsing import (
    format_anthropic_error,
    format_anthropic_response,
    parse_anthropic_request,
)

logger = logging.getLogger(__name__)

# Gateway start time — set when server starts
_gateway_start_time: float = 0.0


class AnthropicProxyHandler(BaseHTTPRequestHandler):
    """Intercepts POST /v1/messages, routes through CCR engine."""

    # Set by CCRGateway before server starts
    engine: CCREngine
    real_base_url: str = "https://api.anthropic.com"
    passthrough_on_error: bool = True

    def do_POST(self) -> None:
        """Handle POST requests — the main interception point."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""

        path = self.path

        # Only intercept /v1/messages
        if path == "/v1/messages":
            self._handle_messages(body)
        else:
            # Passthrough everything else
            self._passthrough(path, body, method="POST")

    def do_GET(self) -> None:
        """Handle GET requests — health checks + passthrough."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/ready":
            self._handle_ready()
        else:
            self._passthrough(self.path, b"", method="GET")

    def _handle_health(self) -> None:
        """Liveness check — is the gateway process alive?"""
        uptime = time.time() - _gateway_start_time if _gateway_start_time else 0
        self._send_json(200, {
            "status": "ok",
            "uptime_seconds": round(uptime, 1),
        })

    def _handle_ready(self) -> None:
        """Readiness check — is the engine initialized and models reachable?"""
        checks: dict[str, dict[str, Any]] = {}
        all_ok = True

        # Check engine initialization
        engine_ok = getattr(self.engine, "_initialized", False)
        checks["engine"] = {"ok": engine_ok}
        if not engine_ok:
            all_ok = False

        # Check sub-model connectivity
        sub_client = getattr(self.engine, "_sub_client", None)
        if sub_client:
            sub_ok, sub_detail = sub_client.check_connectivity()
            checks["sub_model"] = {"ok": sub_ok, "detail": sub_detail}
            if not sub_ok:
                all_ok = False
        else:
            checks["sub_model"] = {"ok": False, "detail": "not initialized"}
            all_ok = False

        # Check Claude API connectivity
        claude_client = getattr(self.engine, "_claude_client", None)
        if claude_client:
            claude_ok, claude_detail = claude_client.check_connectivity()
            checks["claude"] = {"ok": claude_ok, "detail": claude_detail}
            if not claude_ok:
                all_ok = False
        else:
            checks["claude"] = {"ok": False, "detail": "not initialized"}
            all_ok = False

        # Usage stats
        cost_tracker = getattr(self.engine, "cost_tracker", None)
        if cost_tracker:
            report = cost_tracker.get_report()
            checks["usage"] = {
                "total_calls": report.get("total_calls", 0),
                "total_cost_usd": report.get("total_cost_usd", 0),
            }

        status_code = 200 if all_ok else 503
        self._send_json(status_code, {
            "status": "ready" if all_ok else "not_ready",
            "checks": checks,
        })

    def _handle_messages(self, body: bytes) -> None:
        """Process a Messages API request through CCR."""
        try:
            # Check if streaming is requested — passthrough if so (v1 doesn't support streaming)
            data = json.loads(body)
            if data.get("stream", False):
                logger.info("Streaming request — passing through to Anthropic")
                self._passthrough("/v1/messages", body, method="POST")
                return

            # Parse into CCRRequest
            request = parse_anthropic_request(body)
            logger.info(
                f"[{request.request_id}] Intercepted: model={request.model_requested}, "
                f"messages={len(request.original_messages)}"
            )

            # Process through engine
            response = self.engine.process(request)

            # Format and send response
            api_response = format_anthropic_response(
                content=response.content,
                model=response.model_used,
                usage=response.usage,
                request_id=response.request_id,
            )
            self._send_json(200, api_response)

            logger.info(
                f"[{request.request_id}] → {response.routed_to} "
                f"(tier={response.classification.tier.value}, "
                f"tokens={response.usage.total_tokens})"
            )

        except ModelAuthError as e:
            # Auth errors should never passthrough — tell the user immediately
            logger.error(f"Authentication failed: {e}")
            self._send_json(401, format_anthropic_error(str(e), "authentication_error"))
        except CCRError as e:
            logger.error(f"CCR processing failed: {e}", exc_info=True)
            if self.passthrough_on_error and e.recoverable:
                logger.info("Falling back to passthrough")
                self._passthrough("/v1/messages", body, method="POST")
            else:
                self._send_json(500, format_anthropic_error(str(e), "api_error"))
        except Exception as e:
            logger.error(f"CCR processing failed (unexpected): {e}", exc_info=True)
            if self.passthrough_on_error:
                logger.info("Falling back to passthrough")
                self._passthrough("/v1/messages", body, method="POST")
            else:
                self._send_json(500, format_anthropic_error(str(e), "api_error"))

    def _passthrough(self, path: str, body: bytes, method: str = "POST") -> None:
        """Forward request directly to Anthropic API."""
        url = f"{self.real_base_url}{path}"

        # Forward all headers except Host
        headers = {}
        for key, value in self.headers.items():
            if key.lower() not in ("host", "content-length"):
                headers[key] = value

        try:
            with httpx.Client(timeout=300) as client:
                if method == "POST":
                    resp = client.post(url, content=body, headers=headers)
                else:
                    resp = client.get(url, headers=headers)

            self.send_response(resp.status_code)
            for key, value in resp.headers.items():
                if key.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                    self.send_header(key, value)
            response_body = resp.content
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except Exception as e:
            logger.error(f"Passthrough failed: {e}")
            self._send_json(502, format_anthropic_error(f"Proxy error: {e}", "api_error"))

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        """Override to use Python logging instead of stderr."""
        logger.debug(format % args)


class CCRGateway:
    """Manages the proxy server lifecycle.

    Usage:
        gateway = CCRGateway(engine, port=7447)
        gateway.start()
        # Now set ANTHROPIC_BASE_URL=http://127.0.0.1:7447 and start Claude Code
        gateway.stop()
    """

    def __init__(
        self,
        engine: CCREngine,
        host: str = "127.0.0.1",
        port: int = 7447,
        passthrough_on_error: bool = True,
    ):
        self.engine = engine
        self.host = host
        self.port = port
        self.passthrough_on_error = passthrough_on_error
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        """Start the proxy server. Returns the base URL."""
        global _gateway_start_time
        _gateway_start_time = time.time()

        # Initialize engine
        self.engine.initialize()

        # Configure handler class
        AnthropicProxyHandler.engine = self.engine
        AnthropicProxyHandler.real_base_url = self.engine.config.anthropic_real_base_url
        AnthropicProxyHandler.passthrough_on_error = self.passthrough_on_error

        self._server = ThreadingHTTPServer(
            (self.host, self.port),
            AnthropicProxyHandler,
        )

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="ccr-gateway",
        )
        self._thread.start()

        logger.info(f"CCR Gateway started at {self.base_url}")
        return self.base_url

    def stop(self) -> None:
        """Stop the proxy server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self.engine.shutdown()
        logger.info("CCR Gateway stopped")

    def __enter__(self) -> CCRGateway:
        self.start()
        return self

    def __exit__(self, *args) -> bool:
        self.stop()
        return False
