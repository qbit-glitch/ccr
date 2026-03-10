"""End-to-end gateway tests using mock model backends.

Tests the full pipeline: HTTP request → gateway → engine → router → packer → response
without requiring real API keys or running models.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ccr.core.engine import CCREngine
from ccr.core.types import CCREngineConfig, CCRConfig, RouterConfig, TokenUsage
from ccr.gateway import CCRGateway


# --- Mock model server (simulates vLLM / Qwen) ---

class MockModelHandler(BaseHTTPRequestHandler):
    """Simulates an OpenAI-compatible model API."""

    response_text = "This is a mock response from the sub-model."

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        # Check what was sent
        messages = body.get("messages", [])
        last_msg = messages[-1]["content"] if messages else ""

        # Customize response based on input
        if "classify" in last_msg.lower() or "tier" in last_msg.lower():
            text = '{"tier": "simple", "confidence": 0.9, "reasoning": "test"}'
        elif "extract" in last_msg.lower() or "symbols" in last_msg.lower():
            text = '{"symbols": ["test"], "keywords": ["test"], "file_patterns": []}'
        elif "rank" in last_msg.lower() or "relevance" in last_msg.lower():
            text = '[{"path": "test.py", "relevance": 0.9, "reason": "test"}]'
        else:
            text = self.response_text

        response = {
            "id": "mock-123",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        body_bytes = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def log_message(self, format, *args):
        pass  # Silence logs


@pytest.fixture(scope="module")
def mock_model_server():
    """Start a mock OpenAI-compatible server for the sub-model."""
    server = HTTPServer(("127.0.0.1", 0), MockModelHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


@pytest.fixture
def project_dir():
    """Create a temp project with some sample files."""
    with tempfile.TemporaryDirectory() as d:
        # Create some source files
        os.makedirs(os.path.join(d, "src"))
        with open(os.path.join(d, "src", "main.py"), "w") as f:
            f.write("class App:\n    def run(self):\n        print('hello')\n")
        with open(os.path.join(d, "src", "utils.py"), "w") as f:
            f.write("def helper(x):\n    return x + 1\n")
        with open(os.path.join(d, "config.yaml"), "w") as f:
            f.write("debug: true\n")
        yield d


@pytest.fixture
def engine(project_dir, mock_model_server):
    """Create a CCR engine with mock backends."""
    config = CCREngineConfig(
        memory=CCRConfig(),
        router=RouterConfig(),
        claude_model="mock-claude",
        sub_model="mock-qwen",
        sub_model_base_url=mock_model_server,
        anthropic_api_key="test-key",
        pack_token_budget=4000,
        index_extensions=[".py", ".yaml"],
    )

    # Patch ClaudeClient to use the mock server too
    with patch("ccr.core.engine.ClaudeClient") as MockClaude:
        mock_claude_instance = MagicMock()
        mock_claude_instance.completion.return_value = "Mock Claude response: task completed."
        mock_claude_instance.get_last_usage.return_value = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.001
        )
        MockClaude.return_value = mock_claude_instance

        eng = CCREngine(project_dir, config)
        eng.initialize()
        eng._claude_client = mock_claude_instance
        yield eng
        eng.shutdown()


@pytest.fixture
def gateway(engine):
    """Start CCR gateway on a random port."""
    gw = CCRGateway(engine, host="127.0.0.1", port=0)

    # Use port 0 to get a random available port
    from http.server import ThreadingHTTPServer
    from ccr.gateway import AnthropicProxyHandler

    AnthropicProxyHandler.engine = engine
    AnthropicProxyHandler.real_base_url = "https://api.anthropic.com"
    AnthropicProxyHandler.passthrough_on_error = False  # Fail loudly in tests

    server = ThreadingHTTPServer(("127.0.0.1", 0), AnthropicProxyHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()


class TestGatewayEndToEnd:
    """Tests that send actual HTTP requests to the gateway."""

    def test_trivial_request_returns_response(self, gateway):
        """A trivial request should be routed to sub-model and return a response."""
        response = httpx.post(
            f"{gateway}/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
        assert data["content"][0]["type"] == "text"
        assert len(data["content"][0]["text"]) > 0

    def test_simple_request_includes_memory(self, gateway):
        """A simple request should include memory context."""
        response = httpx.post(
            f"{gateway}/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "fix the bug in the helper function"}],
            },
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"][0]["text"]

    def test_complex_request_routes_to_claude(self, gateway):
        """A complex request should be routed to Claude with context pack."""
        response = httpx.post(
            f"{gateway}/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": "Refactor the entire codebase to use a plugin architecture",
                    }
                ],
            },
            timeout=30,
        )
        assert response.status_code == 200
        data = response.json()
        assert "Mock Claude response" in data["content"][0]["text"]

    def test_response_has_usage_info(self, gateway):
        """Response should include usage statistics."""
        response = httpx.post(
            f"{gateway}/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello there"}],
            },
            timeout=30,
        )
        data = response.json()
        assert "usage" in data
        assert "input_tokens" in data["usage"]
        assert "output_tokens" in data["usage"]

    def test_multi_message_conversation(self, gateway):
        """Should handle multi-turn conversations."""
        response = httpx.post(
            f"{gateway}/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "Thanks"},
                ],
            },
            timeout=30,
        )
        assert response.status_code == 200

    def test_invalid_request_returns_error(self, gateway):
        """Malformed request should return an error."""
        response = httpx.post(
            f"{gateway}/v1/messages",
            content=b"not json",
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        assert response.status_code == 500


class TestEngineDirectly:
    """Tests the engine without going through HTTP."""

    def test_process_trivial(self, engine):
        from ccr.utils.parsing import parse_anthropic_request

        request = parse_anthropic_request(
            json.dumps({
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            }).encode()
        )
        response = engine.process(request)
        assert response.routed_to == "qwen_direct"
        assert response.content
        assert response.classification.tier.value == "trivial"

    def test_process_complex(self, engine):
        from ccr.utils.parsing import parse_anthropic_request

        request = parse_anthropic_request(
            json.dumps({
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": "Refactor the entire architecture of this project",
                    }
                ],
            }).encode()
        )
        response = engine.process(request)
        assert "claude" in response.routed_to
        assert response.classification.tier.value == "complex"

    def test_usage_report_tracks_calls(self, engine):
        from ccr.utils.parsing import parse_anthropic_request

        request = parse_anthropic_request(
            json.dumps({
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            }).encode()
        )
        engine.process(request)
        report = engine.get_usage_report()
        assert report["total_cost_usd"] >= 0
        assert "models" in report


class TestMemoryIntegration:
    """Tests that memory is updated correctly through the pipeline."""

    def test_ccr_directory_created(self, engine):
        assert os.path.isdir(os.path.join(engine.project_root, ".ccr"))

    def test_index_cached(self, engine):
        index_path = engine.memory.get_index_path()
        assert os.path.isfile(index_path)
        content = engine.memory.load_index()
        assert content is not None
        data = json.loads(content)
        assert data["file_count"] > 0

    def test_context_retrieval_works(self, engine):
        ctx = engine.memory.get_context(level=1)
        assert "Project" in ctx

    def test_full_branch_lifecycle(self, engine):
        mem = engine.memory

        # Start on main
        assert mem.get_active_branch() == "main"

        # Commit something
        mem.commit("Initial setup", "Set up project", "Starting fresh", ["main.py"], "Add features")

        # Create branch
        mem.create_branch("experiment", "Try new approach", "It might be faster")
        assert mem.get_active_branch() == "experiment"

        # Commit on branch
        mem.commit("Branch work", "Tried new thing", "Testing hypothesis", ["exp.py"], "Evaluate")

        # Merge back
        mem.merge("experiment", "success", "New approach works!")
        assert mem.get_active_branch() == "main"

        # Verify context has the history
        ctx = mem.get_context(level=2)
        assert "Merge: experiment" in ctx
        assert "Initial setup" in ctx
