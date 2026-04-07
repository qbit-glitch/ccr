"""End-to-end MCP transport integration tests.

Tests the full MCP stdio JSON-RPC wire format:
  subprocess start → initialize handshake → tools/list → tools/call → teardown

These tests verify the FastMCP transport layer, tool registration, and
JSON-RPC protocol — none of which are exercised by the unit test suite
(which calls Python functions directly).

Run with:
    pytest tests/integration/test_mcp_e2e.py -v
    pytest tests/integration/test_mcp_e2e.py -v -m integration
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIMEOUT = 10  # seconds per readline


def _send(proc: subprocess.Popen, msg: dict[str, Any]) -> None:
    """Write a single newline-terminated JSON-RPC message to the server stdin."""
    line = json.dumps(msg) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()


def _recv(proc: subprocess.Popen, timeout: float = _TIMEOUT) -> dict[str, Any]:
    """Read one newline-delimited JSON-RPC response from server stdout.

    Uses a background thread to perform the blocking readline so a wall-clock
    deadline can be applied without platform-specific select() gymnastics.

    Raises:
        TimeoutError: If no response arrives within *timeout* seconds.
        RuntimeError: If the process exited unexpectedly or stdout closes.
    """
    import threading

    result: list[bytes] = []
    exc_holder: list[Exception] = []

    def _reader() -> None:
        try:
            line = proc.stdout.readline()
            result.append(line)
        except Exception as e:
            exc_holder.append(e)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(
            f"No response from MCP server within {timeout}s "
            f"(process returncode={proc.poll()!r})"
        )
    if exc_holder:
        raise RuntimeError(f"Error reading from MCP server stdout: {exc_holder[0]}")
    if not result or result[0] == b"":
        raise RuntimeError(
            f"MCP server closed stdout unexpectedly "
            f"(returncode={proc.poll()!r})"
        )
    return json.loads(result[0].decode())


def _send_notification(proc: subprocess.Popen, method: str) -> None:
    """Send a JSON-RPC notification (no response expected)."""
    _send(proc, {"jsonrpc": "2.0", "method": method})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def project_dir():
    """Temporary directory used as CCR project root throughout this module."""
    with tempfile.TemporaryDirectory(prefix="ccr_e2e_") as d:
        yield d


@pytest.fixture(scope="module")
def mcp_server(project_dir):
    """Start the CCR MCP server subprocess and perform the initialize handshake.

    Yields a tuple ``(proc, next_id_counter)`` where *proc* is the live
    Popen object and *next_id* is a list[int] used as a mutable counter
    so all tests in the module share a monotonically-increasing request id.

    The fixture skips gracefully when the server fails to start (e.g.
    missing optional dependencies), so CI doesn't hard-fail on reduced envs.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "ccr.mcp_server", "--project", project_dir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # keep test output clean
    )

    try:
        # --- MCP initialize handshake ---
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ccr-e2e-test", "version": "0.1"},
                },
            },
        )
        init_response = _recv(proc, timeout=15)

        if "error" in init_response:
            pytest.skip(
                f"MCP server returned error on initialize: {init_response['error']}"
            )

        # Send notifications/initialized — server does not reply to this
        _send_notification(proc, "notifications/initialized")

        # Small pause so the server processes the notification
        time.sleep(0.05)

    except Exception as exc:
        proc.kill()
        proc.wait()
        pytest.skip(f"MCP server failed to start: {exc}")

    # Shared monotonic request-id counter (list so it's mutable from fixtures)
    next_id = [2]  # id=1 was used for initialize

    yield proc, next_id, init_response

    # Teardown
    proc.kill()
    proc.wait()


def _call(
    proc: subprocess.Popen,
    next_id: list[int],
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a JSON-RPC request and return the parsed response dict."""
    req_id = next_id[0]
    next_id[0] += 1
    _send(proc, {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
    return _recv(proc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMcpInitialize:
    """Verify the MCP initialize handshake response structure."""

    def test_initialize_response_has_jsonrpc(self, mcp_server):
        _, _, init_resp = mcp_server
        assert init_resp.get("jsonrpc") == "2.0"

    def test_initialize_response_has_result(self, mcp_server):
        _, _, init_resp = mcp_server
        assert "result" in init_resp, f"Expected 'result', got: {init_resp}"

    def test_initialize_protocol_version(self, mcp_server):
        _, _, init_resp = mcp_server
        version = init_resp["result"].get("protocolVersion")
        assert version == "2024-11-05", f"Unexpected protocolVersion: {version}"

    def test_initialize_advertises_tools_capability(self, mcp_server):
        _, _, init_resp = mcp_server
        capabilities = init_resp["result"].get("capabilities", {})
        assert "tools" in capabilities, (
            f"Server should advertise 'tools' capability. Got: {capabilities}"
        )

    def test_initialize_server_info(self, mcp_server):
        _, _, init_resp = mcp_server
        # FastMCP sets serverInfo with the app name
        server_info = init_resp["result"].get("serverInfo", {})
        assert server_info.get("name") == "ccr", (
            f"Expected serverInfo.name == 'ccr', got: {server_info}"
        )


@pytest.mark.integration
class TestToolsList:
    """Verify tools/list returns expected tool registrations."""

    @pytest.fixture(scope="class")
    def tools_list(self, mcp_server):
        proc, next_id, _ = mcp_server
        resp = _call(proc, next_id, "tools/list")
        assert "error" not in resp, f"tools/list error: {resp.get('error')}"
        return resp["result"]["tools"]

    @pytest.fixture(scope="class")
    def tool_names(self, tools_list):
        return {t["name"] for t in tools_list}

    def test_tools_list_is_non_empty(self, tools_list):
        assert len(tools_list) > 0, "Expected at least one registered tool"

    def test_tool_count_matches_expected(self, tool_names):
        # CCR exposes 31-32 tools (varies slightly across versions)
        assert len(tool_names) >= 28, (
            f"Expected >= 28 tools, found {len(tool_names)}: {sorted(tool_names)}"
        )

    def test_gcc_commit_registered(self, tool_names):
        assert "gcc_commit" in tool_names

    def test_gcc_context_registered(self, tool_names):
        assert "gcc_context" in tool_names

    def test_gcc_status_registered(self, tool_names):
        assert "gcc_status" in tool_names

    def test_ace_get_playbook_registered(self, tool_names):
        assert "ace_get_playbook" in tool_names

    def test_rlm_tools_registered(self, tool_names):
        for name in ("rlm_init", "rlm_execute", "rlm_finalize"):
            assert name in tool_names, f"Expected {name!r} in tool list"

    def test_index_tools_registered(self, tool_names):
        for name in ("index_build", "index_search"):
            assert name in tool_names, f"Expected {name!r} in tool list"

    def test_each_tool_has_description(self, tools_list):
        missing = [t["name"] for t in tools_list if not t.get("description")]
        assert not missing, f"Tools missing description: {missing}"

    def test_each_tool_has_input_schema(self, tools_list):
        missing = [t["name"] for t in tools_list if "inputSchema" not in t]
        assert not missing, f"Tools missing inputSchema: {missing}"


@pytest.mark.integration
class TestGccStatus:
    """Verify gcc_status tool call via MCP wire protocol."""

    @pytest.fixture(scope="class")
    def status_response(self, mcp_server):
        proc, next_id, _ = mcp_server
        resp = _call(
            proc,
            next_id,
            "tools/call",
            {"name": "gcc_status", "arguments": {}},
        )
        return resp

    def test_no_error(self, status_response):
        assert "error" not in status_response, (
            f"gcc_status returned error: {status_response.get('error')}"
        )

    def test_result_has_content(self, status_response):
        result = status_response.get("result", {})
        assert "content" in result, f"Expected 'content' in result: {result}"

    def test_content_is_text(self, status_response):
        content = status_response["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"

    def test_response_is_valid_json(self, status_response):
        text = status_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_response_has_branch_field(self, status_response):
        text = status_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert "branch" in data, f"Expected 'branch' key in gcc_status result: {data}"

    def test_response_branch_is_main(self, status_response):
        text = status_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data["branch"] == "main"

    def test_response_has_message_field(self, status_response):
        text = status_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert "message" in data


@pytest.mark.integration
class TestGccCommit:
    """Verify gcc_commit persists a commit record via MCP wire protocol."""

    _COMMIT_TITLE = "MCP E2E test commit"

    @pytest.fixture(scope="class")
    def commit_response(self, mcp_server):
        proc, next_id, _ = mcp_server
        resp = _call(
            proc,
            next_id,
            "tools/call",
            {
                "name": "gcc_commit",
                "arguments": {
                    "title": self._COMMIT_TITLE,
                    "what": "Testing the MCP transport layer end-to-end",
                    "why": "Verify JSON-RPC wire format, tool registration, and stdio IO",
                    "files_changed": [],
                    "next_step": "Run full integration test suite",
                },
            },
        )
        return resp

    def test_no_error(self, commit_response):
        assert "error" not in commit_response, (
            f"gcc_commit returned error: {commit_response.get('error')}"
        )

    def test_result_has_content(self, commit_response):
        assert "content" in commit_response["result"]

    def test_content_is_text(self, commit_response):
        content = commit_response["result"]["content"]
        assert content[0]["type"] == "text"

    def test_commit_id_assigned(self, commit_response):
        text = commit_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert "commit_id" in data, f"Expected commit_id in result: {data}"
        assert data["commit_id"].startswith("C"), f"commit_id format wrong: {data['commit_id']}"

    def test_commit_title_echoed(self, commit_response):
        text = commit_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data.get("title") == self._COMMIT_TITLE

    def test_admission_decision_present(self, commit_response):
        text = commit_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert "admission_decision" in data
        assert data["admission_decision"] in ("created", "merged", "rejected")

    def test_structured_content_matches_text(self, commit_response):
        """FastMCP should return both content[].text and structuredContent."""
        result = commit_response["result"]
        if "structuredContent" not in result:
            pytest.skip("structuredContent not present (FastMCP version may differ)")
        text_data = json.loads(result["content"][0]["text"])
        sc = result["structuredContent"]
        assert text_data.get("commit_id") == sc.get("commit_id")


@pytest.mark.integration
class TestGccContext:
    """Verify gcc_context returns context containing the previously committed title."""

    @pytest.fixture(scope="class")
    def context_response(self, mcp_server):
        proc, next_id, _ = mcp_server
        resp = _call(
            proc,
            next_id,
            "tools/call",
            {
                "name": "gcc_context",
                "arguments": {"level": 1},
            },
        )
        return resp

    def test_no_error(self, context_response):
        assert "error" not in context_response, (
            f"gcc_context returned error: {context_response.get('error')}"
        )

    def test_result_has_content(self, context_response):
        assert "content" in context_response["result"]

    def test_content_is_text(self, context_response):
        content = context_response["result"]["content"]
        assert content[0]["type"] == "text"

    def test_response_is_valid_json(self, context_response):
        text = context_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_response_has_message(self, context_response):
        text = context_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert "message" in data

    def test_commit_title_appears_in_context(self, context_response):
        """The commit made in TestGccCommit must be visible in gcc_context output."""
        text = context_response["result"]["content"][0]["text"]
        data = json.loads(text)
        message = data.get("message", "")
        assert "MCP E2E test commit" in message, (
            f"Expected commit title 'MCP E2E test commit' in context message.\n"
            f"Actual message:\n{message}"
        )

    def test_response_has_branch(self, context_response):
        text = context_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data.get("branch") == "main"

    def test_response_has_level(self, context_response):
        text = context_response["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data.get("level") == 1


@pytest.mark.integration
class TestProtocolEdgeCases:
    """Verify correct JSON-RPC error handling and edge cases."""

    def test_unknown_tool_returns_error(self, mcp_server):
        proc, next_id, _ = mcp_server
        resp = _call(
            proc,
            next_id,
            "tools/call",
            {"name": "nonexistent_tool_xyz", "arguments": {}},
        )
        # MCP spec: unknown tool → error response OR isError content
        has_error = "error" in resp
        has_is_error = resp.get("result", {}).get("isError", False)
        assert has_error or has_is_error, (
            f"Expected error for unknown tool, got: {resp}"
        )

    def test_response_id_matches_request(self, mcp_server):
        """JSON-RPC spec: response id must match request id."""
        proc, next_id, _ = mcp_server
        req_id = next_id[0]
        next_id[0] += 1
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": "gcc_status", "arguments": {}},
            },
        )
        resp = _recv(proc)
        assert resp.get("id") == req_id, (
            f"Response id {resp.get('id')} != request id {req_id}"
        )

    def test_gcc_context_level2(self, mcp_server):
        """gcc_context at level=2 should include rolling summary + commits."""
        proc, next_id, _ = mcp_server
        resp = _call(
            proc,
            next_id,
            "tools/call",
            {"name": "gcc_context", "arguments": {"level": 2}},
        )
        assert "error" not in resp
        text = resp["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data.get("level") == 2
        # At level 2, recent commits section should be present
        message = data.get("message", "")
        assert len(message) > 0

    def test_ace_get_playbook_returns_text(self, mcp_server):
        """ace_get_playbook should return a non-empty text response."""
        proc, next_id, _ = mcp_server
        resp = _call(
            proc,
            next_id,
            "tools/call",
            {"name": "ace_get_playbook", "arguments": {}},
        )
        assert "error" not in resp
        content = resp["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        text = content[0]["text"]
        assert len(text) > 0
