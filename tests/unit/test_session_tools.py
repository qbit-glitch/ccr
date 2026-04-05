"""Unit tests for ccr.mcp.session_tools — MCP tool layer over SessionStore."""

from __future__ import annotations

import json
import os

import pytest

import ccr.mcp.server as _srv
from ccr.mcp_server import _init
from ccr.mcp.session_tools import (
    session_export,
    session_get_history,
    session_log_turn,
    session_search,
)
import ccr.mcp_server as mcp_mod


@pytest.fixture(autouse=True)
def setup_project(tmp_path, monkeypatch):
    """Initialize CCR in a temp directory for each test."""
    fake_home_ccr = tmp_path / "global_ccr"
    fake_home_ccr.mkdir()
    original_expanduser = os.path.expanduser

    def mock_expanduser(path):
        if path.startswith("~/.ccr"):
            return str(fake_home_ccr) + path[5:]
        return original_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)
    (tmp_path / "hello.py").write_text("def greet(): pass\n")
    _init(str(tmp_path))

    # Reset session state so each test starts clean
    _srv._session_store = None
    _srv._current_session_id = ""
    _srv._session_db_path = str(tmp_path / ".ccr" / "sessions.db")

    yield tmp_path

    # Cleanup
    mcp_mod._memory = None
    mcp_mod._playbook = None
    mcp_mod._global_playbook = None
    mcp_mod._repo_index = None
    mcp_mod._repl = None
    _srv._session_store = None
    _srv._current_session_id = ""


def _write_session_id(tmp_path: object, session_id: str) -> None:
    """Helper: write a session ID file as the hook would."""
    id_file = tmp_path / ".ccr" / ".current_session_id"
    id_file.write_text(session_id)


# ===========================================================================
# session_log_turn
# ===========================================================================


class TestSessionLogTurn:
    def test_creates_session_if_none_active(self, tmp_path):
        """session_log_turn should create a transient session when none is active."""
        result = session_log_turn(assistant_message="Hello, world!")
        assert result["turn_number"] == 1
        assert result["session_id"] != ""
        assert "Logged turn" in result["message"]

    def test_uses_current_session_from_file(self, tmp_path):
        """session_log_turn should use the session ID from .current_session_id."""
        from ccr.core.session_store import SessionStore
        store = SessionStore(str(tmp_path / ".ccr" / "sessions.db"))
        sid = store.create_session(project=str(tmp_path))
        store.close()
        _write_session_id(tmp_path, sid)

        result = session_log_turn(assistant_message="My response.")
        assert result["session_id"] == sid
        assert result["turn_number"] == 1

    def test_returns_correct_turn_number_sequence(self, tmp_path):
        """Multiple calls increment turn_number."""
        r1 = session_log_turn(assistant_message="First response.")
        r2 = session_log_turn(assistant_message="Second response.")
        r3 = session_log_turn(assistant_message="Third response.")
        assert r1["turn_number"] == 1
        assert r2["turn_number"] == 2
        assert r3["turn_number"] == 3

    def test_with_explicit_user_message(self, tmp_path):
        """user_message parameter is stored when provided."""
        session_log_turn(
            assistant_message="42.",
            user_message="What is 6 times 7?",
        )
        sid = _srv._current_session_id
        history = session_get_history(session_id=sid)
        assert history["turns"][0]["user_message"] == "What is 6 times 7?"

    def test_reads_pending_user_msg_file(self, tmp_path):
        """user_message auto-read from .pending_user_msg when not provided."""
        pending = tmp_path / ".ccr" / ".pending_user_msg"
        pending.write_text("What is the capital of France?")
        result = session_log_turn(assistant_message="Paris.")
        assert result["turn_number"] == 1
        # File should be deleted after read
        assert not pending.exists()
        sid = result["session_id"]
        history = session_get_history(session_id=sid)
        assert history["turns"][0]["user_message"] == "What is the capital of France?"

    def test_with_tool_calls(self, tmp_path):
        """tool_calls list is persisted correctly."""
        tools = [{"name": "Read", "input": {"file_path": "foo.py"}}]
        session_log_turn(assistant_message="Done.", tool_calls=tools)
        sid = _srv._current_session_id
        history = session_get_history(session_id=sid)
        stored_tools = json.loads(history["turns"][0]["tool_calls_json"])
        assert stored_tools == tools

    def test_with_files_touched(self, tmp_path):
        """files_touched list is persisted correctly."""
        files = ["ccr/core/session_store.py"]
        session_log_turn(assistant_message="Done.", files_touched=files)
        sid = _srv._current_session_id
        history = session_get_history(session_id=sid)
        stored_files = json.loads(history["turns"][0]["files_touched_json"])
        assert stored_files == files

    def test_never_raises_on_bad_session_id(self, tmp_path):
        """session_log_turn should be non-fatal even with corrupt state."""
        _srv._current_session_id = "bad_id_that_does_not_exist_in_db"
        result = session_log_turn(assistant_message="test")
        # Should either succeed or return an error message — never raise
        assert isinstance(result["message"], str)


# ===========================================================================
# session_get_history
# ===========================================================================


class TestSessionGetHistory:
    def test_default_uses_current_session(self, tmp_path):
        """Empty session_id defaults to _current_session_id."""
        session_log_turn(assistant_message="Hello.")
        sid = _srv._current_session_id
        history = session_get_history()
        assert history["session_id"] == sid
        assert history["turn_count"] == 1

    def test_explicit_session_id(self, tmp_path):
        """Can retrieve a session by explicit ID."""
        from ccr.core.session_store import SessionStore
        store = SessionStore(str(tmp_path / ".ccr" / "sessions.db"))
        sid = store.create_session()
        store.log_turn(sid, "q", "a")
        store.close()

        history = session_get_history(session_id=sid)
        assert history["session_id"] == sid
        assert history["turn_count"] == 1

    def test_limit_respected(self, tmp_path):
        """limit parameter caps number of returned turns."""
        for i in range(5):
            session_log_turn(assistant_message=f"Response {i}")
        sid = _srv._current_session_id
        history = session_get_history(session_id=sid, limit=2)
        assert history["turn_count"] == 2

    def test_no_active_session_returns_gracefully(self, tmp_path):
        """Returns empty result when no session is active."""
        _srv._current_session_id = ""
        result = session_get_history()
        assert result["turn_count"] == 0
        assert result["turns"] == []


# ===========================================================================
# session_search
# ===========================================================================


class TestSessionSearch:
    def test_finds_text_in_turns(self, tmp_path):
        """Search locates turns containing the query text."""
        session_log_turn(
            assistant_message="SQLite is a lightweight embedded database.",
            user_message="Tell me about databases.",
        )
        results = session_search(query="SQLite")
        assert results["result_count"] >= 1

    def test_no_results_for_nonexistent_text(self, tmp_path):
        """Returns empty result for unmatched query."""
        session_log_turn(assistant_message="Hello world.")
        results = session_search(query="xyzzynosuchword1234")
        assert results["result_count"] == 0
        assert results["results"] == []


# ===========================================================================
# session_export
# ===========================================================================


class TestSessionExport:
    def test_export_jsonl(self, tmp_path):
        """JSONL export produces one JSON object per line with messages key."""
        session_log_turn(assistant_message="line 1 response", user_message="line 1 user")
        session_log_turn(assistant_message="line 2 response", user_message="line 2 user")
        sid = _srv._current_session_id
        result = session_export(session_id=sid, format="jsonl")
        assert result["format"] == "jsonl"
        lines = [l for l in result["data"].strip().splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "messages" in obj

    def test_export_json(self, tmp_path):
        """JSON export produces a dict with session and turns keys."""
        session_log_turn(assistant_message="test response")
        sid = _srv._current_session_id
        result = session_export(session_id=sid, format="json")
        data = json.loads(result["data"])
        assert "session" in data
        assert "turns" in data

    def test_export_markdown(self, tmp_path):
        """Markdown export contains turn headers."""
        session_log_turn(assistant_message="the answer", user_message="the question")
        sid = _srv._current_session_id
        result = session_export(session_id=sid, format="markdown")
        assert "## Turn 1" in result["data"]

    def test_no_active_session_returns_gracefully(self, tmp_path):
        """Returns empty data when no session is active."""
        _srv._current_session_id = ""
        result = session_export()
        assert result["data"] == ""
        assert "session_id" in result["message"].lower() or result["session_id"] == ""

    def test_invalid_format_returns_error_message(self, tmp_path):
        """Invalid format arg returns error message, never raises."""
        session_log_turn(assistant_message="some response")
        sid = _srv._current_session_id
        result = session_export(session_id=sid, format="xml")
        assert result["data"] == ""
        assert result["message"] != ""

    def test_db_error_returns_gracefully(self, tmp_path, monkeypatch):
        """SQLite operational errors are caught and returned as message, not raised."""
        session_log_turn(assistant_message="first")
        sid = _srv._current_session_id

        def bad_export(*args, **kwargs):
            import sqlite3
            raise sqlite3.OperationalError("disk I/O error")

        store = _srv._ensure_session_store()
        monkeypatch.setattr(store, "export_session", bad_export)
        result = session_export(session_id=sid, format="jsonl")
        assert result["data"] == ""
        assert "error" in result["message"].lower() or "disk" in result["message"].lower()


# ===========================================================================
# Error-path coverage for session_get_history and session_search
# ===========================================================================


class TestSessionToolsErrorPaths:
    def test_get_history_db_error_returns_gracefully(self, tmp_path, monkeypatch):
        """get_session_turns DB error returns error message instead of raising."""
        session_log_turn(assistant_message="test")
        sid = _srv._current_session_id

        def bad_get_turns(*args, **kwargs):
            import sqlite3
            raise sqlite3.OperationalError("disk I/O error")

        store = _srv._ensure_session_store()
        monkeypatch.setattr(store, "get_session_turns", bad_get_turns)
        result = session_get_history(session_id=sid)
        assert result["turn_count"] == 0
        assert result["turns"] == []
        assert "error" in result["message"].lower()

    def test_search_db_error_returns_gracefully(self, tmp_path, monkeypatch):
        """search_turns DB error returns error message instead of raising."""
        session_log_turn(assistant_message="hello")

        def bad_search(*args, **kwargs):
            import sqlite3
            raise sqlite3.OperationalError("disk I/O error")

        store = _srv._ensure_session_store()
        monkeypatch.setattr(store, "search_turns", bad_search)
        result = session_search(query="hello")
        assert result["result_count"] == 0
        assert result["results"] == []
        assert "error" in result["message"].lower()
