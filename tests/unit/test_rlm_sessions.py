"""Tests for RLM multi-session management (Stream C)."""

from __future__ import annotations

import os
import time

import pytest
from mcp.server.fastmcp.exceptions import ToolError

import ccr.mcp_server as mcp_mod
from ccr.mcp_server import (
    _init,
    rlm_execute,
    rlm_finalize,
    rlm_init,
)
import ccr.mcp.server as _srv_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

    src = tmp_path / "hello.py"
    src.write_text(
        "def greet(name):\n    return f'Hello, {name}!'\n\nclass Greeter:\n    pass\n"
    )
    (tmp_path / "utils.py").write_text("def helper():\n    pass\n")

    _init(str(tmp_path))
    yield tmp_path

    # Cleanup: reset all session state and legacy _repl
    _srv_mod._repl_sessions.clear()
    _srv_mod._repl_session_ttl.clear()
    mcp_mod._memory = None
    mcp_mod._playbook = None
    mcp_mod._global_playbook = None
    mcp_mod._repo_index = None
    mcp_mod._repl = None


# ---------------------------------------------------------------------------
# C2: rlm_init session_id management
# ---------------------------------------------------------------------------


class TestRlmInitAutoSessionId:
    """Call rlm_init without session_id; verify auto-generated ID in response."""

    def test_auto_session_id_starts_with_rlm(self):
        result = rlm_init("Analyze the code")
        assert result["session_id"].startswith("rlm-"), (
            f"Expected session_id to start with 'rlm-', got: {result['session_id']!r}"
        )

    def test_auto_session_id_in_message(self):
        result = rlm_init("Analyze the code")
        sid = result["session_id"]
        assert sid in result["message"], (
            f"Expected session_id {sid!r} to appear in message:\n{result['message']}"
        )

    def test_auto_session_id_registered_in_sessions(self):
        result = rlm_init("Test auto-session")
        sid = result["session_id"]
        assert sid in _srv_mod._repl_sessions, (
            f"Session '{sid}' not found in _repl_sessions"
        )

    def test_auto_session_id_unique_each_call(self):
        r1 = rlm_init("First task")
        sid1 = r1["session_id"]
        r2 = rlm_init("Second task")
        sid2 = r2["session_id"]
        assert sid1 != sid2, "Two consecutive rlm_init calls should produce unique session_ids"

    def test_auto_session_ttl_set(self):
        before = time.time()
        result = rlm_init("Test TTL")
        after = time.time()
        sid = result["session_id"]
        ttl = _srv_mod._repl_session_ttl.get(sid)
        assert ttl is not None, "TTL should be set for auto-generated session"
        assert before <= ttl <= after, f"TTL {ttl} not in expected range [{before}, {after}]"


class TestRlmInitExplicitSessionId:
    """Call rlm_init with an explicit session_id."""

    def test_explicit_session_id_stored(self):
        rlm_init("Test", session_id="my-session")
        assert "my-session" in _srv_mod._repl_sessions

    def test_explicit_session_id_in_result(self):
        result = rlm_init("Test", session_id="explicit-id")
        assert result["session_id"] == "explicit-id"

    def test_explicit_session_id_in_message(self):
        result = rlm_init("Test", session_id="named-session")
        assert "named-session" in result["message"]

    def test_explicit_session_ttl_set(self):
        before = time.time()
        rlm_init("Test", session_id="ttl-test")
        after = time.time()
        ttl = _srv_mod._repl_session_ttl.get("ttl-test")
        assert ttl is not None
        assert before <= ttl <= after

    def test_multiple_named_sessions_coexist(self):
        rlm_init("Task A", session_id="session-a")
        rlm_init("Task B", session_id="session-b")
        assert "session-a" in _srv_mod._repl_sessions
        assert "session-b" in _srv_mod._repl_sessions


class TestRlmInitReplacesSameSession:
    """Re-using the same session_id cleans up the old session."""

    def test_replace_session_updates_ttl(self):
        rlm_init("First init", session_id="replace-me")
        old_ttl = _srv_mod._repl_session_ttl.get("replace-me")

        # Small sleep to ensure TTL changes
        time.sleep(0.01)

        rlm_init("Second init", session_id="replace-me")
        new_ttl = _srv_mod._repl_session_ttl.get("replace-me")

        assert new_ttl is not None
        assert new_ttl > old_ttl, "TTL should be updated on session replacement"

    def test_replace_session_sets_session_replaced_flag(self):
        rlm_init("First", session_id="replace-flag")
        result = rlm_init("Second", session_id="replace-flag")
        assert result["session_replaced"] is True

    def test_replace_session_old_repl_is_gone(self):
        rlm_init("First", session_id="replace-repl")
        old_repl = _srv_mod._repl_sessions["replace-repl"]

        rlm_init("Second", session_id="replace-repl")
        new_repl = _srv_mod._repl_sessions["replace-repl"]

        assert new_repl is not old_repl, "New REPL should differ from old"
        assert old_repl._cleaned_up is True, "Old REPL should have been cleaned up"

    def test_replace_does_not_affect_other_sessions(self):
        rlm_init("Session A", session_id="stay-alive")
        rlm_init("Session B", session_id="replace-only-me")

        repl_a = _srv_mod._repl_sessions["stay-alive"]
        rlm_init("Session B v2", session_id="replace-only-me")

        assert "stay-alive" in _srv_mod._repl_sessions
        assert _srv_mod._repl_sessions["stay-alive"] is repl_a


# ---------------------------------------------------------------------------
# C3: rlm_execute session routing
# ---------------------------------------------------------------------------


class TestRlmExecuteSessionLookup:
    """Execute code in a named session."""

    def test_execute_in_explicit_session(self):
        rlm_init("Test", session_id="exec-session")
        result = rlm_execute("x = 42\nprint(x)", session_id="exec-session")
        assert "42" in result["message"]
        assert result["has_error"] is False

    def test_execute_persists_vars_in_session(self):
        rlm_init("Test", session_id="persist-session")
        rlm_execute("my_val = 99", session_id="persist-session")
        result = rlm_execute("print(my_val)", session_id="persist-session")
        assert "99" in result["message"]

    def test_execute_sessions_are_isolated(self):
        rlm_init("Task A", session_id="iso-a")
        rlm_init("Task B", session_id="iso-b")

        rlm_execute("secret = 'alpha'", session_id="iso-a")
        result = rlm_execute("print('secret' in dir())", session_id="iso-b")
        # session-b should not have 'secret'
        assert "True" not in result["message"]

    def test_execute_backward_compat_no_session_id(self):
        """Without session_id, falls back to most recent session / _repl."""
        rlm_init("Backward compat test")
        result = rlm_execute("print(2 + 2)")
        assert "4" in result["message"]

    def test_execute_backward_compat_uses_last_init_session(self):
        """Without session_id, uses _repl which is always the most recently initialized REPL."""
        rlm_init("Last session", session_id="last-bkcompat")
        rlm_execute("bkvar = 777", session_id="last-bkcompat")
        # Call without session_id — should use _repl (the last rlm_init'd session)
        result = rlm_execute("print(bkvar)")
        assert "777" in result["message"]


class TestRlmExecuteSessionNotFound:
    """Execute with unknown session_id raises ToolError with helpful message."""

    def test_unknown_session_raises_tool_error(self):
        with pytest.raises(ToolError) as exc_info:
            rlm_execute("print(1)", session_id="no-such-session")
        msg = str(exc_info.value)
        assert "no-such-session" in msg, f"Expected session name in error: {msg}"

    def test_error_message_lists_active_sessions(self):
        rlm_init("Active session", session_id="active-1")
        with pytest.raises(ToolError) as exc_info:
            rlm_execute("print(1)", session_id="nonexistent")
        msg = str(exc_info.value)
        assert "active-1" in msg, f"Expected active session listed in error: {msg}"

    def test_no_init_no_session_id_raises(self):
        """No rlm_init called and no session_id → should raise."""
        # _repl is None and _repl_sessions is empty
        with pytest.raises(ToolError):
            rlm_execute("print(1)")


# ---------------------------------------------------------------------------
# C4: rlm_finalize multi-variable export + atomic cleanup + session removal
# ---------------------------------------------------------------------------


class TestRlmFinalizeMultiVar:
    """Finalize with variables=[...] returns JSON dict of all values."""

    def test_multi_var_returns_both_values(self):
        import json
        rlm_init("Multi-var test", session_id="multi-session")
        rlm_execute("x = 1\ny = 2", session_id="multi-session")
        result = rlm_finalize(
            variable_name="x",  # ignored when variables= is set
            session_id="multi-session",
            variables=["x", "y"],
        )
        data = json.loads(result["message"])
        assert data["x"] == "1"
        assert data["y"] == "2"

    def test_multi_var_session_removed_after_finalize(self):
        rlm_init("Cleanup test", session_id="cleanup-mv")
        rlm_execute("a = 10\nb = 20", session_id="cleanup-mv")
        rlm_finalize(
            variable_name="a",
            session_id="cleanup-mv",
            variables=["a", "b"],
        )
        assert "cleanup-mv" not in _srv_mod._repl_sessions

    def test_multi_var_ttl_removed_after_finalize(self):
        rlm_init("TTL cleanup", session_id="ttl-cleanup")
        rlm_execute("v = 5", session_id="ttl-cleanup")
        rlm_finalize(
            variable_name="v",
            session_id="ttl-cleanup",
            variables=["v"],
        )
        assert "ttl-cleanup" not in _srv_mod._repl_session_ttl

    def test_multi_var_variable_name_joined_in_result(self):
        import json
        rlm_init("Var name join", session_id="name-join")
        rlm_execute("p = 7\nq = 8", session_id="name-join")
        result = rlm_finalize(
            variable_name="p",
            session_id="name-join",
            variables=["p", "q"],
        )
        assert "p" in result["variable_name"]
        assert "q" in result["variable_name"]

    def test_multi_var_dict_values(self):
        import json
        rlm_init("Dict var", session_id="dict-mv")
        rlm_execute("d = {'key': 'val'}\nn = 42", session_id="dict-mv")
        result = rlm_finalize(
            variable_name="d",
            session_id="dict-mv",
            variables=["d", "n"],
        )
        data = json.loads(result["message"])
        assert "key" in data["d"]  # JSON-encoded dict
        assert data["n"] == "42"


class TestRlmFinalizeAtomicCleanup:
    """Atomic: if one variable fails, session is NOT cleaned up."""

    def test_bad_var_preserves_session(self):
        rlm_init("Atomic test", session_id="atomic-session")
        rlm_execute("good_var = 42", session_id="atomic-session")
        with pytest.raises(ToolError) as exc_info:
            rlm_finalize(
                variable_name="good_var",
                session_id="atomic-session",
                variables=["good_var", "bad_var_does_not_exist"],
            )
        # Session must still exist
        assert "atomic-session" in _srv_mod._repl_sessions, (
            "Session should be preserved when finalize fails atomically"
        )
        msg = str(exc_info.value)
        assert "bad_var_does_not_exist" in msg

    def test_bad_var_does_not_remove_ttl(self):
        rlm_init("Atomic TTL", session_id="atomic-ttl")
        rlm_execute("z = 99", session_id="atomic-ttl")
        with pytest.raises(ToolError):
            rlm_finalize(
                variable_name="z",
                session_id="atomic-ttl",
                variables=["z", "missing_var"],
            )
        assert "atomic-ttl" in _srv_mod._repl_session_ttl

    def test_can_still_execute_after_failed_finalize(self):
        rlm_init("Continue after failure", session_id="continue-session")
        rlm_execute("ok_var = 55", session_id="continue-session")
        with pytest.raises(ToolError):
            rlm_finalize(
                variable_name="ok_var",
                session_id="continue-session",
                variables=["ok_var", "no_such_var"],
            )
        # Session preserved — we can still execute
        result = rlm_execute("print(ok_var)", session_id="continue-session")
        assert "55" in result["message"]

    def test_invalid_var_name_raises_without_cleanup(self):
        rlm_init("Invalid name", session_id="invalid-name-session")
        rlm_execute("good = 1", session_id="invalid-name-session")
        with pytest.raises(ToolError):
            rlm_finalize(
                variable_name="good",
                session_id="invalid-name-session",
                variables=["good", "1invalid-name"],  # invalid Python identifier
            )
        assert "invalid-name-session" in _srv_mod._repl_sessions


class TestRlmFinalizeSessionRemoval:
    """Successful finalize removes session from _repl_sessions."""

    def test_single_var_finalize_removes_session(self):
        rlm_init("Remove session", session_id="remove-single")
        rlm_execute("result = 'done'", session_id="remove-single")
        rlm_finalize(variable_name="result", session_id="remove-single")
        assert "remove-single" not in _srv_mod._repl_sessions

    def test_single_var_finalize_removes_ttl(self):
        rlm_init("Remove TTL", session_id="remove-ttl")
        rlm_execute("val = 1", session_id="remove-ttl")
        rlm_finalize(variable_name="val", session_id="remove-ttl")
        assert "remove-ttl" not in _srv_mod._repl_session_ttl

    def test_finalize_with_keep_session_does_not_remove(self):
        rlm_init("Keep session", session_id="keep-me")
        rlm_execute("answer = 42", session_id="keep-me")
        result = rlm_finalize(
            variable_name="answer",
            session_id="keep-me",
            keep_session=True,
        )
        assert "keep-me" in _srv_mod._repl_sessions, (
            "Session should be preserved when keep_session=True"
        )
        assert "Session preserved" in result["message"]

    def test_multi_var_finalize_removes_session(self):
        rlm_init("Multi remove", session_id="multi-remove")
        rlm_execute("x = 1\ny = 2", session_id="multi-remove")
        rlm_finalize(
            variable_name="x",
            session_id="multi-remove",
            variables=["x", "y"],
        )
        assert "multi-remove" not in _srv_mod._repl_sessions

    def test_finalize_unknown_session_raises_not_found(self):
        with pytest.raises(ToolError) as exc_info:
            rlm_finalize(variable_name="x", session_id="ghost-session")
        assert "ghost-session" in str(exc_info.value)

    def test_session_removed_from_sessions_dict_not_others(self):
        """Removing one session leaves others intact."""
        rlm_init("Session A", session_id="keep-a")
        rlm_init("Session B", session_id="remove-b")
        rlm_execute("done = True", session_id="remove-b")
        rlm_finalize(variable_name="done", session_id="remove-b")

        assert "remove-b" not in _srv_mod._repl_sessions
        assert "keep-a" in _srv_mod._repl_sessions
