"""Tests for auto-commit hook system (Phase 1A).

Tests session state accumulation, auto-commit generation,
and hook lifecycle (start → accumulate → stop).
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from ccr.hooks.state_accumulator import (
    SessionState,
    append_tool_use,
    clear_state,
    initialize_state,
    load_state,
    save_state,
    state_path,
)


class TestSessionState:
    """Tests for the SessionState dataclass."""

    def test_empty_state_not_meaningful(self):
        state = SessionState()
        assert not state.is_meaningful()

    def test_short_what_not_meaningful(self):
        state = SessionState(what_accumulated=["short"])
        assert not state.is_meaningful()

    def test_long_what_is_meaningful(self):
        state = SessionState(what_accumulated=["a" * 60])
        assert state.is_meaningful()

    def test_multiple_what_entries_accumulate(self):
        state = SessionState(what_accumulated=["hello " * 5, "world " * 5])
        assert state.is_meaningful()

    def test_many_files_is_meaningful(self):
        state = SessionState(files_touched=["a.py", "b.py", "c.py"])
        assert state.is_meaningful()

    def test_two_files_not_meaningful_without_what(self):
        state = SessionState(files_touched=["a.py", "b.py"])
        assert not state.is_meaningful()

    def test_custom_min_chars(self):
        state = SessionState(what_accumulated=["hello"])
        assert state.is_meaningful(min_chars=3)
        assert not state.is_meaningful(min_chars=100)

    def test_to_commit_fields_basic(self):
        state = SessionState(
            what_accumulated=["Added auth module", "Fixed tests"],
            files_touched=["auth.py", "tests/test_auth.py"],
            patterns_observed=["When adding modules, update tests"],
        )
        fields = state.to_commit_fields()
        assert "Auto-commit" in fields["title"]
        assert "Added auth module" in fields["what"]
        assert "Fixed tests" in fields["what"]
        assert "auth.py" in fields["files_changed"]
        assert fields["patterns_learned"] == ["When adding modules, update tests"]

    def test_to_commit_fields_empty(self):
        state = SessionState()
        fields = state.to_commit_fields()
        assert "session work" in fields["title"].lower()
        assert fields["files_changed"] == []

    def test_to_commit_fields_deduplicates_files(self):
        state = SessionState(files_touched=["a.py", "b.py", "a.py", "c.py"])
        fields = state.to_commit_fields()
        assert fields["files_changed"] == ["a.py", "b.py", "c.py"]

    def test_to_commit_fields_caps_files_at_20(self):
        state = SessionState(files_touched=[f"file_{i}.py" for i in range(30)])
        fields = state.to_commit_fields()
        assert len(fields["files_changed"]) == 20

    def test_to_commit_fields_truncates_what(self):
        state = SessionState(what_accumulated=["x" * 600])
        fields = state.to_commit_fields()
        assert len(fields["what"]) <= 500


class TestStateIO:
    """Tests for state persistence (load/save/clear)."""

    def test_save_and_load_roundtrip(self, tmp_path):
        ccr_root = str(tmp_path)
        state = SessionState(
            files_touched=["foo.py"],
            what_accumulated=["Did stuff"],
            tool_calls=5,
            start_time=1234567890.0,
        )
        save_state(ccr_root, state)
        loaded = load_state(ccr_root)
        assert loaded.files_touched == ["foo.py"]
        assert loaded.what_accumulated == ["Did stuff"]
        assert loaded.tool_calls == 5
        assert loaded.start_time == 1234567890.0

    def test_load_missing_returns_empty(self, tmp_path):
        state = load_state(str(tmp_path))
        assert state.files_touched == []
        assert state.tool_calls == 0

    def test_load_corrupt_returns_empty(self, tmp_path):
        path = state_path(str(tmp_path))
        with open(path, "w") as f:
            f.write("not json{{{")
        state = load_state(str(tmp_path))
        assert state.files_touched == []

    def test_clear_removes_file(self, tmp_path):
        ccr_root = str(tmp_path)
        save_state(ccr_root, SessionState(tool_calls=3))
        assert os.path.isfile(state_path(ccr_root))
        clear_state(ccr_root)
        assert not os.path.isfile(state_path(ccr_root))

    def test_clear_missing_file_no_error(self, tmp_path):
        clear_state(str(tmp_path))  # Should not raise

    def test_state_path(self, tmp_path):
        p = state_path(str(tmp_path))
        assert p.endswith(".session_state.json")
        assert str(tmp_path) in p


class TestAppendToolUse:
    """Tests for the tool use accumulation function."""

    def test_append_increments_tool_calls(self, tmp_path):
        ccr_root = str(tmp_path)
        initialize_state(ccr_root)
        append_tool_use(ccr_root, "Write", "Created foo.py", ["foo.py"])
        state = load_state(ccr_root)
        assert state.tool_calls == 1
        assert "Created foo.py" in state.what_accumulated
        assert "foo.py" in state.files_touched

    def test_append_multiple_accumulates(self, tmp_path):
        ccr_root = str(tmp_path)
        initialize_state(ccr_root)
        append_tool_use(ccr_root, "Write", "Created foo.py", ["foo.py"])
        append_tool_use(ccr_root, "Edit", "Modified bar.py", ["bar.py"])
        append_tool_use(ccr_root, "Bash", "Ran tests")
        state = load_state(ccr_root)
        assert state.tool_calls == 3
        assert len(state.what_accumulated) == 3
        assert "foo.py" in state.files_touched
        assert "bar.py" in state.files_touched

    def test_append_deduplicates_files(self, tmp_path):
        ccr_root = str(tmp_path)
        initialize_state(ccr_root)
        append_tool_use(ccr_root, "Write", "First edit", ["foo.py"])
        append_tool_use(ccr_root, "Edit", "Second edit", ["foo.py"])
        state = load_state(ccr_root)
        assert state.files_touched.count("foo.py") == 1

    def test_append_empty_summary_only_increments_count(self, tmp_path):
        ccr_root = str(tmp_path)
        initialize_state(ccr_root)
        append_tool_use(ccr_root, "Read", "", None)
        state = load_state(ccr_root)
        assert state.tool_calls == 1
        assert state.what_accumulated == []

    def test_append_truncates_long_summary(self, tmp_path):
        ccr_root = str(tmp_path)
        initialize_state(ccr_root)
        append_tool_use(ccr_root, "Bash", "x" * 500)
        state = load_state(ccr_root)
        assert len(state.what_accumulated[0]) <= 200


class TestInitializeState:
    """Tests for state initialization."""

    def test_initialize_sets_start_time(self, tmp_path):
        ccr_root = str(tmp_path)
        before = time.time()
        initialize_state(ccr_root)
        after = time.time()
        state = load_state(ccr_root)
        assert before <= state.start_time <= after

    def test_initialize_clears_previous(self, tmp_path):
        ccr_root = str(tmp_path)
        save_state(ccr_root, SessionState(tool_calls=99))
        initialize_state(ccr_root)
        state = load_state(ccr_root)
        assert state.tool_calls == 0


class TestOnToolUseHook:
    """Tests for the on_tool_use.py hook main function."""

    def test_hook_detects_write(self, tmp_path, monkeypatch):
        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        initialize_state(ccr_root)

        monkeypatch.setenv("CCR_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("CLAUDE_TOOL_NAME", "Write")
        monkeypatch.setenv("CLAUDE_TOOL_INPUT", json.dumps({"file_path": str(tmp_path / "new.py")}))

        from ccr.hooks.on_tool_use import main
        main()

        state = load_state(ccr_root)
        assert state.tool_calls == 1
        assert any("new.py" in f for f in state.files_touched)

    def test_hook_detects_pytest(self, tmp_path, monkeypatch):
        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        initialize_state(ccr_root)

        monkeypatch.setenv("CCR_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("CLAUDE_TOOL_NAME", "Bash")
        monkeypatch.setenv("CLAUDE_TOOL_INPUT", json.dumps({"command": "pytest tests/ -x -q"}))

        from ccr.hooks.on_tool_use import main
        main()

        state = load_state(ccr_root)
        assert "Ran tests" in state.what_accumulated

    def test_hook_skips_read(self, tmp_path, monkeypatch):
        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        initialize_state(ccr_root)

        monkeypatch.setenv("CCR_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("CLAUDE_TOOL_NAME", "Read")
        monkeypatch.setenv("CLAUDE_TOOL_INPUT", json.dumps({"file_path": "/tmp/foo.py"}))

        from ccr.hooks.on_tool_use import main
        main()

        state = load_state(ccr_root)
        assert state.tool_calls == 0

    def test_hook_skips_without_ccr_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CCR_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("CLAUDE_TOOL_NAME", "Write")
        monkeypatch.setenv("CLAUDE_TOOL_INPUT", json.dumps({"file_path": "/tmp/foo.py"}))

        from ccr.hooks.on_tool_use import main
        main()  # Should not raise


class TestOnStopHook:
    """Tests for the on_stop.py hook main function."""

    def test_stop_auto_commits_meaningful_state(self, tmp_path, monkeypatch):
        # Set up .ccr/ directory
        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        os.makedirs(os.path.join(ccr_root, "branches"), exist_ok=True)
        with open(os.path.join(ccr_root, "active_branch.txt"), "w") as f:
            f.write("main")

        # Save meaningful state
        state = SessionState(
            what_accumulated=["Implemented auth module with JWT tokens and role checks"],
            files_touched=["auth.py", "tests/test_auth.py", "config.py"],
            tool_calls=10,
        )
        save_state(ccr_root, state)

        monkeypatch.setenv("CCR_PROJECT_ROOT", str(tmp_path))

        from ccr.hooks.on_stop import main
        main()

        # Session state should be cleared
        assert not os.path.isfile(state_path(ccr_root))

    def test_stop_skips_trivial_state(self, tmp_path, monkeypatch):
        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        os.makedirs(os.path.join(ccr_root, "branches"), exist_ok=True)
        with open(os.path.join(ccr_root, "active_branch.txt"), "w") as f:
            f.write("main")

        # Save trivial state
        save_state(ccr_root, SessionState(tool_calls=1))

        monkeypatch.setenv("CCR_PROJECT_ROOT", str(tmp_path))

        from ccr.hooks.on_stop import main
        main()

        # State should still be cleared
        assert not os.path.isfile(state_path(ccr_root))


class TestQuickCosine:
    """Tests for the ONNX quick_cosine convenience function."""

    def test_returns_none_without_onnx(self, monkeypatch):
        """Without ONNX deps, quick_cosine returns None."""
        from ccr.context import embeddings
        monkeypatch.setattr(embeddings, "SEMANTIC_AVAILABLE", False)
        monkeypatch.setattr(embeddings, "_cached_model", None)

        result = embeddings.quick_cosine("hello world", "hello world")
        assert result is None

    def test_signature_accepts_two_strings(self):
        """Verify function signature."""
        from ccr.context.embeddings import quick_cosine
        import inspect
        sig = inspect.signature(quick_cosine)
        params = list(sig.parameters.keys())
        assert params == ["text_a", "text_b"]
