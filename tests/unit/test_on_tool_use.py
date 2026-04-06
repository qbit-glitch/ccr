"""Tests for on_tool_use.py hook — stdin fallback and file accumulation."""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from ccr.hooks.on_tool_use import _read_tool_data_from_stdin, main


class TestReadToolDataFromStdin:
    """_read_tool_data_from_stdin() should parse Claude Code's PostToolUse JSON."""

    def test_reads_tool_name_from_stdin(self):
        payload = json.dumps({"tool_name": "Write", "tool_input": {"file_path": "x.py"}})
        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                tool_name, tool_input = _read_tool_data_from_stdin()
        assert tool_name == "Write"
        assert tool_input == {"file_path": "x.py"}

    def test_reads_edit_tool_from_stdin(self):
        payload = json.dumps({"tool_name": "Edit", "tool_input": {"file_path": "model.py"}})
        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                tool_name, tool_input = _read_tool_data_from_stdin()
        assert tool_name == "Edit"
        assert tool_input["file_path"] == "model.py"

    def test_returns_empty_on_tty_stdin(self):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            tool_name, tool_input = _read_tool_data_from_stdin()
        assert tool_name == ""
        assert tool_input == {}

    def test_returns_empty_on_empty_stdin(self):
        with patch("sys.stdin", StringIO("")):
            with patch("sys.stdin.isatty", return_value=False):
                tool_name, tool_input = _read_tool_data_from_stdin()
        assert tool_name == ""
        assert tool_input == {}

    def test_returns_empty_on_invalid_json(self):
        with patch("sys.stdin", StringIO("not json")):
            with patch("sys.stdin.isatty", return_value=False):
                tool_name, tool_input = _read_tool_data_from_stdin()
        assert tool_name == ""
        assert tool_input == {}

    def test_returns_empty_when_tool_name_absent(self):
        payload = json.dumps({"tool_input": {"file_path": "x.py"}})
        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                tool_name, tool_input = _read_tool_data_from_stdin()
        assert tool_name == ""

    def test_tool_input_none_returns_empty_dict(self):
        payload = json.dumps({"tool_name": "Bash", "tool_input": None})
        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                tool_name, tool_input = _read_tool_data_from_stdin()
        assert tool_name == "Bash"
        assert tool_input == {}


class TestMainStdinFallback:
    """main() should use stdin first and fall back to env vars."""

    def _make_ccr_root(self, tmp_path: object) -> str:
        ccr_root = os.path.join(str(tmp_path), ".ccr")
        os.makedirs(ccr_root, exist_ok=True)
        return ccr_root

    def test_stdin_populates_files_touched(self, tmp_path):
        """A Write call via stdin should accumulate the file in session state."""
        ccr_root = self._make_ccr_root(tmp_path)
        payload = json.dumps({
            "tool_name": "Write",
            "tool_input": {"file_path": str(tmp_path / "model.py")},
        })

        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                with patch.dict(os.environ, {"CCR_PROJECT_ROOT": str(tmp_path)}, clear=False):
                    with patch.dict(os.environ, {"CLAUDE_TOOL_NAME": ""}, clear=False):
                        main()

        from ccr.hooks.state_accumulator import load_state
        state = load_state(ccr_root)
        assert any("model.py" in f for f in state.files_touched), (
            f"model.py not in files_touched: {state.files_touched}"
        )

    def test_env_var_fallback_when_stdin_empty(self, tmp_path):
        """When stdin is empty, env vars should still drive accumulation."""
        ccr_root = self._make_ccr_root(tmp_path)
        file_path = str(tmp_path / "config.py")

        with patch("sys.stdin", StringIO("")):
            with patch("sys.stdin.isatty", return_value=False):
                with patch.dict(os.environ, {
                    "CCR_PROJECT_ROOT": str(tmp_path),
                    "CLAUDE_TOOL_NAME": "Write",
                    "CLAUDE_TOOL_INPUT": json.dumps({"file_path": file_path}),
                }, clear=False):
                    main()

        from ccr.hooks.state_accumulator import load_state
        state = load_state(ccr_root)
        assert any("config.py" in f for f in state.files_touched), (
            f"config.py not in files_touched: {state.files_touched}"
        )

    def test_exits_early_when_no_ccr_dir(self, tmp_path):
        """main() should return silently when .ccr/ does not exist."""
        payload = json.dumps({"tool_name": "Write", "tool_input": {"file_path": "x.py"}})
        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                with patch.dict(os.environ, {"CCR_PROJECT_ROOT": str(tmp_path)}, clear=False):
                    # Should not raise — .ccr/ missing is a silent no-op
                    main()


class TestToolCallsAlwaysIncremented:
    """tool_calls must be incremented for ALL tool invocations, not just write/test/git ones."""

    def _make_ccr_root(self, tmp_path: object) -> str:
        ccr_root = os.path.join(str(tmp_path), ".ccr")
        os.makedirs(ccr_root, exist_ok=True)
        return ccr_root

    def test_read_only_tool_increments_tool_calls(self, tmp_path):
        """A Read tool call (no summary, no files) should still increment tool_calls."""
        ccr_root = self._make_ccr_root(tmp_path)
        payload = json.dumps({
            "tool_name": "Read",
            "tool_input": {"file_path": str(tmp_path / "README.md")},
        })

        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                with patch.dict(os.environ, {"CCR_PROJECT_ROOT": str(tmp_path)}, clear=False):
                    with patch.dict(os.environ, {"CLAUDE_TOOL_NAME": ""}, clear=False):
                        main()

        from ccr.hooks.state_accumulator import load_state
        state = load_state(ccr_root)
        assert state.tool_calls == 1, f"Expected tool_calls=1, got {state.tool_calls}"
        assert state.what_accumulated == [], "Read should not append to what_accumulated"
        # Read now tracks files_touched (for commit title quality) but no summary
        assert any("README.md" in f for f in state.files_touched), (
            f"Read should track files_touched, got: {state.files_touched}"
        )

    def test_glob_tool_increments_tool_calls(self, tmp_path):
        """A Glob tool call should increment tool_calls without touching summary/files."""
        ccr_root = self._make_ccr_root(tmp_path)
        payload = json.dumps({
            "tool_name": "Glob",
            "tool_input": {"pattern": "**/*.py"},
        })

        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                with patch.dict(os.environ, {"CCR_PROJECT_ROOT": str(tmp_path)}, clear=False):
                    with patch.dict(os.environ, {"CLAUDE_TOOL_NAME": ""}, clear=False):
                        main()

        from ccr.hooks.state_accumulator import load_state
        state = load_state(ccr_root)
        assert state.tool_calls == 1, f"Expected tool_calls=1, got {state.tool_calls}"
        assert state.what_accumulated == []
        assert state.files_touched == []

    def test_grep_tool_increments_tool_calls(self, tmp_path):
        """A Grep tool call should increment tool_calls without touching summary/files."""
        ccr_root = self._make_ccr_root(tmp_path)
        payload = json.dumps({
            "tool_name": "Grep",
            "tool_input": {"pattern": "def main", "path": str(tmp_path)},
        })

        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                with patch.dict(os.environ, {"CCR_PROJECT_ROOT": str(tmp_path)}, clear=False):
                    with patch.dict(os.environ, {"CLAUDE_TOOL_NAME": ""}, clear=False):
                        main()

        from ccr.hooks.state_accumulator import load_state
        state = load_state(ccr_root)
        assert state.tool_calls == 1

    def test_multiple_read_only_calls_accumulate_tool_calls(self, tmp_path):
        """Multiple read-only calls should each increment tool_calls."""
        ccr_root = self._make_ccr_root(tmp_path)

        for tool in ("Read", "Glob", "Grep"):
            payload = json.dumps({"tool_name": tool, "tool_input": {}})
            with patch("sys.stdin", StringIO(payload)):
                with patch("sys.stdin.isatty", return_value=False):
                    with patch.dict(os.environ, {"CCR_PROJECT_ROOT": str(tmp_path)}, clear=False):
                        with patch.dict(os.environ, {"CLAUDE_TOOL_NAME": ""}, clear=False):
                            main()

        from ccr.hooks.state_accumulator import load_state
        state = load_state(ccr_root)
        assert state.tool_calls == 3, f"Expected 3, got {state.tool_calls}"

    def test_write_tool_increments_tool_calls_via_append_tool_use(self, tmp_path):
        """Write tool should still increment tool_calls (via append_tool_use path)."""
        ccr_root = self._make_ccr_root(tmp_path)
        payload = json.dumps({
            "tool_name": "Write",
            "tool_input": {"file_path": str(tmp_path / "out.py")},
        })

        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                with patch.dict(os.environ, {"CCR_PROJECT_ROOT": str(tmp_path)}, clear=False):
                    with patch.dict(os.environ, {"CLAUDE_TOOL_NAME": ""}, clear=False):
                        main()

        from ccr.hooks.state_accumulator import load_state
        state = load_state(ccr_root)
        assert state.tool_calls == 1
        assert any("out.py" in f for f in state.files_touched)

    def test_bash_without_meaningful_cmd_increments_tool_calls(self, tmp_path):
        """A plain Bash call (e.g. ls) should increment tool_calls but not add summary."""
        ccr_root = self._make_ccr_root(tmp_path)
        payload = json.dumps({
            "tool_name": "Bash",
            "tool_input": {"command": "ls -la"},
        })

        with patch("sys.stdin", StringIO(payload)):
            with patch("sys.stdin.isatty", return_value=False):
                with patch.dict(os.environ, {"CCR_PROJECT_ROOT": str(tmp_path)}, clear=False):
                    with patch.dict(os.environ, {"CLAUDE_TOOL_NAME": ""}, clear=False):
                        main()

        from ccr.hooks.state_accumulator import load_state
        state = load_state(ccr_root)
        assert state.tool_calls == 1
        assert state.what_accumulated == [], "Plain bash (ls) should not append to what_accumulated"
