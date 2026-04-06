"""Tests for auto-baseline commit feature.

Covers:
- extract_baseline_summary() in state_accumulator
- .session_explicit_commit marker written by gcc_commit
- _auto_baseline_commit() in on_stop
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ccr.hooks.state_accumulator import SessionState, extract_baseline_summary


# ---------------------------------------------------------------------------
# extract_baseline_summary
# ---------------------------------------------------------------------------

class TestExtractBaselineSummary:
    def _make_turns(self, messages: list[str]) -> list[dict]:
        return [{"user_message": m, "assistant_message": "ok"} for m in messages]

    def test_uses_first_user_message_as_title_when_no_files_or_ops(self):
        """User message is priority 3 — only used when no files and no what_accumulated."""
        state = SessionState()
        turns = self._make_turns(["How do I train this model?"])
        result = extract_baseline_summary(state, turns)
        assert result["title"] == "[auto] How do I train this model?"

    def test_files_touched_takes_priority_over_user_message(self):
        """files_touched is priority 1 — always preferred over user message."""
        state = SessionState(files_touched=["train.py", "config.yaml"])
        turns = self._make_turns(["How do I train this model?"])
        result = extract_baseline_summary(state, turns)
        assert result["title"].startswith("[auto] Edited ")
        assert "train.py" in result["title"]

    def test_what_accumulated_takes_priority_over_user_message(self):
        """what_accumulated is priority 2 — used when no files but ops were recorded."""
        state = SessionState(what_accumulated=["Modified model.py"])
        turns = self._make_turns(["How do I train this model?"])
        result = extract_baseline_summary(state, turns)
        assert "Modified model.py" in result["title"]
        assert "How do I" not in result["title"]

    def test_title_shows_multiple_files_with_suffix(self):
        """Title for 4+ files shows '+N more' suffix."""
        state = SessionState(files_touched=["a.py", "b.py", "c.py", "d.py"])
        result = extract_baseline_summary(state, [])
        assert "+1 more" in result["title"]
        assert "a.py" in result["title"]

    def test_title_three_files_no_suffix(self):
        """Exactly 3 files — no '+N more' suffix."""
        state = SessionState(files_touched=["a.py", "b.py", "c.py"])
        result = extract_baseline_summary(state, [])
        assert "more" not in result["title"]
        assert "a.py" in result["title"]

    def test_title_capped_at_80_chars(self):
        state = SessionState()
        long_msg = "x" * 100
        turns = self._make_turns([long_msg])
        result = extract_baseline_summary(state, turns)
        assert result["title"].startswith("[auto] ")
        assert len(result["title"]) <= len("[auto] ") + 80

    def test_title_prefixed_with_auto(self):
        state = SessionState()
        turns = self._make_turns(["Fix the bug"])
        result = extract_baseline_summary(state, turns)
        assert result["title"].startswith("[auto] ")

    def test_empty_turns_produces_default_title(self):
        state = SessionState()
        result = extract_baseline_summary(state, [])
        assert result["title"] == "[auto] Session baseline"

    def test_blank_user_messages_treated_as_empty(self):
        state = SessionState()
        turns = [{"user_message": "   ", "assistant_message": "ok"}]
        result = extract_baseline_summary(state, turns)
        assert result["title"] == "[auto] Session baseline"

    def test_what_includes_turn_summaries(self):
        state = SessionState()
        turns = self._make_turns(["First question", "Second question"])
        result = extract_baseline_summary(state, turns)
        assert "Turn 1" in result["what"]
        assert "Turn 2" in result["what"]

    def test_what_capped_at_500_chars(self):
        state = SessionState()
        turns = self._make_turns(["x" * 200, "y" * 200, "z" * 200])
        result = extract_baseline_summary(state, turns)
        assert len(result["what"]) <= 500

    def test_files_from_state_included_in_what(self):
        state = SessionState(files_touched=["train.py", "config.yaml"])
        result = extract_baseline_summary(state, [])
        assert "train.py" in result["what"]

    def test_files_from_state_in_files_changed(self):
        state = SessionState(files_touched=["a.py", "b.py"])
        result = extract_baseline_summary(state, [])
        assert "a.py" in result["files_changed"]
        assert "b.py" in result["files_changed"]

    def test_files_deduped_and_capped_at_20(self):
        files = [f"file{i}.py" for i in range(30)]
        state = SessionState(files_touched=files)
        result = extract_baseline_summary(state, [])
        assert len(result["files_changed"]) <= 20

    def test_why_field_present(self):
        state = SessionState()
        result = extract_baseline_summary(state, [])
        assert "baseline" in result["why"].lower()

    def test_next_step_empty(self):
        state = SessionState()
        result = extract_baseline_summary(state, [])
        assert result["next_step"] == ""

    def test_at_most_3_turns_used(self):
        state = SessionState()
        turns = self._make_turns([f"question {i}" for i in range(10)])
        result = extract_baseline_summary(state, turns)
        # Should not include "Turn 4" or higher
        assert "Turn 4" not in result["what"]
        assert "Turn 3" in result["what"]


# ---------------------------------------------------------------------------
# .session_explicit_commit marker written by gcc_commit
# ---------------------------------------------------------------------------

class TestExplicitCommitMarker:
    def _invoke_gcc_commit(self, tmp_path, commit_return_value: str) -> None:
        """Helper: invoke gcc_commit with _srv fully mocked."""
        mock_mem = MagicMock()
        mock_mem.commit.return_value = commit_return_value
        mock_mem.config.auto_extract_patterns = False

        with patch("ccr.mcp.gcc_tools._srv") as mock_srv:
            mock_srv._project_root = str(tmp_path)
            mock_srv._ensure_memory.return_value = mock_mem
            mock_srv._get_sub_client.return_value = None
            mock_srv._triple_store = None
            mock_srv._state_lock = MagicMock()
            mock_srv._state_lock.__enter__ = MagicMock(return_value=None)
            mock_srv._state_lock.__exit__ = MagicMock(return_value=False)

            from ccr.mcp.gcc_tools import gcc_commit
            gcc_commit(
                title="Test commit",
                what="did something",
                why="because",
                files_changed=["foo.py"],
                next_step="done",
            )

    def test_marker_written_after_successful_commit(self, tmp_path):
        ccr_dir = tmp_path / ".ccr"
        ccr_dir.mkdir()
        self._invoke_gcc_commit(tmp_path, "[C001] Commit saved")
        marker = ccr_dir / ".session_explicit_commit"
        assert marker.exists(), "Marker file should be created after successful gcc_commit"

    def test_marker_not_written_when_commit_returns_error(self, tmp_path):
        ccr_dir = tmp_path / ".ccr"
        ccr_dir.mkdir()
        self._invoke_gcc_commit(tmp_path, "Error: something went wrong")
        marker = ccr_dir / ".session_explicit_commit"
        assert not marker.exists(), "Marker should NOT be written when commit returns Error"

    def test_marker_not_written_when_commit_rejected(self, tmp_path):
        ccr_dir = tmp_path / ".ccr"
        ccr_dir.mkdir()
        self._invoke_gcc_commit(tmp_path, "[REJECTED] Too similar to existing commit")
        marker = ccr_dir / ".session_explicit_commit"
        assert not marker.exists(), "Marker should NOT be written when commit is rejected"

    def test_marker_not_written_when_project_root_empty(self, tmp_path):
        mock_mem = MagicMock()
        mock_mem.commit.return_value = "[C001] Commit saved"
        mock_mem.config.auto_extract_patterns = False

        with patch("ccr.mcp.gcc_tools._srv") as mock_srv:
            mock_srv._project_root = ""  # Empty project root
            mock_srv._ensure_memory.return_value = mock_mem
            mock_srv._get_sub_client.return_value = None
            mock_srv._triple_store = None
            mock_srv._state_lock = MagicMock()
            mock_srv._state_lock.__enter__ = MagicMock(return_value=None)
            mock_srv._state_lock.__exit__ = MagicMock(return_value=False)

            from ccr.mcp.gcc_tools import gcc_commit
            # Should not raise even with empty project_root
            gcc_commit(
                title="Test commit",
                what="did something",
                why="because",
                files_changed=["foo.py"],
                next_step="done",
            )
        # No assertion needed — just verify no crash


# ---------------------------------------------------------------------------
# Double-commit prevention: state.is_meaningful() path writes marker
# ---------------------------------------------------------------------------

class TestDoubleCommitPrevention:
    """Verify that when state.is_meaningful() commits, the marker is written
    so _auto_baseline_commit skips — no two commits per session."""

    def _make_mem(self, tmp_path):
        mem = MagicMock()
        mem.ccr_root = str(tmp_path / ".ccr")
        os.makedirs(mem.ccr_root, exist_ok=True)
        return mem

    def test_after_meaningful_commit_baseline_skips(self, tmp_path):
        """Integration: when explicit marker is present, _auto_baseline_commit skips."""
        from ccr.hooks.on_stop import _auto_baseline_commit

        mem = self._make_mem(tmp_path)
        # Simulate: gcc_commit was called explicitly and wrote the marker
        marker = os.path.join(mem.ccr_root, ".session_explicit_commit")
        with open(marker, "w") as f:
            f.write("1")

        state = SessionState(files_touched=["train.py"])
        store = MagicMock()
        store.get_session_turns.return_value = [
            {"user_message": "Fix the training loop", "assistant_message": "done"}
        ]

        _auto_baseline_commit(mem, state, store, "ses_abc")

        mem.commit.assert_not_called()
        assert not os.path.exists(marker), "Marker cleaned up after use"

    def test_marker_peek_does_not_delete_it(self, tmp_path):
        """on_stop.py peeks at the marker (does not delete it) before is_meaningful().

        The marker must still be present after the peek so _auto_baseline_commit
        can find it and perform cleanup with its own read+delete logic.
        This verifies single-writer semantics: only _auto_baseline_commit deletes the marker.
        """
        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root, exist_ok=True)
        marker = os.path.join(ccr_root, ".session_explicit_commit")
        with open(marker, "w") as f:
            f.write("1")

        # Simulate the peek as written in on_stop.main()
        _explicit_marker = os.path.join(ccr_root, ".session_explicit_commit")
        _had_explicit_commit = os.path.isfile(_explicit_marker)

        # After peek: marker must still exist (peek does not delete)
        assert _had_explicit_commit, "Peek should detect the marker"
        assert os.path.isfile(marker), "Peek must not delete the marker"

        # _auto_baseline_commit is the sole consumer: it reads + deletes the marker
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        mem.ccr_root = ccr_root  # reuse same ccr_root
        state = SessionState(files_touched=["train.py"])
        store = MagicMock()
        store.get_session_turns.return_value = []

        _auto_baseline_commit(mem, state, store, "ses_x")

        # Marker must now be gone — cleaned up by _auto_baseline_commit
        assert not os.path.isfile(marker), "Marker must be cleaned up by _auto_baseline_commit"
        # commit not called because had_explicit=True
        mem.commit.assert_not_called()


# ---------------------------------------------------------------------------
# _auto_baseline_commit
# ---------------------------------------------------------------------------

class TestAutoBaselineCommit:
    """Tests for the _auto_baseline_commit function in on_stop."""

    def _make_mem(self, tmp_path):
        mem = MagicMock()
        mem.ccr_root = str(tmp_path / ".ccr")
        os.makedirs(mem.ccr_root, exist_ok=True)
        return mem

    def _make_store(self, turns=None):
        store = MagicMock()
        store.get_session_turns.return_value = turns or []
        return store

    def test_creates_baseline_when_no_marker_and_has_turns(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        state = SessionState()
        store = self._make_store([{"user_message": "Help me fix this", "assistant_message": "ok"}])

        _auto_baseline_commit(mem, state, store, "ses_123")

        mem.commit.assert_called_once()
        call_kwargs = mem.commit.call_args[1]
        assert call_kwargs["author"] == "[auto-baseline]"
        assert call_kwargs["title"].startswith("[auto]")

    def test_creates_baseline_when_no_marker_and_has_files(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        state = SessionState(files_touched=["train.py"])
        store = self._make_store([])

        _auto_baseline_commit(mem, state, store, "ses_123")

        mem.commit.assert_called_once()

    def test_skips_when_explicit_marker_exists(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        # Write the marker
        marker = os.path.join(mem.ccr_root, ".session_explicit_commit")
        with open(marker, "w") as f:
            f.write("1")

        state = SessionState()
        store = self._make_store([{"user_message": "hello", "assistant_message": "hi"}])

        _auto_baseline_commit(mem, state, store, "ses_123")

        mem.commit.assert_not_called()

    def test_skips_truly_empty_session(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        state = SessionState()  # no files
        store = self._make_store([])  # no turns

        _auto_baseline_commit(mem, state, store, "ses_123")

        mem.commit.assert_not_called()

    def test_skips_trivial_greeting_no_files(self, tmp_path):
        """Quality gate: 'hi' / short messages with no files should not create a baseline."""
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        state = SessionState()  # no files
        store = self._make_store([{"user_message": "hi", "assistant_message": "Hello!"}])

        _auto_baseline_commit(mem, state, store, "ses_123")

        mem.commit.assert_not_called()

    def test_creates_baseline_for_substantive_turn_no_files(self, tmp_path):
        """Quality gate: turn with >= 3 words qualifies even without files."""
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        state = SessionState()
        store = self._make_store([
            {"user_message": "Explain how attention works in transformers", "assistant_message": "ok"}
        ])

        _auto_baseline_commit(mem, state, store, "ses_123")

        mem.commit.assert_called_once()

    def test_marker_cleaned_up_when_present(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        marker = os.path.join(mem.ccr_root, ".session_explicit_commit")
        with open(marker, "w") as f:
            f.write("1")

        state = SessionState()
        store = self._make_store([])

        _auto_baseline_commit(mem, state, store, "ses_123")

        assert not os.path.exists(marker), "Marker should be removed regardless of path taken"

    def test_marker_cleanup_does_not_crash_when_absent(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        # No marker file — should not raise FileNotFoundError
        state = SessionState()
        store = self._make_store([])

        _auto_baseline_commit(mem, state, store, "ses_123")  # must not raise

    def test_no_crash_when_session_store_raises(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        state = SessionState(files_touched=["a.py"])
        store = MagicMock()
        store.get_session_turns.side_effect = Exception("DB error")

        # Should not raise — falls back to empty turns, still commits because has_files=True
        _auto_baseline_commit(mem, state, store, "ses_123")

    def test_no_crash_when_mem_commit_raises(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        mem.commit.side_effect = Exception("Commit failed")
        state = SessionState(files_touched=["a.py"])
        store = self._make_store([{"user_message": "hello", "assistant_message": "hi"}])

        _auto_baseline_commit(mem, state, store, "ses_123")  # must not raise

    def test_skips_when_session_id_empty_and_no_files(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        state = SessionState()
        store = self._make_store([])

        _auto_baseline_commit(mem, state, store, "")  # empty session_id

        mem.commit.assert_not_called()

    def test_baseline_title_contains_first_user_message(self, tmp_path):
        from ccr.hooks.on_stop import _auto_baseline_commit
        mem = self._make_mem(tmp_path)
        state = SessionState()
        store = self._make_store([
            {"user_message": "Explain how transformers work", "assistant_message": "ok"}
        ])

        _auto_baseline_commit(mem, state, store, "ses_abc")

        call_kwargs = mem.commit.call_args[1]
        assert "Explain how transformers work" in call_kwargs["title"]
