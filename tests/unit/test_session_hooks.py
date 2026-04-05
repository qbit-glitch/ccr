"""Tests for on_stop.py transcript reconciliation.

Verifies that the Stop hook:
- Parses JSONL transcripts correctly
- Inserts missing turns with source='transcript'
- Does not duplicate turns already logged via session_log_turn
- Handles missing/malformed transcripts gracefully
- Reads session_id from both the .current_session_id file and from stdin payload
"""

from __future__ import annotations

import io
import json
import os
import sys
from unittest.mock import patch

import pytest

from ccr.core.session_store import SessionStore
from ccr.hooks.on_stop import (
    _extract_text_from_content,
    _parse_transcript,
    _read_stdin_payload,
    _reconcile_transcript,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "sessions.db")


@pytest.fixture
def store(db_path):
    s = SessionStore(db_path)
    yield s
    s.close()


def _write_transcript(path: str, turns: list[dict]) -> None:
    """Write a JSONL transcript file from a list of {user, assistant} dicts."""
    lines = []
    for t in turns:
        lines.append(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": t["user"]},
        }))
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": t["assistant"]}
            ]},
        }))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# _extract_text_from_content
# ---------------------------------------------------------------------------


class TestExtractTextFromContent:
    def test_plain_string(self):
        assert _extract_text_from_content("hello world") == "hello world"

    def test_list_with_text_block(self):
        content = [{"type": "text", "text": "response text"}]
        assert _extract_text_from_content(content) == "response text"

    def test_list_with_multiple_text_blocks(self):
        content = [
            {"type": "text", "text": "part 1"},
            {"type": "text", "text": "part 2"},
        ]
        result = _extract_text_from_content(content)
        assert "part 1" in result
        assert "part 2" in result

    def test_list_skips_tool_use_blocks(self):
        content = [
            {"type": "tool_use", "id": "abc", "name": "Read", "input": {}},
            {"type": "text", "text": "final answer"},
        ]
        assert _extract_text_from_content(content) == "final answer"

    def test_list_with_only_tool_use_returns_empty(self):
        content = [{"type": "tool_use", "id": "abc", "name": "Write", "input": {}}]
        assert _extract_text_from_content(content) == ""

    def test_empty_string(self):
        assert _extract_text_from_content("") == ""

    def test_empty_list(self):
        assert _extract_text_from_content([]) == ""

    def test_non_string_non_list(self):
        assert _extract_text_from_content(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _parse_transcript
# ---------------------------------------------------------------------------


class TestParseTranscript:
    def test_parses_two_turn_transcript(self, tmp_path):
        tp = str(tmp_path / "transcript.jsonl")
        _write_transcript(tp, [
            {"user": "Hello there", "assistant": "Hi! How can I help?"},
            {"user": "What is 2+2?", "assistant": "The answer is 4."},
        ])
        turns = _parse_transcript(tp)
        assert len(turns) == 2
        assert turns[0]["user_message"] == "Hello there"
        assert turns[0]["assistant_message"] == "Hi! How can I help?"
        assert turns[1]["user_message"] == "What is 2+2?"
        assert turns[1]["assistant_message"] == "The answer is 4."

    def test_parses_single_turn(self, tmp_path):
        tp = str(tmp_path / "transcript.jsonl")
        _write_transcript(tp, [
            {"user": "ping", "assistant": "pong"},
        ])
        turns = _parse_transcript(tp)
        assert len(turns) == 1

    def test_missing_file_returns_empty(self, tmp_path):
        turns = _parse_transcript(str(tmp_path / "nonexistent.jsonl"))
        assert turns == []

    def test_empty_file_returns_empty(self, tmp_path):
        tp = str(tmp_path / "empty.jsonl")
        with open(tp, "w") as f:
            f.write("")
        turns = _parse_transcript(tp)
        assert turns == []

    def test_malformed_lines_are_skipped(self, tmp_path):
        tp = str(tmp_path / "bad.jsonl")
        with open(tp, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"type": "user", "message": {"role": "user", "content": "q1"}}) + "\n")
            f.write(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "a1"}]}}) + "\n")
        turns = _parse_transcript(tp)
        assert len(turns) == 1
        assert turns[0]["user_message"] == "q1"

    def test_skips_tool_use_only_assistant_messages(self, tmp_path):
        """Assistant messages with only tool_use blocks don't create a turn."""
        tp = str(tmp_path / "transcript.jsonl")
        lines = [
            json.dumps({"type": "user", "message": {"role": "user", "content": "run a file"}}),
            # Tool-use only assistant — no text
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
            ]}}),
            # Tool result (user turn)
            json.dumps({"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file contents"}
            ]}}),
            # Final assistant text
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Here is the result."}
            ]}}),
        ]
        with open(tp, "w") as f:
            f.write("\n".join(lines) + "\n")
        turns = _parse_transcript(tp)
        # Should get exactly one turn: the user message paired with the final text response
        assert len(turns) == 1
        assert turns[0]["assistant_message"] == "Here is the result."

    def test_string_content_user_message(self, tmp_path):
        """User message content as plain string is handled."""
        tp = str(tmp_path / "transcript.jsonl")
        lines = [
            json.dumps({"type": "user", "message": {"role": "user", "content": "simple question"}}),
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "simple answer"}}),
        ]
        with open(tp, "w") as f:
            f.write("\n".join(lines) + "\n")
        turns = _parse_transcript(tp)
        assert len(turns) == 1
        assert turns[0]["user_message"] == "simple question"
        assert turns[0]["assistant_message"] == "simple answer"


# ---------------------------------------------------------------------------
# _reconcile_transcript
# ---------------------------------------------------------------------------


class TestReconcileTranscript:
    def test_inserts_missing_turns_as_transcript_source(self, tmp_path, store):
        """Turns in transcript but not in DB are inserted with source='transcript'."""
        sid = store.create_session()
        tp = str(tmp_path / "transcript.jsonl")
        _write_transcript(tp, [
            {"user": "user turn 1", "assistant": "assistant turn 1"},
            {"user": "user turn 2", "assistant": "assistant turn 2"},
        ])

        inserted = _reconcile_transcript(store, sid, tp)
        assert inserted == 2

        turns = store.get_session_turns(sid)
        assert len(turns) == 2
        for t in turns:
            assert t["source"] == "transcript"

    def test_no_duplicate_if_already_logged(self, tmp_path, store):
        """Turns already in DB are not re-inserted."""
        sid = store.create_session()
        # Pre-log 2 turns via session_log_turn (source='direct')
        store.log_turn(sid, "user turn 1", "assistant turn 1", source="direct")
        store.log_turn(sid, "user turn 2", "assistant turn 2", source="direct")

        tp = str(tmp_path / "transcript.jsonl")
        _write_transcript(tp, [
            {"user": "user turn 1", "assistant": "assistant turn 1"},
            {"user": "user turn 2", "assistant": "assistant turn 2"},
        ])

        inserted = _reconcile_transcript(store, sid, tp)
        assert inserted == 0

        # DB should still have only 2 turns (no duplicates)
        turns = store.get_session_turns(sid)
        assert len(turns) == 2

    def test_partial_reconciliation(self, tmp_path, store):
        """Only the missing tail of turns is inserted."""
        sid = store.create_session()
        # Already have turn 1 logged
        store.log_turn(sid, "user turn 1", "assistant turn 1", source="direct")

        tp = str(tmp_path / "transcript.jsonl")
        _write_transcript(tp, [
            {"user": "user turn 1", "assistant": "assistant turn 1"},
            {"user": "user turn 2", "assistant": "assistant turn 2"},  # missing
            {"user": "user turn 3", "assistant": "assistant turn 3"},  # missing
        ])

        inserted = _reconcile_transcript(store, sid, tp)
        assert inserted == 2

        turns = store.get_session_turns(sid)
        assert len(turns) == 3
        assert turns[0]["source"] == "direct"
        assert turns[1]["source"] == "transcript"
        assert turns[2]["source"] == "transcript"

    def test_empty_transcript_inserts_nothing(self, tmp_path, store):
        sid = store.create_session()
        tp = str(tmp_path / "empty.jsonl")
        with open(tp, "w") as f:
            f.write("")

        inserted = _reconcile_transcript(store, sid, tp)
        assert inserted == 0
        assert store.get_session_turns(sid) == []

    def test_nonexistent_transcript_inserts_nothing(self, tmp_path, store):
        sid = store.create_session()
        inserted = _reconcile_transcript(store, sid, str(tmp_path / "missing.jsonl"))
        assert inserted == 0

    def test_reconciled_turns_have_correct_content(self, tmp_path, store):
        """Verify user_message and assistant_message are stored correctly."""
        sid = store.create_session()
        tp = str(tmp_path / "transcript.jsonl")
        _write_transcript(tp, [
            {"user": "What is Python?", "assistant": "A high-level programming language."},
        ])

        _reconcile_transcript(store, sid, tp)
        turns = store.get_session_turns(sid)
        assert turns[0]["user_message"] == "What is Python?"
        assert turns[0]["assistant_message"] == "A high-level programming language."


# ---------------------------------------------------------------------------
# _read_stdin_payload
# ---------------------------------------------------------------------------


class TestReadStdinPayload:
    def test_reads_valid_json(self):
        payload = {"session_id": "ses_123", "transcript_path": "/tmp/t.jsonl"}
        with patch("sys.stdin", io.StringIO(json.dumps(payload))):
            with patch.object(sys.stdin, "isatty", return_value=False):
                result = _read_stdin_payload()
        assert result == payload

    def test_returns_empty_on_invalid_json(self):
        with patch("sys.stdin", io.StringIO("not json")):
            with patch.object(sys.stdin, "isatty", return_value=False):
                result = _read_stdin_payload()
        assert result == {}

    def test_returns_empty_on_tty(self):
        with patch("sys.stdin", io.StringIO("")):
            with patch.object(sys.stdin, "isatty", return_value=True):
                result = _read_stdin_payload()
        assert result == {}


# ---------------------------------------------------------------------------
# Integration: on_stop.main() reconciles transcript
# ---------------------------------------------------------------------------


class TestOnStopIntegration:
    def _make_ccr_dir(self, tmp_path) -> str:
        """Set up a minimal .ccr/ directory."""
        ccr_root = tmp_path / ".ccr"
        ccr_root.mkdir()
        # Create a minimal summary.json so MemoryManager doesn't crash
        return str(ccr_root)

    def test_on_stop_reconciles_transcript(self, tmp_path, monkeypatch):
        """main() inserts missing turns when transcript_path is in stdin."""
        ccr_root = self._make_ccr_dir(tmp_path)

        # Set up store + session
        db_path = os.path.join(ccr_root, "sessions.db")
        store = SessionStore(db_path)
        sid = store.create_session(project=str(tmp_path))
        store.close()

        # Write session ID file
        id_file = os.path.join(ccr_root, ".current_session_id")
        with open(id_file, "w") as f:
            f.write(sid)

        # Write transcript with 2 turns
        tp = str(tmp_path / "transcript.jsonl")
        _write_transcript(tp, [
            {"user": "Question one", "assistant": "Answer one"},
            {"user": "Question two", "assistant": "Answer two"},
        ])

        # Stub out MemoryManager so we don't need a full .ccr/ setup
        from ccr.hooks import on_stop
        monkeypatch.setattr(
            "ccr.hooks.on_stop._read_stdin_payload",
            lambda: {"session_id": sid, "transcript_path": tp, "stop_hook_active": True},
        )

        # Stub out the heavy MemoryManager parts
        class _FakeState:
            def is_meaningful(self): return False
            def to_commit_fields(self): return {}

        _ccr_root_capture = ccr_root

        class _FakeMem:
            def __init__(self):
                self.ccr_root = _ccr_root_capture
            def log_ota(self, **kwargs): pass

        class _FakeMemoryManager:
            def __new__(cls, *args, **kwargs):
                return _FakeMem()

        monkeypatch.setattr("ccr.hooks.on_stop.MemoryManager", _FakeMemoryManager, raising=False)
        # Patch the imports inside main() using sys.modules
        import ccr.hooks.state_accumulator as sa_mod
        monkeypatch.setattr(sa_mod, "clear_state", lambda root: None)
        monkeypatch.setattr(sa_mod, "load_state", lambda root: _FakeState())

        # Monkeypatch sys.path so imports work from tmp project root
        monkeypatch.setenv("CCR_PROJECT_ROOT", str(tmp_path))

        on_stop.main()

        # Verify turns were inserted
        store2 = SessionStore(db_path)
        turns = store2.get_session_turns(sid)
        store2.close()

        assert len(turns) == 2
        assert all(t["source"] == "transcript" for t in turns)
        assert turns[0]["user_message"] == "Question one"
        assert turns[1]["user_message"] == "Question two"

    def test_on_stop_no_duplicate_if_already_logged(self, tmp_path, monkeypatch):
        """main() does not duplicate turns already in the DB."""
        ccr_root = self._make_ccr_dir(tmp_path)

        db_path = os.path.join(ccr_root, "sessions.db")
        store = SessionStore(db_path)
        sid = store.create_session(project=str(tmp_path))
        # Pre-log both turns
        store.log_turn(sid, "Question one", "Answer one", source="direct")
        store.log_turn(sid, "Question two", "Answer two", source="direct")
        store.close()

        id_file = os.path.join(ccr_root, ".current_session_id")
        with open(id_file, "w") as f:
            f.write(sid)

        tp = str(tmp_path / "transcript.jsonl")
        _write_transcript(tp, [
            {"user": "Question one", "assistant": "Answer one"},
            {"user": "Question two", "assistant": "Answer two"},
        ])

        from ccr.hooks import on_stop
        monkeypatch.setattr(
            "ccr.hooks.on_stop._read_stdin_payload",
            lambda: {"session_id": sid, "transcript_path": tp, "stop_hook_active": True},
        )

        class _FakeState:
            def is_meaningful(self): return False

        _ccr_root_capture2 = ccr_root

        class _FakeMem2:
            def __init__(self):
                self.ccr_root = _ccr_root_capture2
            def log_ota(self, **kwargs): pass

        class _FakeMemoryManager2:
            def __new__(cls, *args, **kwargs):
                return _FakeMem2()

        monkeypatch.setattr("ccr.hooks.on_stop.MemoryManager", _FakeMemoryManager2, raising=False)
        import ccr.hooks.state_accumulator as sa_mod
        monkeypatch.setattr(sa_mod, "clear_state", lambda root: None)
        monkeypatch.setattr(sa_mod, "load_state", lambda root: _FakeState())
        monkeypatch.setenv("CCR_PROJECT_ROOT", str(tmp_path))

        on_stop.main()

        store2 = SessionStore(db_path)
        turns = store2.get_session_turns(sid)
        store2.close()

        # Still exactly 2 turns — no duplicates
        assert len(turns) == 2
        # Original sources preserved
        assert all(t["source"] == "direct" for t in turns)

    def test_reconcile_tail_turns(self, tmp_path, monkeypatch):
        """Reconciler inserts tail turns when Claude logged first N but missed the rest."""
        ccr_root = self._make_ccr_dir(tmp_path)
        db_path = os.path.join(ccr_root, "sessions.db")
        store = SessionStore(db_path)
        sid = store.create_session(project=str(tmp_path))
        # Claude logged only turn 1; transcript has 3 turns
        store.log_turn(sid, "Question one", "Answer one", source="direct")
        store.close()

        tp = str(tmp_path / "transcript.jsonl")
        _write_transcript(tp, [
            {"user": "Question one", "assistant": "Answer one"},
            {"user": "Question two", "assistant": "Answer two"},
            {"user": "Question three", "assistant": "Answer three"},
        ])

        from ccr.hooks.on_stop import _reconcile_transcript
        store2 = SessionStore(db_path)
        inserted = _reconcile_transcript(store2, sid, tp)
        turns = store2.get_session_turns(sid, limit=10)
        store2.close()

        # Turns 2 and 3 inserted from transcript
        assert inserted == 2
        assert len(turns) == 3
        transcript_turns = [t for t in turns if t["source"] == "transcript"]
        assert len(transcript_turns) == 2
        assert transcript_turns[0]["user_message"] == "Question two"
        assert transcript_turns[1]["user_message"] == "Question three"
