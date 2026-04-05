"""Tests for C1: session token tracking infrastructure.

Covers SessionState new fields, initialize_state session_id, and
_write_session_metrics appending to sessions.jsonl.
"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.hooks.state_accumulator import (
    SessionState,
    initialize_state,
    load_state,
    save_state,
)
from ccr.hooks.on_stop import _write_session_metrics


class TestSessionStateNewFields(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_context_tokens_zero(self):
        state = SessionState()
        self.assertEqual(state.context_tokens, 0)

    def test_default_session_id_empty(self):
        state = SessionState()
        self.assertEqual(state.session_id, "")

    def test_fields_round_trip_through_json(self):
        """context_tokens and session_id survive save/load cycle."""
        state = SessionState(context_tokens=1234, session_id="abc12345")
        save_state(self.tmp, state)
        loaded = load_state(self.tmp)
        self.assertEqual(loaded.context_tokens, 1234)
        self.assertEqual(loaded.session_id, "abc12345")

    def test_initialize_state_sets_session_id(self):
        """initialize_state creates a non-empty 8-char session_id."""
        initialize_state(self.tmp)
        state = load_state(self.tmp)
        self.assertTrue(len(state.session_id) == 8)

    def test_two_initializations_have_different_session_ids(self):
        initialize_state(self.tmp)
        state1 = load_state(self.tmp)
        initialize_state(self.tmp)
        state2 = load_state(self.tmp)
        self.assertNotEqual(state1.session_id, state2.session_id)

    def test_load_state_missing_new_fields_defaults_gracefully(self):
        """Old state file without context_tokens/session_id loads without crash."""
        path = os.path.join(self.tmp, ".session_state.json")
        with open(path, "w") as f:
            json.dump({"files_touched": ["a.py"], "tool_calls": 3}, f)
        state = load_state(self.tmp)
        self.assertEqual(state.context_tokens, 0)
        self.assertEqual(state.session_id, "")
        self.assertEqual(state.tool_calls, 3)


class TestWriteSessionMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _jsonl_path(self):
        return os.path.join(self.tmp, "metrics", "sessions.jsonl")

    def test_writes_record_when_context_tokens_nonzero(self):
        state = SessionState(
            context_tokens=2500,
            session_id="test1234",
            start_time=time.time() - 600,  # 10 minutes ago
        )
        _write_session_metrics(self.tmp, state)
        self.assertTrue(os.path.isfile(self._jsonl_path()))
        with open(self._jsonl_path()) as f:
            record = json.loads(f.readline())
        self.assertEqual(record["context_tokens"], 2500)
        self.assertEqual(record["session_id"], "test1234")
        self.assertGreater(record["duration_min"], 0)

    def test_skips_when_context_tokens_zero(self):
        """Sessions with no injected context are excluded from stats."""
        state = SessionState(context_tokens=0, session_id="empty123")
        _write_session_metrics(self.tmp, state)
        self.assertFalse(os.path.isfile(self._jsonl_path()))

    def test_two_sessions_append_two_lines(self):
        for i in range(2):
            state = SessionState(
                context_tokens=1000 + i * 500,
                session_id=f"sess{i:04d}",
                start_time=time.time() - 300,
            )
            _write_session_metrics(self.tmp, state)
        with open(self._jsonl_path()) as f:
            lines = [l for l in f.readlines() if l.strip()]
        self.assertEqual(len(lines), 2)

    def test_non_fatal_on_unwritable_path(self):
        """Metrics write failure must not raise — hooks must never crash."""
        state = SessionState(context_tokens=999, session_id="x")
        # Pass a file path where a directory is expected (causes os.makedirs to fail)
        bad_root = "/dev/null/fake_ccr"
        try:
            _write_session_metrics(bad_root, state)
        except Exception as e:
            self.fail(f"_write_session_metrics raised unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
