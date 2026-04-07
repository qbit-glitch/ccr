"""Tests for C2: ccr stats ROI dashboard command."""
import json
import os
import sys
import tempfile
import unittest
from click.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.cli import stats


def _make_project(tmp_dir: str, sessions: list[dict] | None = None) -> str:
    """Create a minimal .ccr/ project directory with optional sessions.jsonl."""
    ccr_dir = os.path.join(tmp_dir, ".ccr")
    os.makedirs(os.path.join(ccr_dir, "metrics"), exist_ok=True)

    if sessions:
        jsonl_path = os.path.join(ccr_dir, "metrics", "sessions.jsonl")
        with open(jsonl_path, "w") as f:
            for s in sessions:
                f.write(json.dumps(s) + "\n")

    return tmp_dir


class TestCcrStats(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.runner = CliRunner()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_history_shows_message(self):
        """No sessions.jsonl → user-friendly message, no crash. Exact message depends
        on whether hooks are installed (global or local), so check for the section header."""
        _make_project(self.tmp)
        result = self.runner.invoke(stats, [self.tmp])
        self.assertEqual(result.exit_code, 0)
        # Either "Hooks are active" (hooks installed) or "ccr install" (not installed)
        has_hooks_msg = "Hooks are active" in result.output
        has_install_msg = "ccr install" in result.output or "No session history" in result.output
        self.assertTrue(has_hooks_msg or has_install_msg, result.output)

    def test_known_data_correct_totals(self):
        """Given 2 sessions with known tokens, totals should match."""
        sessions = [
            {"context_tokens": 2000, "start": "2026-04-01T10:00:00+00:00",
             "end": "2026-04-01T10:40:00+00:00", "duration_min": 40.0},
            {"context_tokens": 3000, "start": "2026-04-02T10:00:00+00:00",
             "end": "2026-04-02T10:50:00+00:00", "duration_min": 50.0},
        ]
        _make_project(self.tmp, sessions=sessions)
        result = self.runner.invoke(stats, [self.tmp])
        self.assertEqual(result.exit_code, 0)
        # Total injected = 5000
        self.assertIn("5,000", result.output)
        # Default multiplier ×4 → 20,000 avoided
        self.assertIn("20,000", result.output)

    def test_custom_multiplier(self):
        """--multiplier 2 should give half the default avoided tokens."""
        sessions = [{"context_tokens": 1000, "start": "2026-04-01T10:00:00+00:00",
                     "end": "2026-04-01T10:30:00+00:00", "duration_min": 30.0}]
        _make_project(self.tmp, sessions=sessions)
        result = self.runner.invoke(stats, [self.tmp, "--multiplier", "2"])
        self.assertEqual(result.exit_code, 0)
        # 1000 tokens × 2 = 2,000 avoided
        self.assertIn("2,000", result.output)
        self.assertIn("×2", result.output)

    def test_heuristic_label_present(self):
        """Output must include heuristic disclaimer and --multiplier hint."""
        sessions = [{"context_tokens": 1000, "start": "2026-04-01T10:00:00+00:00",
                     "end": "2026-04-01T10:30:00+00:00", "duration_min": 30.0}]
        _make_project(self.tmp, sessions=sessions)
        result = self.runner.invoke(stats, [self.tmp])
        self.assertEqual(result.exit_code, 0)
        # Disclaimer now references "Gross savings" and "Net savings" instead of "Rough estimate"
        self.assertIn("Gross savings", result.output)
        self.assertIn("Net savings", result.output)
        self.assertIn("--multiplier", result.output)

    def test_broken_jsonl_line_skipped_gracefully(self):
        """Malformed JSON lines in sessions.jsonl must not crash ccr stats."""
        jsonl_path = os.path.join(self.tmp, ".ccr", "metrics", "sessions.jsonl")
        _make_project(self.tmp)
        with open(jsonl_path, "w") as f:
            f.write("INVALID JSON LINE\n")
            f.write(json.dumps({"context_tokens": 999, "start": "2026-04-01T10:00:00+00:00",
                                "end": "2026-04-01T10:30:00+00:00", "duration_min": 30.0}) + "\n")
        result = self.runner.invoke(stats, [self.tmp])
        self.assertEqual(result.exit_code, 0)
        # Valid session still shows
        self.assertIn("999", result.output)

    def test_sessions_with_zero_context_excluded(self):
        """Sessions where context_tokens==0 (pre-install sessions) are excluded from averages."""
        sessions = [
            {"context_tokens": 0, "start": "2026-04-01T10:00:00+00:00",
             "end": "2026-04-01T10:30:00+00:00"},
            {"context_tokens": 2000, "start": "2026-04-02T10:00:00+00:00",
             "end": "2026-04-02T10:30:00+00:00", "duration_min": 30.0},
        ]
        _make_project(self.tmp, sessions=sessions)
        result = self.runner.invoke(stats, [self.tmp])
        self.assertEqual(result.exit_code, 0)
        # Only 1 valid session, total injected should be 2,000 not 2,000+0=2,000
        self.assertIn("2,000", result.output)
        # Should show 1 session, not 2
        self.assertIn("1 session", result.output)


if __name__ == "__main__":
    unittest.main()
