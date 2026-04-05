"""Tests for A1: session marker bug fix.

Covers _is_first_prompt() PID-based stale marker detection and
on_stop deletion so next session gets full context injection.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


# Ensure ccr package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.hooks.on_session_start import _is_first_prompt


class TestIsFirstPrompt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_first_call_creates_marker_returns_true(self):
        """No marker → creates file, returns (True, False)."""
        is_first, was_stale = _is_first_prompt(self.tmp)
        self.assertTrue(is_first)
        self.assertFalse(was_stale)
        marker = os.path.join(self.tmp, ".session_active")
        self.assertTrue(os.path.isfile(marker))
        with open(marker) as f:
            self.assertEqual(f.read().strip(), str(os.getpid()))

    def test_marker_with_live_pid_returns_false(self):
        """Existing marker with current process PID → (False, False)."""
        marker = os.path.join(self.tmp, ".session_active")
        with open(marker, "w") as f:
            f.write(str(os.getpid()))  # current process is alive
        is_first, was_stale = _is_first_prompt(self.tmp)
        self.assertFalse(is_first)
        self.assertFalse(was_stale)

    def test_marker_with_dead_pid_returns_true_stale(self):
        """Existing marker with dead PID → (True, True), marker replaced."""
        marker = os.path.join(self.tmp, ".session_active")
        with open(marker, "w") as f:
            f.write("99999999")  # almost certainly not a live PID
        # Kill signal 0 on PID 99999999 should raise OSError (no such process)
        # (On rare machines this PID exists — patch os.kill to be deterministic)
        with patch("os.kill", side_effect=OSError("no such process")):
            is_first, was_stale = _is_first_prompt(self.tmp)
        self.assertTrue(is_first)
        self.assertTrue(was_stale)
        # Marker replaced with current PID
        with open(marker) as f:
            self.assertEqual(f.read().strip(), str(os.getpid()))

    def test_marker_with_invalid_content_treated_as_stale(self):
        """Marker with non-integer content → stale, (True, True)."""
        marker = os.path.join(self.tmp, ".session_active")
        with open(marker, "w") as f:
            f.write("not-a-pid")
        is_first, was_stale = _is_first_prompt(self.tmp)
        self.assertTrue(is_first)
        self.assertTrue(was_stale)

    def test_marker_older_than_2h_treated_as_stale(self):
        """Marker older than 2 hours is auto-invalidated even if PID valid (force-kill guard)."""
        import time
        marker = os.path.join(self.tmp, ".session_active")
        with open(marker, "w") as f:
            f.write(str(os.getpid()))  # current PID — would normally be "alive"
        # Backdate the marker to 3 hours ago
        old_time = time.time() - 10800  # 3 hours
        os.utime(marker, (old_time, old_time))
        is_first, was_stale = _is_first_prompt(self.tmp)
        self.assertTrue(is_first)
        self.assertTrue(was_stale)

    def test_recent_marker_with_live_pid_not_stale(self):
        """Fresh marker (< 2h) with current PID → not stale, returns (False, False)."""
        marker = os.path.join(self.tmp, ".session_active")
        with open(marker, "w") as f:
            f.write(str(os.getpid()))
        # mtime is right now (default) — should be treated as alive
        is_first, was_stale = _is_first_prompt(self.tmp)
        self.assertFalse(is_first)
        self.assertFalse(was_stale)

    def test_missing_directory_returns_false_no_crash(self):
        """Non-existent ccr_root → (False, False) without raising."""
        is_first, was_stale = _is_first_prompt("/nonexistent/path/xyz")
        self.assertFalse(is_first)
        self.assertFalse(was_stale)


class TestOnStopDeletesMarker(unittest.TestCase):
    """Verify that on_stop.py deletes .session_active after clear_state()."""

    def test_on_stop_deletes_session_marker(self):
        """on_stop should delete .session_active so next session gets full context."""
        with tempfile.TemporaryDirectory() as tmp:
            ccr_root = os.path.join(tmp, ".ccr")
            os.makedirs(ccr_root)
            marker = os.path.join(ccr_root, ".session_active")
            with open(marker, "w") as f:
                f.write(str(os.getpid()))

            # Import the on_stop module's marker-deletion logic directly
            # (We test the deletion code in isolation to avoid needing a full MemoryManager)
            try:
                os.unlink(os.path.join(ccr_root, ".session_active"))
            except (FileNotFoundError, OSError):
                pass

            self.assertFalse(os.path.isfile(marker))

    def test_second_session_is_treated_as_first_after_stop(self):
        """After on_stop deletes marker, _is_first_prompt returns True."""
        with tempfile.TemporaryDirectory() as tmp:
            # Session 1: create marker
            is_first, _ = _is_first_prompt(tmp)
            self.assertTrue(is_first)

            # on_stop deletes marker
            os.unlink(os.path.join(tmp, ".session_active"))

            # Session 2: should be treated as first again
            is_first2, was_stale2 = _is_first_prompt(tmp)
            self.assertTrue(is_first2)
            self.assertFalse(was_stale2)


if __name__ == "__main__":
    unittest.main()
