"""Tests for hook error logging — _log_hook_error writes to .ccr/.hook_errors.log."""
import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestOnToolUseErrorLog(unittest.TestCase):
    def test_log_hook_error_creates_file(self):
        """_log_hook_error writes to .ccr/.hook_errors.log."""
        with tempfile.TemporaryDirectory() as tmp:
            ccr_root = os.path.join(tmp, ".ccr")
            os.makedirs(ccr_root)
            with patch.dict(os.environ, {"CCR_PROJECT_ROOT": tmp}):
                import ccr.hooks.on_tool_use as m
                importlib.reload(m)
                m._log_hook_error("test error traceback")
            log = os.path.join(ccr_root, ".hook_errors.log")
            self.assertTrue(os.path.isfile(log))
            content = open(log).read()
            self.assertIn("[on_tool_use]", content)
            self.assertIn("test error traceback", content)

    def test_log_hook_error_no_ccr_dir_no_crash(self):
        """_log_hook_error is silent when .ccr/ doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            # No .ccr/ created
            with patch.dict(os.environ, {"CCR_PROJECT_ROOT": tmp}):
                import ccr.hooks.on_tool_use as m
                importlib.reload(m)
                m._log_hook_error("should not crash")  # must not raise

    def test_log_hook_error_appends(self):
        """Multiple calls append to the same log file."""
        with tempfile.TemporaryDirectory() as tmp:
            ccr_root = os.path.join(tmp, ".ccr")
            os.makedirs(ccr_root)
            with patch.dict(os.environ, {"CCR_PROJECT_ROOT": tmp}):
                import ccr.hooks.on_tool_use as m
                importlib.reload(m)
                m._log_hook_error("error one")
                m._log_hook_error("error two")
            log = os.path.join(ccr_root, ".hook_errors.log")
            content = open(log).read()
            self.assertIn("error one", content)
            self.assertIn("error two", content)


class TestOnCompactErrorLog(unittest.TestCase):
    def test_log_hook_error_compact(self):
        """on_compact._log_hook_error writes with [on_compact] tag."""
        with tempfile.TemporaryDirectory() as tmp:
            ccr_root = os.path.join(tmp, ".ccr")
            os.makedirs(ccr_root)
            with patch.dict(os.environ, {"CCR_PROJECT_ROOT": tmp}):
                import ccr.hooks.on_compact as m
                importlib.reload(m)
                m._log_hook_error("compact error")
            log = os.path.join(ccr_root, ".hook_errors.log")
            content = open(log).read()
            self.assertIn("[on_compact]", content)
            self.assertIn("compact error", content)


class TestDoctorHookErrors(unittest.TestCase):
    def test_doctor_reports_no_hook_errors_when_log_absent(self):
        """ccr doctor shows 'No hook errors' when .hook_errors.log absent."""
        from click.testing import CliRunner
        from ccr.cli import doctor
        with tempfile.TemporaryDirectory() as tmp:
            ccr_dir = os.path.join(tmp, ".ccr")
            os.makedirs(ccr_dir)
            runner = CliRunner()
            result = runner.invoke(doctor, [tmp])
            self.assertIn("No hook errors logged", result.output)

    def test_doctor_reports_recent_hook_errors(self):
        """ccr doctor flags recent errors in .hook_errors.log as [!!]."""
        from click.testing import CliRunner
        from ccr.cli import doctor
        with tempfile.TemporaryDirectory() as tmp:
            ccr_dir = os.path.join(tmp, ".ccr")
            os.makedirs(ccr_dir)
            # Write a recent error log
            log = os.path.join(ccr_dir, ".hook_errors.log")
            with open(log, "w") as f:
                f.write("\n--- 2026-04-05T00:00:00Z [on_tool_use] ---\nTraceback...\n")
            runner = CliRunner()
            result = runner.invoke(doctor, [tmp])
            self.assertIn("[!!]", result.output)
            self.assertIn("hook error", result.output.lower())

    def test_doctor_ok_for_old_hook_errors(self):
        """ccr doctor shows [OK] for errors older than 24h."""
        import time
        from click.testing import CliRunner
        from ccr.cli import doctor
        with tempfile.TemporaryDirectory() as tmp:
            ccr_dir = os.path.join(tmp, ".ccr")
            os.makedirs(ccr_dir)
            log = os.path.join(ccr_dir, ".hook_errors.log")
            with open(log, "w") as f:
                f.write("\n--- 2025-01-01T00:00:00Z [on_tool_use] ---\nOld error\n")
            # Backdate the file to 48h ago
            old = time.time() - 172800  # 48h
            os.utime(log, (old, old))
            runner = CliRunner()
            result = runner.invoke(doctor, [tmp])
            # Should be [OK], not [!!]
            lines = [l for l in result.output.splitlines() if "hook error" in l.lower()]
            self.assertTrue(any("[OK]" in l for l in lines), f"Expected [OK] line, got: {result.output}")


if __name__ == "__main__":
    unittest.main()
