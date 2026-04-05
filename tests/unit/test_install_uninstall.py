"""Tests for A4: ccr uninstall command.

Verifies that uninstall removes CCR hooks + MCP entry while preserving .ccr/ memory.
"""
import json
import os
import sys
import tempfile
import unittest
from click.testing import CliRunner
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.cli import install, uninstall


def _write_settings(claude_dir: str, hooks: dict) -> str:
    """Write a settings.local.json with given hooks dict."""
    os.makedirs(claude_dir, exist_ok=True)
    path = os.path.join(claude_dir, "settings.local.json")
    with open(path, "w") as f:
        json.dump({"hooks": hooks}, f, indent=2)
    return path


def _write_mcp(project: str, servers: dict) -> str:
    """Write a .mcp.json with given mcpServers dict."""
    path = os.path.join(project, ".mcp.json")
    with open(path, "w") as f:
        json.dump({"mcpServers": servers}, f, indent=2)
    return path


class TestCcrUninstall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.runner = CliRunner()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_claude_dir(self):
        return os.path.join(self.tmp, ".claude")

    def test_uninstall_removes_ccr_hooks(self):
        """ccr uninstall removes all CCR hook entries from settings.local.json."""
        claude_dir = self._fake_claude_dir()
        _write_settings(claude_dir, {
            "UserPromptSubmit": [{"type": "command", "command": "/fake/.venv/bin/python ccr/hooks/on_session_start.py"}],
            "Stop":             [{"type": "command", "command": "/fake/.venv/bin/python ccr/hooks/on_stop.py"}],
        })
        _write_mcp(self.tmp, {"ccr": {"command": "/fake/python", "args": ["-m", "ccr.mcp_server"]}})

        with patch("os.path.expanduser", return_value=claude_dir):
            result = self.runner.invoke(uninstall, [self.tmp, "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        settings = json.loads(open(os.path.join(claude_dir, "settings.local.json")).read())
        self.assertEqual(settings.get("hooks", {}), {})

    def test_uninstall_removes_mcp_entry(self):
        """ccr uninstall removes ccr from .mcp.json."""
        claude_dir = self._fake_claude_dir()
        _write_settings(claude_dir, {
            "Stop": [{"type": "command", "command": "/fake/ccr/hooks/on_stop.py"}],
        })
        _write_mcp(self.tmp, {
            "ccr": {"command": "/fake/python"},
            "other-server": {"command": "/other/cmd"},
        })

        with patch("os.path.expanduser", return_value=claude_dir):
            result = self.runner.invoke(uninstall, [self.tmp, "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        mcp = json.loads(open(os.path.join(self.tmp, ".mcp.json")).read())
        self.assertNotIn("ccr", mcp.get("mcpServers", {}))
        self.assertIn("other-server", mcp.get("mcpServers", {}))

    def test_uninstall_preserves_non_ccr_hooks(self):
        """ccr uninstall must not touch hooks that don't reference ccr."""
        claude_dir = self._fake_claude_dir()
        _write_settings(claude_dir, {
            "UserPromptSubmit": [
                {"type": "command", "command": "/fake/ccr/hooks/on_session_start.py"},
                {"type": "command", "command": "/other/tool/hook.py"},
            ],
        })
        _write_mcp(self.tmp, {"ccr": {"command": "/fake/python"}})

        with patch("os.path.expanduser", return_value=claude_dir):
            result = self.runner.invoke(uninstall, [self.tmp, "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        settings = json.loads(open(os.path.join(claude_dir, "settings.local.json")).read())
        # Non-CCR hook preserved
        hooks = settings.get("hooks", {}).get("UserPromptSubmit", [])
        commands = [h["command"] for h in hooks]
        self.assertIn("/other/tool/hook.py", commands)
        self.assertNotIn("/fake/ccr/hooks/on_session_start.py", commands)

    def test_uninstall_preserves_ccr_memory_directory(self):
        """ccr uninstall must not delete .ccr/ directory."""
        claude_dir = self._fake_claude_dir()
        _write_settings(claude_dir, {
            "Stop": [{"type": "command", "command": "/fake/ccr/hooks/on_stop.py"}],
        })
        _write_mcp(self.tmp, {"ccr": {"command": "/fake/python"}})
        # Create a .ccr/ directory with fake commit
        ccr_dir = os.path.join(self.tmp, ".ccr")
        os.makedirs(ccr_dir, exist_ok=True)
        with open(os.path.join(ccr_dir, "test.txt"), "w") as f:
            f.write("important memory")

        with patch("os.path.expanduser", return_value=claude_dir):
            result = self.runner.invoke(uninstall, [self.tmp, "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(os.path.isdir(ccr_dir))
        self.assertTrue(os.path.isfile(os.path.join(ccr_dir, "test.txt")))

    def test_uninstall_nothing_installed_graceful(self):
        """Uninstalling when CCR is not installed should not crash."""
        claude_dir = self._fake_claude_dir()
        os.makedirs(claude_dir, exist_ok=True)

        with patch("os.path.expanduser", return_value=claude_dir):
            result = self.runner.invoke(uninstall, [self.tmp, "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("not installed", result.output)


if __name__ == "__main__":
    unittest.main()
