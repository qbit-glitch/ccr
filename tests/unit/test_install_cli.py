"""Tests for the `ccr install` command (student-friendly zero-config setup)."""

from __future__ import annotations

import json
import os
import sys

import pytest
from click.testing import CliRunner

from ccr.cli import cli


@pytest.fixture()
def project(tmp_path):
    """Provide a clean temp directory as the project root."""
    return tmp_path


class TestInstallWritesMcpJson:
    """ccr install should write .mcp.json with ccr MCP server entry."""

    def test_install_writes_mcp_json(self, project):
        runner = CliRunner()
        result = runner.invoke(cli, ["install", str(project)])
        assert result.exit_code == 0, result.output

        mcp_path = project / ".mcp.json"
        assert mcp_path.exists(), ".mcp.json not created"

        config = json.loads(mcp_path.read_text())
        assert "mcpServers" in config
        assert "ccr" in config["mcpServers"]
        ccr_entry = config["mcpServers"]["ccr"]
        assert ccr_entry["command"] == sys.executable
        assert "-m" in ccr_entry["args"]
        assert "ccr.mcp_server" in ccr_entry["args"]

    def test_install_merges_existing_mcp_json(self, project):
        """Existing servers in .mcp.json are preserved."""
        mcp_path = project / ".mcp.json"
        mcp_path.write_text(json.dumps({
            "mcpServers": {
                "other-tool": {"command": "/usr/bin/other", "args": []}
            }
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["install", str(project)])
        assert result.exit_code == 0, result.output

        config = json.loads(mcp_path.read_text())
        assert "other-tool" in config["mcpServers"], "Existing server removed"
        assert "ccr" in config["mcpServers"], "CCR server not added"


class TestInstallWritesAllFourHooks:
    """ccr install should write all 4 hooks to settings.local.json."""

    def test_all_four_hooks_present(self, project):
        runner = CliRunner()
        result = runner.invoke(cli, ["install", str(project)])
        assert result.exit_code == 0, result.output

        settings_path = project / ".claude" / "settings.local.json"
        assert settings_path.exists(), "settings.local.json not created"

        settings = json.loads(settings_path.read_text())
        hooks = settings.get("hooks", {})

        assert "UserPromptSubmit" in hooks, "UserPromptSubmit hook missing"
        assert "PostToolUse" in hooks, "PostToolUse hook missing — auto-commit chain broken"
        assert "Stop" in hooks, "Stop hook missing"
        assert "PreCompact" in hooks, "PreCompact hook missing"

    def test_hook_commands_reference_correct_scripts(self, project):
        runner = CliRunner()
        runner.invoke(cli, ["install", str(project)])

        settings = json.loads(
            (project / ".claude" / "settings.local.json").read_text()
        )
        hooks = settings["hooks"]

        assert "on_session_start.py" in hooks["UserPromptSubmit"][0]["command"]
        assert "on_tool_use.py" in hooks["PostToolUse"][0]["command"]
        assert "on_stop.py" in hooks["Stop"][0]["command"]
        assert "on_compact.py" in hooks["PreCompact"][0]["command"]


class TestInstallIdempotent:
    """Running ccr install twice should not duplicate hooks."""

    def test_install_idempotent(self, project):
        runner = CliRunner()
        runner.invoke(cli, ["install", str(project)])
        runner.invoke(cli, ["install", str(project)])

        settings = json.loads(
            (project / ".claude" / "settings.local.json").read_text()
        )
        hooks = settings.get("hooks", {})

        # Each hook key should have exactly one entry (dict.update, not append)
        for hook_name in ("UserPromptSubmit", "PostToolUse", "Stop", "PreCompact"):
            assert len(hooks[hook_name]) == 1, (
                f"{hook_name} has {len(hooks[hook_name])} entries after 2 installs"
            )


class TestInstallSuccessMessage:
    """ccr install should print a clear success message with Quick test prompt."""

    def test_success_message_includes_all_hooks(self, project):
        runner = CliRunner()
        result = runner.invoke(cli, ["install", str(project)])
        assert "PostToolUse" in result.output
        assert "PreCompact" in result.output
        assert "UserPromptSubmit" in result.output
        assert "Stop" in result.output

    def test_success_message_includes_quick_test(self, project):
        runner = CliRunner()
        result = runner.invoke(cli, ["install", str(project)])
        assert "Quick test" in result.output or "gcc_status" in result.output
