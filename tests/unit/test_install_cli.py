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

    def test_hook_commands_include_ccr_project_root(self, project):
        """Hook commands must embed CCR_PROJECT_ROOT so they work from any CWD."""
        runner = CliRunner()
        runner.invoke(cli, ["install", str(project)])

        settings = json.loads(
            (project / ".claude" / "settings.local.json").read_text()
        )
        hooks = settings["hooks"]
        abs_project = str(project.resolve())

        for event in ("UserPromptSubmit", "PostToolUse", "Stop", "PreCompact"):
            cmd = hooks[event][0]["command"]
            # Accept both quoted (paths with spaces) and unquoted (simple paths) forms:
            #   CCR_PROJECT_ROOT=/simple/path ...
            #   CCR_PROJECT_ROOT='/path/with spaces' ...
            assert (
                f"CCR_PROJECT_ROOT={abs_project}" in cmd
                or f"CCR_PROJECT_ROOT='{abs_project}'" in cmd
            ), (
                f"{event} hook missing CCR_PROJECT_ROOT prefix: {cmd!r}"
            )


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


class TestInstallUpgradeReplacesOldHooks:
    """Re-running ccr install after an old-format install replaces old hooks (no duplicates)."""

    def test_old_format_hook_replaced_not_duplicated(self, project):
        """Old-format hooks (no CCR_PROJECT_ROOT prefix) are replaced, not duplicated."""
        import sys as _sys
        from ccr.cli import cli as _cli

        # Simulate an old-format install by writing hooks without the env var prefix
        settings_dir = project / ".claude"
        settings_dir.mkdir(parents=True)
        settings_path = settings_dir / "settings.local.json"

        # Construct an old-style hook command (no CCR_PROJECT_ROOT)
        import os as _os
        import ccr
        ccr_pkg = _os.path.dirname(_os.path.abspath(ccr.__file__))
        old_cmd = f"{_sys.executable} {_os.path.join(ccr_pkg, 'hooks', 'on_session_start.py')}"
        settings_path.write_text(json.dumps({
            "hooks": {"UserPromptSubmit": [{"type": "command", "command": old_cmd}]}
        }))

        runner = CliRunner()
        result = runner.invoke(_cli, ["install", str(project)])
        assert result.exit_code == 0, result.output

        data = json.loads(settings_path.read_text())
        cmds = [c["command"] for c in data["hooks"]["UserPromptSubmit"]]
        # Must have exactly one CCR hook, and it must have the env var prefix
        ccr_cmds = [c for c in cmds if "on_session_start.py" in c]
        assert len(ccr_cmds) == 1, f"Expected 1 CCR hook, got {len(ccr_cmds)}: {cmds}"
        assert "CCR_PROJECT_ROOT=" in ccr_cmds[0], (
            f"Upgraded hook missing CCR_PROJECT_ROOT: {ccr_cmds[0]!r}"
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


class TestInstallPreservesNonCcrHooks:
    """ccr install must not clobber pre-existing hooks from other tools."""

    def test_pre_existing_hook_survives_install(self, project):
        """A hook from another tool on UserPromptSubmit must survive ccr install."""
        settings_dir = project / ".claude"
        settings_dir.mkdir(parents=True)
        settings_path = settings_dir / "settings.local.json"
        settings_path.write_text(json.dumps({
            "hooks": {
                "UserPromptSubmit": [{"type": "command", "command": "other-tool.py"}]
            }
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["install", str(project)])
        assert result.exit_code == 0, result.output

        data = json.loads(settings_path.read_text())
        cmds = [c["command"] for c in data["hooks"]["UserPromptSubmit"]]
        assert "other-tool.py" in cmds, "Pre-existing hook was clobbered by ccr install"
        assert any("on_session_start.py" in c for c in cmds), "CCR hook not added"

    def test_pre_existing_hook_no_duplicate_on_reinstall(self, project):
        """Re-running ccr install must not add the pre-existing hook twice."""
        runner = CliRunner()
        runner.invoke(cli, ["install", str(project)])
        runner.invoke(cli, ["install", str(project)])

        settings = json.loads(
            (project / ".claude" / "settings.local.json").read_text()
        )
        hooks = settings.get("hooks", {})
        for hook_name in ("UserPromptSubmit", "PostToolUse", "Stop", "PreCompact"):
            assert len(hooks[hook_name]) == 1, (
                f"{hook_name} has {len(hooks[hook_name])} entries after 2 installs"
            )


class TestDoctorHookPathCheck:
    """ccr doctor should detect stale hook paths and report them as [WARN]."""

    def test_valid_paths_emit_ok(self, project):
        """When hook paths exist on disk, doctor reports OK."""
        from ccr.cli_doctor import _run_doctor_checks

        runner = CliRunner()
        runner.invoke(cli, ["install", str(project)])

        ok_items, issues, notices = _run_doctor_checks(str(project))
        # The installed paths should all be valid (using current sys.executable + real script)
        ok_text = " ".join(ok_items)
        assert "Hook paths valid" in ok_text, (
            f"Expected 'Hook paths valid' in OK items, got: {ok_items}"
        )
        # No stale-path warning should be emitted
        warn_notices = [n for n in notices if "Hook path stale" in n]
        assert not warn_notices, f"Unexpected stale-path warning for valid paths: {warn_notices}"

    def test_missing_python_emits_warn(self, project):
        """When the python executable path in a hook is missing, doctor emits a WARN notice."""
        from ccr.cli_doctor import _run_doctor_checks

        # Manually write a hook command with a nonexistent python path
        settings_dir = project / ".claude"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_path = settings_dir / "settings.local.json"

        import ccr
        ccr_pkg = os.path.dirname(os.path.abspath(ccr.__file__))
        real_script = os.path.join(ccr_pkg, "hooks", "on_stop.py")
        fake_cmd = f"CCR_PROJECT_ROOT={project} /nonexistent/python {real_script}"
        settings_path.write_text(json.dumps({
            "hooks": {
                "Stop": [{"type": "command", "command": fake_cmd}]
            }
        }))

        _ok, _issues, notices = _run_doctor_checks(str(project))
        warn_notices = [n for n in notices if "Hook path stale" in n]
        assert warn_notices, "Expected a stale-path WARN notice for missing python executable"
        assert "python" in warn_notices[0], f"WARN should mention python path: {warn_notices[0]}"

    def test_missing_script_emits_warn(self, project):
        """When the hook script path in a hook is missing, doctor emits a WARN notice."""
        from ccr.cli_doctor import _run_doctor_checks

        settings_dir = project / ".claude"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_path = settings_dir / "settings.local.json"

        fake_cmd = f"CCR_PROJECT_ROOT={project} {sys.executable} /nonexistent/path/on_stop.py"
        settings_path.write_text(json.dumps({
            "hooks": {
                "Stop": [{"type": "command", "command": fake_cmd}]
            }
        }))

        _ok, _issues, notices = _run_doctor_checks(str(project))
        warn_notices = [n for n in notices if "Hook path stale" in n]
        assert warn_notices, "Expected a stale-path WARN notice for missing script"
        assert "script" in warn_notices[0], f"WARN should mention script path: {warn_notices[0]}"

    def test_non_ccr_hooks_not_flagged(self, project):
        """Hooks from other tools (not CCR scripts) are not flagged as stale."""
        from ccr.cli_doctor import _check_stale_hook_paths

        hooks = {
            "UserPromptSubmit": [
                {"type": "command", "command": "CCR_PROJECT_ROOT=/p /bad/python /bad/other-tool.py"}
            ]
        }
        result = _check_stale_hook_paths(hooks)
        assert result == [], f"Non-CCR hook incorrectly flagged: {result}"

    def test_stale_check_skipped_when_no_ccr_hooks(self, project):
        """_check_stale_hook_paths returns [] for empty hooks dict."""
        from ccr.cli_doctor import _check_stale_hook_paths

        assert _check_stale_hook_paths({}) == []

    def test_doctor_command_output_contains_warn_text(self, project):
        """ccr doctor CLI output contains [WARN] text when a hook path is missing."""
        runner = CliRunner()

        # Write hook with missing python path
        settings_dir = project / ".claude"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_path = settings_dir / "settings.local.json"

        import ccr
        ccr_pkg = os.path.dirname(os.path.abspath(ccr.__file__))
        real_script = os.path.join(ccr_pkg, "hooks", "on_session_start.py")
        fake_cmd = f"CCR_PROJECT_ROOT={project} /nonexistent/bin/python {real_script}"
        settings_path.write_text(json.dumps({
            "hooks": {
                "UserPromptSubmit": [{"type": "command", "command": fake_cmd}]
            }
        }))

        result = runner.invoke(cli, ["doctor", str(project)])
        assert "[WARN]" in result.output or "stale" in result.output.lower(), (
            f"Expected [WARN] in doctor output, got:\n{result.output}"
        )
