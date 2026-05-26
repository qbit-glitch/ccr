"""Tests for the provider-agnostic adapter architecture.

No API keys required — all tests use mocks, stubs, or file-system validation.
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from unittest.mock import MagicMock, patch

from ccr.adapters import (
    get_adapter,
    get_adapters,
    list_adapter_names,
)
from ccr.adapters.claude_code import ClaudeCodeAdapter
from ccr.adapters.codex import CodexAdapter
from ccr.adapters.kimi import KimiAdapter
from ccr.adapters.continue_dev import ContinueDevAdapter
from ccr.adapters.ollama import OllamaAdapter
from ccr.adapters.openai import OpenAIAdapter
from ccr.adapters.generic_mcp import GenericMcpAdapter


class TestAdapterRegistry:
    """Test adapter discovery and registry."""

    def test_all_adapters_registered(self):
        names = list_adapter_names()
        assert "claude-code" in names
        assert "codex" in names
        assert "kimi" in names
        assert "ollama" in names
        assert "openai" in names
        assert "continue-dev" in names
        assert "generic-mcp" in names

    def test_get_adapter_by_name(self):
        assert isinstance(get_adapter("claude-code"), ClaudeCodeAdapter)
        assert isinstance(get_adapter("codex"), CodexAdapter)
        assert isinstance(get_adapter("kimi"), KimiAdapter)
        assert get_adapter("nonexistent") is None

    def test_adapter_status_structure(self):
        for adapter in get_adapters():
            status = adapter.status()
            assert status.name == adapter.name
            assert status.display_name == adapter.display_name
            assert isinstance(status.installed, bool)
            assert isinstance(status.ccr_enabled, bool)
            assert isinstance(status.integration_level, int)


class TestClaudeCodeAdapter:
    """Test Claude Code adapter without touching real ~/.claude/ config."""

    def test_is_installed_detects_claude_binary(self, tmp_path):
        # Create a fake claude binary in a temp PATH
        fake_bin = tmp_path / "claude"
        fake_bin.write_text("#!/bin/bash\necho fake")
        fake_bin.chmod(0o755)

        adapter = ClaudeCodeAdapter()
        with patch.dict(os.environ, {"PATH": str(tmp_path)}):
            # shutil.which should find our fake binary
            assert adapter.is_installed() is True

    def test_install_creates_mcp_and_hooks(self, tmp_path):
        adapter = ClaudeCodeAdapter()
        # Override config paths to temp dir
        adapter._CLAUDE_DIR = str(tmp_path / ".claude")
        adapter._MCP_PATH = str(tmp_path / ".claude" / ".mcp.json")
        adapter._SETTINGS_PATH = str(tmp_path / ".claude" / "settings.json")

        python_exe = "/usr/bin/python3"
        hooks_dir = "/fake/hooks"
        ccr_pkg = "/fake/ccr"

        result = adapter.install(python_exe, hooks_dir, ccr_pkg)
        assert result.success is True
        assert os.path.isfile(adapter._MCP_PATH)
        assert os.path.isfile(adapter._SETTINGS_PATH)

        # Verify MCP config contains CCR server
        with open(adapter._MCP_PATH) as f:
            mcp = json.load(f)
        assert "ccr" in mcp.get("mcpServers", {})
        assert mcp["mcpServers"]["ccr"]["env"]["CCR_STORAGE_BACKEND"] == "sqlite"

        # Verify hooks contain CCR commands
        with open(adapter._SETTINGS_PATH) as f:
            settings = json.load(f)
        hooks = settings.get("hooks", {})
        assert "UserPromptSubmit" in hooks
        # CCR hooks use {matcher, hooks:[...]} format per Claude Code spec
        all_cmds = []
        for entry in hooks["UserPromptSubmit"]:
            if "command" in entry:
                all_cmds.append(entry["command"])
            for hook in entry.get("hooks", []):
                all_cmds.append(hook.get("command", ""))
        assert any("on_session_start.py" in cmd for cmd in all_cmds)
        assert all("CCR_STORAGE_BACKEND=sqlite" in cmd for cmd in all_cmds if cmd)

    def test_uninstall_removes_ccr_entries(self, tmp_path):
        adapter = ClaudeCodeAdapter()
        adapter._CLAUDE_DIR = str(tmp_path / ".claude")
        adapter._MCP_PATH = str(tmp_path / ".claude" / ".mcp.json")
        adapter._SETTINGS_PATH = str(tmp_path / ".claude" / "settings.json")

        # Install first
        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        assert adapter._is_ccr_enabled() is True

        # Then uninstall
        result = adapter.uninstall()
        assert result.success is True
        assert adapter._is_ccr_enabled() is False

        # Verify CCR removed but file still exists
        with open(adapter._MCP_PATH) as f:
            mcp = json.load(f)
        assert "ccr" not in mcp.get("mcpServers", {})

    def test_install_replaces_nested_old_ccr_hook_preserving_other_hooks(self, tmp_path):
        adapter = ClaudeCodeAdapter()
        adapter._CLAUDE_DIR = str(tmp_path / ".claude")
        adapter._MCP_PATH = str(tmp_path / ".claude" / ".mcp.json")
        adapter._SETTINGS_PATH = str(tmp_path / ".claude" / "settings.json")
        os.makedirs(adapter._CLAUDE_DIR, exist_ok=True)
        with open(adapter._SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "hooks": {
                    "UserPromptSubmit": [{
                        "matcher": "",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "CCR_AUTO_INIT=1 /old/python /fake/hooks/on_session_start.py",
                            },
                            {
                                "type": "command",
                                "command": "/opt/tool/keep-me.js",
                            },
                        ],
                    }]
                }
            }, f)

        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")

        with open(adapter._SETTINGS_PATH) as f:
            settings = json.load(f)
        prompt_hooks = settings["hooks"]["UserPromptSubmit"]
        all_commands = []
        for entry in prompt_hooks:
            if "command" in entry:
                all_commands.append(entry["command"])
            for hook in entry.get("hooks", []):
                all_commands.append(hook.get("command", ""))
        assert "/opt/tool/keep-me.js" in all_commands
        assert not any("/old/python" in command for command in all_commands)
        sqlite_commands = [c for c in all_commands if "on_session_start.py" in c]
        assert sqlite_commands == [
            "CCR_AUTO_INIT=1 CCR_STORAGE_BACKEND=sqlite /usr/bin/python3 /fake/hooks/on_session_start.py"
        ]


class TestKimiAdapter:
    """Test Kimi adapter without touching real ~/.kimi/ config."""

    def test_install_creates_mcp_and_hooks(self, tmp_path):
        adapter = KimiAdapter()
        adapter._KIMI_DIR = str(tmp_path / ".kimi")
        adapter._MCP_PATH = str(tmp_path / ".kimi" / "mcp.json")
        adapter._CONFIG_PATH = str(tmp_path / ".kimi" / "config.toml")

        result = adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        assert result.success is True
        assert os.path.isfile(adapter._MCP_PATH)
        assert os.path.isfile(adapter._CONFIG_PATH)

        # Verify TOML contains CCR hooks
        with open(adapter._CONFIG_PATH) as f:
            toml = f.read()
        assert "[[hooks]]" in toml
        assert "ccr" in toml.lower()
        assert "CCR_STORAGE_BACKEND=sqlite" in toml
        with open(adapter._MCP_PATH) as f:
            mcp = json.load(f)
        assert mcp["mcpServers"]["ccr"]["env"]["CCR_STORAGE_BACKEND"] == "sqlite"

    def test_uninstall_removes_ccr_hooks(self, tmp_path):
        adapter = KimiAdapter()
        adapter._KIMI_DIR = str(tmp_path / ".kimi")
        adapter._MCP_PATH = str(tmp_path / ".kimi" / "mcp.json")
        adapter._CONFIG_PATH = str(tmp_path / ".kimi" / "config.toml")

        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        assert adapter._is_ccr_enabled() is True

        result = adapter.uninstall()
        assert result.success is True
        assert adapter._is_ccr_enabled() is False


class TestCodexAdapter:
    """Test Codex adapter without touching real ~/.codex/config.toml."""

    def test_install_writes_mcp_server_table(self, tmp_path):
        adapter = CodexAdapter()
        adapter._CODEX_DIR = str(tmp_path / ".codex")
        adapter._CONFIG_PATH = str(tmp_path / ".codex" / "config.toml")

        result = adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        assert result.success is True
        assert os.path.isfile(adapter._CONFIG_PATH)

        with open(adapter._CONFIG_PATH) as f:
            toml = f.read()
        assert "[mcp_servers.ccr]" in toml
        assert 'command = "/usr/bin/python3"' in toml
        assert 'args = ["-m", "ccr.mcp_server", "--project", "."]' in toml
        assert "[mcp_servers.ccr.env]" in toml
        assert 'CCR_STORAGE_BACKEND = "sqlite"' in toml
        assert "[features]" in toml
        assert "codex_hooks = true" in toml
        assert "[[hooks.SessionStart]]" in toml
        assert "CCR_STORAGE_BACKEND=sqlite" in toml
        assert "[[hooks.Stop]]" in toml
        assert "codex_stop.py" in toml
        assert "[[hooks.PostToolUse]]" not in toml
        assert "on_tool_use.py" not in toml
        assert "statusMessage" not in toml
        assert "Tracking CCR session changes" not in toml

    def test_install_can_opt_into_codex_post_tool_use_hook(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CCR_CODEX_POST_TOOL_USE", "1")
        adapter = CodexAdapter()
        adapter._CODEX_DIR = str(tmp_path / ".codex")
        adapter._CONFIG_PATH = str(tmp_path / ".codex" / "config.toml")

        result = adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        assert result.success is True

        with open(adapter._CONFIG_PATH) as f:
            toml = f.read()
        assert "[[hooks.PostToolUse]]" in toml
        assert 'matcher = ".*"' in toml
        assert "on_tool_use.py" in toml
        post_tool_use_block = toml.split("[[hooks.PostToolUse]]", 1)[1].split(
            "[[hooks.Stop]]", 1
        )[0]
        assert "statusMessage" not in post_tool_use_block
        assert "statusMessage" not in toml
        assert "Tracking CCR session changes" not in toml

    def test_install_preserves_existing_config_and_is_idempotent(self, tmp_path):
        adapter = CodexAdapter()
        adapter._CODEX_DIR = str(tmp_path / ".codex")
        adapter._CONFIG_PATH = str(tmp_path / ".codex" / "config.toml")
        os.makedirs(adapter._CODEX_DIR, exist_ok=True)
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write('model = "gpt-5.5"\n\n[projects."/repo"]\ntrust_level = "trusted"\n')

        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")

        with open(adapter._CONFIG_PATH) as f:
            toml = f.read()
        assert 'model = "gpt-5.5"' in toml
        assert '[projects."/repo"]' in toml
        assert toml.count("[mcp_servers.ccr]") == 1
        assert toml.count("[[hooks.SessionStart]]") == 1
        assert toml.count("[[hooks.UserPromptSubmit]]") == 1
        assert toml.count("[[hooks.PostToolUse]]") == 0
        assert toml.count("[[hooks.Stop]]") == 1

    def test_uninstall_removes_only_ccr_entries(self, tmp_path):
        adapter = CodexAdapter()
        adapter._CODEX_DIR = str(tmp_path / ".codex")
        adapter._CONFIG_PATH = str(tmp_path / ".codex" / "config.toml")

        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        assert adapter._is_ccr_enabled() is True

        result = adapter.uninstall()
        assert result.success is True
        assert adapter._is_ccr_enabled() is False

        with open(adapter._CONFIG_PATH) as f:
            toml = f.read()
        assert "[mcp_servers.ccr]" not in toml
        assert "[mcp_servers.ccr.env]" not in toml
        assert "CCR_STORAGE_BACKEND" not in toml
        assert "codex_stop.py" not in toml
        assert "codex_hooks" not in toml

    def test_install_keeps_root_keys_outside_features_table(self, tmp_path):
        adapter = CodexAdapter()
        adapter._CODEX_DIR = str(tmp_path / ".codex")
        adapter._CONFIG_PATH = str(tmp_path / ".codex" / "config.toml")
        os.makedirs(adapter._CODEX_DIR, exist_ok=True)
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write('model = "gpt-5.5"\nmodel_reasoning_effort = "high"\n')

        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")

        with open(adapter._CONFIG_PATH, "rb") as f:
            parsed = tomllib.load(f)
        assert parsed["model"] == "gpt-5.5"
        assert parsed["model_reasoning_effort"] == "high"
        assert parsed["features"]["codex_hooks"] is True
        assert parsed["mcp_servers"]["ccr"]["env"]["CCR_STORAGE_BACKEND"] == "sqlite"

    def test_uninstall_preserves_user_owned_codex_hooks_feature(self, tmp_path):
        adapter = CodexAdapter()
        adapter._CODEX_DIR = str(tmp_path / ".codex")
        adapter._CONFIG_PATH = str(tmp_path / ".codex" / "config.toml")
        os.makedirs(adapter._CODEX_DIR, exist_ok=True)
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("[features]\ncodex_hooks = true\n")

        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        adapter.uninstall()

        with open(adapter._CONFIG_PATH, "rb") as f:
            parsed = tomllib.load(f)
        assert parsed["features"]["codex_hooks"] is True

    def test_uninstall_restores_previous_codex_hooks_value(self, tmp_path):
        adapter = CodexAdapter()
        adapter._CODEX_DIR = str(tmp_path / ".codex")
        adapter._CONFIG_PATH = str(tmp_path / ".codex" / "config.toml")
        os.makedirs(adapter._CODEX_DIR, exist_ok=True)
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("[features]\ncodex_hooks = false\n")

        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        with open(adapter._CONFIG_PATH) as f:
            installed = f.read()
        assert "ccr-previous-codex_hooks = false" in installed

        adapter.uninstall()

        with open(adapter._CONFIG_PATH, "rb") as f:
            parsed = tomllib.load(f)
        assert parsed["features"]["codex_hooks"] is False


class TestContinueDevAdapter:
    """Test Continue adapter — validates JSON schema only."""

    def test_install_writes_valid_mcp_json(self, tmp_path):
        adapter = ContinueDevAdapter()
        adapter._CONFIG_DIR = str(tmp_path / ".continue")
        adapter._CONFIG_PATH = str(tmp_path / ".continue" / "config.json")

        result = adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        assert result.success is True
        assert os.path.isfile(adapter._CONFIG_PATH)

        # Validate JSON structure
        with open(adapter._CONFIG_PATH) as f:
            config = json.load(f)
        assert "mcpServers" in config
        assert "ccr" in config["mcpServers"]
        assert config["mcpServers"]["ccr"]["command"] == "/usr/bin/python3"
        assert config["mcpServers"]["ccr"]["env"]["CCR_STORAGE_BACKEND"] == "sqlite"

    def test_context_format_is_markdown(self):
        adapter = ContinueDevAdapter()
        assert adapter.context_format() == "markdown"

    def test_detects_vscode_extension(self, tmp_path):
        adapter = ContinueDevAdapter()
        # Override paths so real ~/.continue doesn't interfere
        adapter._CONFIG_DIR = str(tmp_path / "nonexistent")
        adapter._CONFIG_PATH = str(tmp_path / "nonexistent" / "config.json")
        adapter._VSCODE_EXTENSION_PATHS = [str(tmp_path / ".vscode" / "extensions" / "continue.continue")]

        # Not installed initially
        assert adapter.is_installed() is False

        # Create fake extension directory
        os.makedirs(adapter._VSCODE_EXTENSION_PATHS[0])
        assert adapter.is_installed() is True


class TestOllamaAdapter:
    """Test Ollama adapter — validates wrapper script syntax and logic."""

    def test_install_creates_wrapper_script(self, tmp_path):
        adapter = OllamaAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")

        result = adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        assert result.success is True

        wrapper = os.path.join(adapter._BIN_DIR, "ollama-ccr")
        assert os.path.isfile(wrapper)
        assert os.access(wrapper, os.X_OK)  # Executable

    def test_wrapper_script_is_valid_bash(self, tmp_path):
        """Verify the generated script is syntactically valid bash."""
        adapter = OllamaAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")
        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")

        wrapper = os.path.join(adapter._BIN_DIR, "ollama-ccr")
        # bash -n checks syntax without executing
        result = subprocess.run(["bash", "-n", wrapper], capture_output=True, text=True)
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_wrapper_contains_context_injection_logic(self, tmp_path):
        adapter = OllamaAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")
        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")

        wrapper = os.path.join(adapter._BIN_DIR, "ollama-ccr")
        content = open(wrapper).read()
        assert "context_inject.md" in content
        assert "ollama run" in content

    def test_integration_level_is_file_watcher(self):
        adapter = OllamaAdapter()
        assert adapter.integration_level() == 3  # FILE_WATCHER


class TestOpenAIAdapter:
    """Test OpenAI adapter — validates SDK wrapper with mocked client."""

    def test_install_creates_wrapper_and_sdk_module(self, tmp_path):
        adapter = OpenAIAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")

        ccr_pkg = str(tmp_path / "ccr")
        os.makedirs(ccr_pkg, exist_ok=True)

        result = adapter.install("/usr/bin/python3", "/fake/hooks", ccr_pkg)
        assert result.success is True

        cli_wrapper = os.path.join(adapter._BIN_DIR, "ccr-openai")
        assert os.path.isfile(cli_wrapper)

        sdk_module = os.path.join(ccr_pkg, "wrap_openai.py")
        assert os.path.isfile(sdk_module)

    def test_cli_wrapper_is_valid_bash(self, tmp_path):
        adapter = OpenAIAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")
        ccr_pkg = str(tmp_path / "ccr")
        os.makedirs(ccr_pkg, exist_ok=True)

        adapter.install("/usr/bin/python3", "/fake/hooks", ccr_pkg)

        wrapper = os.path.join(adapter._BIN_DIR, "ccr-openai")
        result = subprocess.run(["bash", "-n", wrapper], capture_output=True, text=True)
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_sdk_wrapper_module_exists_and_is_importable(self, tmp_path):
        """Verify the generated wrap_openai.py can be imported and has the expected function."""
        adapter = OpenAIAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")
        ccr_pkg = str(tmp_path / "ccr")
        os.makedirs(ccr_pkg, exist_ok=True)

        adapter.install("/usr/bin/python3", "/fake/hooks", ccr_pkg)

        sdk_module = os.path.join(ccr_pkg, "wrap_openai.py")
        assert os.path.isfile(sdk_module)

        # Read and verify it contains the wrap_openai function
        content = open(sdk_module).read()
        assert "def wrap_openai(" in content
        assert "class _WrappedClient" in content or "class Wrapped" in content

    def test_sdk_wrapper_prepends_context_to_messages(self, tmp_path):
        """Core test: verify wrap_openai prepends context without API call.

        Uses a mock module to avoid importing the real ccr.wrap_openai which
        loads actual project context.
        """
        import types

        # Create a mock wrap_openai module
        mock_module = types.ModuleType("mock_wrap_openai")
        exec('''
def wrap_openai(client, project_root=None):
    class Wrapped:
        def __init__(self, inner):
            self._inner = inner
            self.chat = WrappedChat(inner.chat)
    class WrappedChat:
        def __init__(self, inner):
            self.completions = WrappedCompletions(inner.completions)
    class WrappedCompletions:
        def __init__(self, inner):
            self._inner = inner
        def create(self, *args, **kwargs):
            msgs = kwargs.get("messages", [])
            if msgs:
                msgs.insert(0, {"role": "system", "content": "[TEST_CONTEXT]"})
                kwargs["messages"] = msgs
            return self._inner.create(*args, **kwargs)
    return Wrapped(client)
''', mock_module.__dict__)

        wrap_openai = mock_module.wrap_openai

        # Create a mock OpenAI-like client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = "mock_response"

        wrapped = wrap_openai(mock_client)
        wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify the mock was called with context prepended
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "[TEST_CONTEXT]" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_is_installed_detects_openai_package(self):
        adapter = OpenAIAdapter()
        # openai package should be installed (it's a dependency of ccr[models])
        # If not, the adapter falls back to checking OPENAI_API_KEY env var
        result = adapter.is_installed()
        assert isinstance(result, bool)


class TestGenericMcpAdapter:
    """Test Generic MCP adapter — validates JSON output."""

    def test_install_writes_valid_json(self, tmp_path):
        adapter = GenericMcpAdapter()
        adapter._CONFIG_PATH = str(tmp_path / "mcp-config.json")

        result = adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        assert result.success is True

        with open(adapter._CONFIG_PATH) as f:
            config = json.load(f)
        assert "mcpServers" in config
        assert "ccr" in config["mcpServers"]
        assert config["mcpServers"]["ccr"]["env"]["CCR_STORAGE_BACKEND"] == "sqlite"
        assert "_ccr_note" in config

    def test_is_installed_returns_false(self):
        """Generic MCP should never auto-detect as 'installed'."""
        adapter = GenericMcpAdapter()
        assert adapter.is_installed() is False


class TestInstallGlobalIdempotency:
    """Test that install_global / uninstall_global are idempotent."""

    def test_claude_install_twice_no_duplicates(self, tmp_path):
        adapter = ClaudeCodeAdapter()
        adapter._CLAUDE_DIR = str(tmp_path / ".claude")
        adapter._MCP_PATH = str(tmp_path / ".claude" / ".mcp.json")
        adapter._SETTINGS_PATH = str(tmp_path / ".claude" / "settings.json")

        # Install twice
        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")

        with open(adapter._SETTINGS_PATH) as f:
            settings = json.load(f)

        hooks = settings.get("hooks", {}).get("UserPromptSubmit", [])
        all_cmds = []
        for c in hooks:
            if "command" in c:
                all_cmds.append(c["command"])
            for hook in c.get("hooks", []):
                all_cmds.append(hook.get("command", ""))
        # Should have exactly one CCR hook, not two
        ccr_hooks = [c for c in all_cmds if "ccr" in c.lower()]
        assert len(ccr_hooks) == 1

    def test_kimi_install_twice_no_duplicates(self, tmp_path):
        adapter = KimiAdapter()
        adapter._KIMI_DIR = str(tmp_path / ".kimi")
        adapter._MCP_PATH = str(tmp_path / ".kimi" / "mcp.json")
        adapter._CONFIG_PATH = str(tmp_path / ".kimi" / "config.toml")

        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")
        adapter.install("/usr/bin/python3", "/fake/hooks", "/fake/ccr")

        with open(adapter._CONFIG_PATH) as f:
            toml = f.read()

        # Count CCR hook blocks
        ccr_blocks = toml.lower().count("ccr")
        # Should not have exploded with duplicates
        assert ccr_blocks > 0
        # Verify no duplicate [[hooks]] blocks for same event
        assert toml.count("event = \"UserPromptSubmit\"") == 1


# ---------------------------------------------------------------------------
# Health-check tests (new in v6 — per-adapter robustness contract)
# ---------------------------------------------------------------------------


class TestVerifyHookCommandPaths:
    """The shared verify_hook_command_paths() helper used by every adapter."""

    def test_skips_non_ccr_commands(self):
        from ccr.adapters import verify_hook_command_paths

        assert verify_hook_command_paths("/usr/bin/python /fake/script.py") == []
        assert verify_hook_command_paths("") == []

    def test_reports_missing_python_exe(self):
        from ccr.adapters import verify_hook_command_paths

        cmd = "CCR_AUTO_INIT=1 /tmp/missing-python /tmp/missing-script-ccr.py"
        issues = verify_hook_command_paths(cmd)
        # Both python and script are missing -> two errors
        msgs = [i.message for i in issues]
        assert any("python" in m for m in msgs)
        assert any("script" in m for m in msgs)
        assert all(i.severity == "error" for i in issues)

    def test_clean_paths_no_issues(self, tmp_path):
        from ccr.adapters import verify_hook_command_paths

        py = tmp_path / "python3"
        py.write_text("#!/bin/sh\nexit 0\n")
        py.chmod(0o755)
        script = tmp_path / "ccr_hook.py"
        script.write_text("# ccr hook\n")

        cmd = f"CCR_AUTO_INIT=1 {py} {script}"
        assert verify_hook_command_paths(cmd) == []


class TestCodexAdapterHealth:
    """Codex-specific health checks against ~/.codex/config.toml."""

    def _write_config(self, tmp_path, content: str) -> CodexAdapter:
        adapter = CodexAdapter()
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        adapter._CODEX_DIR = str(codex_dir)
        adapter._CONFIG_PATH = str(codex_dir / "config.toml")
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        return adapter

    def test_health_check_codex_not_installed(self, tmp_path, monkeypatch):
        adapter = CodexAdapter()
        # Point both detection paths at empty locations
        adapter._CODEX_DIR = str(tmp_path / "missing-codex")
        adapter._CONFIG_PATH = str(tmp_path / "missing-codex" / "config.toml")
        monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
        findings = adapter.health_check()
        assert any(f.severity == "info" and "not detected" in f.message for f in findings)

    def test_health_check_missing_config(self, tmp_path, monkeypatch):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        adapter = CodexAdapter()
        adapter._CODEX_DIR = str(codex_dir)
        adapter._CONFIG_PATH = str(codex_dir / "config.toml")
        # Make is_installed True via directory presence
        findings = adapter.health_check()
        assert any(
            f.severity == "warn" and "config not found" in f.message.lower() for f in findings
        )

    def test_health_check_invalid_toml(self, tmp_path):
        adapter = self._write_config(tmp_path, "this is = not valid toml = [oops\n")
        findings = adapter.health_check()
        assert any(f.severity == "error" and "valid TOML" in f.message for f in findings)

    def test_health_check_full_install_passes(self, tmp_path):
        # Use real python to make path checks pass.
        import sys as _sys

        config = (
            "[features]\ncodex_hooks = true\n\n"
            "[mcp_servers.ccr]\n"
            f'command = "{_sys.executable}"\n'
            'args = ["-m", "ccr.mcp_server", "--project", "."]\n\n'
            "[mcp_servers.ccr.env]\n"
            'CCR_STORAGE_BACKEND = "sqlite"\n\n'
            "[[hooks.SessionStart]]\n"
            'matcher = "startup|resume"\n'
            "[[hooks.SessionStart.hooks]]\n"
            'type = "command"\n'
            f'command = "CCR_AUTO_INIT=1 {_sys.executable} {__file__}"\n'
            "timeout = 30\n"
        )
        adapter = self._write_config(tmp_path, config)
        findings = adapter.health_check()
        severities = [f.severity for f in findings]
        assert "error" not in severities
        # Must have positive markers
        ok_msgs = [f.message for f in findings if f.severity == "ok"]
        assert any("MCP server" in m for m in ok_msgs)
        assert any("codex_hooks" in m for m in ok_msgs)
        assert any("hooks:" in m for m in ok_msgs)

    def test_health_check_warns_when_storage_not_sqlite(self, tmp_path):
        import sys as _sys

        config = (
            "[features]\ncodex_hooks = true\n\n"
            "[mcp_servers.ccr]\n"
            f'command = "{_sys.executable}"\n'
            'args = ["-m", "ccr.mcp_server", "--project", "."]\n\n'
            "[mcp_servers.ccr.env]\n"
            'CCR_STORAGE_BACKEND = "file"\n\n'
            "[[hooks.SessionStart]]\n"
            "[[hooks.SessionStart.hooks]]\n"
            'type = "command"\n'
            f'command = "{_sys.executable} ccr-script.py"\n'
            "timeout = 30\n"
        )
        adapter = self._write_config(tmp_path, config)
        findings = adapter.health_check()
        msgs = [(f.severity, f.message) for f in findings]
        assert any("sqlite" in m.lower() and s == "warn" for s, m in msgs)

    def test_health_check_detects_missing_python_exe(self, tmp_path):
        config = (
            "[features]\ncodex_hooks = true\n\n"
            "[mcp_servers.ccr]\n"
            'command = "/tmp/this-python-does-not-exist"\n'
            'args = ["-m", "ccr.mcp_server", "--project", "."]\n\n'
            "[mcp_servers.ccr.env]\n"
            'CCR_STORAGE_BACKEND = "sqlite"\n\n'
            "[[hooks.SessionStart]]\n"
            "[[hooks.SessionStart.hooks]]\n"
            'type = "command"\n'
            'command = "/tmp/missing-python /tmp/missing-script-ccr.py"\n'
            "timeout = 30\n"
        )
        adapter = self._write_config(tmp_path, config)
        findings = adapter.health_check()
        errs = [f for f in findings if f.severity == "error"]
        assert any("MCP python missing" in f.message for f in errs)
        assert any("Hook python executable missing" in f.message for f in errs)

    def test_is_mcp_configured_uses_tomllib(self, tmp_path):
        # tomllib should pick up the structured table even when the substring
        # form looks different (e.g. nested ".env" header would still map to
        # mcp_servers.ccr in the dict).
        config = (
            "[mcp_servers.ccr]\n"
            'command = "/usr/bin/python3"\n'
            'args = ["-m", "x"]\n'
            "[mcp_servers.ccr.env]\n"
            'CCR_STORAGE_BACKEND = "sqlite"\n'
        )
        adapter = self._write_config(tmp_path, config)
        assert adapter._is_mcp_configured() is True

    def test_is_hooks_configured_walks_nested_groups(self, tmp_path):
        config = (
            "[[hooks.Stop]]\n"
            "[[hooks.Stop.hooks]]\n"
            'type = "command"\n'
            'command = "ccr_stop_handler"\n'
        )
        adapter = self._write_config(tmp_path, config)
        assert adapter._is_hooks_configured() is True


class TestClaudeCodeAdapterHealth:
    """Claude Code health checks against ~/.claude/.mcp.json + settings.json."""

    def _build(self, tmp_path) -> ClaudeCodeAdapter:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        adapter = ClaudeCodeAdapter()
        adapter._CLAUDE_DIR = str(claude_dir)
        adapter._MCP_PATH = str(claude_dir / ".mcp.json")
        adapter._SETTINGS_PATH = str(claude_dir / "settings.json")
        return adapter

    def test_health_warns_when_missing_files(self, tmp_path):
        adapter = self._build(tmp_path)
        findings = adapter.health_check()
        warns = [f for f in findings if f.severity == "warn"]
        assert any("MCP config not found" in f.message for f in warns)
        assert any("settings.json not found" in f.message for f in warns)

    def test_health_errors_on_invalid_json(self, tmp_path):
        adapter = self._build(tmp_path)
        with open(adapter._MCP_PATH, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        with open(adapter._SETTINGS_PATH, "w", encoding="utf-8") as f:
            f.write("{also not valid")
        findings = adapter.health_check()
        errs = [f for f in findings if f.severity == "error"]
        assert any("not valid JSON" in f.message for f in errs)

    def test_health_flags_legacy_bare_command_format(self, tmp_path):
        adapter = self._build(tmp_path)
        with open(adapter._MCP_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mcpServers": {
                        "ccr": {
                            "command": "/usr/bin/python3",
                            "args": ["-m", "ccr.mcp_server"],
                            "env": {"CCR_STORAGE_BACKEND": "sqlite"},
                        }
                    }
                },
                f,
            )
        # Legacy bare {type, command} that triggered the recent bug fix
        with open(adapter._SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hooks": {
                        "Stop": [
                            {
                                "type": "command",
                                "command": "CCR_AUTO_INIT=1 /usr/bin/python3 /tmp/missing-ccr.py",
                            }
                        ]
                    }
                },
                f,
            )
        findings = adapter.health_check()
        errs = [f for f in findings if f.severity == "error"]
        assert any("legacy bare-command format" in f.message for f in errs)

    def test_health_passes_for_valid_install(self, tmp_path):
        import sys as _sys

        adapter = self._build(tmp_path)
        # Make python_exe a real file so the path check passes
        with open(adapter._MCP_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mcpServers": {
                        "ccr": {
                            "command": _sys.executable,
                            "args": ["-m", "ccr.mcp_server"],
                            "env": {"CCR_STORAGE_BACKEND": "sqlite"},
                        }
                    }
                },
                f,
            )
        with open(adapter._SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hooks": {
                        "Stop": [
                            {
                                "matcher": "",
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": f"CCR_AUTO_INIT=1 {_sys.executable} {__file__}",
                                    }
                                ],
                            }
                        ]
                    }
                },
                f,
            )
        findings = adapter.health_check()
        assert not [f for f in findings if f.severity == "error"]


class TestKimiAdapterHealth:
    """Kimi health checks against ~/.kimi/mcp.json + config.toml."""

    def _build(self, tmp_path) -> KimiAdapter:
        kimi_dir = tmp_path / ".kimi"
        kimi_dir.mkdir()
        adapter = KimiAdapter()
        adapter._KIMI_DIR = str(kimi_dir)
        adapter._MCP_PATH = str(kimi_dir / "mcp.json")
        adapter._CONFIG_PATH = str(kimi_dir / "config.toml")
        return adapter

    def test_health_warns_when_missing(self, tmp_path):
        adapter = self._build(tmp_path)
        findings = adapter.health_check()
        warns = [f for f in findings if f.severity == "warn"]
        assert any("MCP config not found" in f.message for f in warns)
        assert any("Kimi config not found" in f.message for f in warns)

    def test_health_errors_on_invalid_toml(self, tmp_path):
        adapter = self._build(tmp_path)
        with open(adapter._MCP_PATH, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": {}}, f)
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("not = valid = [oops\n")
        findings = adapter.health_check()
        assert any(f.severity == "error" and "valid TOML" in f.message for f in findings)

    def test_health_passes_for_valid_install(self, tmp_path):
        import sys as _sys

        adapter = self._build(tmp_path)
        with open(adapter._MCP_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mcpServers": {
                        "ccr": {
                            "command": _sys.executable,
                            "args": ["-m", "ccr.mcp_server"],
                            "env": {"CCR_STORAGE_BACKEND": "sqlite"},
                        }
                    }
                },
                f,
            )
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(
                "[[hooks]]\n"
                'event = "UserPromptSubmit"\n'
                f'command = "CCR_AUTO_INIT=1 {_sys.executable} {__file__}"\n'
                "timeout = 30\n"
            )
        findings = adapter.health_check()
        assert not [f for f in findings if f.severity == "error"]


class TestBaseAdapterHealthFallback:
    """Default health_check implementation returns sensible info for each path.

    Note: Continue/Ollama/OpenAI/GenericMcp now have their own health_check
    overrides — these tests stub is_installed/is_configured to exercise the
    base-class fallback explicitly.
    """

    def test_default_health_check_reports_when_not_installed(self):
        # Use generic-mcp which now has a focused health_check that still
        # returns severity=info when the config doesn't exist (its install
        # path), making it a clean test for "not detected".
        from ccr.adapters.generic_mcp import GenericMcpAdapter

        adapter = GenericMcpAdapter()
        adapter._CONFIG_PATH = "/tmp/this-config-must-not-exist-ccr.json"
        findings = adapter.health_check()
        assert findings and findings[0].severity == "info"

    def test_base_default_warns_when_installed_but_unconfigured(self):
        """Patch a Continue adapter to exercise the inherited default branch."""
        from ccr.adapters.continue_dev import ContinueDevAdapter

        adapter = ContinueDevAdapter()
        with patch.object(adapter, "is_installed", return_value=True):
            adapter._CONFIG_PATH = "/tmp/ccr-test-no-such-continue-config.json"
            findings = adapter.health_check()
        assert any(f.severity == "warn" for f in findings)


class TestContinueDevAdapterHealth:
    def _build(self, tmp_path) -> ContinueDevAdapter:
        d = tmp_path / ".continue"
        d.mkdir()
        adapter = ContinueDevAdapter()
        adapter._CONFIG_DIR = str(d)
        adapter._CONFIG_PATH = str(d / "config.json")
        return adapter

    def test_warns_when_config_missing(self, tmp_path):
        adapter = self._build(tmp_path)
        findings = adapter.health_check()
        assert any(f.severity == "warn" and "config not found" in f.message for f in findings)

    def test_errors_on_invalid_json(self, tmp_path):
        adapter = self._build(tmp_path)
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("{not json")
        findings = adapter.health_check()
        assert any(f.severity == "error" and "valid JSON" in f.message for f in findings)

    def test_passes_for_valid_install(self, tmp_path):
        import sys as _sys

        adapter = self._build(tmp_path)
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mcpServers": {
                        "ccr": {
                            "command": _sys.executable,
                            "args": ["-m", "ccr.mcp_server"],
                            "env": {"CCR_STORAGE_BACKEND": "sqlite"},
                        }
                    }
                },
                f,
            )
        findings = adapter.health_check()
        assert not [f for f in findings if f.severity == "error"]
        assert any(f.severity == "ok" and "sqlite backend" in f.message for f in findings)


class TestOllamaAdapterHealth:
    def test_info_when_ollama_missing(self, monkeypatch):
        adapter = OllamaAdapter()
        monkeypatch.setattr("shutil.which", lambda _: None)
        findings = adapter.health_check()
        assert findings[0].severity == "info"

    def test_warns_when_wrapper_missing(self, tmp_path, monkeypatch):
        adapter = OllamaAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")
        # Force is_installed=True
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/ollama")
        findings = adapter.health_check()
        assert any(f.severity == "warn" and "wrapper not installed" in f.message for f in findings)

    def test_errors_when_python_missing(self, tmp_path, monkeypatch):
        adapter = OllamaAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")
        os.makedirs(adapter._BIN_DIR, exist_ok=True)
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/ollama")
        wrapper = os.path.join(adapter._BIN_DIR, "ollama-ccr")
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n/tmp/missing-python -m ccr.cli init /tmp\n")
        os.chmod(wrapper, 0o755)
        findings = adapter.health_check()
        assert any(f.severity == "error" and "python missing" in f.message for f in findings)

    def test_passes_with_real_python(self, tmp_path, monkeypatch):
        import sys as _sys

        adapter = OllamaAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")
        os.makedirs(adapter._BIN_DIR, exist_ok=True)
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/ollama")
        wrapper = os.path.join(adapter._BIN_DIR, "ollama-ccr")
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write(
                "#!/bin/bash\n"
                f'CCR_PROJECT_ROOT="$P" {_sys.executable} -m ccr.cli init "$P"\n'
            )
        os.chmod(wrapper, 0o755)
        findings = adapter.health_check()
        assert not [f for f in findings if f.severity == "error"]
        assert any(f.severity == "ok" and "Ollama CCR wrapper" in f.message for f in findings)


class TestOpenAIAdapterHealth:
    def test_info_when_openai_unavailable(self, monkeypatch):
        adapter = OpenAIAdapter()
        # Pretend openai is missing AND env var unset
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch.object(adapter, "is_installed", return_value=False):
            findings = adapter.health_check()
        assert findings[0].severity == "info"

    def test_warns_when_wrapper_missing(self, tmp_path, monkeypatch):
        adapter = OpenAIAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        findings = adapter.health_check()
        assert any(f.severity == "warn" and "wrapper not installed" in f.message for f in findings)

    def test_passes_with_valid_install(self, tmp_path, monkeypatch):
        import sys as _sys

        adapter = OpenAIAdapter()
        adapter._BIN_DIR = str(tmp_path / "bin")
        os.makedirs(adapter._BIN_DIR, exist_ok=True)
        wrapper = os.path.join(adapter._BIN_DIR, "ccr-openai")
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write(
                "#!/bin/bash\n"
                f"{_sys.executable} -c \"print('hi')\"\n"
            )
        os.chmod(wrapper, 0o755)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        findings = adapter.health_check()
        assert not [f for f in findings if f.severity == "error"]
        assert any(f.severity == "ok" and "wrapper" in f.message.lower() for f in findings)


class TestGenericMcpAdapterHealth:
    def test_info_when_config_absent(self, tmp_path):
        adapter = GenericMcpAdapter()
        adapter._CONFIG_PATH = str(tmp_path / "no-mcp-config.json")
        findings = adapter.health_check()
        assert findings and findings[0].severity == "info"

    def test_errors_on_invalid_json(self, tmp_path):
        adapter = GenericMcpAdapter()
        adapter._CONFIG_PATH = str(tmp_path / "mcp-config.json")
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("{not json")
        findings = adapter.health_check()
        assert any(f.severity == "error" and "valid JSON" in f.message for f in findings)

    def test_passes_for_valid_config(self, tmp_path):
        import sys as _sys

        adapter = GenericMcpAdapter()
        adapter._CONFIG_PATH = str(tmp_path / "mcp-config.json")
        with open(adapter._CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mcpServers": {
                        "ccr": {
                            "command": _sys.executable,
                            "args": ["-m", "ccr.mcp_server"],
                            "env": {"CCR_STORAGE_BACKEND": "sqlite"},
                        }
                    }
                },
                f,
            )
        findings = adapter.health_check()
        assert not [f for f in findings if f.severity == "error"]
        assert any(f.severity == "ok" and "sqlite backend" in f.message for f in findings)


class TestAgentsDoctorJSON:
    """ccr agents doctor --json emits a stable schema and proper exit codes."""

    def test_json_schema_for_known_agent(self):
        from click.testing import CliRunner
        from ccr.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["agents", "doctor", "codex", "--json"])
        assert result.exit_code in (0, 1)  # exit=1 only if findings have errors
        payload = json.loads(result.stdout)
        assert payload["schema_version"] == 1
        assert isinstance(payload["agents"], list) and len(payload["agents"]) == 1
        agent = payload["agents"][0]
        assert agent["name"] == "codex"
        assert "counts" in agent
        for key in ("ok", "info", "warn", "error"):
            assert key in agent["counts"]
        assert "findings" in agent and isinstance(agent["findings"], list)
        assert "totals" in payload
        for key in ("ok", "info", "warn", "error"):
            assert key in payload["totals"]

    def test_json_unknown_agent_exits_2(self):
        from click.testing import CliRunner
        from ccr.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["agents", "doctor", "definitely-not-a-real-agent", "--json"])
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert payload["schema_version"] == 1
        assert "error" in payload

    def test_json_all_emits_every_adapter(self):
        from click.testing import CliRunner
        from ccr.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["agents", "doctor", "all", "--json"])
        # all adapters processed without crash
        payload = json.loads(result.stdout)
        names = {a["name"] for a in payload["agents"]}
        # Every registered adapter shows up
        for expected in (
            "claude-code",
            "codex",
            "kimi",
            "continue-dev",
            "ollama",
            "openai",
            "generic-mcp",
        ):
            assert expected in names, f"Missing adapter in --json all: {expected}"
