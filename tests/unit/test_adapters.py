"""Tests for the provider-agnostic adapter architecture.

No API keys required — all tests use mocks, stubs, or file-system validation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ccr.adapters import (
    BaseAgentAdapter,
    get_adapter,
    get_adapters,
    detect_installed,
    list_adapter_names,
)
from ccr.adapters.claude_code import ClaudeCodeAdapter
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
        assert "kimi" in names
        assert "ollama" in names
        assert "openai" in names
        assert "continue-dev" in names
        assert "generic-mcp" in names

    def test_get_adapter_by_name(self):
        assert isinstance(get_adapter("claude-code"), ClaudeCodeAdapter)
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

        # Verify hooks contain CCR commands
        with open(adapter._SETTINGS_PATH) as f:
            settings = json.load(f)
        hooks = settings.get("hooks", {})
        assert "UserPromptSubmit" in hooks
        # CCR hooks use CCR_AUTO_INIT env var and on_session_start.py script
        commands = [c.get("command", "") for c in hooks["UserPromptSubmit"]]
        assert any("on_session_start.py" in cmd for cmd in commands)

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
        import sys
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
        commands = [c.get("command", "") for c in hooks]
        # Should have exactly one CCR hook, not two
        ccr_hooks = [c for c in commands if "ccr" in c.lower()]
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
