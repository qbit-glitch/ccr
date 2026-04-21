"""Claude Code adapter — MCP + JSON hooks via ~/.claude/ config."""

from __future__ import annotations

import json
import os
import shlex
import shutil

from ccr.adapters import BaseAgentAdapter, InstallResult, UninstallResult


class ClaudeCodeAdapter(BaseAgentAdapter):
    name = "claude-code"
    display_name = "Claude Code"

    # Config paths
    _CLAUDE_DIR = os.path.expanduser("~/.claude")
    _MCP_PATH = os.path.join(_CLAUDE_DIR, ".mcp.json")
    _SETTINGS_PATH = os.path.join(_CLAUDE_DIR, "settings.json")

    def is_installed(self) -> bool:
        """Detect via presence of claude binary or ~/.claude directory."""
        return shutil.which("claude") is not None or os.path.isdir(self._CLAUDE_DIR)

    def supports_mcp(self) -> bool:
        return True

    def supports_hooks(self) -> bool:
        return True

    def install(self, python_exe: str, hooks_dir: str, ccr_pkg: str) -> InstallResult:
        """Write ~/.claude/.mcp.json and ~/.claude/settings.json."""
        os.makedirs(self._CLAUDE_DIR, exist_ok=True)
        files_created: list[str] = []
        files_modified: list[str] = []

        # MCP config
        mcp_path = self._ensure_mcp(python_exe)
        if os.path.exists(mcp_path):
            files_modified.append(mcp_path)
        else:
            files_created.append(mcp_path)

        # Hooks config
        settings_path = self._ensure_hooks(python_exe, hooks_dir)
        if os.path.exists(settings_path):
            files_modified.append(settings_path)
        else:
            files_created.append(settings_path)

        return InstallResult(
            success=True,
            message=f"Claude Code MCP: {mcp_path}\nClaude Code hooks: {settings_path}",
            files_created=files_created,
            files_modified=files_modified,
        )

    def uninstall(self) -> UninstallResult:
        """Remove CCR entries from ~/.claude/.mcp.json and ~/.claude/settings.json."""
        files_modified: list[str] = []

        # MCP
        if os.path.isfile(self._MCP_PATH):
            try:
                with open(self._MCP_PATH, "r", encoding="utf-8") as f:
                    d = json.loads(f.read())
                d.get("mcpServers", {}).pop("ccr", None)
                with open(self._MCP_PATH, "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=2)
                    f.write("\n")
                files_modified.append(self._MCP_PATH)
            except Exception:
                pass

        # Hooks
        if os.path.isfile(self._SETTINGS_PATH):
            try:
                with open(self._SETTINGS_PATH, "r", encoding="utf-8") as f:
                    d = json.loads(f.read())
                for event in list(d.get("hooks", {}).keys()):
                    d["hooks"][event] = [
                        c
                        for c in d["hooks"][event]
                        if "ccr" not in c.get("command", "").lower()
                    ]
                    if not d["hooks"][event]:
                        del d["hooks"][event]
                if not d.get("hooks"):
                    d.pop("hooks", None)
                with open(self._SETTINGS_PATH, "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=2)
                    f.write("\n")
                files_modified.append(self._SETTINGS_PATH)
            except Exception:
                pass

        return UninstallResult(
            success=True,
            message="Removed CCR from Claude Code global config",
            files_modified=files_modified,
        )

    def _is_mcp_configured(self) -> bool:
        if not os.path.isfile(self._MCP_PATH):
            return False
        try:
            with open(self._MCP_PATH, "r", encoding="utf-8") as f:
                d = json.loads(f.read())
            return "ccr" in d.get("mcpServers", {})
        except Exception:
            return False

    def _is_hooks_configured(self) -> bool:
        if not os.path.isfile(self._SETTINGS_PATH):
            return False
        try:
            with open(self._SETTINGS_PATH, "r", encoding="utf-8") as f:
                d = json.loads(f.read())
            for event_cmds in d.get("hooks", {}).values():
                for c in event_cmds:
                    if "ccr" in c.get("command", "").lower():
                        return True
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal: config file manipulation
    # ------------------------------------------------------------------

    def _ensure_mcp(self, python_exe: str) -> str:
        existing: dict = {}
        if os.path.isfile(self._MCP_PATH):
            try:
                with open(self._MCP_PATH, "r", encoding="utf-8") as f:
                    existing = json.loads(f.read())
            except (json.JSONDecodeError, OSError):
                pass

        existing.setdefault("mcpServers", {})
        existing["mcpServers"]["ccr"] = self.get_mcp_config(python_exe)

        with open(self._MCP_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
            f.write("\n")
        return self._MCP_PATH

    def _ensure_hooks(self, python_exe: str, hooks_dir: str) -> str:
        existing: dict = {}
        if os.path.isfile(self._SETTINGS_PATH):
            try:
                with open(self._SETTINGS_PATH, "r", encoding="utf-8") as f:
                    existing = json.loads(f.read())
            except (json.JSONDecodeError, OSError):
                pass

        _q = shlex.quote
        _py = _q(python_exe)

        ccr_hooks = {
            "UserPromptSubmit": [{
                "type": "command",
                "command": f"CCR_AUTO_INIT=1 {_py} {_q(os.path.join(hooks_dir, 'on_session_start.py'))}",
            }],
            "PostToolUse": [{
                "type": "command",
                "command": f"CCR_AUTO_INIT=1 {_py} {_q(os.path.join(hooks_dir, 'on_tool_use.py'))}",
            }],
            "Stop": [{
                "type": "command",
                "command": f"CCR_AUTO_INIT=1 {_py} {_q(os.path.join(hooks_dir, 'on_stop.py'))}",
            }],
            "PreCompact": [{
                "type": "command",
                "command": f"CCR_AUTO_INIT=1 {_py} {_q(os.path.join(hooks_dir, 'on_compact.py'))}",
            }],
        }

        existing.setdefault("hooks", {})
        for event, commands in ccr_hooks.items():
            try:
                hook_script = shlex.split(commands[0]["command"])[-1]
            except (ValueError, IndexError):
                hook_script = commands[0]["command"].split()[-1]
            existing_cmds = [
                c
                for c in existing["hooks"].get(event, [])
                if hook_script not in c.get("command", "")
            ]
            existing["hooks"][event] = existing_cmds + commands

        with open(self._SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
            f.write("\n")
        return self._SETTINGS_PATH
