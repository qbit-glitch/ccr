"""Claude Code adapter — MCP + JSON hooks via ~/.claude/ config."""

from __future__ import annotations

import json
import os
import shlex
import shutil

from ccr.adapters import (
    BaseAgentAdapter,
    HealthIssue,
    InstallResult,
    UninstallResult,
    verify_hook_command_paths,
)


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
                    cleaned: list[dict] = []
                    for c in d["hooks"][event]:
                        if "ccr" in c.get("command", "").lower():
                            continue
                        nested = c.get("hooks")
                        if isinstance(nested, list):
                            kept = [h for h in nested if "ccr" not in h.get("command", "").lower()]
                            if kept:
                                updated = dict(c)
                                updated["hooks"] = kept
                                cleaned.append(updated)
                            continue
                        cleaned.append(c)
                    d["hooks"][event] = cleaned
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
                    for hook in c.get("hooks", []):
                        if "ccr" in hook.get("command", "").lower():
                            return True
            return False
        except Exception:
            return False

    def health_check(self) -> list[HealthIssue]:
        """Claude-Code-specific health checks for ~/.claude/ config files."""
        if not self.is_installed():
            return [
                HealthIssue(
                    severity="info",
                    message="Claude Code not detected on this system",
                    fix="Install Claude Code (npm install -g @anthropic-ai/claude-code), then run 'ccr install-global --agents claude-code'",
                )
            ]

        issues: list[HealthIssue] = []

        # MCP — JSON validity + ccr entry
        if not os.path.isfile(self._MCP_PATH):
            issues.append(
                HealthIssue(
                    severity="warn",
                    message=f"Claude Code MCP config not found: {self._MCP_PATH}",
                    fix="Run: ccr install-global --agents claude-code",
                )
            )
        else:
            try:
                with open(self._MCP_PATH, "r", encoding="utf-8") as f:
                    mcp = json.loads(f.read())
                ccr_entry = (mcp.get("mcpServers") or {}).get("ccr")
                if not isinstance(ccr_entry, dict):
                    issues.append(
                        HealthIssue(
                            severity="warn",
                            message="Claude Code MCP missing ccr server entry",
                            fix="Run: ccr install-global --agents claude-code",
                        )
                    )
                else:
                    cmd = ccr_entry.get("command", "")
                    if cmd and not os.path.isfile(cmd):
                        issues.append(
                            HealthIssue(
                                severity="error",
                                message=f"Claude Code MCP python missing: {cmd}",
                                fix="Re-run 'ccr install-global --agents claude-code' after fixing the venv",
                            )
                        )
                    env = ccr_entry.get("env") or {}
                    if not isinstance(env, dict) or env.get("CCR_STORAGE_BACKEND") != "sqlite":
                        issues.append(
                            HealthIssue(
                                severity="warn",
                                message="Claude Code MCP server not pinned to CCR_STORAGE_BACKEND=sqlite",
                                fix="Run: ccr install-global --agents claude-code",
                            )
                        )
                    else:
                        issues.append(
                            HealthIssue(
                                severity="ok",
                                message="Claude Code MCP server: ccr (sqlite backend)",
                            )
                        )
            except json.JSONDecodeError:
                issues.append(
                    HealthIssue(
                        severity="error",
                        message=f"Claude Code MCP config is not valid JSON: {self._MCP_PATH}",
                        fix="Manually fix syntax or remove the file and run 'ccr install-global --agents claude-code'",
                    )
                )
            except OSError as exc:
                issues.append(
                    HealthIssue(
                        severity="error",
                        message=f"Claude Code MCP config unreadable: {exc}",
                    )
                )

        # Hooks — settings.json must use {matcher, hooks:[...]} shape
        if not os.path.isfile(self._SETTINGS_PATH):
            issues.append(
                HealthIssue(
                    severity="warn",
                    message=f"Claude Code settings.json not found: {self._SETTINGS_PATH}",
                    fix="Run: ccr install-global --agents claude-code",
                )
            )
            return issues

        try:
            with open(self._SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = json.loads(f.read())
        except json.JSONDecodeError:
            issues.append(
                HealthIssue(
                    severity="error",
                    message=f"Claude Code settings.json is not valid JSON: {self._SETTINGS_PATH}",
                    fix="Restore the file or remove and re-run 'ccr install-global --agents claude-code'",
                )
            )
            return issues
        except OSError as exc:
            issues.append(
                HealthIssue(
                    severity="error",
                    message=f"Claude Code settings.json unreadable: {exc}",
                )
            )
            return issues

        hooks = settings.get("hooks") or {}
        hook_event_count = 0
        path_issues: list[HealthIssue] = []
        bare_command_warnings: list[str] = []
        for event_name, entries in hooks.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if "command" in entry and "hooks" not in entry:
                    if "ccr" in str(entry.get("command", "")).lower():
                        bare_command_warnings.append(event_name)
                        path_issues.extend(verify_hook_command_paths(entry["command"]))
                        hook_event_count += 1
                    continue
                nested = entry.get("hooks") or []
                if not isinstance(nested, list):
                    continue
                for hook in nested:
                    if not isinstance(hook, dict):
                        continue
                    cmd = str(hook.get("command", ""))
                    if "ccr" not in cmd.lower():
                        continue
                    hook_event_count += 1
                    path_issues.extend(verify_hook_command_paths(cmd))

        if bare_command_warnings:
            issues.append(
                HealthIssue(
                    severity="error",
                    message=(
                        "Claude Code settings.json has CCR hook entries in the legacy "
                        f"bare-command format under: {', '.join(sorted(set(bare_command_warnings)))}"
                    ),
                    fix="Re-run 'ccr install-global --agents claude-code' to migrate to {matcher, hooks:[...]}",
                )
            )

        if hook_event_count == 0:
            issues.append(
                HealthIssue(
                    severity="warn",
                    message="No CCR hook entries found in Claude Code settings.json",
                    fix="Run: ccr install-global --agents claude-code",
                )
            )
        else:
            issues.append(
                HealthIssue(
                    severity="ok",
                    message=f"Claude Code CCR hooks: {hook_event_count} command(s) registered",
                )
            )

        seen: set[tuple[str, str]] = set()
        for issue in path_issues:
            key = (issue.severity, issue.message)
            if key in seen:
                continue
            seen.add(key)
            issues.append(issue)

        return issues

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

        def _hook_entry(script_file: str) -> dict:
            cmd = f"CCR_AUTO_INIT=1 CCR_STORAGE_BACKEND=sqlite {_py} {_q(os.path.join(hooks_dir, script_file))}"
            return {"matcher": "", "hooks": [{"type": "command", "command": cmd}]}

        ccr_hooks = {
            "UserPromptSubmit": [_hook_entry("on_session_start.py")],
            "PostToolUse": [_hook_entry("on_tool_use.py")],
            "Stop": [_hook_entry("on_stop.py")],
            "PreCompact": [_hook_entry("on_compact.py")],
        }

        existing.setdefault("hooks", {})
        for event, commands in ccr_hooks.items():
            try:
                hook_script = shlex.split(commands[0]["hooks"][0]["command"])[-1]
            except (ValueError, IndexError, KeyError):
                hook_script = commands[0]["hooks"][0]["command"].split()[-1]
            existing_cmds = self._remove_existing_hook_script(
                existing["hooks"].get(event, []),
                hook_script,
            )
            existing["hooks"][event] = existing_cmds + commands

        with open(self._SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
            f.write("\n")
        return self._SETTINGS_PATH

    @staticmethod
    def _remove_existing_hook_script(entries: list[dict], hook_script: str) -> list[dict]:
        """Remove prior CCR hook commands from both Claude hook shapes.

        Claude settings may store commands either as direct command entries or
        as grouped entries with a nested ``hooks`` list. Preserve non-CCR hooks
        in a mixed nested group.
        """
        cleaned: list[dict] = []
        for entry in entries:
            command = entry.get("command", "")
            if command and hook_script in command:
                continue
            nested = entry.get("hooks")
            if isinstance(nested, list):
                kept_hooks = [
                    hook
                    for hook in nested
                    if hook_script not in hook.get("command", "")
                ]
                if kept_hooks:
                    updated = dict(entry)
                    updated["hooks"] = kept_hooks
                    cleaned.append(updated)
                continue
            cleaned.append(entry)
        return cleaned
