"""Kimi Code CLI adapter — MCP + TOML hooks via ~/.kimi/config.toml."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil

from ccr.adapters import (
    BaseAgentAdapter,
    HealthIssue,
    InstallResult,
    UninstallResult,
    verify_hook_command_paths,
)


try:
    import tomllib as _tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - 3.10 fallback
    _tomllib = None  # type: ignore[assignment]


class KimiAdapter(BaseAgentAdapter):
    name = "kimi"
    display_name = "Kimi Code CLI"

    _KIMI_DIR = os.path.expanduser("~/.kimi")
    _MCP_PATH = os.path.join(_KIMI_DIR, "mcp.json")
    _CONFIG_PATH = os.path.join(_KIMI_DIR, "config.toml")

    def is_installed(self) -> bool:
        """Detect via presence of kimi binary or ~/.kimi directory."""
        return shutil.which("kimi") is not None or os.path.isdir(self._KIMI_DIR)

    def supports_mcp(self) -> bool:
        return True

    def supports_hooks(self) -> bool:
        return True

    def install(self, python_exe: str, hooks_dir: str, ccr_pkg: str) -> InstallResult:
        """Write ~/.kimi/mcp.json and append hooks to ~/.kimi/config.toml."""
        os.makedirs(self._KIMI_DIR, exist_ok=True)
        files_created: list[str] = []
        files_modified: list[str] = []

        mcp_path = self._ensure_mcp(python_exe)
        if os.path.exists(mcp_path):
            files_modified.append(mcp_path)
        else:
            files_created.append(mcp_path)

        config_path = self._ensure_hooks(python_exe, hooks_dir)
        if os.path.exists(config_path):
            files_modified.append(config_path)
        else:
            files_created.append(config_path)

        return InstallResult(
            success=True,
            message=f"Kimi MCP: {mcp_path}\nKimi hooks: {config_path}",
            files_created=files_created,
            files_modified=files_modified,
        )

    def uninstall(self) -> UninstallResult:
        """Remove CCR from ~/.kimi/mcp.json and ~/.kimi/config.toml."""
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

        # Config TOML
        if os.path.isfile(self._CONFIG_PATH):
            try:
                with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                    content = f.read()
                content = re.sub(r"^hooks\s*=\s*\[\]\s*\n?", "", content, flags=re.MULTILINE)
                rest = content
                cleaned = ""
                while True:
                    match = re.search(r"\[\[hooks\]\]\n", rest)
                    if not match:
                        cleaned += rest
                        break
                    start = match.start()
                    next_match = re.search(r"\[\[hooks\]\]\n", rest[start + 1 :])
                    if next_match:
                        end = start + 1 + next_match.start()
                        block = rest[start:end]
                        if "ccr" not in block.lower():
                            cleaned += block
                        rest = rest[:start] + rest[end:]
                    else:
                        block = rest[start:]
                        if "ccr" not in block.lower():
                            cleaned += block
                        rest = rest[:start]
                        break
                with open(self._CONFIG_PATH, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                files_modified.append(self._CONFIG_PATH)
            except Exception:
                pass

        return UninstallResult(
            success=True,
            message="Removed CCR from Kimi Code CLI global config",
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
        if not os.path.isfile(self._CONFIG_PATH):
            return False
        try:
            with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            return "ccr" in content.lower() and "[[hooks]]" in content
        except Exception:
            return False

    def health_check(self) -> list[HealthIssue]:
        """Kimi-specific health checks for ~/.kimi/mcp.json + config.toml."""
        if not self.is_installed():
            return [
                HealthIssue(
                    severity="info",
                    message="Kimi Code CLI not detected on this system",
                    fix="Install Kimi, then run 'ccr install-global --agents kimi'",
                )
            ]

        issues: list[HealthIssue] = []

        # MCP — JSON validity + ccr entry
        if not os.path.isfile(self._MCP_PATH):
            issues.append(
                HealthIssue(
                    severity="warn",
                    message=f"Kimi MCP config not found: {self._MCP_PATH}",
                    fix="Run: ccr install-global --agents kimi",
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
                            message="Kimi MCP missing ccr server entry",
                            fix="Run: ccr install-global --agents kimi",
                        )
                    )
                else:
                    cmd = ccr_entry.get("command", "")
                    if cmd and not os.path.isfile(cmd):
                        issues.append(
                            HealthIssue(
                                severity="error",
                                message=f"Kimi MCP python missing: {cmd}",
                                fix="Re-run 'ccr install-global --agents kimi' after fixing the venv",
                            )
                        )
                    env = ccr_entry.get("env") or {}
                    if not isinstance(env, dict) or env.get("CCR_STORAGE_BACKEND") != "sqlite":
                        issues.append(
                            HealthIssue(
                                severity="warn",
                                message="Kimi MCP server not pinned to CCR_STORAGE_BACKEND=sqlite",
                                fix="Run: ccr install-global --agents kimi",
                            )
                        )
                    else:
                        issues.append(
                            HealthIssue(
                                severity="ok",
                                message="Kimi MCP server: ccr (sqlite backend)",
                            )
                        )
            except json.JSONDecodeError:
                issues.append(
                    HealthIssue(
                        severity="error",
                        message=f"Kimi MCP config is not valid JSON: {self._MCP_PATH}",
                        fix="Manually fix syntax or remove the file and run 'ccr install-global --agents kimi'",
                    )
                )
            except OSError as exc:
                issues.append(
                    HealthIssue(
                        severity="error",
                        message=f"Kimi MCP config unreadable: {exc}",
                    )
                )

        # Hooks — TOML
        if not os.path.isfile(self._CONFIG_PATH):
            issues.append(
                HealthIssue(
                    severity="warn",
                    message=f"Kimi config not found: {self._CONFIG_PATH}",
                    fix="Run: ccr install-global --agents kimi",
                )
            )
            return issues

        try:
            with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            issues.append(
                HealthIssue(severity="error", message=f"Kimi config unreadable: {exc}")
            )
            return issues

        parsed: dict | None = None
        if _tomllib is not None:
            try:
                parsed = _tomllib.loads(text)
            except _tomllib.TOMLDecodeError:
                issues.append(
                    HealthIssue(
                        severity="error",
                        message=f"Kimi config is not valid TOML: {self._CONFIG_PATH}",
                        fix="Manually fix syntax or run 'ccr install-global --agents kimi'",
                    )
                )

        hook_event_count = 0
        path_issues: list[HealthIssue] = []
        if parsed is not None:
            entries = parsed.get("hooks") or []
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    cmd = str(entry.get("command", ""))
                    if "ccr" not in cmd.lower():
                        continue
                    hook_event_count += 1
                    path_issues.extend(verify_hook_command_paths(cmd))
        else:
            # Fallback substring scan if TOML failed to parse.
            for match in re.finditer(r"command\s*=\s*\"([^\"]+)\"", text):
                cmd = match.group(1)
                if "ccr" not in cmd.lower():
                    continue
                hook_event_count += 1
                path_issues.extend(verify_hook_command_paths(cmd))

        if hook_event_count == 0:
            issues.append(
                HealthIssue(
                    severity="warn",
                    message="No CCR hook entries found in ~/.kimi/config.toml",
                    fix="Run: ccr install-global --agents kimi",
                )
            )
        else:
            issues.append(
                HealthIssue(
                    severity="ok",
                    message=f"Kimi CCR hooks: {hook_event_count} command(s) registered",
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
        content = ""
        if os.path.isfile(self._CONFIG_PATH):
            with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                content = f.read()

        # Remove existing CCR hooks
        content = re.sub(r"^hooks\s*=\s*\[\]\s*\n?", "", content, flags=re.MULTILINE)
        hook_blocks = []
        rest = content
        while True:
            match = re.search(r"\[\[hooks\]\]\n", rest)
            if not match:
                break
            start = match.start()
            next_match = re.search(r"\[\[hooks\]\]\n", rest[start + 1 :])
            if next_match:
                end = start + 1 + next_match.start()
                block = rest[start:end]
                if "ccr" not in block.lower():
                    hook_blocks.append(block)
                rest = rest[:start] + rest[end:]
            else:
                block = rest[start:]
                if "ccr" not in block.lower():
                    hook_blocks.append(block)
                rest = rest[:start]
                break

        ccr_hooks_toml = f"""
[[hooks]]
event = "UserPromptSubmit"
command = "CCR_AUTO_INIT=1 CCR_STORAGE_BACKEND=sqlite {shlex.quote(python_exe)} {shlex.quote(os.path.join(hooks_dir, 'on_session_start.py'))}"
timeout = 30

[[hooks]]
event = "PostToolUse"
command = "CCR_AUTO_INIT=1 CCR_STORAGE_BACKEND=sqlite {shlex.quote(python_exe)} {shlex.quote(os.path.join(hooks_dir, 'on_tool_use.py'))}"
timeout = 30

[[hooks]]
event = "PreCompact"
command = "CCR_AUTO_INIT=1 CCR_STORAGE_BACKEND=sqlite {shlex.quote(python_exe)} {shlex.quote(os.path.join(hooks_dir, 'on_compact.py'))}"
timeout = 30

[[hooks]]
event = "Stop"
command = "CCR_AUTO_INIT=1 CCR_STORAGE_BACKEND=sqlite {shlex.quote(python_exe)} {shlex.quote(os.path.join(hooks_dir, 'on_stop.py'))}"
timeout = 60
"""

        new_content = rest.rstrip() + "\n" + ccr_hooks_toml.lstrip("\n")
        with open(self._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        return self._CONFIG_PATH
