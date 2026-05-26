"""Codex adapter - global MCP + hooks via ~/.codex/config.toml."""

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


class CodexAdapter(BaseAgentAdapter):
    """OpenAI Codex CLI integration.

    Codex stores global MCP servers in ~/.codex/config.toml using tables like:

        [mcp_servers.ccr]
        command = "/path/to/python"
        args = ["-m", "ccr.mcp_server", "--project", "."]

    The ``--project .`` argument is intentionally relative so Codex launches CCR
    against the current project directory for each session.
    """

    name = "codex"
    display_name = "Codex CLI"

    _CODEX_DIR = os.path.expanduser("~/.codex")
    _CONFIG_PATH = os.path.join(_CODEX_DIR, "config.toml")

    _CCR_TABLE_RE = re.compile(
        r"(?ms)^\[mcp_servers\.ccr(?:\.[^\]]+)?\]\s*\n.*?"
        r"(?=^\[(?!mcp_servers\.ccr(?:\.|\]))|\Z)"
    )
    _HOOK_GROUP_RE = re.compile(
        r"(?ms)^\[\[hooks\.[^. \]]+\]\]\s*\n.*?(?=^\[\[hooks\.[^. \]]+\]\]|^\[[^\[]|\Z)"
    )
    _CCR_HOOK_FEATURE_COMMENT = "# Added by CCR: enables Codex lifecycle hooks."
    _CCR_PREVIOUS_HOOK_FEATURE_PREFIX = "# ccr-previous-codex_hooks = "
    _POST_TOOL_USE_ENV = "CCR_CODEX_POST_TOOL_USE"

    def is_installed(self) -> bool:
        """Detect via presence of codex binary or ~/.codex directory."""
        return shutil.which("codex") is not None or os.path.isdir(self._CODEX_DIR)

    def supports_mcp(self) -> bool:
        return True

    def supports_hooks(self) -> bool:
        return True

    def install(self, python_exe: str, hooks_dir: str, ccr_pkg: str) -> InstallResult:
        """Write or replace the global CCR MCP server and hook entries for Codex."""
        os.makedirs(self._CODEX_DIR, exist_ok=True)
        existed = os.path.isfile(self._CONFIG_PATH)

        content = ""
        if existed:
            try:
                with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                content = ""

        managed_hooks_feature = self._should_manage_hooks_feature(content)
        previous_hooks_value = self._codex_hooks_feature_value(content)

        content = self._remove_ccr_table(content)
        content = self._remove_ccr_hook_groups(content)
        content = self._ensure_hooks_feature(
            content,
            managed=managed_hooks_feature,
            previous_value=previous_hooks_value,
        )
        content = content.rstrip()
        if content:
            content += "\n\n"
        content += self._ccr_table(python_exe)
        content += "\n"
        include_post_tool_use = os.environ.get(self._POST_TOOL_USE_ENV, "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        content += self._ccr_hooks_toml(
            python_exe,
            hooks_dir,
            include_post_tool_use=include_post_tool_use,
        )

        with open(self._CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")

        return InstallResult(
            success=True,
            message=f"Codex MCP + hooks config written to {self._CONFIG_PATH}",
            files_created=[] if existed else [self._CONFIG_PATH],
            files_modified=[self._CONFIG_PATH] if existed else [],
        )

    def uninstall(self) -> UninstallResult:
        """Remove global CCR MCP server and hook entries from Codex config."""
        if not os.path.isfile(self._CONFIG_PATH):
            return UninstallResult(success=True, message="No Codex config found")

        try:
            with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = self._remove_ccr_hook_groups(self._remove_ccr_table(content))
            if not self._has_hook_groups(new_content):
                new_content = self._remove_managed_hooks_feature(new_content)
            new_content = new_content.rstrip()
            if new_content:
                new_content += "\n"
            with open(self._CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(new_content)
            return UninstallResult(
                success=True,
                message="Removed CCR from Codex config",
                files_modified=[self._CONFIG_PATH],
            )
        except Exception as exc:
            return UninstallResult(success=False, message=str(exc))

    def _read_config(self) -> tuple[str, dict | None]:
        """Return (raw_text, parsed_dict_or_None) for ~/.codex/config.toml.

        Falls back to (text, None) when tomllib is unavailable or the file is
        not parseable; callers can degrade to substring checks.
        """
        if not os.path.isfile(self._CONFIG_PATH):
            return "", None
        try:
            with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return "", None
        if _tomllib is None:
            return text, None
        try:
            return text, _tomllib.loads(text)
        except _tomllib.TOMLDecodeError:
            return text, None

    def _is_mcp_configured(self) -> bool:
        text, parsed = self._read_config()
        if parsed is not None:
            servers = parsed.get("mcp_servers") or {}
            return "ccr" in servers if isinstance(servers, dict) else False
        return "[mcp_servers.ccr]" in text if text else False

    def _is_hooks_configured(self) -> bool:
        text, parsed = self._read_config()
        if parsed is not None:
            hooks = parsed.get("hooks") or {}
            if not isinstance(hooks, dict):
                return False
            for entries in hooks.values():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    nested = entry.get("hooks") or []
                    cmds = [h.get("command", "") for h in nested if isinstance(h, dict)]
                    cmds.append(entry.get("command", ""))
                    if any("ccr" in str(c).lower() for c in cmds):
                        return True
            return False
        if not text:
            return False
        lowered = text.lower()
        return "hooks." in lowered and "ccr" in lowered

    def context_format(self) -> str:
        return "markdown"

    def health_check(self) -> list[HealthIssue]:
        """Codex-specific health checks for ~/.codex/config.toml.

        Verifies: file readable, valid TOML, [mcp_servers.ccr] present and
        wired to sqlite, [features].codex_hooks=true, hook commands point
        to existing python/script paths.
        """
        if not self.is_installed():
            return [
                HealthIssue(
                    severity="info",
                    message="Codex CLI not detected on this system",
                    fix="Install codex (https://github.com/openai/codex), then run 'ccr install-global --agents codex'",
                )
            ]

        issues: list[HealthIssue] = []

        if not os.path.isfile(self._CONFIG_PATH):
            issues.append(
                HealthIssue(
                    severity="warn",
                    message=f"Codex config not found: {self._CONFIG_PATH}",
                    fix="Run: ccr install-global --agents codex",
                )
            )
            return issues

        text, parsed = self._read_config()
        if not text:
            issues.append(
                HealthIssue(
                    severity="error",
                    message=f"Codex config unreadable: {self._CONFIG_PATH}",
                    fix="Check file permissions and rerun 'ccr install-global --agents codex'",
                )
            )
            return issues

        if parsed is None:
            issues.append(
                HealthIssue(
                    severity="error",
                    message=f"Codex config is not valid TOML: {self._CONFIG_PATH}",
                    fix="Manually fix syntax or remove the file and run 'ccr install-global --agents codex'",
                )
            )
            return issues

        # MCP server checks
        servers = parsed.get("mcp_servers") or {}
        ccr_server = servers.get("ccr") if isinstance(servers, dict) else None
        if not isinstance(ccr_server, dict):
            issues.append(
                HealthIssue(
                    severity="warn",
                    message="Codex [mcp_servers.ccr] missing",
                    fix="Run: ccr install-global --agents codex",
                )
            )
        else:
            command = ccr_server.get("command", "")
            if command and not os.path.isfile(command):
                issues.append(
                    HealthIssue(
                        severity="error",
                        message=f"Codex MCP python missing: {command}",
                        fix="Re-run 'ccr install-global --agents codex' after fixing the venv",
                    )
                )
            env = ccr_server.get("env") or {}
            if not isinstance(env, dict) or env.get("CCR_STORAGE_BACKEND") != "sqlite":
                issues.append(
                    HealthIssue(
                        severity="warn",
                        message="Codex MCP server not pinned to CCR_STORAGE_BACKEND=sqlite",
                        fix="Run: ccr install-global --agents codex",
                    )
                )
            else:
                issues.append(
                    HealthIssue(severity="ok", message="Codex MCP server: ccr (sqlite backend)")
                )

        # Hooks feature flag
        features = parsed.get("features") or {}
        if isinstance(features, dict) and features.get("codex_hooks") is True:
            issues.append(HealthIssue(severity="ok", message="Codex codex_hooks feature enabled"))
        else:
            issues.append(
                HealthIssue(
                    severity="warn",
                    message="Codex codex_hooks feature is not enabled",
                    fix="Run: ccr install-global --agents codex (sets [features].codex_hooks = true)",
                )
            )

        # Hook command path validation
        hooks = parsed.get("hooks") or {}
        hook_event_count = 0
        path_issues: list[HealthIssue] = []
        if isinstance(hooks, dict):
            for event_name, entries in hooks.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    nested = entry.get("hooks") or []
                    cmds: list[str] = []
                    if isinstance(nested, list):
                        cmds.extend(
                            str(h.get("command", ""))
                            for h in nested
                            if isinstance(h, dict)
                        )
                    cmds.append(str(entry.get("command", "")))
                    for cmd in cmds:
                        if not cmd or "ccr" not in cmd.lower():
                            continue
                        hook_event_count += 1
                        path_issues.extend(verify_hook_command_paths(cmd))

        if hook_event_count == 0:
            issues.append(
                HealthIssue(
                    severity="warn",
                    message="No CCR hook entries found under [[hooks.*]]",
                    fix="Run: ccr install-global --agents codex",
                )
            )
        else:
            issues.append(
                HealthIssue(
                    severity="ok",
                    message=f"Codex CCR hooks: {hook_event_count} command(s) registered",
                )
            )
        # Deduplicate path-issues so the same stale path is only reported once.
        seen: set[tuple[str, str]] = set()
        for issue in path_issues:
            key = (issue.severity, issue.message)
            if key in seen:
                continue
            seen.add(key)
            issues.append(issue)

        return issues

    @classmethod
    def _remove_ccr_table(cls, content: str) -> str:
        """Remove the [mcp_servers.ccr] table while preserving all other config."""
        suffix = "\n" if content.strip() else ""
        return cls._CCR_TABLE_RE.sub("", content).rstrip() + suffix

    @classmethod
    def _remove_ccr_hook_groups(cls, content: str) -> str:
        """Remove CCR-owned [[hooks.<event>]] groups from Codex config."""

        def _keep_or_drop(match: re.Match[str]) -> str:
            block = match.group(0)
            return "" if "ccr" in block.lower() else block

        return cls._HOOK_GROUP_RE.sub(_keep_or_drop, content)

    @classmethod
    def _should_manage_hooks_feature(cls, content: str) -> bool:
        if cls._CCR_HOOK_FEATURE_COMMENT in content:
            return True
        value = cls._codex_hooks_feature_value(content)
        return value is None or value.strip().lower() not in ("true", "1")

    @staticmethod
    def _codex_hooks_feature_value(content: str) -> str | None:
        match = re.search(r"(?m)^\s*(?:features\.)?codex_hooks\s*=\s*([^#\n]+)", content)
        return match.group(1).strip() if match else None

    @staticmethod
    def _has_hook_groups(content: str) -> bool:
        return bool(re.search(r"(?m)^\[\[hooks\.", content))

    @classmethod
    def _remove_all_codex_hooks_feature_lines(cls, content: str) -> str:
        """Remove codex_hooks lines plus CCR marker comments."""
        lines = content.splitlines()
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped == cls._CCR_HOOK_FEATURE_COMMENT:
                continue
            if stripped.startswith(cls._CCR_PREVIOUS_HOOK_FEATURE_PREFIX):
                continue
            if re.match(r"^\s*(?:features\.)?codex_hooks\s*=", line):
                continue
            cleaned.append(line)
        return "\n".join(cleaned) + ("\n" if content.endswith("\n") and cleaned else "")

    @classmethod
    def _ensure_hooks_feature(
        cls,
        content: str,
        managed: bool = True,
        previous_value: str | None = None,
    ) -> str:
        """Ensure Codex hooks are enabled without moving root config keys."""
        content = cls._remove_all_codex_hooks_feature_lines(content)
        content = content.lstrip()
        hook_lines: list[str] = []
        if managed:
            hook_lines.append(cls._CCR_HOOK_FEATURE_COMMENT)
            if previous_value is not None:
                hook_lines.append(f"{cls._CCR_PREVIOUS_HOOK_FEATURE_PREFIX}{previous_value}")
        hook_lines.append("codex_hooks = true")
        hook_block = "\n".join(hook_lines)

        table_re = re.compile(r"(?ms)^\[features\]\s*\n.*?(?=^\[|\Z)")
        match = table_re.search(content)
        if match:
            block = match.group(0).rstrip()
            new_block = block + "\n" + hook_block
            return content[: match.start()] + new_block + "\n" + content[match.end() :]

        first_table = re.search(r"(?m)^\[", content)
        if first_table:
            root = content[: first_table.start()].rstrip()
            rest = content[first_table.start():].lstrip()
            parts = []
            if root:
                parts.append(root)
            parts.append("[features]\n" + hook_block)
            if rest:
                parts.append(rest)
            return "\n\n".join(parts)

        content = content.rstrip()
        if content:
            content += "\n\n"
        return content + "[features]\n" + hook_block + "\n"

    @classmethod
    def _remove_managed_hooks_feature(cls, content: str) -> str:
        """Remove or restore codex_hooks only when CCR marked the line as managed."""
        lines = content.splitlines()
        out: list[str] = []
        i = 0
        while i < len(lines):
            if lines[i].strip() != cls._CCR_HOOK_FEATURE_COMMENT:
                out.append(lines[i])
                i += 1
                continue

            i += 1
            previous_value = None
            if i < len(lines) and lines[i].strip().startswith(cls._CCR_PREVIOUS_HOOK_FEATURE_PREFIX):
                previous_value = lines[i].strip()[len(cls._CCR_PREVIOUS_HOOK_FEATURE_PREFIX):]
                i += 1
            if i < len(lines) and re.match(r"^\s*codex_hooks\s*=", lines[i]):
                if previous_value is not None:
                    out.append(f"codex_hooks = {previous_value}")
                i += 1
                continue

            if previous_value is not None:
                out.append(f"{cls._CCR_PREVIOUS_HOOK_FEATURE_PREFIX}{previous_value}")

        return "\n".join(out) + ("\n" if content.endswith("\n") and out else "")

    @staticmethod
    def _ccr_table(python_exe: str) -> str:
        command = json.dumps(python_exe)
        args = json.dumps(["-m", "ccr.mcp_server", "--project", "."])
        return (
            f"[mcp_servers.ccr]\ncommand = {command}\nargs = {args}\n\n"
            "[mcp_servers.ccr.env]\nCCR_STORAGE_BACKEND = \"sqlite\"\n"
        )

    @staticmethod
    def _ccr_hooks_toml(
        python_exe: str,
        hooks_dir: str,
        include_post_tool_use: bool = False,
    ) -> str:
        """Return Codex-native hook groups for automatic CCR behavior."""
        _q = shlex.quote
        py = _q(python_exe)

        def command(script: str) -> str:
            path = _q(os.path.join(hooks_dir, script))
            return f"CCR_AUTO_INIT=1 CCR_HOOK_AGENT=codex CCR_STORAGE_BACKEND=sqlite {py} {path}"

        groups = [
            (
                "SessionStart",
                'matcher = "startup|resume"\n',
                "on_session_start.py",
                30,
                None,
            ),
            (
                "UserPromptSubmit",
                "",
                "on_user_prompt_submit.py",
                30,
                None,
            ),
            (
                "Stop",
                "",
                "codex_stop.py",
                60,
                None,
            ),
        ]
        if include_post_tool_use:
            groups.insert(
                2,
                (
                    "PostToolUse",
                    'matcher = ".*"\n',
                    "on_tool_use.py",
                    30,
                    None,
                ),
            )

        chunks: list[str] = []
        for event, matcher, script, timeout, status in groups:
            status_line = f"statusMessage = {json.dumps(status)}\n" if status else ""
            chunks.append(
                f"[[hooks.{event}]]\n"
                f"{matcher}"
                f"[[hooks.{event}.hooks]]\n"
                'type = "command"\n'
                f"command = {json.dumps(command(script))}\n"
                f"timeout = {timeout}\n"
                f"{status_line}"
            )
        return "\n".join(chunks)
