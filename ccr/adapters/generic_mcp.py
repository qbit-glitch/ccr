"""Generic MCP adapter — for any agent that supports MCP but has no specific adapter.

Generates a standalone MCP config file that users can copy into their agent's config.
"""

from __future__ import annotations

import json
import os

from ccr.adapters import BaseAgentAdapter, HealthIssue, InstallResult, UninstallResult


class GenericMcpAdapter(BaseAgentAdapter):
    """Fallback adapter for any MCP-capable agent.

    Writes a generic MCP config to ~/.ccr/mcp-config.json that the user
    can manually import into their agent's configuration.
    """

    name = "generic-mcp"
    display_name = "Generic MCP (manual import)"

    _CONFIG_PATH = os.path.expanduser("~/.ccr/mcp-config.json")

    def is_installed(self) -> bool:
        """Always considered 'available' since it's a manual fallback."""
        return False  # Don't auto-detect; user must explicitly request

    def supports_mcp(self) -> bool:
        return True

    def supports_hooks(self) -> bool:
        return False

    def install(self, python_exe: str, hooks_dir: str, ccr_pkg: str) -> InstallResult:
        os.makedirs(os.path.dirname(self._CONFIG_PATH), exist_ok=True)

        config = {
            "mcpServers": {
                "ccr": self.get_mcp_config(python_exe),
            },
            "_ccr_note": (
                "Copy the 'mcpServers.ccr' block into your agent's MCP configuration. "
                "Refer to your agent's documentation for where to place MCP server configs."
            ),
        }

        with open(self._CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        return InstallResult(
            success=True,
            message=f"Generic MCP config written to {self._CONFIG_PATH}",
            files_created=[self._CONFIG_PATH],
        )

    def uninstall(self) -> UninstallResult:
        if os.path.isfile(self._CONFIG_PATH):
            os.remove(self._CONFIG_PATH)
            return UninstallResult(
                success=True,
                message=f"Removed {self._CONFIG_PATH}",
                files_removed=[self._CONFIG_PATH],
            )
        return UninstallResult(success=True, message="No generic MCP config found")

    def context_format(self) -> str:
        return "markdown"

    def health_check(self) -> list[HealthIssue]:
        """Generic MCP fallback — verifies ~/.ccr/mcp-config.json is valid.

        ``is_installed`` always returns False so this adapter is invisible
        to ``ccr agents doctor`` unless the user explicitly targets it.
        """
        issues: list[HealthIssue] = []
        if not os.path.isfile(self._CONFIG_PATH):
            issues.append(
                HealthIssue(
                    severity="info",
                    message=f"Generic MCP config not generated: {self._CONFIG_PATH}",
                    fix="Run: ccr install-global --agents generic-mcp (then copy the JSON into your agent)",
                )
            )
            return issues

        try:
            with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.loads(f.read())
        except json.JSONDecodeError:
            return [
                HealthIssue(
                    severity="error",
                    message=f"Generic MCP config is not valid JSON: {self._CONFIG_PATH}",
                    fix="Remove the file and re-run 'ccr install-global --agents generic-mcp'",
                )
            ]
        except OSError as exc:
            return [HealthIssue(severity="error", message=f"Generic MCP config unreadable: {exc}")]

        ccr_entry = (config.get("mcpServers") or {}).get("ccr")
        if not isinstance(ccr_entry, dict):
            issues.append(
                HealthIssue(
                    severity="warn",
                    message="Generic MCP config missing ccr server entry",
                    fix="Run: ccr install-global --agents generic-mcp",
                )
            )
            return issues

        cmd = ccr_entry.get("command", "")
        if cmd and not os.path.isfile(cmd):
            issues.append(
                HealthIssue(
                    severity="error",
                    message=f"Generic MCP python missing: {cmd}",
                    fix="Re-run 'ccr install-global --agents generic-mcp' after fixing the venv",
                )
            )
        env = ccr_entry.get("env") or {}
        if isinstance(env, dict) and env.get("CCR_STORAGE_BACKEND") == "sqlite":
            issues.append(
                HealthIssue(
                    severity="ok",
                    message="Generic MCP config: ccr (sqlite backend)",
                )
            )
        else:
            issues.append(
                HealthIssue(
                    severity="warn",
                    message="Generic MCP server not pinned to CCR_STORAGE_BACKEND=sqlite",
                    fix="Run: ccr install-global --agents generic-mcp",
                )
            )
        return issues
