"""Generic MCP adapter — for any agent that supports MCP but has no specific adapter.

Generates a standalone MCP config file that users can copy into their agent's config.
"""

from __future__ import annotations

import json
import os

from ccr.adapters import BaseAgentAdapter, InstallResult, UninstallResult


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
