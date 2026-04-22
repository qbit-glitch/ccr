"""Continue adapter — MCP-only integration via ~/.continue/config.json.

Continue is a VS Code / JetBrains extension (continue.dev).
Supports MCP servers via its config.json.
No hooks — users must call gcc_context() manually in their prompts.
"""

from __future__ import annotations

import json
import os

from ccr.adapters import BaseAgentAdapter, InstallResult, UninstallResult


class ContinueDevAdapter(BaseAgentAdapter):
    """Continue IDE extension (VS Code, JetBrains, etc.).

    Continue supports MCP servers via its config.json.
    No hooks — users must call gcc_context() manually in their prompts.
    """

    name = "continue-dev"
    display_name = "Continue (VS Code extension)"

    _CONFIG_DIR = os.path.expanduser("~/.continue")
    _CONFIG_PATH = os.path.join(_CONFIG_DIR, "config.json")

    # VS Code extension paths to check for installation
    _VSCODE_EXTENSION_PATHS = [
        os.path.expanduser("~/.vscode/extensions/continue.continue"),
        os.path.expanduser("~/.vscode-server/extensions/continue.continue"),
        os.path.expanduser("~/Library/Application Support/Code/User/globalStorage/continue.continue"),
    ]

    def is_installed(self) -> bool:
        # Check for config directory, config file, or VS Code extension
        if os.path.isdir(self._CONFIG_DIR) or os.path.isfile(self._CONFIG_PATH):
            return True
        for path in self._VSCODE_EXTENSION_PATHS:
            if os.path.exists(path):
                return True
        return False

    def supports_mcp(self) -> bool:
        return True

    def supports_hooks(self) -> bool:
        return False

    def install(self, python_exe: str, hooks_dir: str, ccr_pkg: str) -> InstallResult:
        os.makedirs(self._CONFIG_DIR, exist_ok=True)

        config: dict = {}
        if os.path.isfile(self._CONFIG_PATH):
            try:
                with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                    config = json.loads(f.read())
            except (json.JSONDecodeError, OSError):
                config = {}

        # Continue.dev MCP servers go under "models" -> "mcpServers" or top-level "mcpServers"
        # As of 2025, Continue uses top-level "mcpServers" dict
        config.setdefault("mcpServers", {})
        config["mcpServers"]["ccr"] = self.get_mcp_config(python_exe)

        with open(self._CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        return InstallResult(
            success=True,
            message=f"Continue.dev MCP config written to {self._CONFIG_PATH}",
            files_created=[self._CONFIG_PATH] if not os.path.exists(self._CONFIG_PATH + ".backup") else [],
            files_modified=[self._CONFIG_PATH],
        )

    def uninstall(self) -> UninstallResult:
        if not os.path.isfile(self._CONFIG_PATH):
            return UninstallResult(success=True, message="No Continue.dev config found")

        try:
            with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.loads(f.read())
            config.get("mcpServers", {}).pop("ccr", None)
            with open(self._CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
                f.write("\n")
            return UninstallResult(
                success=True,
                message="Removed CCR from Continue.dev config",
                files_modified=[self._CONFIG_PATH],
            )
        except Exception as exc:
            return UninstallResult(success=False, message=str(exc))

    def _is_mcp_configured(self) -> bool:
        if not os.path.isfile(self._CONFIG_PATH):
            return False
        try:
            with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.loads(f.read())
            return "ccr" in config.get("mcpServers", {})
        except Exception:
            return False

    def context_format(self) -> str:
        return "markdown"
