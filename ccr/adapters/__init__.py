"""CCR Agent Adapter Layer — pluggable integrations for any AI agent.

Each adapter encapsulates agent-specific configuration paths, hook formats,
and context injection strategies behind a unified interface.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from abc import ABC, abstractmethod
from typing import ClassVar


@dataclasses.dataclass
class InstallResult:
    """Result of an adapter install() call."""

    success: bool
    message: str
    files_created: list[str] = dataclasses.field(default_factory=list)
    files_modified: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class UninstallResult:
    """Result of an adapter uninstall() call."""

    success: bool
    message: str
    files_removed: list[str] = dataclasses.field(default_factory=list)
    files_modified: list[str] = dataclasses.field(default_factory=list)


class IntegrationLevel:
    """Capability levels an agent can support."""

    NONE = 0
    MANUAL = 1          # User copy-pastes context
    SDK_WRAPPER = 2     # Python decorator / CLI prefix
    FILE_WATCHER = 3    # File-based context injection
    MCP_ONLY = 4        # MCP tools, manual context load
    MCP_AND_HOOKS = 5   # Full auto-context + tools


@dataclasses.dataclass
class AgentStatus:
    """Runtime status of an adapter."""

    name: str
    display_name: str
    installed: bool
    ccr_enabled: bool
    integration_level: int
    mcp_configured: bool
    hooks_configured: bool


class BaseAgentAdapter(ABC):
    """Abstract base for all agent integrations.

    Implementations live in ccr.adapters.<agent_name> and are auto-registered
    via the ADAPTERS list at the bottom of this module.
    """

    name: ClassVar[str] = ""           # e.g. "claude-code"
    display_name: ClassVar[str] = ""   # e.g. "Claude Code"

    @abstractmethod
    def is_installed(self) -> bool:
        """Detect whether this agent is installed on the system."""

    @abstractmethod
    def supports_mcp(self) -> bool:
        """Does this agent support the Model Context Protocol?"""

    @abstractmethod
    def supports_hooks(self) -> bool:
        """Does this agent support lifecycle hooks?"""

    def integration_level(self) -> int:
        """Highest integration level this agent supports."""
        if self.supports_mcp() and self.supports_hooks():
            return IntegrationLevel.MCP_AND_HOOKS
        if self.supports_mcp():
            return IntegrationLevel.MCP_ONLY
        return IntegrationLevel.FILE_WATCHER

    @abstractmethod
    def install(self, python_exe: str, hooks_dir: str, ccr_pkg: str) -> InstallResult:
        """Set up CCR integration for this agent."""

    @abstractmethod
    def uninstall(self) -> UninstallResult:
        """Remove CCR integration for this agent."""

    def get_mcp_config(self, python_exe: str) -> dict:
        """Return the MCP server config dict for this agent.

        Most agents use the same stdio MCP config; override if different.
        """
        return {
            "command": python_exe,
            "args": ["-m", "ccr.mcp_server", "--project", "."],
        }

    def context_format(self) -> str:
        """Return preferred context injection format.

        One of: "xml", "markdown", "plain", "frontmatter"
        """
        return "xml"

    def wrap_context(self, context: str, tag: str = "gcc_context") -> str:
        """Wrap context in the adapter's preferred format."""
        fmt = self.context_format()
        if fmt == "xml":
            return f"<{tag}>\n{context}\n</{tag}>"
        if fmt == "markdown":
            return f"## CCR Context\n\n{context}"
        if fmt == "frontmatter":
            return f"---\nsource: ccr\n---\n\n{context}"
        return context

    def status(self) -> AgentStatus:
        """Return current status of this adapter."""
        return AgentStatus(
            name=self.name,
            display_name=self.display_name,
            installed=self.is_installed(),
            ccr_enabled=self._is_ccr_enabled(),
            integration_level=self.integration_level(),
            mcp_configured=self._is_mcp_configured(),
            hooks_configured=self._is_hooks_configured(),
        )

    # ------------------------------------------------------------------
    # Internal helpers subclasses may override
    # ------------------------------------------------------------------

    def _is_ccr_enabled(self) -> bool:
        """Check if CCR is currently configured for this agent."""
        return self._is_mcp_configured() or self._is_hooks_configured()

    def _is_mcp_configured(self) -> bool:
        """Override if the agent stores MCP config differently."""
        return False

    def _is_hooks_configured(self) -> bool:
        """Override if the agent stores hooks differently."""
        return False

    def _shlex_quote(self, path: str) -> str:
        import shlex

        return shlex.quote(path)


# ---------------------------------------------------------------------------
# Lazy adapter imports — avoid circular deps and heavy imports at module level
# ---------------------------------------------------------------------------

_ALL_ADAPTER_CLASSES: list[type[BaseAgentAdapter]] = []


def _load_adapter_classes() -> list[type[BaseAgentAdapter]]:
    """Dynamically import all adapter subclasses."""
    global _ALL_ADAPTER_CLASSES
    if _ALL_ADAPTER_CLASSES:
        return _ALL_ADAPTER_CLASSES

    # Import each adapter module to trigger class registration
    from ccr.adapters import claude_code, continue_dev, generic_mcp, kimi, ollama, openai

    # Collect all concrete BaseAgentAdapter subclasses
    import inspect

    for mod in (claude_code, continue_dev, generic_mcp, kimi, ollama, openai):
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, BaseAgentAdapter)
                and obj is not BaseAgentAdapter
                and getattr(obj, "name", "")
            ):
                _ALL_ADAPTER_CLASSES.append(obj)

    return _ALL_ADAPTER_CLASSES


def get_adapters() -> list[BaseAgentAdapter]:
    """Return instantiated adapters for all registered agent types."""
    return [cls() for cls in _load_adapter_classes()]


def get_adapter(name: str) -> BaseAgentAdapter | None:
    """Get a specific adapter by name."""
    for cls in _load_adapter_classes():
        if cls.name == name:
            return cls()
    return None


def detect_installed() -> list[BaseAgentAdapter]:
    """Return adapters for agents currently installed on the system."""
    return [a for a in get_adapters() if a.is_installed()]


def list_adapter_names() -> list[str]:
    """Return all registered adapter names."""
    return [cls.name for cls in _load_adapter_classes()]


def list_display_names() -> list[tuple[str, str]]:
    """Return (name, display_name) for all registered adapters."""
    return [(cls.name, cls.display_name) for cls in _load_adapter_classes()]
