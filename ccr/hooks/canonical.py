"""Canonical hook event handling — provider-agnostic payload parsing and formatting.

Hook scripts import from this module to:
  1. Parse the standardized JSON event payload from stdin
  2. Format output according to the agent's preferred ContextFormat
  3. Read/write session state in a consistent way
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from ccr.core.types import CanonicalEvent, ContextFormat


class HookPayload:
    """Standardized event payload sent to hook scripts by agent adapters."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.event = CanonicalEvent(raw.get("event", "session_start"))
        self.agent = raw.get("agent", "unknown")
        self.project_root = raw.get("project_root", os.getcwd())
        self.prompt = raw.get("prompt", "")
        self.session_id = raw.get("session_id", "")
        self.timestamp = raw.get("timestamp", "")
        self.format = ContextFormat(raw.get("format", "xml"))

    @classmethod
    def from_stdin(cls) -> "HookPayload":
        """Read and parse the canonical payload from stdin.

        Falls back to a minimal payload if stdin is empty or unreadable.
        """
        try:
            if not sys.stdin.isatty():
                data = sys.stdin.read()
                if data.strip():
                    parsed = json.loads(data)
                    # If the agent sent its native format (e.g. Claude's {"prompt":"..."}),
                    # wrap it into canonical form
                    if "event" not in parsed:
                        return cls._from_native(parsed)
                    return cls(parsed)
        except Exception:
            pass
        return cls({})

    @classmethod
    def _from_native(cls, native: dict[str, Any]) -> "HookPayload":
        """Convert an agent-native payload to canonical form."""
        # Claude Code native: {"prompt": "..."}
        # Kimi native: similar JSON
        canonical = {
            "event": "session_start",
            "prompt": native.get("prompt", ""),
            "agent": "unknown",
            "format": "xml",
        }
        return cls(canonical)


def format_context_block(content: str, tag: str = "gcc_context", fmt: ContextFormat = ContextFormat.XML) -> str:
    """Wrap context content in the requested format."""
    if fmt == ContextFormat.XML:
        return f"<{tag}>\n{content}\n</{tag}>"
    if fmt == ContextFormat.MARKDOWN:
        return f"## CCR Context\n\n{content}"
    if fmt == ContextFormat.FRONTMATTER:
        return f"---\nsource: ccr\n---\n\n{content}"
    return content


def format_playbook_block(global_text: str, project_text: str, fmt: ContextFormat = ContextFormat.XML) -> str:
    """Format playbook content in the requested format."""
    parts = []
    if global_text:
        if fmt == ContextFormat.XML:
            parts.append(f"# GLOBAL STRATEGIES (all projects)\n{global_text}")
        else:
            parts.append(f"## Global Strategies\n\n{global_text}")
    if project_text:
        if fmt == ContextFormat.XML:
            parts.append(f"# PROJECT STRATEGIES (this project)\n{project_text}")
        else:
            parts.append(f"## Project Strategies\n\n{project_text}")

    if not parts:
        return ""

    combined = "\n\n".join(parts)
    if fmt == ContextFormat.XML:
        return f"<ace_playbook>\n{combined}\n</ace_playbook>"
    return f"## ACE Playbook\n\n{combined}"


def format_directive(directive_text: str, fmt: ContextFormat = ContextFormat.XML) -> str:
    """Format a directive/reminder block."""
    if fmt == ContextFormat.XML:
        return f"<MANDATORY_CCR_ACTIONS>\n{directive_text}\n</MANDATORY_CCR_ACTIONS>"
    return f"**CCR Actions Required**\n\n{directive_text}"


def format_reminder(text: str, fmt: ContextFormat = ContextFormat.XML) -> str:
    """Format a lightweight reminder."""
    if fmt == ContextFormat.XML:
        return f"<ccr_reminder>\n{text}\n</ccr_reminder>"
    return f"\n> **CCR Reminder:** {text}\n"


def format_ready_message(text: str, fmt: ContextFormat = ContextFormat.XML) -> str:
    """Format the first-use 'CCR is active' message."""
    if fmt == ContextFormat.XML:
        return f"<ccr_ready>\n{text}\n</ccr_ready>"
    return f"\n**CCR Active** — {text}\n"
