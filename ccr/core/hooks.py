"""Hook system for CCR — fires callbacks at key lifecycle events.

Based on GCC paper's hook system (SessionStart, PostToolUse, PreCompact, Stop).
Unlike open-gcc which spawns Node.js processes, CCR hooks are in-process Python callables.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ccr.core.types import HookEvent

logger = logging.getLogger(__name__)


class HookManager:
    """Manages lifecycle hooks for CCR sessions.

    Hooks fire at:
        SessionStart  — engine initialization (inject context, tool refs)
        PostToolUse   — after a tool operation is detected in response
        PreCompact    — before context compaction (nudge to commit)
        Stop          — engine shutdown (audit trail)
    """

    def __init__(self):
        self._handlers: dict[HookEvent, list[Callable]] = defaultdict(list)

    def register(self, event: HookEvent, handler: Callable[..., Any]) -> None:
        """Register a handler for a lifecycle event."""
        self._handlers[event].append(handler)

    def fire(self, event: HookEvent, context: dict[str, Any] | None = None) -> list[Any]:
        """Fire all handlers for an event. Returns list of handler results."""
        context = context or {}
        results = []
        for handler in self._handlers[event]:
            try:
                result = handler(context)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook {event.value} handler failed: {e}")
                results.append(None)
        return results

    def has_handlers(self, event: HookEvent) -> bool:
        return len(self._handlers[event]) > 0


def create_default_hooks(memory_manager: Any) -> HookManager:
    """Create a HookManager with the standard GCC-paper hooks pre-registered."""
    hooks = HookManager()

    def on_session_start(ctx: dict) -> str:
        """Inject context and log session start."""
        memory_manager.log_ota(
            tool_name="session-start",
            observation="New CCR session initialized",
            thought="Loading project context for grounding",
            action="Injected level-1 context into system prompt",
        )
        return memory_manager.get_session_context()

    def on_post_tool_use(ctx: dict) -> None:
        """Log tool usage to OTA trace."""
        tool = ctx.get("tool_name", "unknown")
        file_path = ctx.get("file_path", "")
        status = ctx.get("status", "OK")
        observation = ctx.get("observation", f"Tool {tool} executed")
        thought = ctx.get("thought", "")
        action = ctx.get("action", f"{tool} on {file_path}")
        memory_manager.log_ota(
            tool_name=tool,
            file_path=file_path,
            status=status,
            observation=observation,
            thought=thought,
            action=action,
        )

    def on_pre_compact(ctx: dict) -> str:
        """Nudge to commit before context compaction."""
        memory_manager.log_ota(
            tool_name="pre-compact",
            observation="Context approaching compaction threshold",
            thought="Should commit progress before compaction to preserve state",
            action="Triggering auto-commit of current progress",
        )
        return "REMINDER: Commit your progress before context is compacted."

    def on_post_task_complete(ctx: dict) -> None:
        """Log ACE adaptation results after task completion."""
        task = ctx.get("task", "unknown")
        added = ctx.get("bullets_added", 0)
        pruned = ctx.get("bullets_pruned", 0)
        size = ctx.get("playbook_size", 0)
        memory_manager.log_ota(
            tool_name="ace-adaptation",
            observation=f"ACE adapted after task: {task[:80]}",
            thought=f"Playbook: +{added} added, -{pruned} pruned, size={size}",
            action="Playbook updated and saved",
        )

    def on_stop(ctx: dict) -> None:
        """Log session end for audit trail."""
        memory_manager.log_ota(
            tool_name="session-end",
            observation="CCR session shutting down",
            thought="Persisting final state",
            action="Session ended cleanly",
        )

    hooks.register(HookEvent.SESSION_START, on_session_start)
    hooks.register(HookEvent.POST_TOOL_USE, on_post_tool_use)
    hooks.register(HookEvent.PRE_COMPACT, on_pre_compact)
    hooks.register(HookEvent.POST_TASK_COMPLETE, on_post_task_complete)
    hooks.register(HookEvent.STOP, on_stop)

    return hooks
