"""GCC TODO tools — durable project task queue."""

from __future__ import annotations

from mcp.types import ToolAnnotations

from ccr.mcp.server import mcp
import ccr.mcp.server as _srv
from ccr.mcp_types import GccTodoDigestResult, GccTodoResult


def _format_todos(todos: list[dict], empty: str = "No TODOs found.") -> str:
    if not todos:
        return empty
    lines = ["# TODOs"]
    for item in todos:
        extra = []
        if item.get("due_at"):
            extra.append(f"due {item['due_at']}")
        if item.get("blocked_by"):
            extra.append("blocked by " + ", ".join(item["blocked_by"]))
        if item.get("labels"):
            extra.append("labels " + ", ".join(item["labels"]))
        suffix = f" ({'; '.join(extra)})" if extra else ""
        lines.append(
            f"- [{item['id']}] {item.get('status', 'todo')}/"
            f"{item.get('priority', 'normal')}: {item.get('title', '')}{suffix}"
        )
    return "\n".join(lines)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_todos(
    action: str = "list",
    status: str | None = None,
    branch: str | None = None,
    include_done: bool = False,
    limit: int = 20,
) -> GccTodoResult:
    """List durable project TODOs.

    Args:
        action: Currently only "list"; reserved for compatibility with
            action-style task APIs.
        status: Optional comma-separated statuses to include.
        branch: Optional branch filter. Defaults to all branches.
        include_done: Include done/canceled items when no status filter is set.
        limit: Maximum TODOs to return.
    """
    if action != "list":
        raise ValueError("gcc_todos currently supports action='list' only.")
    with _srv._state_lock:
        mem = _srv._ensure_memory()
        todos = mem.todo_list(
            status=status,
            branch=branch,
            include_done=include_done,
            limit=limit,
        )
    return GccTodoResult(
        action=action,
        count=len(todos),
        todos=todos,
        message=_format_todos(todos),
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_todo_add(
    title: str,
    description: str = "",
    priority: str = "normal",
    due_at: str = "",
    labels: list[str] | None = None,
    parent_id: str = "",
    assignee: str = "",
    branch: str | None = None,
    status: str = "todo",
    blocked_by: list[str] | None = None,
) -> GccTodoResult:
    """Create a durable project TODO.

    Priority: urgent, high, normal, low.
    Status: backlog, todo, in_progress, blocked, review, done, canceled.
    """
    with _srv._state_lock:
        mem = _srv._ensure_memory()
        todo = mem.todo_add(
            title=title,
            description=description,
            priority=priority,
            due_at=due_at,
            labels=labels,
            parent_id=parent_id,
            assignee=assignee,
            branch=branch,
            status=status,
            blocked_by=blocked_by,
        )
    return GccTodoResult(
        action="add",
        count=1,
        todos=[todo],
        message=f"Added TODO [{todo['id']}] {todo['title']}",
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
def gcc_todo_update(
    todo_id: str,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    due_at: str | None = None,
    labels: list[str] | None = None,
    parent_id: str | None = None,
    blocked_by: list[str] | None = None,
    assignee: str | None = None,
    percent_complete: int | None = None,
    order_index: int | None = None,
) -> GccTodoResult:
    """Update fields on an existing TODO."""
    updates = {
        "title": title,
        "description": description,
        "status": status,
        "priority": priority,
        "due_at": due_at,
        "labels": labels,
        "parent_id": parent_id,
        "blocked_by": blocked_by,
        "assignee": assignee,
        "percent_complete": percent_complete,
        "order_index": order_index,
    }
    with _srv._state_lock:
        mem = _srv._ensure_memory()
        todo = mem.todo_update(todo_id, updates)
    if todo is None:
        return GccTodoResult(
            action="update",
            count=0,
            todos=[],
            message=f"TODO '{todo_id}' not found.",
        )
    return GccTodoResult(
        action="update",
        count=1,
        todos=[todo],
        message=f"Updated TODO [{todo['id']}] {todo['title']}",
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
def gcc_todo_done(
    todo_id: str,
    note: str = "",
    source_commit: str = "",
) -> GccTodoResult:
    """Mark a TODO as done."""
    with _srv._state_lock:
        mem = _srv._ensure_memory()
        todo = mem.todo_done(todo_id, note=note, source_commit=source_commit)
    if todo is None:
        return GccTodoResult(
            action="done",
            count=0,
            todos=[],
            message=f"TODO '{todo_id}' not found.",
        )
    return GccTodoResult(
        action="done",
        count=1,
        todos=[todo],
        message=f"Completed TODO [{todo['id']}] {todo['title']}",
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
def gcc_todo_delete(todo_id: str, hard: bool = False) -> GccTodoResult:
    """Delete a TODO.

    By default this soft-deletes by moving the TODO to canceled. Set hard=True
    to permanently remove it.
    """
    with _srv._state_lock:
        mem = _srv._ensure_memory()
        ok = mem.todo_delete(todo_id, hard=hard)
    return GccTodoResult(
        action="delete" if hard else "cancel",
        count=1 if ok else 0,
        todos=[],
        message=(
            f"{'Deleted' if hard else 'Canceled'} TODO '{todo_id}'."
            if ok else f"TODO '{todo_id}' not found."
        ),
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_todo_digest(
    limit: int = 5,
    branch: str | None = None,
    include_blocked: bool = True,
) -> GccTodoDigestResult:
    """Return a compact active-TODO digest for agent context."""
    with _srv._state_lock:
        mem = _srv._ensure_memory()
        digest = mem.todo_digest(limit=limit, branch=branch, include_blocked=include_blocked)
        message = mem.format_todo_digest(limit=limit, branch=branch)
    return GccTodoDigestResult(
        count=digest["count"],
        overdue=digest["overdue"],
        blocked=digest["blocked"],
        todos=digest["todos"],
        message=message or "No active TODOs.",
    )
