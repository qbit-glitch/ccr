"""MCP tools for the Session Logger.

Four tools:
  - session_log_turn      — log a Q&A turn after each Claude response
  - session_get_history   — retrieve recent turns for a session
  - session_search        — full-text search across all sessions
  - session_export        — export a session as JSON/JSONL/Markdown
"""

from __future__ import annotations

import os

import ccr.mcp.server as _srv
from ccr.mcp.server import mcp
from ccr.mcp_types import (
    SessionExportResult,
    SessionGetHistoryResult,
    SessionLogTurnResult,
    SessionSearchResult,
)

try:
    from mcp.types import ToolAnnotations
except ImportError:
    ToolAnnotations = None  # type: ignore[assignment,misc]


def _annotations(**kwargs):
    if ToolAnnotations is not None:
        return ToolAnnotations(**kwargs)
    return None


def _get_store():
    """Return the active SessionStore via the server helper."""
    return _srv._ensure_session_store()


def _active_session_id() -> str:
    """Return the cached current session ID (may be empty if no session started)."""
    return _srv._current_session_id


def _read_pending_user_msg(ccr_root: str) -> str:
    """Read and delete the buffered user message written by the hook."""
    path = os.path.join(ccr_root, ".pending_user_msg")
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            os.unlink(path)
            return content.strip()
    except OSError:
        pass
    return ""


# ---------------------------------------------------------------------------
# session_log_turn
# ---------------------------------------------------------------------------

_log_turn_ann = _annotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


@mcp.tool(annotations=_log_turn_ann)
def session_log_turn(
    assistant_message: str,
    user_message: str = "",
    tool_calls: list[dict] | None = None,
    files_touched: list[str] | None = None,
) -> SessionLogTurnResult:
    """Log a complete Q&A turn to the session database.

    Call this after EVERY response you give to the user.

    Args:
        assistant_message: Your full response text (required).
        user_message: The user's message. If empty, read automatically
            from the hook-buffered ``.ccr/.pending_user_msg`` file.
        tool_calls: Optional list of tool call summaries for this turn.
        files_touched: Optional list of files modified during this turn.
    """
    try:
        store = _get_store()
        session_id = _active_session_id()

        # Auto-read user message from hook buffer if not provided
        if not user_message and _srv._project_root:
            ccr_root = os.path.join(_srv._project_root, ".ccr")
            user_message = _read_pending_user_msg(ccr_root)

        # Ensure we have an active session
        if not session_id:
            # Create a transient session so we never silently drop data
            session_id = store.create_session(project=_srv._project_root or "")
            _srv._current_session_id = session_id

        turn_number = store.log_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            files_touched=files_touched,
        )
        return SessionLogTurnResult(
            session_id=session_id,
            turn_number=turn_number,
            message=f"Logged turn {turn_number} in session {session_id}",
        )
    except Exception as exc:
        return SessionLogTurnResult(
            session_id="",
            turn_number=0,
            message=f"session_log_turn failed (non-fatal): {exc}",
        )


# ---------------------------------------------------------------------------
# session_get_history
# ---------------------------------------------------------------------------

_get_history_ann = _annotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)


@mcp.tool(annotations=_get_history_ann)
def session_get_history(
    session_id: str = "",
    limit: int = 20,
    offset: int = 0,
) -> SessionGetHistoryResult:
    """Retrieve Q&A turns for a session.

    Args:
        session_id: Session to retrieve. Defaults to the current session.
        limit: Maximum number of turns to return (default 20).
        offset: Number of turns to skip (for pagination).
    """
    store = _get_store()
    sid = session_id or _active_session_id()
    if not sid:
        return SessionGetHistoryResult(
            session_id="",
            turn_count=0,
            turns=[],
            message="No active session. Provide a session_id or start a new session.",
        )

    turns = store.get_session_turns(sid, limit=limit, offset=offset)
    lines = []
    for t in turns:
        source = t.get("source", "direct")
        source_label = " [from transcript]" if source == "transcript" else ""
        lines.append(
            f"[Turn {t['turn_number']} — {t['timestamp']}{source_label}]\n"
            f"User: {t['user_message'][:200]}{'...' if len(t['user_message']) > 200 else ''}\n"
            f"Assistant: {t['assistant_message'][:200]}{'...' if len(t['assistant_message']) > 200 else ''}"
        )
    summary = "\n\n".join(lines) if lines else "(no turns recorded)"

    return SessionGetHistoryResult(
        session_id=sid,
        turn_count=len(turns),
        turns=turns,
        message=f"Session {sid}: {len(turns)} turn(s) (offset={offset})\n\n{summary}",
    )


# ---------------------------------------------------------------------------
# session_search
# ---------------------------------------------------------------------------

_search_ann = _annotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)


@mcp.tool(annotations=_search_ann)
def session_search(
    query: str,
    limit: int = 10,
) -> SessionSearchResult:
    """Full-text search across all session turns.

    Uses FTS5 when available; falls back to LIKE-based search.

    Args:
        query: Search terms.
        limit: Maximum number of results (default 10).
    """
    store = _get_store()
    results = store.search_turns(query, limit=limit)

    lines = []
    for r in results:
        snippet_u = r.get("snippet_user", "")
        snippet_a = r.get("snippet_asst", "")
        lines.append(
            f"[{r['session_id']} / Turn {r['turn_number']} — {r['timestamp']}]\n"
            f"  User: {snippet_u}\n"
            f"  Asst: {snippet_a}"
        )
    summary = "\n\n".join(lines) if lines else f"No results for: {query!r}"

    return SessionSearchResult(
        result_count=len(results),
        results=results,
        message=f"Found {len(results)} turn(s) matching {query!r}\n\n{summary}",
    )


# ---------------------------------------------------------------------------
# session_export
# ---------------------------------------------------------------------------

_export_ann = _annotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)


@mcp.tool(annotations=_export_ann)
def session_export(
    session_id: str = "",
    format: str = "jsonl",
) -> SessionExportResult:
    """Export a session for replay, debugging, or training data.

    Args:
        session_id: Session to export. Defaults to the current session.
        format: ``"json"`` (full object), ``"jsonl"`` (OpenAI fine-tune,
            one message-pair per line), or ``"markdown"`` (human-readable).
    """
    store = _get_store()
    sid = session_id or _active_session_id()
    if not sid:
        return SessionExportResult(
            session_id="",
            format=format,
            data="",
            message="No active session. Provide a session_id.",
        )

    try:
        data = store.export_session(sid, fmt=format)
    except ValueError as exc:
        return SessionExportResult(
            session_id=sid,
            format=format,
            data="",
            message=str(exc),
        )

    line_count = data.count("\n") + 1 if data.strip() else 0
    return SessionExportResult(
        session_id=sid,
        format=format,
        data=data,
        message=f"Exported session {sid} as {format!r} ({line_count} lines)",
    )
