"""Shared session state accumulator for Claude Code hooks.

Provides atomic read/write of session state that accumulates across
tool use events and gets committed on session end.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

_STATE_FILENAME = ".session_state.json"


@dataclass
class SessionState:
    """Accumulated session state for auto-commit."""

    files_touched: list[str] = field(default_factory=list)
    tasks_completed: list[str] = field(default_factory=list)
    what_accumulated: list[str] = field(default_factory=list)
    patterns_observed: list[str] = field(default_factory=list)
    tool_calls: int = 0
    start_time: float = 0.0
    context_tokens: int = 0   # C1: tokens injected at session start
    session_id: str = ""      # C1: short UUID for sessions.jsonl correlation

    def is_meaningful(self, min_chars: int = 50) -> bool:
        """Check if accumulated state has enough content to warrant a commit."""
        total = sum(len(w) for w in self.what_accumulated)
        return total >= min_chars or len(self.files_touched) >= 3

    def to_commit_fields(self) -> dict[str, str | list[str]]:
        """Convert accumulated state to gcc_commit-compatible fields."""
        what_text = "; ".join(self.what_accumulated) if self.what_accumulated else "Session work"
        files = list(dict.fromkeys(self.files_touched))[:20]  # Dedup, cap at 20
        return {
            "title": f"Auto-commit: {self.what_accumulated[0][:60]}" if self.what_accumulated else "Auto-commit: session work",
            "what": what_text[:500],
            "why": "Auto-committed by session hook",
            "files_changed": files,
            "next_step": "",
            "patterns_learned": self.patterns_observed[:10],
        }


def state_path(ccr_root: str) -> str:
    """Return path to session state file."""
    return os.path.join(ccr_root, _STATE_FILENAME)


def load_state(ccr_root: str) -> SessionState:
    """Load session state from disk. Returns empty state if missing/corrupt."""
    path = state_path(ccr_root)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        return SessionState(
            files_touched=data.get("files_touched", []),
            tasks_completed=data.get("tasks_completed", []),
            what_accumulated=data.get("what_accumulated", []),
            patterns_observed=data.get("patterns_observed", []),
            tool_calls=data.get("tool_calls", 0),
            start_time=data.get("start_time", 0.0),
            context_tokens=data.get("context_tokens", 0),
            session_id=data.get("session_id", ""),
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return SessionState()


def save_state(ccr_root: str, state: SessionState) -> None:
    """Save session state to disk with atomic write."""
    path = state_path(ccr_root)
    tmp = path + ".tmp"
    try:
        data = json.dumps(asdict(state), indent=2, default=str)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        if os.path.isfile(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def clear_state(ccr_root: str) -> None:
    """Remove session state file (called on session start)."""
    path = state_path(ccr_root)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Failed to clear session state: %s", exc)


def append_tool_use(ccr_root: str, tool_name: str, summary: str,
                    files: list[str] | None = None) -> None:
    """Append a tool use event to the session state.

    Thread-safe via atomic read-modify-write (acceptable race window for hooks
    which fire sequentially in Claude Code).
    """
    state = load_state(ccr_root)
    state.tool_calls += 1

    if summary:
        state.what_accumulated.append(summary[:200])

    if files:
        for f in files:
            if f and f not in state.files_touched:
                state.files_touched.append(f)

    save_state(ccr_root, state)


def initialize_state(ccr_root: str) -> None:
    """Initialize a fresh session state with start timestamp and session ID."""
    state = SessionState(
        start_time=time.time(),
        session_id=str(uuid.uuid4())[:8],  # Short 8-char ID for sessions.jsonl
    )
    save_state(ccr_root, state)


def extract_baseline_summary(
    state: SessionState,
    session_turns: list[dict],
) -> dict[str, str | list[str]]:
    """Build a baseline commit dict when no explicit commit was made.

    Uses the first 1-3 user messages as the summary source. Returns a dict
    compatible with mem.commit() kwargs.

    Args:
        state: Already-loaded SessionState (do not reload after clear_state).
        session_turns: Turns from SessionStore.get_session_turns(), may be empty.

    Returns:
        Dict with title, what, why, files_changed, next_step.
    """
    user_msgs = [
        t["user_message"] for t in session_turns[:10]
        if t.get("user_message", "").strip()
    ][:3]

    if user_msgs:
        topic = user_msgs[0][:80].replace("\n", " ").strip()
        title = f"[auto] {topic}" if len(topic) > 5 else "[auto] Session baseline"
        what_parts = [f"Turn {i + 1}: {m[:100]}" for i, m in enumerate(user_msgs)]
    else:
        title = "[auto] Session baseline"
        what_parts = ["Session with no logged turns"]

    files = list(dict.fromkeys(state.files_touched))[:20]
    if files:
        what_parts.append(f"Files touched: {', '.join(files[:5])}")

    return {
        "title": title,
        "what": "; ".join(what_parts)[:500],
        "why": "Session baseline: auto-captured when no explicit commit was made",
        "files_changed": files,
        "next_step": "",
    }
