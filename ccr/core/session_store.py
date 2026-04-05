"""SQLite-backed chat session logger.

Persists every Q&A turn (user message + assistant response) to
``.ccr/sessions.db`` for full replay, debugging, and training-data use.

Follows the SqliteVecStore pattern: thread-local connections, WAL mode,
CREATE TABLE IF NOT EXISTS, graceful FTS5 fallback.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _make_session_id() -> str:
    """Generate a human-readable unique session ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"ses_{ts}_{short}"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: len // 4 (no tokenizer needed)."""
    return max(1, len(text) // 4)


_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    project     TEXT NOT NULL DEFAULT '',
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    turn_count  INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS turns (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL REFERENCES sessions(id),
    turn_number         INTEGER NOT NULL,
    timestamp           TEXT NOT NULL,
    user_message        TEXT NOT NULL DEFAULT '',
    assistant_message   TEXT NOT NULL DEFAULT '',
    tool_calls_json     TEXT NOT NULL DEFAULT '[]',
    files_touched_json  TEXT NOT NULL DEFAULT '[]',
    token_estimate      INTEGER NOT NULL DEFAULT 0,
    source              TEXT NOT NULL DEFAULT 'direct',
    UNIQUE(session_id, turn_number)
);
"""

_CREATE_TURNS_IDX_SESSION = """
CREATE INDEX IF NOT EXISTS idx_turns_session_id ON turns(session_id);
"""

_CREATE_SESSIONS_IDX_STARTED = """
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    user_message,
    assistant_message,
    content='turns',
    content_rowid='id'
);
"""

_CREATE_FTS_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, user_message, assistant_message)
    VALUES (new.id, new.user_message, new.assistant_message);
END;
"""

_CREATE_FTS_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, user_message, assistant_message)
    VALUES ('delete', old.id, old.user_message, old.assistant_message);
END;
"""

_CREATE_FTS_TRIGGER_BU = """
CREATE TRIGGER IF NOT EXISTS turns_bu BEFORE UPDATE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, user_message, assistant_message)
    VALUES ('delete', old.id, old.user_message, old.assistant_message);
END;
"""

_CREATE_FTS_TRIGGER_AU = """
CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
    INSERT INTO turns_fts(rowid, user_message, assistant_message)
    VALUES (new.id, new.user_message, new.assistant_message);
END;
"""


class SessionStore:
    """Persistent Q&A turn store using SQLite.

    Thread-safe via per-thread connections and WAL journal mode.
    FTS5 enabled by default; falls back to LIKE search if unavailable.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._write_lock = threading.Lock()   # serializes turn inserts across threads
        self._fts_available: bool = True
        self._ensure_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self._local.conn = self._create_connection()
        return self._local.conn

    def _create_connection(self) -> sqlite3.Connection:
        """Open a new connection with WAL mode and row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        """Create tables and indexes if they don't exist (idempotent)."""
        with self._init_lock:
            parent = os.path.dirname(self._db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = self._get_conn()
            conn.execute(_CREATE_SESSIONS)
            conn.execute(_CREATE_TURNS)
            conn.execute(_CREATE_TURNS_IDX_SESSION)
            conn.execute(_CREATE_SESSIONS_IDX_STARTED)
            # FTS5 may not be compiled in — catch and disable gracefully
            try:
                conn.execute(_CREATE_FTS)
                conn.execute(_CREATE_FTS_TRIGGER_INSERT)
                conn.execute(_CREATE_FTS_TRIGGER_DELETE)
                conn.execute(_CREATE_FTS_TRIGGER_BU)
                conn.execute(_CREATE_FTS_TRIGGER_AU)
            except sqlite3.OperationalError:
                self._fts_available = False
            conn.commit()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, project: str = "") -> str:
        """Insert a new session row and return its ID."""
        sid = _make_session_id()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO sessions (id, project, started_at) VALUES (?, ?, ?)",
            (sid, project, _utcnow()),
        )
        conn.commit()
        return sid

    def finalize_session(self, session_id: str) -> None:
        """Set ended_at on the session (called by on_stop hook)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (_utcnow(), session_id),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Turn logging
    # ------------------------------------------------------------------

    def log_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        tool_calls: list[dict[str, Any]] | None = None,
        files_touched: list[str] | None = None,
        source: str = "direct",
    ) -> int:
        """Insert a Q&A turn and return its turn_number (1-based).

        Thread-safe: a write lock prevents concurrent turn-number races.

        Args:
            session_id: Session to log the turn under.
            user_message: The user's message text.
            assistant_message: Claude's response text.
            tool_calls: Optional list of tool call summaries.
            files_touched: Optional list of files modified during this turn.
            source: Origin of the turn — ``'direct'`` (logged by session_log_turn)
                or ``'transcript'`` (reconstructed from the Stop-hook transcript).
        """
        with self._write_lock:
            return self._log_turn_locked(
                session_id, user_message, assistant_message, tool_calls, files_touched,
                source=source,
            )

    def _log_turn_locked(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        tool_calls: list[dict[str, Any]] | None,
        files_touched: list[str] | None,
        source: str = "direct",
    ) -> int:
        """Internal: insert turn while holding _write_lock."""
        conn = self._get_conn()

        # Determine next turn number for this session
        row = conn.execute(
            "SELECT turn_count FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            # Session not found — create a transient one
            conn.execute(
                "INSERT INTO sessions (id, project, started_at) VALUES (?, ?, ?)",
                (session_id, "", _utcnow()),
            )
            turn_number = 1
        else:
            turn_number = (row["turn_count"] or 0) + 1

        token_est = _estimate_tokens(user_message) + _estimate_tokens(assistant_message)

        conn.execute(
            """
            INSERT INTO turns
                (session_id, turn_number, timestamp, user_message, assistant_message,
                 tool_calls_json, files_touched_json, token_estimate, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_number,
                _utcnow(),
                user_message,
                assistant_message,
                json.dumps(tool_calls or []),
                json.dumps(files_touched or []),
                token_est,
                source,
            ),
        )
        conn.execute(
            "UPDATE sessions SET turn_count = ? WHERE id = ?",
            (turn_number, session_id),
        )
        conn.commit()
        return turn_number

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_session_turns(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return turns for a session, ordered by turn_number."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT turn_number, timestamp, user_message, assistant_message,
                   tool_calls_json, files_touched_json, token_estimate, source
            FROM turns
            WHERE session_id = ?
            ORDER BY turn_number
            LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent sessions ordered newest-first."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT id, project, started_at, ended_at, turn_count
            FROM sessions
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_turns(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Full-text search across all turns.  Falls back to LIKE when FTS unavailable."""
        conn = self._get_conn()
        if self._fts_available:
            try:
                rows = conn.execute(
                    """
                    SELECT t.session_id, t.turn_number, t.timestamp,
                           snippet(turns_fts, 0, '[', ']', '...', 12) AS snippet_user,
                           snippet(turns_fts, 1, '[', ']', '...', 12) AS snippet_asst
                    FROM turns_fts
                    JOIN turns t ON t.id = turns_fts.rowid
                    WHERE turns_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                self._fts_available = False

        # LIKE fallback
        like = f"%{query}%"
        rows = conn.execute(
            """
            SELECT session_id, turn_number, timestamp,
                   substr(user_message, 1, 100) AS snippet_user,
                   substr(assistant_message, 1, 100) AS snippet_asst
            FROM turns
            WHERE user_message LIKE ? OR assistant_message LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (like, like, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_session(self, session_id: str, fmt: str = "jsonl") -> str:
        """Export a session as JSON, JSONL, or Markdown.

        JSONL uses the OpenAI fine-tuning format:
        ``{"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}``
        """
        conn = self._get_conn()
        session_row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        turns = self.get_session_turns(session_id, limit=10000)

        if fmt == "json":
            data = {
                "session": dict(session_row) if session_row else {"id": session_id},
                "turns": turns,
            }
            return json.dumps(data, indent=2, ensure_ascii=False)

        if fmt == "jsonl":
            lines = []
            for t in turns:
                obj = {
                    "messages": [
                        {"role": "user", "content": t["user_message"]},
                        {"role": "assistant", "content": t["assistant_message"]},
                    ]
                }
                lines.append(json.dumps(obj, ensure_ascii=False))
            return "\n".join(lines)

        if fmt == "markdown":
            parts = [f"# Session {session_id}\n"]
            if session_row:
                parts.append(f"**Started:** {session_row['started_at']}")
                if session_row["ended_at"]:
                    parts.append(f"  **Ended:** {session_row['ended_at']}")
                parts.append(f"  **Turns:** {session_row['turn_count']}\n")
            for t in turns:
                parts.append(f"## Turn {t['turn_number']} — {t['timestamp']}\n")
                parts.append(f"**User:**\n{t['user_message']}\n")
                parts.append(f"**Assistant:**\n{t['assistant_message']}\n")
            return "\n".join(parts)

        raise ValueError(f"Unknown export format: {fmt!r}. Use 'json', 'jsonl', or 'markdown'.")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the thread-local connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
