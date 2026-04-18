"""Phase 1 SQLite mixin: scratchpad, metrics, log, metadata."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from ccr.core.storage._sqlite_utils import _utcnow


class Phase1Mixin:
    """Scratchpad, metrics, log, and metadata methods.

    Requires self.memory_conn from SqliteStorageBackend.
    """

    # ── Scratchpad ──────────────────────────────────────────────

    def scratchpad_set(
        self, key: str, value: str, ttl_seconds: int | None = None,
    ) -> dict:
        conn = self.memory_conn
        now = _utcnow()
        expires_at: str | None = None
        if ttl_seconds is not None:
            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            ).isoformat()

        existing = conn.execute(
            "SELECT created_at, access_count FROM scratchpad WHERE key = ?",
            (key,),
        ).fetchone()

        created_at = existing["created_at"] if existing else now
        access_count = existing["access_count"] if existing else 0

        conn.execute(
            """INSERT OR REPLACE INTO scratchpad
               (key, value, created_at, updated_at, access_count, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key, value, created_at, now, access_count, expires_at),
        )
        conn.commit()
        return {
            "key": key, "value": value, "created_at": created_at,
            "updated_at": now, "access_count": access_count,
            "expires_at": expires_at,
        }

    def scratchpad_get(self, key: str) -> dict | None:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT * FROM scratchpad WHERE key = ?", (key,),
        ).fetchone()
        if row is None:
            return None

        if row["expires_at"]:
            if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
                conn.execute("DELETE FROM scratchpad WHERE key = ?", (key,))
                conn.commit()
                return None

        conn.execute(
            "UPDATE scratchpad SET access_count = access_count + 1 WHERE key = ?",
            (key,),
        )
        conn.commit()
        new_count = row["access_count"] + 1
        return {
            "key": row["key"], "value": row["value"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "access_count": new_count, "expires_at": row["expires_at"],
        }

    def scratchpad_list(self) -> list[dict]:
        conn = self.memory_conn
        now = datetime.now(timezone.utc)
        rows = conn.execute("SELECT * FROM scratchpad").fetchall()
        result = []
        expired_keys = []
        for r in rows:
            if r["expires_at"]:
                if datetime.fromisoformat(r["expires_at"]) < now:
                    expired_keys.append(r["key"])
                    continue
            result.append({
                "key": r["key"], "value": r["value"],
                "created_at": r["created_at"], "updated_at": r["updated_at"],
                "access_count": r["access_count"], "expires_at": r["expires_at"],
            })
        if expired_keys:
            placeholders = ",".join("?" * len(expired_keys))
            conn.execute(
                f"DELETE FROM scratchpad WHERE key IN ({placeholders})",
                expired_keys,
            )
            conn.commit()
        return result

    def scratchpad_delete(self, key: str) -> bool:
        conn = self.memory_conn
        cursor = conn.execute("DELETE FROM scratchpad WHERE key = ?", (key,))
        conn.commit()
        return cursor.rowcount > 0

    def scratchpad_clear(self) -> int:
        conn = self.memory_conn
        count = conn.execute("SELECT COUNT(*) FROM scratchpad").fetchone()[0]
        conn.execute("DELETE FROM scratchpad")
        conn.commit()
        return count

    def scratchpad_search(self, query: str, top_k: int = 5) -> list[dict]:
        entries = self.scratchpad_list()
        query_lower = query.lower()
        matched = [
            e for e in entries
            if query_lower in f"{e['key']} {e['value']}".lower()
        ]
        return matched[:top_k]

    # ── Metrics ─────────────────────────────────────────────────

    def metrics_increment(self, key: str, amount: int = 1) -> None:
        conn = self.memory_conn
        now = _utcnow()
        conn.execute(
            """INSERT INTO metrics (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
               value = value + excluded.value, updated_at = excluded.updated_at""",
            (key, amount, now),
        )
        conn.commit()

    def metrics_get(self) -> dict[str, Any]:
        conn = self.memory_conn
        rows = conn.execute("SELECT key, value FROM metrics").fetchall()
        result: dict[str, Any] = {
            "total_commits": 0, "search_calls": 0, "link_creations": 0,
        }
        for r in rows:
            result[r["key"]] = r["value"]
        return result

    # ── Log ─────────────────────────────────────────────────────

    def log_append(self, branch: str, line: str, max_lines: int = 500) -> None:
        conn = self.memory_conn
        conn.execute(
            "INSERT INTO log_entries (branch, line, created_at) VALUES (?, ?, ?)",
            (branch, line, _utcnow()),
        )
        conn.execute(
            """DELETE FROM log_entries WHERE branch = ? AND id NOT IN (
                SELECT id FROM log_entries WHERE branch = ?
                ORDER BY id DESC LIMIT ?
            )""",
            (branch, branch, max_lines),
        )
        conn.commit()

    def log_read(self, branch: str, count: int = 50) -> str:
        conn = self.memory_conn
        rows = conn.execute(
            """SELECT line FROM log_entries WHERE branch = ?
               ORDER BY id DESC LIMIT ?""",
            (branch, count),
        ).fetchall()
        lines = [r["line"] for r in reversed(rows)]
        return "\n".join(lines)

    # ── Metadata ────────────────────────────────────────────────

    def metadata_load(self) -> dict:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT value_json FROM metadata WHERE key = '_root'",
        ).fetchone()
        if row is None:
            return {}
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return {}

    def metadata_save(self, data: dict) -> None:
        conn = self.memory_conn
        conn.execute(
            """INSERT OR REPLACE INTO metadata (key, value_json, updated_at)
               VALUES ('_root', ?, ?)""",
            (json.dumps(data, default=str), _utcnow()),
        )
        conn.commit()
