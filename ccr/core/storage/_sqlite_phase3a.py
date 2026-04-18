"""Phase 3a SQLite mixin: commits, rolling summaries, branches."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from ccr.core.storage._sqlite_utils import (
    _BRANCH_COLUMNS,
    _COMMIT_COLUMNS,
    _escape_like,
    _utcnow,
)


class Phase3aMixin:
    """Commit, rolling summary, and branch methods.

    Requires self.memory_conn from SqliteStorageBackend.
    """

    # ── Commits ────────────────────────────────────────────────

    def commit_insert(self, branch: str, data: dict) -> None:
        conn = self.memory_conn
        conn.execute(
            """INSERT INTO commits
               (id, branch, timestamp, title, what, why, files_json,
                next_step, patterns_json, score, author, ci_json,
                experiment_json, ota_trace, raw_block, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["id"], branch, data.get("timestamp", _utcnow()),
                data.get("title", ""), data.get("what", ""),
                data.get("why", ""),
                json.dumps(data.get("files", []), default=str),
                data.get("next_step", ""),
                json.dumps(data.get("patterns", []), default=str) if data.get("patterns") else None,
                data.get("score"),
                data.get("author", ""),
                json.dumps(data.get("ci_context"), default=str) if data.get("ci_context") else None,
                json.dumps(data.get("experiment"), default=str) if data.get("experiment") else None,
                data.get("ota_trace"),
                data.get("raw_block"),
                data.get("created_at", _utcnow()),
            ),
        )
        conn.commit()

    def commit_get(self, branch: str, commit_id: str) -> dict | None:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT * FROM commits WHERE branch = ? AND id = ?",
            (branch, commit_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_commit_dict(row)

    def commit_list(self, branch: str, limit: int = 10, offset: int = 0) -> list[dict]:
        conn = self.memory_conn
        rows = conn.execute(
            """SELECT * FROM commits WHERE branch = ?
               ORDER BY CAST(SUBSTR(id, 2) AS INTEGER) DESC
               LIMIT ? OFFSET ?""",
            (branch, limit, offset),
        ).fetchall()
        return [self._row_to_commit_dict(r) for r in rows]

    def commit_get_next_id(self, branch: str) -> str:
        conn = self.memory_conn
        row = conn.execute(
            """SELECT MAX(CAST(SUBSTR(id, 2) AS INTEGER))
               FROM commits WHERE branch = ?""",
            (branch,),
        ).fetchone()
        max_num = row[0] if row[0] is not None else 0
        return f"C{max_num + 1:03d}"

    def commit_update(self, branch: str, commit_id: str, updates: dict) -> bool:
        conn = self.memory_conn
        if not updates:
            return False
        set_parts = []
        values: list[Any] = []
        for key, val in updates.items():
            if key in ("files", "patterns"):
                set_parts.append(f"{key}_json = ?")
                values.append(json.dumps(val, default=str))
            elif key in ("ci_context", "experiment"):
                col = "ci_json" if key == "ci_context" else "experiment_json"
                set_parts.append(f"{col} = ?")
                values.append(json.dumps(val, default=str) if val else None)
            elif key in _COMMIT_COLUMNS:
                set_parts.append(f"{key} = ?")
                values.append(val)
        if not set_parts:
            return False
        values.extend([branch, commit_id])
        conn.execute(
            f"UPDATE commits SET {', '.join(set_parts)} WHERE branch = ? AND id = ?",
            values,
        )
        changed = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
        return changed > 0

    def commit_search_text(self, branch: str, term: str, max_results: int = 5) -> list[dict]:
        conn = self.memory_conn
        # FTS5 path — fast + ranked
        if getattr(self, "_fts_available", False):
            try:
                rows = conn.execute(
                    """SELECT c.* FROM commits c
                       JOIN commits_fts f ON f.rowid = c.rowid
                       WHERE c.branch = ? AND commits_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (branch, term, max_results),
                ).fetchall()
                if rows:
                    return [self._row_to_commit_dict(r) for r in rows]
            except sqlite3.OperationalError:
                pass  # Malformed FTS query → fall through to LIKE
        like_term = f"%{_escape_like(term)}%"
        rows = conn.execute(
            """SELECT * FROM commits WHERE branch = ?
               AND (title LIKE ? ESCAPE '\\' OR what LIKE ? ESCAPE '\\'
                    OR why LIKE ? ESCAPE '\\' OR next_step LIKE ? ESCAPE '\\'
                    OR files_json LIKE ? ESCAPE '\\')
               ORDER BY CAST(SUBSTR(id, 2) AS INTEGER) DESC
               LIMIT ?""",
            (branch, like_term, like_term, like_term, like_term, like_term, max_results),
        ).fetchall()
        return [self._row_to_commit_dict(r) for r in rows]

    def commit_search_with_snippet(
        self, branch: str, term: str, max_results: int = 5,
    ) -> list[dict]:
        """FTS5-aware commit search returning snippets + ranks.

        Returns [] when FTS5 is unavailable or the query fails.
        Callers must fall back to commit_search_text in that case.
        """
        if not getattr(self, "_fts_available", False):
            return []
        conn = self.memory_conn
        try:
            rows = conn.execute(
                """SELECT c.*,
                          snippet(commits_fts, -1, '[', ']', '...', 12) AS snippet_text,
                          rank
                   FROM commits c
                   JOIN commits_fts f ON f.rowid = c.rowid
                   WHERE c.branch = ? AND commits_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (branch, term, max_results),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for r in rows:
            d = self._row_to_commit_dict(r)
            d["snippet"] = r["snippet_text"]
            d["rank"] = r["rank"]
            out.append(d)
        return out

    def commit_count(self, branch: str) -> int:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT COUNT(*) FROM commits WHERE branch = ?", (branch,),
        ).fetchone()
        return row[0]

    @staticmethod
    def _row_to_commit_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["files"] = json.loads(d.pop("files_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            d["files"] = []
            d.pop("files_json", None)
        try:
            raw = d.pop("patterns_json", None)
            d["patterns"] = json.loads(raw) if raw else []
        except (json.JSONDecodeError, TypeError):
            d["patterns"] = []
        for json_col, key in [("ci_json", "ci_context"), ("experiment_json", "experiment")]:
            raw = d.pop(json_col, None)
            if raw:
                try:
                    d[key] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d[key] = None
            else:
                d[key] = None
        d.setdefault("next", d.pop("next_step", ""))
        d.setdefault("stored_score", d.get("score"))
        return d

    # ── Rolling Summaries ──────────────────────────────────────

    def rolling_summary_get(self, branch: str) -> str:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT summary FROM rolling_summaries WHERE branch = ?",
            (branch,),
        ).fetchone()
        return row["summary"] if row else ""

    def rolling_summary_set(self, branch: str, summary: str) -> None:
        conn = self.memory_conn
        conn.execute(
            """INSERT OR REPLACE INTO rolling_summaries (branch, summary, updated_at)
               VALUES (?, ?, ?)""",
            (branch, summary, _utcnow()),
        )
        conn.commit()

    # ── Branches ───────────────────────────────────────────────

    def branch_create(self, name: str, data: dict) -> None:
        conn = self.memory_conn
        conn.execute(
            """INSERT INTO branches
               (name, status, parent, purpose, hypothesis, conclusion,
                linked_issue, team_owner, priority, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                data.get("status", "active"),
                data.get("parent"),
                data.get("purpose"),
                data.get("hypothesis"),
                data.get("conclusion"),
                data.get("linked_issue"),
                data.get("team_owner"),
                data.get("priority"),
                data.get("created_at", _utcnow()),
            ),
        )
        conn.commit()

    def branch_get(self, name: str) -> dict | None:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT * FROM branches WHERE name = ?", (name,),
        ).fetchone()
        return dict(row) if row else None

    def branch_list(self, status: str | None = None) -> list[dict]:
        conn = self.memory_conn
        if status:
            rows = conn.execute(
                "SELECT * FROM branches WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM branches ORDER BY created_at DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def branch_update(self, name: str, updates: dict) -> bool:
        conn = self.memory_conn
        if not updates:
            return False
        set_parts = []
        values: list[Any] = []
        for key, val in updates.items():
            if key not in _BRANCH_COLUMNS:
                continue
            set_parts.append(f"{key} = ?")
            values.append(val)
        if not set_parts:
            return False
        values.append(name)
        conn.execute(
            f"UPDATE branches SET {', '.join(set_parts)} WHERE name = ?",
            values,
        )
        changed = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
        return changed > 0

    def branch_update_status(self, name: str, status: str) -> bool:
        return self.branch_update(name, {"status": status})
