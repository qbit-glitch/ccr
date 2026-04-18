"""Phase 3c SQLite mixin: discussions, session/phase summaries, summary meta, project state."""

from __future__ import annotations

import json
import sqlite3

from ccr.core.storage._sqlite_utils import _escape_like, _utcnow


class Phase3cMixin:
    """Discussion, summary, and project state methods.

    Requires self.memory_conn from SqliteStorageBackend.
    """

    # ── Discussions ─────────────────────────────────────────────

    def discussion_insert(self, branch: str, data: dict) -> None:
        conn = self.memory_conn
        conn.execute(
            """INSERT OR REPLACE INTO discussions
               (id, branch, timestamp, topic, hypothesis, alternatives,
                decision, rationale, uncertainty, linked_commit, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["id"], branch, data.get("timestamp", _utcnow()),
                data.get("topic", ""), data.get("hypothesis", ""),
                data.get("alternatives", ""), data.get("decision", ""),
                data.get("rationale", ""), data.get("uncertainty", ""),
                data.get("linked_commit"), data.get("created_at", _utcnow()),
            ),
        )
        conn.commit()

    def discussion_list(
        self, branch: str, search: str | None = None,
        topic: str | None = None, date_range: list[str] | None = None,
    ) -> list[dict]:
        conn = self.memory_conn
        # FTS5 path — only when `search` is the sole text filter so that the
        # MATCH query benefits from the inverted index. Additional filters
        # (topic / date_range) are still AND-combined on the base table.
        if search and getattr(self, "_fts_available", False):
            try:
                sql_fts = (
                    "SELECT d.* FROM discussions d "
                    "JOIN discussions_fts f ON f.rowid = d.rowid "
                    "WHERE d.branch = ? AND discussions_fts MATCH ?"
                )
                fts_params: list = [branch, search]
                if topic:
                    sql_fts += " AND d.topic LIKE ? ESCAPE '\\'"
                    fts_params.append(f"%{_escape_like(topic)}%")
                if date_range and len(date_range) >= 2:
                    sql_fts += " AND d.timestamp >= ? AND d.timestamp <= ?"
                    fts_params.extend([date_range[0], date_range[1]])
                sql_fts += " ORDER BY rank"
                rows = conn.execute(sql_fts, fts_params).fetchall()
                if rows:
                    return [self._row_to_discussion_dict(r) for r in rows]
            except sqlite3.OperationalError:
                pass  # Malformed FTS query → fall through to LIKE

        sql = "SELECT * FROM discussions WHERE branch = ?"
        params: list = [branch]

        if search:
            sql += (" AND (topic LIKE ? ESCAPE '\\' OR hypothesis LIKE ? ESCAPE '\\'"
                    " OR decision LIKE ? ESCAPE '\\' OR rationale LIKE ? ESCAPE '\\')")
            pat = f"%{_escape_like(search)}%"
            params.extend([pat, pat, pat, pat])
        if topic:
            sql += " AND topic LIKE ? ESCAPE '\\'"
            params.append(f"%{_escape_like(topic)}%")
        if date_range and len(date_range) >= 2:
            sql += " AND timestamp >= ? AND timestamp <= ?"
            params.extend([date_range[0], date_range[1]])

        sql += " ORDER BY id DESC"
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_discussion_dict(r) for r in rows]

    def discussion_search_text(
        self, branch: str, term: str, max_results: int = 10,
    ) -> list[dict]:
        """Return discussions on `branch` matching `term`. FTS5 when available."""
        conn = self.memory_conn
        if getattr(self, "_fts_available", False):
            try:
                rows = conn.execute(
                    """SELECT d.* FROM discussions d
                       JOIN discussions_fts f ON f.rowid = d.rowid
                       WHERE d.branch = ? AND discussions_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (branch, term, max_results),
                ).fetchall()
                if rows:
                    return [self._row_to_discussion_dict(r) for r in rows]
            except sqlite3.OperationalError:
                pass  # Malformed FTS query → fall through to LIKE
        pat = f"%{_escape_like(term)}%"
        rows = conn.execute(
            """SELECT * FROM discussions WHERE branch = ?
               AND (topic LIKE ? ESCAPE '\\' OR hypothesis LIKE ? ESCAPE '\\'
                    OR alternatives LIKE ? ESCAPE '\\' OR decision LIKE ? ESCAPE '\\'
                    OR rationale LIKE ? ESCAPE '\\' OR uncertainty LIKE ? ESCAPE '\\')
               ORDER BY id DESC LIMIT ?""",
            (branch, pat, pat, pat, pat, pat, pat, max_results),
        ).fetchall()
        return [self._row_to_discussion_dict(r) for r in rows]

    @staticmethod
    def _row_to_discussion_dict(r) -> dict:
        return {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "topic": r["topic"],
            "hypothesis": r["hypothesis"],
            "alternatives": r["alternatives"],
            "decision": r["decision"],
            "rationale": r["rationale"],
            "uncertainty": r["uncertainty"],
            "linked_commit": r["linked_commit"],
        }

    def discussion_get_next_id(self, branch: str) -> str:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT id FROM discussions WHERE branch = ? ORDER BY id DESC LIMIT 1",
            (branch,),
        ).fetchone()
        if row is None:
            return "D001"
        try:
            num = int(row["id"][1:])
            return f"D{num + 1:03d}"
        except (ValueError, IndexError):
            return "D001"

    # ── Session Summaries ───────────────────────────────────────

    def session_summary_insert(self, branch: str, data: dict) -> None:
        conn = self.memory_conn
        conn.execute(
            """INSERT OR REPLACE INTO session_summaries
               (id, branch, start_date, end_date, commit_range,
                accomplished, files_touched, key_decisions, direction, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["id"], branch, data.get("start_date", ""),
                data.get("end_date", ""), data.get("commit_range", ""),
                data.get("accomplished", ""), data.get("files_touched", ""),
                data.get("key_decisions", ""), data.get("direction", ""),
                data.get("created_at", _utcnow()),
            ),
        )
        conn.commit()

    def session_summary_list(self, branch: str, count: int = 3) -> list[dict]:
        conn = self.memory_conn
        rows = conn.execute(
            "SELECT * FROM session_summaries WHERE branch = ? ORDER BY id DESC LIMIT ?",
            (branch, count),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "start_date": r["start_date"],
                "end_date": r["end_date"],
                "branch": r["branch"],
                "commit_range": r["commit_range"] or "",
                "accomplished": r["accomplished"] or "",
                "files_touched": r["files_touched"] or "",
                "key_decisions": r["key_decisions"] or "",
                "direction": r["direction"] or "",
            }
            for r in rows
        ]

    def session_summary_get_next_id(self, branch: str) -> str:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT id FROM session_summaries WHERE branch = ? ORDER BY id DESC LIMIT 1",
            (branch,),
        ).fetchone()
        if row is None:
            return "S001"
        try:
            num = int(row["id"][1:])
            return f"S{num + 1:03d}"
        except (ValueError, IndexError):
            return "S001"

    # ── Phase Summaries ─────────────────────────────────────────

    def phase_summary_insert(self, data: dict) -> None:
        conn = self.memory_conn
        conn.execute(
            """INSERT OR REPLACE INTO phase_summaries
               (id, start_date, end_date, scope, goal, outcome,
                accomplishments, files_changed, branch_summary, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["id"], data.get("start_date", ""),
                data.get("end_date", ""), data.get("scope", ""),
                data.get("goal", ""), data.get("outcome", ""),
                data.get("accomplishments", ""),
                data.get("files_changed", ""),
                data.get("branch_summary", ""),
                data.get("created_at", _utcnow()),
            ),
        )
        conn.commit()

    def phase_summary_list(self, count: int = 3) -> list[dict]:
        conn = self.memory_conn
        rows = conn.execute(
            "SELECT * FROM phase_summaries ORDER BY id DESC LIMIT ?",
            (count,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "start_date": r["start_date"] or "",
                "end_date": r["end_date"] or "",
                "scope": r["scope"] or "",
                "goal": r["goal"] or "",
                "outcome": r["outcome"] or "",
                "accomplishments": r["accomplishments"] or "",
                "files_changed": r["files_changed"] or "",
                "branch_summary": r["branch_summary"] or "",
            }
            for r in rows
        ]

    def phase_summary_get_next_id(self) -> str:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT id FROM phase_summaries ORDER BY id DESC LIMIT 1",
        ).fetchone()
        if row is None:
            return "P001"
        try:
            num = int(row["id"][1:])
            return f"P{num + 1:03d}"
        except (ValueError, IndexError):
            return "P001"

    # ── Summary Meta ────────────────────────────────────────────

    def summary_meta_load(self) -> dict:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT value_json FROM summary_meta WHERE key = '_root'",
        ).fetchone()
        if row is None:
            return {
                "version": 1, "session": {},
                "phase": {"last_commit_id": None, "last_summary_id": None, "last_generated": None},
                "overview": {"last_generated": None, "phase_count_at_generation": 0},
            }
        try:
            return json.loads(row["value_json"])
        except (json.JSONDecodeError, TypeError):
            return {"version": 1, "session": {}, "phase": {}, "overview": {}}

    def summary_meta_save(self, data: dict) -> None:
        conn = self.memory_conn
        conn.execute(
            "INSERT OR REPLACE INTO summary_meta (key, value_json, updated_at) VALUES ('_root', ?, ?)",
            (json.dumps(data, default=str), _utcnow()),
        )
        conn.commit()

    # ── Project State ───────────────────────────────────────────

    def project_state_get(self, key: str) -> str | None:
        conn = self.memory_conn
        row = conn.execute(
            "SELECT value FROM project_state WHERE key = ?", (key,),
        ).fetchone()
        return row["value"] if row else None

    def project_state_set(self, key: str, value: str) -> None:
        conn = self.memory_conn
        conn.execute(
            "INSERT OR REPLACE INTO project_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, _utcnow()),
        )
        conn.commit()
