"""Phase 3d SQLite mixin: durable project TODOs."""

from __future__ import annotations

import json
import sqlite3

from ccr.core.storage._sqlite_utils import _utcnow


_DONE_STATUSES = {"done", "canceled"}
_TODO_COLUMNS = {
    "title",
    "description",
    "status",
    "priority",
    "order_index",
    "branch",
    "scope",
    "parent_id",
    "blocked_by",
    "labels",
    "assignee",
    "due_at",
    "started_at",
    "completed_at",
    "completion_note",
    "percent_complete",
    "source",
    "source_commit",
    "created_at",
    "updated_at",
}


class Phase3dMixin:
    """TODO list persistence for SQLite CCR storage."""

    def todo_insert(self, data: dict) -> dict:
        conn = self.memory_conn
        todo = dict(data)
        now = _utcnow()
        todo.setdefault("created_at", now)
        todo.setdefault("updated_at", now)
        if not todo.get("id"):
            todo["id"] = self.todo_get_next_id()
        conn.execute(
            """INSERT OR REPLACE INTO todos
               (id, title, description, status, priority, order_index, branch,
                scope, parent_id, blocked_by_json, labels_json, assignee,
                due_at, started_at, completed_at, completion_note,
                percent_complete, source, source_commit, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                todo["id"],
                todo.get("title", ""),
                todo.get("description", ""),
                todo.get("status", "todo"),
                todo.get("priority", "normal"),
                int(todo.get("order_index", todo.get("order", 1000)) or 1000),
                todo.get("branch", "main"),
                todo.get("scope", "project"),
                todo.get("parent_id", ""),
                json.dumps(todo.get("blocked_by", [])),
                json.dumps(todo.get("labels", [])),
                todo.get("assignee", ""),
                todo.get("due_at", ""),
                todo.get("started_at", ""),
                todo.get("completed_at", ""),
                todo.get("completion_note", ""),
                int(todo.get("percent_complete", 0) or 0),
                todo.get("source", "manual"),
                todo.get("source_commit", ""),
                todo.get("created_at", now),
                todo.get("updated_at", now),
            ),
        )
        conn.commit()
        stored = self.todo_get(todo["id"])
        return stored or todo

    def todo_get(self, todo_id: str) -> dict | None:
        row = self.memory_conn.execute(
            "SELECT * FROM todos WHERE id = ?", (todo_id,),
        ).fetchone()
        return self._row_to_todo_dict(row) if row else None

    def todo_list(
        self,
        status: str | None = None,
        branch: str | None = None,
        include_done: bool = False,
        limit: int = 20,
    ) -> list[dict]:
        sql = "SELECT * FROM todos WHERE 1=1"
        params: list = []
        if status:
            statuses = [s.strip() for s in status.split(",") if s.strip()]
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                sql += f" AND status IN ({placeholders})"
                params.extend(statuses)
        elif not include_done:
            placeholders = ",".join("?" for _ in _DONE_STATUSES)
            sql += f" AND status NOT IN ({placeholders})"
            params.extend(sorted(_DONE_STATUSES))
        if branch:
            sql += " AND branch = ?"
            params.append(branch)
        sql += (
            " ORDER BY "
            "CASE status "
            "WHEN 'blocked' THEN 0 WHEN 'in_progress' THEN 1 WHEN 'review' THEN 2 "
            "WHEN 'todo' THEN 3 WHEN 'backlog' THEN 4 WHEN 'done' THEN 5 "
            "WHEN 'canceled' THEN 6 ELSE 9 END, "
            "CASE WHEN due_at = '' THEN '9999-12-31T23:59:59Z' ELSE due_at END, "
            "CASE priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 "
            "WHEN 'normal' THEN 2 WHEN 'low' THEN 3 ELSE 9 END, "
            "order_index ASC, updated_at DESC LIMIT ?"
        )
        params.append(max(0, int(limit)))
        rows = self.memory_conn.execute(sql, params).fetchall()
        return [self._row_to_todo_dict(row) for row in rows]

    def todo_update(self, todo_id: str, updates: dict) -> bool:
        clean = {k: v for k, v in updates.items() if k in _TODO_COLUMNS}
        if not clean:
            return self.todo_get(todo_id) is not None
        clean["updated_at"] = clean.get("updated_at") or _utcnow()
        assignments = []
        params = []
        for key, value in clean.items():
            column = key
            if key == "blocked_by":
                column = "blocked_by_json"
                value = json.dumps(value or [])
            elif key == "labels":
                column = "labels_json"
                value = json.dumps(value or [])
            assignments.append(f"{column} = ?")
            params.append(value)
        params.append(todo_id)
        cursor = self.memory_conn.execute(
            f"UPDATE todos SET {', '.join(assignments)} WHERE id = ?",
            params,
        )
        self.memory_conn.commit()
        return cursor.rowcount > 0

    def todo_delete(self, todo_id: str) -> bool:
        cursor = self.memory_conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        self.memory_conn.commit()
        return cursor.rowcount > 0

    def todo_get_next_id(self) -> str:
        row = self.memory_conn.execute(
            "SELECT id FROM todos WHERE id GLOB 'T[0-9]*' ORDER BY CAST(substr(id, 2) AS INTEGER) DESC LIMIT 1",
        ).fetchone()
        if row is None:
            return "T001"
        try:
            return f"T{int(row['id'][1:]) + 1:03d}"
        except (ValueError, IndexError, TypeError):
            return "T001"

    def todo_reorder(
        self,
        todo_id: str,
        before_id: str | None = None,
        after_id: str | None = None,
        order_index: int | None = None,
    ) -> bool:
        if order_index is None:
            order_index = 1000
            if before_id:
                before = self.todo_get(before_id)
                if before:
                    order_index = int(before.get("order_index", 1000)) - 1
            elif after_id:
                after = self.todo_get(after_id)
                if after:
                    order_index = int(after.get("order_index", 1000)) + 1
        return self.todo_update(todo_id, {"order_index": int(order_index)})

    @staticmethod
    def _loads_list(raw: str | None) -> list:
        try:
            value = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []

    @classmethod
    def _row_to_todo_dict(cls, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "description": row["description"] or "",
            "status": row["status"],
            "priority": row["priority"],
            "order_index": row["order_index"],
            "order": row["order_index"],
            "branch": row["branch"],
            "scope": row["scope"],
            "parent_id": row["parent_id"] or "",
            "blocked_by": cls._loads_list(row["blocked_by_json"]),
            "labels": cls._loads_list(row["labels_json"]),
            "assignee": row["assignee"] or "",
            "due_at": row["due_at"] or "",
            "started_at": row["started_at"] or "",
            "completed_at": row["completed_at"] or "",
            "completion_note": row["completion_note"] or "",
            "percent_complete": row["percent_complete"],
            "source": row["source"] or "manual",
            "source_commit": row["source_commit"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
