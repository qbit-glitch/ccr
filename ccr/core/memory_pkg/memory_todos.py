"""TodoMixin -- durable project TODO operations and digest formatting."""

from __future__ import annotations

from datetime import datetime, timezone


TODO_STATUSES = {"backlog", "todo", "in_progress", "blocked", "review", "done", "canceled"}
TODO_OPEN_STATUSES = {"backlog", "todo", "in_progress", "blocked", "review"}
TODO_PRIORITIES = {"urgent", "high", "normal", "low"}
TODO_DONE_STATUSES = {"done", "canceled"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TodoMixin:
    """Durable TODO list helpers.

    Expects the composite class to provide:
        self._storage
        self.get_active_branch()
    """

    def todo_add(
        self,
        title: str,
        description: str = "",
        priority: str = "normal",
        due_at: str = "",
        labels: list[str] | None = None,
        parent_id: str = "",
        branch: str | None = None,
        assignee: str = "",
        status: str = "todo",
        blocked_by: list[str] | None = None,
        source: str = "manual",
        source_commit: str = "",
    ) -> dict:
        title = self._clean_title(title)
        status = self._validate_status(status)
        priority = self._validate_priority(priority)
        self._validate_parent(parent_id)
        self._validate_blockers(blocked_by or [])
        now = _utcnow()
        todo = {
            "id": self._storage.todo_get_next_id(),
            "title": title,
            "description": description.strip(),
            "status": status,
            "priority": priority,
            "order_index": 1000,
            "branch": branch or self.get_active_branch(),
            "scope": "project",
            "parent_id": parent_id.strip(),
            "blocked_by": list(blocked_by or []),
            "labels": [label.strip() for label in labels or [] if label.strip()],
            "assignee": assignee.strip(),
            "due_at": due_at.strip(),
            "started_at": now if status == "in_progress" else "",
            "completed_at": now if status in TODO_DONE_STATUSES else "",
            "completion_note": "",
            "percent_complete": 100 if status == "done" else 0,
            "source": source.strip() or "manual",
            "source_commit": source_commit.strip(),
            "created_at": now,
            "updated_at": now,
        }
        return self._storage.todo_insert(todo)

    def todo_list(
        self,
        status: str | None = None,
        branch: str | None = None,
        include_done: bool = False,
        limit: int = 20,
    ) -> list[dict]:
        return self._storage.todo_list(
            status=status,
            branch=branch,
            include_done=include_done,
            limit=max(0, int(limit)),
        )

    def todo_get(self, todo_id: str) -> dict | None:
        return self._storage.todo_get(todo_id.strip())

    def todo_update(self, todo_id: str, updates: dict) -> dict | None:
        todo_id = todo_id.strip()
        existing = self._storage.todo_get(todo_id)
        if existing is None:
            return None
        clean: dict = {}
        for key, value in updates.items():
            if value is None:
                continue
            if key == "title":
                clean[key] = self._clean_title(str(value))
            elif key == "status":
                status = self._validate_status(str(value))
                clean[key] = status
                if status == "in_progress" and not existing.get("started_at"):
                    clean["started_at"] = _utcnow()
                if status in TODO_DONE_STATUSES:
                    clean["completed_at"] = _utcnow()
                    clean["percent_complete"] = 100 if status == "done" else existing.get("percent_complete", 0)
            elif key == "priority":
                clean[key] = self._validate_priority(str(value))
            elif key == "parent_id":
                parent_id = str(value).strip()
                self._validate_parent(parent_id, current_id=todo_id)
                clean[key] = parent_id
            elif key == "blocked_by":
                blockers = list(value or [])
                self._validate_blockers(blockers, current_id=todo_id)
                clean[key] = blockers
            elif key == "labels":
                clean[key] = [str(label).strip() for label in value or [] if str(label).strip()]
            elif key == "percent_complete":
                pct = int(value)
                clean[key] = min(100, max(0, pct))
            elif key in {
                "description", "due_at", "assignee", "branch", "scope",
                "started_at", "completed_at", "completion_note", "source",
                "source_commit",
            }:
                clean[key] = str(value).strip()
            elif key in {"order_index", "order"}:
                clean["order_index"] = int(value)
        if not clean:
            return existing
        clean["updated_at"] = _utcnow()
        if not self._storage.todo_update(todo_id, clean):
            return None
        return self._storage.todo_get(todo_id)

    def todo_done(
        self,
        todo_id: str,
        note: str = "",
        source_commit: str = "",
    ) -> dict | None:
        return self.todo_update(
            todo_id,
            {
                "status": "done",
                "percent_complete": 100,
                "completed_at": _utcnow(),
                "completion_note": note,
                "source_commit": source_commit,
            },
        )

    def todo_delete(self, todo_id: str, hard: bool = False) -> bool:
        todo_id = todo_id.strip()
        if hard:
            return self._storage.todo_delete(todo_id)
        return self.todo_update(todo_id, {"status": "canceled", "completed_at": _utcnow()}) is not None

    def todo_reorder(
        self,
        todo_id: str,
        before_id: str | None = None,
        after_id: str | None = None,
        order_index: int | None = None,
    ) -> dict | None:
        todo_id = todo_id.strip()
        ok = self._storage.todo_reorder(
            todo_id,
            before_id=before_id.strip() if before_id else None,
            after_id=after_id.strip() if after_id else None,
            order_index=order_index,
        )
        return self._storage.todo_get(todo_id) if ok else None

    def todo_counts(self, branch: str | None = None) -> dict:
        items = self.todo_list(branch=branch, include_done=True, limit=100000)
        counts = {
            "open": 0,
            "backlog": 0,
            "todo": 0,
            "in_progress": 0,
            "blocked": 0,
            "review": 0,
            "done": 0,
            "canceled": 0,
            "overdue": 0,
        }
        today = datetime.now(timezone.utc).date().isoformat()
        for item in items:
            status = item.get("status", "todo")
            counts[status] = counts.get(status, 0) + 1
            if status in TODO_OPEN_STATUSES:
                counts["open"] += 1
                due = str(item.get("due_at") or "")[:10]
                if due and due < today:
                    counts["overdue"] += 1
        return counts

    def todo_digest(
        self,
        limit: int = 5,
        branch: str | None = None,
        include_blocked: bool = True,
    ) -> dict:
        items = self.todo_list(branch=branch, include_done=False, limit=100000)
        if not include_blocked:
            items = [item for item in items if item.get("status") != "blocked"]
        counts = self.todo_counts(branch=branch)
        return {
            "count": counts["open"],
            "overdue": counts["overdue"],
            "blocked": counts["blocked"],
            "todos": items[: max(0, int(limit))],
        }

    def format_todo_digest(
        self,
        limit: int = 5,
        branch: str | None = None,
        max_chars: int = 600,
    ) -> str:
        digest = self.todo_digest(limit=limit, branch=branch)
        todos = digest["todos"]
        if not todos:
            return ""
        lines = ["## Active TODOs"]
        for item in todos:
            extra = []
            if item.get("due_at"):
                extra.append(f"due {item['due_at']}")
            if item.get("blocked_by"):
                extra.append("blocked by " + ",".join(item["blocked_by"]))
            suffix = f" ({'; '.join(extra)})" if extra else ""
            lines.append(
                f"- [{item['id']}] {item.get('status', 'todo')}/"
                f"{item.get('priority', 'normal')}: {item.get('title', '')}{suffix}"
            )
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "..."
        return text

    @staticmethod
    def _clean_title(title: str) -> str:
        title = " ".join((title or "").split())
        if not title:
            raise ValueError("TODO title must not be empty.")
        if len(title) > 200:
            title = title[:200].rstrip()
        return title

    @staticmethod
    def _validate_status(status: str) -> str:
        status = status.strip().lower()
        if status not in TODO_STATUSES:
            raise ValueError(f"Invalid TODO status '{status}'. Valid: {', '.join(sorted(TODO_STATUSES))}")
        return status

    @staticmethod
    def _validate_priority(priority: str) -> str:
        priority = priority.strip().lower()
        if priority not in TODO_PRIORITIES:
            raise ValueError(f"Invalid TODO priority '{priority}'. Valid: {', '.join(sorted(TODO_PRIORITIES))}")
        return priority

    def _validate_parent(self, parent_id: str, current_id: str = "") -> None:
        if not parent_id:
            return
        if parent_id == current_id:
            raise ValueError("TODO cannot be its own parent.")
        if self._storage.todo_get(parent_id) is None:
            raise ValueError(f"Parent TODO '{parent_id}' does not exist.")

    def _validate_blockers(self, blocked_by: list[str], current_id: str = "") -> None:
        for blocker in blocked_by:
            blocker = str(blocker).strip()
            if not blocker:
                continue
            if blocker == current_id:
                raise ValueError("TODO cannot block itself.")
            if self._storage.todo_get(blocker) is None:
                raise ValueError(f"Blocking TODO '{blocker}' does not exist.")
