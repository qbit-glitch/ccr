"""Phase 3d file-backend mixin: durable project TODOs."""

from __future__ import annotations

import json
import os

from ccr.core.storage._sqlite_utils import _utcnow


_DONE_STATUSES = {"done", "canceled"}
_STATUS_ORDER = {
    "blocked": 0,
    "in_progress": 1,
    "review": 2,
    "todo": 3,
    "backlog": 4,
    "done": 5,
    "canceled": 6,
}
_PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2, "low": 3}


class FilePhase3dMixin:
    """TODO list persistence for flat-file CCR storage."""

    def _todos_path(self) -> str:
        return os.path.join(self.ccr_root, "todos.json")

    def _load_todos_data(self) -> dict:
        path = self._todos_path()
        if not os.path.isfile(path):
            return {"version": 1, "next_id": 1, "todos": []}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "next_id": 1, "todos": []}
        if not isinstance(data, dict):
            return {"version": 1, "next_id": 1, "todos": []}
        todos = data.get("todos")
        if not isinstance(todos, list):
            data["todos"] = []
        data.setdefault("version", 1)
        data.setdefault("next_id", self._next_num_from_todos(data["todos"]))
        return data

    def _save_todos_data(self, data: dict) -> None:
        path = self._todos_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _next_num_from_todos(todos: list[dict]) -> int:
        max_num = 0
        for item in todos:
            raw = str(item.get("id", ""))
            if raw.startswith("T"):
                try:
                    max_num = max(max_num, int(raw[1:]))
                except ValueError:
                    pass
        return max_num + 1

    @staticmethod
    def _todo_sort_key(item: dict) -> tuple:
        status = str(item.get("status", "todo"))
        priority = str(item.get("priority", "normal"))
        due = item.get("due_at") or "9999-12-31T23:59:59Z"
        return (
            _STATUS_ORDER.get(status, 9),
            due,
            _PRIORITY_ORDER.get(priority, 9),
            int(item.get("order_index", item.get("order", 1000)) or 1000),
            item.get("updated_at", ""),
        )

    def todo_insert(self, data: dict) -> dict:
        with self._lock:
            store = self._load_todos_data()
            todo = dict(data)
            if not todo.get("id"):
                todo["id"] = f"T{int(store.get('next_id', 1)):03d}"
            now = _utcnow()
            todo.setdefault("created_at", now)
            todo.setdefault("updated_at", now)
            todos = [t for t in store["todos"] if t.get("id") != todo["id"]]
            todos.append(todo)
            store["todos"] = todos
            store["next_id"] = max(
                int(store.get("next_id", 1)),
                self._next_num_from_todos(todos),
            )
            self._save_todos_data(store)
            return todo

    def todo_get(self, todo_id: str) -> dict | None:
        store = self._load_todos_data()
        for item in store["todos"]:
            if item.get("id") == todo_id:
                return dict(item)
        return None

    def todo_list(
        self,
        status: str | None = None,
        branch: str | None = None,
        include_done: bool = False,
        limit: int = 20,
    ) -> list[dict]:
        items = [dict(t) for t in self._load_todos_data()["todos"]]
        if status:
            wanted = {s.strip() for s in status.split(",") if s.strip()}
            items = [t for t in items if t.get("status") in wanted]
        elif not include_done:
            items = [t for t in items if t.get("status") not in _DONE_STATUSES]
        if branch:
            items = [t for t in items if t.get("branch") == branch]
        items.sort(key=self._todo_sort_key)
        return items[: max(0, limit)]

    def todo_update(self, todo_id: str, updates: dict) -> bool:
        with self._lock:
            store = self._load_todos_data()
            found = False
            for item in store["todos"]:
                if item.get("id") == todo_id:
                    item.update(updates)
                    item["updated_at"] = updates.get("updated_at") or _utcnow()
                    found = True
                    break
            if found:
                self._save_todos_data(store)
            return found

    def todo_delete(self, todo_id: str) -> bool:
        with self._lock:
            store = self._load_todos_data()
            before = len(store["todos"])
            store["todos"] = [t for t in store["todos"] if t.get("id") != todo_id]
            if len(store["todos"]) == before:
                return False
            self._save_todos_data(store)
            return True

    def todo_get_next_id(self) -> str:
        store = self._load_todos_data()
        return f"T{int(store.get('next_id', 1)):03d}"

    def todo_reorder(
        self,
        todo_id: str,
        before_id: str | None = None,
        after_id: str | None = None,
        order_index: int | None = None,
    ) -> bool:
        if order_index is None:
            items = self.todo_list(include_done=True, limit=100000)
            index_by_id = {item["id"]: item for item in items}
            if before_id and before_id in index_by_id:
                order_index = int(index_by_id[before_id].get("order_index", 1000)) - 1
            elif after_id and after_id in index_by_id:
                order_index = int(index_by_id[after_id].get("order_index", 1000)) + 1
            else:
                order_index = 1000
        return self.todo_update(todo_id, {"order_index": int(order_index)})
