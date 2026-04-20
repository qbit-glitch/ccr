"""Phase 2 SQLite mixin: playbook bullets, sections, failure lessons, schema, audit."""

from __future__ import annotations

import json
from typing import Any

from ccr.core.storage._sqlite_utils import _BULLET_COLUMNS, _utcnow


class Phase2Mixin:
    """Playbook bullet, section, failure lesson, schema, and audit methods.

    Requires self._get_scoped_conn(scope) from SqliteStorageBackend.
    """

    # ── Playbook Bullets ───────────────────────────────────────

    def bullet_get(self, bullet_id: str, scope: str = "project") -> dict | None:
        conn = self._get_scoped_conn(scope)
        row = conn.execute(
            "SELECT * FROM playbook_bullets WHERE id = ?", (bullet_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def bullet_list(self, section: str | None = None, scope: str = "project") -> list[dict]:
        conn = self._get_scoped_conn(scope)
        if section is not None:
            rows = conn.execute(
                "SELECT * FROM playbook_bullets WHERE section = ?", (section,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM playbook_bullets").fetchall()
        return [dict(r) for r in rows]

    def bullet_insert(self, bullet: dict, scope: str = "project") -> None:
        conn = self._get_scoped_conn(scope)
        self._bullet_insert_nc(conn, bullet)
        conn.commit()

    def _bullet_insert_nc(self, conn: Any, bullet: dict) -> None:
        """No-commit variant of bullet_insert; call inside an outer txn.

        Phase 5a review C2 fix: save_to_backend wraps multiple primitives in a
        single transaction. Inner commits would fragment that txn (an exception
        after the commit cannot roll the inserted row back). _nc variants let
        save_to_backend run as one atomic unit.
        """
        conn.execute(
            """INSERT INTO playbook_bullets
               (id, section, content, helpful, harmful, scope, when_to_apply,
                trigger_text, action, weighted_helpful, weighted_harmful,
                personal_decay_rate, grpo_advantage, last_updated, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bullet["id"], bullet["section"], bullet["content"],
                bullet.get("helpful", 0), bullet.get("harmful", 0),
                bullet.get("scope", "general"), bullet.get("when_to_apply", ""),
                bullet.get("trigger_text", ""), bullet.get("action", ""),
                bullet.get("weighted_helpful", 0.0), bullet.get("weighted_harmful", 0.0),
                bullet.get("personal_decay_rate", 0.0), bullet.get("grpo_advantage", 0.0),
                bullet.get("last_updated"), bullet.get("created_at", _utcnow()),
            ),
        )

    def bullet_update(self, bullet_id: str, updates: dict, scope: str = "project") -> bool:
        conn = self._get_scoped_conn(scope)
        changed = self._bullet_update_nc(conn, bullet_id, updates)
        conn.commit()
        return changed

    def _bullet_update_nc(self, conn: Any, bullet_id: str, updates: dict) -> bool:
        """No-commit variant of bullet_update; call inside an outer txn."""
        if not updates:
            return False
        set_parts = []
        values: list[Any] = []
        for key, val in updates.items():
            if key not in _BULLET_COLUMNS:
                continue
            set_parts.append(f"{key} = ?")
            values.append(val)
        if not set_parts:
            return False
        values.append(bullet_id)
        conn.execute(
            f"UPDATE playbook_bullets SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )
        changed = conn.execute("SELECT changes()").fetchone()[0]
        return changed > 0

    def bullet_delete(self, bullet_id: str, scope: str = "project") -> bool:
        conn = self._get_scoped_conn(scope)
        changed = self._bullet_delete_nc(conn, bullet_id)
        conn.commit()
        return changed

    def _bullet_delete_nc(self, conn: Any, bullet_id: str) -> bool:
        """No-commit variant of bullet_delete; call inside an outer txn."""
        conn.execute("DELETE FROM playbook_bullets WHERE id = ?", (bullet_id,))
        changed = conn.execute("SELECT changes()").fetchone()[0]
        return changed > 0

    def bullet_update_counters(self, bullet_tags: list[dict], scope: str = "project") -> int:
        conn = self._get_scoped_conn(scope)
        updated = 0
        now = _utcnow()
        for tag in bullet_tags:
            bid = tag.get("id") or tag.get("bullet", "")
            if not bid:
                continue
            raw_weight = tag.get("weight", 1.0)
            try:
                weight = max(0.0, min(1.0, float(raw_weight)))
            except (TypeError, ValueError):
                weight = 1.0
            tag_val = tag.get("tag", "neutral")
            if tag_val == "helpful":
                conn.execute(
                    """UPDATE playbook_bullets SET
                        helpful = helpful + 1,
                        weighted_helpful = weighted_helpful + ?,
                        last_updated = ?,
                        personal_decay_rate = max(0.90, min(0.99,
                            0.95 + (helpful + 1 - harmful) * 0.002))
                        WHERE id = ?""",
                    (weight, now, bid),
                )
            elif tag_val == "harmful":
                conn.execute(
                    """UPDATE playbook_bullets SET
                        harmful = harmful + 1,
                        weighted_harmful = weighted_harmful + ?,
                        last_updated = ?,
                        personal_decay_rate = max(0.90, min(0.99,
                            0.95 + (helpful - harmful - 1) * 0.002))
                        WHERE id = ?""",
                    (weight, now, bid),
                )
                lesson = tag.get("failure_lesson")
                if isinstance(lesson, dict) and lesson:
                    self.failure_lessons_insert(bid, lesson, scope)
            else:
                continue
            if conn.execute("SELECT changes()").fetchone()[0] > 0:
                updated += 1
        conn.commit()
        return updated

    def bullet_get_next_id(self, scope: str = "project") -> int:
        conn = self._get_scoped_conn(scope)
        row = conn.execute(
            """SELECT MAX(CAST(SUBSTR(id, INSTR(id, '-') + 1) AS INTEGER))
               FROM playbook_bullets""",
        ).fetchone()
        max_num = row[0] if row[0] is not None else 0
        return max_num + 1

    # ── Playbook Sections ──────────────────────────────────────

    def playbook_sections_get(self, scope: str = "project") -> list[str]:
        conn = self._get_scoped_conn(scope)
        rows = conn.execute(
            "SELECT name FROM playbook_sections ORDER BY position",
        ).fetchall()
        return [r["name"] for r in rows]

    def playbook_sections_set(self, sections: list[str], scope: str = "project") -> None:
        conn = self._get_scoped_conn(scope)
        self._playbook_sections_set_nc(conn, sections)
        conn.commit()

    def _playbook_sections_set_nc(self, conn: Any, sections: list[str]) -> None:
        """No-commit variant of playbook_sections_set; call inside an outer txn."""
        conn.execute("DELETE FROM playbook_sections")
        for i, name in enumerate(sections):
            conn.execute(
                "INSERT INTO playbook_sections (position, name) VALUES (?, ?)",
                (i, name),
            )

    # ── Failure Lessons ────────────────────────────────────────

    def failure_lessons_for_bullet(self, bullet_id: str, scope: str = "project") -> list[dict]:
        conn = self._get_scoped_conn(scope)
        rows = conn.execute(
            "SELECT * FROM failure_lessons WHERE bullet_id = ?", (bullet_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def failure_lessons_insert(self, bullet_id: str, lesson: dict, scope: str = "project") -> None:
        conn = self._get_scoped_conn(scope)
        self._failure_lessons_insert_nc(conn, bullet_id, lesson)
        conn.commit()

    def _failure_lessons_insert_nc(self, conn: Any, bullet_id: str, lesson: dict) -> None:
        """No-commit variant of failure_lessons_insert; call inside an outer txn."""
        conn.execute(
            """INSERT INTO failure_lessons
               (bullet_id, failure_point, flawed_reasoning, counterfactual,
                prevention_principle, task_context, timestamp, evolved)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bullet_id,
                lesson.get("failure_point", ""),
                lesson.get("flawed_reasoning", ""),
                lesson.get("counterfactual", ""),
                lesson.get("prevention_principle", ""),
                lesson.get("task_context", ""),
                lesson.get("timestamp", _utcnow()),
                int(lesson.get("evolved", False)),
            ),
        )

    def failure_lessons_mark_evolved(self, bullet_id: str, scope: str = "project") -> int:
        conn = self._get_scoped_conn(scope)
        changed = self._failure_lessons_mark_evolved_nc(conn, bullet_id)
        conn.commit()
        return changed

    def _failure_lessons_mark_evolved_nc(self, conn: Any, bullet_id: str) -> int:
        """No-commit variant of failure_lessons_mark_evolved; call inside an outer txn."""
        conn.execute(
            "UPDATE failure_lessons SET evolved = 1 WHERE bullet_id = ? AND evolved = 0",
            (bullet_id,),
        )
        changed = conn.execute("SELECT changes()").fetchone()[0]
        return changed

    def failure_lessons_all(self, scope: str = "project") -> dict[str, list[dict]]:
        conn = self._get_scoped_conn(scope)
        rows = conn.execute("SELECT * FROM failure_lessons ORDER BY bullet_id").fetchall()
        result: dict[str, list[dict]] = {}
        for r in rows:
            bid = r["bullet_id"]
            if bid not in result:
                result[bid] = []
            result[bid].append(dict(r))
        return result

    # ── Playbook Schema ────────────────────────────────────────

    def playbook_schema_load(self, scope: str = "project") -> dict:
        conn = self._get_scoped_conn(scope)
        row = conn.execute(
            "SELECT data_json FROM playbook_schema ORDER BY version DESC LIMIT 1",
        ).fetchone()
        if row is None:
            return {}
        try:
            return json.loads(row["data_json"])
        except json.JSONDecodeError:
            return {}

    def playbook_schema_save(self, schema: dict, history_entry: dict | None = None, scope: str = "project") -> None:
        conn = self._get_scoped_conn(scope)
        version = schema.get("version", 1)
        conn.execute(
            """INSERT OR REPLACE INTO playbook_schema
               (version, data_json, parent_version, change_description, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                version,
                json.dumps(schema, default=str),
                schema.get("parent_version"),
                schema.get("change_description", ""),
                schema.get("created_at", _utcnow()),
            ),
        )
        if history_entry:
            h_version = history_entry.get("version", version - 1)
            conn.execute(
                """INSERT OR IGNORE INTO playbook_schema
                   (version, data_json, parent_version, change_description, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    h_version,
                    json.dumps(history_entry, default=str),
                    history_entry.get("parent_version"),
                    history_entry.get("change_description", ""),
                    history_entry.get("created_at", _utcnow()),
                ),
            )
        conn.commit()

    def playbook_schema_history(self, scope: str = "project") -> list[dict]:
        conn = self._get_scoped_conn(scope)
        rows = conn.execute(
            "SELECT * FROM playbook_schema ORDER BY version",
        ).fetchall()
        result = []
        for r in rows:
            entry = dict(r)
            try:
                entry["data"] = json.loads(entry.pop("data_json"))
            except (json.JSONDecodeError, KeyError):
                pass
            result.append(entry)
        return result

    # ── Audit ──────────────────────────────────────────────────

    def delta_history_append(self, entry: dict, scope: str = "project") -> None:
        conn = self._get_scoped_conn(scope)
        conn.execute(
            """INSERT INTO delta_history
               (timestamp, author, ops_count, applied_count, scope,
                operations_json, failed_ids_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.get("timestamp", _utcnow()),
                entry.get("author", ""),
                entry.get("ops_count", 0),
                entry.get("applied_count", 0),
                entry.get("scope", scope),
                json.dumps(entry.get("operations", []), default=str) if entry.get("operations") else None,
                json.dumps(entry.get("failed_ids", []), default=str) if entry.get("failed_ids") else None,
            ),
        )
        conn.commit()

    def archived_bullets_insert(self, bullets: list[dict], reason: str, scope: str = "project") -> int:
        conn = self._get_scoped_conn(scope)
        now = _utcnow()
        for b in bullets:
            conn.execute(
                """INSERT INTO archived_bullets
                   (id, section, content, helpful, harmful, archived_at, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    b.get("id", ""),
                    b.get("section", ""),
                    b.get("content", ""),
                    b.get("helpful", 0),
                    b.get("harmful", 0),
                    now,
                    reason,
                ),
            )
        conn.commit()
        return len(bullets)
