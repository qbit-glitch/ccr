"""Phase 2 file-backend mixin: playbook, failure lessons, schema, audit."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from ccr.core.storage._sqlite_utils import _utcnow

logger = logging.getLogger(__name__)

_BULLET_RE = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)")


class FilePhase2Mixin:
    """Playbook bullets, sections, failure lessons, schema, and audit methods.

    Requires self.ccr_root, self.global_ccr_root, and self._lock
    from FileStorageBackend.
    """

    # ── Playbook (Phase 2) ─────────────────────────────────────

    def _playbook_path(self, scope: str) -> str:
        if scope == "global" and self.global_ccr_root:
            return os.path.join(self.global_ccr_root, "global_playbook.txt")
        return os.path.join(self.ccr_root, "playbook.txt")

    def _failure_lessons_path(self, scope: str) -> str:
        if scope == "global" and self.global_ccr_root:
            return os.path.join(self.global_ccr_root, "global_failure_lessons.json")
        return os.path.join(self.ccr_root, "failure_lessons.json")

    def _schema_path(self, scope: str) -> str:
        if scope == "global" and self.global_ccr_root:
            return os.path.join(self.global_ccr_root, "global_playbook_schema.json")
        return os.path.join(self.ccr_root, "playbook_schema.json")

    def _history_path(self, scope: str) -> str:
        return os.path.join(self.ccr_root, "playbook_history.json")

    def _archived_path(self, scope: str) -> str:
        return os.path.join(self.ccr_root, "archived_bullets.json")

    def _load_playbook_data(self, scope: str) -> tuple[list[dict], list[str]]:
        path = self._playbook_path(scope)
        if not os.path.isfile(path):
            return [], []
        with open(path, encoding="utf-8") as f:
            text = f.read()
        bullets: list[dict] = []
        sections: list[str] = []
        current_section = ""
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("## "):
                current_section = line[3:].strip()
                if current_section not in sections:
                    sections.append(current_section)
                continue
            m = _BULLET_RE.match(line)
            if m:
                bullets.append({
                    "id": m.group(1),
                    "helpful": int(m.group(2)),
                    "harmful": int(m.group(3)),
                    "content": m.group(4),
                    "section": current_section,
                    "scope": "general",
                    "when_to_apply": "",
                    "trigger_text": "",
                    "action": "",
                    "weighted_helpful": 0.0,
                    "weighted_harmful": 0.0,
                    "personal_decay_rate": 0.0,
                    "grpo_advantage": 0.0,
                    "last_updated": None,
                    "created_at": _utcnow(),
                })
        fl_data = self._load_failure_lessons_json(scope)
        for b in bullets:
            bid = b["id"]
            if bid in fl_data:
                ext = fl_data[bid]
                b["scope"] = ext.get("scope", b["scope"])
                b["when_to_apply"] = ext.get("when_to_apply", "")
                b["trigger_text"] = ext.get("trigger", "")
                b["action"] = ext.get("action", "")
                b["weighted_helpful"] = ext.get("weighted_helpful", 0.0)
                b["weighted_harmful"] = ext.get("weighted_harmful", 0.0)
                b["personal_decay_rate"] = ext.get("personal_decay_rate", 0.0)
        return bullets, sections

    def _save_playbook_data(self, bullets: list[dict], sections: list[str], scope: str) -> None:
        path = self._playbook_path(scope)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        lines: list[str] = []
        by_section: dict[str, list[dict]] = {}
        for b in bullets:
            sec = b.get("section", "")
            by_section.setdefault(sec, []).append(b)
        for sec in sections:
            lines.append(f"## {sec}")
            for b in by_section.get(sec, []):
                lines.append(
                    f"[{b['id']}] helpful={b.get('helpful', 0)} "
                    f"harmful={b.get('harmful', 0)} :: {b.get('content', '')}"
                )
            lines.append("")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _load_failure_lessons_json(self, scope: str) -> dict[str, Any]:
        path = self._failure_lessons_path(scope)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_failure_lessons_json(self, data: dict, scope: str) -> None:
        path = self._failure_lessons_path(scope)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def bullet_get(self, bullet_id: str, scope: str = "project") -> dict | None:
        with self._lock:
            bullets, _ = self._load_playbook_data(scope)
            for b in bullets:
                if b["id"] == bullet_id:
                    return b
            return None

    def bullet_list(self, section: str | None = None, scope: str = "project") -> list[dict]:
        with self._lock:
            bullets, _ = self._load_playbook_data(scope)
            if section is not None:
                return [b for b in bullets if b.get("section") == section]
            return bullets

    def bullet_insert(self, bullet: dict, scope: str = "project") -> None:
        with self._lock:
            bullets, sections = self._load_playbook_data(scope)
            bullets.append(bullet)
            sec = bullet.get("section", "")
            if sec and sec not in sections:
                sections.append(sec)
            self._save_playbook_data(bullets, sections, scope)

    def bullet_update(self, bullet_id: str, updates: dict, scope: str = "project") -> bool:
        with self._lock:
            bullets, sections = self._load_playbook_data(scope)
            for b in bullets:
                if b["id"] == bullet_id:
                    b.update(updates)
                    self._save_playbook_data(bullets, sections, scope)
                    return True
            return False

    def bullet_delete(self, bullet_id: str, scope: str = "project") -> bool:
        with self._lock:
            bullets, sections = self._load_playbook_data(scope)
            new_bullets = [b for b in bullets if b["id"] != bullet_id]
            if len(new_bullets) == len(bullets):
                return False
            self._save_playbook_data(new_bullets, sections, scope)
            fl_data = self._load_failure_lessons_json(scope)
            fl_data.pop(bullet_id, None)
            self._save_failure_lessons_json(fl_data, scope)
            return True

    def bullet_update_counters(self, bullet_tags: list[dict], scope: str = "project") -> int:
        with self._lock:
            bullets, sections = self._load_playbook_data(scope)
            id_index = {b["id"]: b for b in bullets}
            updated = 0
            for tag in bullet_tags:
                bid = tag.get("id") or tag.get("bullet", "")
                if not bid or bid not in id_index:
                    continue
                b = id_index[bid]
                raw_weight = tag.get("weight", 1.0)
                try:
                    weight = max(0.0, min(1.0, float(raw_weight)))
                except (TypeError, ValueError):
                    weight = 1.0
                tag_val = tag.get("tag", "neutral")
                if tag_val == "helpful":
                    b["helpful"] = b.get("helpful", 0) + 1
                    b["weighted_helpful"] = b.get("weighted_helpful", 0.0) + weight
                    h, harm = b["helpful"], b.get("harmful", 0)
                    b["personal_decay_rate"] = max(0.90, min(0.99, 0.95 + (h - harm) * 0.002))
                    b["last_updated"] = _utcnow()
                    updated += 1
                elif tag_val == "harmful":
                    b["harmful"] = b.get("harmful", 0) + 1
                    b["weighted_harmful"] = b.get("weighted_harmful", 0.0) + weight
                    h, harm = b.get("helpful", 0), b["harmful"]
                    b["personal_decay_rate"] = max(0.90, min(0.99, 0.95 + (h - harm) * 0.002))
                    b["last_updated"] = _utcnow()
                    lesson = tag.get("failure_lesson")
                    if isinstance(lesson, dict) and lesson:
                        self._append_failure_lesson_unlocked(bid, lesson, scope)
                    updated += 1
            self._save_playbook_data(bullets, sections, scope)
            return updated

    def _append_failure_lesson_unlocked(self, bullet_id: str, lesson: dict, scope: str) -> None:
        fl_data = self._load_failure_lessons_json(scope)
        if bullet_id not in fl_data:
            fl_data[bullet_id] = {"lessons": []}
        elif isinstance(fl_data[bullet_id], list):
            fl_data[bullet_id] = {"lessons": fl_data[bullet_id]}
        fl_data[bullet_id]["lessons"].append({
            "failure_point": lesson.get("failure_point", ""),
            "flawed_reasoning": lesson.get("flawed_reasoning", ""),
            "counterfactual": lesson.get("counterfactual", ""),
            "prevention_principle": lesson.get("prevention_principle", ""),
            "task_context": lesson.get("task_context", ""),
            "timestamp": lesson.get("timestamp", _utcnow()),
            "evolved": lesson.get("evolved", False),
        })
        self._save_failure_lessons_json(fl_data, scope)

    def bullet_get_next_id(self, scope: str = "project") -> int:
        with self._lock:
            bullets, _ = self._load_playbook_data(scope)
            max_num = 0
            for b in bullets:
                bid = b.get("id", "")
                dash = bid.rfind("-")
                if dash >= 0:
                    try:
                        num = int(bid[dash + 1:])
                        max_num = max(max_num, num)
                    except ValueError:
                        pass
            return max_num + 1

    # ── Failure Lessons ────────────────────────────────────────

    def failure_lessons_for_bullet(self, bullet_id: str, scope: str = "project") -> list[dict]:
        with self._lock:
            fl_data = self._load_failure_lessons_json(scope)
            entry = fl_data.get(bullet_id)
            if entry is None:
                return []
            if isinstance(entry, list):
                return entry
            return entry.get("lessons", [])

    def failure_lessons_insert(self, bullet_id: str, lesson: dict, scope: str = "project") -> None:
        with self._lock:
            self._append_failure_lesson_unlocked(bullet_id, lesson, scope)

    def failure_lessons_mark_evolved(self, bullet_id: str, scope: str = "project") -> int:
        with self._lock:
            fl_data = self._load_failure_lessons_json(scope)
            entry = fl_data.get(bullet_id)
            if entry is None:
                return 0
            lessons = entry.get("lessons", []) if isinstance(entry, dict) else entry
            count = 0
            for lesson in lessons:
                if not lesson.get("evolved", False):
                    lesson["evolved"] = True
                    count += 1
            if isinstance(entry, dict):
                entry["lessons"] = lessons
            else:
                fl_data[bullet_id] = {"lessons": lessons}
            self._save_failure_lessons_json(fl_data, scope)
            return count

    def failure_lessons_all(self, scope: str = "project") -> dict[str, list[dict]]:
        with self._lock:
            fl_data = self._load_failure_lessons_json(scope)
            result: dict[str, list[dict]] = {}
            for bid, entry in fl_data.items():
                if isinstance(entry, list):
                    result[bid] = entry
                else:
                    result[bid] = entry.get("lessons", [])
            return result

    # ── Playbook Sections ──────────────────────────────────────

    def playbook_sections_get(self, scope: str = "project") -> list[str]:
        with self._lock:
            _, sections = self._load_playbook_data(scope)
            return sections

    def playbook_sections_set(self, sections: list[str], scope: str = "project") -> None:
        with self._lock:
            bullets, _ = self._load_playbook_data(scope)
            self._save_playbook_data(bullets, sections, scope)

    # ── Playbook Schema ────────────────────────────────────────

    def playbook_schema_load(self, scope: str = "project") -> dict:
        path = self._schema_path(scope)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("current", data)
        except (json.JSONDecodeError, OSError):
            return {}

    def playbook_schema_save(self, schema: dict, history_entry: dict | None = None, scope: str = "project") -> None:
        path = self._schema_path(scope)
        existing: dict[str, Any] = {}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        existing["current"] = schema
        if history_entry:
            existing.setdefault("history", []).append(history_entry)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)

    def playbook_schema_history(self, scope: str = "project") -> list[dict]:
        path = self._schema_path(scope)
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("history", [])
        except (json.JSONDecodeError, OSError):
            return []

    # ── Audit ──────────────────────────────────────────────────

    def delta_history_append(self, entry: dict, scope: str = "project") -> None:
        path = self._history_path(scope)
        history: list[dict] = []
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        entry.setdefault("timestamp", _utcnow())
        history.append(entry)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)

    def archived_bullets_insert(self, bullets: list[dict], reason: str, scope: str = "project") -> int:
        path = self._archived_path(scope)
        archived: list[dict] = []
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    archived = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        now = _utcnow()
        for b in bullets:
            archived.append({**b, "archived_at": now, "reason": reason})
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(archived, f, indent=2, default=str)
        return len(bullets)
