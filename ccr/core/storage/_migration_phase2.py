"""Phase 2 migration: playbook flat files.

Flat-file sources: playbook.txt, failure_lessons.json,
playbook_schema.json, playbook_history.json, archived_bullets.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from typing import Any

from ccr.core.storage._migration_utils import _backup_file

logger = logging.getLogger(__name__)

_BULLET_RE = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)")


def migrate_phase_2(
    ccr_root: str, db_path: str,
) -> dict[str, Any]:
    """Migrate playbook flat files to SQLite tables.

    Parses: playbook.txt, failure_lessons.json, playbook_schema.json,
    playbook_history.json, archived_bullets.json.
    """
    result: dict[str, Any] = {"migrated": 0, "errors": []}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    try:
        conn.execute("BEGIN")
        result["migrated"] += _migrate_playbook_txt(ccr_root, conn)
        result["migrated"] += _migrate_failure_lessons_json(ccr_root, conn)
        result["migrated"] += _migrate_playbook_schema_json(ccr_root, conn)
        result["migrated"] += _migrate_playbook_history_json(ccr_root, conn)
        result["migrated"] += _migrate_archived_bullets_json(ccr_root, conn)
        conn.commit()
        logger.info("Phase 2 migration: %d records migrated", result["migrated"])
    except Exception as exc:
        conn.rollback()
        result["errors"].append(f"Phase 2 failed: {exc}")
        logger.error("Phase 2 migration failed: %s", exc)
    finally:
        conn.close()

    return result


def _migrate_playbook_txt(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse playbook.txt → INSERT into playbook_bullets + playbook_sections."""
    path = os.path.join(ccr_root, "playbook.txt")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        logger.warning("Could not read playbook.txt: %s", exc)
        return 0

    sections: list[str] = []
    current_section = ""
    count = 0
    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            if current_section not in sections:
                sections.append(current_section)
            continue

        m = _BULLET_RE.match(stripped)
        if m:
            bullet_id = m.group(1)
            conn.execute(
                """INSERT OR IGNORE INTO playbook_bullets
                   (id, section, content, helpful, harmful, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (bullet_id, current_section, m.group(4),
                 int(m.group(2)), int(m.group(3)), now),
            )
            count += 1

    for i, sec in enumerate(sections):
        conn.execute(
            "INSERT OR IGNORE INTO playbook_sections (position, name) VALUES (?, ?)",
            (i, sec),
        )

    if count:
        _backup_file(path)
    return count


def _migrate_failure_lessons_json(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse failure_lessons.json → INSERT into failure_lessons + UPDATE bullet extended fields."""
    path = os.path.join(ccr_root, "failure_lessons.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse failure_lessons.json: %s", exc)
        return 0

    count = 0
    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()

    for bid, entry in data.items():
        if isinstance(entry, list):
            lessons = entry
            ext = {}
        else:
            lessons = entry.get("lessons", [])
            ext = entry

        conn.execute(
            """UPDATE playbook_bullets SET
                scope = ?,
                trigger_text = ?,
                when_to_apply = ?,
                weighted_helpful = ?,
                weighted_harmful = ?,
                personal_decay_rate = ?
                WHERE id = ?""",
            (
                ext.get("scope", "general"),
                ext.get("trigger", ""),
                ext.get("when_to_apply", ""),
                ext.get("weighted_helpful", 0.0),
                ext.get("weighted_harmful", 0.0),
                ext.get("personal_decay_rate", 0.0),
                bid,
            ),
        )

        for lesson in lessons:
            conn.execute(
                """INSERT INTO failure_lessons
                   (bullet_id, failure_point, flawed_reasoning, counterfactual,
                    prevention_principle, task_context, timestamp, evolved)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    bid,
                    lesson.get("failure_point", ""),
                    lesson.get("flawed_reasoning", ""),
                    lesson.get("counterfactual", ""),
                    lesson.get("prevention_principle", ""),
                    lesson.get("task_context", ""),
                    lesson.get("timestamp", now),
                    int(lesson.get("evolved", False)),
                ),
            )
            count += 1

    if count or data:
        _backup_file(path)
    return count


def _migrate_playbook_schema_json(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse playbook_schema.json → INSERT into playbook_schema."""
    path = os.path.join(ccr_root, "playbook_schema.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse playbook_schema.json: %s", exc)
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    current = data.get("current", data)
    if current:
        version = current.get("version", 1)
        conn.execute(
            """INSERT OR IGNORE INTO playbook_schema
               (version, data_json, parent_version, change_description, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (version, json.dumps(current, default=str),
             current.get("parent_version"), current.get("change_description", ""), now),
        )
        count += 1

    for entry in data.get("history", []):
        v = entry.get("version", 0)
        conn.execute(
            """INSERT OR IGNORE INTO playbook_schema
               (version, data_json, parent_version, change_description, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (v, json.dumps(entry, default=str),
             entry.get("parent_version"), entry.get("change_description", ""),
             entry.get("created_at", now)),
        )
        count += 1

    if count:
        _backup_file(path)
    return count


def _migrate_playbook_history_json(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse playbook_history.json → INSERT into delta_history."""
    path = os.path.join(ccr_root, "playbook_history.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse playbook_history.json: %s", exc)
        return 0

    if not isinstance(history, list):
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for entry in history:
        conn.execute(
            """INSERT INTO delta_history
               (timestamp, author, ops_count, applied_count, scope,
                operations_json, failed_ids_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.get("timestamp", now),
                entry.get("author", ""),
                entry.get("ops_count", 0),
                entry.get("applied_count", 0),
                entry.get("scope", "project"),
                json.dumps(entry.get("operations", []), default=str) if entry.get("operations") else None,
                json.dumps(entry.get("failed_ids", []), default=str) if entry.get("failed_ids") else None,
            ),
        )
        count += 1

    if count:
        _backup_file(path)
    return count


def _migrate_archived_bullets_json(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse archived_bullets.json → INSERT into archived_bullets."""
    path = os.path.join(ccr_root, "archived_bullets.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            archived = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse archived_bullets.json: %s", exc)
        return 0

    if not isinstance(archived, list):
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    now = _utcnow()
    count = 0

    for b in archived:
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
                b.get("archived_at", now),
                b.get("reason", ""),
            ),
        )
        count += 1

    if count:
        _backup_file(path)
    return count
