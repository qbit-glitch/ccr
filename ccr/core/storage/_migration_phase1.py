"""Phase 1 migration: scratchpad, metrics, logs, metadata.

Flat-file sources: scratchpad.json, memory_metrics.json,
branches/*/log.md, metadata.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any

from ccr.core.storage._migration_utils import _backup_file

logger = logging.getLogger(__name__)


def migrate_phase_1(
    ccr_root: str, db_path: str,
) -> dict[str, Any]:
    """Migrate scratchpad, metrics, log, and metadata from flat files to SQLite.

    Runs inside a single transaction. Backs up source files to .bak.
    """
    result: dict[str, Any] = {"migrated": 0, "errors": []}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("BEGIN")

        result["migrated"] += _migrate_scratchpad(ccr_root, conn)
        result["migrated"] += _migrate_metrics(ccr_root, conn)
        result["migrated"] += _migrate_logs(ccr_root, conn)
        result["migrated"] += _migrate_metadata(ccr_root, conn)

        conn.commit()
        logger.info("Phase 1 migration: %d records migrated", result["migrated"])
    except Exception as exc:
        conn.rollback()
        result["errors"].append(f"Phase 1 failed: {exc}")
        logger.error("Phase 1 migration failed: %s", exc)
    finally:
        conn.close()

    return result


def _migrate_scratchpad(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse scratchpad.json → INSERT into scratchpad table."""
    path = os.path.join(ccr_root, "scratchpad.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse scratchpad.json: %s", exc)
        return 0

    entries = data.get("entries", {})
    count = 0
    for key, entry in entries.items():
        conn.execute(
            """INSERT OR IGNORE INTO scratchpad
               (key, value, created_at, updated_at, access_count, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                key,
                entry.get("value", ""),
                entry.get("created_at", ""),
                entry.get("updated_at", ""),
                entry.get("access_count", 0),
                entry.get("expires_at"),
            ),
        )
        count += 1

    if count:
        _backup_file(path)
    return count


def _migrate_metrics(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse memory_metrics.json → INSERT into metrics table."""
    path = os.path.join(ccr_root, "memory_metrics.json")
    if not os.path.isfile(path):
        return 0

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse memory_metrics.json: %s", exc)
        return 0

    count = 0
    for key, value in data.items():
        if key == "last_updated":
            continue
        if not isinstance(value, (int, float)):
            continue
        conn.execute(
            """INSERT OR IGNORE INTO metrics (key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, int(value), data.get("last_updated")),
        )
        count += 1

    if count:
        _backup_file(path)
    return count


def _migrate_logs(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse branches/*/log.md → INSERT into log_entries table."""
    branches_dir = os.path.join(ccr_root, "branches")
    if not os.path.isdir(branches_dir):
        return 0

    count = 0
    for branch_name in os.listdir(branches_dir):
        if branch_name.startswith(("_", ".")):
            continue
        log_path = os.path.join(branches_dir, branch_name, "log.md")
        if not os.path.isfile(log_path):
            continue

        try:
            with open(log_path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue

        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            conn.execute(
                "INSERT INTO log_entries (branch, line) VALUES (?, ?)",
                (branch_name, stripped),
            )
            count += 1

        if count:
            _backup_file(log_path)

    return count


def _migrate_metadata(ccr_root: str, conn: sqlite3.Connection) -> int:
    """Parse metadata.yaml → INSERT into metadata table."""
    path = os.path.join(ccr_root, "metadata.yaml")
    if not os.path.isfile(path):
        return 0

    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Could not parse metadata.yaml: %s", exc)
        return 0

    if not data:
        return 0

    from ccr.core.storage._sqlite_utils import _utcnow
    conn.execute(
        """INSERT OR IGNORE INTO metadata (key, value_json, updated_at)
           VALUES ('_root', ?, ?)""",
        (json.dumps(data, default=str), _utcnow()),
    )

    _backup_file(path)
    return 1
