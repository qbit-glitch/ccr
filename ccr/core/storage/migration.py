"""Migration functions for flat-file → SQLite conversion.

Each phase has its own migration function. The auto_migrate orchestrator
detects which phases are needed and runs them in order.

Migration safety:
  - Runs inside a single transaction (all or nothing)
  - Flat files renamed to .bak (never deleted)
  - Idempotent: running twice produces no duplicates
  - .ccr/.migrated sentinel prevents re-migration

Implementation split:
  _migration_utils.py  — _write_sentinel, _backup_file
  _migration_phase1.py — migrate_phase_1 + helpers
  _migration_phase2.py — migrate_phase_2 + helpers
  _migration_phase3.py — migrate_phase_3a/3b/3c + helpers
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ccr.core.storage._migration_phase1 import migrate_phase_1
from ccr.core.storage._migration_phase2 import migrate_phase_2
from ccr.core.storage._migration_phase3 import (
    migrate_phase_3a,
    migrate_phase_3b,
    migrate_phase_3c,
)
from ccr.core.storage._migration_phase5a import migrate_phase_5a
from ccr.core.storage._migration_utils import _backup_file, _write_sentinel

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 3

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "needs_migration",
    "auto_migrate",
    "migrate_phase_1",
    "migrate_phase_2",
    "migrate_phase_3a",
    "migrate_phase_3b",
    "migrate_phase_3c",
    "migrate_phase_5a",
    "_backup_file",
    "_write_sentinel",
]


def needs_migration(ccr_root: str) -> bool:
    """Check if flat files exist that haven't been migrated."""
    sentinel = os.path.join(ccr_root, ".migrated")
    if os.path.isfile(sentinel):
        return False
    metadata = os.path.join(ccr_root, "metadata.yaml")
    return os.path.isfile(metadata)


def auto_migrate(ccr_root: str, db_path: str) -> dict[str, Any]:
    """Run all needed migration phases.

    Returns: {"phases_run": list[int], "total_migrated": int, "errors": list[str]}
    """
    result: dict[str, Any] = {"phases_run": [], "total_migrated": 0, "errors": []}

    if not needs_migration(ccr_root):
        return result

    p1 = migrate_phase_1(ccr_root, db_path)
    result["phases_run"].append(1)
    result["total_migrated"] += p1["migrated"]
    result["errors"].extend(p1["errors"])

    p2 = migrate_phase_2(ccr_root, db_path)
    result["phases_run"].append(2)
    result["total_migrated"] += p2["migrated"]
    result["errors"].extend(p2["errors"])

    p3a = migrate_phase_3a(ccr_root, db_path)
    result["phases_run"].append("3a")
    result["total_migrated"] += p3a["migrated"]
    result["errors"].extend(p3a["errors"])

    p3b = migrate_phase_3b(ccr_root, db_path)
    result["phases_run"].append("3b")
    result["total_migrated"] += p3b["migrated"]
    result["errors"].extend(p3b["errors"])

    p3c = migrate_phase_3c(ccr_root, db_path)
    result["phases_run"].append("3c")
    result["total_migrated"] += p3c["migrated"]
    result["errors"].extend(p3c["errors"])

    # ── Phase 5a: playbook flat-file → SQLite ──────────────────────────────
    # Construct a backend locally, pass it in, close it in finally.
    try:
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        playbook_path = os.path.join(ccr_root, "playbook.txt")
        failure_lessons_path = os.path.join(ccr_root, "failure_lessons.json")
        backend = SqliteStorageBackend(ccr_root)
        try:
            p5a = migrate_phase_5a(
                backend=backend,
                playbook_path=playbook_path,
                failure_lessons_path=failure_lessons_path,
                scope="project",
            )
            if not p5a.get("skipped"):
                result["phases_run"].append("5a")
                result["total_migrated"] += p5a["migrated"]
        finally:
            backend.close()
    except Exception as exc:
        result["errors"].append(f"phase5a: {exc}")

    if not result["errors"]:
        _write_sentinel(ccr_root)

    logger.info("Auto-migration complete: %s", result)
    return result
