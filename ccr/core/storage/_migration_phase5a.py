"""Phase 5a migration: flat playbook.txt / global_playbook.txt → SQLite.

One-shot, idempotent, per-DB, user_version-guarded backfill. Re-running is a
no-op once user_version has been bumped. Wrapped in a single
`with conn:` transaction so a crash leaves user_version unchanged and the DB
unpopulated (atomic — same pattern as Phase 4 sqlite-vec backfill).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ccr.ace.playbook import Playbook

if TYPE_CHECKING:
    from ccr.core.storage.sqlite_backend import SqliteStorageBackend

# Target user_version per scope — memory.db (project) was at 2 after Phase 4;
# global.db has never been migrated (starts at 0).
_TARGET_VERSION_PROJECT = 3
_TARGET_VERSION_GLOBAL = 1


def migrate_phase_5a(
    backend: "SqliteStorageBackend",
    playbook_path: str,
    failure_lessons_path: str,
    scope: str = "project",
) -> dict:
    """Backfill the flat playbook.txt into SQLite playbook_bullets/sections/failure_lessons.

    Idempotent: If `PRAGMA user_version` is already at the target, returns
    `{"migrated": 0, "skipped": True}`. Atomic: backfill + version bump happen
    inside a single `with conn:` transaction.

    The caller owns the `backend` lifecycle — this function does NOT close it.
    This keeps the helper safe to call from `SqliteStorageBackend.__init__`
    (passing `self`) without infinite recursion.

    Args:
        backend: A fully-constructed `SqliteStorageBackend`. For `scope="global"`
            the backend must have been initialized with a `global_ccr_root`.
        playbook_path: Full path to the flat playbook.txt or global_playbook.txt.
        failure_lessons_path: Full path to failure_lessons.json (project) or
            global_failure_lessons.json (global). May not exist.
        scope: "project" or "global". Determines target user_version.

    Returns:
        dict with keys:
          - "migrated": int (number of bullets inserted)
          - "version_before": int
          - "version_after": int
          - "skipped": bool (True when user_version already at target)
          - "scope": str
    """
    if scope not in ("project", "global"):
        raise ValueError(f"scope must be 'project' or 'global', got {scope!r}")
    target_version = (
        _TARGET_VERSION_PROJECT if scope == "project" else _TARGET_VERSION_GLOBAL
    )

    if scope == "project":
        mgr = backend._memory_mgr
        conn = backend.memory_conn
    else:
        if backend._global_mgr is None or backend.global_conn is None:
            raise ValueError(
                "scope='global' requires backend initialized with global_ccr_root"
            )
        mgr = backend._global_mgr
        conn = backend.global_conn

    current_version = mgr.get_user_version()
    if current_version >= target_version:
        return {
            "migrated": 0,
            "version_before": current_version,
            "version_after": current_version,
            "skipped": True,
            "scope": scope,
        }

    # If no flat file, just bump version and exit (fresh install path)
    if not os.path.isfile(playbook_path):
        with conn:
            mgr.set_user_version(target_version)
        return {
            "migrated": 0,
            "version_before": current_version,
            "version_after": target_version,
            "skipped": False,
            "scope": scope,
        }

    # Parse flat playbook.txt into a Playbook instance
    with open(playbook_path, "r", encoding="utf-8") as f:
        pb = Playbook(f.read())

    # Load failure lessons from companion JSON if present
    if os.path.isfile(failure_lessons_path):
        pb.load_failure_lessons(failure_lessons_path)

    migrated_count = len(pb.bullets)

    # Atomic: save_to_backend + user_version bump in one txn
    with conn:
        pb.save_to_backend(backend, scope=scope)
        mgr.set_user_version(target_version)

    return {
        "migrated": migrated_count,
        "version_before": current_version,
        "version_after": target_version,
        "skipped": False,
        "scope": scope,
    }
