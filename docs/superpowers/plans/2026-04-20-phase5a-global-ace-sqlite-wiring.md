# Phase 5a: Global ACE Playbook on SQLite — MCP Server Wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the MCP server's playbook load/save helpers to use the existing `SqliteStorageBackend` (with `_global_mgr` pointing at `~/.ccr/global.db`) via the already-implemented `Playbook.from_backend()` / `save_to_backend()` classmethods, giving cross-project concurrent-write safety for `scope="global"` bullet mutations and atomic counter updates via SQLite WAL row-level locking.

**Architecture:** Phase 2 landed `global.db` with the full `playbook_bullets` / `failure_lessons` / `playbook_sections` schema and dual-backend CRUD methods, plus `Playbook.from_backend()` / `save_to_backend()` classmethods on the Playbook class — but the MCP layer in `ccr/mcp/server.py` never called them. This phase wires `_load_playbook` / `_save_playbook` / `_load_global_playbook` / `_save_global_playbook` to go through the MemoryManager's `_storage` backend (branching on `isinstance(backend, SqliteStorageBackend)`), then adds one-shot idempotent backfill migrations that parse the legacy flat `playbook.txt` / `global_playbook.txt` into the SQLite tables behind a per-DB `PRAGMA user_version` guard (memory.db 2→3, global.db 0→1). File-backend path is unchanged (keeps reading/writing flat files via the file-phase-2 helpers).

**Tech Stack:** Python 3.12+, SQLite WAL mode, existing `SqliteConnectionManager` + `SqliteStorageBackend` from Phases 0–4, existing `FileStorageBackend`, `Playbook.from_backend/save_to_backend` (Phase 2), pytest with dual-backend parametrization.

---

## Scope & Affected Files

### Create
- `ccr/core/storage/_migration_phase5a.py` — `migrate_phase_5a(ccr_root, db_path, playbook_path, failure_lessons_path, scope)` helper (one-shot flat-file → SQLite backfill, user_version guarded, atomic per-DB)
- `tests/unit/test_mcp_playbook_sqlite.py` — dedicated MCP-layer tests (load/save round-trip + scope routing + backfill idempotency + cross-project smoke)

### Modify
- `ccr/mcp/server.py`:
  - Rewrite `_load_playbook()` to branch on backend type: `SqliteStorageBackend → Playbook.from_backend(_memory._storage, "project")`; `FileStorageBackend → existing flat-file read`
  - Rewrite `_save_playbook()` symmetrically, saving via `pb.save_to_backend(_memory._storage, "project")` when SQLite
  - Same for `_load_global_playbook()` / `_save_global_playbook()` with `scope="global"`
  - In `_init()`, after `_memory` is built and before first `_load_playbook()`, call `migrate_phase_5a(...)` once per DB
- `ccr/core/storage/migration.py`:
  - Add `from ccr.core.storage._migration_phase5a import migrate_phase_5a`
  - Add `migrate_phase_5a` to `__all__`
  - Extend `auto_migrate(...)` to call `migrate_phase_5a(ccr_root, db_path, playbook_path, failure_lessons_path, scope="project")` after phase 3c
- `ccr/core/storage/sqlite_backend.py`:
  - Extend the `__init__` migration block (just after the vec-migration check, ~line 440) to also run `migrate_phase_5a(...)` for project scope when `user_version < 3`; add a second call for the global manager when `global.db user_version < 1`
- `tests/unit/test_storage_backend.py` — add Phase 5a round-trip parity test (both backends)

**Non-goals for this phase:**
- No changes to `playbook_schema.json` persistence (still flat-file — Phase 5b candidate)
- No change to default `CCRConfig().storage_backend = "files"` — opt-in via `CCR_STORAGE_BACKEND=sqlite` stays as-is (default flip is a separate decision)
- No MCP tool signature changes (AcePlaybookResult, AceApplyDeltaResult, etc. keep their shapes)
- No changes to `delta_history.json` / `archived_bullets.json` flat fallbacks written by `ace_apply_delta` / `ace_prune` — these remain as per-scope audit logs (SQLite `delta_history` / `archived_bullets` tables exist but are only populated through `save_to_backend`; audit-log JSON mirrors are additive, not source of truth)
- No migration in the opposite direction (SQLite → flat file); downgrade path = delete global.db and let flat file resume

**Principle:** Phase 2 built the plumbing; this phase turns it on for the MCP layer. All wire-up is additive and idempotent — the flat-file path survives unchanged for `storage_backend="files"` users. Schema version on `memory.db` bumps from 2 (Phase 4 vec) to 3 (Phase 5a playbook); `global.db user_version` bumps from 0 to 1 (first migration on that DB).

**Relationship to Phase 2:** `migrate_phase_2` already backfills `.ccr/playbook.txt` → `memory.db.playbook_bullets` today via `INSERT OR IGNORE` (unguarded, re-runs on every `auto_migrate`). Phase 5a's project-scope migration is therefore mostly redundant on `memory.db` — it serves as a version-guard marker ("MCP layer now reads from SQLite") rather than a data-move. The **new** work Phase 5a does for project scope: user_version bump 2→3. The **new** work for global scope: parse `~/.ccr/global_playbook.txt` → `global.db` (Phase 2 never touched global.db). Version bump 0→1 on global.db.

---

## Task 5a.1: Add `migrate_phase_5a` helper (flat → SQLite backfill)

**Files:**
- Create: `ccr/core/storage/_migration_phase5a.py`
- Create: `tests/unit/test_mcp_playbook_sqlite.py` (test-first)

### - [ ] Step 1: Write the failing backfill test

Create `tests/unit/test_mcp_playbook_sqlite.py`:

```python
"""Phase 5a: MCP-layer wiring of ACE playbook to SqliteStorageBackend."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from ccr.core.storage._migration_phase5a import migrate_phase_5a
from ccr.core.storage.sqlite_backend import SqliteStorageBackend


FIXTURE_PLAYBOOK = """\
## STRATEGIES & INSIGHTS
[str-00001] helpful=3 harmful=0 :: When wiring MCP tools, test round-trip first.
[str-00002] helpful=1 harmful=0 :: Prefer dependency injection over globals.

## COMMON MISTAKES TO AVOID
[mis-00001] helpful=0 harmful=2 :: Mutating shared state without a lock.
"""


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    ccr = tmp_path / ".ccr"
    ccr.mkdir()
    (ccr / "playbook.txt").write_text(FIXTURE_PLAYBOOK)
    return tmp_path


def test_migrate_phase_5a_backfills_flat_playbook_into_sqlite(tmp_project: Path) -> None:
    ccr_root = str(tmp_project / ".ccr")
    db_path = os.path.join(ccr_root, "memory.db")
    playbook_path = os.path.join(ccr_root, "playbook.txt")
    failure_lessons_path = os.path.join(ccr_root, "failure_lessons.json")

    # Prime memory.db at user_version=2 (post-Phase-4) so migration triggers 2→3
    backend = SqliteStorageBackend(ccr_root)
    backend._memory_mgr.set_user_version(2)
    backend.close()

    result = migrate_phase_5a(
        ccr_root=ccr_root,
        db_path=db_path,
        playbook_path=playbook_path,
        failure_lessons_path=failure_lessons_path,
        scope="project",
    )

    assert result["migrated"] == 3  # 3 bullets in FIXTURE_PLAYBOOK
    assert result["version_before"] == 2
    assert result["version_after"] == 3

    # Verify bullets landed in SQLite via re-open
    backend = SqliteStorageBackend(ccr_root)
    try:
        bullets = backend.bullet_list(scope="project")
        assert {b["id"] for b in bullets} == {"str-00001", "str-00002", "mis-00001"}
        assert backend._memory_mgr.get_user_version() == 3
    finally:
        backend.close()


def test_migrate_phase_5a_is_idempotent(tmp_project: Path) -> None:
    """Second run is a no-op — user_version already >= 3."""
    ccr_root = str(tmp_project / ".ccr")
    db_path = os.path.join(ccr_root, "memory.db")
    playbook_path = os.path.join(ccr_root, "playbook.txt")
    failure_lessons_path = os.path.join(ccr_root, "failure_lessons.json")

    backend = SqliteStorageBackend(ccr_root)
    backend._memory_mgr.set_user_version(2)
    backend.close()

    migrate_phase_5a(ccr_root, db_path, playbook_path, failure_lessons_path, scope="project")
    second = migrate_phase_5a(ccr_root, db_path, playbook_path, failure_lessons_path, scope="project")

    assert second["migrated"] == 0
    assert second["skipped"] is True


def test_migrate_phase_5a_no_flat_file_is_noop(tmp_path: Path) -> None:
    """When playbook.txt doesn't exist, migrate cleanly with no bullets."""
    ccr_root = str(tmp_path / ".ccr")
    Path(ccr_root).mkdir()
    db_path = os.path.join(ccr_root, "memory.db")
    playbook_path = os.path.join(ccr_root, "playbook.txt")  # missing
    failure_lessons_path = os.path.join(ccr_root, "failure_lessons.json")

    backend = SqliteStorageBackend(ccr_root)
    backend._memory_mgr.set_user_version(2)
    backend.close()

    result = migrate_phase_5a(ccr_root, db_path, playbook_path, failure_lessons_path, scope="project")
    assert result["migrated"] == 0
    assert result["version_after"] == 3  # still bumps version to mark migration done


def test_migrate_phase_5a_atomic_on_crash(tmp_project: Path, monkeypatch) -> None:
    """If backfill throws mid-way, user_version stays at 2 and a re-run redoes it cleanly."""
    ccr_root = str(tmp_project / ".ccr")
    db_path = os.path.join(ccr_root, "memory.db")
    playbook_path = os.path.join(ccr_root, "playbook.txt")
    failure_lessons_path = os.path.join(ccr_root, "failure_lessons.json")

    backend = SqliteStorageBackend(ccr_root)
    backend._memory_mgr.set_user_version(2)
    backend.close()

    # Monkey-patch Playbook._parse to throw on second bullet
    import ccr.ace.playbook as pbmod
    original = pbmod.Playbook._parse
    call_count = {"n": 0}

    def flaky_parse(self, text):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated crash")
        return original(self, text)

    monkeypatch.setattr(pbmod.Playbook, "_parse", flaky_parse)

    with pytest.raises(RuntimeError):
        migrate_phase_5a(ccr_root, db_path, playbook_path, failure_lessons_path, scope="project")

    # DB should still be at v2 — no partial rows
    backend = SqliteStorageBackend(ccr_root)
    try:
        assert backend._memory_mgr.get_user_version() == 2
        assert backend.bullet_list(scope="project") == []
    finally:
        backend.close()

    # Undo monkeypatch and re-run — should succeed
    monkeypatch.setattr(pbmod.Playbook, "_parse", original)
    result = migrate_phase_5a(ccr_root, db_path, playbook_path, failure_lessons_path, scope="project")
    assert result["migrated"] == 3
```

### - [ ] Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py::test_migrate_phase_5a_backfills_flat_playbook_into_sqlite -xvs`

Expected: FAIL with `ModuleNotFoundError: No module named 'ccr.core.storage._migration_phase5a'`

### - [ ] Step 3: Write the minimal implementation

Create `ccr/core/storage/_migration_phase5a.py`:

```python
"""Phase 5a migration: flat playbook.txt / global_playbook.txt → SQLite.

One-shot, idempotent, per-DB, user_version-guarded backfill. Re-running is a
no-op once user_version has been bumped. Wrapped in a single
`with conn:` transaction so a crash leaves user_version unchanged and the DB
unpopulated (atomic — same pattern as Phase 4 sqlite-vec backfill).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

from ccr.ace.playbook import Playbook

logger = logging.getLogger(__name__)

# Target user_version per scope — memory.db (project) was at 2 after Phase 4;
# global.db has never been migrated (starts at 0).
_TARGET_VERSION_PROJECT = 3
_TARGET_VERSION_GLOBAL = 1


def migrate_phase_5a(
    ccr_root: str,
    db_path: str,
    playbook_path: str,
    failure_lessons_path: str,
    scope: str = "project",
) -> dict:
    """Backfill the flat playbook.txt into SQLite playbook_bullets/sections/failure_lessons.

    Idempotent: If `PRAGMA user_version` is already at the target, returns
    `{"migrated": 0, "skipped": True}`. Atomic: backfill + version bump happen
    inside a single `with conn:` transaction.

    Args:
        ccr_root: The .ccr/ directory (project) or ~/.ccr/ (global).
        db_path: Full path to memory.db (project) or global.db (global).
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

    # Local import to avoid cycle — sqlite_backend imports from migration
    from ccr.core.storage.sqlite_backend import SqliteStorageBackend

    # For global scope, the backend's ccr_root IS the global root — caller passes
    # global_ccr_root (e.g. "~/.ccr") as `ccr_root`. We don't want the backend to
    # also create a project memory.db in that dir, so only construct with
    # global_ccr_root when scope == "global".
    if scope == "project":
        backend = SqliteStorageBackend(ccr_root)
        mgr = backend._memory_mgr
        conn = backend.memory_conn
    else:
        # For global-scope migration, we open the global.db directly via a
        # dedicated backend pointing at that root. SqliteStorageBackend(ccr_root)
        # creates a memory.db at `{ccr_root}/memory.db` — in global case that
        # path IS {global_ccr}/memory.db which is fine (unused by the playbook
        # tables — they live on memory_conn's DB which the test body targets).
        # Simpler: instantiate with global_ccr_root=ccr_root so _global_mgr is
        # set; the "playbook_bullets" insert will route to the global_conn.
        backend = SqliteStorageBackend(ccr_root, global_ccr_root=ccr_root)
        mgr = backend._global_mgr
        conn = backend.global_conn

    try:
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
    finally:
        backend.close()
```

### - [ ] Step 4: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py -xvs -k "migrate_phase_5a"`

Expected: 4 passed (all four `test_migrate_phase_5a_*` tests).

### - [ ] Step 5: Commit

```bash
git add ccr/core/storage/_migration_phase5a.py tests/unit/test_mcp_playbook_sqlite.py
git commit -m "feat(storage): add Phase 5a migration — flat playbook.txt → SQLite backfill"
```

---

## Task 5a.2: Register migration in `migration.py` + `SqliteStorageBackend.__init__`

**Files:**
- Modify: `ccr/core/storage/migration.py`
- Modify: `ccr/core/storage/sqlite_backend.py` (inside `__init__`, after vec migration)

### - [ ] Step 1: Write the failing auto-migration test

Append to `tests/unit/test_mcp_playbook_sqlite.py`:

```python
def test_sqlite_backend_auto_runs_phase5a_on_init(tmp_project: Path) -> None:
    """Opening SqliteStorageBackend triggers phase 5a when memory.db is at v<3."""
    ccr_root = str(tmp_project / ".ccr")

    # Pre-condition: memory.db at v2 (Phase 4 baseline), flat playbook has 3 bullets
    backend = SqliteStorageBackend(ccr_root)
    backend._memory_mgr.set_user_version(2)
    assert backend.bullet_list(scope="project") == []
    backend.close()

    # Act: re-open — auto-migration should run
    backend = SqliteStorageBackend(ccr_root)
    try:
        assert backend._memory_mgr.get_user_version() == 3
        bullets = backend.bullet_list(scope="project")
        assert len(bullets) == 3
        assert {b["id"] for b in bullets} == {"str-00001", "str-00002", "mis-00001"}
    finally:
        backend.close()
```

### - [ ] Step 2: Run to verify it fails

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py::test_sqlite_backend_auto_runs_phase5a_on_init -xvs`

Expected: FAIL — `backend._memory_mgr.get_user_version()` returns 2 (migration not wired).

### - [ ] Step 3: Wire migration into `SqliteStorageBackend.__init__`

Open `ccr/core/storage/sqlite_backend.py`. Find the existing Phase 4 `user_version` migration block (grep for `backfill_vec` inside `__init__`) — the new block goes directly after it.

Add this block (adjust line numbers if the surrounding code has drifted; the semantic insertion point is "after all phase-N migrations for memory.db, still inside __init__, before returning"):

```python
        # ── Phase 5a: playbook flat-file → SQLite backfill (memory.db) ──────
        try:
            if self._memory_mgr.get_user_version() < 3:
                from ccr.core.storage._migration_phase5a import migrate_phase_5a
                playbook_path = os.path.join(self.ccr_root, "playbook.txt")
                failure_lessons_path = os.path.join(self.ccr_root, "failure_lessons.json")
                migrate_phase_5a(
                    ccr_root=self.ccr_root,
                    db_path=self._db_path,
                    playbook_path=playbook_path,
                    failure_lessons_path=failure_lessons_path,
                    scope="project",
                )
        except Exception as exc:
            logger.warning("Phase 5a project migration failed: %s", exc)

        # ── Phase 5a: global.db playbook backfill ───────────────────────────
        if self._global_mgr is not None and self.global_ccr_root:
            try:
                global_version = self._global_mgr.get_user_version()
                if global_version < 1:
                    from ccr.core.storage._migration_phase5a import migrate_phase_5a
                    global_playbook_path = os.path.join(
                        self.global_ccr_root, "global_playbook.txt"
                    )
                    global_fl_path = os.path.join(
                        self.global_ccr_root, "global_failure_lessons.json"
                    )
                    migrate_phase_5a(
                        ccr_root=self.global_ccr_root,
                        db_path=os.path.join(self.global_ccr_root, "global.db"),
                        playbook_path=global_playbook_path,
                        failure_lessons_path=global_fl_path,
                        scope="global",
                    )
            except Exception as exc:
                logger.warning("Phase 5a global migration failed: %s", exc)
```

**Important:** verify that `self.ccr_root`, `self.db_path`, `self._global_mgr`, and `self._global_ccr_root` attrs all exist on `SqliteStorageBackend` at the insertion point. If any attribute name differs (e.g. `_ccr_root`), update the references accordingly. Grep `ccr/core/storage/sqlite_backend.py` for `self\\.ccr_root` to confirm.

### - [ ] Step 4: Register in `migration.auto_migrate`

Open `ccr/core/storage/migration.py`. After the Phase 3c block in `auto_migrate(...)`, add:

```python
    # ── Phase 5a: playbook flat-file → SQLite ──────────────────────────────
    try:
        from ccr.core.storage._migration_phase5a import migrate_phase_5a
        playbook_path = os.path.join(ccr_root, "playbook.txt")
        failure_lessons_path = os.path.join(ccr_root, "failure_lessons.json")
        p5a = migrate_phase_5a(
            ccr_root=ccr_root,
            db_path=db_path,
            playbook_path=playbook_path,
            failure_lessons_path=failure_lessons_path,
            scope="project",
        )
        if not p5a.get("skipped"):
            result["phases_run"].append("5a")
            result["total_migrated"] += p5a["migrated"]
    except Exception as exc:
        result["errors"].append(f"phase5a: {exc}")
```

And add the import at the top of the file (after `from ccr.core.storage._migration_phase3 import migrate_phase_3a, ...`):

```python
from ccr.core.storage._migration_phase5a import migrate_phase_5a
```

Add `"migrate_phase_5a"` to `__all__`.

### - [ ] Step 5: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py::test_sqlite_backend_auto_runs_phase5a_on_init -xvs`

Expected: PASS.

Also run the full migration test suite to check for regressions:

Run: `.venv/bin/python -m pytest tests/unit/test_migration.py -xvs`

Expected: all existing tests still pass; no new failures.

### - [ ] Step 6: Commit

```bash
git add ccr/core/storage/sqlite_backend.py ccr/core/storage/migration.py tests/unit/test_mcp_playbook_sqlite.py
git commit -m "feat(storage): auto-run Phase 5a migration on SqliteStorageBackend init"
```

---

## Task 5a.3: Rewrite `_load_playbook` / `_save_playbook` to route through backend

**Files:**
- Modify: `ccr/mcp/server.py:327-346` (both helpers)

### - [ ] Step 1: Write the failing round-trip test

Append to `tests/unit/test_mcp_playbook_sqlite.py`:

```python
def test_load_save_playbook_roundtrip_via_sqlite(tmp_project: Path, monkeypatch) -> None:
    """_save_playbook writes to SQLite when backend is SqliteStorageBackend; _load_playbook reads from it."""
    monkeypatch.setenv("CCR_STORAGE_BACKEND", "sqlite")

    import ccr.mcp.server as srv
    # Fresh _init on the tmp project root
    srv._init(str(tmp_project))

    try:
        pb = srv._ensure_playbook()
        initial_ids = {b.id for b in pb.bullets}
        assert initial_ids == {"str-00001", "str-00002", "mis-00001"}

        # Mutate: add a new bullet via apply_delta (goes through _save_playbook)
        from ccr.ace.playbook import DeltaOperation
        op = DeltaOperation(
            op_type="ADD",
            section="STRATEGIES & INSIGHTS",
            content="Phase 5a round-trip marker",
        )
        pb.apply_delta([op])
        srv._save_playbook()

        # Directly query SQLite — the new bullet must be visible
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        backend = SqliteStorageBackend(str(tmp_project / ".ccr"))
        try:
            bullets = backend.bullet_list(scope="project")
            assert any("round-trip marker" in b["content"] for b in bullets)
        finally:
            backend.close()

        # Re-run _load_playbook and confirm the bullet survives
        reloaded = srv._load_playbook()
        assert any("round-trip marker" in b.content for b in reloaded.bullets)
    finally:
        # Cleanup to avoid leaking state into other tests
        if srv._memory and hasattr(srv._memory, "_storage"):
            try:
                srv._memory._storage.close()
            except Exception:
                pass
```

### - [ ] Step 2: Run to verify it fails

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py::test_load_save_playbook_roundtrip_via_sqlite -xvs`

Expected: FAIL — new bullet is not visible when directly querying SQLite because `_save_playbook` still writes only to flat file.

### - [ ] Step 3: Rewrite the helpers in `ccr/mcp/server.py`

Open `ccr/mcp/server.py`. Replace the existing `_load_playbook` (lines ~327-337) and `_save_playbook` (lines ~340-346) bodies.

Existing `_load_playbook`:
```python
def _load_playbook() -> Playbook:
    """Load playbook from disk or create empty. Also loads failure lessons."""
    if os.path.isfile(_playbook_path):
        with open(_playbook_path, "r", encoding="utf-8") as f:
            pb = Playbook(f.read())
    else:
        pb = Playbook()
    # Load structured failure lessons from companion JSON
    if _failure_lessons_path:
        pb.load_failure_lessons(_failure_lessons_path)
    return pb
```

Replace with:
```python
def _load_playbook() -> Playbook:
    """Load playbook. Prefer SQLite backend when available; else flat file."""
    # SQLite path: hydrate from memory.db playbook_bullets/sections/failure_lessons
    if _memory is not None:
        storage = getattr(_memory, "_storage", None)
        if storage is not None:
            from ccr.core.storage.sqlite_backend import SqliteStorageBackend
            if isinstance(storage, SqliteStorageBackend):
                try:
                    return Playbook.from_backend(storage, scope="project")
                except Exception as exc:
                    logger.warning(
                        "SQLite playbook load failed, falling back to flat file: %s", exc,
                    )

    # Flat-file path (default backend or SQLite error fallback)
    if os.path.isfile(_playbook_path):
        with open(_playbook_path, "r", encoding="utf-8") as f:
            pb = Playbook(f.read())
    else:
        pb = Playbook()
    if _failure_lessons_path:
        pb.load_failure_lessons(_failure_lessons_path)
    return pb
```

Existing `_save_playbook`:
```python
def _save_playbook() -> None:
    """Persist playbook and failure lessons to disk (H5: atomic writes)."""
    if _playbook is not None:
        _atomic_write(_playbook_path, _playbook.serialize())
        # Save failure lessons to companion JSON
        if _failure_lessons_path:
            _playbook.save_failure_lessons(_failure_lessons_path)
```

Replace with:
```python
def _save_playbook() -> None:
    """Persist playbook. Prefer SQLite backend when available; else flat file."""
    if _playbook is None:
        return

    # SQLite path
    if _memory is not None:
        storage = getattr(_memory, "_storage", None)
        if storage is not None:
            from ccr.core.storage.sqlite_backend import SqliteStorageBackend
            if isinstance(storage, SqliteStorageBackend):
                try:
                    _playbook.save_to_backend(storage, scope="project")
                    return
                except Exception as exc:
                    logger.warning(
                        "SQLite playbook save failed, falling back to flat file: %s", exc,
                    )

    # Flat-file path (default backend or SQLite error fallback)
    _atomic_write(_playbook_path, _playbook.serialize())
    if _failure_lessons_path:
        _playbook.save_failure_lessons(_failure_lessons_path)
```

### - [ ] Step 4: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py::test_load_save_playbook_roundtrip_via_sqlite -xvs`

Expected: PASS.

### - [ ] Step 5: Run the full MCP-server test suite to catch regressions

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_server.py -x -q`

Expected: all existing MCP tests still pass.

### - [ ] Step 6: Commit

```bash
git add ccr/mcp/server.py tests/unit/test_mcp_playbook_sqlite.py
git commit -m "feat(mcp): route _load_playbook/_save_playbook through SqliteStorageBackend when available"
```

---

## Task 5a.4: Rewrite `_load_global_playbook` / `_save_global_playbook`

**Files:**
- Modify: `ccr/mcp/server.py:349-366` (both helpers)

### - [ ] Step 1: Write the failing cross-project test

Append to `tests/unit/test_mcp_playbook_sqlite.py`:

```python
def test_global_playbook_shared_across_projects_via_sqlite(tmp_path: Path, monkeypatch) -> None:
    """Adding a global bullet in project A is visible in project B via ~/.ccr/global.db."""
    monkeypatch.setenv("CCR_STORAGE_BACKEND", "sqlite")

    # Redirect ~/.ccr/ to a tmp dir so we don't pollute the real user dir
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    project_a = tmp_path / "proj_a"
    project_b = tmp_path / "proj_b"
    for p in (project_a, project_b):
        (p / ".ccr").mkdir(parents=True)

    import ccr.mcp.server as srv

    # Session 1: open project A, add a global bullet
    srv._init(str(project_a))
    gpb = srv._ensure_global_playbook()
    from ccr.ace.playbook import DeltaOperation
    op = DeltaOperation(
        op_type="ADD",
        section="STRATEGIES & INSIGHTS",
        content="Cross-project marker from proj_a",
    )
    gpb.apply_delta([op])
    srv._save_global_playbook()
    if srv._memory and hasattr(srv._memory, "_storage"):
        srv._memory._storage.close()

    # Reset module globals (simulating a new Claude Code session on proj_b)
    srv._memory = None
    srv._playbook = None
    srv._global_playbook = None

    # Session 2: open project B
    srv._init(str(project_b))
    gpb_b = srv._ensure_global_playbook()

    try:
        assert any("proj_a" in b.content for b in gpb_b.bullets), (
            "Global bullet from proj_a should be visible in proj_b session"
        )
    finally:
        if srv._memory and hasattr(srv._memory, "_storage"):
            srv._memory._storage.close()
```

### - [ ] Step 2: Run to verify it fails

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py::test_global_playbook_shared_across_projects_via_sqlite -xvs`

Expected: FAIL — bullet added in session 1 writes to `global_playbook.txt` under proj_a's view of ~/.ccr/, not the shared SQLite table.

### - [ ] Step 3: Rewrite the global helpers

In `ccr/mcp/server.py`, replace `_load_global_playbook` (lines ~349-358):

```python
def _load_global_playbook() -> Playbook:
    """Load global playbook. Prefer SQLite backend when available; else flat file."""
    if _memory is not None:
        storage = getattr(_memory, "_storage", None)
        if storage is not None:
            from ccr.core.storage.sqlite_backend import SqliteStorageBackend
            if isinstance(storage, SqliteStorageBackend):
                try:
                    return Playbook.from_backend(storage, scope="global")
                except Exception as exc:
                    logger.warning(
                        "SQLite global-playbook load failed, falling back to flat file: %s", exc,
                    )

    # Flat-file path
    if os.path.isfile(_global_playbook_path):
        with open(_global_playbook_path, "r", encoding="utf-8") as f:
            pb = Playbook(f.read())
    else:
        pb = Playbook()
    if _global_failure_lessons_path:
        pb.load_failure_lessons(_global_failure_lessons_path)
    return pb
```

And `_save_global_playbook` (lines ~361-366):

```python
def _save_global_playbook() -> None:
    """Persist global playbook. Prefer SQLite backend when available; else flat file."""
    if _global_playbook is None:
        return

    if _memory is not None:
        storage = getattr(_memory, "_storage", None)
        if storage is not None:
            from ccr.core.storage.sqlite_backend import SqliteStorageBackend
            if isinstance(storage, SqliteStorageBackend):
                try:
                    _global_playbook.save_to_backend(storage, scope="global")
                    return
                except Exception as exc:
                    logger.warning(
                        "SQLite global-playbook save failed, falling back to flat file: %s", exc,
                    )

    _atomic_write(_global_playbook_path, _global_playbook.serialize())
    if _global_failure_lessons_path:
        _global_playbook.save_failure_lessons(_global_failure_lessons_path)
```

### - [ ] Step 4: Run the cross-project test to verify it passes

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py::test_global_playbook_shared_across_projects_via_sqlite -xvs`

Expected: PASS.

### - [ ] Step 5: Commit

```bash
git add ccr/mcp/server.py tests/unit/test_mcp_playbook_sqlite.py
git commit -m "feat(mcp): route global playbook I/O through SqliteStorageBackend for cross-project sharing"
```

---

## Task 5a.5: Dual-backend parity test (file vs sqlite round-trip)

**Files:**
- Modify: `tests/unit/test_storage_backend.py`

### - [ ] Step 1: Add the parity test

Open `tests/unit/test_storage_backend.py` and append:

```python
def test_phase5a_playbook_roundtrip_parity_both_backends(tmp_path):
    """Playbook round-trip through from_backend/save_to_backend behaves identically on both backends."""
    from ccr.ace.playbook import Playbook, DeltaOperation
    from ccr.core.storage.file_backend import FileStorageBackend
    from ccr.core.storage.sqlite_backend import SqliteStorageBackend

    ccr_file = tmp_path / "file_ccr"
    ccr_sqlite = tmp_path / "sqlite_ccr"
    ccr_file.mkdir()
    ccr_sqlite.mkdir()

    backends = [
        ("file", FileStorageBackend(str(ccr_file))),
        ("sqlite", SqliteStorageBackend(str(ccr_sqlite))),
    ]

    try:
        for label, backend in backends:
            pb = Playbook()
            pb.apply_delta([
                DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content=f"{label} bullet 1"),
                DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content=f"{label} bullet 2"),
            ])
            pb.save_to_backend(backend, scope="project")
            reloaded = Playbook.from_backend(backend, scope="project")

            assert len(reloaded.bullets) == 2
            assert {b.content for b in reloaded.bullets} == {f"{label} bullet 1", f"{label} bullet 2"}
    finally:
        for _, backend in backends:
            try:
                backend.close()
            except Exception:
                pass
```

### - [ ] Step 2: Run the parity test

Run: `.venv/bin/python -m pytest tests/unit/test_storage_backend.py::test_phase5a_playbook_roundtrip_parity_both_backends -xvs`

Expected: PASS — both backends must produce identical `{"label bullet 1", "label bullet 2"}` sets.

### - [ ] Step 3: Commit

```bash
git add tests/unit/test_storage_backend.py
git commit -m "test(storage): add Phase 5a dual-backend playbook round-trip parity"
```

---

## Task 5a.6: Idempotent re-init — existing SQLite data is not clobbered by flat-file backfill

**Files:**
- Append to `tests/unit/test_mcp_playbook_sqlite.py` (no code change — regression test)

### - [ ] Step 1: Write the test

```python
def test_phase5a_skips_backfill_when_sqlite_already_populated(tmp_project: Path) -> None:
    """If SQLite already has bullets (user_version >= 3), migrate is a no-op even if flat file still exists."""
    ccr_root = str(tmp_project / ".ccr")
    db_path = os.path.join(ccr_root, "memory.db")
    playbook_path = os.path.join(ccr_root, "playbook.txt")
    failure_lessons_path = os.path.join(ccr_root, "failure_lessons.json")

    # Seed SQLite with a different set of bullets, then bump version to 3
    backend = SqliteStorageBackend(ccr_root)
    try:
        backend._memory_mgr.set_user_version(2)
        from ccr.ace.playbook import Playbook, DeltaOperation
        pb = Playbook()
        pb.apply_delta([
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="sqlite-only content"),
        ])
        pb.save_to_backend(backend, scope="project")
        backend._memory_mgr.set_user_version(3)
    finally:
        backend.close()

    # Run migration — should skip (version already at target)
    result = migrate_phase_5a(ccr_root, db_path, playbook_path, failure_lessons_path, scope="project")
    assert result["skipped"] is True
    assert result["migrated"] == 0

    # Verify the SQLite-only content survives; flat-file bullets were NOT injected
    backend = SqliteStorageBackend(ccr_root)
    try:
        bullets = backend.bullet_list(scope="project")
        contents = {b["content"] for b in bullets}
        assert "sqlite-only content" in contents
        assert not any("str-00001" in c for c in contents), "flat-file bullets should not be re-injected"
    finally:
        backend.close()
```

### - [ ] Step 2: Run to verify it passes (should already be green from Task 5a.1 design)

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py::test_phase5a_skips_backfill_when_sqlite_already_populated -xvs`

Expected: PASS.

### - [ ] Step 3: Commit

```bash
git add tests/unit/test_mcp_playbook_sqlite.py
git commit -m "test(storage): Phase 5a skip-backfill regression test"
```

---

## Task 5a.7: End-to-end MCP-tool smoke test (ace_apply_delta round-trip via SQLite)

**Files:**
- Append to `tests/unit/test_mcp_playbook_sqlite.py`

### - [ ] Step 1: Write the smoke test

```python
def test_ace_apply_delta_persists_via_sqlite_and_survives_reinit(tmp_project: Path, monkeypatch) -> None:
    """ace_apply_delta writes bullets to SQLite; re-init reads them back."""
    monkeypatch.setenv("CCR_STORAGE_BACKEND", "sqlite")

    import ccr.mcp.server as srv
    from ccr.mcp.ace_tools import ace_apply_delta

    srv._init(str(tmp_project))
    try:
        result = ace_apply_delta(
            operations=[
                {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "MCP smoke bullet"},
            ],
            scope="project",
        )
        assert result.applied == 1

        # Directly query SQLite
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        backend = SqliteStorageBackend(str(tmp_project / ".ccr"))
        try:
            bullets = backend.bullet_list(scope="project")
            assert any("MCP smoke bullet" in b["content"] for b in bullets)
        finally:
            backend.close()
    finally:
        if srv._memory and hasattr(srv._memory, "_storage"):
            srv._memory._storage.close()

    # Reset globals and re-init — bullet should still be loaded
    srv._memory = None
    srv._playbook = None
    srv._global_playbook = None
    srv._init(str(tmp_project))
    try:
        pb = srv._ensure_playbook()
        assert any("MCP smoke bullet" in b.content for b in pb.bullets)
    finally:
        if srv._memory and hasattr(srv._memory, "_storage"):
            srv._memory._storage.close()
```

### - [ ] Step 2: Run the smoke test

Run: `.venv/bin/python -m pytest tests/unit/test_mcp_playbook_sqlite.py::test_ace_apply_delta_persists_via_sqlite_and_survives_reinit -xvs`

Expected: PASS.

### - [ ] Step 3: Commit

```bash
git add tests/unit/test_mcp_playbook_sqlite.py
git commit -m "test(mcp): end-to-end ace_apply_delta round-trip via SqliteStorageBackend"
```

---

## Task 5a.8: Full-suite regression run + live-data smoke

### - [ ] Step 1: Run the full test suite

Run: `.venv/bin/python -m pytest tests/unit/ tests/integration/ -x -q`

Expected: ~2860+ passed / 2 skipped. Baseline before this plan was 2853 passed / 2 skipped (from Phase 4 C112); expected new count: baseline + 8 new tests from this plan = 2861 passed.

If any pre-existing tests fail, investigate — most likely suspects are `tests/unit/test_mcp_server.py` tests that construct a Playbook from a fixture flat file (they should still work because the SQLite path is only taken when backend is `SqliteStorageBackend` and the test harness typically uses `FileStorageBackend` via default `CCRConfig`).

### - [ ] Step 2: Live-data smoke test on the real project

With the current repo (env var set):

```bash
CCR_STORAGE_BACKEND=sqlite .venv/bin/python -c "
import ccr.mcp.server as srv
srv._init('.')
pb = srv._ensure_playbook()
gpb = srv._ensure_global_playbook()
print(f'project bullets: {len(pb.bullets)}')
print(f'global bullets:  {len(gpb.bullets)}')

import sqlite3
for label, path in [('memory.db', '.ccr/memory.db'), ('global.db', '/Users/qbit-glitch/.ccr/global.db')]:
    conn = sqlite3.connect(path)
    v = conn.execute('PRAGMA user_version').fetchone()[0]
    c = conn.execute('SELECT COUNT(*) FROM playbook_bullets').fetchone()[0]
    print(f'{label}: user_version={v} rows={c}')
    conn.close()
"
```

Expected output sketch (exact numbers depend on live state):
```
project bullets: 38   # or however many are currently in the .ccr/playbook.txt / memory.db
global bullets:  0    # or whatever ~/.ccr/global_playbook.txt has
memory.db: user_version=3 rows=38
global.db: user_version=1 rows=0
```

If `user_version` is still 2 on memory.db or 0 on global.db, the migration didn't run — investigate the `SqliteStorageBackend.__init__` migration block (Task 5a.2 Step 3).

### - [ ] Step 3: Commit

```bash
git add -A
git commit -m "ccr: Phase 5a complete — global ACE playbook on SQLite with cross-project sharing"
```

---

## Task 5a.9: Documentation update

**Files:**
- Modify: `CLAUDE.md` (project instructions)

### - [ ] Step 1: Update the Key Patterns section

In `CLAUDE.md`, under "Key Patterns", update the two-tier playbook line. Existing:

```markdown
- **Two-tier playbook**: Global (~/.ccr/) + project (.ccr/); `scope="global"|"project"` on ACE tools
```

Replace with:

```markdown
- **Two-tier playbook**: Global (~/.ccr/global.db) + project (.ccr/memory.db) via SQLite; `scope="global"|"project"` on ACE tools. WAL locking makes concurrent Claude Code sessions on different projects safe for global-scope writes.
```

### - [ ] Step 2: Commit

```bash
git add CLAUDE.md
git commit -m "docs: note Phase 5a SQLite + cross-project WAL locking on two-tier playbook"
```

---

## Self-Review Checklist

After all tasks complete, verify:

1. **Spec coverage** — ✔ Every non-goal is explicitly listed; every goal has a dedicated task:
   - `_load_playbook` / `_save_playbook` → Task 5a.3
   - `_load_global_playbook` / `_save_global_playbook` → Task 5a.4
   - Backfill migration → Task 5a.1 + Task 5a.2
   - Dual-backend parity → Task 5a.5
   - Cross-project smoke → Task 5a.4 Step 1 + Task 5a.7

2. **Placeholder scan** — ✔ No TBD / TODO / "similar to" / "add error handling" — every step shows the exact code.

3. **Type consistency** — ✔ `Playbook.from_backend(backend, scope)` matches signature in `ccr/ace/playbook.py:60`. `Playbook.save_to_backend(backend, scope)` matches `ccr/ace/playbook.py:108`. `SqliteStorageBackend` / `FileStorageBackend` class names match imports. `migrate_phase_5a(ccr_root, db_path, playbook_path, failure_lessons_path, scope)` consistent across Tasks 5a.1 / 5a.2 / 5a.6. `_global_mgr` attr name matches `ccr/core/storage/sqlite_backend.py:388`.

4. **Known risks documented**:
   - `SqliteStorageBackend` constructor attributes referenced in Task 5a.2 Step 3 (`self.ccr_root`, `self.db_path`, `self._global_ccr_root`) — the implementer must verify these exact names via grep before writing the migration block. If the real names differ, substitute them.
   - Cross-project test in Task 5a.4 Step 1 uses `monkeypatch.setenv("HOME", ...)` to redirect `~/.ccr/`. If any code path resolves `~/.ccr/` via `pwd.getpwuid(os.getuid()).pw_dir` instead of `os.path.expanduser("~")`, the test may still touch the real user dir. Mitigation: use `tmp_path` + patch `os.path.expanduser` directly if HOME env isn't enough.
   - The flat-file `playbook_history.json` / `archived_bullets.json` audit logs are written separately in `ace_apply_delta` / `ace_prune` via `_get_playbook_dir(scope)`. These remain on-disk JSON in `~/.ccr/` or `.ccr/` (not SQLite) — intentional non-goal. Users with mixed backends (SQLite for bullets, JSON for audit) should understand the split.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-20-phase5a-global-ace-sqlite-wiring.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
