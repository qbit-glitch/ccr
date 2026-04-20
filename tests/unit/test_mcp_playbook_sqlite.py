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
