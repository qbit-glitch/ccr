"""Phase 5a: MCP-layer wiring of ACE playbook to SqliteStorageBackend."""
from __future__ import annotations

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
    playbook_path = str(tmp_project / ".ccr" / "playbook.txt")
    failure_lessons_path = str(tmp_project / ".ccr" / "failure_lessons.json")

    backend = SqliteStorageBackend(ccr_root)
    backend._memory_mgr.set_user_version(2)
    try:
        result = migrate_phase_5a(
            backend=backend,
            playbook_path=playbook_path,
            failure_lessons_path=failure_lessons_path,
            scope="project",
        )

        assert result["migrated"] == 3  # 3 bullets in FIXTURE_PLAYBOOK
        assert result["version_before"] == 2
        assert result["version_after"] == 3
        assert result["scope"] == "project"

        # Query the same backend that did the migration — no close/reopen dance.
        bullets = backend.bullet_list(scope="project")
        assert {b["id"] for b in bullets} == {"str-00001", "str-00002", "mis-00001"}
        assert backend._memory_mgr.get_user_version() == 3
    finally:
        backend.close()


def test_migrate_phase_5a_is_idempotent(tmp_project: Path) -> None:
    """Second run is a no-op — user_version already >= 3."""
    ccr_root = str(tmp_project / ".ccr")
    playbook_path = str(tmp_project / ".ccr" / "playbook.txt")
    failure_lessons_path = str(tmp_project / ".ccr" / "failure_lessons.json")

    backend = SqliteStorageBackend(ccr_root)
    backend._memory_mgr.set_user_version(2)
    try:
        first = migrate_phase_5a(
            backend=backend,
            playbook_path=playbook_path,
            failure_lessons_path=failure_lessons_path,
            scope="project",
        )
        assert first["migrated"] == 3

        second = migrate_phase_5a(
            backend=backend,
            playbook_path=playbook_path,
            failure_lessons_path=failure_lessons_path,
            scope="project",
        )
        assert second["migrated"] == 0
        assert second["skipped"] is True
    finally:
        backend.close()


def test_migrate_phase_5a_no_flat_file_is_noop(tmp_path: Path) -> None:
    """When playbook.txt doesn't exist, migrate cleanly with no bullets."""
    ccr_root = str(tmp_path / ".ccr")
    Path(ccr_root).mkdir()
    playbook_path = str(tmp_path / ".ccr" / "playbook.txt")  # missing
    failure_lessons_path = str(tmp_path / ".ccr" / "failure_lessons.json")

    backend = SqliteStorageBackend(ccr_root)
    backend._memory_mgr.set_user_version(2)
    try:
        result = migrate_phase_5a(
            backend=backend,
            playbook_path=playbook_path,
            failure_lessons_path=failure_lessons_path,
            scope="project",
        )
        assert result["migrated"] == 0
        assert result["version_after"] == 3  # still bumps version to mark migration done
    finally:
        backend.close()


def test_migrate_phase_5a_atomic_on_crash(tmp_project: Path, monkeypatch) -> None:
    """If backfill throws mid-way, user_version stays at 2 and a re-run redoes it cleanly."""
    ccr_root = str(tmp_project / ".ccr")
    playbook_path = str(tmp_project / ".ccr" / "playbook.txt")
    failure_lessons_path = str(tmp_project / ".ccr" / "failure_lessons.json")

    backend = SqliteStorageBackend(ccr_root)
    backend._memory_mgr.set_user_version(2)
    try:
        # Monkey-patch Playbook._parse to throw on first invocation
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
            migrate_phase_5a(
                backend=backend,
                playbook_path=playbook_path,
                failure_lessons_path=failure_lessons_path,
                scope="project",
            )

        # Same backend is still usable — verify atomicity directly.
        assert backend._memory_mgr.get_user_version() == 2
        assert backend.bullet_list(scope="project") == []

        # Undo monkeypatch and re-run on the same backend — should succeed.
        monkeypatch.setattr(pbmod.Playbook, "_parse", original)
        result = migrate_phase_5a(
            backend=backend,
            playbook_path=playbook_path,
            failure_lessons_path=failure_lessons_path,
            scope="project",
        )
        assert result["migrated"] == 3
    finally:
        backend.close()


def test_migrate_phase_5a_global_scope(tmp_path: Path) -> None:
    """Global scope: backfills global_playbook.txt into global.db, bumps global.db user_version 0→1."""
    global_root = tmp_path / "home_ccr"
    global_root.mkdir()
    (global_root / "global_playbook.txt").write_text(FIXTURE_PLAYBOOK)

    backend = SqliteStorageBackend(str(global_root), global_ccr_root=str(global_root))
    try:
        assert backend._global_mgr.get_user_version() == 0
        result = migrate_phase_5a(
            backend=backend,
            playbook_path=str(global_root / "global_playbook.txt"),
            failure_lessons_path=str(global_root / "global_failure_lessons.json"),
            scope="global",
        )
        assert result["migrated"] == 3
        assert result["version_before"] == 0
        assert result["version_after"] == 1
        assert result["scope"] == "global"

        bullets = backend.bullet_list(scope="global")
        assert {b["id"] for b in bullets} == {"str-00001", "str-00002", "mis-00001"}
        assert backend._global_mgr.get_user_version() == 1
    finally:
        backend.close()
