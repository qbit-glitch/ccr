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


def test_migrate_phase_5a_atomic_on_crash(tmp_path: Path, monkeypatch) -> None:
    """If backfill throws mid-way, user_version stays at 2 and a re-run redoes it cleanly."""
    ccr_root_dir = tmp_path / ".ccr"
    ccr_root_dir.mkdir()
    ccr_root = str(ccr_root_dir)
    playbook_path = str(ccr_root_dir / "playbook.txt")
    failure_lessons_path = str(ccr_root_dir / "failure_lessons.json")

    # Construct backend first (no flat playbook yet) so init's auto-migration is
    # a no-op. Then write the playbook, rewind user_version to 2, and exercise
    # the helper manually to verify atomicity.
    backend = SqliteStorageBackend(ccr_root)
    (ccr_root_dir / "playbook.txt").write_text(FIXTURE_PLAYBOOK)
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

    # Construct backend first (no global_playbook.txt yet) so init's auto-migration
    # is a no-op for the global scope. Then write the playbook and roll the
    # version back to 0 so we can exercise the helper directly.
    backend = SqliteStorageBackend(str(global_root), global_ccr_root=str(global_root))
    (global_root / "global_playbook.txt").write_text(FIXTURE_PLAYBOOK)
    backend._global_mgr.set_user_version(0)
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


def test_sqlite_backend_auto_runs_phase5a_on_init(tmp_path: Path) -> None:
    """Opening SqliteStorageBackend triggers phase 5a when memory.db is at v<3."""
    ccr_root = tmp_path / ".ccr"
    ccr_root.mkdir()
    ccr_root_str = str(ccr_root)

    # Build an empty backend first (no playbook.txt yet) — version auto-advances
    # to 3 because no flat file exists. Roll version back to simulate a Phase-4
    # baseline DB from before Phase 5a shipped.
    backend = SqliteStorageBackend(ccr_root_str)
    backend._memory_mgr.set_user_version(2)
    assert backend.bullet_list(scope="project") == []
    backend.close()

    # Drop the flat playbook.txt now, then re-open — auto-migration should run.
    (ccr_root / "playbook.txt").write_text(FIXTURE_PLAYBOOK)

    backend = SqliteStorageBackend(ccr_root_str)
    try:
        assert backend._memory_mgr.get_user_version() == 3
        bullets = backend.bullet_list(scope="project")
        assert len(bullets) == 3
        assert {b["id"] for b in bullets} == {"str-00001", "str-00002", "mis-00001"}
    finally:
        backend.close()
