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


@pytest.fixture
def srv_reset():
    """Auto-reset all ccr.mcp.server module globals after the test.

    Review-fix C3: tests that call srv._init(...) leak module-level globals
    (_memory, _playbook, _global_playbook, _repo_index, _scratchpad, _triple_store,
    _index_db, _session_store) into subsequent tests. This fixture tears down
    the SQLite connection and nulls all attached globals on teardown.
    """
    import ccr.mcp.server as srv
    yield srv
    try:
        if srv._memory is not None and hasattr(srv._memory, "_storage"):
            try:
                srv._memory._storage.close()
            except Exception:
                pass
    finally:
        srv._memory = None
        srv._playbook = None
        srv._global_playbook = None
        srv._repo_index = None
        srv._scratchpad = None
        srv._triple_store = None
        srv._index_db = None
        srv._session_store = None


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


def test_load_save_playbook_roundtrip_via_sqlite(
    tmp_project: Path, monkeypatch, srv_reset
) -> None:
    """_save_playbook writes to SQLite when backend is SqliteStorageBackend; _load_playbook reads from it."""
    monkeypatch.setenv("CCR_STORAGE_BACKEND", "sqlite")

    srv = srv_reset
    # Fresh _init on the tmp project root
    srv._init(str(tmp_project))

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


def test_save_to_backend_persists_failure_lessons(tmp_path: Path) -> None:
    """Review-fix C1: save_to_backend writes failure_lessons (not just bullets).

    Previously save_to_backend only persisted sections + bullets; failure_lessons
    were loaded by from_backend but never saved — asymmetric round-trip. This
    test proves the symmetric round-trip now holds.
    """
    from ccr.ace.playbook import FailureLesson, Playbook
    from ccr.ace.playbook_types import Bullet

    ccr_root = tmp_path / ".ccr"
    ccr_root.mkdir()
    backend = SqliteStorageBackend(str(ccr_root))
    try:
        # Build a Playbook with a single bullet + one FailureLesson attached.
        pb = Playbook()
        bullet = Bullet(
            id="str-00099",
            helpful=2,
            harmful=1,
            content="Always validate inputs before processing.",
            section="STRATEGIES & INSIGHTS",
        )
        lesson = FailureLesson(
            failure_point="Missed a null check in process_user_input",
            flawed_reasoning="Assumed client-side validation sufficient",
            counterfactual="Should validate on server side too",
            prevention_principle="Never trust inputs crossing a trust boundary",
            task_context="PR #42 review",
            timestamp="2026-04-20T10:00:00+00:00",
            evolved=False,
        )
        bullet.failure_lessons.append(lesson)
        pb._bullets.append(bullet)
        pb._id_index[bullet.id] = bullet

        # Save and re-load via a fresh Playbook instance.
        pb.save_to_backend(backend, scope="project")

        loaded = Playbook.from_backend(backend, scope="project")
        assert len(loaded.bullets) == 1
        loaded_bullet = loaded.bullets[0]
        assert loaded_bullet.id == "str-00099"
        assert len(loaded_bullet.failure_lessons) == 1
        loaded_lesson = loaded_bullet.failure_lessons[0]
        assert loaded_lesson.failure_point == lesson.failure_point
        assert loaded_lesson.flawed_reasoning == lesson.flawed_reasoning
        assert loaded_lesson.counterfactual == lesson.counterfactual
        assert loaded_lesson.prevention_principle == lesson.prevention_principle
        assert loaded_lesson.task_context == lesson.task_context
        assert loaded_lesson.timestamp == lesson.timestamp
        assert loaded_lesson.evolved is False

        # Idempotent re-save must not duplicate lessons (dedup by
        # (failure_point, timestamp)).
        loaded.save_to_backend(backend, scope="project")
        reloaded = Playbook.from_backend(backend, scope="project")
        assert len(reloaded.bullets[0].failure_lessons) == 1
    finally:
        backend.close()


def test_save_to_backend_is_atomic(tmp_path: Path, monkeypatch) -> None:
    """Review-fix C2: save_to_backend runs as a single transaction.

    A primitive that raises mid-save rolls back the whole change so concurrent
    readers never observe partial state. Simulated by monkeypatching
    _bullet_insert_nc to raise on the 2nd invocation — the 1st insert must
    NOT survive.
    """
    from ccr.ace.playbook import Playbook
    from ccr.ace.playbook_types import Bullet

    ccr_root = tmp_path / ".ccr"
    ccr_root.mkdir()
    backend = SqliteStorageBackend(str(ccr_root))
    try:
        # Seed an in-memory Playbook with two bullets and a baseline state.
        pb = Playbook()
        pb._bullets.append(Bullet(
            id="str-00101",
            helpful=0,
            harmful=0,
            content="bullet A",
            section="STRATEGIES & INSIGHTS",
        ))
        pb._bullets.append(Bullet(
            id="str-00102",
            helpful=0,
            harmful=0,
            content="bullet B",
            section="STRATEGIES & INSIGHTS",
        ))
        pb._id_index = {b.id: b for b in pb._bullets}

        # Capture pre-save DB state (baseline: zero bullets).
        pre_bullets = backend.bullet_list(scope="project")
        assert pre_bullets == []

        # Monkey-patch _bullet_insert_nc to raise on the 2nd call so the
        # 1st insert is already on the outer txn but not committed yet.
        original = backend.__class__._bullet_insert_nc
        call_count = {"n": 0}

        def flaky(self, conn, bullet):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated mid-save failure")
            return original(self, conn, bullet)

        monkeypatch.setattr(
            backend.__class__, "_bullet_insert_nc", flaky
        )

        with pytest.raises(RuntimeError, match="simulated mid-save failure"):
            pb.save_to_backend(backend, scope="project")

        # Atomicity check: DB must match pre-save state (no partial write).
        post_bullets = backend.bullet_list(scope="project")
        assert post_bullets == [], (
            f"Atomicity violated: DB has partial writes: {post_bullets}"
        )
    finally:
        backend.close()


def test_global_playbook_shared_across_projects_via_sqlite(
    tmp_path: Path, monkeypatch, srv_reset
) -> None:
    """Adding a global bullet in project A is visible in project B via ~/.ccr/global.db.

    Task 5a.4: symmetric to the project-scope round-trip test, but proves that
    the SQLite-preferring wiring of ``_load_global_playbook`` /
    ``_save_global_playbook`` enables cross-project sharing through
    ``~/.ccr/global.db``. HOME is redirected to a tmp dir so the real user dir
    is not polluted (``os.path.expanduser`` honors ``$HOME`` on macOS/Linux).
    """
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

    # Verify the bullet actually landed in global.db (not only the flat file).
    from ccr.core.storage.sqlite_backend import SqliteStorageBackend
    verify_backend = SqliteStorageBackend(
        str(fake_home / ".ccr"), global_ccr_root=str(fake_home / ".ccr"),
    )
    try:
        rows = verify_backend.bullet_list(scope="global")
        assert any("proj_a" in r["content"] for r in rows), (
            "Session 1 _save_global_playbook must persist to global.db"
        )
    finally:
        verify_backend.close()

    if srv._memory and hasattr(srv._memory, "_storage"):
        srv._memory._storage.close()

    # Delete the flat-file global_playbook.txt so Session 2's _load_global_playbook
    # MUST route through global.db (SQLite). If the wiring is missing, Session 2
    # loads an empty playbook and this test fails.
    flat_path = fake_home / ".ccr" / "global_playbook.txt"
    if flat_path.exists():
        flat_path.unlink()

    # Reset module globals (simulating a new Claude Code session on proj_b)
    srv._memory = None
    srv._playbook = None
    srv._global_playbook = None

    # Session 2: open project B
    srv._init(str(project_b))
    gpb_b = srv._ensure_global_playbook()

    assert any("proj_a" in b.content for b in gpb_b.bullets), (
        "Global bullet from proj_a should be visible in proj_b session via global.db"
    )
    # Cleanup handled by srv_reset fixture
