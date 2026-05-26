"""Tests for the storage abstraction layer (Phase 0).

Covers:
  - Feature flag routing (get_backend returns correct class)
  - SqliteConnectionManager thread safety and WAL mode
  - FileStorageBackend basic operations
  - StorageBackend ABC contract
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from ccr.core.storage import StorageBackend, get_backend
from ccr.core.storage.file_backend import FileStorageBackend
from ccr.core.storage.sqlite_backend import SqliteConnectionManager, SqliteStorageBackend
from ccr.core.types import CCRConfig


# ── Feature Flag Routing ────────────────────────────────────────


class TestGetBackend:
    def test_default_returns_file_backend(self, tmp_path):
        backend = get_backend("files", str(tmp_path))
        assert isinstance(backend, FileStorageBackend)
        assert backend.backend_type == "files"

    def test_sqlite_returns_sqlite_backend(self, tmp_path):
        ccr_root = tmp_path / ".ccr"
        ccr_root.mkdir()
        backend = get_backend("sqlite", str(ccr_root))
        assert isinstance(backend, SqliteStorageBackend)
        assert backend.backend_type == "sqlite"

    def test_backend_has_ccr_root(self, tmp_path):
        backend = get_backend("files", str(tmp_path))
        assert backend.ccr_root == str(tmp_path)

    def test_global_ccr_root_passed_through(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        global_ccr = tmp_path / "global"
        global_ccr.mkdir()
        backend = get_backend("sqlite", str(ccr), str(global_ccr))
        assert backend.global_ccr_root == str(global_ccr)

    def test_config_storage_backend_field(self):
        cfg = CCRConfig()
        assert cfg.storage_backend == "files"
        cfg2 = CCRConfig(storage_backend="sqlite")
        assert cfg2.storage_backend == "sqlite"


# ── SqliteConnectionManager ─────────────────────────────────────


class TestSqliteConnectionManager:
    def test_creates_db_file(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        assert db.exists()
        mgr.close()

    def test_creates_parent_dirs(self, tmp_path):
        db = tmp_path / "deep" / "nested" / "test.db"
        mgr = SqliteConnectionManager(str(db))
        assert db.exists()
        mgr.close()

    def test_wal_mode(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        mode = mgr.get_conn().execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        mgr.close()

    def test_foreign_keys_enabled(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        fk = mgr.get_conn().execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        mgr.close()

    def test_row_factory_is_row(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        assert mgr.get_conn().row_factory is sqlite3.Row
        mgr.close()

    def test_user_version_defaults_to_zero(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        assert mgr.get_user_version() == 0
        mgr.close()

    def test_set_user_version(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        mgr.set_user_version(3)
        assert mgr.get_user_version() == 3
        mgr.close()

    def test_thread_local_connections(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        main_conn = mgr.get_conn()
        other_conn = [None]

        def worker():
            other_conn[0] = mgr.get_conn()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert other_conn[0] is not None
        assert other_conn[0] is not main_conn
        mgr.close()

    def test_same_thread_reuses_connection(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        conn1 = mgr.get_conn()
        conn2 = mgr.get_conn()
        assert conn1 is conn2
        mgr.close()

    def test_close_allows_reopen(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        conn1 = mgr.get_conn()
        mgr.close()
        conn2 = mgr.get_conn()
        assert conn1 is not conn2
        mgr.close()

    def test_concurrent_writes(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        mgr.get_conn().execute("CREATE TABLE t (val INTEGER)")
        mgr.get_conn().commit()

        errors = []

        def writer(n):
            try:
                conn = mgr.get_conn()
                for i in range(50):
                    conn.execute("INSERT INTO t VALUES (?)", (n * 1000 + i,))
                conn.commit()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes failed: {errors}"
        count = mgr.get_conn().execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert count == 200
        mgr.close()

    def test_db_path_property(self, tmp_path):
        db = tmp_path / "test.db"
        mgr = SqliteConnectionManager(str(db))
        assert mgr.db_path == str(db)
        mgr.close()


# ── SqliteStorageBackend ────────────────────────────────────────


class TestSqliteStorageBackend:
    def test_creates_memory_db(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        assert (ccr / "memory.db").exists()
        backend.close()

    def test_creates_global_db(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        global_ccr = tmp_path / "global"
        global_ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr), str(global_ccr))
        assert (global_ccr / "global.db").exists()
        backend.close()

    def test_no_global_db_when_none(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        assert backend.global_conn is None
        backend.close()

    def test_phase1_methods_work(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        assert backend.scratchpad_get("nonexistent") is None
        assert backend.metrics_get()["total_commits"] == 0
        assert backend.log_read("main") == ""
        assert backend.metadata_load() == {}
        backend.close()

    def test_memory_conn_returns_connection(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        assert isinstance(backend.memory_conn, sqlite3.Connection)
        backend.close()

    def test_syncs_flat_file_commits_after_migrated_sentinel(self, tmp_path):
        ccr = tmp_path / ".ccr"
        branch = ccr / "branches" / "main"
        branch.mkdir(parents=True)
        (ccr / ".migrated").write_text("2026-04-18")
        (branch / "commits.md").write_text(
            "# Branch: main\n\n"
            "# Milestone Journal\n\n"
            "## [C101] 2026-04-26 16:44 | branch:main | SQLite sync works\n"
            "**What**: Synced missing flat-file commit into SQLite.\n"
            "**Why**: Backend switched after initial migration.\n"
            "**Files**: ccr/core/storage/sqlite_backend.py\n"
            "**Next**: Keep SQLite current.\n"
        )

        backend = SqliteStorageBackend(str(ccr))
        try:
            commits = backend.commit_list("main", limit=5)
        finally:
            backend.close()

        assert [c["id"] for c in commits] == ["C101"]
        assert commits[0]["title"] == "SQLite sync works"
        assert commits[0]["what"] == "Synced missing flat-file commit into SQLite."


# ── FileStorageBackend ──────────────────────────────────────────


class TestFileStorageBackend:
    def test_scratchpad_set_get(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        result = backend.scratchpad_set("key1", "value1")
        assert result["key"] == "key1"
        assert result["value"] == "value1"

        got = backend.scratchpad_get("key1")
        assert got is not None
        assert got["value"] == "value1"
        assert got["access_count"] == 1

    def test_scratchpad_missing_key(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        assert backend.scratchpad_get("nonexistent") is None

    def test_scratchpad_delete(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        backend.scratchpad_set("key1", "value1")
        assert backend.scratchpad_delete("key1") is True
        assert backend.scratchpad_delete("key1") is False
        assert backend.scratchpad_get("key1") is None

    def test_scratchpad_clear(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        backend.scratchpad_set("a", "1")
        backend.scratchpad_set("b", "2")
        count = backend.scratchpad_clear()
        assert count == 2
        assert backend.scratchpad_list() == []

    def test_scratchpad_list(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        backend.scratchpad_set("x", "1")
        backend.scratchpad_set("y", "2")
        entries = backend.scratchpad_list()
        keys = {e["key"] for e in entries}
        assert keys == {"x", "y"}

    def test_scratchpad_search(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        backend.scratchpad_set("alpha", "hello world")
        backend.scratchpad_set("beta", "goodbye moon")
        results = backend.scratchpad_search("hello")
        assert len(results) == 1
        assert results[0]["key"] == "alpha"

    def test_scratchpad_ttl_expiry(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        backend.scratchpad_set("temp", "data", ttl_seconds=-1)
        assert backend.scratchpad_get("temp") is None

    def test_metrics_increment(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        backend.metrics_increment("total_commits", 1)
        backend.metrics_increment("total_commits", 2)
        metrics = backend.metrics_get()
        assert metrics["total_commits"] == 3

    def test_metrics_default(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        metrics = backend.metrics_get()
        assert "total_commits" in metrics
        assert metrics["total_commits"] == 0

    def test_log_append_and_read(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        (ccr / "branches" / "main").mkdir(parents=True)
        backend = FileStorageBackend(str(ccr))
        backend.log_append("main", "line1")
        backend.log_append("main", "line2")
        content = backend.log_read("main", count=10)
        assert "line1" in content
        assert "line2" in content

    def test_log_rotation(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        (ccr / "branches" / "main").mkdir(parents=True)
        backend = FileStorageBackend(str(ccr))
        for i in range(20):
            backend.log_append("main", f"line{i}", max_lines=10)
        content = backend.log_read("main", count=100)
        lines = [line for line in content.split("\n") if line]
        assert len(lines) == 10

    def test_log_read_empty(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        assert backend.log_read("main") == ""

    def test_metadata_roundtrip(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        data = {"version": 1, "branches": [{"name": "main"}]}
        backend.metadata_save(data)
        loaded = backend.metadata_load()
        assert loaded["version"] == 1
        assert loaded["branches"][0]["name"] == "main"

    def test_metadata_missing(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        assert backend.metadata_load() == {}


# ── Phase 2 Dual-Backend Parity ────────────────────────────────


@pytest.fixture(params=["files", "sqlite"])
def phase2_backend(request, tmp_path):
    ccr = tmp_path / ".ccr"
    ccr.mkdir()
    (ccr / "branches" / "main").mkdir(parents=True)
    b = get_backend(request.param, str(ccr))
    yield b
    b.close()


class TestPhase2DualBackend:
    def test_bullet_insert_get(self, phase2_backend):
        phase2_backend.bullet_insert({
            "id": "str-00001", "section": "STRATEGIES & INSIGHTS",
            "content": "Test", "created_at": "2026-04-17T00:00:00+00:00",
        })
        got = phase2_backend.bullet_get("str-00001")
        assert got is not None
        assert got["content"] == "Test"

    def test_bullet_list(self, phase2_backend):
        phase2_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "A",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        phase2_backend.bullet_insert({
            "id": "str-00002", "section": "S", "content": "B",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert len(phase2_backend.bullet_list()) == 2

    def test_bullet_update(self, phase2_backend):
        phase2_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "Old",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert phase2_backend.bullet_update("str-00001", {"content": "New"})
        assert phase2_backend.bullet_get("str-00001")["content"] == "New"

    def test_bullet_delete(self, phase2_backend):
        phase2_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert phase2_backend.bullet_delete("str-00001") is True
        assert phase2_backend.bullet_get("str-00001") is None

    def test_bullet_update_counters(self, phase2_backend):
        phase2_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        updated = phase2_backend.bullet_update_counters([
            {"id": "str-00001", "tag": "helpful"},
        ])
        assert updated == 1
        b = phase2_backend.bullet_get("str-00001")
        assert b["helpful"] == 1

    def test_bullet_get_next_id(self, phase2_backend):
        assert phase2_backend.bullet_get_next_id() == 1
        phase2_backend.bullet_insert({
            "id": "str-00010", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert phase2_backend.bullet_get_next_id() == 11

    def test_sections_roundtrip(self, phase2_backend):
        phase2_backend.playbook_sections_set(["A", "B", "C"])
        assert phase2_backend.playbook_sections_get() == ["A", "B", "C"]

    def test_failure_lessons(self, phase2_backend):
        phase2_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        phase2_backend.failure_lessons_insert("str-00001", {
            "failure_point": "broke", "flawed_reasoning": "wrong",
            "counterfactual": "X", "prevention_principle": "always X",
        })
        lessons = phase2_backend.failure_lessons_for_bullet("str-00001")
        assert len(lessons) == 1

    def test_failure_lessons_mark_evolved(self, phase2_backend):
        phase2_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        phase2_backend.failure_lessons_insert("str-00001", {
            "failure_point": "broke", "flawed_reasoning": "wrong",
            "counterfactual": "X", "prevention_principle": "always X",
        })
        assert phase2_backend.failure_lessons_mark_evolved("str-00001") == 1

    def test_schema_roundtrip(self, phase2_backend):
        phase2_backend.playbook_schema_save({"version": 1, "decay_rate": 0.95})
        loaded = phase2_backend.playbook_schema_load()
        assert loaded["version"] == 1

    def test_delta_history(self, phase2_backend):
        phase2_backend.delta_history_append({"author": "test", "ops_count": 1})

    def test_archived_bullets(self, phase2_backend):
        count = phase2_backend.archived_bullets_insert([
            {"id": "str-00001", "content": "X", "helpful": 0, "harmful": 5},
        ], reason="pruned")
        assert count == 1


# ── StorageBackend ABC ──────────────────────────────────────────


class TestStorageBackendABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            StorageBackend("root")

    def test_close_is_noop_by_default(self, tmp_path):
        backend = FileStorageBackend(str(tmp_path))
        backend.close()


# ── Migration Module ────────────────────────────────────────────


class TestMigrationHelpers:
    def test_needs_migration_false_without_metadata(self, tmp_path):
        from ccr.core.storage.migration import needs_migration
        assert needs_migration(str(tmp_path)) is False

    def test_needs_migration_true_with_metadata(self, tmp_path):
        from ccr.core.storage.migration import needs_migration
        (tmp_path / "metadata.yaml").write_text("version: 1")
        assert needs_migration(str(tmp_path)) is True

    def test_needs_migration_false_with_sentinel(self, tmp_path):
        from ccr.core.storage.migration import needs_migration
        (tmp_path / "metadata.yaml").write_text("version: 1")
        (tmp_path / ".migrated").write_text("done")
        assert needs_migration(str(tmp_path)) is False

    def test_backup_file(self, tmp_path):
        from ccr.core.storage.migration import _backup_file
        f = tmp_path / "test.json"
        f.write_text('{"a": 1}')
        bak = _backup_file(str(f))
        assert bak == str(f) + ".bak"
        assert not f.exists()
        assert (tmp_path / "test.json.bak").exists()

    def test_backup_missing_file(self, tmp_path):
        from ccr.core.storage.migration import _backup_file
        assert _backup_file(str(tmp_path / "nope.json")) is None


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
