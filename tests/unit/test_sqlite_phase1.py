"""Phase 1 SQLite backend tests + parameterized dual-backend tests.

Covers:
  - SqliteStorageBackend: scratchpad CRUD, TTL, search, metrics, log, metadata
  - Dual-backend parity: identical behaviour on files vs sqlite
  - Migration: flat files → SQLite data integrity
"""

from __future__ import annotations

import json
import os

import pytest

from ccr.core.storage import get_backend
from ccr.core.storage.file_backend import FileStorageBackend
from ccr.core.storage.sqlite_backend import SqliteStorageBackend


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sqlite_backend(tmp_path):
    ccr = tmp_path / ".ccr"
    ccr.mkdir()
    backend = SqliteStorageBackend(str(ccr))
    yield backend
    backend.close()


@pytest.fixture
def file_backend(tmp_path):
    ccr = tmp_path / ".ccr"
    ccr.mkdir()
    (ccr / "branches" / "main").mkdir(parents=True)
    backend = FileStorageBackend(str(ccr))
    yield backend


@pytest.fixture(params=["files", "sqlite"])
def backend(request, tmp_path):
    ccr = tmp_path / ".ccr"
    ccr.mkdir()
    (ccr / "branches" / "main").mkdir(parents=True)
    b = get_backend(request.param, str(ccr))
    yield b
    b.close()


# ── SQLite Scratchpad ──────────────────────────────────────────────


class TestSqliteScratchpad:
    def test_set_and_get(self, sqlite_backend):
        result = sqlite_backend.scratchpad_set("k1", "v1")
        assert result["key"] == "k1"
        assert result["value"] == "v1"
        assert result["access_count"] == 0

        got = sqlite_backend.scratchpad_get("k1")
        assert got is not None
        assert got["value"] == "v1"
        assert got["access_count"] == 1

    def test_get_missing_returns_none(self, sqlite_backend):
        assert sqlite_backend.scratchpad_get("nope") is None

    def test_set_preserves_created_at(self, sqlite_backend):
        r1 = sqlite_backend.scratchpad_set("k", "v1")
        created = r1["created_at"]
        r2 = sqlite_backend.scratchpad_set("k", "v2")
        assert r2["created_at"] == created
        assert r2["value"] == "v2"

    def test_set_preserves_access_count(self, sqlite_backend):
        sqlite_backend.scratchpad_set("k", "v1")
        sqlite_backend.scratchpad_get("k")
        sqlite_backend.scratchpad_get("k")
        r = sqlite_backend.scratchpad_set("k", "v2")
        assert r["access_count"] == 2

    def test_ttl_expiry(self, sqlite_backend):
        sqlite_backend.scratchpad_set("temp", "data", ttl_seconds=-1)
        assert sqlite_backend.scratchpad_get("temp") is None

    def test_ttl_not_expired(self, sqlite_backend):
        sqlite_backend.scratchpad_set("temp", "data", ttl_seconds=3600)
        got = sqlite_backend.scratchpad_get("temp")
        assert got is not None
        assert got["value"] == "data"

    def test_list_excludes_expired(self, sqlite_backend):
        sqlite_backend.scratchpad_set("live", "yes")
        sqlite_backend.scratchpad_set("dead", "no", ttl_seconds=-1)
        entries = sqlite_backend.scratchpad_list()
        keys = [e["key"] for e in entries]
        assert "live" in keys
        assert "dead" not in keys

    def test_list_cleans_expired(self, sqlite_backend):
        sqlite_backend.scratchpad_set("dead", "no", ttl_seconds=-1)
        sqlite_backend.scratchpad_list()
        row = sqlite_backend.memory_conn.execute(
            "SELECT COUNT(*) FROM scratchpad WHERE key = 'dead'",
        ).fetchone()[0]
        assert row == 0

    def test_delete(self, sqlite_backend):
        sqlite_backend.scratchpad_set("k", "v")
        assert sqlite_backend.scratchpad_delete("k") is True
        assert sqlite_backend.scratchpad_delete("k") is False
        assert sqlite_backend.scratchpad_get("k") is None

    def test_clear(self, sqlite_backend):
        sqlite_backend.scratchpad_set("a", "1")
        sqlite_backend.scratchpad_set("b", "2")
        count = sqlite_backend.scratchpad_clear()
        assert count == 2
        assert sqlite_backend.scratchpad_list() == []

    def test_clear_empty(self, sqlite_backend):
        assert sqlite_backend.scratchpad_clear() == 0

    def test_search_matches(self, sqlite_backend):
        sqlite_backend.scratchpad_set("alpha", "hello world")
        sqlite_backend.scratchpad_set("beta", "goodbye moon")
        results = sqlite_backend.scratchpad_search("hello")
        assert len(results) == 1
        assert results[0]["key"] == "alpha"

    def test_search_case_insensitive(self, sqlite_backend):
        sqlite_backend.scratchpad_set("k", "FooBar")
        results = sqlite_backend.scratchpad_search("foobar")
        assert len(results) == 1

    def test_search_matches_key(self, sqlite_backend):
        sqlite_backend.scratchpad_set("important_note", "anything")
        results = sqlite_backend.scratchpad_search("important")
        assert len(results) == 1

    def test_search_top_k(self, sqlite_backend):
        for i in range(10):
            sqlite_backend.scratchpad_set(f"k{i}", "match")
        results = sqlite_backend.scratchpad_search("match", top_k=3)
        assert len(results) == 3


# ── SQLite Metrics ─────────────────────────────────────────────────


class TestSqliteMetrics:
    def test_increment(self, sqlite_backend):
        sqlite_backend.metrics_increment("total_commits", 1)
        sqlite_backend.metrics_increment("total_commits", 2)
        m = sqlite_backend.metrics_get()
        assert m["total_commits"] == 3

    def test_defaults(self, sqlite_backend):
        m = sqlite_backend.metrics_get()
        assert m["total_commits"] == 0
        assert m["search_calls"] == 0
        assert m["link_creations"] == 0

    def test_multiple_keys(self, sqlite_backend):
        sqlite_backend.metrics_increment("total_commits", 5)
        sqlite_backend.metrics_increment("search_calls", 3)
        m = sqlite_backend.metrics_get()
        assert m["total_commits"] == 5
        assert m["search_calls"] == 3


# ── SQLite Log ─────────────────────────────────────────────────────


class TestSqliteLog:
    def test_append_and_read(self, sqlite_backend):
        sqlite_backend.log_append("main", "line1")
        sqlite_backend.log_append("main", "line2")
        content = sqlite_backend.log_read("main", count=10)
        assert "line1" in content
        assert "line2" in content

    def test_read_empty(self, sqlite_backend):
        assert sqlite_backend.log_read("main") == ""

    def test_read_count_limit(self, sqlite_backend):
        for i in range(20):
            sqlite_backend.log_append("main", f"line{i}")
        content = sqlite_backend.log_read("main", count=5)
        lines = [l for l in content.split("\n") if l]
        assert len(lines) == 5
        assert "line19" in content
        assert "line15" in content

    def test_chronological_order(self, sqlite_backend):
        sqlite_backend.log_append("main", "first")
        sqlite_backend.log_append("main", "second")
        sqlite_backend.log_append("main", "third")
        content = sqlite_backend.log_read("main", count=10)
        lines = [l for l in content.split("\n") if l]
        assert lines == ["first", "second", "third"]

    def test_rotation(self, sqlite_backend):
        for i in range(20):
            sqlite_backend.log_append("main", f"line{i}", max_lines=10)
        count = sqlite_backend.memory_conn.execute(
            "SELECT COUNT(*) FROM log_entries WHERE branch = 'main'",
        ).fetchone()[0]
        assert count == 10

    def test_branch_isolation(self, sqlite_backend):
        sqlite_backend.log_append("main", "main_line")
        sqlite_backend.log_append("feature", "feature_line")
        assert "main_line" in sqlite_backend.log_read("main")
        assert "feature_line" not in sqlite_backend.log_read("main")
        assert "feature_line" in sqlite_backend.log_read("feature")


# ── SQLite Metadata ────────────────────────────────────────────────


class TestSqliteMetadata:
    def test_roundtrip(self, sqlite_backend):
        data = {"version": 1, "branches": [{"name": "main"}]}
        sqlite_backend.metadata_save(data)
        loaded = sqlite_backend.metadata_load()
        assert loaded["version"] == 1
        assert loaded["branches"][0]["name"] == "main"

    def test_load_empty(self, sqlite_backend):
        assert sqlite_backend.metadata_load() == {}

    def test_overwrite(self, sqlite_backend):
        sqlite_backend.metadata_save({"v": 1})
        sqlite_backend.metadata_save({"v": 2})
        assert sqlite_backend.metadata_load()["v"] == 2

    def test_complex_nested(self, sqlite_backend):
        data = {
            "created": "2026-04-17",
            "branches": [
                {"name": "main", "status": "active", "commits": 50},
                {"name": "feature-x", "status": "merged"},
            ],
            "tags": ["v1.0", "v2.0"],
        }
        sqlite_backend.metadata_save(data)
        loaded = sqlite_backend.metadata_load()
        assert len(loaded["branches"]) == 2
        assert loaded["tags"] == ["v1.0", "v2.0"]


# ── Dual-Backend Parity ───────────────────────────────────────────


class TestDualBackendParity:
    def test_scratchpad_set_get(self, backend):
        result = backend.scratchpad_set("k", "v")
        assert result["key"] == "k"
        got = backend.scratchpad_get("k")
        assert got["value"] == "v"
        assert got["access_count"] == 1

    def test_scratchpad_ttl_expiry(self, backend):
        backend.scratchpad_set("t", "d", ttl_seconds=-1)
        assert backend.scratchpad_get("t") is None

    def test_scratchpad_delete(self, backend):
        backend.scratchpad_set("k", "v")
        assert backend.scratchpad_delete("k") is True
        assert backend.scratchpad_delete("k") is False

    def test_scratchpad_clear(self, backend):
        backend.scratchpad_set("a", "1")
        backend.scratchpad_set("b", "2")
        assert backend.scratchpad_clear() == 2
        assert backend.scratchpad_list() == []

    def test_scratchpad_search(self, backend):
        backend.scratchpad_set("note", "hello world")
        backend.scratchpad_set("other", "goodbye")
        results = backend.scratchpad_search("hello")
        assert len(results) == 1
        assert results[0]["key"] == "note"

    def test_metrics_increment(self, backend):
        backend.metrics_increment("total_commits", 3)
        m = backend.metrics_get()
        assert m["total_commits"] == 3

    def test_log_append_read(self, backend):
        backend.log_append("main", "line1")
        backend.log_append("main", "line2")
        content = backend.log_read("main", count=10)
        assert "line1" in content
        assert "line2" in content

    def test_log_read_empty(self, backend):
        assert backend.log_read("main") == ""

    def test_metadata_roundtrip(self, backend):
        data = {"version": 1, "name": "test"}
        backend.metadata_save(data)
        loaded = backend.metadata_load()
        assert loaded["version"] == 1

    def test_metadata_empty(self, backend):
        assert backend.metadata_load() == {}


# ── Migration ──────────────────────────────────────────────────────


class TestMigratePhase1:
    def _setup_flat_files(self, ccr_root):
        """Create flat files matching the existing format."""
        os.makedirs(os.path.join(ccr_root, "branches", "main"), exist_ok=True)

        scratchpad = {
            "version": 1,
            "entries": {
                "note1": {
                    "value": "hello",
                    "created_at": "2026-04-01T00:00:00+00:00",
                    "updated_at": "2026-04-01T12:00:00+00:00",
                    "access_count": 3,
                    "expires_at": None,
                },
                "note2": {
                    "value": "world",
                    "created_at": "2026-04-02T00:00:00+00:00",
                    "updated_at": "2026-04-02T00:00:00+00:00",
                    "access_count": 0,
                    "expires_at": None,
                },
            },
        }
        with open(os.path.join(ccr_root, "scratchpad.json"), "w") as f:
            json.dump(scratchpad, f)

        metrics = {"total_commits": 42, "search_calls": 7, "last_updated": "2026-04-01"}
        with open(os.path.join(ccr_root, "memory_metrics.json"), "w") as f:
            json.dump(metrics, f)

        with open(os.path.join(ccr_root, "branches", "main", "log.md"), "w") as f:
            f.write("first log\nsecond log\nthird log")

        import yaml
        metadata = {"version": 1, "created": "2026-04-01", "branches": [{"name": "main"}]}
        with open(os.path.join(ccr_root, "metadata.yaml"), "w") as f:
            yaml.dump(metadata, f)

    def test_full_migration(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_1

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_flat_files(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")

        result = migrate_phase_1(str(ccr), db_path)
        assert result["errors"] == []
        assert result["migrated"] > 0

        backend_fresh = SqliteStorageBackend(str(ccr))

        got = backend_fresh.scratchpad_get("note1")
        assert got is not None
        assert got["value"] == "hello"
        assert got["access_count"] == 4  # 3 original + 1 from get

        m = backend_fresh.metrics_get()
        assert m["total_commits"] == 42
        assert m["search_calls"] == 7

        log = backend_fresh.log_read("main", count=10)
        assert "first log" in log
        assert "third log" in log

        meta = backend_fresh.metadata_load()
        assert meta["version"] == 1

        backend.close()
        backend_fresh.close()

    def test_migration_backs_up_files(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_1

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_flat_files(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        migrate_phase_1(str(ccr), db_path)

        assert (ccr / "scratchpad.json.bak").exists()
        assert (ccr / "memory_metrics.json.bak").exists()
        assert (ccr / "metadata.yaml.bak").exists()
        backend.close()

    def test_migration_idempotent(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_1

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_flat_files(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")

        r1 = migrate_phase_1(str(ccr), db_path)
        self._setup_flat_files(str(ccr))
        r2 = migrate_phase_1(str(ccr), db_path)

        assert r1["errors"] == []
        assert r2["errors"] == []

        m = backend.metrics_get()
        assert m["total_commits"] == 42
        backend.close()

    def test_migration_empty_files(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_1

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        os.makedirs(os.path.join(str(ccr), "branches", "main"), exist_ok=True)

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_1(str(ccr), db_path)
        assert result["migrated"] == 0
        assert result["errors"] == []
        backend.close()

    def test_auto_migrate(self, tmp_path):
        from ccr.core.storage.migration import auto_migrate, needs_migration

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_flat_files(str(ccr))

        assert needs_migration(str(ccr)) is True

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = auto_migrate(str(ccr), db_path)

        assert 1 in result["phases_run"]
        assert result["total_migrated"] > 0
        assert result["errors"] == []
        assert (ccr / ".migrated").exists()
        assert needs_migration(str(ccr)) is False
        backend.close()


# ── Phase 2: SQLite Bullets ──────────────────────────────────────


class TestSqliteBullets:
    def test_insert_and_get(self, sqlite_backend):
        bullet = {
            "id": "str-00001", "section": "STRATEGIES & INSIGHTS",
            "content": "Test strategy", "created_at": "2026-04-17T00:00:00+00:00",
        }
        sqlite_backend.bullet_insert(bullet)
        got = sqlite_backend.bullet_get("str-00001")
        assert got is not None
        assert got["content"] == "Test strategy"
        assert got["helpful"] == 0

    def test_get_missing(self, sqlite_backend):
        assert sqlite_backend.bullet_get("nope") is None

    def test_list_all(self, sqlite_backend):
        for i in range(3):
            sqlite_backend.bullet_insert({
                "id": f"str-{i:05d}", "section": "STRATEGIES & INSIGHTS",
                "content": f"Strategy {i}", "created_at": "2026-04-17T00:00:00+00:00",
            })
        assert len(sqlite_backend.bullet_list()) == 3

    def test_list_by_section(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "STRATEGIES & INSIGHTS",
            "content": "A", "created_at": "2026-04-17T00:00:00+00:00",
        })
        sqlite_backend.bullet_insert({
            "id": "heu-00001", "section": "PROBLEM-SOLVING HEURISTICS",
            "content": "B", "created_at": "2026-04-17T00:00:00+00:00",
        })
        result = sqlite_backend.bullet_list(section="STRATEGIES & INSIGHTS")
        assert len(result) == 1
        assert result[0]["id"] == "str-00001"

    def test_update(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "STRATEGIES & INSIGHTS",
            "content": "Old", "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert sqlite_backend.bullet_update("str-00001", {"content": "New"})
        assert sqlite_backend.bullet_get("str-00001")["content"] == "New"

    def test_update_missing(self, sqlite_backend):
        assert sqlite_backend.bullet_update("nope", {"content": "X"}) is False

    def test_delete(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert sqlite_backend.bullet_delete("str-00001") is True
        assert sqlite_backend.bullet_delete("str-00001") is False
        assert sqlite_backend.bullet_get("str-00001") is None

    def test_update_counters_helpful(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        updated = sqlite_backend.bullet_update_counters([
            {"id": "str-00001", "tag": "helpful", "weight": 0.8},
        ])
        assert updated == 1
        b = sqlite_backend.bullet_get("str-00001")
        assert b["helpful"] == 1
        assert abs(b["weighted_helpful"] - 0.8) < 0.001

    def test_update_counters_harmful_with_lesson(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        updated = sqlite_backend.bullet_update_counters([{
            "id": "str-00001", "tag": "harmful",
            "failure_lesson": {
                "failure_point": "broke here",
                "flawed_reasoning": "wrong assumption",
                "counterfactual": "should have done X",
                "prevention_principle": "always do X",
            },
        }])
        assert updated == 1
        b = sqlite_backend.bullet_get("str-00001")
        assert b["harmful"] == 1
        lessons = sqlite_backend.failure_lessons_for_bullet("str-00001")
        assert len(lessons) == 1
        assert lessons[0]["failure_point"] == "broke here"

    def test_update_counters_missing_id(self, sqlite_backend):
        assert sqlite_backend.bullet_update_counters([
            {"id": "nope", "tag": "helpful"},
        ]) == 0

    def test_get_next_id_empty(self, sqlite_backend):
        assert sqlite_backend.bullet_get_next_id() == 1

    def test_get_next_id(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00042", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert sqlite_backend.bullet_get_next_id() == 43

    def test_decay_rate_helpful(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        sqlite_backend.bullet_update_counters([
            {"id": "str-00001", "tag": "helpful"},
        ])
        b = sqlite_backend.bullet_get("str-00001")
        expected = max(0.90, min(0.99, 0.95 + (1 - 0) * 0.002))
        assert abs(b["personal_decay_rate"] - expected) < 0.001

    def test_global_scope(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        gcr = tmp_path / "global"
        gcr.mkdir()
        backend = SqliteStorageBackend(str(ccr), str(gcr))
        backend.bullet_insert(
            {"id": "g-00001", "section": "S", "content": "Global",
             "created_at": "2026-04-17T00:00:00+00:00"},
            scope="global",
        )
        assert backend.bullet_get("g-00001", scope="global") is not None
        assert backend.bullet_get("g-00001", scope="project") is None
        backend.close()


# ── Phase 2: SQLite Sections ────────────────────────────────────


class TestSqliteSections:
    def test_set_and_get(self, sqlite_backend):
        sections = ["STRATEGIES & INSIGHTS", "OTHERS"]
        sqlite_backend.playbook_sections_set(sections)
        got = sqlite_backend.playbook_sections_get()
        assert got == sections

    def test_preserves_order(self, sqlite_backend):
        sections = ["C", "A", "B"]
        sqlite_backend.playbook_sections_set(sections)
        assert sqlite_backend.playbook_sections_get() == ["C", "A", "B"]

    def test_empty(self, sqlite_backend):
        assert sqlite_backend.playbook_sections_get() == []

    def test_replace(self, sqlite_backend):
        sqlite_backend.playbook_sections_set(["A", "B"])
        sqlite_backend.playbook_sections_set(["X", "Y", "Z"])
        assert sqlite_backend.playbook_sections_get() == ["X", "Y", "Z"]


# ── Phase 2: SQLite Failure Lessons ─────────────────────────────


class TestSqliteFailureLessons:
    def test_insert_and_get(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        sqlite_backend.failure_lessons_insert("str-00001", {
            "failure_point": "broke",
            "flawed_reasoning": "wrong",
            "counterfactual": "do X",
            "prevention_principle": "always X",
        })
        lessons = sqlite_backend.failure_lessons_for_bullet("str-00001")
        assert len(lessons) == 1
        assert lessons[0]["failure_point"] == "broke"
        assert lessons[0]["evolved"] == 0

    def test_mark_evolved(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        sqlite_backend.failure_lessons_insert("str-00001", {
            "failure_point": "broke", "flawed_reasoning": "wrong",
            "counterfactual": "do X", "prevention_principle": "always X",
        })
        count = sqlite_backend.failure_lessons_mark_evolved("str-00001")
        assert count == 1
        lessons = sqlite_backend.failure_lessons_for_bullet("str-00001")
        assert lessons[0]["evolved"] == 1

    def test_all_grouped(self, sqlite_backend):
        for bid in ["str-00001", "str-00002"]:
            sqlite_backend.bullet_insert({
                "id": bid, "section": "S", "content": "X",
                "created_at": "2026-04-17T00:00:00+00:00",
            })
            sqlite_backend.failure_lessons_insert(bid, {
                "failure_point": f"broke-{bid}", "flawed_reasoning": "wrong",
                "counterfactual": "do X", "prevention_principle": "always X",
            })
        grouped = sqlite_backend.failure_lessons_all()
        assert len(grouped) == 2
        assert "str-00001" in grouped
        assert "str-00002" in grouped

    def test_cascade_delete(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        sqlite_backend.failure_lessons_insert("str-00001", {
            "failure_point": "broke", "flawed_reasoning": "wrong",
            "counterfactual": "do X", "prevention_principle": "always X",
        })
        sqlite_backend.bullet_delete("str-00001")
        assert sqlite_backend.failure_lessons_for_bullet("str-00001") == []


# ── Phase 2: SQLite Schema ──────────────────────────────────────


class TestSqlitePlaybookSchema:
    def test_save_and_load(self, sqlite_backend):
        schema = {"version": 1, "decay_rate": 0.95, "token_budget": 80000}
        sqlite_backend.playbook_schema_save(schema)
        loaded = sqlite_backend.playbook_schema_load()
        assert loaded["decay_rate"] == 0.95

    def test_load_empty(self, sqlite_backend):
        assert sqlite_backend.playbook_schema_load() == {}

    def test_history(self, sqlite_backend):
        sqlite_backend.playbook_schema_save({"version": 1, "decay_rate": 0.95})
        sqlite_backend.playbook_schema_save({"version": 2, "decay_rate": 0.90, "parent_version": 1})
        history = sqlite_backend.playbook_schema_history()
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[1]["version"] == 2


# ── Phase 2: SQLite Audit ───────────────────────────────────────


class TestSqliteAudit:
    def test_delta_history_append(self, sqlite_backend):
        sqlite_backend.delta_history_append({
            "author": "claude", "ops_count": 3, "applied_count": 2,
            "scope": "project", "failed_ids": ["str-00099"],
        })
        conn = sqlite_backend.memory_conn
        row = conn.execute("SELECT * FROM delta_history").fetchone()
        assert row["author"] == "claude"
        assert row["ops_count"] == 3

    def test_archived_bullets_insert(self, sqlite_backend):
        count = sqlite_backend.archived_bullets_insert([
            {"id": "str-00001", "section": "S", "content": "X", "helpful": 1, "harmful": 5},
            {"id": "str-00002", "section": "S", "content": "Y", "helpful": 0, "harmful": 3},
        ], reason="pruned")
        assert count == 2
        rows = sqlite_backend.memory_conn.execute("SELECT * FROM archived_bullets").fetchall()
        assert len(rows) == 2
        assert rows[0]["reason"] == "pruned"


# ── Phase 2: Migration ──────────────────────────────────────────


class TestMigratePhase2:
    def _setup_playbook_files(self, ccr_root):
        os.makedirs(ccr_root, exist_ok=True)
        with open(os.path.join(ccr_root, "playbook.txt"), "w") as f:
            f.write("## STRATEGIES & INSIGHTS\n")
            f.write("[str-00001] helpful=5 harmful=1 :: Test strategy\n")
            f.write("[str-00002] helpful=0 harmful=0 :: Another strategy\n")
            f.write("\n## PROBLEM-SOLVING HEURISTICS\n")
            f.write("[heu-00001] helpful=2 harmful=3 :: Heuristic\n")

        with open(os.path.join(ccr_root, "failure_lessons.json"), "w") as f:
            json.dump({
                "str-00001": {
                    "lessons": [{
                        "failure_point": "broke",
                        "flawed_reasoning": "wrong",
                        "counterfactual": "X",
                        "prevention_principle": "always X",
                    }],
                    "scope": "general",
                    "trigger": "when X",
                    "weighted_helpful": 2.5,
                },
            }, f)

        with open(os.path.join(ccr_root, "playbook_schema.json"), "w") as f:
            json.dump({
                "current": {"version": 1, "decay_rate": 0.95},
                "history": [],
            }, f)

        with open(os.path.join(ccr_root, "playbook_history.json"), "w") as f:
            json.dump([
                {"author": "claude", "ops_count": 2, "applied_count": 2, "timestamp": "2026-04-17"},
            ], f)

        with open(os.path.join(ccr_root, "archived_bullets.json"), "w") as f:
            json.dump([
                {"id": "old-00001", "section": "S", "content": "Old", "helpful": 0,
                 "harmful": 5, "archived_at": "2026-04-17", "reason": "pruned"},
            ], f)

    def test_full_migration(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_2

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_playbook_files(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_2(str(ccr), db_path)

        assert result["errors"] == []
        assert result["migrated"] > 0

        bullets = backend.bullet_list()
        assert len(bullets) == 3

        b1 = backend.bullet_get("str-00001")
        assert b1["helpful"] == 5
        assert b1["harmful"] == 1
        assert b1["trigger_text"] == "when X"
        assert abs(b1["weighted_helpful"] - 2.5) < 0.01

        lessons = backend.failure_lessons_for_bullet("str-00001")
        assert len(lessons) == 1
        assert lessons[0]["failure_point"] == "broke"

        sections = backend.playbook_sections_get()
        assert "STRATEGIES & INSIGHTS" in sections
        assert "PROBLEM-SOLVING HEURISTICS" in sections

        schema = backend.playbook_schema_load()
        assert schema["decay_rate"] == 0.95

        delta_rows = backend.memory_conn.execute("SELECT * FROM delta_history").fetchall()
        assert len(delta_rows) == 1
        assert delta_rows[0]["author"] == "claude"

        archive_rows = backend.memory_conn.execute("SELECT * FROM archived_bullets").fetchall()
        assert len(archive_rows) == 1
        assert archive_rows[0]["reason"] == "pruned"

        backend.close()

    def test_migration_backs_up_files(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_2

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_playbook_files(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        migrate_phase_2(str(ccr), db_path)

        assert (ccr / "playbook.txt.bak").exists()
        assert (ccr / "failure_lessons.json.bak").exists()
        assert (ccr / "playbook_schema.json.bak").exists()
        assert (ccr / "playbook_history.json.bak").exists()
        assert (ccr / "archived_bullets.json.bak").exists()
        backend.close()

    def test_migration_no_files(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_2

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_2(str(ccr), db_path)
        assert result["migrated"] == 0
        assert result["errors"] == []
        backend.close()

    def test_migration_idempotent(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_2

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_playbook_files(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        r1 = migrate_phase_2(str(ccr), db_path)
        self._setup_playbook_files(str(ccr))
        r2 = migrate_phase_2(str(ccr), db_path)

        assert r1["errors"] == []
        assert r2["errors"] == []
        bullets = backend.bullet_list()
        assert len(bullets) == 3
        backend.close()


# ── Phase 3a: SQLite Commits ──────────────────────────────────


class TestSqliteCommits:
    def test_insert_and_get(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C001", "timestamp": "2026-04-17 10:00",
            "title": "Add storage layer", "what": "Created base.py",
            "why": "Need abstraction", "files": ["base.py", "init.py"],
            "next_step": "Implement sqlite", "score": 0.85,
            "author": "claude",
        })
        got = sqlite_backend.commit_get("main", "C001")
        assert got is not None
        assert got["id"] == "C001"
        assert got["title"] == "Add storage layer"
        assert got["what"] == "Created base.py"
        assert got["files"] == ["base.py", "init.py"]
        assert got["score"] == 0.85
        assert got["author"] == "claude"

    def test_get_missing(self, sqlite_backend):
        assert sqlite_backend.commit_get("main", "C999") is None

    def test_get_branch_isolation(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C001", "title": "Main commit",
            "what": "x", "why": "y",
        })
        sqlite_backend.commit_insert("feature", {
            "id": "C001", "title": "Feature commit",
            "what": "a", "why": "b",
        })
        assert sqlite_backend.commit_get("main", "C001")["title"] == "Main commit"
        assert sqlite_backend.commit_get("feature", "C001")["title"] == "Feature commit"

    def test_list_order_desc(self, sqlite_backend):
        for i in range(1, 6):
            sqlite_backend.commit_insert("main", {
                "id": f"C{i:03d}", "title": f"Commit {i}",
                "what": "x", "why": "y",
            })
        result = sqlite_backend.commit_list("main", limit=3)
        assert len(result) == 3
        assert result[0]["id"] == "C005"
        assert result[1]["id"] == "C004"
        assert result[2]["id"] == "C003"

    def test_list_offset(self, sqlite_backend):
        for i in range(1, 6):
            sqlite_backend.commit_insert("main", {
                "id": f"C{i:03d}", "title": f"Commit {i}",
                "what": "x", "why": "y",
            })
        result = sqlite_backend.commit_list("main", limit=2, offset=2)
        assert len(result) == 2
        assert result[0]["id"] == "C003"
        assert result[1]["id"] == "C002"

    def test_list_empty_branch(self, sqlite_backend):
        assert sqlite_backend.commit_list("nonexistent") == []

    def test_get_next_id_empty(self, sqlite_backend):
        assert sqlite_backend.commit_get_next_id("main") == "C001"

    def test_get_next_id(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C042", "title": "X", "what": "x", "why": "y",
        })
        assert sqlite_backend.commit_get_next_id("main") == "C043"

    def test_update(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C001", "title": "Old", "what": "old what", "why": "y",
        })
        assert sqlite_backend.commit_update("main", "C001", {"title": "New"})
        got = sqlite_backend.commit_get("main", "C001")
        assert got["title"] == "New"
        assert got["what"] == "old what"

    def test_update_json_fields(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C001", "title": "X", "what": "x", "why": "y",
            "files": ["a.py"],
        })
        assert sqlite_backend.commit_update("main", "C001", {
            "files": ["a.py", "b.py"],
            "patterns": ["P001"],
        })
        got = sqlite_backend.commit_get("main", "C001")
        assert got["files"] == ["a.py", "b.py"]
        assert got["patterns"] == ["P001"]

    def test_update_ci_experiment(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C001", "title": "X", "what": "x", "why": "y",
        })
        sqlite_backend.commit_update("main", "C001", {
            "ci_context": {"status": "green", "url": "http://ci/1"},
            "experiment": {"id": "exp-1", "metric": 0.9},
        })
        got = sqlite_backend.commit_get("main", "C001")
        assert got["ci_context"]["status"] == "green"
        assert got["experiment"]["metric"] == 0.9

    def test_update_missing(self, sqlite_backend):
        assert sqlite_backend.commit_update("main", "C999", {"title": "X"}) is False

    def test_search_text(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C001", "title": "Fix auth bug",
            "what": "Patched JWT validation", "why": "Security issue",
        })
        sqlite_backend.commit_insert("main", {
            "id": "C002", "title": "Add dashboard",
            "what": "Created UI components", "why": "Feature request",
        })
        results = sqlite_backend.commit_search_text("main", "auth")
        assert len(results) == 1
        assert results[0]["id"] == "C001"

    def test_search_text_across_fields(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C001", "title": "Update", "what": "x",
            "why": "security hardening", "next_step": "deploy",
        })
        assert len(sqlite_backend.commit_search_text("main", "security")) == 1
        assert len(sqlite_backend.commit_search_text("main", "deploy")) == 1

    def test_search_text_empty(self, sqlite_backend):
        assert sqlite_backend.commit_search_text("main", "nothing") == []

    def test_count(self, sqlite_backend):
        assert sqlite_backend.commit_count("main") == 0
        for i in range(1, 4):
            sqlite_backend.commit_insert("main", {
                "id": f"C{i:03d}", "title": "X", "what": "x", "why": "y",
            })
        assert sqlite_backend.commit_count("main") == 3

    def test_row_to_commit_dict_aliases(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C001", "title": "X", "what": "x", "why": "y",
            "next_step": "do more", "score": 0.7,
        })
        got = sqlite_backend.commit_get("main", "C001")
        assert got["next"] == "do more"
        assert got["stored_score"] == 0.7

    def test_optional_fields_default(self, sqlite_backend):
        sqlite_backend.commit_insert("main", {
            "id": "C001", "title": "Minimal", "what": "x", "why": "y",
        })
        got = sqlite_backend.commit_get("main", "C001")
        assert got["files"] == []
        assert got["patterns"] == []
        assert got["ci_context"] is None
        assert got["experiment"] is None
        assert got["author"] == ""


# ── Phase 3a: SQLite Rolling Summaries ─────────────────────────


class TestSqliteRollingSummary:
    def test_set_and_get(self, sqlite_backend):
        sqlite_backend.rolling_summary_set("main", "Project started with storage layer")
        got = sqlite_backend.rolling_summary_get("main")
        assert got == "Project started with storage layer"

    def test_get_empty(self, sqlite_backend):
        assert sqlite_backend.rolling_summary_get("main") == ""

    def test_overwrite(self, sqlite_backend):
        sqlite_backend.rolling_summary_set("main", "First")
        sqlite_backend.rolling_summary_set("main", "Second")
        assert sqlite_backend.rolling_summary_get("main") == "Second"

    def test_branch_isolation(self, sqlite_backend):
        sqlite_backend.rolling_summary_set("main", "Main summary")
        sqlite_backend.rolling_summary_set("feature", "Feature summary")
        assert sqlite_backend.rolling_summary_get("main") == "Main summary"
        assert sqlite_backend.rolling_summary_get("feature") == "Feature summary"


# ── Phase 3a: SQLite Branches ─────────────────────────────────


class TestSqliteBranches:
    def test_create_and_get(self, sqlite_backend):
        sqlite_backend.branch_create("feature-x", {
            "purpose": "Test new approach",
            "hypothesis": "This will improve perf",
            "parent": "main",
        })
        got = sqlite_backend.branch_get("feature-x")
        assert got is not None
        assert got["name"] == "feature-x"
        assert got["purpose"] == "Test new approach"
        assert got["parent"] == "main"
        assert got["status"] == "active"

    def test_get_missing(self, sqlite_backend):
        assert sqlite_backend.branch_get("nope") is None

    def test_list_all(self, sqlite_backend):
        sqlite_backend.branch_create("main", {"purpose": "Main dev"})
        sqlite_backend.branch_create("feat-a", {"purpose": "Feature A"})
        branches = sqlite_backend.branch_list()
        assert len(branches) == 2

    def test_list_by_status(self, sqlite_backend):
        sqlite_backend.branch_create("main", {"purpose": "Main"})
        sqlite_backend.branch_create("old", {"purpose": "Old", "status": "merged"})
        active = sqlite_backend.branch_list(status="active")
        assert len(active) == 1
        assert active[0]["name"] == "main"
        merged = sqlite_backend.branch_list(status="merged")
        assert len(merged) == 1
        assert merged[0]["name"] == "old"

    def test_update(self, sqlite_backend):
        sqlite_backend.branch_create("feat", {"purpose": "Test"})
        assert sqlite_backend.branch_update("feat", {"conclusion": "Worked well"})
        got = sqlite_backend.branch_get("feat")
        assert got["conclusion"] == "Worked well"

    def test_update_missing(self, sqlite_backend):
        assert sqlite_backend.branch_update("nope", {"purpose": "X"}) is False

    def test_update_status(self, sqlite_backend):
        sqlite_backend.branch_create("feat", {"purpose": "Test"})
        assert sqlite_backend.branch_update_status("feat", "merged")
        got = sqlite_backend.branch_get("feat")
        assert got["status"] == "merged"

    def test_update_status_missing(self, sqlite_backend):
        assert sqlite_backend.branch_update_status("nope", "merged") is False

    def test_all_fields(self, sqlite_backend):
        sqlite_backend.branch_create("full", {
            "purpose": "P", "hypothesis": "H", "parent": "main",
            "linked_issue": "#42", "team_owner": "alice",
            "priority": "high", "status": "active",
        })
        got = sqlite_backend.branch_get("full")
        assert got["linked_issue"] == "#42"
        assert got["team_owner"] == "alice"
        assert got["priority"] == "high"


# ── Phase 3a: Dual-Backend Parity ──────────────────────────────


class TestPhase3aDualBackend:
    def test_commit_get_missing(self, backend):
        assert backend.commit_get("main", "C999") is None

    def test_commit_list_empty(self, backend):
        assert backend.commit_list("main") == []

    def test_commit_count_empty(self, backend):
        assert backend.commit_count("main") == 0

    def test_commit_search_empty(self, backend):
        assert backend.commit_search_text("main", "nothing") == []

    def test_commit_get_next_id_empty(self, backend):
        assert backend.commit_get_next_id("main") == "C001"

    def test_rolling_summary_empty(self, backend):
        assert backend.rolling_summary_get("main") == ""

    def test_branch_get_missing(self, backend):
        assert backend.branch_get("nope") is None


# ── Phase 3a: Migration ────────────────────────────────────────


class TestMigratePhase3a:
    def _setup_commits_md(self, ccr_root, branch="main"):
        branch_dir = os.path.join(ccr_root, "branches", branch)
        os.makedirs(branch_dir, exist_ok=True)
        content = """# Memory: main

## Rolling Summary
Project implements a storage abstraction layer for flat-file to SQLite migration.

---

## [C003] 2026-04-17 12:00 | branch:main | Add Phase 2 tables
**What**: Created playbook bullet tables
**Why**: Need structured bullet storage
**Files**: sqlite_backend.py, base.py
**Next**: Implement Phase 3
**Patterns**: SQL schema | migration
**Score**: 0.90
**Author**: claude

## [C002] 2026-04-17 11:00 | branch:main | Implement scratchpad
**What**: Added scratchpad CRUD
**Why**: Working memory for ephemeral data
**Files**: scratchpad.py
**Next**: Add metrics
**Score**: 0.75
**Author**: claude

## [C001] 2026-04-17 10:00 | branch:main | Init project
**What**: Set up project structure
**Why**: Starting point
**Files**: (none)
**Next**: Build scratchpad
"""
        with open(os.path.join(branch_dir, "commits.md"), "w") as f:
            f.write(content)

    def _setup_metadata_yaml(self, ccr_root):
        import yaml
        meta = {
            "version": 1,
            "created": "2026-04-17",
            "branches": [
                {"name": "main", "status": "active", "purpose": "Main development"},
                {"name": "feature-x", "status": "merged", "parent": "main"},
            ],
        }
        with open(os.path.join(ccr_root, "metadata.yaml"), "w") as f:
            yaml.dump(meta, f)

    def test_full_migration(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3a

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_commits_md(str(ccr))
        self._setup_metadata_yaml(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_3a(str(ccr), db_path)

        assert result["errors"] == []
        assert result["migrated"] > 0

        commits = backend.commit_list("main", limit=10)
        assert len(commits) == 3
        assert commits[0]["id"] == "C003"
        assert commits[0]["title"] == "Add Phase 2 tables"
        assert commits[0]["score"] == 0.90

        c1 = backend.commit_get("main", "C001")
        assert c1["title"] == "Init project"
        assert c1["files"] == []

        c2 = backend.commit_get("main", "C002")
        assert c2["files"] == ["scratchpad.py"]

        summary = backend.rolling_summary_get("main")
        assert "storage abstraction" in summary

        main_branch = backend.branch_get("main")
        assert main_branch is not None
        assert main_branch["status"] == "active"

        feat_branch = backend.branch_get("feature-x")
        assert feat_branch is not None
        assert feat_branch["status"] == "merged"
        assert feat_branch["parent"] == "main"

        backend.close()

    def test_migration_no_files(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3a

        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_3a(str(ccr), db_path)
        assert result["migrated"] == 0
        assert result["errors"] == []
        backend.close()

    def test_migration_idempotent(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3a

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_commits_md(str(ccr))
        self._setup_metadata_yaml(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")

        r1 = migrate_phase_3a(str(ccr), db_path)
        self._setup_commits_md(str(ccr))
        self._setup_metadata_yaml(str(ccr))
        r2 = migrate_phase_3a(str(ccr), db_path)

        assert r1["errors"] == []
        assert r2["errors"] == []
        assert backend.commit_count("main") == 3
        backend.close()

    def test_migration_multiple_branches(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3a

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_commits_md(str(ccr), branch="main")
        self._setup_commits_md(str(ccr), branch="feature")

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_3a(str(ccr), db_path)

        assert result["errors"] == []
        assert backend.commit_count("main") == 3
        assert backend.commit_count("feature") == 3
        backend.close()

    def test_migration_commit_fields_parsed(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3a

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_commits_md(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        migrate_phase_3a(str(ccr), db_path)

        c3 = backend.commit_get("main", "C003")
        assert c3["what"] == "Created playbook bullet tables"
        assert c3["why"] == "Need structured bullet storage"
        assert c3["files"] == ["sqlite_backend.py", "base.py"]
        assert c3["next"] == "Implement Phase 3"
        assert "SQL schema" in c3["patterns"]
        assert "migration" in c3["patterns"]
        assert c3["author"] == "claude"
        backend.close()

    def test_auto_migrate_includes_phase_3a(self, tmp_path):
        from ccr.core.storage.migration import auto_migrate

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_commits_md(str(ccr))
        self._setup_metadata_yaml(str(ccr))

        os.makedirs(os.path.join(str(ccr), "branches", "main"), exist_ok=True)
        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = auto_migrate(str(ccr), db_path)

        assert "3a" in result["phases_run"]
        assert result["errors"] == []
        assert backend.commit_count("main") == 3
        backend.close()


# ── Phase 3b: Links ──────────────────────────────────────────────


class TestSqliteLinks:
    def test_insert_and_get(self, sqlite_backend):
        sqlite_backend.link_insert_batch("C001", [
            {"target": "C002", "link_type": "entity", "score": 0.9,
             "shared_files": ["a.py"], "snippet": "shared entity"},
        ])
        got = sqlite_backend.link_get_for_commit("C001")
        assert "entity" in got
        assert got["entity"][0]["target"] == "C002"
        assert got["entity"][0]["score"] == 0.9

    def test_get_missing(self, sqlite_backend):
        got = sqlite_backend.link_get_for_commit("C999")
        assert got == {}

    def test_multiple_link_types(self, sqlite_backend):
        sqlite_backend.link_insert_batch("C001", [
            {"target": "C002", "link_type": "entity", "score": 0.8},
            {"target": "C003", "link_type": "causal", "score": 0.7},
            {"target": "C004", "link_type": "semantic", "score": 0.6},
        ])
        got = sqlite_backend.link_get_for_commit("C001")
        assert len(got) == 3
        assert "entity" in got
        assert "causal" in got
        assert "semantic" in got

    def test_get_all(self, sqlite_backend):
        sqlite_backend.link_insert_batch("C001", [
            {"target": "C002", "link_type": "entity", "score": 0.9},
        ])
        sqlite_backend.link_insert_batch("C003", [
            {"target": "C004", "link_type": "causal", "score": 0.5},
        ])
        all_links = sqlite_backend.link_get_all()
        assert "C001" in all_links
        assert "C003" in all_links

    def test_replace_on_duplicate(self, sqlite_backend):
        sqlite_backend.link_insert_batch("C001", [
            {"target": "C002", "link_type": "entity", "score": 0.5},
        ])
        sqlite_backend.link_insert_batch("C001", [
            {"target": "C002", "link_type": "entity", "score": 0.9},
        ])
        got = sqlite_backend.link_get_for_commit("C001")
        assert got["entity"][0]["score"] == 0.9

    def test_prune(self, sqlite_backend):
        for i in range(1, 6):
            sqlite_backend.link_insert_batch(f"C{i:03d}", [
                {"target": f"C{i+10:03d}", "link_type": "entity", "score": 0.5},
            ])
        evicted = sqlite_backend.link_prune(max_nodes=3)
        assert evicted == 2
        all_links = sqlite_backend.link_get_all()
        assert len(all_links) <= 3

    def test_prune_empty(self, sqlite_backend):
        assert sqlite_backend.link_prune(max_nodes=10) == 0

    def test_prune_under_limit(self, sqlite_backend):
        sqlite_backend.link_insert_batch("C001", [
            {"target": "C002", "link_type": "entity", "score": 0.5},
        ])
        assert sqlite_backend.link_prune(max_nodes=10) == 0

    def test_bidirectional_get(self, sqlite_backend):
        sqlite_backend.link_insert_batch("C001", [
            {"target": "C002", "link_type": "entity", "score": 0.8},
        ])
        got = sqlite_backend.link_get_for_commit("C002")
        assert "entity" in got
        assert got["entity"][0]["target"] == "C001"

    def test_shared_files_none(self, sqlite_backend):
        sqlite_backend.link_insert_batch("C001", [
            {"target": "C002", "link_type": "causal", "score": 0.7},
        ])
        got = sqlite_backend.link_get_for_commit("C001")
        assert got["causal"][0].get("shared_files") is None


# ── Phase 3b: Patterns ───────────────────────────────────────────


class TestSqlitePatterns:
    def test_load_all_empty(self, sqlite_backend):
        got = sqlite_backend.pattern_load_all()
        assert got["patterns"] == {}
        assert got["next_id"] == 1

    def test_save_and_load_roundtrip(self, sqlite_backend):
        data = {
            "version": 1,
            "next_id": 3,
            "patterns": {
                "P001": {
                    "text": "test pattern",
                    "commit_ids": ["C001", "C002"],
                    "occurrence_count": 2,
                    "promoted": True,
                    "success_count": 5,
                    "failure_count": 1,
                    "quality_score": 0.8,
                    "first_seen": "2026-04-17",
                    "last_seen": "2026-04-18",
                    "created_at": "2026-04-17",
                },
                "P002": {
                    "text": "another pattern",
                    "commit_ids": ["C003"],
                    "occurrence_count": 1,
                    "promoted": False,
                    "success_count": 0,
                    "failure_count": 0,
                    "quality_score": 0.5,
                    "created_at": "2026-04-18",
                },
            },
        }
        sqlite_backend.pattern_save_all(data)
        got = sqlite_backend.pattern_load_all()
        assert got["next_id"] == 3
        assert len(got["patterns"]) == 2
        assert got["patterns"]["P001"]["text"] == "test pattern"
        assert got["patterns"]["P001"]["promoted"] is True
        assert got["patterns"]["P001"]["commit_ids"] == ["C001", "C002"]
        assert got["patterns"]["P002"]["promoted"] is False

    def test_get_by_id(self, sqlite_backend):
        data = {
            "version": 1, "next_id": 2,
            "patterns": {
                "P001": {"text": "p1", "commit_ids": [], "occurrence_count": 1,
                         "promoted": False, "quality_score": 0.5, "created_at": "now"},
            },
        }
        sqlite_backend.pattern_save_all(data)
        p = sqlite_backend.pattern_get("P001")
        assert p is not None
        assert p["text"] == "p1"

    def test_get_missing(self, sqlite_backend):
        assert sqlite_backend.pattern_get("P999") is None

    def test_update(self, sqlite_backend):
        data = {
            "version": 1, "next_id": 2,
            "patterns": {
                "P001": {"text": "old", "commit_ids": ["C001"], "occurrence_count": 1,
                         "promoted": False, "quality_score": 0.5, "created_at": "now"},
            },
        }
        sqlite_backend.pattern_save_all(data)
        ok = sqlite_backend.pattern_update("P001", {
            "text": "updated",
            "quality_score": 0.9,
            "promoted": True,
            "commit_ids": ["C001", "C002"],
        })
        assert ok is True
        p = sqlite_backend.pattern_get("P001")
        assert p["text"] == "updated"
        assert p["quality_score"] == 0.9
        assert p["promoted"] is True
        assert p["commit_ids"] == ["C001", "C002"]

    def test_update_missing(self, sqlite_backend):
        assert sqlite_backend.pattern_update("P999", {"text": "x"}) is False

    def test_get_next_id(self, sqlite_backend):
        assert sqlite_backend.pattern_get_next_id() == 1
        data = {
            "version": 1, "next_id": 42,
            "patterns": {},
        }
        sqlite_backend.pattern_save_all(data)
        assert sqlite_backend.pattern_get_next_id() == 42

    def test_save_all_replaces(self, sqlite_backend):
        data1 = {
            "version": 1, "next_id": 2,
            "patterns": {
                "P001": {"text": "first", "commit_ids": [], "occurrence_count": 1,
                         "promoted": False, "quality_score": 0.5, "created_at": "now"},
            },
        }
        sqlite_backend.pattern_save_all(data1)
        data2 = {
            "version": 1, "next_id": 3,
            "patterns": {
                "P002": {"text": "second", "commit_ids": [], "occurrence_count": 1,
                         "promoted": False, "quality_score": 0.5, "created_at": "now"},
            },
        }
        sqlite_backend.pattern_save_all(data2)
        got = sqlite_backend.pattern_load_all()
        assert "P001" not in got["patterns"]
        assert "P002" in got["patterns"]


# ── Phase 3b: Triples ────────────────────────────────────────────


class TestSqliteTriples:
    def test_insert_and_list(self, sqlite_backend):
        count = sqlite_backend.triple_insert_batch([
            {"subject": "A", "predicate": "uses", "object": "B",
             "source_commit": "C001", "confidence": 0.9},
        ])
        assert count == 1
        triples = sqlite_backend.triple_list(top_k=10)
        assert len(triples) == 1
        assert triples[0]["subject"] == "A"
        assert triples[0]["predicate"] == "uses"

    def test_dedup(self, sqlite_backend):
        t = {"subject": "A", "predicate": "uses", "object": "B",
             "source_commit": "C001", "confidence": 0.9}
        c1 = sqlite_backend.triple_insert_batch([t])
        c2 = sqlite_backend.triple_insert_batch([t])
        assert c1 == 1
        assert c2 == 0
        assert sqlite_backend.triple_count() == 1

    def test_count(self, sqlite_backend):
        assert sqlite_backend.triple_count() == 0
        sqlite_backend.triple_insert_batch([
            {"subject": "A", "predicate": "uses", "object": "B",
             "source_commit": "C001", "confidence": 0.9},
            {"subject": "C", "predicate": "calls", "object": "D",
             "source_commit": "C002", "confidence": 0.8},
        ])
        assert sqlite_backend.triple_count() == 2

    def test_list_by_commit(self, sqlite_backend):
        sqlite_backend.triple_insert_batch([
            {"subject": "A", "predicate": "uses", "object": "B",
             "source_commit": "C001", "confidence": 0.9},
            {"subject": "C", "predicate": "calls", "object": "D",
             "source_commit": "C002", "confidence": 0.8},
        ])
        got = sqlite_backend.triple_list(top_k=10, commit_id="C001")
        assert len(got) == 1
        assert got[0]["subject"] == "A"

    def test_list_by_entity(self, sqlite_backend):
        sqlite_backend.triple_insert_batch([
            {"subject": "A", "predicate": "uses", "object": "B",
             "source_commit": "C001", "confidence": 0.9},
            {"subject": "C", "predicate": "calls", "object": "A",
             "source_commit": "C002", "confidence": 0.8},
        ])
        got = sqlite_backend.triple_list(top_k=10, entity="A")
        assert len(got) == 2

    def test_search(self, sqlite_backend):
        sqlite_backend.triple_insert_batch([
            {"subject": "MemoryManager", "predicate": "extends", "object": "BaseMixin",
             "source_commit": "C001", "confidence": 0.9},
            {"subject": "Indexer", "predicate": "calls", "object": "FTS5",
             "source_commit": "C002", "confidence": 0.8},
        ])
        got = sqlite_backend.triple_search("Memory", top_k=10)
        assert len(got) == 1
        assert got[0]["subject"] == "MemoryManager"

    def test_search_no_results(self, sqlite_backend):
        assert sqlite_backend.triple_search("nonexistent") == []

    def test_top_k_limit(self, sqlite_backend):
        triples = [
            {"subject": f"E{i}", "predicate": "p", "object": f"O{i}",
             "source_commit": f"C{i:03d}", "confidence": 0.5}
            for i in range(20)
        ]
        sqlite_backend.triple_insert_batch(triples)
        got = sqlite_backend.triple_list(top_k=5)
        assert len(got) == 5


# ── Phase 3b: Evolved Summaries ──────────────────────────────────


class TestSqliteEvolution:
    def test_set_and_get(self, sqlite_backend):
        data = {
            "evolved_what": "Improved implementation",
            "evolution_reason": "Found better approach",
            "evolved_at": "2026-04-18",
            "source_commit_id": "C001",
            "original_what": "Basic implementation",
        }
        sqlite_backend.evolved_summary_set("C002", data)
        got = sqlite_backend.evolved_summary_get("C002")
        assert got is not None
        assert got["evolved_what"] == "Improved implementation"
        assert got["source_commit_id"] == "C001"

    def test_get_missing(self, sqlite_backend):
        assert sqlite_backend.evolved_summary_get("C999") is None

    def test_overwrite(self, sqlite_backend):
        sqlite_backend.evolved_summary_set("C001", {
            "evolved_what": "v1", "evolution_reason": "r1",
            "evolved_at": "t1", "source_commit_id": "C000",
            "original_what": "o1",
        })
        sqlite_backend.evolved_summary_set("C001", {
            "evolved_what": "v2", "evolution_reason": "r2",
            "evolved_at": "t2", "source_commit_id": "C000",
            "original_what": "o1",
        })
        got = sqlite_backend.evolved_summary_get("C001")
        assert got["evolved_what"] == "v2"

    def test_all(self, sqlite_backend):
        sqlite_backend.evolved_summary_set("C001", {
            "evolved_what": "e1", "evolution_reason": "r1",
            "evolved_at": "t1", "source_commit_id": "C000",
            "original_what": "o1",
        })
        sqlite_backend.evolved_summary_set("C002", {
            "evolved_what": "e2", "evolution_reason": "r2",
            "evolved_at": "t2", "source_commit_id": "C001",
            "original_what": "o2",
        })
        all_ev = sqlite_backend.evolved_summary_all()
        assert len(all_ev) == 2
        assert "C001" in all_ev
        assert "C002" in all_ev

    def test_all_empty(self, sqlite_backend):
        assert sqlite_backend.evolved_summary_all() == {}


# ── Phase 3b: Clusters ───────────────────────────────────────────


class TestSqliteClusters:
    def test_save_and_load(self, sqlite_backend):
        clusters = [
            {"name": "Core", "commit_ids": ["C001", "C002"], "top_keywords": ["memory"]},
            {"name": "Tests", "commit_ids": ["C003"], "top_keywords": ["pytest"]},
        ]
        sqlite_backend.cluster_save(clusters)
        got = sqlite_backend.cluster_load()
        assert len(got["clusters"]) == 2
        assert got["clusters"][0]["name"] in ("Core", "Tests")
        assert "C001" in got["commit_to_cluster"]
        assert "C003" in got["commit_to_cluster"]

    def test_load_empty(self, sqlite_backend):
        got = sqlite_backend.cluster_load()
        assert got["clusters"] == []
        assert got["commit_to_cluster"] == {}

    def test_overwrite(self, sqlite_backend):
        sqlite_backend.cluster_save([
            {"name": "Old", "commit_ids": ["C001"]},
        ])
        sqlite_backend.cluster_save([
            {"name": "New", "commit_ids": ["C002", "C003"]},
        ])
        got = sqlite_backend.cluster_load()
        assert len(got["clusters"]) == 1
        assert got["clusters"][0]["name"] == "New"
        assert "C001" not in got["commit_to_cluster"]

    def test_cluster_mapping(self, sqlite_backend):
        sqlite_backend.cluster_save([
            {"name": "A", "commit_ids": ["C001", "C002"]},
            {"name": "B", "commit_ids": ["C003"]},
        ])
        got = sqlite_backend.cluster_load()
        a_id = got["commit_to_cluster"]["C001"]
        assert got["commit_to_cluster"]["C002"] == a_id
        assert got["commit_to_cluster"]["C003"] != a_id


# ── Phase 3b: Dual-Backend Parity ────────────────────────────────


class TestPhase3bDualBackend:
    def test_link_get_missing(self, backend):
        assert backend.link_get_for_commit("C999") == {}

    def test_link_get_all_empty(self, backend):
        assert backend.link_get_all() == {}

    def test_pattern_load_empty(self, backend):
        got = backend.pattern_load_all()
        assert got["patterns"] == {}

    def test_triple_count_empty(self, backend):
        assert backend.triple_count() == 0

    def test_evolved_summary_missing(self, backend):
        assert backend.evolved_summary_get("C999") is None

    def test_cluster_load_empty(self, backend):
        got = backend.cluster_load()
        assert got["clusters"] == []
        assert got["commit_to_cluster"] == {}

    def test_link_roundtrip(self, backend):
        backend.link_insert_batch("C001", [
            {"target": "C002", "link_type": "entity", "score": 0.8},
        ])
        got = backend.link_get_for_commit("C001")
        assert "entity" in got
        assert got["entity"][0]["target"] == "C002"

    def test_triple_roundtrip(self, backend):
        count = backend.triple_insert_batch([
            {"subject": "X", "predicate": "uses", "object": "Y",
             "source_commit": "C001", "confidence": 0.9},
        ])
        assert count == 1
        assert backend.triple_count() == 1

    def test_pattern_roundtrip(self, backend):
        backend.pattern_save_all({
            "version": 1, "next_id": 2,
            "patterns": {
                "P001": {"text": "t", "commit_ids": [], "occurrence_count": 1,
                         "promoted": False, "quality_score": 0.5, "created_at": "now"},
            },
        })
        got = backend.pattern_load_all()
        assert "P001" in got["patterns"]

    def test_evolved_summary_roundtrip(self, backend):
        backend.evolved_summary_set("C001", {
            "evolved_what": "e", "evolution_reason": "r",
            "evolved_at": "t", "source_commit_id": "C000",
            "original_what": "o",
        })
        got = backend.evolved_summary_get("C001")
        assert got["evolved_what"] == "e"


# ── Phase 3b: Migration ──────────────────────────────────────────


class TestMigratePhase3b:
    def _write_json(self, ccr_root, filename, data):
        path = os.path.join(ccr_root, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_full_migration(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3b

        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        self._write_json(str(ccr), "commit_links.json", {
            "version": 1,
            "links": {
                "C001": {
                    "entity": [{"target": "C002", "score": 0.9, "shared_files": ["a.py"]}],
                    "causal": [{"target": "C003", "score": 0.7}],
                },
            },
        })
        self._write_json(str(ccr), "patterns.json", {
            "version": 1, "next_id": 3,
            "patterns": {
                "P001": {"text": "SQL migration", "commit_ids": ["C001"],
                         "occurrence_count": 2, "promoted": True,
                         "quality_score": 0.8, "created_at": "2026-04-17"},
                "P002": {"text": "test-first", "commit_ids": [],
                         "occurrence_count": 1, "promoted": False,
                         "quality_score": 0.5, "created_at": "2026-04-18"},
            },
        })
        self._write_json(str(ccr), "triples.json", {
            "version": 1,
            "triples": [
                {"subject": "A", "predicate": "uses", "object": "B",
                 "source_commit": "C001", "confidence": 0.9},
                {"subject": "C", "predicate": "calls", "object": "D",
                 "source_commit": "C002", "confidence": 0.8},
            ],
        })
        self._write_json(str(ccr), "evolved_summaries.json", {
            "version": 1,
            "evolved": {
                "C002": {
                    "evolved_what": "Better approach",
                    "evolution_reason": "Found optimization",
                    "evolved_at": "2026-04-18",
                    "source_commit_id": "C001",
                    "original_what": "Basic approach",
                },
            },
        })
        self._write_json(str(ccr), "commit_clusters.json", {
            "version": 1,
            "clusters": [
                {"id": 1, "name": "Core", "commit_ids": ["C001", "C002"],
                 "top_keywords": ["memory", "storage"]},
            ],
            "commit_to_cluster": {"C001": 1, "C002": 1},
        })

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_3b(str(ccr), db_path)

        assert result["errors"] == []
        assert result["migrated"] > 0

        links = backend.link_get_for_commit("C001")
        assert "entity" in links
        assert links["entity"][0]["target"] == "C002"

        patterns = backend.pattern_load_all()
        assert len(patterns["patterns"]) == 2
        assert patterns["next_id"] == 3
        assert patterns["patterns"]["P001"]["text"] == "SQL migration"

        assert backend.triple_count() == 2

        ev = backend.evolved_summary_get("C002")
        assert ev["evolved_what"] == "Better approach"

        cl = backend.cluster_load()
        assert len(cl["clusters"]) == 1
        assert "C001" in cl["commit_to_cluster"]

        backend.close()

    def test_migration_no_files(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3b

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_3b(str(ccr), db_path)
        assert result["migrated"] == 0
        assert result["errors"] == []
        backend.close()

    def test_migration_idempotent(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3b

        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        self._write_json(str(ccr), "triples.json", {
            "version": 1,
            "triples": [
                {"subject": "A", "predicate": "uses", "object": "B",
                 "source_commit": "C001", "confidence": 0.9},
            ],
        })

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")

        r1 = migrate_phase_3b(str(ccr), db_path)
        self._write_json(str(ccr), "triples.json", {
            "version": 1,
            "triples": [
                {"subject": "A", "predicate": "uses", "object": "B",
                 "source_commit": "C001", "confidence": 0.9},
            ],
        })
        r2 = migrate_phase_3b(str(ccr), db_path)

        assert r1["errors"] == []
        assert r2["errors"] == []
        assert backend.triple_count() == 1
        backend.close()

    def test_migration_backs_up_files(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3b

        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        self._write_json(str(ccr), "patterns.json", {
            "version": 1, "next_id": 2,
            "patterns": {"P001": {"text": "t", "commit_ids": [],
                                  "occurrence_count": 1, "promoted": False,
                                  "quality_score": 0.5, "created_at": "now"}},
        })

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        migrate_phase_3b(str(ccr), db_path)

        assert os.path.isfile(os.path.join(str(ccr), "patterns.json.bak"))
        assert not os.path.isfile(os.path.join(str(ccr), "patterns.json"))
        backend.close()

    def test_auto_migrate_includes_3b(self, tmp_path):
        from ccr.core.storage.migration import auto_migrate

        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        import yaml
        meta = {"version": 1, "created": "2026-04-17", "branches": []}
        with open(os.path.join(str(ccr), "metadata.yaml"), "w") as f:
            yaml.dump(meta, f)

        self._write_json(str(ccr), "triples.json", {
            "version": 1,
            "triples": [
                {"subject": "X", "predicate": "p", "object": "Y",
                 "source_commit": "C001", "confidence": 0.5},
            ],
        })

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = auto_migrate(str(ccr), db_path)

        assert "3b" in result["phases_run"]
        assert result["errors"] == []
        assert backend.triple_count() == 1
        backend.close()


# ── Phase 3c: Discussions ───────────────────────────────────────


class TestSqliteDiscussions:
    def test_insert_and_list(self, sqlite_backend):
        sqlite_backend.discussion_insert("main", {
            "id": "D001", "timestamp": "2026-04-18 10:00",
            "topic": "SQLite migration", "hypothesis": "Will improve perf",
            "alternatives": "Keep flat files", "decision": "Migrate",
            "rationale": "Better queries", "uncertainty": "Low",
            "linked_commit": "C042",
        })
        got = sqlite_backend.discussion_list("main")
        assert len(got) == 1
        assert got[0]["id"] == "D001"
        assert got[0]["topic"] == "SQLite migration"
        assert got[0]["decision"] == "Migrate"
        assert got[0]["linked_commit"] == "C042"

    def test_list_empty(self, sqlite_backend):
        assert sqlite_backend.discussion_list("main") == []

    def test_branch_isolation(self, sqlite_backend):
        sqlite_backend.discussion_insert("main", {
            "id": "D001", "timestamp": "2026-04-18 10:00", "topic": "Main topic",
        })
        sqlite_backend.discussion_insert("feature", {
            "id": "D001", "timestamp": "2026-04-18 11:00", "topic": "Feature topic",
        })
        main = sqlite_backend.discussion_list("main")
        feature = sqlite_backend.discussion_list("feature")
        assert len(main) == 1
        assert main[0]["topic"] == "Main topic"
        assert len(feature) == 1
        assert feature[0]["topic"] == "Feature topic"

    def test_filter_search(self, sqlite_backend):
        sqlite_backend.discussion_insert("main", {
            "id": "D001", "timestamp": "2026-04-18 10:00",
            "topic": "SQLite migration", "decision": "Yes",
        })
        sqlite_backend.discussion_insert("main", {
            "id": "D002", "timestamp": "2026-04-18 11:00",
            "topic": "Indexer rewrite", "decision": "No",
        })
        got = sqlite_backend.discussion_list("main", search="SQLite")
        assert len(got) == 1
        assert got[0]["id"] == "D001"

    def test_filter_topic(self, sqlite_backend):
        sqlite_backend.discussion_insert("main", {
            "id": "D001", "topic": "Architecture", "timestamp": "2026-04-18 10:00",
        })
        sqlite_backend.discussion_insert("main", {
            "id": "D002", "topic": "Testing", "timestamp": "2026-04-18 11:00",
        })
        got = sqlite_backend.discussion_list("main", topic="Arch")
        assert len(got) == 1
        assert got[0]["id"] == "D001"

    def test_filter_date_range(self, sqlite_backend):
        sqlite_backend.discussion_insert("main", {
            "id": "D001", "timestamp": "2026-04-17 10:00", "topic": "Old",
        })
        sqlite_backend.discussion_insert("main", {
            "id": "D002", "timestamp": "2026-04-18 10:00", "topic": "New",
        })
        got = sqlite_backend.discussion_list(
            "main", date_range=["2026-04-18 00:00", "2026-04-18 23:59"],
        )
        assert len(got) == 1
        assert got[0]["id"] == "D002"

    def test_get_next_id_empty(self, sqlite_backend):
        assert sqlite_backend.discussion_get_next_id("main") == "D001"

    def test_get_next_id_after_inserts(self, sqlite_backend):
        sqlite_backend.discussion_insert("main", {
            "id": "D003", "timestamp": "2026-04-18 10:00", "topic": "Test",
        })
        assert sqlite_backend.discussion_get_next_id("main") == "D004"

    def test_order_desc(self, sqlite_backend):
        for i in range(1, 4):
            sqlite_backend.discussion_insert("main", {
                "id": f"D{i:03d}", "timestamp": f"2026-04-18 {i:02d}:00",
                "topic": f"Topic {i}",
            })
        got = sqlite_backend.discussion_list("main")
        assert [d["id"] for d in got] == ["D003", "D002", "D001"]


# ── Phase 3c: Session Summaries ─────────────────────────────────


class TestSqliteSessionSummaries:
    def test_insert_and_list(self, sqlite_backend):
        sqlite_backend.session_summary_insert("main", {
            "id": "S001", "start_date": "2026-04-18 09:00",
            "end_date": "2026-04-18 17:00", "commit_range": "C001-C005",
            "accomplished": "Built storage layer",
            "files_touched": "5 files", "key_decisions": "Use WAL mode",
            "direction": "Continue with Phase 3c",
        })
        got = sqlite_backend.session_summary_list("main")
        assert len(got) == 1
        assert got[0]["id"] == "S001"
        assert got[0]["accomplished"] == "Built storage layer"
        assert got[0]["branch"] == "main"

    def test_list_empty(self, sqlite_backend):
        assert sqlite_backend.session_summary_list("main") == []

    def test_count_limit(self, sqlite_backend):
        for i in range(1, 6):
            sqlite_backend.session_summary_insert("main", {
                "id": f"S{i:03d}", "start_date": f"2026-04-{i:02d} 09:00",
                "end_date": f"2026-04-{i:02d} 17:00",
            })
        got = sqlite_backend.session_summary_list("main", count=2)
        assert len(got) == 2
        assert got[0]["id"] == "S005"
        assert got[1]["id"] == "S004"

    def test_branch_isolation(self, sqlite_backend):
        sqlite_backend.session_summary_insert("main", {
            "id": "S001", "start_date": "2026-04-18 09:00",
            "end_date": "2026-04-18 17:00",
        })
        sqlite_backend.session_summary_insert("dev", {
            "id": "S001", "start_date": "2026-04-18 10:00",
            "end_date": "2026-04-18 18:00",
        })
        assert len(sqlite_backend.session_summary_list("main")) == 1
        assert len(sqlite_backend.session_summary_list("dev")) == 1

    def test_get_next_id_empty(self, sqlite_backend):
        assert sqlite_backend.session_summary_get_next_id("main") == "S001"

    def test_get_next_id_after_inserts(self, sqlite_backend):
        sqlite_backend.session_summary_insert("main", {
            "id": "S005", "start_date": "2026-04-18 09:00",
            "end_date": "2026-04-18 17:00",
        })
        assert sqlite_backend.session_summary_get_next_id("main") == "S006"


# ── Phase 3c: Phase Summaries ───────────────────────────────────


class TestSqlitePhaseSummaries:
    def test_insert_and_list(self, sqlite_backend):
        sqlite_backend.phase_summary_insert({
            "id": "P001", "start_date": "2026-04-01 00:00",
            "end_date": "2026-04-15 00:00", "scope": "Storage migration",
            "goal": "Move to SQLite", "outcome": "Success",
            "accomplishments": "4 phases done",
            "files_changed": "20 files",
            "branch_summary": "All on main",
        })
        got = sqlite_backend.phase_summary_list()
        assert len(got) == 1
        assert got[0]["id"] == "P001"
        assert got[0]["scope"] == "Storage migration"
        assert got[0]["goal"] == "Move to SQLite"

    def test_list_empty(self, sqlite_backend):
        assert sqlite_backend.phase_summary_list() == []

    def test_count_limit(self, sqlite_backend):
        for i in range(1, 6):
            sqlite_backend.phase_summary_insert({
                "id": f"P{i:03d}", "start_date": f"2026-0{i}-01 00:00",
                "end_date": f"2026-0{i}-15 00:00",
            })
        got = sqlite_backend.phase_summary_list(count=2)
        assert len(got) == 2
        assert got[0]["id"] == "P005"

    def test_get_next_id_empty(self, sqlite_backend):
        assert sqlite_backend.phase_summary_get_next_id() == "P001"

    def test_get_next_id_after_inserts(self, sqlite_backend):
        sqlite_backend.phase_summary_insert({
            "id": "P003", "start_date": "2026-04-01 00:00",
            "end_date": "2026-04-15 00:00",
        })
        assert sqlite_backend.phase_summary_get_next_id() == "P004"


# ── Phase 3c: Summary Meta ─────────────────────────────────────


class TestSqliteSummaryMeta:
    def test_load_default(self, sqlite_backend):
        got = sqlite_backend.summary_meta_load()
        assert got["version"] == 1
        assert "session" in got
        assert "phase" in got

    def test_save_and_load(self, sqlite_backend):
        data = {
            "version": 1,
            "session": {"main": {"last_commit_id": "C010"}},
            "phase": {"last_commit_id": "C010", "last_summary_id": "P002"},
            "overview": {"last_generated": "2026-04-18"},
        }
        sqlite_backend.summary_meta_save(data)
        got = sqlite_backend.summary_meta_load()
        assert got["session"]["main"]["last_commit_id"] == "C010"
        assert got["phase"]["last_summary_id"] == "P002"

    def test_overwrite(self, sqlite_backend):
        sqlite_backend.summary_meta_save({"version": 1, "session": {"a": 1}})
        sqlite_backend.summary_meta_save({"version": 2, "session": {"b": 2}})
        got = sqlite_backend.summary_meta_load()
        assert got["version"] == 2
        assert "b" in got["session"]


# ── Phase 3c: Project State ────────────────────────────────────


class TestSqliteProjectState:
    def test_get_missing(self, sqlite_backend):
        assert sqlite_backend.project_state_get("overview") is None

    def test_set_and_get(self, sqlite_backend):
        sqlite_backend.project_state_set("overview", "# Project Overview\nThis project...")
        got = sqlite_backend.project_state_get("overview")
        assert "Project Overview" in got

    def test_overwrite(self, sqlite_backend):
        sqlite_backend.project_state_set("overview", "v1")
        sqlite_backend.project_state_set("overview", "v2")
        assert sqlite_backend.project_state_get("overview") == "v2"

    def test_multiple_keys(self, sqlite_backend):
        sqlite_backend.project_state_set("overview", "the overview")
        sqlite_backend.project_state_set("focus", "current focus")
        assert sqlite_backend.project_state_get("overview") == "the overview"
        assert sqlite_backend.project_state_get("focus") == "current focus"


# ── Phase 3c: Dual-Backend Parity ──────────────────────────────


class TestPhase3cDualBackend:
    def test_discussion_list_empty(self, backend):
        assert backend.discussion_list("main") == []

    def test_discussion_get_next_id_empty(self, backend):
        assert backend.discussion_get_next_id("main") == "D001"

    def test_session_summary_list_empty(self, backend):
        assert backend.session_summary_list("main") == []

    def test_session_summary_get_next_id_empty(self, backend):
        assert backend.session_summary_get_next_id("main") == "S001"

    def test_phase_summary_list_empty(self, backend):
        assert backend.phase_summary_list() == []

    def test_phase_summary_get_next_id_empty(self, backend):
        assert backend.phase_summary_get_next_id() == "P001"

    def test_summary_meta_default(self, backend):
        got = backend.summary_meta_load()
        assert "version" in got

    def test_project_state_missing(self, backend):
        assert backend.project_state_get("nonexistent") is None

    def test_discussion_roundtrip(self, backend):
        backend.discussion_insert("main", {
            "id": "D001", "timestamp": "2026-04-18 10:00",
            "topic": "Test topic", "decision": "Approved",
        })
        got = backend.discussion_list("main")
        assert len(got) == 1
        assert got[0]["topic"] == "Test topic"

    def test_session_summary_roundtrip(self, backend):
        backend.session_summary_insert("main", {
            "id": "S001", "start_date": "2026-04-18 09:00",
            "end_date": "2026-04-18 17:00",
            "accomplished": "Tests written",
        })
        got = backend.session_summary_list("main")
        assert len(got) == 1
        assert got[0]["accomplished"] == "Tests written"

    def test_phase_summary_roundtrip(self, backend):
        backend.phase_summary_insert({
            "id": "P001", "start_date": "2026-04-01 00:00",
            "end_date": "2026-04-15 00:00", "goal": "Migrate storage",
        })
        got = backend.phase_summary_list()
        assert len(got) == 1
        assert got[0]["goal"] == "Migrate storage"

    def test_summary_meta_roundtrip(self, backend):
        data = {"version": 1, "session": {"x": 1}, "phase": {}}
        backend.summary_meta_save(data)
        got = backend.summary_meta_load()
        assert got["session"]["x"] == 1

    def test_project_state_roundtrip(self, backend):
        backend.project_state_set("overview", "Hello world")
        assert backend.project_state_get("overview") == "Hello world"


# ── Phase 3c: Migration ────────────────────────────────────────


class TestMigratePhase3c:
    def test_full_migration(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3c

        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        branch_dir = ccr / "branches" / "main"
        branch_dir.mkdir(parents=True)

        (branch_dir / "discussions.md").write_text(
            "## [D001] 2026-04-18 10:00 | SQLite migration\n"
            "**Hypothesis**: Will improve perf\n"
            "**Alternatives**: Keep flat files\n"
            "**Decision**: Migrate\n"
            "**Rationale**: Better queries\n"
            "**Uncertainty**: Low\n"
            "**Linked Commit**: C042\n"
            "\n---\n\n"
            "## [D002] 2026-04-17 14:00 | Index design\n"
            "**Hypothesis**: FTS5 is fast enough\n"
            "**Alternatives**: BM25 only\n"
            "**Decision**: Use FTS5 with fallback\n"
            "**Rationale**: Better search quality\n",
            encoding="utf-8",
        )

        (branch_dir / "summaries.md").write_text(
            "## [S001] 2026-04-18 09:00 - 2026-04-18 17:00 | main | Session Summary\n"
            "**Commits**: C001-C005\n"
            "**Accomplished**: Built storage layer\n"
            "**Files touched**: 5 files\n"
            "**Key decisions**: Use WAL mode\n"
            "**Direction**: Continue with Phase 3c\n",
            encoding="utf-8",
        )

        summaries_dir = ccr / "summaries"
        summaries_dir.mkdir()
        (summaries_dir / "phases.md").write_text(
            "## [P001] 2026-04-01 00:00 - 2026-04-15 00:00 | Phase Summary\n"
            "**Scope**: Storage migration\n"
            "**Goal**: Move to SQLite\n"
            "**Outcome**: Success\n"
            "**Key accomplishments**: 4 phases done\n"
            "**Files changed**: 20 files\n"
            "**Branch summary**: All on main\n",
            encoding="utf-8",
        )

        import yaml
        (ccr / "summary_meta.yaml").write_text(
            yaml.dump({"version": 1, "session": {"main": {"last": "C005"}}}),
            encoding="utf-8",
        )

        (ccr / "overview.md").write_text(
            "# Project Overview\nCCR migration project.\n",
            encoding="utf-8",
        )

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_3c(str(ccr), db_path)

        assert result["errors"] == []
        assert result["migrated"] >= 5

        discs = backend.discussion_list("main")
        assert len(discs) == 2
        assert discs[0]["topic"] in ("SQLite migration", "Index design")

        sessions = backend.session_summary_list("main")
        assert len(sessions) == 1
        assert sessions[0]["accomplished"] == "Built storage layer"

        phases = backend.phase_summary_list()
        assert len(phases) == 1
        assert phases[0]["scope"] == "Storage migration"

        meta = backend.summary_meta_load()
        assert meta["session"]["main"]["last"] == "C005"

        overview = backend.project_state_get("overview")
        assert "Project Overview" in overview

        backend.close()

    def test_migration_no_files(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3c

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = migrate_phase_3c(str(ccr), db_path)
        assert result["migrated"] == 0
        assert result["errors"] == []
        backend.close()

    def test_migration_idempotent(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3c

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        branch_dir = ccr / "branches" / "main"
        branch_dir.mkdir(parents=True)

        disc_content = (
            "## [D001] 2026-04-18 10:00 | Test topic\n"
            "**Hypothesis**: H\n"
            "**Alternatives**: A\n"
            "**Decision**: D\n"
            "**Rationale**: R\n"
        )
        (branch_dir / "discussions.md").write_text(disc_content, encoding="utf-8")

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")

        r1 = migrate_phase_3c(str(ccr), db_path)
        (branch_dir / "discussions.md").write_text(disc_content, encoding="utf-8")
        r2 = migrate_phase_3c(str(ccr), db_path)

        assert r1["errors"] == []
        assert r2["errors"] == []
        discs = backend.discussion_list("main")
        assert len(discs) == 1
        backend.close()

    def test_migration_backs_up_phases(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_3c

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        summaries_dir = ccr / "summaries"
        summaries_dir.mkdir()

        (summaries_dir / "phases.md").write_text(
            "## [P001] 2026-04-01 00:00 - 2026-04-15 00:00 | Phase Summary\n"
            "**Scope**: Test\n**Goal**: Verify backup\n**Outcome**: Done\n"
            "**Key accomplishments**: N/A\n**Files changed**: 0\n**Branch summary**: main\n",
            encoding="utf-8",
        )

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        migrate_phase_3c(str(ccr), db_path)

        assert os.path.isfile(str(summaries_dir / "phases.md.bak"))
        assert not os.path.isfile(str(summaries_dir / "phases.md"))
        backend.close()

    def test_auto_migrate_includes_3c(self, tmp_path):
        from ccr.core.storage.migration import auto_migrate

        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        import yaml
        meta = {"version": 1, "created": "2026-04-17", "branches": []}
        with open(os.path.join(str(ccr), "metadata.yaml"), "w") as f:
            yaml.dump(meta, f)

        (ccr / "overview.md").write_text("# My Project\n", encoding="utf-8")

        backend = SqliteStorageBackend(str(ccr))
        db_path = os.path.join(str(ccr), "memory.db")
        result = auto_migrate(str(ccr), db_path)

        assert "3c" in result["phases_run"]
        assert result["errors"] == []
        assert backend.project_state_get("overview") == "# My Project\n"
        backend.close()
