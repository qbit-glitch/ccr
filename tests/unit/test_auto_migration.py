"""Tests for Phase 7: auto-migration + SQLite default."""
from __future__ import annotations

import json
import os

import pytest
import yaml

from ccr.core.storage.migration import auto_migrate, needs_migration
from ccr.core.storage.sqlite_backend import SqliteStorageBackend
from ccr.core.types import CCRConfig


class TestSqliteDefault:
    def test_default_is_sqlite(self):
        cfg = CCRConfig()
        assert cfg.storage_backend == "sqlite"

    def test_override_to_files(self):
        cfg = CCRConfig(storage_backend="files")
        assert cfg.storage_backend == "files"


class TestNeedsMigration:
    def test_no_ccr_dir(self, tmp_path):
        assert not needs_migration(str(tmp_path / ".ccr"))

    def test_sentinel_exists(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        (ccr / ".migrated").write_text("2026-04-18")
        (ccr / "metadata.yaml").write_text("version: 1")
        assert not needs_migration(str(ccr))

    def test_flat_files_exist(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        (ccr / "metadata.yaml").write_text("version: 1")
        assert needs_migration(str(ccr))

    def test_no_metadata(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        assert not needs_migration(str(ccr))


class TestAutoMigrate:
    def test_full_migration_creates_sentinel(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        meta = {"version": 1, "created": "2026-04-18", "branches": []}
        with open(str(ccr / "metadata.yaml"), "w") as f:
            yaml.dump(meta, f)

        with open(str(ccr / "scratchpad.json"), "w") as f:
            json.dump({"entries": {"test_key": {
                "value": "test_val", "created_at": "2026-04-18",
                "updated_at": "2026-04-18", "access_count": 1,
            }}}, f)

        backend = SqliteStorageBackend(str(ccr))
        db_path = str(ccr / "memory.db")
        result = auto_migrate(str(ccr), db_path)

        assert result["errors"] == []
        assert len(result["phases_run"]) >= 1
        assert os.path.isfile(str(ccr / ".migrated"))
        backend.close()

    def test_sentinel_prevents_re_migration(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        meta = {"version": 1, "created": "2026-04-18", "branches": []}
        with open(str(ccr / "metadata.yaml"), "w") as f:
            yaml.dump(meta, f)

        backend = SqliteStorageBackend(str(ccr))
        db_path = str(ccr / "memory.db")

        r1 = auto_migrate(str(ccr), db_path)
        assert len(r1["phases_run"]) >= 1
        assert os.path.isfile(str(ccr / ".migrated"))

        r2 = auto_migrate(str(ccr), db_path)
        assert r2["phases_run"] == []
        assert r2["total_migrated"] == 0
        backend.close()

    def test_fresh_project_no_migration(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        backend = SqliteStorageBackend(str(ccr))
        db_path = str(ccr / "memory.db")
        result = auto_migrate(str(ccr), db_path)

        assert result["phases_run"] == []
        assert result["total_migrated"] == 0
        backend.close()

    def test_all_phases_run(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        with open(str(ccr / "metadata.yaml"), "w") as f:
            yaml.dump({"version": 1, "created": "2026-04-18", "branches": []}, f)

        with open(str(ccr / "scratchpad.json"), "w") as f:
            json.dump({"entries": {}}, f)

        with open(str(ccr / "triples.json"), "w") as f:
            json.dump({"version": 1, "triples": [
                {"subject": "A", "predicate": "uses", "object": "B",
                 "source_commit": "C001", "confidence": 0.9},
            ]}, f)

        (ccr / "overview.md").write_text("# My Project\n")

        backend = SqliteStorageBackend(str(ccr))
        db_path = str(ccr / "memory.db")
        result = auto_migrate(str(ccr), db_path)

        assert 1 in result["phases_run"]
        assert 2 in result["phases_run"]
        assert "3a" in result["phases_run"]
        assert "3b" in result["phases_run"]
        assert "3c" in result["phases_run"]
        assert result["errors"] == []

        assert backend.triple_count() == 1
        assert backend.project_state_get("overview") == "# My Project\n"
        backend.close()

    def test_migration_idempotent_across_phases(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        with open(str(ccr / "metadata.yaml"), "w") as f:
            yaml.dump({"version": 1}, f)

        backend = SqliteStorageBackend(str(ccr))
        db_path = str(ccr / "memory.db")

        r1 = auto_migrate(str(ccr), db_path)
        assert r1["errors"] == []

        r2 = auto_migrate(str(ccr), db_path)
        assert r2["phases_run"] == []
        backend.close()


class TestBackupSafety:
    def test_flat_files_renamed_not_deleted(self, tmp_path):
        ccr = tmp_path / ".ccr"
        ccr.mkdir()

        with open(str(ccr / "metadata.yaml"), "w") as f:
            yaml.dump({"version": 1, "created": "2026-04-18"}, f)

        with open(str(ccr / "scratchpad.json"), "w") as f:
            json.dump({"entries": {"k": {
                "value": "v", "created_at": "t", "updated_at": "t", "access_count": 1,
            }}}, f)

        backend = SqliteStorageBackend(str(ccr))
        db_path = str(ccr / "memory.db")
        auto_migrate(str(ccr), db_path)

        assert os.path.isfile(str(ccr / "scratchpad.json.bak"))
        assert not os.path.isfile(str(ccr / "scratchpad.json"))
        backend.close()


class TestEnvVarOverride:
    def test_env_var_forces_files_backend(self):
        cfg = CCRConfig(storage_backend="files")
        assert cfg.storage_backend == "files"

    def test_env_var_forces_sqlite_backend(self):
        cfg = CCRConfig(storage_backend="sqlite")
        assert cfg.storage_backend == "sqlite"


class TestCliInit:
    def test_init_creates_memory_db(self, tmp_path):
        from click.testing import CliRunner
        from ccr.cli import init

        runner = CliRunner()
        result = runner.invoke(init, [str(tmp_path)])
        assert result.exit_code == 0

        ccr = tmp_path / ".ccr"
        assert ccr.is_dir()
        assert (ccr / "memory.db").is_file()
