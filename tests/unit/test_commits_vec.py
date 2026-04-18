"""Phase 4: commits_vec vec0 virtual table + sqlite-vec extension loading."""
from __future__ import annotations

import pytest

from ccr.core.storage.sqlite_backend import SqliteStorageBackend


def _sqlite_vec_available() -> bool:
    try:
        import sqlite_vec  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _sqlite_vec_available(), reason="sqlite-vec not installed"
)


class TestCommitsVecSchema:
    def test_vec_available_true_when_extension_loadable(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        assert backend.vec_available is True
        backend.close()

    def test_commits_vec_table_created(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        tables = {
            r[0]
            for r in backend.memory_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "commits_vec" in tables
        backend.close()

    def test_commits_vec_dim_is_384(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        sql = backend.memory_conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='commits_vec'"
        ).fetchone()[0]
        assert "float[384]" in sql
        backend.close()
