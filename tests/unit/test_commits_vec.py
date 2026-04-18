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


class TestVectorUpsert:
    def test_upsert_round_trip(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        vec = [0.1] * 384
        backend.commit_upsert_vector("C001", vec)
        row = backend.memory_conn.execute(
            "SELECT id FROM commits_vec WHERE id = ?", ("C001",)
        ).fetchone()
        assert row is not None
        assert row[0] == "C001"
        backend.close()

    def test_upsert_rejects_wrong_dim(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        with pytest.raises(ValueError, match="dim"):
            backend.commit_upsert_vector("C002", [0.1] * 10)
        backend.close()

    def test_upsert_is_idempotent(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        backend.commit_upsert_vector("C003", [0.2] * 384)
        backend.commit_upsert_vector("C003", [0.3] * 384)
        count = backend.memory_conn.execute(
            "SELECT COUNT(*) FROM commits_vec WHERE id = ?", ("C003",)
        ).fetchone()[0]
        assert count == 1
        backend.close()


class TestSemanticSearch:
    @staticmethod
    def _seed_commit(backend, cid, branch, title):
        from ccr.core.storage._sqlite_utils import _utcnow
        now = _utcnow()
        backend.memory_conn.execute(
            """INSERT INTO commits (id, branch, timestamp, title, what, why,
               files_json, next_step, patterns_json, score, author, ci_json,
               experiment_json, ota_trace, raw_block, created_at)
               VALUES (?, ?, ?, ?, ?, '', '[]', '', NULL, NULL, '',
                       NULL, NULL, NULL, ?, ?)""",
            (cid, branch, now, title, title, f"## [{cid}] {title}", now),
        )
        backend.memory_conn.commit()

    def test_knn_returns_closest_commit(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        self._seed_commit(backend, "C001", "main", "auth bug")
        self._seed_commit(backend, "C002", "main", "logging refactor")
        v1 = [1.0] + [0.0] * 383
        v2 = [0.0, 1.0] + [0.0] * 382
        backend.commit_upsert_vector("C001", v1)
        backend.commit_upsert_vector("C002", v2)

        q = [1.0] + [0.0] * 383
        hits = backend.commit_semantic_search("main", q, top_k=2)
        assert len(hits) >= 1
        assert hits[0]["id"] == "C001"
        assert "title" in hits[0]
        assert hits[0]["distance"] <= hits[-1]["distance"]
        backend.close()

    def test_semantic_search_empty_when_no_vectors(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        assert backend.commit_semantic_search("main", [0.1] * 384, top_k=5) == []
        backend.close()

    def test_semantic_search_filters_by_branch(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        self._seed_commit(backend, "C100", "main", "main commit")
        self._seed_commit(backend, "C101", "experiment", "branch commit")
        backend.commit_upsert_vector("C100", [1.0] + [0.0] * 383)
        backend.commit_upsert_vector("C101", [1.0] + [0.0] * 383)
        hits = backend.commit_semantic_search("main", [1.0] + [0.0] * 383, top_k=5)
        assert {h["id"] for h in hits} == {"C100"}
        backend.close()

    def test_semantic_search_rejects_wrong_dim(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        with pytest.raises(ValueError, match="dim"):
            backend.commit_semantic_search("main", [0.1] * 10, top_k=5)
        backend.close()
