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


class TestBackfill:
    def test_gzip_json_source_imported_on_upgrade(self, tmp_path):
        """Pre-Phase-4 DB with commit_embeddings.json.gz gets backfilled."""
        import gzip
        import json
        import sqlite3

        ccr_root = tmp_path / "ccr"
        ccr_root.mkdir()
        legacy_vecs = {"C001": [0.5] * 384, "C002": [0.1] * 384}
        gz_path = ccr_root / "commit_embeddings.json.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(legacy_vecs, f)

        db = sqlite3.connect(str(ccr_root / "memory.db"))
        db.execute("PRAGMA user_version = 1")
        db.commit()
        db.close()

        backend = SqliteStorageBackend(str(ccr_root))
        ids = {
            r[0]
            for r in backend.memory_conn.execute(
                "SELECT id FROM commits_vec"
            ).fetchall()
        }
        assert ids == {"C001", "C002"}
        assert backend._memory_mgr.get_user_version() == 2
        backend.close()

    def test_legacy_embeddings_db_imported_on_upgrade(self, tmp_path):
        """Pre-Phase-4 DB with .ccr/embeddings.db sidecar gets backfilled."""
        try:
            import sqlite_vec  # noqa: F401
        except ImportError:
            pytest.skip("sqlite-vec required for legacy side-car import")
        from ccr.context.vec_store import get_vec_store

        ccr_root = tmp_path / "ccr"
        ccr_root.mkdir()

        legacy = get_vec_store(str(ccr_root / "embeddings.db"))
        assert legacy is not None
        legacy.upsert("C101", [0.2] * 384, namespace="commit")
        legacy.close()

        backend = SqliteStorageBackend(str(ccr_root))
        row = backend.memory_conn.execute(
            "SELECT id FROM commits_vec WHERE id = ?", ("C101",)
        ).fetchone()
        assert row is not None and row[0] == "C101"
        backend.close()

    def test_backfill_idempotent(self, tmp_path):
        """Re-opening migrated DB doesn't re-run backfill."""
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        backend.commit_upsert_vector("C001", [0.1] * 384)
        assert backend._memory_mgr.get_user_version() == 2
        backend.close()

        backend2 = SqliteStorageBackend(str(tmp_path / "ccr"))
        rows = backend2.memory_conn.execute(
            "SELECT COUNT(*) FROM commits_vec"
        ).fetchone()[0]
        assert rows == 1  # unchanged
        backend2.close()


class TestSemanticWiring:
    def test_search_commits_uses_backend_knn_when_vec_available(
        self, tmp_path, monkeypatch
    ):
        """_search_commits prefers backend.commit_semantic_search over in-Python cosine."""
        import numpy as np

        from ccr.core.memory import MemoryManager
        from ccr.core.storage._sqlite_utils import _utcnow
        from ccr.core.types import CCRConfig

        project_root = tmp_path / "proj"
        project_root.mkdir()
        cfg = CCRConfig(storage_backend="sqlite")
        mem = MemoryManager(str(project_root), config=cfg)
        assert mem._storage.vec_available is True

        for cid, title in (("C001", "authentication refactor"), ("C002", "loader tweak")):
            mem._storage.memory_conn.execute(
                """INSERT INTO commits (id, branch, timestamp, title, what, why,
                   files_json, next_step, patterns_json, score, author, ci_json,
                   experiment_json, ota_trace, raw_block, created_at)
                   VALUES (?, 'main', ?, ?, ?, '', '[]', '', NULL, NULL, '',
                           NULL, NULL, NULL, ?, ?)""",
                (cid, _utcnow(), title, title,
                 f"## [{cid}] {title}\n**What**: {title}\n**Why**: test\n", _utcnow()),
            )
        mem._storage.memory_conn.commit()

        v_auth = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        v_loader = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)
        mem._storage.commit_upsert_vector("C001", v_auth.tolist())
        mem._storage.commit_upsert_vector("C002", v_loader.tolist())

        class FakeModel:
            def embed_query(self, text):
                return v_auth

        import ccr.core.memory_pkg.memory_context as mod_ctx
        monkeypatch.setattr(mod_ctx, "get_embedding_model", lambda: FakeModel())

        # Query term that will NOT text-match either commit to force semantic path
        result = mem._search_commits("main", "xyzqqq_no_text_match")
        assert "C001" in result
        mem._storage.close()
