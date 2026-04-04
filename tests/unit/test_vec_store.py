"""Tests for sqlite-vec persistent vector store."""

import threading

import pytest

# Check if sqlite-vec is available
try:
    import sqlite_vec  # noqa: F401

    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

pytestmark = pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec not installed")

from ccr.context.vec_store import (
    SQLITE_VEC_AVAILABLE,
    SqliteVecStore,
    deserialize_float32,
    get_vec_store,
    serialize_float32,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_embeddings.db")


@pytest.fixture
def store(db_path):
    s = SqliteVecStore(db_path, dim=4)
    yield s
    s.close()


def make_vec(vals: list[float]) -> list[float]:
    """Create a simple test vector."""
    return vals


class TestSerialization:
    def test_serialize_deserialize_roundtrip(self):
        vec = [1.0, 2.0, 3.0, 4.0]
        data = serialize_float32(vec)
        result = deserialize_float32(data, 4)
        assert result == pytest.approx(vec)

    def test_serialize_empty(self):
        data = serialize_float32([])
        result = deserialize_float32(data, 0)
        assert result == []

    def test_serialize_negative_values(self):
        vec = [-1.5, 0.0, 3.14, -0.001]
        data = serialize_float32(vec)
        result = deserialize_float32(data, 4)
        assert result == pytest.approx(vec, rel=1e-5)

    def test_serialize_returns_bytes(self):
        data = serialize_float32([1.0, 2.0])
        assert isinstance(data, bytes)
        assert len(data) == 8  # 2 floats * 4 bytes each


class TestSqliteVecStore:
    def test_upsert_and_get(self, store):
        vec = make_vec([1.0, 0.0, 0.0, 0.0])
        store.upsert("C001", vec, "commit")
        result = store.get("C001")
        assert result == pytest.approx(vec)

    def test_get_missing(self, store):
        assert store.get("nonexistent") is None

    def test_upsert_updates_existing(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0])
        store.upsert("C001", [0.0, 1.0, 0.0, 0.0])
        result = store.get("C001")
        assert result == pytest.approx([0.0, 1.0, 0.0, 0.0])

    def test_wrong_dimension_raises(self, store):
        with pytest.raises(ValueError, match="dim"):
            store.upsert("C001", [1.0, 0.0])  # dim 2, expected 4

    def test_search_wrong_dimension_raises(self, store):
        with pytest.raises(ValueError, match="dim"):
            store.search([1.0, 0.0], "commit")

    def test_search_knn(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0], "commit")
        store.upsert("C002", [0.0, 1.0, 0.0, 0.0], "commit")
        store.upsert("C003", [0.9, 0.1, 0.0, 0.0], "commit")
        results = store.search([1.0, 0.0, 0.0, 0.0], "commit", top_k=2)
        assert len(results) == 2
        # C001 should be closest (distance 0), C003 second
        assert results[0][0] == "C001"

    def test_search_returns_distances(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0], "commit")
        results = store.search([1.0, 0.0, 0.0, 0.0], "commit", top_k=1)
        assert len(results) == 1
        _id, distance = results[0]
        assert _id == "C001"
        assert isinstance(distance, float)

    def test_search_namespace_isolation(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0], "commit")
        store.upsert("F001", [1.0, 0.0, 0.0, 0.0], "file")
        results = store.search([1.0, 0.0, 0.0, 0.0], "commit", top_k=10)
        ids = [r[0] for r in results]
        assert "C001" in ids
        assert "F001" not in ids

    def test_search_empty_namespace(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0], "commit")
        results = store.search([1.0, 0.0, 0.0, 0.0], "file", top_k=10)
        assert results == []

    def test_delete(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0])
        assert store.delete("C001") is True
        assert store.get("C001") is None

    def test_delete_nonexistent(self, store):
        assert store.delete("ghost") is False

    def test_delete_removes_from_search(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0], "commit")
        store.delete("C001")
        results = store.search([1.0, 0.0, 0.0, 0.0], "commit", top_k=10)
        assert results == []

    def test_count(self, store):
        assert store.count() == 0
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0], "commit")
        store.upsert("C002", [0.0, 1.0, 0.0, 0.0], "commit")
        store.upsert("F001", [0.0, 0.0, 1.0, 0.0], "file")
        assert store.count() == 3
        assert store.count("commit") == 2
        assert store.count("file") == 1

    def test_get_batch(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0])
        store.upsert("C002", [0.0, 1.0, 0.0, 0.0])
        store.upsert("C003", [0.0, 0.0, 1.0, 0.0])
        batch = store.get_batch(["C001", "C003"])
        assert len(batch) == 2
        assert "C001" in batch
        assert "C003" in batch
        assert batch["C001"] == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_get_batch_empty(self, store):
        assert store.get_batch([]) == {}

    def test_get_batch_partial(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0])
        batch = store.get_batch(["C001", "missing"])
        assert len(batch) == 1
        assert "C001" in batch

    def test_list_ids(self, store):
        store.upsert("C003", [0.0, 0.0, 1.0, 0.0], "commit")
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0], "commit")
        store.upsert("F001", [0.0, 1.0, 0.0, 0.0], "file")
        ids = store.list_ids("commit")
        assert sorted(ids) == ["C001", "C003"]

    def test_list_ids_empty_namespace(self, store):
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0], "commit")
        ids = store.list_ids("file")
        assert ids == []

    def test_thread_safety(self, store):
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    store.upsert(f"T{n}-{i}", [float(n), float(i), 0.0, 0.0])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert store.count() == 30

    def test_persistence(self, db_path):
        s1 = SqliteVecStore(db_path, dim=4)
        s1.upsert("C001", [1.0, 2.0, 3.0, 4.0])
        s1.close()
        s2 = SqliteVecStore(db_path, dim=4)
        result = s2.get("C001")
        assert result == pytest.approx([1.0, 2.0, 3.0, 4.0])
        s2.close()

    def test_default_namespace(self, store):
        """Upsert without explicit namespace uses 'commit'."""
        store.upsert("C001", [1.0, 0.0, 0.0, 0.0])
        assert store.count("commit") == 1

    def test_upsert_preserves_metadata_namespace(self, store):
        """Updating a vector preserves the new namespace."""
        store.upsert("X001", [1.0, 0.0, 0.0, 0.0], "commit")
        store.upsert("X001", [0.0, 1.0, 0.0, 0.0], "file")
        assert store.count("commit") == 0
        assert store.count("file") == 1

    def test_close_idempotent(self, store):
        """Closing multiple times does not raise."""
        store.close()
        store.close()  # Should not raise

    def test_creates_parent_directory(self, tmp_path):
        """Store creates parent directories for the DB file."""
        db_path = str(tmp_path / "subdir" / "nested" / "embeddings.db")
        s = SqliteVecStore(db_path, dim=4)
        s.upsert("C001", [1.0, 0.0, 0.0, 0.0])
        assert s.get("C001") is not None
        s.close()


class TestGetVecStore:
    def test_returns_store_when_available(self, db_path):
        store = get_vec_store(db_path, dim=4)
        assert store is not None
        store.close()

    def test_returns_none_when_unavailable(self, db_path):
        import ccr.context.vec_store as mod

        orig = mod.SQLITE_VEC_AVAILABLE
        mod.SQLITE_VEC_AVAILABLE = False
        try:
            store = get_vec_store(db_path)
            assert store is None
        finally:
            mod.SQLITE_VEC_AVAILABLE = orig

    def test_custom_dimension(self, db_path):
        store = get_vec_store(db_path, dim=8)
        assert store is not None
        store.upsert("C001", [1.0] * 8)
        result = store.get("C001")
        assert len(result) == 8
        store.close()


class TestMigration:
    """Test the gzip-to-sqlite migration helper in embeddings.py."""

    def test_migrate_gzip_to_sqlite(self, tmp_path):
        from ccr.context.embeddings import (
            migrate_gzip_to_sqlite,
            save_embeddings,
        )

        # Create a gzip file with embeddings
        gzip_path = str(tmp_path / "embeddings.json.gz")
        db_path = str(tmp_path / "embeddings.db")
        save_embeddings(
            {
                "C001": [1.0, 0.0, 0.0, 0.0],
                "C002": [0.0, 1.0, 0.0, 0.0],
            },
            gzip_path,
        )

        count = migrate_gzip_to_sqlite(gzip_path, db_path, namespace="commit", dim=4)
        assert count == 2

        # Verify vectors are in sqlite
        store = SqliteVecStore(db_path, dim=4)
        assert store.count("commit") == 2
        result = store.get("C001")
        assert result == pytest.approx([1.0, 0.0, 0.0, 0.0])
        store.close()

    def test_migrate_with_matching_dims(self, tmp_path):
        """End-to-end migration with proper dimensions."""
        from ccr.context.embeddings import save_embeddings

        from ccr.context.vec_store import SqliteVecStore

        gzip_path = str(tmp_path / "embeddings.json.gz")
        db_path = str(tmp_path / "embeddings.db")

        # Create vectors with dim=4 and a store with dim=4
        vecs = {
            "C001": [1.0, 0.0, 0.0, 0.0],
            "C002": [0.0, 1.0, 0.0, 0.0],
            "C003": [0.0, 0.0, 1.0, 0.0],
        }
        save_embeddings(vecs, gzip_path)

        # Manually migrate (since migrate_gzip_to_sqlite uses default dim=384)
        from ccr.context.embeddings import load_embeddings

        store = SqliteVecStore(db_path, dim=4)
        cache = load_embeddings(gzip_path)
        for id_, vec in cache.items():
            store.upsert(id_, vec, "commit")

        assert store.count("commit") == 3
        result = store.get("C001")
        assert result == pytest.approx([1.0, 0.0, 0.0, 0.0])
        store.close()

    def test_migrate_empty_gzip(self, tmp_path):
        from ccr.context.embeddings import migrate_gzip_to_sqlite

        gzip_path = str(tmp_path / "nonexistent.json.gz")
        db_path = str(tmp_path / "embeddings.db")

        count = migrate_gzip_to_sqlite(gzip_path, db_path)
        assert count == 0

    def test_migrate_when_sqlite_vec_unavailable(self, tmp_path):
        from ccr.context.embeddings import (
            migrate_gzip_to_sqlite,
            save_embeddings,
        )

        import ccr.context.vec_store as mod

        gzip_path = str(tmp_path / "embeddings.json.gz")
        db_path = str(tmp_path / "embeddings.db")
        save_embeddings({"C001": [1.0, 0.0, 0.0, 0.0]}, gzip_path)

        orig = mod.SQLITE_VEC_AVAILABLE
        mod.SQLITE_VEC_AVAILABLE = False
        try:
            count = migrate_gzip_to_sqlite(gzip_path, db_path)
            assert count == 0
        finally:
            mod.SQLITE_VEC_AVAILABLE = orig
