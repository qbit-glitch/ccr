"""Phase 4 tests: IndexDB SQLite persistence, FTS5 search, incremental builds.

Covers:
  - IndexDB: file CRUD, chunk CRUD, meta KV, FTS5/LIKE search
  - RepoIndex: save_to_db / from_db roundtrip, incremental_build
  - Integration with indexer search (fts5_search method)
"""

from __future__ import annotations

import os
import time

import pytest

from ccr.context.index_db import IndexDB
from ccr.context.indexer import RepoIndex


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "index.db")
    index_db = IndexDB(db_path)
    yield index_db
    index_db.close()


@pytest.fixture
def sample_repo(tmp_path):
    """Create a small sample repo for indexing."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(
        "import os\n\nclass Application:\n    def run(self):\n        pass\n"
    )
    (root / "utils.py").write_text(
        "def helper():\n    return 42\n\ndef parse(data):\n    return data\n"
    )
    (root / "config.yaml").write_text("key: value\nname: test\n")
    return root


# ── IndexDB: File CRUD ────────────────────────────────────────────


class TestIndexDBFiles:
    def test_upsert_and_get(self, db):
        db.upsert_file(
            path="src/main.py", language="python", line_count=50,
            size_bytes=1200, mtime=1713456789.0,
            symbols=["Application", "main"], imports=["os", "sys"],
            git_hash="abc123", indexed_at="2026-04-18T00:00:00",
        )
        db.conn.commit()
        got = db.get_file("src/main.py")
        assert got is not None
        assert got["path"] == "src/main.py"
        assert got["language"] == "python"
        assert got["line_count"] == 50
        assert got["symbols"] == ["Application", "main"]
        assert got["imports"] == ["os", "sys"]

    def test_get_missing(self, db):
        assert db.get_file("nonexistent.py") is None

    def test_save_batch_and_load(self, db):
        files = [
            {"path": "a.py", "language": "python", "line_count": 10,
             "size_bytes": 100, "mtime": 1000.0, "symbols": ["foo"],
             "imports": ["os"]},
            {"path": "b.ts", "language": "typescript", "line_count": 20,
             "size_bytes": 200, "mtime": 2000.0, "symbols": ["bar"],
             "imports": ["react"]},
        ]
        count = db.save_files_batch(files, "2026-04-18T00:00:00")
        assert count == 2

        loaded = db.load_files()
        assert len(loaded) == 2
        paths = {f["path"] for f in loaded}
        assert "a.py" in paths
        assert "b.ts" in paths

    def test_file_count(self, db):
        assert db.file_count() == 0
        db.save_files_batch([
            {"path": "x.py", "language": "python", "line_count": 1,
             "size_bytes": 10, "mtime": 1.0},
        ], "now")
        assert db.file_count() == 1

    def test_upsert_updates_existing(self, db):
        db.upsert_file(
            path="f.py", language="python", line_count=10,
            size_bytes=100, mtime=1.0, symbols=["old"],
            imports=[], git_hash="", indexed_at="t1",
        )
        db.conn.commit()
        db.upsert_file(
            path="f.py", language="python", line_count=20,
            size_bytes=200, mtime=2.0, symbols=["new"],
            imports=["os"], git_hash="abc", indexed_at="t2",
        )
        db.conn.commit()
        got = db.get_file("f.py")
        assert got["line_count"] == 20
        assert got["symbols"] == ["new"]
        assert db.file_count() == 1

    def test_delete_file(self, db):
        db.save_files_batch([
            {"path": "del.py", "language": "python", "line_count": 1,
             "size_bytes": 10, "mtime": 1.0},
        ], "now")
        assert db.delete_file("del.py") is True
        assert db.get_file("del.py") is None
        assert db.delete_file("del.py") is False

    def test_delete_missing_files(self, db):
        db.save_files_batch([
            {"path": "keep.py", "language": "python", "line_count": 1,
             "size_bytes": 10, "mtime": 1.0},
            {"path": "gone.py", "language": "python", "line_count": 1,
             "size_bytes": 10, "mtime": 1.0},
        ], "now")
        deleted = db.delete_missing({"keep.py"})
        assert deleted == 1
        assert db.file_count() == 1
        assert db.get_file("keep.py") is not None

    def test_get_file_mtimes(self, db):
        db.save_files_batch([
            {"path": "a.py", "mtime": 100.0},
            {"path": "b.py", "mtime": 200.0},
        ], "now")
        mtimes = db.get_file_mtimes()
        assert mtimes["a.py"] == 100.0
        assert mtimes["b.py"] == 200.0


# ── IndexDB: Chunks ───────────────────────────────────────────────


class TestIndexDBChunks:
    def test_save_and_load_chunks(self, db):
        chunks = [
            {"file_path": "main.py", "chunk_idx": 0,
             "start_line": 1, "end_line": 10, "text": "chunk 0"},
            {"file_path": "main.py", "chunk_idx": 1,
             "start_line": 11, "end_line": 20, "text": "chunk 1"},
        ]
        count = db.save_chunks_batch(chunks)
        assert count == 2

        loaded = db.load_chunks("main.py")
        assert len(loaded) == 2
        assert loaded[0]["chunk_idx"] == 0
        assert loaded[1]["text"] == "chunk 1"

    def test_load_all_chunks(self, db):
        db.save_chunks_batch([
            {"file_path": "a.py", "chunk_idx": 0,
             "start_line": 1, "end_line": 5, "text": "a0"},
            {"file_path": "b.py", "chunk_idx": 0,
             "start_line": 1, "end_line": 5, "text": "b0"},
        ])
        all_chunks = db.load_chunks()
        assert len(all_chunks) == 2

    def test_chunk_count(self, db):
        assert db.chunk_count() == 0
        db.save_chunks_batch([
            {"file_path": "x.py", "chunk_idx": 0,
             "start_line": 1, "end_line": 5, "text": "x"},
        ])
        assert db.chunk_count() == 1

    def test_chunk_upsert(self, db):
        db.save_chunks_batch([
            {"file_path": "f.py", "chunk_idx": 0,
             "start_line": 1, "end_line": 5, "text": "old"},
        ])
        db.save_chunks_batch([
            {"file_path": "f.py", "chunk_idx": 0,
             "start_line": 1, "end_line": 10, "text": "new"},
        ])
        loaded = db.load_chunks("f.py")
        assert len(loaded) == 1
        assert loaded[0]["text"] == "new"
        assert loaded[0]["end_line"] == 10


# ── IndexDB: Meta KV ─────────────────────────────────────────────


class TestIndexDBMeta:
    def test_set_and_get(self, db):
        db.set_meta("built_at", "12345.0")
        assert db.get_meta("built_at") == "12345.0"

    def test_get_missing(self, db):
        assert db.get_meta("nonexistent") is None

    def test_overwrite(self, db):
        db.set_meta("key", "v1")
        db.set_meta("key", "v2")
        assert db.get_meta("key") == "v2"


# ── IndexDB: FTS5 / LIKE Search ──────────────────────────────────


class TestIndexDBSearch:
    def test_fts_search(self, db):
        db.save_files_batch([
            {"path": "src/auth/login.py", "language": "python",
             "line_count": 50, "size_bytes": 1000, "mtime": 1.0,
             "symbols": ["LoginController", "authenticate"],
             "imports": ["flask"]},
            {"path": "src/db/queries.py", "language": "python",
             "line_count": 30, "size_bytes": 500, "mtime": 1.0,
             "symbols": ["execute_query", "connect"],
             "imports": ["sqlite3"]},
        ], "now")
        results = db.fts_search("login", top_k=10)
        assert len(results) >= 1
        assert results[0]["path"] == "src/auth/login.py"

    def test_fts_search_symbols(self, db):
        db.save_files_batch([
            {"path": "model.py", "language": "python",
             "line_count": 100, "size_bytes": 2000, "mtime": 1.0,
             "symbols": ["NeuralNetwork", "forward", "backward"]},
        ], "now")
        results = db.fts_search("NeuralNetwork", top_k=10)
        assert len(results) >= 1

    def test_fts_search_no_results(self, db):
        db.save_files_batch([
            {"path": "a.py", "language": "python",
             "line_count": 10, "size_bytes": 100, "mtime": 1.0},
        ], "now")
        results = db.fts_search("nonexistent_symbol_xyz", top_k=10)
        assert results == []

    def test_fts_search_empty_query(self, db):
        results = db.fts_search("", top_k=10)
        assert results == []

    def test_fts_sanitize_query(self):
        assert IndexDB._sanitize_fts_query("hello world") == '"hello" OR "world"'
        assert IndexDB._sanitize_fts_query("test") == '"test"'
        assert IndexDB._sanitize_fts_query("") == ""

    def test_like_fallback(self, db):
        db.save_files_batch([
            {"path": "src/utils.py", "language": "python",
             "line_count": 10, "size_bytes": 100, "mtime": 1.0,
             "symbols": ["helper"]},
        ], "now")
        results = db._like_search("utils", top_k=10)
        assert len(results) >= 1
        assert results[0]["path"] == "src/utils.py"


# ── RepoIndex: save_to_db / from_db ──────────────────────────────


class TestIndexDBRoundtrip:
    def test_save_and_load(self, db, sample_repo):
        idx = RepoIndex.build(str(sample_repo))
        count = idx.save_to_db(db)
        assert count >= 2

        loaded = RepoIndex.from_db(str(sample_repo), db)
        assert loaded is not None
        assert len(loaded.files) == len(idx.files)
        for path in idx.files:
            assert path in loaded.files
            assert loaded.files[path].language == idx.files[path].language

    def test_load_preserves_metadata(self, db, sample_repo):
        idx = RepoIndex.build(str(sample_repo))
        idx.save_to_db(db)

        loaded = RepoIndex.from_db(str(sample_repo), db)
        assert loaded._built_at is not None
        assert loaded._mtime_sig != ""

    def test_load_empty_db(self, db, sample_repo):
        loaded = RepoIndex.from_db(str(sample_repo), db)
        assert loaded is None

    def test_load_skips_deleted_files(self, db, sample_repo):
        idx = RepoIndex.build(str(sample_repo))
        idx.save_to_db(db)

        os.remove(str(sample_repo / "config.yaml"))
        loaded = RepoIndex.from_db(str(sample_repo), db)
        assert loaded is not None
        assert "config.yaml" not in loaded.files

    def test_chunks_roundtrip(self, db, sample_repo):
        idx = RepoIndex.build(str(sample_repo))
        idx.save_to_db(db)

        chunks = db.load_chunks()
        main_chunks = db.load_chunks("main.py")
        assert isinstance(chunks, list)
        assert isinstance(main_chunks, list)


# ── Incremental Build ─────────────────────────────────────────────


class TestIncrementalBuild:
    def test_incremental_detects_changes(self, db, sample_repo):
        idx1 = RepoIndex.build(str(sample_repo))
        idx1.save_to_db(db)

        time.sleep(0.05)
        (sample_repo / "utils.py").write_text("def new_func():\n    pass\n")

        idx2, changed = RepoIndex.incremental_build(str(sample_repo), db)
        assert changed >= 1
        assert len(idx2.files) >= 2

    def test_incremental_no_changes(self, db, sample_repo):
        idx1 = RepoIndex.build(str(sample_repo))
        idx1.save_to_db(db)

        idx2, changed = RepoIndex.incremental_build(str(sample_repo), db)
        assert changed == 0

    def test_incremental_detects_deletions(self, db, sample_repo):
        idx1 = RepoIndex.build(str(sample_repo))
        idx1.save_to_db(db)

        os.remove(str(sample_repo / "config.yaml"))
        idx2, changed = RepoIndex.incremental_build(str(sample_repo), db)
        assert changed >= 1
        assert "config.yaml" not in idx2.files

    def test_incremental_new_file(self, db, sample_repo):
        idx1 = RepoIndex.build(str(sample_repo))
        idx1.save_to_db(db)

        (sample_repo / "new_file.py").write_text("x = 1\n")
        idx2, changed = RepoIndex.incremental_build(str(sample_repo), db)
        assert changed >= 1
        assert "new_file.py" in idx2.files


# ── fts5_search Integration ───────────────────────────────────────


class TestFTS5SearchIntegration:
    def test_fts5_search_with_index(self, db, sample_repo):
        idx = RepoIndex.build(str(sample_repo))
        idx.save_to_db(db)
        results = idx.fts5_search("Application", db, top_k=10)
        assert len(results) >= 1
        assert any(r["path"] == "main.py" for r in results)

    def test_fts5_search_falls_back_without_db(self, sample_repo):
        idx = RepoIndex.build(str(sample_repo))
        results = idx.fts5_search("helper", None, top_k=10)
        assert isinstance(results, list)

    def test_fts5_search_returns_metadata(self, db, sample_repo):
        idx = RepoIndex.build(str(sample_repo))
        idx.save_to_db(db)
        results = idx.fts5_search("main", db, top_k=10)
        if results:
            r = results[0]
            assert "path" in r
            assert "language" in r
            assert "symbols" in r
