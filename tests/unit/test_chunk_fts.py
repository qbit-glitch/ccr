"""Phase 3: Chunk-level FTS5 full-text search on index.db."""
from __future__ import annotations

import sqlite3

import pytest

from ccr.context.index_db import IndexDB


def _has_fts5() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _has_fts5(), reason="SQLite FTS5 not available")


class TestChunkFtsSchema:
    def test_chunks_fts_table_created(self, tmp_path):
        db = IndexDB(str(tmp_path / "index.db"))
        tables = {
            r[0] for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "index_chunks_fts" in tables
        db.close()

    def test_all_four_chunk_triggers_created(self, tmp_path):
        db = IndexDB(str(tmp_path / "index.db"))
        triggers = {
            r[0] for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        for t in (
            "index_chunks_fts_ai",
            "index_chunks_fts_ad",
            "index_chunks_fts_bu",
            "index_chunks_fts_au",
        ):
            assert t in triggers, f"missing trigger {t}"
        db.close()


class TestChunkSearch:
    def test_chunk_search_matches_content(self, tmp_path):
        db = IndexDB(str(tmp_path / "index.db"))
        db.save_chunks_batch([
            {
                "file_path": "src/auth.py",
                "chunk_idx": 0,
                "start_line": 1,
                "end_line": 20,
                "text": "def authenticate(user, password):\n    return verify_password(user, password)",
            },
            {
                "file_path": "src/log.py",
                "chunk_idx": 0,
                "start_line": 1,
                "end_line": 10,
                "text": "logger.info('request received')",
            },
        ])
        results = db.chunk_search("authenticate", top_k=5)
        assert len(results) == 1
        assert results[0]["file_path"] == "src/auth.py"
        assert "authenticate" in results[0]["text"]
        db.close()

    def test_chunk_search_empty_on_no_match(self, tmp_path):
        db = IndexDB(str(tmp_path / "index.db"))
        db.save_chunks_batch([{
            "file_path": "a.py", "chunk_idx": 0,
            "start_line": 1, "end_line": 1, "text": "hello world",
        }])
        assert db.chunk_search("nonexistent", top_k=5) == []
        db.close()

    def test_chunk_search_malformed_query_falls_back(self, tmp_path):
        """Unclosed quote would break FTS5 MATCH → LIKE fallback returns safely."""
        db = IndexDB(str(tmp_path / "index.db"))
        db.save_chunks_batch([{
            "file_path": "a.py", "chunk_idx": 0,
            "start_line": 1, "end_line": 1, "text": 'he said "hello"',
        }])
        result = db.chunk_search('"', top_k=5)
        assert isinstance(result, list)
        db.close()


class TestChunkSnippets:
    def test_snippet_highlights_match(self, tmp_path):
        db = IndexDB(str(tmp_path / "index.db"))
        db.save_chunks_batch([{
            "file_path": "src/auth.py",
            "chunk_idx": 0,
            "start_line": 10,
            "end_line": 30,
            "text": (
                "def authenticate(user, password):\n"
                "    '''Verify the user credentials.'''\n"
                "    return verify_password(user, password)"
            ),
        }])
        hits = db.chunk_search_with_snippet("authenticate", top_k=5)
        assert len(hits) == 1
        assert "[authenticate]" in hits[0]["snippet"]
        assert hits[0]["file_path"] == "src/auth.py"
        assert hits[0]["rank"] < 0  # BM25 rank is negative
        db.close()

    def test_snippet_returns_empty_when_fts5_unavailable(self, tmp_path, monkeypatch):
        db = IndexDB(str(tmp_path / "index.db"))
        db.save_chunks_batch([{
            "file_path": "a.py", "chunk_idx": 0,
            "start_line": 1, "end_line": 1, "text": "hello",
        }])
        monkeypatch.setattr(db, "_fts_available", False)
        assert db.chunk_search_with_snippet("hello", top_k=5) == []
        db.close()


class TestChunkBackfill:
    def test_backfill_on_upgrade(self, tmp_path):
        """DB created before chunk FTS5 still gets searchable chunks after upgrade."""
        db_path = tmp_path / "index.db"
        # Simulate a pre-Phase-3 DB: chunks exist, no chunk FTS.
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE index_files (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                language TEXT NOT NULL DEFAULT '',
                line_count INTEGER NOT NULL DEFAULT 0,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                mtime REAL NOT NULL DEFAULT 0,
                symbols_json TEXT NOT NULL DEFAULT '[]',
                imports_json TEXT NOT NULL DEFAULT '[]',
                git_hash TEXT DEFAULT '',
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE index_chunks (
                file_path TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (file_path, chunk_idx)
            );
            CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO index_chunks VALUES ('pre/existing.py', 0, 1, 20,
                'def legacy_function():\n    return 42');
            PRAGMA user_version = 0;
        """)
        conn.commit()
        conn.close()

        # Now open with Phase-3-enabled IndexDB — backfill should run.
        db = IndexDB(str(db_path))
        hits = db.chunk_search("legacy_function", top_k=5)
        assert len(hits) == 1
        assert hits[0]["file_path"] == "pre/existing.py"
        ver = db.conn.execute("PRAGMA user_version").fetchone()[0]
        assert ver >= 1
        db.close()

    def test_backfill_idempotent(self, tmp_path):
        """Re-opening an already-migrated DB doesn't re-run backfill."""
        db_path = tmp_path / "index.db"
        db = IndexDB(str(db_path))
        db.save_chunks_batch([{
            "file_path": "a.py", "chunk_idx": 0,
            "start_line": 1, "end_line": 1, "text": "first open",
        }])
        db.close()
        db2 = IndexDB(str(db_path))
        hits = db2.chunk_search("first", top_k=5)
        assert len(hits) == 1
        db2.close()
