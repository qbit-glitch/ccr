"""Phase 2: FTS5 full-text search on memory.db -- SQLite-only tests.

Tests the FTS5 helper module (`ccr/core/storage/_sqlite_fts5.py`):
- Availability probe
- Virtual-table creation (idempotent)
- Trigger propagation (INSERT / DELETE / UPDATE)

Module-level skip when SQLite's FTS5 module is not compiled in.
"""
from __future__ import annotations

import sqlite3

import pytest

from ccr.core.storage._sqlite_fts5 import (
    CREATE_COMMITS_FTS,
    CREATE_DISCUSSIONS_FTS,
    CREATE_PATTERNS_FTS,
    CREATE_TRIPLES_FTS,
    fts5_available,
    install_fts5,
)


def _has_fts5() -> bool:
    """Probe whether the running SQLite build has FTS5 compiled in."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_fts5(), reason="SQLite FTS5 not available"
)


# ---------- Helpers ---------------------------------------------------------


_BASE_SCHEMA = """
CREATE TABLE commits (
    rowid INTEGER PRIMARY KEY,
    title TEXT, what TEXT, why TEXT, next_step TEXT, files_json TEXT
);
CREATE TABLE discussions (
    rowid INTEGER PRIMARY KEY,
    topic TEXT, hypothesis TEXT, alternatives TEXT,
    decision TEXT, rationale TEXT, uncertainty TEXT
);
CREATE TABLE triples (
    id INTEGER PRIMARY KEY,
    subject TEXT, predicate TEXT, object TEXT
);
CREATE TABLE patterns (
    rowid INTEGER PRIMARY KEY,
    text TEXT
);
"""


def _fresh_conn(tmp_path, name: str = "m.db") -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_path / name))
    conn.executescript(_BASE_SCHEMA)
    conn.commit()
    return conn


# ---------- Availability ----------------------------------------------------


class TestFts5Availability:
    def test_probe_returns_true_on_fts5_capable_sqlite(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "t.db"))
        assert fts5_available(conn) is True
        conn.close()

    def test_probe_is_side_effect_free(self, tmp_path):
        """Probe must not leave any tables behind."""
        conn = sqlite3.connect(str(tmp_path / "t.db"))
        fts5_available(conn)
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            ).fetchall()
        ]
        assert tables == []
        conn.close()


# ---------- Table creation --------------------------------------------------


class TestCreateTables:
    def test_install_creates_all_fts_tables(self, tmp_path):
        conn = _fresh_conn(tmp_path)
        assert install_fts5(conn) is True
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "commits_fts",
            "discussions_fts",
            "triples_fts",
            "patterns_fts",
        } <= tables
        conn.close()

    def test_install_is_idempotent(self, tmp_path):
        conn = _fresh_conn(tmp_path)
        assert install_fts5(conn) is True
        # Second call must not raise and must still return True.
        assert install_fts5(conn) is True
        conn.close()


# ---------- Trigger propagation ---------------------------------------------


class TestTriggers:
    def test_insert_propagates_to_fts(self, tmp_path):
        conn = _fresh_conn(tmp_path)
        install_fts5(conn)
        conn.execute(
            "INSERT INTO commits (title, what, why, next_step, files_json) "
            "VALUES (?,?,?,?,?)",
            ("Add auth", "Added authentication", "security", "ship it", "[]"),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT rowid FROM commits_fts WHERE commits_fts MATCH 'authentication'"
        ).fetchall()
        assert len(rows) == 1
        conn.close()

    def test_delete_propagates_to_fts(self, tmp_path):
        conn = _fresh_conn(tmp_path)
        install_fts5(conn)
        conn.execute(
            "INSERT INTO commits (title, what, why, next_step, files_json) "
            "VALUES (?,?,?,?,?)",
            ("Auth", "authentication middleware", "sec", "ship", "[]"),
        )
        conn.commit()
        # Sanity: row is indexed.
        assert conn.execute(
            "SELECT count(*) FROM commits_fts WHERE commits_fts MATCH 'authentication'"
        ).fetchone()[0] == 1
        # Delete from base table -> FTS entry disappears via AFTER DELETE trigger.
        conn.execute("DELETE FROM commits")
        conn.commit()
        assert conn.execute(
            "SELECT count(*) FROM commits_fts WHERE commits_fts MATCH 'authentication'"
        ).fetchone()[0] == 0
        conn.close()

    def test_update_propagates_to_fts(self, tmp_path):
        conn = _fresh_conn(tmp_path)
        install_fts5(conn)
        conn.execute(
            "INSERT INTO commits (title, what, why, next_step, files_json) "
            "VALUES (?,?,?,?,?)",
            ("Auth", "authentication middleware", "sec", "ship", "[]"),
        )
        conn.commit()
        # Update: replace 'authentication' with 'authorization'.
        conn.execute(
            "UPDATE commits SET what = ? WHERE rowid = 1",
            ("authorization middleware",),
        )
        conn.commit()
        # Old term gone, new term findable.
        assert conn.execute(
            "SELECT count(*) FROM commits_fts WHERE commits_fts MATCH 'authentication'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT count(*) FROM commits_fts WHERE commits_fts MATCH 'authorization'"
        ).fetchone()[0] == 1
        conn.close()


# Sanity check that all DDL constants are accessible (smoke).
def test_ddl_constants_present():
    assert "commits_fts" in CREATE_COMMITS_FTS
    assert "discussions_fts" in CREATE_DISCUSSIONS_FTS
    assert "triples_fts" in CREATE_TRIPLES_FTS
    assert "patterns_fts" in CREATE_PATTERNS_FTS


# ---------- Backend integration --------------------------------------------


class TestBackendIntegration:
    """SqliteStorageBackend installs FTS5 during init."""

    def test_backend_init_sets_fts_available_true(self, tmp_path):
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        try:
            assert backend.fts_available is True
        finally:
            backend.close()

    def test_backend_init_creates_commits_fts(self, tmp_path):
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        try:
            conn = backend._memory_mgr.get_conn()
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            assert "commits_fts" in tables
            assert "discussions_fts" in tables
            assert "triples_fts" in tables
            assert "patterns_fts" in tables
        finally:
            backend.close()

    def test_backend_init_installs_triggers(self, tmp_path):
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        try:
            conn = backend._memory_mgr.get_conn()
            triggers = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()}
            # 4 events x 4 sources = 16 triggers
            expected = {
                f"{src}_{evt}"
                for src in ("commits", "discussions", "triples", "patterns")
                for evt in ("ai", "ad", "bu", "au")
            }
            assert expected <= triggers
        finally:
            backend.close()


# ---------- Task 2.3: Upgraded search methods -------------------------------


class TestSearchMethodsFTS5:
    """SQLite backend: FTS5-based search preferred, LIKE fallback preserved."""

    def _backend(self, tmp_path):
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        return SqliteStorageBackend(str(ccr))

    def test_commit_search_uses_fts5(self, tmp_path):
        b = self._backend(tmp_path)
        try:
            b.commit_insert("main", {
                "id": "C001", "title": "Auth",
                "what": "Added authentication middleware",
                "why": "security", "next_step": "deploy",
                "files": ["auth.py"], "patterns": [],
                "timestamp": "2026-04-18T10:00:00+00:00",
                "raw_block": "raw", "score": 0.5,
            })
            b.commit_insert("main", {
                "id": "C002", "title": "Logs",
                "what": "Added structured logging",
                "why": "observability", "next_step": "monitor",
                "files": ["log.py"], "patterns": [],
                "timestamp": "2026-04-18T11:00:00+00:00",
                "raw_block": "raw", "score": 0.5,
            })
            hits = b.commit_search_text("main", "authentication", max_results=5)
            assert len(hits) == 1
            assert "authentication" in hits[0]["what"].lower()
        finally:
            b.close()

    def test_commit_search_fts5_malformed_falls_back(self, tmp_path):
        """Malformed FTS query (unclosed quote) must not crash — falls back to LIKE."""
        b = self._backend(tmp_path)
        try:
            b.commit_insert("main", {
                "id": "C001", "title": "Test", "what": "test content",
                "why": "", "next_step": "", "files": [], "patterns": [],
                "timestamp": "2026-04-18T10:00:00+00:00",
                "raw_block": "raw", "score": 0.5,
            })
            # Malformed FTS5 syntax — should not raise
            hits = b.commit_search_text("main", 'unclosed "quote', max_results=5)
            # Either returns [] or matches via LIKE — key is no exception
            assert isinstance(hits, list)
        finally:
            b.close()

    def test_pattern_search_text_sqlite_and_file(self, tmp_path):
        b = self._backend(tmp_path)
        try:
            # Insert a pattern directly
            data = {"version": 1, "patterns": {
                "p-001": {"text": "When adding auth, always test edge cases",
                          "occurrence_count": 1, "success_count": 0, "failure_count": 0,
                          "quality_score": 0.5, "first_seen": "2026-04-18",
                          "last_seen": "2026-04-18", "promoted": False,
                          "commit_ids": []},
            }}
            b.pattern_save_all(data)
            hits = b.pattern_search_text("auth", max_results=5)
            assert len(hits) == 1
            assert "auth" in hits[0]["text"].lower()
        finally:
            b.close()

    def test_discussion_search_text_sqlite(self, tmp_path):
        b = self._backend(tmp_path)
        try:
            b.discussion_insert("main", {
                "id": "D001",
                "topic": "cache strategy",
                "hypothesis": "LRU vs TTL",
                "alternatives": "", "decision": "LRU",
                "rationale": "simpler",
                "uncertainty": "", "linked_commit": "",
                "timestamp": "2026-04-18T10:00:00+00:00",
            })
            hits = b.discussion_search_text("main", "cache", max_results=5)
            assert len(hits) == 1
            assert "cache" in hits[0]["topic"].lower()
        finally:
            b.close()


class TestBackfill:
    """Phase 2.4: Backfill FTS5 for databases created before Phase 2."""

    def test_backfill_populates_commits_fts_on_upgrade(self, tmp_path):
        """Pre-Phase-2 DB: commits rows exist but commits_fts is empty.
        After reopening with Phase-2 backend, FTS5 must contain them."""
        import sqlite3
        db = tmp_path / ".ccr" / "memory.db"
        db.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(db))
        # Simulate pre-Phase-2 schema: commits table + rows, NO FTS tables/triggers
        conn.executescript('''
            CREATE TABLE commits (
                rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
                id          TEXT NOT NULL,
                branch      TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                title       TEXT NOT NULL DEFAULT '',
                what        TEXT NOT NULL DEFAULT '',
                why         TEXT NOT NULL DEFAULT '',
                files_json  TEXT NOT NULL DEFAULT '[]',
                next_step   TEXT NOT NULL DEFAULT '',
                patterns_json TEXT NOT NULL DEFAULT '[]',
                score       REAL DEFAULT 0.0,
                author      TEXT DEFAULT '',
                ci_json     TEXT DEFAULT '',
                experiment_json TEXT DEFAULT '',
                ota_trace   TEXT DEFAULT '',
                raw_block   TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
            );
            INSERT INTO commits (id, branch, timestamp, title, what, why, next_step, files_json, raw_block, created_at)
            VALUES ('C001', 'main', '2026-04-18T10:00:00+00:00', 'Auth', 'Added authentication', 'security', 'ship', '[]', 'raw', '2026-04-18T10:00:00+00:00');
            PRAGMA user_version = 0;
        ''')
        conn.commit()
        conn.close()

        # Open with Phase 2 backend — should install FTS5 and backfill
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        backend = SqliteStorageBackend(str(tmp_path / ".ccr"))
        try:
            assert backend.fts_available is True
            # user_version bumped
            ver = backend._memory_mgr.get_user_version()
            assert ver >= 1
            # Search works - commits_fts is populated
            hits = backend.commit_search_text("main", "authentication", max_results=5)
            assert len(hits) == 1
            assert hits[0]["id"] == "C001"
        finally:
            backend.close()

    def test_backfill_idempotent(self, tmp_path):
        """Re-opening a Phase-2 DB does not re-run backfill."""
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        # First open: creates schema + installs FTS5 + backfills (on empty)
        b1 = SqliteStorageBackend(str(ccr))
        try:
            b1.commit_insert("main", {
                "id": "C001", "title": "T", "what": "W",
                "why": "Y", "next_step": "N",
                "files": [], "patterns": [],
                "timestamp": "2026-04-18T10:00:00+00:00",
                "raw_block": "raw", "score": 0.5,
            })
            assert b1._memory_mgr.get_user_version() >= 1
        finally:
            b1.close()
        # Second open — should NOT re-run backfill (but search still works)
        b2 = SqliteStorageBackend(str(ccr))
        try:
            hits = b2.commit_search_text("main", "W", max_results=5)
            assert len(hits) == 1
        finally:
            b2.close()

    def test_backfill_fresh_db_no_op(self, tmp_path):
        """Fresh DB: empty commits table; backfill runs but inserts nothing."""
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        try:
            assert backend.fts_available is True
            # user_version bumped even on fresh DB
            ver = backend._memory_mgr.get_user_version()
            assert ver >= 1
        finally:
            backend.close()


# ---------- Task 2.5: gcc_search snippets -----------------------------------


class TestGccSearchSnippets:
    """Phase 2.5: gcc_search returns FTS5 snippets + rank when available."""

    def test_commit_search_with_snippet_returns_snippet_field(self, tmp_path):
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        b = SqliteStorageBackend(str(ccr))
        try:
            b.commit_insert("main", {
                "id": "C001", "title": "Auth",
                "what": "Added authentication middleware for the login flow",
                "why": "security", "next_step": "deploy",
                "files": ["auth.py"], "patterns": [],
                "timestamp": "2026-04-18T10:00:00+00:00",
                "raw_block": "raw", "score": 0.5,
            })
            results = b.commit_search_with_snippet("main", "authentication", max_results=5)
            assert len(results) == 1
            assert "snippet" in results[0]
            assert "[authentication]" in results[0]["snippet"] or "authentication" in results[0]["snippet"]
            assert "rank" in results[0]
            assert isinstance(results[0]["rank"], float)
        finally:
            b.close()

    def test_commit_search_with_snippet_returns_empty_on_no_match(self, tmp_path):
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        b = SqliteStorageBackend(str(ccr))
        try:
            b.commit_insert("main", {
                "id": "C001", "title": "Auth",
                "what": "Added authentication",
                "why": "security", "next_step": "deploy",
                "files": [], "patterns": [],
                "timestamp": "2026-04-18T10:00:00+00:00",
                "raw_block": "raw", "score": 0.5,
            })
            results = b.commit_search_with_snippet("main", "nonsense_term", max_results=5)
            assert results == []
        finally:
            b.close()

    def test_commit_search_with_snippet_malformed_query_safe(self, tmp_path):
        """Malformed FTS5 query must not crash — returns []."""
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        b = SqliteStorageBackend(str(ccr))
        try:
            b.commit_insert("main", {
                "id": "C001", "title": "T", "what": "text", "why": "",
                "next_step": "", "files": [], "patterns": [],
                "timestamp": "2026-04-18T10:00:00+00:00",
                "raw_block": "r", "score": 0.5,
            })
            # Malformed FTS5 syntax (unclosed quote)
            results = b.commit_search_with_snippet("main", 'bad "syntax', max_results=5)
            assert results == []  # Graceful — no exception
        finally:
            b.close()
