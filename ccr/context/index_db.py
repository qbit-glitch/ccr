"""SQLite persistence layer for the repo index.

Stores file metadata, chunks, and FTS5 full-text index in index.db.
Supports incremental builds via mtime comparison and FTS5 search.

Schema:
  index_files — file metadata (path, language, symbols, imports, etc.)
  index_fts   — FTS5 virtual table for full-text search on path + symbols
  index_chunks — chunk metadata + text for chunk-level search
  index_meta   — key/value store for build metadata (built_at, mtime_sig)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS index_files (
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
CREATE INDEX IF NOT EXISTS idx_files_path ON index_files(path);

CREATE TABLE IF NOT EXISTS index_chunks (
    file_path TEXT NOT NULL,
    chunk_idx INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (file_path, chunk_idx)
);

CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS index_fts USING fts5(
    path,
    symbols_text,
    imports_text,
    content='index_files',
    content_rowid='rowid'
);
"""

_FTS_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS index_fts_ai AFTER INSERT ON index_files BEGIN
    INSERT INTO index_fts(rowid, path, symbols_text, imports_text)
    VALUES (new.rowid, new.path,
            REPLACE(new.symbols_json, '"', ''),
            REPLACE(new.imports_json, '"', ''));
END;
"""

_FTS_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS index_fts_ad AFTER DELETE ON index_files BEGIN
    INSERT INTO index_fts(index_fts, rowid, path, symbols_text, imports_text)
    VALUES ('delete', old.rowid, old.path,
            REPLACE(old.symbols_json, '"', ''),
            REPLACE(old.imports_json, '"', ''));
END;
"""

_FTS_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS index_fts_au AFTER UPDATE ON index_files BEGIN
    INSERT INTO index_fts(index_fts, rowid, path, symbols_text, imports_text)
    VALUES ('delete', old.rowid, old.path,
            REPLACE(old.symbols_json, '"', ''),
            REPLACE(old.imports_json, '"', ''));
    INSERT INTO index_fts(rowid, path, symbols_text, imports_text)
    VALUES (new.rowid, new.path,
            REPLACE(new.symbols_json, '"', ''),
            REPLACE(new.imports_json, '"', ''));
END;
"""

# ── Phase 3: chunk-level FTS5 on index_chunks.text ────────────────────
# Mirrors the Phase 2 memory.db pattern (external content + 4 triggers).

_CHUNKS_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS index_chunks_fts USING fts5(
    text,
    content='index_chunks',
    content_rowid='rowid'
);
"""

_CHUNKS_FTS_TRIGGER_AI = """
CREATE TRIGGER IF NOT EXISTS index_chunks_fts_ai AFTER INSERT ON index_chunks BEGIN
    INSERT INTO index_chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""

_CHUNKS_FTS_TRIGGER_AD = """
CREATE TRIGGER IF NOT EXISTS index_chunks_fts_ad AFTER DELETE ON index_chunks BEGIN
    INSERT INTO index_chunks_fts(index_chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;
"""

_CHUNKS_FTS_TRIGGER_BU = """
CREATE TRIGGER IF NOT EXISTS index_chunks_fts_bu BEFORE UPDATE ON index_chunks BEGIN
    INSERT INTO index_chunks_fts(index_chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;
"""

_CHUNKS_FTS_TRIGGER_AU = """
CREATE TRIGGER IF NOT EXISTS index_chunks_fts_au AFTER UPDATE ON index_chunks BEGIN
    INSERT INTO index_chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""


class IndexDB:
    """SQLite persistence for the repo index.

    Thread-safe via per-thread connections. WAL mode for concurrent reads.
    FTS5 for full-text search on file paths and symbols.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        self._fts_available = False
        self._setup()

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self._db_path)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=5000")
            self._local.conn = c
        return c

    def _setup(self) -> None:
        conn = self.conn
        conn.executescript(_DDL)
        try:
            conn.executescript(_FTS_DDL)
            conn.executescript(_FTS_TRIGGER_INSERT)
            conn.executescript(_FTS_TRIGGER_DELETE)
            conn.executescript(_FTS_TRIGGER_UPDATE)
            conn.executescript(_CHUNKS_FTS_DDL)
            conn.executescript(_CHUNKS_FTS_TRIGGER_AI)
            conn.executescript(_CHUNKS_FTS_TRIGGER_AD)
            conn.executescript(_CHUNKS_FTS_TRIGGER_BU)
            conn.executescript(_CHUNKS_FTS_TRIGGER_AU)
            self._fts_available = True
        except sqlite3.OperationalError:
            logger.debug("FTS5 not available, falling back to LIKE search")
            self._fts_available = False
            return
        # Schema-version guard: backfill existing chunks on first upgrade.
        # v0 → pre-Phase-3 (no chunks_fts index); v1 → chunks backfilled.
        cur_ver = conn.execute("PRAGMA user_version").fetchone()[0]
        if cur_ver < 1:
            self._backfill_chunks_fts()
            conn.execute("PRAGMA user_version = 1")
            conn.commit()

    # ── File CRUD ────────────────────────────────────────────────

    def upsert_file(
        self, path: str, language: str, line_count: int,
        size_bytes: int, mtime: float, symbols: list[str],
        imports: list[str], git_hash: str, indexed_at: str,
    ) -> None:
        self.conn.execute(
            """INSERT INTO index_files
               (path, language, line_count, size_bytes, mtime,
                symbols_json, imports_json, git_hash, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                language=excluded.language,
                line_count=excluded.line_count,
                size_bytes=excluded.size_bytes,
                mtime=excluded.mtime,
                symbols_json=excluded.symbols_json,
                imports_json=excluded.imports_json,
                git_hash=excluded.git_hash,
                indexed_at=excluded.indexed_at""",
            (path, language, line_count, size_bytes, mtime,
             json.dumps(symbols), json.dumps(imports), git_hash, indexed_at),
        )

    def save_files_batch(
        self, files: list[dict], indexed_at: str,
    ) -> int:
        """Bulk upsert files. Returns count of files saved."""
        conn = self.conn
        count = 0
        for f in files:
            conn.execute(
                """INSERT INTO index_files
                   (path, language, line_count, size_bytes, mtime,
                    symbols_json, imports_json, git_hash, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                    language=excluded.language,
                    line_count=excluded.line_count,
                    size_bytes=excluded.size_bytes,
                    mtime=excluded.mtime,
                    symbols_json=excluded.symbols_json,
                    imports_json=excluded.imports_json,
                    git_hash=excluded.git_hash,
                    indexed_at=excluded.indexed_at""",
                (
                    f["path"], f.get("language", ""),
                    f.get("line_count", 0), f.get("size_bytes", 0),
                    f.get("mtime", 0), json.dumps(f.get("symbols", [])),
                    json.dumps(f.get("imports", [])),
                    f.get("git_hash", ""), indexed_at,
                ),
            )
            count += 1
        conn.commit()
        return count

    def load_files(self) -> list[dict]:
        """Load all indexed files as dicts."""
        rows = self.conn.execute("SELECT * FROM index_files").fetchall()
        result = []
        for r in rows:
            try:
                symbols = json.loads(r["symbols_json"])
            except (json.JSONDecodeError, TypeError):
                symbols = []
            try:
                imports = json.loads(r["imports_json"])
            except (json.JSONDecodeError, TypeError):
                imports = []
            result.append({
                "path": r["path"],
                "language": r["language"],
                "line_count": r["line_count"],
                "size_bytes": r["size_bytes"],
                "mtime": r["mtime"],
                "symbols": symbols,
                "imports": imports,
                "git_hash": r["git_hash"] or "",
                "indexed_at": r["indexed_at"],
            })
        return result

    def get_file(self, path: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM index_files WHERE path = ?", (path,),
        ).fetchone()
        if row is None:
            return None
        try:
            symbols = json.loads(row["symbols_json"])
        except (json.JSONDecodeError, TypeError):
            symbols = []
        try:
            imports = json.loads(row["imports_json"])
        except (json.JSONDecodeError, TypeError):
            imports = []
        return {
            "path": row["path"],
            "language": row["language"],
            "line_count": row["line_count"],
            "size_bytes": row["size_bytes"],
            "mtime": row["mtime"],
            "symbols": symbols,
            "imports": imports,
            "git_hash": row["git_hash"] or "",
            "indexed_at": row["indexed_at"],
        }

    def delete_file(self, path: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM index_files WHERE path = ?", (path,),
        )
        self.conn.execute(
            "DELETE FROM index_chunks WHERE file_path = ?", (path,),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_missing(self, existing_paths: set[str]) -> int:
        """Remove files from index that no longer exist on disk."""
        rows = self.conn.execute(
            "SELECT path FROM index_files",
        ).fetchall()
        to_delete = [r["path"] for r in rows if r["path"] not in existing_paths]
        if not to_delete:
            return 0
        for p in to_delete:
            self.conn.execute("DELETE FROM index_files WHERE path = ?", (p,))
            self.conn.execute("DELETE FROM index_chunks WHERE file_path = ?", (p,))
        self.conn.commit()
        return len(to_delete)

    def get_file_mtimes(self) -> dict[str, float]:
        """Return {path: mtime} for all indexed files. Used for incremental builds."""
        rows = self.conn.execute(
            "SELECT path, mtime FROM index_files",
        ).fetchall()
        return {r["path"]: r["mtime"] for r in rows}

    def file_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as c FROM index_files").fetchone()
        return row["c"] if row else 0

    # ── Chunk CRUD ───────────────────────────────────────────────

    def save_chunks_batch(self, chunks: list[dict]) -> int:
        conn = self.conn
        count = 0
        for ch in chunks:
            conn.execute(
                """INSERT INTO index_chunks
                   (file_path, chunk_idx, start_line, end_line, text)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(file_path, chunk_idx) DO UPDATE SET
                    start_line=excluded.start_line,
                    end_line=excluded.end_line,
                    text=excluded.text""",
                (
                    ch["file_path"], ch["chunk_idx"],
                    ch["start_line"], ch["end_line"],
                    ch.get("text", ""),
                ),
            )
            count += 1
        conn.commit()
        return count

    def load_chunks(self, file_path: str | None = None) -> list[dict]:
        if file_path:
            rows = self.conn.execute(
                "SELECT * FROM index_chunks WHERE file_path = ? ORDER BY chunk_idx",
                (file_path,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM index_chunks ORDER BY file_path, chunk_idx",
            ).fetchall()
        return [
            {
                "file_path": r["file_path"],
                "chunk_idx": r["chunk_idx"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "text": r["text"],
            }
            for r in rows
        ]

    def chunk_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as c FROM index_chunks").fetchone()
        return row["c"] if row else 0

    # ── FTS5 Search ──────────────────────────────────────────────

    def fts_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Full-text search on file paths and symbols via FTS5.

        Falls back to LIKE search if FTS5 is unavailable.
        """
        if self._fts_available:
            return self._fts5_search(query, top_k)
        return self._like_search(query, top_k)

    def _fts5_search(self, query: str, top_k: int = 10) -> list[dict]:
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []
        try:
            rows = self.conn.execute(
                """SELECT f.path, f.language, f.symbols_json, f.imports_json,
                          f.line_count, f.size_bytes,
                          rank AS score
                   FROM index_fts fts
                   JOIN index_files f ON f.rowid = fts.rowid
                   WHERE index_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return self._like_search(query, top_k)
        return self._rows_to_results(rows)

    def _like_search(self, query: str, top_k: int = 10) -> list[dict]:
        pattern = f"%{query}%"
        rows = self.conn.execute(
            """SELECT path, language, symbols_json, imports_json,
                      line_count, size_bytes, 0 as score
               FROM index_files
               WHERE path LIKE ? OR symbols_json LIKE ? OR imports_json LIKE ?
               LIMIT ?""",
            (pattern, pattern, pattern, top_k),
        ).fetchall()
        return self._rows_to_results(rows)

    def _rows_to_results(self, rows: list) -> list[dict]:
        results = []
        for r in rows:
            try:
                symbols = json.loads(r["symbols_json"])
            except (json.JSONDecodeError, TypeError):
                symbols = []
            try:
                imports = json.loads(r["imports_json"])
            except (json.JSONDecodeError, TypeError):
                imports = []
            results.append({
                "path": r["path"],
                "language": r["language"],
                "symbols": symbols,
                "imports": imports,
                "line_count": r["line_count"],
                "size_bytes": r["size_bytes"],
                "score": abs(r["score"]) if r["score"] else 0,
            })
        return results

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize a query for FTS5 MATCH syntax."""
        tokens = []
        for word in query.split():
            clean = "".join(c for c in word if c.isalnum() or c in "_-./")
            if clean:
                tokens.append(f'"{clean}"')
        return " OR ".join(tokens)

    # ── Chunk-level FTS5 search (Phase 3) ────────────────────────

    def chunk_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Full-text search on chunk code content via FTS5.

        Falls back to LIKE search if FTS5 is unavailable or query is malformed.
        Returns up to ``top_k`` chunk dicts ordered by BM25 rank (best first).
        """
        if self._fts_available:
            safe = self._sanitize_fts_query(query)
            if safe:
                try:
                    rows = self.conn.execute(
                        """SELECT c.file_path, c.chunk_idx, c.start_line, c.end_line,
                                  c.text, rank AS score
                           FROM index_chunks_fts f
                           JOIN index_chunks c ON f.rowid = c.rowid
                           WHERE index_chunks_fts MATCH ?
                           ORDER BY rank
                           LIMIT ?""",
                        (safe, top_k),
                    ).fetchall()
                    return [
                        {
                            "file_path": r["file_path"],
                            "chunk_idx": r["chunk_idx"],
                            "start_line": r["start_line"],
                            "end_line": r["end_line"],
                            "text": r["text"],
                            "score": abs(r["score"]) if r["score"] else 0,
                        }
                        for r in rows
                    ]
                except sqlite3.OperationalError:
                    pass  # Fall through to LIKE
        pat = f"%{query}%"
        rows = self.conn.execute(
            """SELECT file_path, chunk_idx, start_line, end_line, text, 0 AS score
               FROM index_chunks
               WHERE text LIKE ?
               LIMIT ?""",
            (pat, top_k),
        ).fetchall()
        return [
            {
                "file_path": r["file_path"],
                "chunk_idx": r["chunk_idx"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "text": r["text"],
                "score": 0,
            }
            for r in rows
        ]

    def chunk_search_with_snippet(
        self, query: str, top_k: int = 10,
    ) -> list[dict]:
        """FTS5-only chunk search returning BM25 rank + highlighted snippet.

        Returns empty list if FTS5 is unavailable or query is malformed —
        callers should fall back to ``chunk_search`` (LIKE) in those cases.
        The snippet wraps matches in ``[...]`` and uses ``...`` ellipsis for
        elided context, with up to 12 surrounding tokens.
        """
        if not self._fts_available:
            return []
        safe = self._sanitize_fts_query(query)
        if not safe:
            return []
        try:
            rows = self.conn.execute(
                """SELECT c.file_path, c.chunk_idx, c.start_line, c.end_line,
                          c.text,
                          snippet(index_chunks_fts, -1, '[', ']', '...', 12)
                              AS snippet_text,
                          rank
                   FROM index_chunks_fts f
                   JOIN index_chunks c ON f.rowid = c.rowid
                   WHERE index_chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {
                "file_path": r["file_path"],
                "chunk_idx": r["chunk_idx"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "text": r["text"],
                "snippet": r["snippet_text"],
                "rank": r["rank"],
            }
            for r in rows
        ]

    def _backfill_chunks_fts(self) -> int:
        """Rebuild index_chunks_fts from index_chunks.

        Uses the FTS5 'rebuild' command — triggers do not fire during rebuild.
        Returns the row count in the FTS index after rebuild.
        """
        if not self._fts_available:
            return 0
        got = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            ("index_chunks_fts",),
        ).fetchone()
        if got is None:
            return 0
        self.conn.execute(
            "INSERT INTO index_chunks_fts(index_chunks_fts) VALUES ('rebuild')"
        )
        n = self.conn.execute(
            "SELECT count(*) FROM index_chunks_fts"
        ).fetchone()[0]
        self.conn.commit()
        return n

    # ── Meta KV ──────────────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,),
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def fts_available(self) -> bool:
        return self._fts_available

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None
