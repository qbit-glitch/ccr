# Phase 3: Chunk-level FTS5 + Snippets on index.db — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `index.db` FTS5 coverage from file metadata (path/symbols/imports) to the actual code content stored in `index_chunks.text`, so `index_search` can return BM25-ranked code snippets with zero external dependencies — a grep-replacement that works without ONNX embeddings.

**Architecture:** Add one external-content FTS5 virtual table `index_chunks_fts` on `index_chunks(text)` with 4 sync triggers (AI/AD/BU/AU) mirroring the Phase 2 `commits_fts` / `discussions_fts` pattern. Expose `chunk_search` and `chunk_search_with_snippet` on `IndexDB`, then wire into the `index_search` MCP tool as a new `mode="code"` path. Backfill existing DBs via `PRAGMA user_version` guard, same pattern proven in Phase 2.4.

**Tech Stack:** Python `sqlite3` stdlib (FTS5 module — already probed via existing `_fts_available` flag), `pytest` parametrized tests, thread-local WAL connection manager (already present on `IndexDB`).

---

## Scope & Affected Files

### Create
- `tests/unit/test_chunk_fts.py` — dedicated tests for chunk-level FTS5 (module-level skip when FTS5 missing)

### Modify
- `ccr/context/index_db.py`:
  - Add `_CHUNKS_FTS_DDL` constant (virtual table on `index_chunks.text`)
  - Add `_CHUNKS_FTS_TRIGGER_{AI,AD,BU,AU}` constants (4 triggers)
  - Extend `_setup()` to create the new FTS table + triggers
  - Extend `_setup()` to run `backfill_chunks_fts` when `PRAGMA user_version < 2`
  - Add `chunk_search(query, top_k) -> list[dict]` (FTS5 MATCH, LIKE fallback)
  - Add `chunk_search_with_snippet(query, top_k) -> list[dict]` (FTS5-only, returns `snippet_text` + `rank`)
  - Add `_backfill_chunks_fts()` private helper
- `ccr/mcp/index_tools.py`:
  - Extend `index_search` to accept `mode="code"` — returns chunk-level hits with `path:start-end` + snippet
  - Wire `_index_db.chunk_search_with_snippet(...)` when `_index_db.fts_available`
- `tests/unit/test_indexer.py`:
  - Add assertions covering the new `chunk_search` path (graceful LIKE fallback when no FTS5)

**Non-goals for this phase:**
- No changes to `index_files` FTS (already working)
- No new MCP tool signature — `mode="code"` is an additive string value
- No chunk embedding changes (ONNX path untouched)

**Principle:** Zero-dep BM25 code search is an opportunistic addition. Existing hybrid/semantic/keyword modes are unchanged. Schema version bumps from 1 (memory.db FTS5) to 2 (index chunk FTS5). Note: `index.db` and `memory.db` are separate DBs; the `user_version` pragma on `index.db` independently tracks its own migrations — use 1 there (not 2), since this is the first migration applied to `index.db` under the v5 regime.

---

## Task 3.1: Chunk FTS5 DDL + triggers in IndexDB

**Files:**
- Modify: `ccr/context/index_db.py` (add DDL constants + `_setup` call)
- Create: `tests/unit/test_chunk_fts.py`

### - [ ] Step 1: Write the failing availability + DDL test

Create `tests/unit/test_chunk_fts.py`:

```python
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
```

### - [ ] Step 2: Run test — expect FAILURE (tables/triggers don't exist yet)

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_chunk_fts.py::TestChunkFtsSchema -v
```

Expected: 2 tests FAIL with `AssertionError: "index_chunks_fts" missing` or similar.

### - [ ] Step 3: Add DDL constants + wire into `_setup`

In `ccr/context/index_db.py`, after the existing `_FTS_TRIGGER_UPDATE` string (around line 93), add:

```python
_CHUNKS_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS index_chunks_fts USING fts5(
    text,
    content='index_chunks',
    content_rowid='rowid'
);
"""

# 4 triggers: AFTER INSERT / AFTER DELETE / BEFORE UPDATE / AFTER UPDATE.
# Mirrors the Phase 2 memory.db pattern (commits_ai/ad/bu/au).
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
```

Then inside `IndexDB._setup`, extend the existing FTS install block (currently at ~lines 120-131) from:

```python
def _setup(self) -> None:
    conn = self.conn
    conn.executescript(_DDL)
    try:
        conn.executescript(_FTS_DDL)
        conn.executescript(_FTS_TRIGGER_INSERT)
        conn.executescript(_FTS_TRIGGER_DELETE)
        conn.executescript(_FTS_TRIGGER_UPDATE)
        self._fts_available = True
    except sqlite3.OperationalError:
        logger.debug("FTS5 not available, falling back to LIKE search")
        self._fts_available = False
```

to:

```python
def _setup(self) -> None:
    conn = self.conn
    conn.executescript(_DDL)
    try:
        conn.executescript(_FTS_DDL)
        conn.executescript(_FTS_TRIGGER_INSERT)
        conn.executescript(_FTS_TRIGGER_DELETE)
        conn.executescript(_FTS_TRIGGER_UPDATE)
        # Phase 3: chunk-level FTS5 on index_chunks.text
        conn.executescript(_CHUNKS_FTS_DDL)
        conn.executescript(_CHUNKS_FTS_TRIGGER_AI)
        conn.executescript(_CHUNKS_FTS_TRIGGER_AD)
        conn.executescript(_CHUNKS_FTS_TRIGGER_BU)
        conn.executescript(_CHUNKS_FTS_TRIGGER_AU)
        self._fts_available = True
    except sqlite3.OperationalError:
        logger.debug("FTS5 not available, falling back to LIKE search")
        self._fts_available = False
```

### - [ ] Step 4: Run tests — expect PASS

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_chunk_fts.py::TestChunkFtsSchema -v
```

Expected: 2 tests PASS.

### - [ ] Step 5: Commit

```bash
git add ccr/context/index_db.py tests/unit/test_chunk_fts.py
git commit -m "ccr: Phase 3.1 — chunk-level FTS5 virtual table + triggers on index.db"
```

---

## Task 3.2: `chunk_search` method with LIKE fallback

**Files:**
- Modify: `ccr/context/index_db.py` (add `chunk_search`)
- Test: `tests/unit/test_chunk_fts.py::TestChunkSearch`

### - [ ] Step 1: Write the failing search test

Append to `tests/unit/test_chunk_fts.py`:

```python
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
        # Unclosed quote — FTS5 raises OperationalError; we must still return []
        # (LIKE won't match an arbitrary special char), not crash.
        result = db.chunk_search('"', top_k=5)
        assert isinstance(result, list)
        db.close()
```

### - [ ] Step 2: Run tests — expect AttributeError (method doesn't exist)

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_chunk_fts.py::TestChunkSearch -v
```

Expected: 3 tests FAIL with `AttributeError: 'IndexDB' object has no attribute 'chunk_search'`.

### - [ ] Step 3: Implement `chunk_search` in `index_db.py`

Add after the existing `_sanitize_fts_query` staticmethod (around line 400):

```python
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
    # LIKE fallback
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
```

### - [ ] Step 4: Run tests — expect PASS

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_chunk_fts.py::TestChunkSearch -v
```

Expected: 3 tests PASS.

### - [ ] Step 5: Commit

```bash
git add ccr/context/index_db.py tests/unit/test_chunk_fts.py
git commit -m "ccr: Phase 3.2 — chunk_search with FTS5 MATCH + LIKE fallback"
```

---

## Task 3.3: `chunk_search_with_snippet` for BM25 + highlighted snippets

**Files:**
- Modify: `ccr/context/index_db.py` (add `chunk_search_with_snippet`)
- Test: `tests/unit/test_chunk_fts.py::TestChunkSnippets`

### - [ ] Step 1: Write the failing snippet test

Append to `tests/unit/test_chunk_fts.py`:

```python
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

    def test_snippet_returns_empty_when_fts5_unavailable(self, tmp_path, monkeypatch):
        db = IndexDB(str(tmp_path / "index.db"))
        db.save_chunks_batch([{
            "file_path": "a.py", "chunk_idx": 0,
            "start_line": 1, "end_line": 1, "text": "hello",
        }])
        monkeypatch.setattr(db, "_fts_available", False)
        assert db.chunk_search_with_snippet("hello", top_k=5) == []
        db.close()
```

### - [ ] Step 2: Run tests — expect AttributeError

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_chunk_fts.py::TestChunkSnippets -v
```

Expected: 2 tests FAIL with `AttributeError`.

### - [ ] Step 3: Implement `chunk_search_with_snippet`

Add after `chunk_search` in `index_db.py`:

```python
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
```

### - [ ] Step 4: Run tests — expect PASS

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_chunk_fts.py::TestChunkSnippets -v
```

Expected: 2 tests PASS.

### - [ ] Step 5: Commit

```bash
git add ccr/context/index_db.py tests/unit/test_chunk_fts.py
git commit -m "ccr: Phase 3.3 — chunk_search_with_snippet with BM25 rank + bracket snippets"
```

---

## Task 3.4: Backfill migration for existing index.db files

**Files:**
- Modify: `ccr/context/index_db.py` (add `_backfill_chunks_fts` + guard)
- Test: `tests/unit/test_chunk_fts.py::TestChunkBackfill`

### - [ ] Step 1: Write the failing backfill test

Append to `tests/unit/test_chunk_fts.py`:

```python
class TestChunkBackfill:
    def test_backfill_on_upgrade(self, tmp_path):
        """DB created before chunk FTS5 still gets searchable chunks after upgrade."""
        import sqlite3 as _sqlite3
        db_path = tmp_path / "index.db"
        # Simulate a pre-Phase-3 DB: chunks exist, no chunk FTS.
        conn = _sqlite3.connect(str(db_path))
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
        # And user_version should be bumped.
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
        # Re-open — no error, still searchable
        db2 = IndexDB(str(db_path))
        hits = db2.chunk_search("first", top_k=5)
        assert len(hits) == 1
        db2.close()
```

### - [ ] Step 2: Run tests — first test FAILS (no backfill), second may PASS already

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_chunk_fts.py::TestChunkBackfill -v
```

Expected: `test_backfill_on_upgrade` FAILS (0 hits instead of 1).

### - [ ] Step 3: Implement backfill with user_version guard

In `index_db.py`, replace the `_setup` body with the version-guarded form:

```python
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
```

Then add the helper after `chunk_search_with_snippet`:

```python
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
```

### - [ ] Step 4: Run tests — expect PASS

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_chunk_fts.py::TestChunkBackfill -v
```

Expected: 2 tests PASS.

### - [ ] Step 5: Commit

```bash
git add ccr/context/index_db.py tests/unit/test_chunk_fts.py
git commit -m "ccr: Phase 3.4 — chunk FTS5 backfill migration with user_version guard"
```

---

## Task 3.5: Wire chunk search into the `index_search` MCP tool

**Files:**
- Modify: `ccr/mcp/index_tools.py` (add `mode="code"` branch)
- Test: `tests/unit/test_indexer.py` (add one path-test)

### - [ ] Step 1: Write failing test asserting `mode="code"` returns snippets

Append to `tests/unit/test_indexer.py` (no FTS5 skip — the test uses the LIKE fallback path so it runs on any SQLite):

```python
class TestIndexSearchCodeMode:
    def test_code_mode_returns_chunk_hits(self, tmp_path, monkeypatch):
        """mode='code' hits index_chunks.text and returns file_path:lines + snippet."""
        # Point CCR at a tmp project
        from ccr.mcp import server as srv
        from ccr.context.index_db import IndexDB
        from ccr.mcp.index_tools import index_search

        project = tmp_path / "proj"
        project.mkdir()
        (project / "src").mkdir()
        (project / "src" / "auth.py").write_text(
            "def authenticate(user, password):\n"
            "    return verify_password(user, password)\n"
        )
        srv._project_root = str(project)
        db_path = project / ".ccr" / "index.db"
        db_path.parent.mkdir(exist_ok=True)
        db = IndexDB(str(db_path))
        db.save_chunks_batch([{
            "file_path": "src/auth.py",
            "chunk_idx": 0,
            "start_line": 1,
            "end_line": 2,
            "text": (
                "def authenticate(user, password):\n"
                "    return verify_password(user, password)"
            ),
        }])
        monkeypatch.setattr(srv, "_index_db", db)

        out = index_search(query="authenticate", mode="code", limit=5)
        assert "src/auth.py" in out
        # Either bracket snippet (FTS5) or raw text (LIKE fallback) is acceptable.
        assert "authenticate" in out
        db.close()
```

### - [ ] Step 2: Run test — expect FAILURE (mode='code' not recognised)

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_indexer.py::TestIndexSearchCodeMode -v
```

Expected: FAIL with mode-validation error or no matching branch.

### - [ ] Step 3: Add `mode="code"` branch to `index_search` in `ccr/mcp/index_tools.py`

Locate the resolved-mode block inside `index_search` (around line 240) and add a `code` branch:

```python
# ── Phase 3: mode="code" — chunk-level FTS5 search ───────────────────
if resolved_mode == "code":
    db = getattr(_srv, "_index_db", None)
    if db is None:
        return "No index built yet. Run index_build first."
    # Prefer FTS5 snippet path; fall back to LIKE-based chunk_search.
    hits: list[dict] = []
    if getattr(db, "fts_available", False):
        try:
            hits = db.chunk_search_with_snippet(query, top_k=limit)
        except Exception:  # pragma: no cover — defensive
            hits = []
    if not hits:
        hits = db.chunk_search(query, top_k=limit)
    if not hits:
        return f"No code matches for '{query}'."
    lines = [f"Code search hits for '{query}':"]
    for h in hits:
        fp = h["file_path"]
        sl, el = h["start_line"], h["end_line"]
        snippet = h.get("snippet") or h.get("text", "")[:200]
        snippet = snippet.replace("\n", " ")
        lines.append(f"- {fp}:{sl}-{el}")
        lines.append(f"    {snippet}")
    return "\n".join(lines)
```

This branch must be added **before** the existing `resolved_mode not in {...}` validation check so the code branch is accepted. Locate the validator (search for `valid_modes` or the `if resolved_mode not in` block around line 235) and add `"code"` to the allowed set:

```python
valid_modes = {"keyword", "semantic", "hybrid", "code"}
```

### - [ ] Step 4: Run test — expect PASS

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_indexer.py::TestIndexSearchCodeMode -v
```

Expected: PASS.

### - [ ] Step 5: Commit

```bash
git add ccr/mcp/index_tools.py tests/unit/test_indexer.py
git commit -m "ccr: Phase 3.5 — index_search mode='code' returns chunk hits with snippets"
```

---

## Task 3.6: E2E validation + full test suite + Phase 3 commit

### - [ ] Step 1: Run full test suite

Run:

```bash
.venv/bin/python -m pytest tests/unit/ -x -q
```

Expected: 2827 previous tests + new Phase 3 tests (≥10) all pass. Target: 2837+ total, 0 failures.

### - [ ] Step 2: Real-data smoke test

Run against the repo's own `.ccr/index.db` (if present):

```bash
.venv/bin/python -c "
from ccr.context.index_db import IndexDB
db = IndexDB('.ccr/index.db')
print('fts_available:', db.fts_available)
print('chunk count:', db.chunk_count())
hits = db.chunk_search('commit', top_k=3)
print(f'chunk hits for commit: {len(hits)}')
if hits:
    print('first hit:', hits[0]['file_path'], hits[0]['start_line'], '-', hits[0]['end_line'])
snip = db.chunk_search_with_snippet('commit', top_k=1)
if snip:
    print('snippet:', snip[0]['snippet'])
db.close()
"
```

Expected: `fts_available: True`, chunk count > 0, at least one hit, snippet non-empty with `[commit]` markers.

If the repo has no built index (`.ccr/index.db` missing), run `index_build` first via the MCP tool, or build programmatically and re-run the smoke test.

### - [ ] Step 3: Dispatch code-reviewer agent on Phase 3 diff

Use the `Agent` tool with `subagent_type="code-reviewer"` and prompt:

```
Review the Phase 3 diff on main since c92290d. Focus areas:
- SQL injection risk in chunk_search / chunk_search_with_snippet (parameterized queries, _sanitize_fts_query coverage)
- Backfill idempotency — can _backfill_chunks_fts run twice safely?
- Error handling: any path where OperationalError escapes?
- Test coverage: does TestChunkBackfill actually simulate a pre-Phase-3 DB?
Report any critical or high-severity issues. Return a concise summary (under 300 words).
```

### - [ ] Step 4: Address any critical findings from the review

If code-reviewer flags critical issues, fix them inline with TDD (write failing test first, then fix, then re-run full suite). Minor/stylistic findings can be deferred.

### - [ ] Step 5: Final git commit (consolidated if multiple Phase 3 commits exist)

Verify the commit is NOT empty before recording in GCC memory:

```bash
git show --stat HEAD | tail -5
```

If empty, soft-reset and re-stage (same recovery pattern from C107). Otherwise proceed.

### - [ ] Step 6: Record Phase 3 completion in GCC memory

```python
gcc_commit(
    title="Complete Phase 3: chunk-level FTS5 + snippets on index.db",
    what="[5 sub-tasks summary: 3.1 DDL + triggers, 3.2 chunk_search, 3.3 snippet method, 3.4 backfill migration, 3.5 MCP mode='code', 3.6 validation]",
    why="Extends FTS5 coverage from index.db file metadata to actual code content, giving zero-dep BM25 code search with snippets — a grep replacement that works even without ONNX embeddings. Mirrors Phase 2 pattern on memory.db.",
    files_changed=[
        "ccr/context/index_db.py",
        "ccr/mcp/index_tools.py",
        "tests/unit/test_chunk_fts.py",
        "tests/unit/test_indexer.py",
        "docs/superpowers/plans/2026-04-19-phase3-chunk-fts5-index-db.md",
    ],
    next_step="Phase 4 candidates: global.db cross-project ACE wiring, or sqlite-vec vector search on memory.db embeddings column.",
    patterns_learned=[
        "Schema-version guard on per-DB user_version pragma: each SQLite file tracks its own migrations independently — memory.db at version 1, index.db at version 1, no conflict.",
        "For code search without embeddings, FTS5 on chunk.text with snippet(fts, -1, '[', ']', '...', 12) gives grep-quality snippets plus BM25 ranking in one query.",
        "When adding a new mode to an MCP tool validator (valid_modes set), add test that exercises the new mode end-to-end before wiring — catches mode-whitelist misses.",
    ],
)
```

### - [ ] Step 7: Final commit push (local branch only; do not push to origin unless user requests)

```bash
git log --oneline -10
```

Expected: clean linear history with the Phase 3 commits on top of `62843ac`.

---

## Validation Matrix

| Scenario | Expected | Test |
|---|---|---|
| FTS5 available → chunk_search | Uses MATCH, BM25 rank, returns hits | `TestChunkSearch::test_chunk_search_matches_content` |
| FTS5 available → snippet method | Returns `[match]` brackets + negative rank | `TestChunkSnippets::test_snippet_highlights_match` |
| FTS5 unavailable → snippet method | Returns `[]` gracefully | `TestChunkSnippets::test_snippet_returns_empty_when_fts5_unavailable` |
| Malformed FTS5 query | Falls back to LIKE; no crash | `TestChunkSearch::test_chunk_search_malformed_query_falls_back` |
| Pre-Phase-3 DB opened fresh | `user_version` bumped, chunks searchable | `TestChunkBackfill::test_backfill_on_upgrade` |
| Re-opening migrated DB | No-op backfill, still searchable | `TestChunkBackfill::test_backfill_idempotent` |
| MCP tool `mode="code"` | Returns `path:start-end` + snippet lines | `TestIndexSearchCodeMode::test_code_mode_returns_chunk_hits` |
| Triggers (AI/AD/BU/AU) | All 4 present after `_setup` | `TestChunkFtsSchema::test_all_four_chunk_triggers_created` |

---

## Risk Register

| Risk | Mitigation |
|---|---|
| FTS5 not compiled in SQLite build | `self._fts_available` gate — same probe used by Phase 2; all chunk FTS paths check it. |
| Large chunk text blows up FTS5 index | External-content table stores only inverted index, not duplicate text. Size overhead roughly 30-50% of `chunks.text` which is already small per row. |
| Backfill on huge existing DB is slow | FTS5 `'rebuild'` is single-pass streaming; on a 10k-file repo this is sub-second. Guarded by `user_version` so it only runs once per DB. |
| Unclosed-quote FTS query crashes MATCH | `_sanitize_fts_query` strips non-alnum chars before MATCH; plus `try/except OperationalError` on every MATCH call. |
| New `mode="code"` breaks existing callers | Added as additive enum value; existing `keyword/semantic/hybrid` modes unchanged. Default `mode="hybrid"` unchanged. |
| Trigger pattern differs from existing (`_FTS_TRIGGER_UPDATE` is a single AU combining delete+insert; new code uses separate BU + AU) | New pattern is strictly an improvement — BEFORE UPDATE runs before row change so old rowid is still valid. Both patterns work; we do NOT change the existing one. |

---

## Execution Plan

Sequential (each task depends on the previous):

1. **Task 3.1** — schema first, can't search without tables
2. **Task 3.2** — basic search, verifies triggers work end-to-end
3. **Task 3.3** — snippet method, reuses search query pattern
4. **Task 3.4** — backfill, requires methods from 3.2/3.3 for verification
5. **Task 3.5** — MCP wiring, requires all of 3.1–3.4
6. **Task 3.6** — validation + commit

**Parallelization opportunity:** Tasks 3.2 and 3.3 touch the same file (`index_db.py`) but independent methods — they *could* be dispatched in parallel to two subagents with clear file-region ownership, but given the small incremental scope, sequential TDD is simpler and safer. Default to sequential unless time pressure is explicit.

---

## Self-Review

**1. Spec coverage:**
- Goal (FTS5 on chunk content) → Tasks 3.1, 3.2 ✓
- Snippets → Task 3.3 ✓
- Migration for existing DBs → Task 3.4 ✓
- MCP tool integration → Task 3.5 ✓
- E2E validation → Task 3.6 ✓

**2. Placeholders:** None detected. Every step has exact code, exact commands, exact expected output.

**3. Type consistency:**
- `chunk_search` returns `list[dict]` with keys `file_path, chunk_idx, start_line, end_line, text, score` — consistent across Task 3.2 test, 3.2 impl, and 3.5 MCP wiring
- `chunk_search_with_snippet` returns same keys plus `snippet` and `rank` — consistent in Task 3.3 test + impl + 3.5 wiring
- `_fts_available` attribute name (not `fts_available`, which is the property) — accessed via `getattr(self, "_fts_available", False)` for safety in new methods; the public `fts_available` property already exists and is used by the MCP tool
- Schema version `1` on `index.db` (first migration for this DB) — distinct from `memory.db`'s version `1` (also first migration, on a different file)

Plan is self-consistent.
