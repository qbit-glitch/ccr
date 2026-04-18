# Phase 2: FTS5 Full-Text Search on memory.db — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite FTS5 full-text search to `memory.db` (commits, discussions, triples, patterns) so `gcc_search` and related tools get precision + ranking + snippets when FTS5 is available.

**Architecture:** Mirror the proven `sessions.db` pattern — external-content FTS5 virtual tables (`content='<source>', content_rowid='id'`) plus AFTER-INSERT / AFTER-DELETE / BEFORE-UPDATE / AFTER-UPDATE triggers. Graceful fallback to existing LIKE when FTS5 is unavailable at build time. File backend unchanged — keeps current substring search. Existing abstract methods (`commit_search_text`, `triple_search`) are upgraded in-place; no new public API.

**Tech Stack:** Python `sqlite3` stdlib (FTS5 module), `pytest` (+ `@pytest.fixture(params=[...])` for dual-backend coverage), thread-local WAL connection manager (already present).

---

## Scope & Affected Files

### Create
- `ccr/core/storage/_sqlite_fts5.py` — FTS5 DDL + trigger strings + helpers (availability probe, backfill, search)
- `tests/unit/test_fts5.py` — SQLite-only FTS5 tests (module-level skip when FTS5 missing)

### Modify
- `ccr/core/storage/sqlite_backend.py` — init FTS5 tables/triggers after base schema; set `self._fts_available` flag
- `ccr/core/storage/_sqlite_phase3a.py` — upgrade `commit_search_text` to use `MATCH` when FTS5 available, retain LIKE fallback
- `ccr/core/storage/_sqlite_phase3b.py` — upgrade `triple_search` + `pattern_load_all` (add `pattern_search_text` method)
- `ccr/core/storage/_sqlite_phase3c.py` — add FTS5 path to `discussion_list(search=...)`
- `ccr/core/storage/base.py` — add optional `discussion_search_text`, `pattern_search_text` abstract methods (with default LIKE impl in file backend)
- `ccr/core/storage/_file_phase3b.py` — add `pattern_search_text` stub (substring)
- `ccr/core/storage/_file_phase3c.py` — add `discussion_search_text` stub (substring; reuse existing filter)
- `ccr/mcp/gcc_search_tools.py` — pass through FTS5 snippets + rank when backend reports availability
- `tests/unit/test_wiring.py` — add dual-backend search assertions (both must return results; FTS5-specific rank order tested separately)

**Principle:** No change to tool signatures, no new abstract methods that file backend can't satisfy. FTS5 is an opportunistic SQLite-side upgrade.

---

## Task 2.1: FTS5 Helper Module

**Files:**
- Create: `ccr/core/storage/_sqlite_fts5.py` (~200 lines)
- Test: `tests/unit/test_fts5.py::TestFts5Availability`, `::TestCreateTables`, `::TestTriggers`

### - [ ] Step 1: Write the availability probe test first

```python
# tests/unit/test_fts5.py
"""Phase 2: FTS5 full-text search on memory.db — SQLite-only tests."""
import sqlite3
import pytest

from ccr.core.storage._sqlite_fts5 import (
    fts5_available,
    install_fts5,
    CREATE_COMMITS_FTS,
    CREATE_DISCUSSIONS_FTS,
    CREATE_TRIPLES_FTS,
    CREATE_PATTERNS_FTS,
)


def _has_fts5() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _has_fts5(), reason="SQLite FTS5 not available")


class TestFts5Availability:
    def test_probe_returns_true_on_fts5_capable_sqlite(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "t.db"))
        assert fts5_available(conn) is True
        conn.close()

    def test_probe_is_read_only(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "t.db"))
        fts5_available(conn)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()]
        assert tables == []
        conn.close()
```

### - [ ] Step 2: Run test, confirm it fails with ImportError

Run: `.venv/bin/python -m pytest tests/unit/test_fts5.py::TestFts5Availability -v`
Expected: ImportError — module doesn't exist yet.

### - [ ] Step 3: Implement the helper module

```python
# ccr/core/storage/_sqlite_fts5.py
"""FTS5 full-text search helpers for memory.db.

External-content FTS5 tables shadow the canonical tables (commits,
discussions, triples, patterns). Auto-synced via triggers. No data
duplication — FTS5 only stores its inverted index.

Graceful fallback: if the SQLite build lacks FTS5 compile flags, the
top-level ``install_fts5`` returns False and the caller must keep using
LIKE-based search.
"""
from __future__ import annotations
import sqlite3


# ---------- Virtual-table DDL -----------------------------------------------

CREATE_COMMITS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS commits_fts USING fts5(
    title, what, why, next_step, files_json,
    content='commits',
    content_rowid='rowid'
);
"""

CREATE_DISCUSSIONS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS discussions_fts USING fts5(
    topic, hypothesis, alternatives, decision, rationale, uncertainty,
    content='discussions',
    content_rowid='rowid'
);
"""

CREATE_TRIPLES_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS triples_fts USING fts5(
    subject, predicate, object,
    content='triples',
    content_rowid='id'
);
"""

CREATE_PATTERNS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS patterns_fts USING fts5(
    text,
    content='patterns',
    content_rowid='rowid'
);
"""

_ALL_TABLES = (
    CREATE_COMMITS_FTS, CREATE_DISCUSSIONS_FTS,
    CREATE_TRIPLES_FTS, CREATE_PATTERNS_FTS,
)


# ---------- Trigger DDL ------------------------------------------------------
# Pattern: AFTER INSERT, AFTER DELETE, BEFORE UPDATE (delete), AFTER UPDATE (insert)
# Mirrors session_store.turns_fts trigger family. One trigger per source table
# per event — 16 triggers total for 4 sources.

_TRIGGER_TEMPLATES = {
    "commits": {
        "cols": ("title", "what", "why", "next_step", "files_json"),
        "rowid": "rowid",
    },
    "discussions": {
        "cols": ("topic", "hypothesis", "alternatives", "decision", "rationale", "uncertainty"),
        "rowid": "rowid",
    },
    "triples": {
        "cols": ("subject", "predicate", "object"),
        "rowid": "id",
    },
    "patterns": {
        "cols": ("text",),
        "rowid": "rowid",
    },
}


def _trigger_sql(source: str) -> list[str]:
    spec = _TRIGGER_TEMPLATES[source]
    cols = ", ".join(spec["cols"])
    new_vals = ", ".join(f"new.{c}" for c in spec["cols"])
    old_vals = ", ".join(f"old.{c}" for c in spec["cols"])
    rowid = spec["rowid"]
    fts = f"{source}_fts"
    return [
        # AFTER INSERT
        f"""CREATE TRIGGER IF NOT EXISTS {source}_ai AFTER INSERT ON {source} BEGIN
            INSERT INTO {fts}(rowid, {cols}) VALUES (new.{rowid}, {new_vals});
        END;""",
        # AFTER DELETE
        f"""CREATE TRIGGER IF NOT EXISTS {source}_ad AFTER DELETE ON {source} BEGIN
            INSERT INTO {fts}({fts}, rowid, {cols}) VALUES ('delete', old.{rowid}, {old_vals});
        END;""",
        # BEFORE UPDATE — remove old row from FTS
        f"""CREATE TRIGGER IF NOT EXISTS {source}_bu BEFORE UPDATE ON {source} BEGIN
            INSERT INTO {fts}({fts}, rowid, {cols}) VALUES ('delete', old.{rowid}, {old_vals});
        END;""",
        # AFTER UPDATE — add new row
        f"""CREATE TRIGGER IF NOT EXISTS {source}_au AFTER UPDATE ON {source} BEGIN
            INSERT INTO {fts}(rowid, {cols}) VALUES (new.{rowid}, {new_vals});
        END;""",
    ]


# ---------- Public API -------------------------------------------------------

def fts5_available(conn: sqlite3.Connection) -> bool:
    """Return True iff SQLite build has FTS5 module compiled in.

    Side-effect-free: creates + drops a throwaway virtual table, rolling back
    so no schema change is committed.
    """
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE __fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def install_fts5(conn: sqlite3.Connection) -> bool:
    """Create FTS5 virtual tables + triggers. Idempotent.

    Returns True on success, False if FTS5 is not available.
    Raises sqlite3.OperationalError on errors other than missing FTS5.
    """
    if not fts5_available(conn):
        return False
    for ddl in _ALL_TABLES:
        conn.execute(ddl)
    for source in _TRIGGER_TEMPLATES:
        for trig in _trigger_sql(source):
            conn.execute(trig)
    conn.commit()
    return True


def backfill_fts5(conn: sqlite3.Connection) -> dict[str, int]:
    """Rebuild FTS5 indexes from the canonical tables.

    Safe to run on a fresh DB (no rows → no inserts). Uses the FTS5 'rebuild'
    command which re-reads from the content table in one pass — no triggers
    fire during rebuild.
    """
    counts = {}
    for source in _TRIGGER_TEMPLATES:
        fts = f"{source}_fts"
        # Check if FTS table exists (skip gracefully on non-FTS5 builds)
        got = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (fts,),
        ).fetchone()
        if got is None:
            continue
        conn.execute(f"INSERT INTO {fts}({fts}) VALUES ('rebuild')")
        n = conn.execute(f"SELECT count(*) FROM {fts}").fetchone()[0]
        counts[source] = n
    conn.commit()
    return counts
```

### - [ ] Step 4: Run tests — availability probe passes

Run: `.venv/bin/python -m pytest tests/unit/test_fts5.py::TestFts5Availability -v`
Expected: PASS (2 tests)

### - [ ] Step 5: Add table-creation + trigger tests

```python
class TestCreateTables:
    def test_install_creates_all_fts_tables(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        # Create canonical tables first (minimal schema)
        conn.executescript("""
            CREATE TABLE commits (rowid INTEGER PRIMARY KEY, title TEXT, what TEXT, why TEXT, next_step TEXT, files_json TEXT);
            CREATE TABLE discussions (rowid INTEGER PRIMARY KEY, topic TEXT, hypothesis TEXT, alternatives TEXT, decision TEXT, rationale TEXT, uncertainty TEXT);
            CREATE TABLE triples (id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT);
            CREATE TABLE patterns (rowid INTEGER PRIMARY KEY, text TEXT);
        """)
        assert install_fts5(conn) is True
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert {"commits_fts", "discussions_fts", "triples_fts", "patterns_fts"} <= tables
        conn.close()

    def test_install_is_idempotent(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        conn.executescript("""
            CREATE TABLE commits (rowid INTEGER PRIMARY KEY, title TEXT, what TEXT, why TEXT, next_step TEXT, files_json TEXT);
            CREATE TABLE discussions (rowid INTEGER PRIMARY KEY, topic TEXT, hypothesis TEXT, alternatives TEXT, decision TEXT, rationale TEXT, uncertainty TEXT);
            CREATE TABLE triples (id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT);
            CREATE TABLE patterns (rowid INTEGER PRIMARY KEY, text TEXT);
        """)
        assert install_fts5(conn) is True
        assert install_fts5(conn) is True  # Second call does not raise
        conn.close()


class TestTriggers:
    def test_insert_into_commits_propagates_to_fts(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "m.db"))
        conn.executescript("""
            CREATE TABLE commits (rowid INTEGER PRIMARY KEY, title TEXT, what TEXT, why TEXT, next_step TEXT, files_json TEXT);
            CREATE TABLE discussions (rowid INTEGER PRIMARY KEY, topic TEXT, hypothesis TEXT, alternatives TEXT, decision TEXT, rationale TEXT, uncertainty TEXT);
            CREATE TABLE triples (id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT);
            CREATE TABLE patterns (rowid INTEGER PRIMARY KEY, text TEXT);
        """)
        install_fts5(conn)
        conn.execute(
            "INSERT INTO commits (title, what, why, next_step, files_json) VALUES (?,?,?,?,?)",
            ("Add auth", "Added authentication", "security", "ship it", "[]"),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT rowid FROM commits_fts WHERE commits_fts MATCH 'authentication'"
        ).fetchall()
        assert len(rows) == 1
        conn.close()

    def test_delete_propagates(self, tmp_path):
        # ... same setup ...
        # INSERT + DELETE, then assert MATCH returns 0 rows
        pass  # expand in implementation
```

### - [ ] Step 6: Run tests, all green, commit

Run: `.venv/bin/python -m pytest tests/unit/test_fts5.py -v`
Expected: PASS all.

```bash
git add ccr/core/storage/_sqlite_fts5.py tests/unit/test_fts5.py
git commit -m "ccr: Phase 2.1 — FTS5 helper module + triggers"
```

---

## Task 2.2: Wire FTS5 into sqlite_backend

**Files:**
- Modify: `ccr/core/storage/sqlite_backend.py`
- Test: `tests/unit/test_fts5.py::TestBackendIntegration`

### - [ ] Step 1: Write integration test first

```python
class TestBackendIntegration:
    def test_backend_init_installs_fts5(self, tmp_path):
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))
        assert backend.fts_available is True
        # commits_fts table must exist
        conn = backend._conn_mgr.get_conn()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "commits_fts" in tables
        backend.close()
```

### - [ ] Step 2: Add FTS5 install call to SqliteStorageBackend

In `sqlite_backend.py` after the schema DDL block runs in `__init__`, add:

```python
# After base schema creation, opportunistically install FTS5.
from ._sqlite_fts5 import install_fts5
self._fts_available: bool = install_fts5(self._conn_mgr.get_conn())

@property
def fts_available(self) -> bool:
    return self._fts_available
```

### - [ ] Step 3: Run test, commit

```bash
git add ccr/core/storage/sqlite_backend.py tests/unit/test_fts5.py
git commit -m "ccr: Phase 2.2 — install FTS5 during sqlite_backend init"
```

---

## Task 2.3: Upgrade SQLite search methods to FTS5

**Files:**
- Modify: `ccr/core/storage/_sqlite_phase3a.py` — `commit_search_text`
- Modify: `ccr/core/storage/_sqlite_phase3b.py` — `triple_search`, add `pattern_search_text`
- Modify: `ccr/core/storage/_sqlite_phase3c.py` — add FTS5 path in `discussion_list` search branch
- Modify: `ccr/core/storage/base.py` — add `pattern_search_text`, `discussion_search_text` abstract methods
- Modify: `ccr/core/storage/_file_phase3b.py`, `_file_phase3c.py` — substring implementations
- Test: `tests/unit/test_fts5.py::TestSearchMethods`, `tests/unit/test_wiring.py` (add search cases)

### - [ ] Step 1: Write failing dual-backend test in test_wiring.py

```python
# tests/unit/test_wiring.py
class TestSearchWiring:
    def test_commit_search_text_returns_matches(self, mem):
        mem.commit("Auth", "Added authentication middleware", "security", ["auth.py"], "ship it")
        mem.commit("Logs", "Added structured logging", "observability", ["log.py"], "monitor")
        results = mem._storage.commit_search_text("main", "authentication", max_results=5)
        assert len(results) == 1
        assert "authentication" in results[0].get("what", "").lower()

    def test_discussion_search_returns_matches(self, mem):
        mem.add_discussion("cache strategy", "LRU vs TTL", "", "LRU", "simpler", "", "")
        results = mem._storage.discussion_list("main", search="cache")
        assert len(results) == 1
```

### - [ ] Step 2: Run test — fails for SQLite backend (LIKE works, but verify)

Run: `.venv/bin/python -m pytest tests/unit/test_wiring.py::TestSearchWiring -v`
Expected: currently passes with LIKE. New assertion: FTS5 branch used when `fts_available`.

### - [ ] Step 3: Update `commit_search_text` to prefer FTS5

```python
# ccr/core/storage/_sqlite_phase3a.py

def commit_search_text(self, branch: str, term: str, max_results: int = 5) -> list[dict]:
    conn = self.memory_conn
    # FTS5 path — fast + ranked
    if getattr(self, "_fts_available", False):
        try:
            rows = conn.execute(
                """SELECT c.* FROM commits c
                   JOIN commits_fts f ON f.rowid = c.rowid
                   WHERE c.branch = ? AND commits_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (branch, term, max_results),
            ).fetchall()
            if rows:
                return [self._row_to_commit_dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass  # Malformed FTS query → fall through to LIKE
    # LIKE fallback — preserves existing behaviour
    like_term = f"%{_escape_like(term)}%"
    rows = conn.execute(
        """SELECT * FROM commits WHERE branch = ?
           AND (title LIKE ? ESCAPE '\\' OR what LIKE ? ESCAPE '\\'
                OR why LIKE ? ESCAPE '\\' OR next_step LIKE ? ESCAPE '\\'
                OR files_json LIKE ? ESCAPE '\\')
           ORDER BY CAST(SUBSTR(id, 2) AS INTEGER) DESC
           LIMIT ?""",
        (branch, like_term, like_term, like_term, like_term, like_term, max_results),
    ).fetchall()
    return [self._row_to_commit_dict(r) for r in rows]
```

### - [ ] Step 4: Analogous updates to `triple_search` and `discussion_list`

Apply the same pattern: check `_fts_available`, try FTS5 MATCH, fall back to LIKE/substring on OperationalError.

### - [ ] Step 5: Add `pattern_search_text` — brand new method

```python
# ccr/core/storage/base.py (in abstract class)
def pattern_search_text(self, term: str, max_results: int = 10) -> list[dict]:
    """Find patterns matching `term`. FTS5 in SQLite, substring in files."""
    raise NotImplementedError
```

```python
# ccr/core/storage/_sqlite_phase3b.py
def pattern_search_text(self, term: str, max_results: int = 10) -> list[dict]:
    conn = self.memory_conn
    if getattr(self, "_fts_available", False):
        try:
            rows = conn.execute(
                """SELECT p.* FROM patterns p
                   JOIN patterns_fts f ON f.rowid = p.rowid
                   WHERE patterns_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (term, max_results),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass
    like = f"%{_escape_like(term)}%"
    rows = conn.execute(
        "SELECT * FROM patterns WHERE text LIKE ? ESCAPE '\\' LIMIT ?",
        (like, max_results),
    ).fetchall()
    return [dict(r) for r in rows]
```

```python
# ccr/core/storage/_file_phase3b.py
def pattern_search_text(self, term: str, max_results: int = 10) -> list[dict]:
    term_lower = term.lower()
    patterns = self.pattern_load_all().get("patterns", {})
    hits = [p for p in patterns.values() if term_lower in p.get("text", "").lower()]
    return hits[:max_results]
```

### - [ ] Step 6: Add `discussion_search_text` abstract + impls (same pattern)

### - [ ] Step 7: Run full dual-backend wiring tests

Run: `.venv/bin/python -m pytest tests/unit/test_wiring.py tests/unit/test_fts5.py -v`
Expected: all green.

### - [ ] Step 8: Commit

```bash
git add ccr/core/storage/_sqlite_phase3a.py ccr/core/storage/_sqlite_phase3b.py \
         ccr/core/storage/_sqlite_phase3c.py ccr/core/storage/base.py \
         ccr/core/storage/_file_phase3b.py ccr/core/storage/_file_phase3c.py \
         tests/unit/test_wiring.py tests/unit/test_fts5.py
git commit -m "ccr: Phase 2.3 — upgrade commit/triple/discussion/pattern search to FTS5 when available"
```

---

## Task 2.4: Backfill migration for existing data

**Files:**
- Modify: `ccr/core/storage/sqlite_backend.py` — call `backfill_fts5` after install on first run
- Test: `tests/unit/test_fts5.py::TestBackfill`

### - [ ] Step 1: Write test — data inserted via file backend, migrated, FTS5 rebuilt

```python
class TestBackfill:
    def test_backfill_after_migration_fills_commits_fts(self, tmp_path):
        # Simulate: existing SQLite DB with commits but no FTS5 index
        # (e.g. DB created before Phase 2 upgrade)
        import sqlite3
        db = tmp_path / ".ccr" / "memory.db"
        db.parent.mkdir()
        conn = sqlite3.connect(str(db))
        conn.executescript("""
            CREATE TABLE commits (rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                branch TEXT, id TEXT, title TEXT, what TEXT, why TEXT,
                next_step TEXT, files_json TEXT DEFAULT '[]');
            INSERT INTO commits (branch, id, title, what, why, next_step, files_json)
            VALUES ('main', 'C001', 'Auth', 'Added auth', 'security', 'ship', '[]');
        """)
        conn.commit()
        conn.close()

        # Now initialize Phase-2-enabled backend — backfill should run
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        backend = SqliteStorageBackend(str(tmp_path / ".ccr"))
        hits = backend.commit_search_text("main", "auth", max_results=5)
        assert len(hits) == 1
        backend.close()
```

### - [ ] Step 2: Add backfill guard (run once per DB version)

In `sqlite_backend.py`, after `install_fts5`:

```python
# Schema version guard — bump when FTS5 is rolled out.
# user_version 0 → pre-FTS5; 1 → FTS5 installed + backfilled.
if self._fts_available:
    conn = self._conn_mgr.get_conn()
    cur_version = self._conn_mgr.get_user_version()
    if cur_version < 1:
        from ._sqlite_fts5 import backfill_fts5
        backfill_fts5(conn)
        self._conn_mgr.set_user_version(1)
```

### - [ ] Step 3: Run tests, commit

```bash
git add ccr/core/storage/sqlite_backend.py tests/unit/test_fts5.py
git commit -m "ccr: Phase 2.4 — backfill FTS5 index on first upgrade"
```

---

## Task 2.5: Enhance gcc_search with FTS5 snippets + ranking

**Files:**
- Modify: `ccr/mcp/gcc_search_tools.py` (commit search branch around line 433)
- Test: `tests/unit/test_fts5.py::TestGccSearchEnhancement`

### - [ ] Step 1: Write test — search output includes snippet

```python
class TestGccSearchEnhancement:
    def test_search_returns_snippet_when_fts5_available(self, tmp_path):
        # Build project with SQLite backend, insert commits, run gcc_search
        # Verify result includes `**Snippet**:` line with [match] markers
        pass  # Full impl in step 3
```

### - [ ] Step 2: Add snippet-aware commit search helper to storage

```python
# ccr/core/storage/_sqlite_phase3a.py — ADD alongside commit_search_text
def commit_search_with_snippet(self, branch: str, term: str, max_results: int = 5) -> list[dict]:
    """FTS5-only search returning snippets. Returns [] if FTS5 unavailable."""
    if not getattr(self, "_fts_available", False):
        return []
    conn = self.memory_conn
    try:
        rows = conn.execute(
            """SELECT c.*,
                      snippet(commits_fts, -1, '[', ']', '...', 12) AS snippet_text,
                      rank
               FROM commits c
               JOIN commits_fts f ON f.rowid = c.rowid
               WHERE c.branch = ? AND commits_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (branch, term, max_results),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        d = self._row_to_commit_dict(r)
        d["snippet"] = r["snippet_text"]
        d["rank"] = r["rank"]
        out.append(d)
    return out
```

### - [ ] Step 3: Wire into gcc_search_tools.py

In the commits branch of `gcc_search`, prefer `commit_search_with_snippet` when backend is SQLite + FTS available. Emit snippet in markdown output:

```python
# Inside gcc_search commit loop
if getattr(mem._storage, "fts_available", False):
    results = mem._storage.commit_search_with_snippet(branch, query, max_results=limit)
    for r in results:
        lines.append(f"- **{r['id']}** {r['title']}")
        lines.append(f"  - **Snippet**: {r['snippet']}")
        lines.append(f"  - **Rank**: {r['rank']:.3f}")
else:
    # Existing 3-phase fallback unchanged
    ...
```

### - [ ] Step 4: Run tests, commit

```bash
git add ccr/core/storage/_sqlite_phase3a.py ccr/mcp/gcc_search_tools.py tests/unit/test_fts5.py
git commit -m "ccr: Phase 2.5 — gcc_search returns FTS5 snippets + rank when available"
```

---

## Task 2.6: E2E validation + full test suite

### - [ ] Step 1: Run full test suite

Run: `.venv/bin/python -m pytest tests/unit/ -x -q`
Expected: 2802 existing tests + new Phase 2 tests all pass. Target: 2820+ total.

### - [ ] Step 2: Backwards-compat smoke test

Run against existing `.ccr/` in repo root (real data from 107 commits):

```bash
.venv/bin/python -c "
from ccr.core.storage.sqlite_backend import SqliteStorageBackend
b = SqliteStorageBackend('.ccr')
print('fts_available:', b.fts_available)
print('fts5 hits for \"phase\":', len(b.commit_search_text('main', 'phase', 5)))
print('snippet hit:', b.commit_search_with_snippet('main', 'phase', 1))
b.close()
"
```

Expected: `fts_available: True`, hits > 0, snippet non-empty.

### - [ ] Step 3: Dispatch code-reviewer agent on Phase 2 diff

### - [ ] Step 4: Final git commit + gcc_commit

```bash
git add -A && git commit -m "ccr: Phase 2 — FTS5 full-text search on memory.db complete"
```

Then `gcc_commit(title="Complete Phase 2: FTS5 full-text search on memory.db", ...)`.

---

## Validation Matrix

| Scenario | Expected | Test |
|---|---|---|
| FTS5 available → commit search | Uses MATCH, returns ranked rows | `TestSearchMethods::test_commit_search_uses_fts5` |
| FTS5 unavailable → commit search | Falls back to LIKE | `TestSearchMethods::test_commit_search_falls_back_to_like` (mock) |
| Fresh DB → backfill runs once | `user_version = 1` set, FTS tables populated | `TestBackfill::test_backfill_sets_user_version` |
| Re-open DB → backfill skipped | user_version already 1, no-op | `TestBackfill::test_backfill_idempotent` |
| File backend → unchanged | Substring search works, no FTS code path | `test_wiring.py::TestSearchWiring` (params=['files','sqlite']) |
| Malformed FTS query | Graceful LIKE fallback, no crash | `TestSearchMethods::test_fts5_malformed_falls_back` |
| Snippet in gcc_search output | Markdown includes `**Snippet**:` | `TestGccSearchEnhancement::test_snippet_in_output` |

---

## Risk Register

| Risk | Mitigation |
|---|---|
| FTS5 not compiled in Python's bundled SQLite | `fts5_available()` probe + graceful fallback — `_fts_available` flag gates all FTS5 paths |
| Triggers silently drop data if schema mismatch | All FTS tables use `content='<source>', content_rowid='<id-col>'` explicitly; `backfill_fts5` can rebuild at any time |
| BEFORE UPDATE trigger fires before row change → old rowid still valid | Mirrors `session_store` pattern which works in production; tested at `TestTriggers::test_update_propagates` |
| Malformed user query (e.g., unclosed quote) crashes FTS5 MATCH | Wrapped in `try/except OperationalError` → falls back to LIKE |
| Large commits slow FTS5 insert trigger | External-content FTS5 stores only index, not content duplicate — negligible overhead |

---

## Execution plan

Two parallel-dispatch opportunities in otherwise-sequential execution:

1. **After 2.3** completes, dispatch 2.4 (backfill) and 2.5 (gcc_search enhancement) as parallel agents — different files, no interference.
2. **Final validation**: dispatch code-reviewer + full-test-suite-runner in parallel.

Tasks 2.1 → 2.2 → 2.3 are strictly sequential (schema must exist before wire-in, wire-in must exist before enhanced search).
