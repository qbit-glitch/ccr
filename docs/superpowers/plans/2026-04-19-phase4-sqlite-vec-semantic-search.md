# Phase 4: Unified sqlite-vec Semantic Search on memory.db — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move commit embeddings from the side-car `.ccr/embeddings.db` + gzipped JSON to a `commits_vec` vec0 virtual table on `memory.db`, and expose a native KNN path through `gcc_search(mode=semantic)` — eliminating the Python-side `O(N)` cosine scan in `_search_commits`.

**Architecture:** Load `sqlite-vec` via `enable_load_extension` inside `SqliteConnectionManager._create_connection` (graceful fallback when extension or build is missing). Create one vec0 virtual table on `memory.db` holding `(id TEXT PRIMARY KEY, embedding float[384])`. Upsert each new commit's vector inside `commit_insert`'s transaction. Expose `commit_semantic_search(query_vec, branch, top_k)` that issues a single `embedding MATCH ? AND k = ?` query joined back to `commits` for raw_block+title. Wire `memory_context._search_commits` to prefer the backend KNN when available; fall back to the existing ONNX-cosine-in-Python chain when not. One-shot migration backfills legacy `.ccr/embeddings.db` and `.ccr/commit_embeddings.json.gz` into `commits_vec` behind a `PRAGMA user_version 1 → 2` guard. Same per-DB migration pattern proven in Phase 2.4 (memory.db 0→1) and Phase 3.4 (index.db 0→1).

**Tech Stack:** Python `sqlite3` stdlib with `conn.enable_load_extension(True)`, `sqlite-vec` (already in `ccr[vector]` optional deps — confirmed at `.venv/lib/python3.14/site-packages/sqlite_vec/__init__.py`), ONNX all-MiniLM-L6-v2 via existing `ccr.context.embeddings.get_embedding_model()`, `pytest` with module-level `skipif(not SQLITE_VEC_AVAILABLE)`.

---

## Scope & Affected Files

### Create
- `ccr/core/storage/_sqlite_vec.py` — vec0 DDL helpers + `install_vec(conn)` + `backfill_vec(conn, ccr_root)` (mirrors `_sqlite_fts5.py` layout)
- `tests/unit/test_commits_vec.py` — dedicated tests (module-level skip when sqlite-vec missing)

### Modify
- `ccr/core/storage/sqlite_backend.py`:
  - Import `install_vec`, `backfill_vec` from `_sqlite_vec`
  - Extend `SqliteConnectionManager._create_connection` to load the sqlite-vec extension (graceful fallback)
  - Add `self._vec_available: bool = install_vec(self.memory_conn)` after the FTS5 install call in `__init__`
  - Add `vec_available` `@property` on `SqliteStorageBackend` returning `self._vec_available`
  - Extend `user_version` migration block: if `vec_available and current_version < 2`, run `backfill_vec` then `set_user_version(2)`
- `ccr/core/storage/base.py`:
  - Add abstract `vec_available` `@property` → `bool`
  - Add abstract `commit_upsert_vector(self, commit_id: str, vector: list[float]) -> None`
  - Add abstract `commit_semantic_search(self, branch: str, query_vec: list[float], top_k: int) -> list[dict]`
- `ccr/core/storage/file_backend.py`:
  - Implement `vec_available = False` property (stub)
  - Implement `commit_upsert_vector` as a no-op
  - Implement `commit_semantic_search` returning `[]`
- `ccr/core/storage/_sqlite_phase3a.py`:
  - Add `commit_upsert_vector(commit_id, vector)` — `INSERT OR REPLACE` into `commits_vec`
  - Add `commit_semantic_search(branch, query_vec, top_k)` — vec0 MATCH + JOIN back to `commits`
- `ccr/core/memory_pkg/memory_commit.py`:
  - After `_embed_commit` returns a vec, call `self._storage.commit_upsert_vector(commit_id, vec.tolist())` when backend supports it
- `ccr/core/memory_pkg/memory_context.py`:
  - In `_search_commits`, replace the ONNX-cosine-in-Python block with a call to `self._storage.commit_semantic_search(branch, query_vec, remaining)` when `getattr(self._storage, "vec_available", False)` is True; keep the existing block as fallback

**Non-goals for this phase:**
- No changes to `.ccr/embeddings.db` schema — the legacy store remains read-only during backfill; after migration it is orphaned (NOT deleted — safe rollback)
- No MCP tool signature changes — `gcc_search(mode=semantic)` keeps its existing `GccSearchResult` shape
- No embedding-model upgrade — stays on all-MiniLM-L6-v2 (384 dim L2-normalized)
- No cross-project ACE wiring (global.db) — that is a candidate for Phase 5

**Principle:** KNN via vec0 replaces a per-call `O(N) @` numpy matmul with an indexed query. The migration is additive and idempotent: commits still get a gzip fallback entry if sqlite-vec is missing at install time, so downgrades never lose data. Schema version on `memory.db` bumps from 1 (Phase 2 FTS5) to 2 (Phase 4 vec). `index.db` `user_version` is untouched (stays at 1).

---

## Task 4.1: commits_vec DDL + sqlite-vec loader in SqliteConnectionManager

**Files:**
- Create: `ccr/core/storage/_sqlite_vec.py`
- Create: `tests/unit/test_commits_vec.py`
- Modify: `ccr/core/storage/sqlite_backend.py:333-339` (extension loader in `_create_connection`)
- Modify: `ccr/core/storage/sqlite_backend.py:384-395` (`__init__` install + migration block)
- Modify: `ccr/core/storage/base.py` (add `vec_available` abstract property)

### - [ ] Step 1: Write the failing availability + DDL test

Create `tests/unit/test_commits_vec.py`:

```python
"""Phase 4: commits_vec vec0 virtual table + sqlite-vec extension loading."""
from __future__ import annotations

import os

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
        # vec0 reports virtual-table definition via sqlite_master.sql
        sql = backend.memory_conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='commits_vec'"
        ).fetchone()[0]
        assert "float[384]" in sql
        backend.close()
```

### - [ ] Step 2: Run test — expect missing `vec_available` property

Run: `.venv/bin/python -m pytest tests/unit/test_commits_vec.py::TestCommitsVecSchema -v`
Expected: FAIL with `AttributeError: 'SqliteStorageBackend' object has no attribute 'vec_available'`

### - [ ] Step 3: Create `_sqlite_vec.py` helper module

Create `ccr/core/storage/_sqlite_vec.py`:

```python
"""sqlite-vec virtual-table helpers for ``memory.db``.

Mirrors the ``_sqlite_fts5.py`` layout: a probe, an ``install_vec`` that creates
the ``commits_vec`` vec0 virtual table, and a ``backfill_vec`` that imports
legacy vectors from ``.ccr/embeddings.db`` and ``.ccr/commit_embeddings.json.gz``.

Graceful fallback: if the Python process cannot load the extension (extension
missing, build without ``enable_load_extension``, or platform restriction),
``install_vec`` returns ``False`` and callers must keep using the
ONNX-cosine-in-Python fallback.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import struct
from typing import Dict

logger = logging.getLogger(__name__)

VEC_DIM = 384

# ---------- Virtual-table DDL ----------------------------------------------

CREATE_COMMITS_VEC = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS commits_vec"
    f" USING vec0(id TEXT PRIMARY KEY, embedding float[{VEC_DIM}])"
)


# ---------- Public API ------------------------------------------------------


def _serialize(vec) -> bytes:
    """Serialize an iterable of floats into the byte layout vec0 expects."""
    lst = list(vec)
    if len(lst) != VEC_DIM:
        raise ValueError(f"Vector dim {len(lst)} != expected {VEC_DIM}")
    return struct.pack(f"{VEC_DIM}f", *lst)


def vec_available(conn: sqlite3.Connection) -> bool:
    """Return True iff the sqlite-vec extension is already loaded on ``conn``.

    Side-effect-free: tries to CREATE+DROP a throwaway vec0 table.
    """
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS __vec_probe USING vec0(id TEXT PRIMARY KEY, embedding float[{VEC_DIM}])"
        )
        conn.execute("DROP TABLE __vec_probe")
        return True
    except sqlite3.OperationalError:
        return False


def install_vec(conn: sqlite3.Connection) -> bool:
    """Create ``commits_vec`` virtual table. Idempotent. Returns False if sqlite-vec missing."""
    if not vec_available(conn):
        return False
    conn.execute(CREATE_COMMITS_VEC)
    conn.commit()
    return True


def backfill_vec(conn: sqlite3.Connection, ccr_root: str) -> Dict[str, int]:
    """One-shot import of legacy vectors into ``commits_vec``.

    Sources (in order, deduped by commit_id):
      1. ``.ccr/embeddings.db`` — legacy sqlite-vec side-car store
      2. ``.ccr/commit_embeddings.json.gz`` — legacy gzip JSON fallback

    Returns: {"sqlite_vec": N, "gzip_json": M, "total": N+M}. Sources that
    don't exist are skipped gracefully.
    """
    counts = {"sqlite_vec": 0, "gzip_json": 0, "total": 0}
    seen: set[str] = set()

    # Source 1: legacy sqlite-vec side-car
    legacy_db = os.path.join(ccr_root, "embeddings.db")
    if os.path.exists(legacy_db):
        try:
            import sqlite_vec  # soft dep
            src = sqlite3.connect(legacy_db)
            src.enable_load_extension(True)
            sqlite_vec.load(src)
            src.enable_load_extension(False)
            rows = src.execute(
                "SELECT id, embedding FROM vec_embeddings"
            ).fetchall()
            for cid, blob in rows:
                if cid in seen:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO commits_vec (id, embedding) VALUES (?, ?)",
                    (cid, blob),
                )
                seen.add(cid)
                counts["sqlite_vec"] += 1
            src.close()
        except Exception as exc:
            logger.warning("backfill_vec: legacy embeddings.db import failed: %s", exc)

    # Source 2: legacy gzip JSON
    legacy_json = os.path.join(ccr_root, "commit_embeddings.json.gz")
    if os.path.exists(legacy_json):
        try:
            from ccr.context.embeddings import load_embeddings
            cache = load_embeddings(legacy_json)
            for cid, vec_list in cache.items():
                if cid in seen:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO commits_vec (id, embedding) VALUES (?, ?)",
                    (cid, _serialize(vec_list)),
                )
                seen.add(cid)
                counts["gzip_json"] += 1
        except Exception as exc:
            logger.warning("backfill_vec: gzip JSON import failed: %s", exc)

    conn.commit()
    counts["total"] = counts["sqlite_vec"] + counts["gzip_json"]
    return counts
```

### - [ ] Step 4: Wire extension loading into `SqliteConnectionManager._create_connection`

Edit `ccr/core/storage/sqlite_backend.py`. Replace lines 333-339 (the `_create_connection` method body) with:

```python
    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        # Load sqlite-vec extension if available. Silent failure is correct —
        # SqliteStorageBackend probes vec_available() and falls back cleanly.
        try:
            import sqlite_vec  # soft dep
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            pass
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
```

### - [ ] Step 5: Add `install_vec` call + `vec_available` flag in backend `__init__`

Edit `ccr/core/storage/sqlite_backend.py`. Add import at top with the other `_sqlite_*` imports:

```python
from ccr.core.storage._sqlite_vec import install_vec
```

Then locate the block at lines 384-395 (FTS5 install + backfill). Append after it — before `def _ensure_phase1_tables`:

```python
        # Phase 4: install commits_vec virtual table (graceful when sqlite-vec missing)
        self._vec_available: bool = install_vec(self.memory_conn)

        # Phase 4 migration: backfill legacy embeddings into commits_vec.
        # user_version 1 → 2 guard (1 = Phase 2 FTS5 backfill complete).
        if self._vec_available:
            current_version = self._memory_mgr.get_user_version()
            if current_version < 2:
                from ._sqlite_vec import backfill_vec
                backfill_vec(self.memory_conn, ccr_root)
                self._memory_mgr.set_user_version(2)
```

### - [ ] Step 6: Add `vec_available` property on backend + abstract method on base

Edit `ccr/core/storage/sqlite_backend.py`. After `fts_available` property (at ~line 428), add:

```python
    @property
    def vec_available(self) -> bool:
        """True iff sqlite-vec extension loaded and commits_vec table created."""
        return self._vec_available
```

Edit `ccr/core/storage/base.py`. Locate the `fts_available` abstract property and add right after it:

```python
    @property
    @abstractmethod
    def vec_available(self) -> bool:
        """True iff the backend supports vector KNN search on commits."""
```

Edit `ccr/core/storage/file_backend.py` (at ~line 45 where `fts_available` is defined). Add after it:

```python
    @property
    def vec_available(self) -> bool:
        return False
```

### - [ ] Step 7: Run tests — expect all three passing

Run: `.venv/bin/python -m pytest tests/unit/test_commits_vec.py::TestCommitsVecSchema -v`
Expected: PASS (3/3)

### - [ ] Step 8: Run the full wiring test matrix to confirm no regression

Run: `.venv/bin/python -m pytest tests/unit/test_wiring.py tests/unit/test_fts5.py tests/unit/test_storage_backend.py -x -q`
Expected: PASS (no regressions in the dual-backend matrix)

### - [ ] Step 9: Commit

```bash
git add ccr/core/storage/_sqlite_vec.py \
        ccr/core/storage/base.py \
        ccr/core/storage/file_backend.py \
        ccr/core/storage/sqlite_backend.py \
        tests/unit/test_commits_vec.py
git commit -m "ccr: Phase 4.1 — commits_vec DDL + sqlite-vec extension loader"
git show --stat HEAD | tail -5
```

Expected: non-empty commit showing 5 files changed. If the stat output is empty, soft-reset to prior SHA and re-stage explicitly (third-occurrence empty-commit pattern documented in C110).

---

## Task 4.2: Upsert vectors on commit_insert

**Files:**
- Modify: `ccr/core/storage/base.py` (abstract method)
- Modify: `ccr/core/storage/file_backend.py` (no-op stub)
- Modify: `ccr/core/storage/_sqlite_phase3a.py` (implementation)
- Modify: `ccr/core/memory_pkg/memory_commit.py:251` (wire from the embed call site)
- Modify: `tests/unit/test_commits_vec.py` (new TestVectorUpsert class)

### - [ ] Step 1: Write the failing upsert test

Append to `tests/unit/test_commits_vec.py`:

```python
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
```

### - [ ] Step 2: Run — expect `AttributeError` on `commit_upsert_vector`

Run: `.venv/bin/python -m pytest tests/unit/test_commits_vec.py::TestVectorUpsert -v`
Expected: FAIL

### - [ ] Step 3: Add abstract method on `base.py`

Edit `ccr/core/storage/base.py`. In the commit-methods block, add:

```python
    @abstractmethod
    def commit_upsert_vector(self, commit_id: str, vector: list[float]) -> None:
        """Insert or replace the embedding for a commit (no-op on file backend)."""
```

### - [ ] Step 4: Add no-op stub on `file_backend.py`

Edit `ccr/core/storage/file_backend.py`. In the `FileStorageBackend` class body add:

```python
    def commit_upsert_vector(self, commit_id: str, vector: list[float]) -> None:
        # File backend keeps using gzip JSON via MemoryManager._embed_commit path.
        return None
```

### - [ ] Step 5: Add the SQLite implementation

Edit `ccr/core/storage/_sqlite_phase3a.py`. Add import at top:

```python
from ccr.core.storage._sqlite_vec import _serialize as _serialize_vec
```

Add method inside `Phase3aMixin` (after `commit_update`, before `commit_search_text`):

```python
    def commit_upsert_vector(self, commit_id: str, vector: list[float]) -> None:
        """Insert or replace a 384-dim L2-normalized embedding for a commit.

        Silently no-ops when sqlite-vec is unavailable so the caller can remain
        backend-agnostic. Dimension mismatches still raise ValueError.
        """
        if not getattr(self, "_vec_available", False):
            return
        conn = self.memory_conn
        blob = _serialize_vec(vector)  # raises ValueError on wrong dim
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO commits_vec (id, embedding) VALUES (?, ?)",
                (commit_id, blob),
            )
```

### - [ ] Step 6: Wire from the commit-embedding call site

Edit `ccr/core/memory_pkg/memory_commit.py`. Locate lines 249-257 (the `_embed_commit` + `_compute_links` block). Replace with:

```python
        commit_links: list = []
        try:
            new_vec = self._embed_commit(
                commit_id, f"{title} {what} {why} {next_step}"
            )
            # Phase 4: upsert into memory.db commits_vec when backend supports it.
            if new_vec is not None:
                try:
                    self._storage.commit_upsert_vector(commit_id, new_vec.tolist())
                except Exception:
                    pass  # vector indexing is supplementary
            commit_links = self._compute_links(
                branch, commit_id, title, what, why, files_changed, next_step,
                new_vec=new_vec,
            )
            if commit_links:
                self._update_links(commit_id, commit_links)
        except Exception:
            pass  # Linking is supplementary — never fail the commit
```

### - [ ] Step 7: Run upsert tests — expect 3/3 passing

Run: `.venv/bin/python -m pytest tests/unit/test_commits_vec.py::TestVectorUpsert -v`
Expected: PASS (3/3)

### - [ ] Step 8: Run commit-flow integration

Run: `.venv/bin/python -m pytest tests/unit/test_memory.py -k "commit" -q`
Expected: PASS (no regressions in the 80+ commit-flow tests)

### - [ ] Step 9: Commit

```bash
git add ccr/core/storage/base.py \
        ccr/core/storage/file_backend.py \
        ccr/core/storage/_sqlite_phase3a.py \
        ccr/core/memory_pkg/memory_commit.py \
        tests/unit/test_commits_vec.py
git commit -m "ccr: Phase 4.2 — upsert commit vectors into commits_vec"
git show --stat HEAD | tail -5
```

Expected: non-empty commit (5 files). Empty-commit check per Phase 3 pattern.

---

## Task 4.3: commit_semantic_search KNN method

**Files:**
- Modify: `ccr/core/storage/base.py` (abstract method)
- Modify: `ccr/core/storage/file_backend.py` (stub returns `[]`)
- Modify: `ccr/core/storage/_sqlite_phase3a.py` (implementation)
- Modify: `tests/unit/test_commits_vec.py` (new TestSemanticSearch class)

### - [ ] Step 1: Write the failing KNN search test

Append to `tests/unit/test_commits_vec.py`:

```python
class TestSemanticSearch:
    def test_knn_returns_closest_commit(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        # Insert two commits with distinct vectors into the real commits table
        # so JOIN-back-to-commits can hydrate raw_block + title.
        from ccr.core.storage._sqlite_utils import _utcnow
        for cid, title in (("C001", "auth bug"), ("C002", "logging refactor")):
            backend.memory_conn.execute(
                """INSERT INTO commits (id, branch, timestamp, title, what, why,
                   files_json, next_step, patterns_json, score, author, ci_json,
                   experiment_json, ota_trace, raw_block, created_at)
                   VALUES (?, 'main', ?, ?, ?, '', '[]', '', NULL, NULL, '',
                           NULL, NULL, NULL, ?, ?)""",
                (cid, _utcnow(), title, title, f"## [{cid}] {title}", _utcnow()),
            )
            backend.memory_conn.commit()
        # C001 ≈ [1, 0, 0, …], C002 ≈ [0, 1, 0, …]
        v1 = [1.0] + [0.0] * 383
        v2 = [0.0, 1.0] + [0.0] * 382
        backend.commit_upsert_vector("C001", v1)
        backend.commit_upsert_vector("C002", v2)

        # Query vector aligned with C001
        q = [1.0] + [0.0] * 383
        hits = backend.commit_semantic_search("main", q, top_k=2)
        assert len(hits) >= 1
        assert hits[0]["id"] == "C001"
        assert "title" in hits[0]
        assert hits[0]["distance"] <= hits[-1]["distance"]  # sorted ascending
        backend.close()

    def test_semantic_search_empty_when_no_vectors(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        assert backend.commit_semantic_search("main", [0.1] * 384, top_k=5) == []
        backend.close()

    def test_semantic_search_filters_by_branch(self, tmp_path):
        backend = SqliteStorageBackend(str(tmp_path / "ccr"))
        from ccr.core.storage._sqlite_utils import _utcnow
        # Same-vector commits on different branches
        for cid, branch in (("C100", "main"), ("C101", "experiment")):
            backend.memory_conn.execute(
                """INSERT INTO commits (id, branch, timestamp, title, what, why,
                   files_json, next_step, patterns_json, score, author, ci_json,
                   experiment_json, ota_trace, raw_block, created_at)
                   VALUES (?, ?, ?, 't', 'w', 'y', '[]', '', NULL, NULL, '',
                           NULL, NULL, NULL, 'rb', ?)""",
                (cid, branch, _utcnow(), _utcnow()),
            )
        backend.memory_conn.commit()
        backend.commit_upsert_vector("C100", [1.0] + [0.0] * 383)
        backend.commit_upsert_vector("C101", [1.0] + [0.0] * 383)
        hits = backend.commit_semantic_search("main", [1.0] + [0.0] * 383, top_k=5)
        assert {h["id"] for h in hits} == {"C100"}
        backend.close()
```

### - [ ] Step 2: Run — expect `AttributeError: commit_semantic_search`

Run: `.venv/bin/python -m pytest tests/unit/test_commits_vec.py::TestSemanticSearch -v`
Expected: FAIL

### - [ ] Step 3: Add abstract method on `base.py`

Edit `ccr/core/storage/base.py`. Add alongside `commit_upsert_vector`:

```python
    @abstractmethod
    def commit_semantic_search(
        self, branch: str, query_vec: list[float], top_k: int,
    ) -> list[dict]:
        """Return top-k nearest commit dicts (id, title, raw_block, distance).

        Returns [] when no vectors indexed or backend lacks KNN support.
        """
```

### - [ ] Step 4: Stub on `file_backend.py`

Edit `ccr/core/storage/file_backend.py`. Add:

```python
    def commit_semantic_search(
        self, branch: str, query_vec: list[float], top_k: int,
    ) -> list[dict]:
        return []
```

### - [ ] Step 5: SQLite implementation

Edit `ccr/core/storage/_sqlite_phase3a.py`. Add after `commit_upsert_vector`:

```python
    def commit_semantic_search(
        self, branch: str, query_vec: list[float], top_k: int,
    ) -> list[dict]:
        """vec0 KNN joined back to commits, filtered by branch.

        Over-fetches 3× to tolerate branch filtering in the wrapping SELECT.
        Returns list of dicts with keys: id, title, raw_block, distance.
        """
        if not getattr(self, "_vec_available", False):
            return []
        if top_k <= 0:
            return []
        blob = _serialize_vec(query_vec)
        fetch_k = max(top_k * 3, top_k)
        conn = self.memory_conn
        try:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.raw_block, v.distance
                FROM commits_vec v
                JOIN commits c ON c.id = v.id
                WHERE v.embedding MATCH ? AND k = ? AND c.branch = ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (blob, fetch_k, branch, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # vec0 MATCH unavailable at runtime
        return [
            {
                "id": r["id"],
                "title": r["title"] or "",
                "raw_block": r["raw_block"] or "",
                "distance": float(r["distance"]),
            }
            for r in rows
        ]
```

### - [ ] Step 6: Run KNN tests

Run: `.venv/bin/python -m pytest tests/unit/test_commits_vec.py::TestSemanticSearch -v`
Expected: PASS (3/3)

### - [ ] Step 7: Commit

```bash
git add ccr/core/storage/base.py \
        ccr/core/storage/file_backend.py \
        ccr/core/storage/_sqlite_phase3a.py \
        tests/unit/test_commits_vec.py
git commit -m "ccr: Phase 4.3 — commit_semantic_search KNN method"
git show --stat HEAD | tail -5
```

Expected: non-empty commit (4 files).

---

## Task 4.4: Legacy embedding backfill migration

**Files:**
- Modify: `tests/unit/test_commits_vec.py` (new TestBackfill class)
- No new code — the migration block and `backfill_vec` helper already exist from Task 4.1. This task verifies correctness end-to-end.

### - [ ] Step 1: Write the failing gzip-JSON backfill test

Append to `tests/unit/test_commits_vec.py`:

```python
class TestBackfill:
    def test_gzip_json_source_imported_on_upgrade(self, tmp_path):
        """Pre-Phase-4 DB with commit_embeddings.json.gz gets backfilled."""
        import gzip
        import json

        ccr_root = tmp_path / "ccr"
        ccr_root.mkdir()
        # Seed a legacy gzip cache with two commit embeddings
        legacy_vecs = {"C001": [0.5] * 384, "C002": [0.1] * 384}
        gz_path = ccr_root / "commit_embeddings.json.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(legacy_vecs, f)

        # Create memory.db at user_version=1 (Phase 2 done, Phase 4 pending)
        import sqlite3
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
```

### - [ ] Step 2: Run backfill tests

Run: `.venv/bin/python -m pytest tests/unit/test_commits_vec.py::TestBackfill -v`
Expected: PASS (3/3) — the migration block from Task 4.1 already performs the backfill

### - [ ] Step 3: Smoke test on the real `.ccr/` directory

Run:
```bash
.venv/bin/python -c "
from ccr.core.storage.sqlite_backend import SqliteStorageBackend
b = SqliteStorageBackend('.ccr')
print('vec_available:', b.vec_available)
print('user_version:', b._memory_mgr.get_user_version())
n = b.memory_conn.execute('SELECT COUNT(*) FROM commits_vec').fetchone()[0]
print('commits_vec rows:', n)
b.close()
"
```

Expected: `vec_available: True`, `user_version: 2`, and `commits_vec rows:` greater than 100 (matches the current 110 commits).

### - [ ] Step 4: Commit

```bash
git add tests/unit/test_commits_vec.py
git commit -m "ccr: Phase 4.4 — legacy embedding backfill tests (migration guard verification)"
git show --stat HEAD | tail -5
```

Expected: non-empty commit (1 file).

---

## Task 4.5: Wire gcc_search semantic path to KNN

**Files:**
- Modify: `ccr/core/memory_pkg/memory_context.py:329-387` (`_search_commits`)
- Modify: `tests/unit/test_commits_vec.py` (new TestSemanticWiring class)

### - [ ] Step 1: Write the failing integration test

Append to `tests/unit/test_commits_vec.py`:

```python
class TestSemanticWiring:
    def test_memory_search_uses_backend_knn_when_vec_available(self, tmp_path):
        """MemoryManager._search_commits prefers backend KNN path."""
        from ccr.core.memory import MemoryManager

        mem = MemoryManager(ccr_root=str(tmp_path / "ccr"))
        # Seed a real commit through the public API so triggers + vector upsert fire
        mem.commit(
            title="authentication refactor",
            what="rewrote login flow",
            why="race condition in session setup",
            files_changed=["src/auth.py"],
            next_step="add CSRF test",
        )
        # ONNX might not be installed in test env — patch out embed_query to
        # return a deterministic vector aligned with the stored commit.
        import numpy as np
        stored = mem._load_commit_embeddings(["C001"]).get("C001")
        if stored is None:
            pytest.skip("ONNX model not available in test env")

        from ccr.context.embeddings import get_embedding_model
        model = get_embedding_model()
        assert model is not None

        result = mem._search_commits("main", "authentication")
        assert "C001" in result or "authentication" in result.lower()
```

### - [ ] Step 2: Run — expect PASS via existing ONNX path OR skip when ONNX missing

Run: `.venv/bin/python -m pytest tests/unit/test_commits_vec.py::TestSemanticWiring -v`
Expected: PASS or skip. If PASS, the existing ONNX-cosine path already works; this test becomes a regression guard for Step 3.

### - [ ] Step 3: Rewrite the semantic branch in `_search_commits`

Edit `ccr/core/memory_pkg/memory_context.py`. Locate lines 345-367 (the `remaining > 0 and model is not None` block). Replace with:

```python
        if remaining > 0:
            model = get_embedding_model()
            if model is not None:
                try:
                    import numpy as np

                    query_vec = model.embed_query(term)

                    # Phase 4: prefer backend KNN when sqlite-vec is wired
                    if getattr(self._storage, "vec_available", False):
                        hits = self._storage.commit_semantic_search(
                            branch, query_vec.tolist(), remaining + len(exact_ids),
                        )
                        for h in hits:
                            cid = h["id"]
                            if cid in exact_ids:
                                continue
                            block = h.get("raw_block", "").strip()
                            if block:
                                semantic_matches.append(block)
                            if len(semantic_matches) >= remaining:
                                break
                    else:
                        # Legacy path: ONNX-cosine-in-Python over gzip JSON cache
                        all_embeddings = self._load_all_commit_embeddings()
                        if all_embeddings:
                            ids = list(all_embeddings.keys())
                            vecs = np.stack([all_embeddings[cid] for cid in ids])
                            scores = vecs @ query_vec
                            ranked = sorted(zip(ids, scores), key=lambda x: -x[1])
                            for cid, score in ranked:
                                if score < 0.3 or cid in exact_ids:
                                    continue
                                block = self._find_commit_by_id(branch, cid)
                                if block:
                                    semantic_matches.append(block.strip())
                                if len(semantic_matches) >= remaining:
                                    break
                except Exception:
                    pass
```

### - [ ] Step 4: Re-run the wiring test

Run: `.venv/bin/python -m pytest tests/unit/test_commits_vec.py::TestSemanticWiring -v`
Expected: PASS (or skip when ONNX missing)

### - [ ] Step 5: Run the broader search-integration suite

Run: `.venv/bin/python -m pytest tests/unit/test_memory.py tests/unit/test_gcc_search.py -k "search" -q`
Expected: PASS — no regression; semantic hits still appear for keyword-less queries.

### - [ ] Step 6: Commit

```bash
git add ccr/core/memory_pkg/memory_context.py \
        tests/unit/test_commits_vec.py
git commit -m "ccr: Phase 4.5 — wire _search_commits semantic path to backend KNN"
git show --stat HEAD | tail -5
```

Expected: non-empty commit (2 files).

---

## Task 4.6: E2E validation + code review + milestone commit

**Files:** no new code — validation, review, and milestone commit only.

### - [ ] Step 1: Full suite

Run: `.venv/bin/python -m pytest tests/unit/ tests/integration/ -x -q`
Expected: ~2840+ passed, 2 skipped, 0 failed (prior baseline 2839 + ~12 new Phase 4 tests).

### - [ ] Step 2: Real-data smoke test on live `.ccr/memory.db`

Run:
```bash
.venv/bin/python -c "
from ccr.core.memory import MemoryManager
import numpy as np
mem = MemoryManager(ccr_root='.ccr')
# Assumes Phase 4 migrated current 110 commits into commits_vec
result = mem._search_commits('main', 'FTS5 snippet BM25')
print('Semantic hit length:', len(result))
print(result[:500] if result else '(empty)')
"
```
Expected: non-empty result mentioning a Phase 2/3 FTS5-related commit (C108, C109, or C110).

### - [ ] Step 3: Dispatch code-reviewer agent on the Phase 4 diff

Use the `Agent` tool with `subagent_type=code-reviewer` and this prompt:

> Review the full Phase 4 diff (`git diff <phase-3-tip>..HEAD`) for sqlite-vec commits_vec integration. Focus areas: (1) connection-level extension loading — is `enable_load_extension(True)` safely scoped to a narrow window? (2) migration atomicity — is `backfill_vec` + `set_user_version(2)` idempotent across crash-mid-migration? (3) is `commit_upsert_vector` inside the commit transaction, or does it race against `commit_insert`'s `conn.commit()`? (4) check the `commit_semantic_search` SQL — does it use parameterized binding everywhere? Report anything HIGH or CRITICAL; MEDIUMs are nice-to-have.

Expected: HIGH/CRITICAL count 0. Address anything HIGH before proceeding.

### - [ ] Step 4: Verify the code-reviewer feedback is in CCR memory

Run:
```bash
.venv/bin/python -c "
from ccr.mcp.gcc_search_tools import gcc_search
# literal placeholder — during execution, run the actual MCP tool from Claude
print('Skip — exercised via MCP in Claude session')
"
```

Or call `gcc_search(query='Phase 4 review')` from the Claude session and verify the review landed.

### - [ ] Step 5: Milestone commit

```bash
git status
git add -A
git commit -m "ccr: Complete Phase 4 — sqlite-vec semantic search on memory.db commits_vec"
git show --stat HEAD | tail -5
```

Expected: non-empty commit. Empty-commit check (third-occurrence pattern): if stat is empty, `git reset --soft HEAD~1` then re-stage explicitly.

### - [ ] Step 6: Update CCR project memory

Call `gcc_commit` with:
- `title`: "Complete Phase 4: sqlite-vec semantic search on memory.db commits_vec"
- `what`: What was landed — the six sub-tasks + test counts.
- `why`: Eliminates per-call O(N) numpy cosine scan; unifies memory.db storage; schema version 1 → 2.
- `files_changed`: list of all 8 modified/created files.
- `next_step`: "Phase 5 candidates — (a) global.db cross-project ACE wiring, (b) address MEDIUM review notes from Phases 2/3/4, (c) index.db semantic chunk search via chunks_vec mirror. Recommend (a) as user-visible."
- `patterns_learned`: Update C110's post-commit-check pattern + two new Phase 4 patterns (see self-review below).

---

## Validation Matrix

| Case | Pre-Phase-4 behavior | Post-Phase-4 behavior |
|---|---|---|
| Fresh DB, sqlite-vec installed | No `commits_vec`, ONNX-cosine-in-Python semantic path | `commits_vec` created, `user_version=2`, backend KNN semantic path |
| Fresh DB, sqlite-vec missing | No vec table, ONNX-cosine-in-Python | `vec_available=False`, ONNX-cosine-in-Python fallback; `user_version` stays at 1 |
| Pre-Phase-4 DB, gzip JSON cache exists | Cache read each search | Migrated to `commits_vec` on first open; cache orphaned |
| Pre-Phase-4 DB, `.ccr/embeddings.db` sidecar exists | Sidecar read each search | Sidecar vectors imported into `commits_vec`; sidecar orphaned |
| New commit via `mem.commit()` | `_embed_commit` writes gzip JSON + sidecar | Same + `commit_upsert_vector` adds row to `commits_vec` |
| Restart after migration | Reads side-car / gzip JSON | Reads `commits_vec` only; `user_version=2` guards re-backfill |
| File backend path | ONNX-cosine-in-Python | Unchanged — `vec_available=False` for file backend |
| Malformed query vector (wrong dim) | Passes through, numpy shape error | `ValueError` raised by `_serialize_vec`; search returns `[]` |

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| sqlite-vec extension loads at session start but fails on some platforms | Low | `_create_connection` swallows extension errors; `install_vec` probes via `CREATE VIRTUAL TABLE` and sets `_vec_available=False` on `OperationalError` |
| Crash mid-migration leaves `commits_vec` half-populated at `user_version=1` | Low | `backfill_vec` uses `INSERT OR REPLACE` — re-running on a partial table just completes the missing rows before bumping to `user_version=2` |
| Migration OOMs on repos with >10k commits | Low | Backfill streams by `INSERT OR REPLACE` row-by-row within a single transaction; sqlite-vec handles ~100k rows comfortably. Commit every 1000 rows only if profiling shows it matters |
| `commit_upsert_vector` inside `mem.commit` fails silently | Low | Wrapped in `try/except: pass` — the gzip JSON fallback in `_embed_commit` still runs, so downgrades don't lose data |
| Empty commit slips into git history (third-occurrence pattern) | Medium | Every `git commit` step in this plan is followed by `git show --stat HEAD \| tail -5` — operator must verify non-empty before moving to the next task |

---

## Self-Review Checklist

Run this checklist after reading the plan end-to-end.

- [x] **Spec coverage:** Each of the 6 sub-tasks maps 1:1 to the Phase 4 scope (DDL, upsert, KNN, backfill, wiring, validation). No spec item missing.
- [x] **No placeholders:** Every code step shows complete code. No "TBD", no "similar to X", no "add error handling as appropriate". The `gcc_search` Step 4 documentation call-out is the one note suggesting operator action, not a placeholder.
- [x] **Type consistency:** `commit_upsert_vector(commit_id: str, vector: list[float]) -> None` and `commit_semantic_search(branch: str, query_vec: list[float], top_k: int) -> list[dict]` match across `base.py`, `file_backend.py`, `_sqlite_phase3a.py`, and all three test classes. `vec_available` is a `bool` property everywhere. `VEC_DIM = 384` is the single source of truth in `_sqlite_vec.py`.
- [x] **Empty-commit guard:** Every commit step explicitly runs `git show --stat HEAD | tail -5` and documents soft-reset recovery (third-occurrence pattern recorded in C110).
- [x] **Migration idempotency:** `user_version < 2` guard + `INSERT OR REPLACE` make backfill safe to re-run.
- [x] **Dual-backend symmetry:** `FileStorageBackend` gets no-op stubs so the abstract interface stays intact.

### Patterns to record on milestone commit

1. **"Load SQLite extensions at the connection-factory level, never per-query"** — `SqliteConnectionManager._create_connection` loads sqlite-vec once; every thread-local connection inherits. Prevents `no such module` errors on fresh threads.
2. **"One `user_version` per SQLite file, monotonically increasing across phases"** — memory.db: 0 (Phase 1) → 1 (Phase 2 FTS5) → 2 (Phase 4 vec); index.db: 0 → 1 (Phase 3 chunk FTS5). Independent migration streams avoid cross-DB coupling.
3. **"Migrations must be crash-idempotent: INSERT OR REPLACE + version bump in a single logical step"** — avoids half-migrated state after power loss or keyboard-interrupt.

---

## Handoff to Execution

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.

**2. Inline Execution** — batch-execute through this session, checkpointing between tasks.

Auto mode is active, so defaulting to **Inline Execution** via `superpowers:executing-plans`.
