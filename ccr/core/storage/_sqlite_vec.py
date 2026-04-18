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


CREATE_COMMITS_VEC = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS commits_vec"
    f" USING vec0(id TEXT PRIMARY KEY, embedding float[{VEC_DIM}])"
)


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
    # DDL autocommits in SQLite; wrapping in `with conn:` just ensures any
    # in-flight implicit transaction from the caller doesn't tangle with us.
    with conn:
        conn.execute(CREATE_COMMITS_VEC)
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

    legacy_db = os.path.join(ccr_root, "embeddings.db")
    if os.path.exists(legacy_db):
        src = None
        try:
            import sqlite_vec  # soft dep
            src = sqlite3.connect(legacy_db)
            src.enable_load_extension(True)
            try:
                sqlite_vec.load(src)
            finally:
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
        except Exception as exc:
            logger.warning("backfill_vec: legacy embeddings.db import failed: %s", exc)
        finally:
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass

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

    # Caller wraps in `with self.memory_conn:` for atomicity w/ set_user_version.
    counts["total"] = counts["sqlite_vec"] + counts["gzip_json"]
    return counts
