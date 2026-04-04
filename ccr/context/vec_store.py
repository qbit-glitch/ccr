"""Persistent vector store using sqlite-vec extension.

Optional dependency: install with `pip install ccr[vector]`.
Falls back to existing gzip JSON when sqlite-vec unavailable.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import threading
from datetime import datetime, timezone


def _is_sqlite_vec_available() -> bool:
    """Check if sqlite-vec extension is available."""
    try:
        import sqlite_vec  # noqa: F401

        return True
    except ImportError:
        return False


SQLITE_VEC_AVAILABLE = _is_sqlite_vec_available()


def serialize_float32(vec: list[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_float32(data: bytes, dim: int) -> list[float]:
    """Deserialize bytes to float32 vector."""
    return list(struct.unpack(f"{dim}f", data))


class SqliteVecStore:
    """Persistent vector store using sqlite-vec extension.

    Uses a single SQLite database with namespace-based isolation.
    Supports: upsert, search (KNN), get, batch get, delete, count.
    Thread-safe via connection-per-thread with WAL mode.
    """

    def __init__(self, db_path: str, dim: int = 384):
        self._db_path = db_path
        self._dim = dim
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._ensure_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_connection()
        return self._local.conn

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new connection with sqlite-vec loaded."""
        import sqlite_vec

        conn = sqlite3.connect(self._db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_db(self) -> None:
        """Create tables if they don't exist."""
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = self._get_conn()
        # Metadata table for ID-to-namespace mapping
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_metadata (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # Virtual table for vector search
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings"
            f" USING vec0(id TEXT PRIMARY KEY, embedding float[{self._dim}])"
        )
        conn.commit()

    def upsert(self, id: str, vector: list[float], namespace: str = "commit") -> None:
        """Insert or update a vector."""
        if len(vector) != self._dim:
            raise ValueError(f"Vector dim {len(vector)} != expected {self._dim}")
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        vec_bytes = serialize_float32(vector)
        # Upsert metadata
        conn.execute(
            "INSERT OR REPLACE INTO vec_metadata (id, namespace, created_at) VALUES (?, ?, ?)",
            (id, namespace, now),
        )
        # Upsert vector (delete + insert for virtual tables)
        conn.execute("DELETE FROM vec_embeddings WHERE id = ?", (id,))
        conn.execute(
            "INSERT INTO vec_embeddings (id, embedding) VALUES (?, ?)",
            (id, vec_bytes),
        )
        conn.commit()

    def search(
        self, query_vec: list[float], namespace: str = "commit", top_k: int = 10
    ) -> list[tuple[str, float]]:
        """KNN search. Returns list of (id, distance) sorted by distance ascending.

        Uses a two-step approach: KNN query on vec0 (which requires k=?),
        then filters by namespace from the metadata table.
        Over-fetches by 3x to account for namespace filtering.
        """
        if len(query_vec) != self._dim:
            raise ValueError(f"Query dim {len(query_vec)} != expected {self._dim}")
        conn = self._get_conn()
        query_bytes = serialize_float32(query_vec)
        # Over-fetch from vec0 (KNN requires k=? constraint), then filter by namespace.
        # Fetch up to 3x top_k to ensure enough results survive namespace filtering.
        fetch_k = top_k * 3
        rows = conn.execute(
            """
            SELECT v.id, v.distance
            FROM vec_embeddings v
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (query_bytes, fetch_k),
        ).fetchall()
        # Filter by namespace
        ns_ids = set(
            r[0]
            for r in conn.execute(
                "SELECT id FROM vec_metadata WHERE namespace = ?", (namespace,)
            ).fetchall()
        )
        results = [(row[0], row[1]) for row in rows if row[0] in ns_ids]
        return results[:top_k]

    def get(self, id: str) -> list[float] | None:
        """Get a single vector by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT embedding FROM vec_embeddings WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None
        return deserialize_float32(row[0], self._dim)

    def get_batch(self, ids: list[str]) -> dict[str, list[float]]:
        """Get multiple vectors by ID."""
        if not ids:
            return {}
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT id, embedding FROM vec_embeddings WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row[0]: deserialize_float32(row[1], self._dim) for row in rows}

    def delete(self, id: str) -> bool:
        """Delete a vector. Returns True if it existed."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM vec_metadata WHERE id = ?", (id,))
        conn.execute("DELETE FROM vec_embeddings WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0

    def count(self, namespace: str | None = None) -> int:
        """Count vectors, optionally filtered by namespace."""
        conn = self._get_conn()
        if namespace:
            row = conn.execute(
                "SELECT COUNT(*) FROM vec_metadata WHERE namespace = ?", (namespace,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM vec_metadata").fetchone()
        return row[0] if row else 0

    def list_ids(self, namespace: str = "commit") -> list[str]:
        """List all IDs in a namespace."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM vec_metadata WHERE namespace = ? ORDER BY id",
            (namespace,),
        ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


def get_vec_store(db_path: str, dim: int = 384) -> SqliteVecStore | None:
    """Factory that returns SqliteVecStore if sqlite-vec available, else None."""
    if not SQLITE_VEC_AVAILABLE:
        return None
    try:
        return SqliteVecStore(db_path, dim)
    except Exception:
        return None
