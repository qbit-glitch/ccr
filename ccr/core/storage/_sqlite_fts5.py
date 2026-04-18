"""FTS5 full-text search helpers for ``memory.db``.

External-content FTS5 virtual tables shadow the canonical tables
(``commits``, ``discussions``, ``triples``, ``patterns``). Auto-synced via
INSERT / DELETE / UPDATE triggers so the inverted index always reflects the
base tables. External-content means FTS5 stores only the inverted index --
the document text lives in the canonical table -- so there is no data
duplication.

Graceful fallback: if the SQLite build lacks FTS5 compile flags,
``install_fts5`` returns ``False`` and the caller must keep using LIKE-based
search. The module itself imports fine without FTS5; only the probe/install
paths touch the FTS5 virtual-table machinery.

This mirrors the ``turns_fts`` pattern in ``ccr/core/session_store.py`` which
has been running in production. Four triggers per source table (AFTER INSERT,
AFTER DELETE, BEFORE UPDATE, AFTER UPDATE) keep the FTS index consistent.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List


# ---------- Virtual-table DDL ------------------------------------------------

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
    CREATE_COMMITS_FTS,
    CREATE_DISCUSSIONS_FTS,
    CREATE_TRIPLES_FTS,
    CREATE_PATTERNS_FTS,
)


# ---------- Trigger templates -----------------------------------------------
# Four triggers per source table -- mirrors session_store.turns_fts:
#   - {source}_ai  AFTER INSERT   -> push new row to FTS
#   - {source}_ad  AFTER DELETE   -> tombstone old rowid in FTS
#   - {source}_bu  BEFORE UPDATE  -> remove stale copy from FTS
#   - {source}_au  AFTER UPDATE   -> push updated row to FTS
#
# Total: 16 triggers across 4 sources.

_TRIGGER_TEMPLATES: Dict[str, Dict[str, object]] = {
    "commits": {
        "cols": ("title", "what", "why", "next_step", "files_json"),
        "rowid": "rowid",
    },
    "discussions": {
        "cols": (
            "topic",
            "hypothesis",
            "alternatives",
            "decision",
            "rationale",
            "uncertainty",
        ),
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


def _trigger_sql(source: str) -> List[str]:
    """Generate the 4 trigger SQL strings for the given source table.

    Args:
        source: One of ``commits``, ``discussions``, ``triples``, ``patterns``.

    Returns:
        List of 4 ``CREATE TRIGGER IF NOT EXISTS`` statements: AI, AD, BU, AU.
    """
    spec = _TRIGGER_TEMPLATES[source]
    cols_tuple = spec["cols"]  # type: ignore[index]
    rowid = spec["rowid"]  # type: ignore[index]
    cols = ", ".join(cols_tuple)  # type: ignore[arg-type]
    new_vals = ", ".join(f"new.{c}" for c in cols_tuple)  # type: ignore[union-attr]
    old_vals = ", ".join(f"old.{c}" for c in cols_tuple)  # type: ignore[union-attr]
    fts = f"{source}_fts"
    return [
        # AFTER INSERT -> push new row into FTS.
        f"""CREATE TRIGGER IF NOT EXISTS {source}_ai AFTER INSERT ON {source} BEGIN
            INSERT INTO {fts}(rowid, {cols}) VALUES (new.{rowid}, {new_vals});
        END;""",
        # AFTER DELETE -> tombstone old rowid in FTS.
        f"""CREATE TRIGGER IF NOT EXISTS {source}_ad AFTER DELETE ON {source} BEGIN
            INSERT INTO {fts}({fts}, rowid, {cols}) VALUES ('delete', old.{rowid}, {old_vals});
        END;""",
        # BEFORE UPDATE -> remove stale copy from FTS (uses OLD values).
        f"""CREATE TRIGGER IF NOT EXISTS {source}_bu BEFORE UPDATE ON {source} BEGIN
            INSERT INTO {fts}({fts}, rowid, {cols}) VALUES ('delete', old.{rowid}, {old_vals});
        END;""",
        # AFTER UPDATE -> push updated row into FTS.
        f"""CREATE TRIGGER IF NOT EXISTS {source}_au AFTER UPDATE ON {source} BEGIN
            INSERT INTO {fts}(rowid, {cols}) VALUES (new.{rowid}, {new_vals});
        END;""",
    ]


# ---------- Public API ------------------------------------------------------


def fts5_available(conn: sqlite3.Connection) -> bool:
    """Return ``True`` iff the SQLite build has FTS5 compiled in.

    Side-effect-free: creates and immediately drops a throwaway virtual table
    named ``__fts5_probe``. If FTS5 is unavailable, ``sqlite3.OperationalError``
    is raised by ``CREATE VIRTUAL TABLE`` and swallowed here.

    Args:
        conn: An open SQLite connection.

    Returns:
        ``True`` if FTS5 is supported, ``False`` otherwise.
    """
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_probe USING fts5(x)"
        )
        conn.execute("DROP TABLE __fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def install_fts5(conn: sqlite3.Connection) -> bool:
    """Create FTS5 virtual tables and their sync triggers. Idempotent.

    Every ``CREATE`` uses ``IF NOT EXISTS`` so calling this twice is a no-op.

    Args:
        conn: An open SQLite connection whose base schema (``commits``,
            ``discussions``, ``triples``, ``patterns``) already exists.

    Returns:
        ``True`` on success. ``False`` if FTS5 is not available in this build
        (caller must fall back to LIKE search).

    Raises:
        sqlite3.OperationalError: For errors other than missing FTS5
            (e.g. missing base tables).
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


def backfill_fts5(conn: sqlite3.Connection) -> Dict[str, int]:
    """Rebuild FTS5 indexes from the canonical tables.

    Uses the FTS5 ``'rebuild'`` command which re-reads rows from the external
    content table in one pass. Triggers do NOT fire during rebuild.

    Safe on a fresh DB (no rows -> no inserts). Sources whose FTS table is
    missing (e.g. on builds without FTS5, or partial installs) are skipped
    gracefully.

    Args:
        conn: An open SQLite connection.

    Returns:
        Dict mapping source table name -> row count in its FTS index after
        rebuild. Only includes sources whose FTS table actually exists.
    """
    counts: Dict[str, int] = {}
    for source in _TRIGGER_TEMPLATES:
        fts = f"{source}_fts"
        got = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (fts,),
        ).fetchone()
        if got is None:
            continue
        conn.execute(f"INSERT INTO {fts}({fts}) VALUES ('rebuild')")
        n = conn.execute(f"SELECT count(*) FROM {fts}").fetchone()[0]
        counts[source] = n
    conn.commit()
    return counts
