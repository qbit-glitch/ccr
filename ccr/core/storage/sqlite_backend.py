"""SQLite storage backend for CCR.

Follows the SessionStore / SqliteVecStore pattern:
  - WAL journal mode for concurrent reads during writes
  - Thread-local connections via threading.local()
  - CREATE TABLE IF NOT EXISTS for idempotent setup
  - PRAGMA busy_timeout for write contention

Method implementations are split across phase-specific mixin files:
  _sqlite_phase1.py  — scratchpad, metrics, log, metadata
  _sqlite_phase2.py  — playbook bullets, sections, failure lessons, schema, audit
  _sqlite_phase3a.py — commits, rolling summaries, branches
  _sqlite_phase3b.py — links, patterns, triples, evolved summaries, clusters
  _sqlite_phase3c.py — discussions, session/phase summaries, summary meta, project state
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading

from ccr.core.storage._migration_phase5a import migrate_phase_5a
from ccr.core.storage._sqlite_fts5 import install_fts5
from ccr.core.storage._sqlite_vec import install_vec
from ccr.core.storage._sqlite_phase1 import Phase1Mixin
from ccr.core.storage._sqlite_phase2 import Phase2Mixin
from ccr.core.storage._sqlite_phase3a import Phase3aMixin
from ccr.core.storage._sqlite_phase3b import Phase3bMixin
from ccr.core.storage._sqlite_phase3c import Phase3cMixin
from ccr.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)


# ── DDL Constants ──────────────────────────────────────────────

_PHASE1_TABLES = """
CREATE TABLE IF NOT EXISTS scratchpad (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    expires_at  TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    key         TEXT PRIMARY KEY,
    value       INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS log_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    branch      TEXT NOT NULL,
    line        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_log_branch ON log_entries(branch);

CREATE TABLE IF NOT EXISTS metadata (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    updated_at  TEXT
);
"""


_PHASE3A_TABLES = """
CREATE TABLE IF NOT EXISTS commits (
    id              TEXT NOT NULL,
    branch          TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    title           TEXT NOT NULL,
    what            TEXT NOT NULL,
    why             TEXT NOT NULL,
    files_json      TEXT NOT NULL DEFAULT '[]',
    next_step       TEXT NOT NULL DEFAULT '',
    patterns_json   TEXT,
    score           REAL,
    author          TEXT NOT NULL DEFAULT '',
    ci_json         TEXT,
    experiment_json TEXT,
    ota_trace       TEXT,
    raw_block       TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (branch, id)
);
CREATE INDEX IF NOT EXISTS idx_commits_branch ON commits(branch);
CREATE INDEX IF NOT EXISTS idx_commits_id ON commits(id);

CREATE TABLE IF NOT EXISTS rolling_summaries (
    branch      TEXT PRIMARY KEY,
    summary     TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS branches (
    name        TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'active',
    parent      TEXT,
    purpose     TEXT,
    hypothesis  TEXT,
    conclusion  TEXT,
    linked_issue TEXT,
    team_owner  TEXT,
    priority    TEXT,
    created_at  TEXT NOT NULL
);
"""

_PHASE3B_TABLES = """
CREATE TABLE IF NOT EXISTS commit_links (
    source_id       TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    link_type       TEXT NOT NULL,
    score           REAL NOT NULL DEFAULT 0.0,
    shared_files_json TEXT,
    snippet         TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, link_type)
);
CREATE INDEX IF NOT EXISTS idx_links_source ON commit_links(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON commit_links(target_id);

CREATE TABLE IF NOT EXISTS patterns (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    first_seen      TEXT,
    commit_ids_json TEXT NOT NULL DEFAULT '[]',
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    promoted        INTEGER NOT NULL DEFAULT 0,
    success_count   INTEGER NOT NULL DEFAULT 0,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    quality_score   REAL NOT NULL DEFAULT 0.5,
    last_seen       TEXT,
    last_quality_update TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pattern_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS triples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 0.8,
    timestamp     TEXT,
    UNIQUE(subject, predicate, object)
);
CREATE INDEX IF NOT EXISTS idx_triples_commit ON triples(source_commit);
CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);

CREATE TABLE IF NOT EXISTS evolved_summaries (
    commit_id        TEXT PRIMARY KEY,
    evolved_what     TEXT NOT NULL,
    evolution_reason TEXT NOT NULL,
    evolved_at       TEXT NOT NULL,
    source_commit_id TEXT NOT NULL,
    original_what    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    label           TEXT,
    commit_ids_json TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_mapping (
    commit_id  TEXT PRIMARY KEY,
    cluster_id INTEGER NOT NULL REFERENCES clusters(id)
);
"""

_PHASE3C_TABLES = """
CREATE TABLE IF NOT EXISTS discussions (
    id          TEXT NOT NULL,
    branch      TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    topic       TEXT NOT NULL,
    hypothesis  TEXT NOT NULL DEFAULT '',
    alternatives TEXT NOT NULL DEFAULT '',
    decision    TEXT NOT NULL DEFAULT '',
    rationale   TEXT NOT NULL DEFAULT '',
    uncertainty TEXT NOT NULL DEFAULT '',
    linked_commit TEXT,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (branch, id)
);
CREATE INDEX IF NOT EXISTS idx_discussions_branch ON discussions(branch);

CREATE TABLE IF NOT EXISTS session_summaries (
    id          TEXT NOT NULL,
    branch      TEXT NOT NULL,
    start_date  TEXT NOT NULL,
    end_date    TEXT NOT NULL,
    commit_range TEXT,
    accomplished TEXT,
    files_touched TEXT,
    key_decisions TEXT,
    direction   TEXT,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (branch, id)
);
CREATE INDEX IF NOT EXISTS idx_session_branch ON session_summaries(branch);

CREATE TABLE IF NOT EXISTS phase_summaries (
    id              TEXT PRIMARY KEY,
    start_date      TEXT,
    end_date        TEXT,
    scope           TEXT,
    goal            TEXT,
    outcome         TEXT,
    accomplishments TEXT,
    files_changed   TEXT,
    branch_summary  TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_meta (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS project_state (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT
);
"""

_PHASE2_TABLES = """
CREATE TABLE IF NOT EXISTS playbook_bullets (
    id                  TEXT PRIMARY KEY,
    section             TEXT NOT NULL,
    content             TEXT NOT NULL,
    helpful             INTEGER NOT NULL DEFAULT 0,
    harmful             INTEGER NOT NULL DEFAULT 0,
    scope               TEXT NOT NULL DEFAULT 'general',
    when_to_apply       TEXT NOT NULL DEFAULT '',
    trigger_text        TEXT NOT NULL DEFAULT '',
    action              TEXT NOT NULL DEFAULT '',
    weighted_helpful    REAL NOT NULL DEFAULT 0.0,
    weighted_harmful    REAL NOT NULL DEFAULT 0.0,
    personal_decay_rate REAL NOT NULL DEFAULT 0.0,
    grpo_advantage      REAL NOT NULL DEFAULT 0.0,
    last_updated        TEXT,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS failure_lessons (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    bullet_id            TEXT NOT NULL REFERENCES playbook_bullets(id) ON DELETE CASCADE,
    failure_point        TEXT NOT NULL,
    flawed_reasoning     TEXT NOT NULL,
    counterfactual       TEXT NOT NULL,
    prevention_principle TEXT NOT NULL,
    task_context         TEXT NOT NULL DEFAULT '',
    timestamp            TEXT,
    evolved              INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_fl_bullet ON failure_lessons(bullet_id);

CREATE TABLE IF NOT EXISTS playbook_sections (
    position    INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS playbook_schema (
    version            INTEGER PRIMARY KEY,
    data_json          TEXT NOT NULL,
    parent_version     INTEGER,
    change_description TEXT,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS delta_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    author          TEXT NOT NULL DEFAULT '',
    ops_count       INTEGER NOT NULL DEFAULT 0,
    applied_count   INTEGER NOT NULL DEFAULT 0,
    scope           TEXT NOT NULL DEFAULT 'project',
    operations_json TEXT,
    failed_ids_json TEXT
);

CREATE TABLE IF NOT EXISTS archived_bullets (
    id          TEXT NOT NULL,
    section     TEXT,
    content     TEXT,
    helpful     INTEGER,
    harmful     INTEGER,
    archived_at TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT ''
);
"""


# ── Connection Manager ─────────────────────────────────────────

class SqliteConnectionManager:
    """Thread-safe SQLite connection manager with WAL mode."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        self._ensure_db()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        conn = self.get_conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")

    def get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_connection()
        return self._local.conn

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        # Phase 4: Load sqlite-vec extension if available. Silent failure is
        # correct — backends probe vec_available() and fall back cleanly.
        # try/finally ensures enable_load_extension(False) runs even if
        # sqlite_vec.load() raises, closing the dynamic-loading window.
        try:
            import sqlite_vec  # soft dep
            conn.enable_load_extension(True)
            try:
                sqlite_vec.load(conn)
            finally:
                conn.enable_load_extension(False)
        except Exception as exc:
            logger.debug(
                "sqlite-vec not loaded (%s): %s", type(exc).__name__, exc,
            )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def get_user_version(self) -> int:
        return self.get_conn().execute("PRAGMA user_version").fetchone()[0]

    def set_user_version(self, version: int) -> None:
        version = int(version)
        self.get_conn().execute(f"PRAGMA user_version = {version}")


# ── Backend Facade ─────────────────────────────────────────────

class SqliteStorageBackend(
    Phase1Mixin, Phase2Mixin, Phase3aMixin, Phase3bMixin, Phase3cMixin,
    StorageBackend,
):
    """SQLite-backed storage for CCR data."""

    def __init__(self, ccr_root: str, global_ccr_root: str | None = None):
        super().__init__(ccr_root, global_ccr_root)
        db_path = os.path.join(ccr_root, "memory.db")
        self._memory_mgr = SqliteConnectionManager(db_path)

        if global_ccr_root:
            global_db = os.path.join(global_ccr_root, "global.db")
            self._global_mgr = SqliteConnectionManager(global_db)
        else:
            self._global_mgr = None

        self._ensure_phase1_tables()
        self._ensure_phase2_tables()
        self._ensure_phase3a_tables()
        self._ensure_phase3b_tables()
        self._ensure_phase3c_tables()
        self._sync_flat_file_commits()

        # Phase 2: install FTS5 virtual tables + triggers (graceful if FTS5 missing)
        self._fts_available: bool = install_fts5(self.memory_conn)

        # Phase 2.4: backfill FTS5 indexes from existing rows for pre-Phase-2 DBs.
        # Use PRAGMA user_version as a one-way schema guard:
        #   0 -> pre-Phase-2 (needs backfill), 1 -> Phase 2 backfill complete.
        if self._fts_available:
            conn = self.memory_conn
            current_version = self._memory_mgr.get_user_version()
            if current_version < 1:
                from ._sqlite_fts5 import backfill_fts5
                backfill_fts5(conn)
                self._memory_mgr.set_user_version(1)

        # Phase 4: install commits_vec virtual table (graceful when sqlite-vec missing)
        self._vec_available: bool = install_vec(self.memory_conn)

        # Phase 4 migration: backfill legacy embeddings into commits_vec.
        # user_version 1 -> 2 guard (1 = Phase 2 FTS5 backfill complete).
        # Wrapped in a single transaction so a crash between backfill and
        # the user_version bump leaves user_version=1 (backfill re-runs,
        # but INSERT OR REPLACE makes it idempotent).
        if self._vec_available:
            current_version = self._memory_mgr.get_user_version()
            if current_version < 2:
                from ._sqlite_vec import backfill_vec
                with self.memory_conn:
                    backfill_vec(self.memory_conn, ccr_root)
                    self._memory_mgr.set_user_version(2)

        # ── Phase 5a: playbook flat-file → SQLite backfill (memory.db) ──────
        # Pass `self` (no recursion — helper no longer constructs a backend).
        try:
            if self._memory_mgr.get_user_version() < 3:
                playbook_path = os.path.join(self.ccr_root, "playbook.txt")
                failure_lessons_path = os.path.join(self.ccr_root, "failure_lessons.json")
                migrate_phase_5a(
                    backend=self,
                    playbook_path=playbook_path,
                    failure_lessons_path=failure_lessons_path,
                    scope="project",
                )
        except Exception as exc:
            logger.warning("Phase 5a project migration failed: %s", exc)

        # ── Phase 5a: global.db playbook backfill ───────────────────────────
        if self._global_mgr is not None and self.global_ccr_root:
            try:
                global_version = self._global_mgr.get_user_version()
                if global_version < 1:
                    global_playbook_path = os.path.join(
                        self.global_ccr_root, "global_playbook.txt"
                    )
                    global_fl_path = os.path.join(
                        self.global_ccr_root, "global_failure_lessons.json"
                    )
                    migrate_phase_5a(
                        backend=self,
                        playbook_path=global_playbook_path,
                        failure_lessons_path=global_fl_path,
                        scope="global",
                    )
            except Exception as exc:
                logger.warning("Phase 5a global migration failed: %s", exc)

    def _sync_flat_file_commits(self) -> None:
        """Incrementally backfill commits.md rows missing from memory.db.

        The one-shot migration sentinel can be older than recent flat-file
        commits when users flip CCR_STORAGE_BACKEND=sqlite later. Keep this
        sync idempotent so SQLite startup never falls behind the canonical
        commits.md files.
        """
        branches_dir = os.path.join(self.ccr_root, "branches")
        if not os.path.isdir(branches_dir):
            return
        try:
            from ccr.core.storage._migration_phase3 import _migrate_commits  # noqa: PLC0415

            before = self.memory_conn.total_changes
            with self.memory_conn:
                _migrate_commits(self.ccr_root, self.memory_conn)
            inserted = self.memory_conn.total_changes - before
            if inserted:
                logger.info("Synced %d flat-file commit row changes into SQLite", inserted)
        except Exception as exc:
            logger.warning("SQLite flat-file commit sync failed: %s", exc)

    def _ensure_phase1_tables(self) -> None:
        self.memory_conn.executescript(_PHASE1_TABLES)

    def _ensure_phase2_tables(self) -> None:
        self.memory_conn.executescript(_PHASE2_TABLES)
        if self._global_mgr:
            self.global_conn.executescript(_PHASE2_TABLES)

    def _ensure_phase3a_tables(self) -> None:
        self.memory_conn.executescript(_PHASE3A_TABLES)

    def _ensure_phase3b_tables(self) -> None:
        self.memory_conn.executescript(_PHASE3B_TABLES)

    def _ensure_phase3c_tables(self) -> None:
        self.memory_conn.executescript(_PHASE3C_TABLES)

    def _get_scoped_conn(self, scope: str) -> sqlite3.Connection:
        if scope == "global":
            if self.global_conn is None:
                raise ValueError("No global database configured")
            return self.global_conn
        return self.memory_conn

    @property
    def backend_type(self) -> str:
        return "sqlite"

    @property
    def fts_available(self) -> bool:
        """True iff FTS5 virtual tables + triggers were installed at init."""
        return self._fts_available

    @property
    def vec_available(self) -> bool:
        """True iff sqlite-vec extension loaded and commits_vec table created."""
        return self._vec_available

    @property
    def memory_conn(self) -> sqlite3.Connection:
        return self._memory_mgr.get_conn()

    @property
    def global_conn(self) -> sqlite3.Connection | None:
        return self._global_mgr.get_conn() if self._global_mgr else None

    def close(self) -> None:
        self._memory_mgr.close()
        if self._global_mgr:
            self._global_mgr.close()
