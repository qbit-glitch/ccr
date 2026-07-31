"""Abstract base class for CCR storage backends.

Methods are grouped by subsystem and added phase-by-phase:
  Phase 1: scratchpad, metrics, log, metadata
  Phase 2: playbook, failure_lessons, schema
  Phase 3a: commits, rolling_summaries, branches
  Phase 3b: links, patterns, triples, evolved_summaries, clusters
  Phase 3c: discussions, session_summaries, phase_summaries, summary_meta, project_state
  Phase 3d: project todos
  Phase 4: index (future)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Unified interface for CCR data persistence.

    Concrete implementations: FileStorageBackend (flat files),
    SqliteStorageBackend (SQLite WAL mode).
    """

    def __init__(self, ccr_root: str, global_ccr_root: str | None = None):
        self.ccr_root = ccr_root
        self.global_ccr_root = global_ccr_root

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return 'files' or 'sqlite'."""

    # ── Phase 1: Scratchpad ─────────────────────────────────────

    @abstractmethod
    def scratchpad_set(
        self, key: str, value: str, ttl_seconds: int | None = None,
    ) -> dict:
        """Create or update a scratchpad entry. Returns entry dict."""

    @abstractmethod
    def scratchpad_get(self, key: str) -> dict | None:
        """Get a scratchpad entry by key. None if missing or expired."""

    @abstractmethod
    def scratchpad_list(self) -> list[dict]:
        """List all non-expired scratchpad entries."""

    @abstractmethod
    def scratchpad_delete(self, key: str) -> bool:
        """Delete a scratchpad entry. Returns True if existed."""

    @abstractmethod
    def scratchpad_clear(self) -> int:
        """Clear all entries. Returns count of entries removed."""

    @abstractmethod
    def scratchpad_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search scratchpad entries by keyword match."""

    # ── Phase 1: Metrics ────────────────────────────────────────

    @abstractmethod
    def metrics_increment(self, key: str, amount: int = 1) -> None:
        """Increment a named counter."""

    @abstractmethod
    def metrics_get(self) -> dict[str, Any]:
        """Return all metrics as a dict."""

    # ── Phase 1: Log ────────────────────────────────────────────

    @abstractmethod
    def log_append(self, branch: str, line: str, max_lines: int = 500) -> None:
        """Append a line to the branch log with rotation."""

    @abstractmethod
    def log_read(self, branch: str, count: int = 50) -> str:
        """Read the last N lines of the branch log."""

    # ── Phase 1: Metadata ───────────────────────────────────────

    @abstractmethod
    def metadata_load(self) -> dict:
        """Load project metadata (branches, version, etc.)."""

    @abstractmethod
    def metadata_save(self, data: dict) -> None:
        """Save project metadata."""

    # ── Phase 2: Playbook Bullets ───────────────────────────────

    @abstractmethod
    def bullet_get(self, bullet_id: str, scope: str = "project") -> dict | None:
        """Get a single bullet by ID. Returns dict with all fields or None."""

    @abstractmethod
    def bullet_list(self, section: str | None = None, scope: str = "project") -> list[dict]:
        """List all bullets, optionally filtered by section."""

    @abstractmethod
    def bullet_insert(self, bullet: dict, scope: str = "project") -> None:
        """Insert a new bullet. Dict must have 'id', 'section', 'content', 'created_at'."""

    @abstractmethod
    def bullet_update(self, bullet_id: str, updates: dict, scope: str = "project") -> bool:
        """Update bullet fields. Returns True if bullet existed."""

    @abstractmethod
    def bullet_delete(self, bullet_id: str, scope: str = "project") -> bool:
        """Delete a bullet by ID. Returns True if existed."""

    @abstractmethod
    def bullet_update_counters(self, bullet_tags: list[dict], scope: str = "project") -> int:
        """Increment helpful/harmful counters. Returns count of updated bullets."""

    @abstractmethod
    def bullet_get_next_id(self, scope: str = "project") -> int:
        """Return the next available bullet ID number."""

    # ── Phase 2: Failure Lessons ────────────────────────────────

    @abstractmethod
    def failure_lessons_for_bullet(self, bullet_id: str, scope: str = "project") -> list[dict]:
        """Get all failure lessons for a bullet."""

    @abstractmethod
    def failure_lessons_insert(self, bullet_id: str, lesson: dict, scope: str = "project") -> None:
        """Insert a failure lesson for a bullet."""

    @abstractmethod
    def failure_lessons_mark_evolved(self, bullet_id: str, scope: str = "project") -> int:
        """Mark all lessons for a bullet as evolved. Returns count updated."""

    @abstractmethod
    def failure_lessons_all(self, scope: str = "project") -> dict[str, list[dict]]:
        """Get all failure lessons grouped by bullet_id."""

    # ── Phase 2: Playbook Sections ──────────────────────────────

    @abstractmethod
    def playbook_sections_get(self, scope: str = "project") -> list[str]:
        """Get ordered section names."""

    @abstractmethod
    def playbook_sections_set(self, sections: list[str], scope: str = "project") -> None:
        """Set ordered section names (replaces all)."""

    # ── Phase 2: Playbook Schema ────────────────────────────────

    @abstractmethod
    def playbook_schema_load(self, scope: str = "project") -> dict:
        """Load the current playbook schema as dict."""

    @abstractmethod
    def playbook_schema_save(self, schema: dict, history_entry: dict | None = None, scope: str = "project") -> None:
        """Save schema. If history_entry provided, append to version history."""

    @abstractmethod
    def playbook_schema_history(self, scope: str = "project") -> list[dict]:
        """Get schema version history."""

    # ── Phase 2: Audit ──────────────────────────────────────────

    @abstractmethod
    def delta_history_append(self, entry: dict, scope: str = "project") -> None:
        """Append a delta operation audit entry."""

    @abstractmethod
    def archived_bullets_insert(self, bullets: list[dict], reason: str, scope: str = "project") -> int:
        """Archive pruned bullets. Returns count archived."""

    # ── Phase 3a: Commits ─────────────────────────────────────────

    @abstractmethod
    def commit_insert(self, branch: str, data: dict) -> None:
        """Insert a structured commit record."""

    @abstractmethod
    def commit_get(self, branch: str, commit_id: str) -> dict | None:
        """Get a commit by ID. Returns dict or None."""

    @abstractmethod
    def commit_list(self, branch: str, limit: int = 10, offset: int = 0) -> list[dict]:
        """List commits on a branch, most recent first."""

    @abstractmethod
    def commit_get_next_id(self, branch: str) -> str:
        """Return the next available commit ID (e.g. 'C042')."""

    @abstractmethod
    def commit_update(self, branch: str, commit_id: str, updates: dict) -> bool:
        """Update commit fields. Returns True if commit existed."""

    @abstractmethod
    def commit_search_text(self, branch: str, term: str, max_results: int = 5) -> list[dict]:
        """Search commits by substring match across text fields."""

    def commit_search_with_snippet(
        self, branch: str, term: str, max_results: int = 5,
    ) -> list[dict]:
        """FTS5-aware commit search with snippets. Returns [] if unavailable."""
        raise NotImplementedError

    @abstractmethod
    def commit_count(self, branch: str) -> int:
        """Return total commit count on a branch."""

    # ── Phase 3a: Rolling Summaries ────────────────────────────────

    @abstractmethod
    def rolling_summary_get(self, branch: str) -> str:
        """Get the rolling summary for a branch. Empty string if none."""

    @abstractmethod
    def rolling_summary_set(self, branch: str, summary: str) -> None:
        """Set the rolling summary for a branch."""

    # ── Phase 3a: Branches ─────────────────────────────────────────

    @abstractmethod
    def branch_create(self, name: str, data: dict) -> None:
        """Create a branch record. Dict should have purpose, hypothesis, parent, created_at."""

    @abstractmethod
    def branch_get(self, name: str) -> dict | None:
        """Get branch metadata by name."""

    @abstractmethod
    def branch_list(self, status: str | None = None) -> list[dict]:
        """List branches, optionally filtered by status."""

    @abstractmethod
    def branch_update(self, name: str, updates: dict) -> bool:
        """Update branch fields. Returns True if branch existed."""

    @abstractmethod
    def branch_update_status(self, name: str, status: str) -> bool:
        """Update branch status. Returns True if branch existed."""

    # ── Phase 3b: Links ────────────────────────────────────────────

    @abstractmethod
    def link_insert_batch(self, source_id: str, links: list[dict]) -> None:
        """Insert links from source_id. Each dict has target, link_type, score, etc."""

    @abstractmethod
    def link_get_for_commit(self, commit_id: str) -> dict:
        """Get all links for a commit. Returns {link_type: [entries]}."""

    @abstractmethod
    def link_get_all(self) -> dict:
        """Get entire link graph. Returns {source_id: {link_type: [entries]}}."""

    @abstractmethod
    def link_prune(self, max_nodes: int) -> int:
        """Evict oldest nodes if graph exceeds max_nodes. Returns evicted count."""

    # ── Phase 3b: Patterns ─────────────────────────────────────────

    @abstractmethod
    def pattern_load_all(self) -> dict:
        """Load full pattern buffer. Returns {version, patterns: {pid: dict}, next_id}."""

    @abstractmethod
    def pattern_save_all(self, data: dict) -> None:
        """Atomically replace all patterns. Dict has version, patterns, next_id."""

    @abstractmethod
    def pattern_get(self, pattern_id: str) -> dict | None:
        """Get a single pattern by ID."""

    @abstractmethod
    def pattern_update(self, pattern_id: str, updates: dict) -> bool:
        """Update pattern fields. Returns True if pattern existed."""

    @abstractmethod
    def pattern_get_next_id(self) -> int:
        """Return the next available pattern ID number."""

    @abstractmethod
    def pattern_search_text(self, term: str, max_results: int = 10) -> list[dict]:
        """Search patterns by substring / FTS5 match against the text field.

        Returned dicts share the shape of `pattern_load_all()["patterns"]` values:
        id, text, commit_ids (list), promoted (bool), and scalar columns.
        """

    # ── Phase 3b: Triples ──────────────────────────────────────────

    @abstractmethod
    def triple_insert_batch(self, triples: list[dict]) -> int:
        """Insert triples with dedup. Returns count of newly added."""

    @abstractmethod
    def triple_list(
        self, top_k: int = 10, commit_id: str | None = None,
        entity: str | None = None,
    ) -> list[dict]:
        """List triples, optionally filtered by commit or entity."""

    @abstractmethod
    def triple_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search triples by substring match across fields."""

    @abstractmethod
    def triple_count(self) -> int:
        """Return total triple count."""

    # ── Phase 3b: Evolved Summaries ────────────────────────────────

    @abstractmethod
    def evolved_summary_get(self, commit_id: str) -> dict | None:
        """Get evolved summary for a commit. None if not evolved."""

    @abstractmethod
    def evolved_summary_set(self, commit_id: str, data: dict) -> None:
        """Set or update evolved summary for a commit."""

    @abstractmethod
    def evolved_summary_all(self) -> dict:
        """Get all evolved summaries. Returns {commit_id: data}."""

    # ── Phase 3b: Clusters ─────────────────────────────────────────

    @abstractmethod
    def cluster_save(self, clusters: list[dict]) -> None:
        """Save cluster definitions with commit mapping."""

    @abstractmethod
    def cluster_load(self) -> dict:
        """Load clusters. Returns {clusters: [...], commit_to_cluster: {...}}."""

    # ── Phase 3c: Discussions ─────────────────────────────────────

    @abstractmethod
    def discussion_insert(self, branch: str, data: dict) -> None:
        """Insert a discussion record. Dict has id, timestamp, topic, hypothesis, etc."""

    @abstractmethod
    def discussion_list(
        self, branch: str, search: str | None = None,
        topic: str | None = None, date_range: list[str] | None = None,
    ) -> list[dict]:
        """List discussions with optional filters. Most recent first."""

    @abstractmethod
    def discussion_get_next_id(self, branch: str) -> str:
        """Return next available discussion ID (e.g. 'D001')."""

    @abstractmethod
    def discussion_search_text(
        self, branch: str, term: str, max_results: int = 10,
    ) -> list[dict]:
        """Return discussions on `branch` matching `term` (substring / FTS5)."""

    # ── Phase 3c: Session Summaries ───────────────────────────────

    @abstractmethod
    def session_summary_insert(self, branch: str, data: dict) -> None:
        """Insert a session summary. Dict has id, start_date, end_date, etc."""

    @abstractmethod
    def session_summary_list(self, branch: str, count: int = 3) -> list[dict]:
        """List recent session summaries for a branch."""

    @abstractmethod
    def session_summary_get_next_id(self, branch: str) -> str:
        """Return next available session summary ID (e.g. 'S001')."""

    # ── Phase 3c: Phase Summaries ─────────────────────────────────

    @abstractmethod
    def phase_summary_insert(self, data: dict) -> None:
        """Insert a phase summary. Dict has id, start_date, end_date, scope, etc."""

    @abstractmethod
    def phase_summary_list(self, count: int = 3) -> list[dict]:
        """List recent phase summaries."""

    @abstractmethod
    def phase_summary_get_next_id(self) -> str:
        """Return next available phase summary ID (e.g. 'P001')."""

    # ── Phase 3c: Summary Meta ────────────────────────────────────

    @abstractmethod
    def summary_meta_load(self) -> dict:
        """Load summary metadata (version, session/phase tracking)."""

    @abstractmethod
    def summary_meta_save(self, data: dict) -> None:
        """Save summary metadata."""

    # ── Phase 3c: Project State ───────────────────────────────────

    @abstractmethod
    def project_state_get(self, key: str) -> str | None:
        """Get a project state value by key (e.g. 'overview')."""

    @abstractmethod
    def project_state_set(self, key: str, value: str) -> None:
        """Set a project state value."""

    # ── Phase 3d: TODOs ─────────────────────────────────────────

    @abstractmethod
    def todo_insert(self, data: dict) -> dict:
        """Insert a structured TODO item and return the stored record."""

    @abstractmethod
    def todo_get(self, todo_id: str) -> dict | None:
        """Get a TODO by ID."""

    @abstractmethod
    def todo_list(
        self,
        status: str | None = None,
        branch: str | None = None,
        include_done: bool = False,
        limit: int = 20,
    ) -> list[dict]:
        """List TODOs, sorted for active-work display."""

    @abstractmethod
    def todo_update(self, todo_id: str, updates: dict) -> bool:
        """Update a TODO. Returns True if it existed."""

    @abstractmethod
    def todo_delete(self, todo_id: str) -> bool:
        """Hard-delete a TODO. Returns True if it existed."""

    @abstractmethod
    def todo_get_next_id(self) -> str:
        """Return the next TODO ID, e.g. T001."""

    @abstractmethod
    def todo_reorder(
        self,
        todo_id: str,
        before_id: str | None = None,
        after_id: str | None = None,
        order_index: int | None = None,
    ) -> bool:
        """Update TODO ordering. Returns True if the TODO existed."""

    # ── Lifecycle ────────────────────────────────────────────────

    def close(self) -> None:
        """Release resources. Override in backends that hold connections."""
