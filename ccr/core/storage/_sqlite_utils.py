"""Shared utilities for SQLite backend mixins."""

from __future__ import annotations

import re
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _escape_like(term: str) -> str:
    """Escape LIKE wildcards so user input is treated literally."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_SAFE_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_BULLET_COLUMNS = frozenset({
    "section", "content", "helpful", "harmful", "scope", "when_to_apply",
    "trigger_text", "action", "weighted_helpful", "weighted_harmful",
    "personal_decay_rate", "grpo_advantage", "last_updated",
})
_COMMIT_COLUMNS = frozenset({
    "timestamp", "title", "what", "why", "next_step", "score", "author",
    "ota_trace", "raw_block", "files_json", "patterns_json", "ci_json",
    "experiment_json",
})
_BRANCH_COLUMNS = frozenset({
    "status", "parent", "purpose", "hypothesis", "conclusion",
    "linked_issue", "team_owner", "priority", "created_at",
})
_PATTERN_COLUMNS = frozenset({
    "text", "first_seen", "commit_ids_json", "occurrence_count",
    "promoted", "success_count", "failure_count", "quality_score",
    "last_seen", "last_quality_update",
})
