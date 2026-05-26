"""Temporal fact ledger for evidence-first memory recall.

This module is deliberately file-backed and optional. It gives CCR a durable
place to record project facts with temporal validity without changing the
existing commit/session/playbook storage contracts.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
    "current", "did", "do", "does", "how", "memory", "our", "record",
    "records", "status", "we", "what", "when", "where", "which", "who", "why",
}

_NEGATION_WORDS = {
    "no", "not", "never", "without", "disabled", "unsupported", "missing",
    "cannot", "can't", "wont", "won't", "isn't", "doesn't", "failed",
}


def utc_now() -> str:
    """Return a stable ISO UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json_write(path: str, data: dict[str, Any]) -> None:
    """Write JSON atomically in the same directory as ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".facts.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def normalize_fact_key(text: str) -> str:
    """Build a compact stable key from a fact statement."""
    words = [
        w for w in re.findall(r"[a-z0-9_]+", text.lower())
        if len(w) > 1 and w not in _STOP_WORDS
    ]
    return "_".join(words[:8]) or "fact"


def tokenize(text: str) -> set[str]:
    """Tokenize text for lightweight lexical matching."""
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 1 and w not in _STOP_WORDS
    }


def lexical_score(query: str, text: str) -> float:
    """Return a small deterministic relevance score in [0, 1]."""
    q = tokenize(query)
    t = tokenize(text)
    if not q or not t:
        return 0.0
    overlap = len(q & t)
    if overlap == 0:
        return 0.0
    return overlap / max(len(q), 1)


def _has_negation(text: str) -> bool:
    words = tokenize(text)
    return bool(words & _NEGATION_WORDS)


@dataclass
class FactRecord:
    """A durable project fact with temporal and provenance metadata."""

    id: str
    key: str
    statement: str
    observed_at: str
    valid_from: str = ""
    valid_to: str = ""
    superseded_by: str = ""
    confidence: float = 0.75
    source_commit: str = ""
    source_session: str = ""
    source_file: str = ""
    pinned: bool = False
    classification: str = "confirmed"
    evidence_ids: list[str] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @property
    def is_active(self) -> bool:
        return not self.superseded_by and not self.valid_to

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FactRecord":
        return cls(
            id=str(data.get("id", "")),
            key=str(data.get("key") or normalize_fact_key(str(data.get("statement", "")))),
            statement=str(data.get("statement", "")),
            observed_at=str(data.get("observed_at") or data.get("created_at") or utc_now()),
            valid_from=str(data.get("valid_from", "")),
            valid_to=str(data.get("valid_to", "")),
            superseded_by=str(data.get("superseded_by", "")),
            confidence=float(data.get("confidence", 0.75)),
            source_commit=str(data.get("source_commit", "")),
            source_session=str(data.get("source_session", "")),
            source_file=str(data.get("source_file", "")),
            pinned=bool(data.get("pinned", False)),
            classification=str(data.get("classification") or "confirmed"),
            evidence_ids=list(data.get("evidence_ids") or []),
            episode_ids=list(data.get("episode_ids") or []),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
        )


@dataclass
class FactConflict:
    """A candidate conflict between two active facts."""

    key: str
    fact_a: str
    fact_b: str
    reason: str
    severity: str = "medium"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class FactLedger:
    """Small JSON-backed temporal fact ledger."""

    def __init__(self, ccr_root: str):
        self.ccr_root = ccr_root
        self.path = os.path.join(ccr_root, "facts.json")

    def _load_data(self) -> dict[str, Any]:
        if not os.path.isfile(self.path):
            return {"version": 1, "facts": []}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.loads(fh.read() or "{}")
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "facts": []}
        if not isinstance(data, dict):
            return {"version": 1, "facts": []}
        facts = data.get("facts")
        if not isinstance(facts, list):
            data["facts"] = []
        data.setdefault("version", 1)
        return data

    def _save_facts(self, facts: list[FactRecord]) -> None:
        _atomic_json_write(
            self.path,
            {"version": 1, "facts": [f.to_dict() for f in facts]},
        )

    def list_facts(
        self,
        query: str = "",
        include_inactive: bool = False,
        limit: int = 50,
    ) -> list[FactRecord]:
        """List facts, optionally filtered by lexical query."""
        facts = [FactRecord.from_dict(f) for f in self._load_data().get("facts", [])]
        if not include_inactive:
            facts = [f for f in facts if f.is_active]
        if query.strip():
            facts = [
                f for f in facts
                if lexical_score(query, f"{f.id} {f.key} {f.statement} {f.source_commit}") > 0
                or query.strip().lower() == f.id.lower()
            ]
            facts.sort(
                key=lambda f: lexical_score(
                    query,
                    f"{f.id} {f.key} {f.statement} {f.source_commit}",
                ),
                reverse=True,
            )
        else:
            facts.sort(key=lambda f: f.updated_at or f.created_at, reverse=True)
        return facts[: max(1, limit)]

    def add_fact(
        self,
        statement: str,
        *,
        key: str = "",
        observed_at: str = "",
        valid_from: str = "",
        valid_to: str = "",
        superseded_by: str = "",
        confidence: float = 0.75,
        source_commit: str = "",
        source_session: str = "",
        source_file: str = "",
        pinned: bool = False,
        classification: str = "confirmed",
        evidence_ids: list[str] | None = None,
        episode_ids: list[str] | None = None,
    ) -> FactRecord:
        """Append a fact without mutating existing facts."""
        statement = statement.strip()
        if not statement:
            raise ValueError("Fact statement must not be empty.")
        facts = [FactRecord.from_dict(f) for f in self._load_data().get("facts", [])]
        next_id = self._next_fact_id(facts)
        now = utc_now()
        fact = FactRecord(
            id=next_id,
            key=(key.strip() or normalize_fact_key(statement)),
            statement=statement,
            observed_at=observed_at.strip() or now,
            valid_from=valid_from.strip(),
            valid_to=valid_to.strip(),
            superseded_by=superseded_by.strip(),
            confidence=max(0.0, min(1.0, float(confidence))),
            source_commit=source_commit.strip(),
            source_session=source_session.strip(),
            source_file=source_file.strip(),
            pinned=pinned,
            classification=classification.strip() or "confirmed",
            evidence_ids=list(evidence_ids or []),
            episode_ids=list(episode_ids or []),
            created_at=now,
            updated_at=now,
        )
        facts.append(fact)
        self._save_facts(facts)
        return fact

    def supersede_fact(self, fact_id: str, superseded_by: str = "", valid_to: str = "") -> bool:
        """Mark a fact inactive while preserving its record."""
        facts = [FactRecord.from_dict(f) for f in self._load_data().get("facts", [])]
        now = utc_now()
        changed = False
        for fact in facts:
            if fact.id == fact_id:
                fact.superseded_by = superseded_by.strip()
                fact.valid_to = valid_to.strip() or now
                fact.updated_at = now
                changed = True
                break
        if changed:
            self._save_facts(facts)
        return changed

    def detect_conflicts(self, query: str = "", limit: int = 50) -> list[FactConflict]:
        """Find candidate conflicts among active facts.

        The first pass is intentionally conservative: facts with the same key and
        different statements are flagged. If one side contains a negation marker
        and the other does not, severity is high.
        """
        facts = self.list_facts(query=query, include_inactive=False, limit=1000)
        grouped: dict[str, list[FactRecord]] = {}
        for fact in facts:
            grouped.setdefault(fact.key, []).append(fact)

        conflicts: list[FactConflict] = []
        for key, items in grouped.items():
            if len(items) < 2:
                continue
            for i, left in enumerate(items):
                for right in items[i + 1:]:
                    if left.statement.strip().lower() == right.statement.strip().lower():
                        continue
                    left_neg = _has_negation(left.statement)
                    right_neg = _has_negation(right.statement)
                    severity = "high" if left_neg != right_neg else "medium"
                    reason = (
                        "same key with opposing polarity"
                        if severity == "high"
                        else "same key with different active statements"
                    )
                    conflicts.append(FactConflict(
                        key=key,
                        fact_a=left.id,
                        fact_b=right.id,
                        reason=reason,
                        severity=severity,
                    ))
        conflicts.sort(key=lambda c: (c.severity != "high", c.key, c.fact_a, c.fact_b))
        return conflicts[: max(1, limit)]

    @staticmethod
    def _next_fact_id(facts: list[FactRecord]) -> str:
        max_id = 0
        for fact in facts:
            m = re.match(r"F(\d+)$", fact.id)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return f"F{max_id + 1:03d}"
