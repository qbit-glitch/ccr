"""Memory write quarantine for inferred and speculative facts.

Confirmed and tool-observed memories can be written directly to the fact
ledger.  Inferred or speculative memories are queued here first so teams can
promote, reject, or review them before they affect trusted recall.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ccr.core.events import atomic_write_json
from ccr.core.facts import FactLedger, normalize_fact_key


VALID_CLASSIFICATIONS = {"confirmed", "inferred", "speculative", "tool_observed"}
QUARANTINE_CLASSIFICATIONS = {"inferred", "speculative"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class QuarantineItem:
    """One pending memory candidate."""

    id: str
    statement: str
    key: str
    classification: str
    reason: str = ""
    confidence: float = 0.5
    source_commit: str = ""
    source_session: str = ""
    source_file: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    promoted_fact_id: str = ""
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuarantineItem":
        return cls(
            id=str(data.get("id", "")),
            statement=str(data.get("statement", "")),
            key=str(data.get("key") or normalize_fact_key(str(data.get("statement", "")))),
            classification=str(data.get("classification") or "inferred"),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.5)),
            source_commit=str(data.get("source_commit", "")),
            source_session=str(data.get("source_session", "")),
            source_file=str(data.get("source_file", "")),
            evidence_ids=list(data.get("evidence_ids") or []),
            episode_ids=list(data.get("episode_ids") or []),
            metadata=dict(data.get("metadata") or {}),
            status=str(data.get("status") or "pending"),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            promoted_fact_id=str(data.get("promoted_fact_id", "")),
            rejection_reason=str(data.get("rejection_reason", "")),
        )


class MemoryQuarantine:
    """JSON-backed quarantine queue."""

    def __init__(self, ccr_root: str):
        self.ccr_root = ccr_root
        self.path = os.path.join(ccr_root, "quarantine.json")

    def submit(
        self,
        statement: str,
        *,
        key: str = "",
        classification: str = "inferred",
        reason: str = "",
        confidence: float = 0.5,
        source_commit: str = "",
        source_session: str = "",
        source_file: str = "",
        evidence_ids: list[str] | None = None,
        episode_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QuarantineItem:
        """Add a memory candidate to quarantine."""
        statement = statement.strip()
        if not statement:
            raise ValueError("statement must not be empty.")
        classification = normalize_classification(classification)
        if classification not in QUARANTINE_CLASSIFICATIONS:
            raise ValueError("Only inferred/speculative memories should be quarantined.")
        items = self.list_items(status="all")
        now = utc_now()
        item = QuarantineItem(
            id=self._next_id(items),
            statement=statement,
            key=key.strip() or normalize_fact_key(statement),
            classification=classification,
            reason=reason.strip(),
            confidence=max(0.0, min(1.0, float(confidence))),
            source_commit=source_commit.strip(),
            source_session=source_session.strip(),
            source_file=source_file.strip(),
            evidence_ids=list(evidence_ids or []),
            episode_ids=list(episode_ids or []),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        items.append(item)
        self._save(items)
        return item

    def list_items(self, status: str = "pending", limit: int = 100) -> list[QuarantineItem]:
        """List quarantine items."""
        data = self._load()
        items = [QuarantineItem.from_dict(i) for i in data.get("items", [])]
        status = status.strip().lower() or "pending"
        if status != "all":
            items = [item for item in items if item.status == status]
        items.sort(key=lambda item: item.updated_at or item.created_at, reverse=True)
        return items[: max(1, limit)]

    def promote(self, item_id: str, ledger: FactLedger) -> QuarantineItem | None:
        """Promote a pending quarantine item into the fact ledger."""
        items = self.list_items(status="all", limit=100000)
        now = utc_now()
        promoted: QuarantineItem | None = None
        for item in items:
            if item.id == item_id and item.status == "pending":
                fact = ledger.add_fact(
                    item.statement,
                    key=item.key,
                    confidence=item.confidence,
                    source_commit=item.source_commit,
                    source_session=item.source_session,
                    source_file=item.source_file,
                    classification=item.classification,
                    evidence_ids=item.evidence_ids,
                    episode_ids=item.episode_ids,
                )
                item.status = "promoted"
                item.updated_at = now
                item.promoted_fact_id = fact.id
                promoted = item
                break
        if promoted:
            self._save(items)
        return promoted

    def reject(self, item_id: str, reason: str = "") -> QuarantineItem | None:
        """Reject a pending quarantine item."""
        items = self.list_items(status="all", limit=100000)
        now = utc_now()
        rejected: QuarantineItem | None = None
        for item in items:
            if item.id == item_id and item.status == "pending":
                item.status = "rejected"
                item.updated_at = now
                item.rejection_reason = reason.strip()
                rejected = item
                break
        if rejected:
            self._save(items)
        return rejected

    def _load(self) -> dict[str, Any]:
        if not os.path.isfile(self.path):
            return {"version": 1, "items": []}
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.loads(fh.read() or "{}")
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "items": []}
        if not isinstance(data, dict):
            return {"version": 1, "items": []}
        data.setdefault("version", 1)
        if not isinstance(data.get("items"), list):
            data["items"] = []
        return data

    def _save(self, items: list[QuarantineItem]) -> None:
        atomic_write_json(self.path, {"version": 1, "items": [i.to_dict() for i in items]})

    @staticmethod
    def _next_id(items: list[QuarantineItem]) -> str:
        max_id = 0
        for item in items:
            match = re.match(r"Q(\d+)$", item.id)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return f"Q{max_id + 1:03d}"


def normalize_classification(value: str) -> str:
    """Normalize and validate memory classification."""
    normalized = (value or "confirmed").strip().lower().replace("-", "_")
    aliases = {
        "tool": "tool_observed",
        "tool-observed": "tool_observed",
        "observed": "tool_observed",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_CLASSIFICATIONS:
        raise ValueError(
            "classification must be one of: confirmed, inferred, speculative, tool_observed"
        )
    return normalized
