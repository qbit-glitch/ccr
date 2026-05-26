"""Recall replay and trace utilities.

Trace records preserve why a memory was recalled: the query, planner decision,
pre-rerank candidates, final evidence, warnings, and confidence.  This gives
CCR a local "why was this recalled?" audit trail without requiring an external
observability backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ccr.core.facts import lexical_score


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _body_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class RecallTrace:
    """One append-only recall replay record."""

    id: str
    query: str
    plan: dict[str, Any]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    stale_notes: list[str] = field(default_factory=list)
    conflict_notes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = field(default_factory=utc_now)
    prev_hash: str = ""
    hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecallTrace":
        return cls(
            id=str(data.get("id", "")),
            query=str(data.get("query", "")),
            plan=dict(data.get("plan") or {}),
            candidates=list(data.get("candidates") or []),
            evidence=list(data.get("evidence") or []),
            stale_notes=list(data.get("stale_notes") or []),
            conflict_notes=list(data.get("conflict_notes") or []),
            confidence=float(data.get("confidence", 0.0)),
            created_at=str(data.get("created_at") or utc_now()),
            prev_hash=str(data.get("prev_hash", "")),
            hash=str(data.get("hash", "")),
        )


class RecallTraceStore:
    """Append-only JSONL recall trace store."""

    def __init__(self, ccr_root: str):
        self.ccr_root = ccr_root
        self.path = os.path.join(ccr_root, "recall_traces.jsonl")

    def append_trace(
        self,
        *,
        query: str,
        plan: dict[str, Any],
        candidates: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        stale_notes: list[str],
        conflict_notes: list[str],
        confidence: float,
    ) -> RecallTrace:
        """Append a recall trace and return the stored record."""
        traces = list(reversed(self.list_traces(limit=100000)))
        prev_hash = traces[-1].hash if traces else ""
        trace = RecallTrace(
            id=self._next_id(traces),
            query=query,
            plan=plan,
            candidates=self._trim_candidates(candidates),
            evidence=self._trim_candidates(evidence),
            stale_notes=list(stale_notes),
            conflict_notes=list(conflict_notes),
            confidence=max(0.0, min(1.0, float(confidence))),
            prev_hash=prev_hash,
        )
        data = trace.to_dict()
        data["hash"] = _body_hash(data)
        trace.hash = data["hash"]
        os.makedirs(self.ccr_root, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, sort_keys=True))
            fh.write("\n")
        return trace

    def list_traces(self, query: str = "", limit: int = 25) -> list[RecallTrace]:
        """Return recent recall traces, optionally query-filtered."""
        if not os.path.isfile(self.path):
            return []
        traces: list[RecallTrace] = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(RecallTrace.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        if query.strip():
            scored = [
                (lexical_score(query, f"{t.id} {t.query} {json.dumps(t.plan, sort_keys=True)}"), t)
                for t in traces
            ]
            traces = [t for score, t in scored if score > 0 or query.lower() == t.id.lower()]
            traces.sort(
                key=lambda t: lexical_score(query, f"{t.id} {t.query} {json.dumps(t.plan, sort_keys=True)}"),
                reverse=True,
            )
        else:
            traces = list(reversed(traces))
        return traces[: max(1, limit)]

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the recall trace hash chain."""
        errors: list[str] = []
        prev_hash = ""
        if not os.path.isfile(self.path):
            return True, []
        with open(self.path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {line_no}: invalid json: {exc}")
                    continue
                trace = RecallTrace.from_dict(data)
                if trace.prev_hash != prev_hash:
                    errors.append(f"line {line_no}: prev_hash mismatch")
                if trace.hash != _body_hash(data):
                    errors.append(f"line {line_no}: hash mismatch")
                prev_hash = trace.hash
        return not errors, errors

    @staticmethod
    def _next_id(traces: list[RecallTrace]) -> str:
        max_id = 0
        for trace in traces:
            match = re.match(r"R(\d+)$", trace.id)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return f"R{max_id + 1:03d}"

    @staticmethod
    def _trim_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trimmed: list[dict[str, Any]] = []
        for candidate in candidates[:25]:
            copy = dict(candidate)
            snippet = str(copy.get("snippet", ""))
            if len(snippet) > 500:
                copy["snippet"] = snippet[:497].rstrip() + "..."
            trimmed.append(copy)
        return trimmed
