"""Evidence-first recall over CCR memory sources.

``gcc_context`` is optimized for loading useful context. This module is
optimized for auditable recall: a compact answer, explicit evidence IDs, and
warnings about stale or conflicting facts.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ccr.core.facts import FactLedger, FactRecord, lexical_score, tokenize
from ccr.core.observability import ccr_span
from ccr.core.replay import RecallTraceStore
from ccr.core.rerankers import get_reranker
from ccr.core.temporal_graph import TemporalGraphStore


@dataclass
class RecallEvidence:
    """One cited memory item used by recall."""

    id: str
    source: str
    title: str
    snippet: str
    score: float
    path: str = ""
    timestamp: str = ""
    stale: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecallPlan:
    """Planner decision for one recall query."""

    intent: str
    sources: list[str]
    strategy: str
    reranker: str = "pending"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecallResult:
    """Structured result returned by the recall engine."""

    query: str
    answer: str
    confidence: float
    evidence: list[RecallEvidence]
    stale_notes: list[str] = field(default_factory=list)
    conflict_notes: list[str] = field(default_factory=list)
    plan: RecallPlan | None = None
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "stale_notes": list(self.stale_notes),
            "conflict_notes": list(self.conflict_notes),
            "plan": self.plan.to_dict() if self.plan else {},
            "trace_id": self.trace_id,
            "message": self.to_markdown(),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Evidence-First Recall",
            f"**Query**: {self.query}",
            f"**Confidence**: {self.confidence:.2f}",
        ]
        if self.plan:
            lines.append(
                f"**Plan**: {self.plan.intent} via {self.plan.strategy} "
                f"({', '.join(self.plan.sources)})"
            )
        if self.trace_id:
            lines.append(f"**Trace**: {self.trace_id}")
        lines.extend([
            "",
            "## Answer",
            self.answer,
        ])
        if self.evidence:
            lines.extend(["", "## Evidence"])
            for ev in self.evidence:
                stale = " [stale]" if ev.stale else ""
                where = f" ({ev.path})" if ev.path else ""
                ts = f" | {ev.timestamp}" if ev.timestamp else ""
                refs = _format_source_refs(ev.metadata)
                refs_text = f" | refs: {refs}" if refs else ""
                lines.append(
                    f"- **[{ev.id}]** {ev.source}{stale}{where}{ts} "
                    f"score={ev.score:.3f}{refs_text}: {ev.title}"
                )
                if ev.snippet:
                    lines.append(f"  {ev.snippet}")
        if self.stale_notes:
            lines.extend(["", "## Stale Notes"])
            lines.extend(f"- {note}" for note in self.stale_notes)
        if self.conflict_notes:
            lines.extend(["", "## Conflict Notes"])
            lines.extend(f"- {note}" for note in self.conflict_notes)
        return "\n".join(lines)


class RecallEngine:
    """Deterministic, local-first recall engine."""

    def __init__(self, memory_manager: Any):
        self.mem = memory_manager
        self.ledger = FactLedger(memory_manager.ccr_root)
        self.graph = TemporalGraphStore(memory_manager.ccr_root)
        self.traces = RecallTraceStore(memory_manager.ccr_root)

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        sources: list[str] | None = None,
        include_sessions: bool = True,
    ) -> RecallResult:
        """Return a compact evidence-backed answer for ``query``."""
        query = query.strip()
        if not query:
            raise ValueError("Recall query must not be empty.")
        limit = max(1, min(limit, 25))
        plan = self._plan(query, sources=sources, include_sessions=include_sessions)
        active_sources = set(plan.sources)
        evidence: list[RecallEvidence] = []
        candidate_snapshot: list[dict[str, Any]] = []

        project_root = getattr(self.mem, "project_root", "") or os.path.dirname(self.mem.ccr_root)
        with ccr_span(
            "ccr.memory.recall",
            project_root=project_root,
            attributes={
                "query_len": len(query),
                "limit": limit,
                "sources": ",".join(sorted(active_sources)),
                "intent": plan.intent,
                "strategy": plan.strategy,
            },
        ):
            if "facts" in active_sources:
                with ccr_span("ccr.memory.retrieve.facts", project_root=project_root):
                    evidence.extend(self._fact_evidence(query, limit=limit))
            if "graph" in active_sources:
                with ccr_span("ccr.memory.retrieve.temporal_graph", project_root=project_root):
                    evidence.extend(self._graph_evidence(query, limit=limit))
            if "commits" in active_sources:
                with ccr_span("ccr.memory.retrieve.commits", project_root=project_root):
                    evidence.extend(self._commit_evidence(query, limit=limit))
            if "discussions" in active_sources:
                with ccr_span("ccr.memory.retrieve.discussions", project_root=project_root):
                    evidence.extend(self._discussion_evidence(query, limit=limit))
            if include_sessions and "sessions" in active_sources:
                with ccr_span("ccr.memory.retrieve.sessions", project_root=project_root):
                    evidence.extend(self._session_evidence(query, limit=limit))

            evidence = self._dedupe_and_rank(evidence, limit=max(limit * 3, limit))
            candidate_snapshot = [ev.to_dict() for ev in evidence]
            with ccr_span("ccr.memory.rerank", project_root=project_root, attributes={"candidates": len(evidence)}):
                reranker = get_reranker()
                plan.reranker = reranker.provider
                evidence = reranker.rerank(query, evidence)[:limit]
            stale_notes = self._stale_notes(evidence)
            conflict_notes = self._conflict_notes(query)
            confidence = self._confidence(evidence, conflict_notes)
            answer = self._answer(query, evidence)
            trace = self.traces.append_trace(
                query=query,
                plan=plan.to_dict(),
                candidates=candidate_snapshot,
                evidence=[ev.to_dict() for ev in evidence],
                stale_notes=stale_notes,
                conflict_notes=conflict_notes,
                confidence=confidence,
            )
        return RecallResult(
            query=query,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            stale_notes=stale_notes,
            conflict_notes=conflict_notes,
            plan=plan,
            trace_id=trace.id,
        )

    def _plan(
        self,
        query: str,
        *,
        sources: list[str] | None,
        include_sessions: bool,
    ) -> RecallPlan:
        """Build a deterministic recall plan from query shape."""
        if sources:
            requested = [s.strip().lower() for s in sources if s.strip()]
            requested = [
                s for s in requested
                if s in {"facts", "graph", "commits", "discussions", "sessions"}
            ]
            if requested:
                return RecallPlan(
                    intent="custom",
                    sources=requested,
                    strategy="caller-source-filter",
                    reason="Caller supplied an explicit source filter.",
                )
        lower = query.lower()
        intent = "recall"
        strategy = "broad-then-rerank"
        source_order = ["facts", "commits", "discussions"]
        reason = "Default evidence-first recall across durable project memory."
        if any(w in lower for w in ("current", "then", "when", "changed", "stale", "superseded", "valid")):
            intent = "temporal"
            strategy = "temporal-graph-first"
            source_order = ["graph", "facts", "commits", "discussions"]
            reason = "Query asks about time, staleness, or change; graph-backed fact lifecycle evidence is prioritized."
        elif any(w in lower for w in ("conflict", "contradict", "contradiction", "disagree", "inconsistent")):
            intent = "conflict"
            strategy = "conflict-aware"
            source_order = ["graph", "facts", "commits", "discussions"]
            reason = "Query asks about contradictions; fact conflicts and temporal graph evidence are prioritized."
        elif any(w in lower for w in ("prove", "source", "cite", "evidence", "why recalled", "why was")):
            intent = "evidence"
            strategy = "citation-heavy"
            source_order = ["facts", "graph", "commits", "discussions"]
            reason = "Query asks for proof; fact and graph citations are emphasized."
        elif any(w in lower for w in ("decision", "decide", "chose", "chosen", "rationale")):
            intent = "decision"
            strategy = "decision-history"
            source_order = ["discussions", "commits", "facts", "graph"]
            reason = "Query asks about a decision; discussions and commits are emphasized."
        elif any(w in lower for w in ("how", "workflow", "command", "procedure", "runbook")):
            intent = "procedural"
            strategy = "commit-session-procedure"
            source_order = ["commits", "sessions", "facts", "discussions"]
            reason = "Query asks for a procedure; commits and sessions are emphasized."
        if include_sessions and "sessions" not in source_order:
            source_order.append("sessions")
        if not include_sessions:
            source_order = [s for s in source_order if s != "sessions"]
        return RecallPlan(
            intent=intent,
            sources=source_order,
            strategy=strategy,
            reason=reason,
        )

    def _fact_evidence(self, query: str, limit: int) -> list[RecallEvidence]:
        facts = self.ledger.list_facts(query=query, include_inactive=True, limit=limit)
        results = []
        for fact in facts:
            score = lexical_score(query, f"{fact.key} {fact.statement} {fact.source_commit}")
            if score <= 0:
                continue
            results.append(self._fact_to_evidence(fact, score))
        return results

    def _fact_to_evidence(self, fact: FactRecord, score: float) -> RecallEvidence:
        stale = bool(fact.valid_to or fact.superseded_by)
        meta = {
            "key": fact.key,
            "valid_from": fact.valid_from,
            "valid_to": fact.valid_to,
            "superseded_by": fact.superseded_by,
            "confidence": fact.confidence,
            "pinned": fact.pinned,
            "source_commit": fact.source_commit,
            "source_session": fact.source_session,
            "source_file": fact.source_file,
            "classification": getattr(fact, "classification", "confirmed"),
            "evidence_ids": list(getattr(fact, "evidence_ids", []) or []),
            "episode_ids": list(getattr(fact, "episode_ids", []) or []),
        }
        return RecallEvidence(
            id=fact.id,
            source="fact",
            title=fact.key,
            snippet=fact.statement,
            score=score * max(0.1, fact.confidence),
            path=fact.source_file,
            timestamp=fact.observed_at,
            stale=stale,
            metadata=meta,
        )

    def _graph_evidence(self, query: str, limit: int) -> list[RecallEvidence]:
        hits = self.graph.search(query, limit=limit)
        evidence: list[RecallEvidence] = []
        for hit in hits:
            fact = hit.fact
            stale = bool(fact.valid_to or fact.superseded_by)
            evidence.append(RecallEvidence(
                id=f"G:{fact.id}",
                source="graph",
                title=f"{fact.key} ({hit.relation})",
                snippet=fact.statement,
                score=hit.score,
                path=fact.source_file,
                timestamp=fact.observed_at,
                stale=stale,
                metadata={
                    "entity_id": hit.entity_id,
                    "source_fact": fact.id,
                    "key": fact.key,
                    "valid_from": fact.valid_from,
                    "valid_to": fact.valid_to,
                    "superseded_by": fact.superseded_by,
                    "confidence": fact.confidence,
                    "source_commit": fact.source_commit,
                    "source_session": fact.source_session,
                    "source_file": fact.source_file,
                    "classification": getattr(fact, "classification", "confirmed"),
                    "evidence_ids": list(getattr(fact, "evidence_ids", []) or []),
                    "episode_ids": list(getattr(fact, "episode_ids", []) or []),
                },
            ))
        return evidence

    def _commit_evidence(self, query: str, limit: int) -> list[RecallEvidence]:
        branch = self.mem.get_active_branch()
        raw_hits: list[dict[str, Any]] = []
        try:
            raw_hits = self.mem._storage.commit_search_text(branch, query, max_results=limit * 3)
        except Exception:
            raw_hits = []
        if not raw_hits:
            try:
                text = self.mem._search_commits(branch, query)
                raw_hits = self._parse_commit_blocks(text)
            except Exception:
                raw_hits = []

        results = []
        for hit in raw_hits:
            cid = str(hit.get("id", ""))
            title = str(hit.get("title", ""))
            what = str(hit.get("what", ""))
            why = str(hit.get("why", ""))
            raw_block = str(hit.get("raw_block", ""))
            text = " ".join([title, what, why, raw_block])
            score = lexical_score(query, text)
            if score <= 0 and query.lower() not in cid.lower():
                continue
            results.append(RecallEvidence(
                id=cid or "?",
                source="commit",
                title=title or cid or "commit",
                snippet=_compact_snippet(what or raw_block or why, query),
                score=max(score, 0.15 if query.lower() in cid.lower() else 0.0),
                timestamp=str(hit.get("timestamp", "")),
                metadata={"branch": branch, "files": hit.get("files", [])},
            ))
        return results

    def _discussion_evidence(self, query: str, limit: int) -> list[RecallEvidence]:
        try:
            result = self.mem.get_discussions(search=query)
        except Exception:
            return []
        records = result.get("records", []) if isinstance(result, dict) else []
        evidence = []
        for record in records[:limit]:
            text = " ".join(str(record.get(k, "")) for k in (
                "topic", "hypothesis", "decision", "rationale", "uncertainty",
            ))
            score = lexical_score(query, text)
            if score <= 0:
                continue
            rid = str(record.get("id", "D???"))
            evidence.append(RecallEvidence(
                id=rid,
                source="discussion",
                title=str(record.get("topic", rid)),
                snippet=_compact_snippet(text, query),
                score=score,
                timestamp=str(record.get("date", "")),
                metadata={"linked_commit": record.get("linked_commit", "")},
            ))
        return evidence

    def _session_evidence(self, query: str, limit: int) -> list[RecallEvidence]:
        db_path = os.path.join(self.mem.ccr_root, "sessions.db")
        if not os.path.isfile(db_path):
            return []
        try:
            from ccr.core.session_store import SessionStore
            store = SessionStore(db_path)
            turns = store.search_turns(query, limit=limit)
        except Exception:
            return []
        evidence = []
        for turn in turns:
            sid = str(turn.get("session_id", "ses"))[:12]
            turn_no = str(turn.get("turn_number", "?"))
            user_snip = str(turn.get("snippet_user") or turn.get("user_message") or "")
            asst_snip = str(turn.get("snippet_asst") or turn.get("assistant_message") or "")
            text = f"{user_snip} {asst_snip}"
            score = lexical_score(query, text)
            if score <= 0:
                continue
            evidence.append(RecallEvidence(
                id=f"{sid}#{turn_no}",
                source="session",
                title=f"session {sid} turn {turn_no}",
                snippet=_compact_snippet(text, query),
                score=score * 0.8,
                timestamp=str(turn.get("timestamp", "")),
            ))
        return evidence

    @staticmethod
    def _parse_commit_blocks(text: str) -> list[dict[str, Any]]:
        blocks = []
        for match in re.finditer(r"## \[(C\d{3,})\](.*?)(?=\n## \[C\d{3,}\]|\Z)", text, re.S):
            block = match.group(0)
            title_line = block.splitlines()[0] if block.splitlines() else ""
            title = title_line.split("|")[-1].strip() if "|" in title_line else title_line
            what_match = re.search(r"\*\*What\*\*:\s*(.*?)(?=\n\*\*|\Z)", block, re.S)
            why_match = re.search(r"\*\*Why\*\*:\s*(.*?)(?=\n\*\*|\Z)", block, re.S)
            blocks.append({
                "id": match.group(1),
                "title": title,
                "what": what_match.group(1).strip() if what_match else "",
                "why": why_match.group(1).strip() if why_match else "",
                "raw_block": block,
            })
        return blocks

    @staticmethod
    def _dedupe_and_rank(evidence: list[RecallEvidence], limit: int) -> list[RecallEvidence]:
        by_id: dict[tuple[str, str], RecallEvidence] = {}
        for ev in evidence:
            key = (ev.source, ev.id)
            if key not in by_id or ev.score > by_id[key].score:
                by_id[key] = ev
        ranked = sorted(by_id.values(), key=lambda ev: ev.score, reverse=True)
        return ranked[:limit]

    @staticmethod
    def _confidence(evidence: list[RecallEvidence], conflict_notes: list[str]) -> float:
        if not evidence:
            return 0.0
        top = evidence[0].score
        source_diversity = min(0.2, 0.05 * len({ev.source for ev in evidence}))
        penalty = 0.2 if conflict_notes else 0.0
        return max(0.0, min(1.0, top + source_diversity - penalty))

    @staticmethod
    def _answer(query: str, evidence: list[RecallEvidence]) -> str:
        if not evidence:
            return (
                "I do not have enough CCR memory evidence to answer this. "
                "Use broader search terms or add a fact/commit that records the decision."
            )
        top = evidence[0]
        if top.source in {"fact", "graph"}:
            return top.snippet
        return (
            f"The strongest available memory is {top.source} [{top.id}]: "
            f"{top.title}. {top.snippet}".strip()
        )

    @staticmethod
    def _stale_notes(evidence: list[RecallEvidence]) -> list[str]:
        notes = []
        for ev in evidence:
            if ev.stale:
                replacement = ev.metadata.get("superseded_by") or ev.metadata.get("valid_to")
                suffix = f" superseded/expired by {replacement}" if replacement else ""
                notes.append(f"{ev.id} is stale{suffix}.")
        return notes

    def _conflict_notes(self, query: str) -> list[str]:
        conflicts = self.ledger.detect_conflicts(query=query, limit=5)
        return [
            f"{c.severity} conflict on '{c.key}': {c.fact_a} vs {c.fact_b} ({c.reason})"
            for c in conflicts
        ]


def _compact_snippet(text: str, query: str, max_chars: int = 240) -> str:
    """Return a short query-aware snippet."""
    text = " ".join(str(text).split())
    if not text:
        return ""
    q_terms = tokenize(query)
    if q_terms:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sentence in sentences:
            if tokenize(sentence) & q_terms:
                text = sentence.strip()
                break
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_source_refs(metadata: dict[str, Any]) -> str:
    """Format compact provenance refs for markdown evidence lines."""
    refs: list[str] = []
    source_commit = str(metadata.get("source_commit") or "").strip()
    source_session = str(metadata.get("source_session") or "").strip()
    source_file = str(metadata.get("source_file") or "").strip()
    linked_commit = str(metadata.get("linked_commit") or "").strip()
    files = metadata.get("files") or []

    if source_commit:
        refs.append(f"commit {source_commit}")
    elif linked_commit:
        refs.append(f"commit {linked_commit}")
    if source_session:
        refs.append(f"session {source_session}")
    if source_file:
        refs.append(source_file)
    if isinstance(files, list) and files:
        refs.extend(str(f) for f in files[:3] if str(f).strip())
    return ", ".join(dict.fromkeys(refs))
