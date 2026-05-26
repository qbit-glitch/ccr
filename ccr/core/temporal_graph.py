"""Temporal graph view over CCR facts.

The graph is a lightweight derived index, not a source of truth.  Facts remain
in ``facts.json``; this module materializes entity/fact nodes and validity
edges so recall can reason about "what is true now" versus "what was believed
then" without forcing a heavyweight graph database into local projects.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from ccr.core.events import atomic_write_json
from ccr.core.facts import FactLedger, FactRecord, lexical_score


@dataclass
class GraphNode:
    id: str
    kind: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    valid_from: str = ""
    valid_to: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphHit:
    fact: FactRecord
    entity_id: str
    score: float
    relation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact": self.fact.to_dict(),
            "entity_id": self.entity_id,
            "score": self.score,
            "relation": self.relation,
        }


class TemporalGraphStore:
    """Derived graph index for fact-ledger recall."""

    def __init__(self, ccr_root: str):
        self.ccr_root = ccr_root
        self.path = os.path.join(ccr_root, "temporal_graph.json")

    def rebuild_from_facts(self, ledger: FactLedger | None = None) -> dict[str, Any]:
        """Rebuild graph data from the current fact ledger."""
        ledger = ledger or FactLedger(self.ccr_root)
        facts = ledger.list_facts(include_inactive=True, limit=100000)
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        for fact in facts:
            entity_id = f"entity:{fact.key}"
            nodes.setdefault(entity_id, GraphNode(
                id=entity_id,
                kind="entity",
                label=fact.key,
                properties={"key": fact.key},
            ))
            fact_node_id = f"fact:{fact.id}"
            nodes[fact_node_id] = GraphNode(
                id=fact_node_id,
                kind="fact",
                label=fact.statement,
                properties={
                    "fact_id": fact.id,
                    "key": fact.key,
                    "observed_at": fact.observed_at,
                    "active": fact.is_active,
                    "confidence": fact.confidence,
                    "source_commit": fact.source_commit,
                    "source_session": fact.source_session,
                    "source_file": fact.source_file,
                    "classification": getattr(fact, "classification", "confirmed"),
                },
            )
            edges.append(GraphEdge(
                source=entity_id,
                target=fact_node_id,
                relation="has_fact",
                valid_from=fact.valid_from or fact.observed_at,
                valid_to=fact.valid_to,
                properties={
                    "superseded_by": fact.superseded_by,
                    "pinned": fact.pinned,
                    "evidence_ids": list(getattr(fact, "evidence_ids", []) or []),
                    "episode_ids": list(getattr(fact, "episode_ids", []) or []),
                },
            ))
            if fact.superseded_by:
                edges.append(GraphEdge(
                    source=fact_node_id,
                    target=f"fact:{fact.superseded_by}",
                    relation="superseded_by",
                    valid_from=fact.valid_to or fact.observed_at,
                    properties={"source_fact": fact.id},
                ))
        data = {
            "version": 1,
            "nodes": [n.to_dict() for n in nodes.values()],
            "edges": [e.to_dict() for e in edges],
        }
        atomic_write_json(self.path, data)
        return data

    def search(self, query: str, limit: int = 10) -> list[GraphHit]:
        """Search graph-backed facts by query relevance."""
        ledger = FactLedger(self.ccr_root)
        self.rebuild_from_facts(ledger)
        facts = ledger.list_facts(query=query, include_inactive=True, limit=100000)
        hits: list[GraphHit] = []
        for fact in facts:
            relation = "active_fact" if fact.is_active else "stale_fact"
            score = lexical_score(
                query,
                f"{fact.id} {fact.key} {fact.statement} {fact.source_commit} {fact.source_file}",
            )
            if score <= 0 and query.strip().lower() != fact.id.lower():
                continue
            temporal_boost = 0.15 if any(
                word in query.lower()
                for word in ("current", "then", "when", "changed", "stale", "superseded", "valid")
            ) else 0.0
            hits.append(GraphHit(
                fact=fact,
                entity_id=f"entity:{fact.key}",
                score=min(1.0, (score * max(0.1, fact.confidence)) + temporal_boost),
                relation=relation,
            ))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(1, limit)]

    def load(self) -> dict[str, Any]:
        """Load the materialized graph, returning an empty graph if absent."""
        if not os.path.isfile(self.path):
            return {"version": 1, "nodes": [], "edges": []}
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.loads(fh.read() or "{}")
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "nodes": [], "edges": []}
        if not isinstance(data, dict):
            return {"version": 1, "nodes": [], "edges": []}
        data.setdefault("version", 1)
        data.setdefault("nodes", [])
        data.setdefault("edges", [])
        return data
