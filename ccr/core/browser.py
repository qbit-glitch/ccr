"""Static local memory browser generator."""

from __future__ import annotations

import html
import os
from typing import Any

from ccr.core.facts import FactLedger
from ccr.core.memory_tiers import MemoryTierInspector
from ccr.core.quarantine import MemoryQuarantine
from ccr.core.replay import RecallTraceStore
from ccr.core.temporal_graph import TemporalGraphStore


class MemoryBrowserBuilder:
    def __init__(self, memory_manager: Any):
        self.mem = memory_manager
        self.ccr_root = memory_manager.ccr_root

    def build_html(self) -> str:
        branch = self.mem.get_active_branch()
        commits = self._commits(branch)
        facts = FactLedger(self.ccr_root).list_facts(include_inactive=True, limit=200)
        conflicts = FactLedger(self.ccr_root).detect_conflicts(limit=200)
        tiers = MemoryTierInspector(self.mem).snapshot()
        traces = RecallTraceStore(self.ccr_root).list_traces(limit=50)
        quarantined = MemoryQuarantine(self.ccr_root).list_items(status="pending", limit=50)
        graph = TemporalGraphStore(self.ccr_root).rebuild_from_facts()
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CCR Memory Browser</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #111827; background: #f7f7f5; }}
    header {{ padding: 20px 28px; background: #ffffff; border-bottom: 1px solid #d8d8d2; }}
    main {{ padding: 20px 28px 36px; display: grid; gap: 24px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .meta {{ margin-top: 6px; color: #4b5563; font-size: 13px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: #d8d8d2; border: 1px solid #d8d8d2; }}
    .metric {{ background: #fff; padding: 14px; }}
    .metric strong {{ display: block; font-size: 22px; }}
    section {{ background: #fff; border: 1px solid #d8d8d2; padding: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: #374151; background: #fafafa; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <header>
    <h1>CCR Memory Browser</h1>
    <div class="meta">{html.escape(os.path.dirname(self.ccr_root))} · branch {html.escape(branch)}</div>
  </header>
  <main>
    <div class="metrics">
      <div class="metric"><strong>{len(commits)}</strong> commits shown</div>
      <div class="metric"><strong>{len(facts)}</strong> facts</div>
      <div class="metric"><strong>{len(conflicts)}</strong> active conflicts</div>
      <div class="metric"><strong>{sum(t.count for t in tiers)}</strong> tier records</div>
      <div class="metric"><strong>{len(traces)}</strong> recall traces</div>
      <div class="metric"><strong>{len(quarantined)}</strong> quarantined</div>
      <div class="metric"><strong>{len(graph.get('nodes', []))}</strong> graph nodes</div>
    </div>
    {self._tier_section(tiers)}
    {self._trace_section(traces)}
    {self._quarantine_section(quarantined)}
    {self._conflict_section(conflicts)}
    {self._fact_section(facts)}
    {self._graph_section(graph)}
    {self._commit_section(commits)}
  </main>
</body>
</html>
"""

    def write(self, output: str) -> str:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(self.build_html())
            fh.write("\n")
        return output

    def _commits(self, branch: str) -> list[dict[str, Any]]:
        try:
            return self.mem._storage.commit_list(branch, limit=100)
        except Exception:
            return []

    @staticmethod
    def _tier_section(tiers: list[Any]) -> str:
        rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(t.tier)}</code></td>"
            f"<td>{t.count}</td>"
            f"<td>{html.escape(t.rule)}</td>"
            f"<td>{html.escape(', '.join(t.examples[:3]) or '-')}</td>"
            "</tr>"
            for t in tiers
        )
        return f"<section><h2>Memory Tiers</h2><table><tr><th>Tier</th><th>Count</th><th>Rule</th><th>Examples</th></tr>{rows}</table></section>"

    @staticmethod
    def _conflict_section(conflicts: list[Any]) -> str:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(c.severity)}</td><td><code>{html.escape(c.key)}</code></td>"
            f"<td>{html.escape(c.fact_a)} vs {html.escape(c.fact_b)}</td>"
            f"<td>{html.escape(c.reason)}</td>"
            "</tr>"
            for c in conflicts
        ) or "<tr><td colspan='4'>No active conflicts.</td></tr>"
        return f"<section><h2>Conflicts</h2><table><tr><th>Severity</th><th>Key</th><th>Facts</th><th>Reason</th></tr>{rows}</table></section>"

    @staticmethod
    def _trace_section(traces: list[Any]) -> str:
        rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(t.id)}</code></td>"
            f"<td>{html.escape(t.created_at)}</td>"
            f"<td>{html.escape(str(t.plan.get('intent', 'recall')))}</td>"
            f"<td>{html.escape(str(t.plan.get('strategy', '-')))}</td>"
            f"<td>{t.confidence:.2f}</td>"
            f"<td>{html.escape(', '.join(str(e.get('id', '')) for e in t.evidence[:5]) or '-')}</td>"
            f"<td>{html.escape(str(t.plan.get('reason', '')))}</td>"
            f"<td>{html.escape(t.query)}</td>"
            "</tr>"
            for t in traces
        ) or "<tr><td colspan='8'>No recall traces recorded.</td></tr>"
        return (
            "<section><h2>Recall Replay: Why Was This Recalled?</h2>"
            "<table><tr><th>Trace</th><th>Time</th><th>Intent</th><th>Strategy</th>"
            "<th>Confidence</th><th>Evidence</th><th>Why</th><th>Query</th></tr>"
            f"{rows}</table></section>"
        )

    @staticmethod
    def _quarantine_section(items: list[Any]) -> str:
        rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(i.id)}</code></td>"
            f"<td>{html.escape(i.classification)}</td>"
            f"<td>{html.escape(i.key)}</td>"
            f"<td>{i.confidence:.2f}</td>"
            f"<td>{html.escape(i.statement)}</td>"
            f"<td>{html.escape(i.reason or '-')}</td>"
            "</tr>"
            for i in items
        ) or "<tr><td colspan='6'>No pending quarantine items.</td></tr>"
        return (
            "<section><h2>Memory Write Quarantine</h2>"
            "<table><tr><th>ID</th><th>Class</th><th>Key</th><th>Confidence</th><th>Statement</th><th>Reason</th></tr>"
            f"{rows}</table></section>"
        )

    @staticmethod
    def _fact_section(facts: list[Any]) -> str:
        rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(f.id)}</code></td><td>{html.escape(f.key)}</td>"
            f"<td>{html.escape(f.statement)}</td><td>{html.escape(f.source_commit or f.source_file or '-')}</td>"
            f"<td>{html.escape(getattr(f, 'classification', 'confirmed'))}</td>"
            f"<td>{html.escape(f.superseded_by or f.valid_to or 'active')}</td>"
            "</tr>"
            for f in facts
        ) or "<tr><td colspan='6'>No facts recorded.</td></tr>"
        return f"<section><h2>Facts</h2><table><tr><th>ID</th><th>Key</th><th>Statement</th><th>Source</th><th>Class</th><th>Status</th></tr>{rows}</table></section>"

    @staticmethod
    def _graph_section(graph: dict[str, Any]) -> str:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(str(n.get('id', '')))}</code></td>"
            f"<td>{html.escape(str(n.get('kind', '')))}</td>"
            f"<td>{html.escape(str(n.get('label', ''))[:180])}</td>"
            "</tr>"
            for n in nodes[:100]
        ) or "<tr><td colspan='3'>No graph nodes materialized.</td></tr>"
        return (
            f"<section><h2>Temporal Graph</h2><div class='meta'>{len(nodes)} nodes · {len(edges)} edges</div>"
            "<table><tr><th>Node</th><th>Kind</th><th>Label</th></tr>"
            f"{rows}</table></section>"
        )

    @staticmethod
    def _commit_section(commits: list[dict[str, Any]]) -> str:
        rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(str(c.get('id', '')))}</code></td>"
            f"<td>{html.escape(str(c.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(c.get('title', '')))}</td>"
            f"<td>{html.escape(str(c.get('why', ''))[:160])}</td>"
            "</tr>"
            for c in commits
        ) or "<tr><td colspan='4'>No commits recorded.</td></tr>"
        return f"<section><h2>Timeline</h2><table><tr><th>ID</th><th>Time</th><th>Title</th><th>Why</th></tr>{rows}</table></section>"
