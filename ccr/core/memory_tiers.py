"""Explicit CCR memory tier inspection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from ccr.core.facts import FactLedger


TIER_RULES = {
    "scratchpad": "ephemeral working memory; expires or is cleared",
    "session": "searchable Q&A/tool-turn history",
    "commit": "versioned project milestones and decisions",
    "fact": "durable truths with temporal validity and provenance",
    "pattern": "recurring reusable project behaviors",
    "playbook": "promoted strategies with helpful/harmful feedback",
}


@dataclass
class TierSnapshot:
    tier: str
    count: int
    rule: str
    examples: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "count": self.count,
            "rule": self.rule,
            "examples": self.examples,
        }


class MemoryTierInspector:
    def __init__(self, memory_manager: Any):
        self.mem = memory_manager
        self.ccr_root = memory_manager.ccr_root

    def snapshot(self) -> list[TierSnapshot]:
        return [
            self._scratchpad(),
            self._sessions(),
            self._commits(),
            self._facts(),
            self._patterns(),
            self._playbook(),
        ]

    def to_markdown(self) -> str:
        lines = [
            "# CCR Memory Tiers",
            "",
            "`scratchpad -> session -> commit -> fact -> pattern -> playbook`",
            "",
            "| Tier | Count | Rule | Examples |",
            "|------|------:|------|----------|",
        ]
        for snap in self.snapshot():
            examples = ", ".join(snap.examples[:3]) or "-"
            lines.append(f"| {snap.tier} | {snap.count} | {snap.rule} | {examples} |")
        return "\n".join(lines)

    def _scratchpad(self) -> TierSnapshot:
        path = os.path.join(self.ccr_root, "scratchpad.json")
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.loads(fh.read() or "{}")
            except (OSError, json.JSONDecodeError):
                data = {}
        entries = data.get("entries", data if isinstance(data, dict) else {})
        keys = list(entries.keys()) if isinstance(entries, dict) else []
        return TierSnapshot("scratchpad", len(keys), TIER_RULES["scratchpad"], keys[:3])

    def _sessions(self) -> TierSnapshot:
        path = os.path.join(self.ccr_root, "sessions.db")
        if not os.path.isfile(path):
            return TierSnapshot("session", 0, TIER_RULES["session"], [])
        try:
            import sqlite3
            from ccr.core.session_store import SessionStore
            store = SessionStore(path)
            rows = store.search_turns("", limit=3)
            conn = sqlite3.connect(path)
            count = int(conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0])
            conn.close()
            examples = [str(r.get("user_message") or r.get("snippet_user") or "")[:60] for r in rows]
            return TierSnapshot("session", count, TIER_RULES["session"], examples)
        except Exception:
            return TierSnapshot("session", 0, TIER_RULES["session"], [])

    def _commits(self) -> TierSnapshot:
        branch = self.mem.get_active_branch()
        try:
            commits = self.mem._storage.commit_list(branch, limit=3)
            count = len(self.mem._storage.commit_list(branch, limit=100000))
        except Exception:
            commits = []
            count = 0
        examples = [f"{c.get('id', '?')}: {c.get('title', '')}" for c in commits]
        return TierSnapshot("commit", count, TIER_RULES["commit"], examples)

    def _facts(self) -> TierSnapshot:
        ledger = FactLedger(self.ccr_root)
        facts = ledger.list_facts(include_inactive=True, limit=100000)
        examples = [f"{f.id}: {f.key}" for f in facts[:3]]
        return TierSnapshot("fact", len(facts), TIER_RULES["fact"], examples)

    def _patterns(self) -> TierSnapshot:
        try:
            data = self.mem._storage.pattern_load_all()
            patterns = data.get("patterns", {})
        except Exception:
            patterns = {}
        examples = list(patterns.keys())[:3]
        return TierSnapshot("pattern", len(patterns), TIER_RULES["pattern"], examples)

    def _playbook(self) -> TierSnapshot:
        try:
            bullets = self.mem._storage.bullet_list(scope="project")
        except Exception:
            bullets = []
        examples = [str(b.get("id", "")) for b in bullets[:3]]
        return TierSnapshot("playbook", len(bullets), TIER_RULES["playbook"], examples)
