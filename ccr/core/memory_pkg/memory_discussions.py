"""DiscussionsMixin — persistent decision and hypothesis log per branch.

Stores discussion/decision records to .ccr/branches/{branch}/discussions.md.
Each record captures a topic, hypothesis, alternatives considered, decision,
rationale, uncertainty, and optional link to a commit.

Discussion block format:
    ## [D001] 2026-04-05 10:30 | dataset preprocessing approach
    **Hypothesis**: Use TorchDataset for batch loading
    **Alternatives**: pandas CSV loader, HDF5
    **Decision**: TorchDataset
    **Rationale**: 40% throughput gain in profiling (train.py benchmark)
    **Uncertainty**: Not tested on datasets >10GB
    **Linked Commit**: C045

    ---
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_DISCUSSION_HEADER = re.compile(
    r"^## \[(D\d{3,})\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \| (.+)$",
    re.MULTILINE,
)


def _discussions_path(ccr_root: str, branch: str) -> str:
    return os.path.join(ccr_root, "branches", branch, "discussions.md")


def _next_discussion_id(existing_text: str) -> str:
    """Find the highest D### ID and return the next one."""
    ids = [int(m) for m in re.findall(r"\[D(\d{3,})\]", existing_text)]
    next_num = (max(ids) + 1) if ids else 1
    return f"D{next_num:03d}"


def _parse_discussion_blocks(text: str) -> list[dict[str, Any]]:
    """Parse discussions.md and return a list of discussion dicts."""
    parts = _DISCUSSION_HEADER.split(text)
    results = []
    idx = 1
    while idx + 2 < len(parts):
        disc_id = parts[idx]
        disc_date = parts[idx + 1]
        topic_and_rest = parts[idx + 2]
        idx += 3

        # First line of body after the header is already consumed by the split group
        # topic is captured in group 3 of _DISCUSSION_HEADER — but the split
        # puts the header parts interleaved, so topic is in parts[idx-1].
        # Actually: _DISCUSSION_HEADER has 3 groups, so split gives:
        # [pre, id, date, topic, body_after_header, id2, date2, topic2, body2, ...]
        # We need to re-check: groups = (D###, date, topic)
        # Already corrected: disc_id=D###, disc_date=date, body includes the topic line
        # The topic was captured as group 3. Since split with 3 groups interleaves:
        # parts = [pre, g1, g2, g3, rest_after_match, g1_next, ...]
        # So we need 4 items per match: [id, date, topic, body]
        break  # Will redo below

    # Redo with correct group count
    results = []
    parts = _DISCUSSION_HEADER.split(text)
    # groups: D###, date, topic → 3 groups → 4 parts per match (pre + 3 groups) but
    # split inserts groups inline: [pre, g1_0, g2_0, g3_0, rest_0, g1_1, ...]
    idx = 1
    while idx + 3 <= len(parts):
        disc_id = parts[idx].strip()
        disc_date = parts[idx + 1].strip()
        topic = parts[idx + 2].strip()
        body = parts[idx + 3] if idx + 3 < len(parts) else ""
        idx += 4

        rec: dict[str, Any] = {
            "id": disc_id,
            "date": disc_date,
            "topic": topic,
            "hypothesis": "",
            "alternatives": "",
            "decision": "",
            "rationale": "",
            "uncertainty": "",
            "linked_commit": "",
        }

        for field, pattern in [
            ("hypothesis",   r"\*\*Hypothesis\*\*:\s*(.+)"),
            ("alternatives", r"\*\*Alternatives\*\*:\s*(.+)"),
            ("decision",     r"\*\*Decision\*\*:\s*(.+)"),
            ("rationale",    r"\*\*Rationale\*\*:\s*(.+)"),
            ("uncertainty",  r"\*\*Uncertainty\*\*:\s*(.+)"),
            ("linked_commit",r"\*\*Linked Commit\*\*:\s*(.+)"),
        ]:
            m = re.search(pattern, body)
            if m:
                rec[field] = m.group(1).strip()

        results.append(rec)

    return results


class DiscussionsMixin:
    """Persistent decision and hypothesis log stored in discussions.md."""

    def _get_discussions_path(self, branch: str | None = None) -> str:
        if branch is None:
            branch = self.get_active_branch()
        return _discussions_path(self.ccr_root, branch)

    def add_discussion(
        self,
        topic: str,
        hypothesis: str,
        alternatives_considered: str,
        decision: str,
        rationale: str,
        uncertainty: str = "",
        linked_commit: str | None = None,
    ) -> dict[str, Any]:
        """Append a discussion record via storage backend.

        Returns dict with id, date, topic, message (formatted block).
        """
        branch = self.get_active_branch()
        disc_id = self._storage.discussion_get_next_id(branch)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        data = {
            "id": disc_id,
            "timestamp": now,
            "topic": topic,
            "hypothesis": hypothesis,
            "alternatives": alternatives_considered,
            "decision": decision,
            "rationale": rationale,
            "uncertainty": uncertainty,
            "linked_commit": linked_commit or "",
        }
        self._storage.discussion_insert(branch, data)

        lines = [
            f"## [{disc_id}] {now} | {topic}",
            f"**Hypothesis**: {hypothesis}",
            f"**Alternatives**: {alternatives_considered}",
            f"**Decision**: {decision}",
            f"**Rationale**: {rationale}",
        ]
        if uncertainty:
            lines.append(f"**Uncertainty**: {uncertainty}")
        if linked_commit:
            lines.append(f"**Linked Commit**: {linked_commit}")

        return {
            "id": disc_id,
            "date": now,
            "topic": topic,
            "message": "\n".join(lines),
        }

    def get_discussions(
        self,
        search: str | None = None,
        topic: str | None = None,
        date_range: list[str] | None = None,
    ) -> dict[str, Any]:
        """Query discussion records via storage backend.

        Args:
            search: Full-text substring match (case-insensitive) across all fields.
            topic: Exact topic match.
            date_range: [start_date, end_date] as "YYYY-MM-DD" strings.

        Returns:
            dict with count, records, message (formatted markdown).
        """
        branch = self.get_active_branch()
        records = self._storage.discussion_list(branch, search, topic, date_range)

        for r in records:
            if "timestamp" in r and "date" not in r:
                r["date"] = r["timestamp"]

        if not records:
            msg = "No discussions logged yet." if not (search or topic or date_range) else "No matching discussions found."
            return {"count": 0, "records": [], "message": msg}

        lines = [
            "| ID | Date | Topic | Decision | Rationale |",
            "|----|------|-------|----------|-----------|",
        ]
        for r in records:
            lines.append(
                f"| {r['id']} | {r.get('date', r.get('timestamp', ''))} | {r.get('topic', '')[:50]} "
                f"| {r.get('decision', '')[:50]} | {r.get('rationale', '')[:60]} |"
            )

        return {"count": len(records), "records": records, "message": "\n".join(lines)}
