"""OTAMixin — Observation-Thought-Action logging.

Methods: log_ota, _format_ota_log, _get_next_ota_id, _get_ota_slice_since_last_commit.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class OTAMixin:
    """OTA (Observation-Thought-Action) logging for MemoryManager.

    Implements GCC paper OTA triple format for tool usage tracking.
    """

    def log_ota(
        self,
        tool_name: str,
        file_path: str = "",
        status: str = "OK",
        observation: str = "",
        thought: str = "",
        action: str = "",
    ) -> None:
        """Append an OTA log entry to the active branch's log.

        If observation/thought/action are provided, uses the full GCC paper
        OTA triple format. Otherwise falls back to simple table format.
        """
        branch = self.get_active_branch()
        line = self._format_ota_log(tool_name, file_path, status, observation, thought, action)
        self._append_log(branch, line)

    def _format_ota_log(
        self,
        tool_name: str,
        file_path: str = "",
        status: str = "OK",
        observation: str = "",
        thought: str = "",
        action: str = "",
    ) -> str:
        """Format an OTA log entry in the GCC paper triple format."""
        branch = self.get_active_branch()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        ota_id = self._get_next_ota_id(branch)

        if observation or thought or action:
            # Full OTA triple format (GCC paper spec)
            return (
                f"---\n"
                f"**[{ota_id}]** {now} | Branch: {branch}\n"
                f"- **Observation**: {observation or f'{tool_name} on {file_path}'}\n"
                f"- **Thought**: {thought or 'Processing'}\n"
                f"- **Action**: {action or f'{tool_name} ({status})'}\n"
            )
        # Simple table format (backward compat)
        return f"| {now} | {tool_name} | {file_path or '-'} | {status} |"

    def _get_next_ota_id(self, branch: str) -> str:
        """Get the next OTA sequential ID for a branch (M2: cached to avoid O(n) scan)."""
        if not hasattr(self, '_ota_counters'):
            self._ota_counters: dict[str, int] = {}
        if branch not in self._ota_counters:
            content = self._read_file(self._get_log_path(branch))
            if content:
                matches = re.findall(r"\[OTA-(\d+)\]", content)
                self._ota_counters[branch] = max(int(m) for m in matches) if matches else 0
            else:
                self._ota_counters[branch] = 0
        self._ota_counters[branch] += 1
        return f"OTA-{self._ota_counters[branch]:03d}"

    def _get_ota_slice_since_last_commit(self, branch: str, max_entries: int = 5) -> str:
        """Get recent OTA entries since the last commit (for reference in commit)."""
        content = self._read_file(self._get_log_path(branch))
        if not content:
            return ""
        entries = re.split(r"(?=---\n\*\*\[OTA-)", content)
        entries = [e.strip() for e in entries if e.strip()]
        recent = entries[-max_entries:] if entries else []
        if not recent:
            return ""
        # Compact: just IDs and observations
        refs = []
        for entry in recent:
            id_match = re.search(r"\[OTA-(\d+)\]", entry)
            obs_match = re.search(r"\*\*Observation\*\*: (.+)", entry)
            if id_match:
                ota_id = f"OTA-{id_match.group(1)}"
                obs = obs_match.group(1)[:80] if obs_match else ""
                refs.append(f"{ota_id}: {obs}")
        return "; ".join(refs) if refs else ""
