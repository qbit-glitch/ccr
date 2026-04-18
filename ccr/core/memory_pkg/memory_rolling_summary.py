"""RollingSummaryMixin — Rolling summary management for MemoryManager.

Methods:
    _get_rolling_summary
    _update_rolling_summary
    _mechanical_compress_summary (staticmethod)
    _structured_truncate_summary (staticmethod)
    _write_rolling_summary
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


class RollingSummaryMixin:
    """Rolling summary management methods for MemoryManager."""

    def _get_rolling_summary(self, branch: str) -> str:
        """Read the current rolling summary S_{t-1} from commits.md."""
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return ""
        match = re.search(r"## Rolling Summary\n(.*?)(?=\n---|\n# |\Z)", content, re.DOTALL)
        if match:
            summary = match.group(1).strip()
            if summary == "(none yet)":
                return ""
            return summary
        return ""

    def _update_rolling_summary(
        self, branch: str, what: str, why: str, next_step: str,
        compressed_summary: str | None = None,
    ) -> None:
        """Regenerate rolling summary: S_t = f(S_{t-1}, D_t).

        Per the GCC paper: each commit regenerates a coarse-grained summary
        combining the previous summary with the new contribution. This creates
        a progressively refined chain that captures the full branch history
        in a compact form — no need to re-read all individual commits.

        Three strategies in priority order:
        1. If compressed_summary is provided (by Claude Code via two-call pattern),
           use it directly — this restores the GCC paper's LLM-compressed S_t.
        2. If sub_client is available, use LLM to compress (legacy sub-model path).
        3. Fallback: concatenation with structured truncation that preserves the
           first sentence (project context) and last 3 entries, instead of blind
           tail truncation.

        Args:
            compressed_summary: Optional LLM-compressed summary provided by the
                caller (e.g., Claude Code responding to the compression prompt).
                When provided, replaces the entire rolling summary.
        """
        # Strategy 1: Caller-provided compressed summary (two-call pattern)
        # This is how MCP mode restores the GCC paper's S_t = f(S_{t-1}, D_t)
        # property — Claude Code IS the LLM that compresses the summary.
        if compressed_summary is not None:
            self._write_rolling_summary(branch, compressed_summary.strip()[:1500])
            return

        previous_summary = self._get_rolling_summary(branch)
        new_contribution = f"{what} (because: {why}). Next: {next_step}"

        # Strategy 2: Sub-client LLM compression (legacy, not used in MCP mode)
        if self.sub_client is not None:
            try:
                prompt = (
                    f"Compress this branch progress into a concise summary (max 300 words):\n\n"
                    f"Previous summary: {previous_summary or '(first commit)'}\n\n"
                    f"New contribution: {new_contribution}\n\n"
                    f"Output ONLY the compressed summary, no other text."
                )
                messages = [{"role": "user", "content": prompt}]
                new_summary = self.sub_client.completion(messages)
                if new_summary and len(new_summary.strip()) > 10:
                    self._write_rolling_summary(branch, new_summary.strip()[:1500])
                    return
            except Exception:
                pass  # Fall through to mechanical fallback

        # Strategy 2.5: Mechanical auto-compression when previous_summary is long
        # Fires before concatenation to prevent unbounded growth even when Claude Code
        # ignores the compression warning. The warning still fires (as a "could do better"
        # hint), but now it is not mandatory — auto-compression provides a safety net.
        if previous_summary and len(previous_summary) > 1200:
            try:
                previous_summary = self._mechanical_compress_summary(previous_summary)
            except Exception:
                pass  # Fall through with original previous_summary on any error

        # Strategy 3: Mechanical concatenation with structured truncation
        if previous_summary:
            new_summary = f"{previous_summary}; {new_contribution}"
        else:
            new_summary = new_contribution

        # Cap rolling summary length with structured truncation
        if len(new_summary) > 1500:
            new_summary = self._structured_truncate_summary(new_summary)

        self._write_rolling_summary(branch, new_summary)

    @staticmethod
    def _mechanical_compress_summary(summary: str) -> str:
        """Mechanically compress a long rolling summary by keeping anchor + tail.

        Used in Strategy 2.5 of _update_rolling_summary() as a safety net when
        the previous_summary exceeds 1200 chars and no compressed_summary was
        provided by the caller.

        Algorithm:
        1. Split on "; " delimiter (the concatenation separator)
        2. Keep the first segment (project context anchor)
        3. Keep the last 3 segments (most recent contributions)
        4. Join with "; " — result is typically well under 900 chars
        5. If still > 900 chars, fall back to tail-truncation at 900

        This is intentionally lossy for middle segments — the full LLM-compressed
        summary (via compressed_summary= parameter) is always preferred.
        """
        segments = [s.strip() for s in summary.split("; ") if s.strip()]

        if len(segments) <= 4:
            # Not enough segments to meaningfully compress — just return as-is
            # (structured truncation in Strategy 3 will handle length enforcement)
            return summary

        # Keep first (anchor) + last 3 (most recent)
        first = segments[0]
        last_three = segments[-3:]
        compressed = "; ".join([first] + last_three)

        # Safety cap at 900 chars
        if len(compressed) > 900:
            compressed = "..." + compressed[-897:]

        return compressed

    @staticmethod
    def _structured_truncate_summary(summary: str, max_chars: int = 1500) -> str:
        """Structured truncation preserving context and recency.

        Instead of blind "..." + last_1200_chars (lossy FIFO), this keeps:
        1. The FIRST sentence — captures project context / initial direction
        2. The LAST 3 semicolon-delimited entries — most recent work in full
        3. For older entries in between, only the first clause (before any
           parenthetical or period) — compressed but not lost

        This is still mechanical (no LLM) but preserves significantly more
        structure than tail truncation. For true S_t = f(S_{t-1}, D_t),
        the caller should provide compressed_summary via the two-call pattern.
        """
        if len(summary) <= max_chars:
            return summary

        # Split on semicolons (the delimiter used by concatenation)
        entries = [e.strip() for e in summary.split(";") if e.strip()]

        if len(entries) <= 3:
            # Too few entries to structure — fall back to tail truncation
            return "..." + summary[-(max_chars - 3):]

        # Keep first entry (project context) and last 3 entries in full
        first_entry = entries[0]
        last_three = entries[-3:]
        middle_entries = entries[1:-3]

        # Compress middle entries: keep only first clause
        compressed_middle = []
        for entry in middle_entries:
            # Take text before first parenthetical or period, whichever comes first
            cut = len(entry)
            paren_pos = entry.find(" (")
            period_pos = entry.find(". ")
            if paren_pos > 0:
                cut = min(cut, paren_pos)
            if period_pos > 0:
                cut = min(cut, period_pos)
            compressed = entry[:cut].rstrip(" ,;.")
            if compressed:
                compressed_middle.append(compressed)

        # Reassemble: first + compressed middle + last 3
        parts = [first_entry] + compressed_middle + last_three
        result = "; ".join(parts)

        # If still too long, progressively drop compressed middle entries
        while len(result) > max_chars and compressed_middle:
            compressed_middle.pop(0)
            parts = [first_entry] + compressed_middle + last_three
            result = "; ".join(parts)

        # Final safety: if still over budget, hard truncate (shouldn't happen often)
        if len(result) > max_chars:
            result = "..." + result[-(max_chars - 3):]

        return result

    def _write_rolling_summary(self, branch: str, summary: str) -> None:
        """Write the rolling summary into the commits.md file."""
        path = self._get_commits_path(branch)
        with self._locks[path], self._file_lock(path):
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                r"(## Rolling Summary\n).*?(?=\n---|\n# |\Z)",
                lambda m: f"{m.group(1)}{summary}\n",
                content,
                count=1,
                flags=re.DOTALL,
            )
            self._write_file_unlocked(path, content)
        self._invalidate_commit_index(branch)
