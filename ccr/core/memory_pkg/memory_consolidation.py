"""ConsolidationMixin -- hierarchical summary generation (TiMem-inspired)."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ConsolidationMixin:
    """Hierarchical summary consolidation for MemoryManager.

    Expects the composite class to provide:
        self.ccr_root: str
        self.branches_dir: str
        self.config: CCRConfig  (with .session_summary_interval,
            .session_summary_max_chars, .phase_summary_max_items,
            .overview_staleness_threshold)
        self._locks: dict
        self.get_active_branch() -> str
        self._read_file(path) -> str
        self._read_file_unlocked(path) -> str
        self._write_file(path, content)
        self._write_file_unlocked(path, content)
        self._get_branch_dir(branch) -> str
        self._get_commits_path(branch) -> str
        self._jaccard(a, b) -> float (staticmethod)
        self._parse_recent_commit_data(branch, k) -> list[dict]
        self._get_rolling_summary(branch) -> str
        self._get_branch_header(branch) -> str
        self._load_metadata() -> dict
        self._file_lock(path) -> context manager
    """

    def _get_summary_meta_path(self) -> str:
        return os.path.join(self.ccr_root, "summary_meta.yaml")

    def _get_summaries_path(self, branch: str) -> str:
        """Path to session summaries file for a branch."""
        return os.path.join(self._get_branch_dir(branch), "summaries.md")

    def _get_phases_path(self) -> str:
        """Path to phase summaries file."""
        return os.path.join(self.ccr_root, "summaries", "phases.md")

    def _get_overview_path(self) -> str:
        """Path to project overview file."""
        return os.path.join(self.ccr_root, "overview.md")

    @staticmethod
    def _default_summary_meta() -> dict:
        return {
            "version": 1,
            "session": {},
            "phase": {
                "last_commit_id": None,
                "last_summary_id": None,
                "last_generated": None,
            },
            "overview": {
                "last_generated": None,
                "phase_count_at_generation": 0,
            },
        }

    def _load_summary_meta(self) -> dict:
        data = self._storage.summary_meta_load()
        return data if data else self._default_summary_meta()

    def _save_summary_meta(self, data: dict) -> None:
        self._storage.summary_meta_save(data)

    def _get_commit_count(self, branch: str) -> int:
        """Count total commits on a branch via storage backend."""
        return self._storage.commit_count(branch)

    def _get_next_session_summary_id(self, branch: str) -> str:
        """Get next S### ID for session summaries on a branch."""
        return self._storage.session_summary_get_next_id(branch)

    def _get_next_phase_summary_id(self) -> str:
        """Get next P### ID for phase summaries."""
        return self._storage.phase_summary_get_next_id()

    def _read_session_summaries(self, branch: str, count: int = 3) -> list[dict]:
        """Read last `count` session summaries via storage backend.

        Returns list of dicts with: id, start_date, end_date, branch,
        commits, accomplished, files, direction.
        """
        records = self._storage.session_summary_list(branch, count)
        for r in records:
            r.setdefault("commits", r.get("commit_range", ""))
            r.setdefault("files", r.get("files_touched", ""))
        return records

    def _read_phase_summaries(self, count: int = 3) -> list[dict]:
        """Read last `count` phase summaries via storage backend."""
        return self._storage.phase_summary_list(count)

    # --- Session Summary Generation (TiMem §3.2 Phi_2) ---

    def _generate_session_summary(self, branch: str) -> str:
        """Generate a session summary from recent commits (mechanical consolidation).

        Corresponds to TiMem's Phi_2: consolidate L1 (raw commits) into L2 (session).
        Uses template-based extraction + Jaccard dedup (w_i=3 historical window per §3.2).
        """
        interval = self.config.session_summary_interval
        commits = self._parse_recent_commit_data(branch, k=interval)
        if len(commits) < 3:
            return ""  # Not enough data for meaningful summary

        # TiMem H_i: last 3 session summaries for deduplication (Eq. 3)
        historical = self._read_session_summaries(branch, count=3)
        historical_whats: set[str] = set()
        for h in historical:
            historical_whats.update(h.get("accomplished", "").lower().split("; "))

        # Extract and deduplicate what statements (within batch + against historical)
        whats = []
        seen_whats: set[str] = set()
        for c in commits:
            what_text = c.get("what", "").strip()
            if not what_text:
                continue
            what_words = set(what_text.lower().split())
            # Check against historical session summaries (TiMem H_i dedup)
            is_duplicate = False
            for hist_what in historical_whats:
                if self._jaccard(what_words, set(hist_what.split())) > 0.7:
                    is_duplicate = True
                    break
            # Also dedup within current batch
            if not is_duplicate:
                for seen in seen_whats:
                    if self._jaccard(what_words, set(seen.split())) > 0.7:
                        is_duplicate = True
                        break
            if not is_duplicate:
                seen_whats.add(what_text.lower())
                whats.append(what_text)

        if not whats:
            whats = [c.get("what", "") for c in commits[:1]]  # Fallback: at least one

        # Union files
        all_files: set[str] = set()
        for c in commits:
            all_files.update(c.get("files", []))

        # Key decisions from why fields (unique)
        seen_whys: set[str] = set()
        decisions = []
        for c in commits:
            why_text = c.get("why", "").strip()
            if why_text and why_text.lower() not in seen_whys:
                seen_whys.add(why_text.lower())
                decisions.append(why_text)

        # Time span
        timestamps = [c.get("timestamp", "") for c in commits if c.get("timestamp")]
        start_date = timestamps[-1] if timestamps else "unknown"
        end_date = timestamps[0] if timestamps else "unknown"

        # Commit range
        commit_ids = [c.get("id", "") for c in commits]
        commit_range = f"{commit_ids[-1]}-{commit_ids[0]}" if commit_ids else "unknown"

        # Most recent next_step
        direction = commits[0].get("next", "") if commits else ""

        # Generate summary
        summary_id = self._get_next_session_summary_id(branch)
        accomplished = "; ".join(whats)[:self.config.session_summary_max_chars]
        files_str = ", ".join(sorted(all_files)[:15])
        decisions_str = "; ".join(decisions)[:200]

        entry = (
            f"## [{summary_id}] {start_date} - {end_date} | {branch} | Session Summary\n"
            f"**Commits**: {commit_range}\n"
            f"**Accomplished**: {accomplished}\n"
            f"**Files touched**: {files_str}\n"
            f"**Key decisions**: {decisions_str}\n"
            f"**Direction**: {direction}\n\n---\n\n"
        )

        data = {
            "id": summary_id,
            "start_date": start_date,
            "end_date": end_date,
            "commit_range": commit_range,
            "accomplished": accomplished,
            "files_touched": files_str,
            "key_decisions": decisions_str,
            "direction": direction,
        }
        self._storage.session_summary_insert(branch, data)

        return entry

    def _maybe_generate_session_summary(self, branch: str) -> str | None:
        """Check if a session summary should be generated after a commit.

        Called at the end of commit(). Checks commit count delta vs config threshold.
        """
        meta = self._load_summary_meta()
        session_meta = meta.get("session", {})
        branch_meta = session_meta.get(branch, {})
        last_count = branch_meta.get("last_commit_count", 0)
        current_count = self._get_commit_count(branch)

        if current_count - last_count < self.config.session_summary_interval:
            return None

        summary = self._generate_session_summary(branch)
        if not summary:
            return None

        # Extract the ID from the generated summary (e.g., "[S001]")
        id_match = re.search(r"\[(S\d{3,})\]", summary)
        generated_id = id_match.group(1) if id_match else "S???"

        # Update meta
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        session_meta[branch] = {
            "last_commit_count": current_count,
            "last_summary_id": generated_id,
            "last_generated": now,
        }
        meta["session"] = session_meta
        self._save_summary_meta(meta)

        return summary

    # --- Phase Summary Generation (TiMem §3.2 Phi_3-4) ---

    def generate_phase_summary(self, branch_name: str | None = None, trigger: str = "merge") -> str:
        """Generate a phase summary from session summaries + branch metadata.

        Called on merge or when commit count exceeds threshold on main.
        Corresponds to TiMem L3-L4 consolidation.
        """
        branch = branch_name or self.get_active_branch()

        # Collect session summaries since last phase summary
        all_sessions_raw = self._read_session_summaries(branch, count=50)
        meta = self._load_summary_meta()
        last_phase_generated = meta.get("phase", {}).get("last_generated")

        # Filter: only include sessions created AFTER the last phase summary
        # Without this, phase summaries accumulate stale data from prior phases
        all_sessions = all_sessions_raw
        if last_phase_generated:
            try:
                cutoff = datetime.strptime(last_phase_generated, "%Y-%m-%dT%H:%M:%S")
                filtered = []
                for s in all_sessions_raw:
                    start_str = s.get("start_date", "")
                    if start_str:
                        try:
                            sess_time = datetime.strptime(start_str.strip(), "%Y-%m-%d %H:%M")
                            if sess_time > cutoff:
                                filtered.append(s)
                        except (ValueError, TypeError):
                            filtered.append(s)  # Include if unparseable
                    else:
                        filtered.append(s)
                if filtered:  # Only use filtered if it yields results
                    all_sessions = filtered
            except (ValueError, TypeError):
                pass  # Use all sessions if cutoff is unparseable

        # If merge: include branch metadata
        branch_purpose = ""
        branch_conclusion = ""
        branch_summary = ""
        if trigger == "merge" and branch != "main":
            header = self._get_branch_header(branch)
            purpose_m = re.search(r"## Purpose\n(.*?)(?=\n##|\Z)", header, re.DOTALL)
            conclusion_m = re.search(r"## Conclusion\n(.*?)(?=\n##|\Z)", header, re.DOTALL)
            branch_purpose = purpose_m.group(1).strip() if purpose_m else ""
            branch_conclusion = conclusion_m.group(1).strip() if conclusion_m else ""
            branch_summary = self._get_rolling_summary(branch)

        # Aggregate accomplishments from session summaries
        accomplishments = []
        all_files: set[str] = set()
        total_commits = 0
        start_date = ""
        end_date = ""

        for sess in all_sessions:
            if sess.get("accomplished"):
                accomplishments.append(sess["accomplished"])
            if sess.get("files"):
                all_files.update(f.strip() for f in sess["files"].split(","))
            # Parse commit range for count
            commits_str = sess.get("commits", "")
            if "-" in commits_str:
                try:
                    c_start, c_end = commits_str.split("-")
                    s_num = int(re.search(r"\d+", c_start).group())
                    e_num = int(re.search(r"\d+", c_end).group())
                    total_commits += e_num - s_num + 1
                except (ValueError, AttributeError):
                    total_commits += 1
            if sess.get("end_date") and not end_date:
                end_date = sess["end_date"]
            if sess.get("start_date"):
                start_date = sess["start_date"]

        if not all_sessions:
            # Fallback: use raw commits if no session summaries exist yet
            recent = self._parse_recent_commit_data(branch, k=20)
            for c in recent:
                if c.get("what"):
                    accomplishments.append(c["what"])
                all_files.update(c.get("files", []))
                total_commits += 1
            if recent:
                start_date = recent[-1].get("timestamp", "")
                end_date = recent[0].get("timestamp", "")

        # Deduplicate accomplishments
        unique_accomplishments = []
        seen: set[str] = set()
        for acc in accomplishments:
            words = set(acc.lower().split())
            is_dup = False
            for s in seen:
                if self._jaccard(words, set(s.split())) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                seen.add(acc.lower())
                unique_accomplishments.append(acc)

        # Format
        phase_id = self._get_next_phase_summary_id()
        scope = branch_name or "main milestone"
        goal = branch_purpose or (unique_accomplishments[0] if unique_accomplishments else "")
        outcome = branch_conclusion or ("ongoing" if trigger != "merge" else "merged")

        items = unique_accomplishments[:self.config.phase_summary_max_items]
        acc_text = "\n".join(f"- {item}" for item in items)
        files_list = sorted(all_files)[:10]
        files_str = ", ".join(files_list)

        entry = (
            f"## [{phase_id}] {start_date} - {end_date} | Phase Summary\n"
            f"**Scope**: {scope}\n"
            f"**Commits covered**: {total_commits} commits, {len(all_sessions)} sessions\n"
            f"**Goal**: {goal}\n"
            f"**Outcome**: {outcome}\n"
            f"**Key accomplishments**:\n{acc_text}\n"
            f"**Files changed**: {files_str}\n"
        )
        if branch_summary:
            entry += f"**Branch summary**: {branch_summary[:300]}\n"
        entry += "\n---\n\n"

        data = {
            "id": phase_id,
            "start_date": start_date,
            "end_date": end_date,
            "scope": scope,
            "goal": goal,
            "outcome": outcome,
            "accomplishments": acc_text,
            "files_changed": files_str,
            "branch_summary": branch_summary[:300] if branch_summary else "",
        }
        self._storage.phase_summary_insert(data)

        # Update meta
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        # Get the actual last commit ID (not _get_next which returns one past)
        recent = self._parse_recent_commit_data(branch, k=1)
        last_cid = recent[0]["id"] if recent else None
        meta["phase"] = {
            "last_commit_id": last_cid,
            "last_summary_id": phase_id,
            "last_generated": now,
        }
        self._save_summary_meta(meta)

        return entry

    # --- Project Overview (TiMem §3.2 Phi_5, caller-driven) ---

    def get_consolidation_prompt(self, tier: str = "project") -> str:
        """Return prompt + raw material for Claude Code to generate a summary.

        For tier="project": returns phase summaries + metadata for Claude to consolidate.
        For tier="session"/"phase": generates mechanically and returns the result.
        """
        if tier == "session":
            branch = self.get_active_branch()
            result = self._generate_session_summary(branch)
            return result or "No session summary generated (insufficient commits)."

        if tier == "phase":
            result = self.generate_phase_summary(trigger="manual")
            return result or "No phase summary generated (insufficient data)."

        # tier == "project": return prompt for Claude Code
        existing_overview = self._storage.project_state_get("overview") or "(none)"

        phases = self._read_phase_summaries(count=5)
        phases_text = ""
        for p in phases:
            phases_text += (
                f"### {p.get('id', '?')} ({p.get('start_date', '?')} - {p.get('end_date', '?')})\n"
                f"Scope: {p.get('scope', '?')}\n"
                f"Goal: {p.get('goal', '?')}\n"
                f"Outcome: {p.get('outcome', '?')}\n"
                f"{p.get('accomplishments', '')}\n\n"
            )
        if not phases_text:
            phases_text = "(no phase summaries yet)"

        meta = self._load_metadata()
        lang = meta.get("config", {}).get("language", "unknown")
        framework = meta.get("config", {}).get("framework", "unknown")
        file_count = len(meta.get("file_tree", []))

        return (
            f"Please generate a project overview from these phase summaries and the current project state.\n\n"
            f"## Current Overview:\n{existing_overview}\n\n"
            f"## Recent Phase Summaries:\n{phases_text}\n"
            f"## Project Metadata:\n"
            f"- Language: {lang}\n"
            f"- Framework: {framework}\n"
            f"- Files indexed: {file_count}\n\n"
            f"Instructions: Write a concise project overview (max 500 words) covering:\n"
            f"1. What this project does\n"
            f"2. Key architectural decisions\n"
            f"3. Major accomplishments\n"
            f"4. Current state and direction\n"
        )

    def save_overview(self, content: str) -> None:
        """Persist a Claude-Code-generated project overview."""
        self._storage.project_state_set("overview", content)
        # Update meta
        meta = self._load_summary_meta()
        phases = self._read_phase_summaries(count=100)
        meta["overview"] = {
            "last_generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "phase_count_at_generation": len(phases),
        }
        self._save_summary_meta(meta)

    def get_summaries(self, tier: str = "all", count: int = 5) -> str:
        """Retrieve hierarchical memory summaries."""
        parts = []
        branch = self.get_active_branch()

        if tier in ("session", "all"):
            sessions = self._read_session_summaries(branch, count=count)
            if sessions:
                lines = [f"# Session Summaries ({branch})"]
                for s in sessions:
                    lines.append(
                        f"**[{s['id']}]** {s.get('start_date', '?')} - {s.get('end_date', '?')}\n"
                        f"  Commits: {s.get('commits', '?')}\n"
                        f"  Accomplished: {s.get('accomplished', '?')}\n"
                        f"  Direction: {s.get('direction', '?')}"
                    )
                parts.append("\n".join(lines))
            elif tier == "session":
                parts.append("No session summaries yet.")

        if tier in ("phase", "all"):
            phases = self._read_phase_summaries(count=count)
            if phases:
                lines = ["# Phase Summaries"]
                for p in phases:
                    lines.append(
                        f"**[{p['id']}]** {p.get('start_date', '?')} - {p.get('end_date', '?')}\n"
                        f"  Scope: {p.get('scope', '?')}\n"
                        f"  Goal: {p.get('goal', '?')}\n"
                        f"  Outcome: {p.get('outcome', '?')}"
                    )
                parts.append("\n".join(lines))
            elif tier == "phase":
                parts.append("No phase summaries yet.")

        if tier in ("project", "all"):
            overview = self._storage.project_state_get("overview")
            if overview:
                parts.append(f"# Project Overview\n{overview}")
                # Check staleness: suggest regen if many phases since last overview
                meta = self._load_summary_meta()
                phase_at_gen = meta.get("overview", {}).get("phase_count_at_generation", 0)
                current_phases = len(self._read_phase_summaries(count=100))
                delta = current_phases - phase_at_gen
                if delta >= self.config.overview_staleness_threshold:
                    parts.append(
                        f"(Overview may be stale: {delta} new phase(s) since last generation. "
                        f"Use gcc_consolidate(tier='project') to regenerate.)"
                    )
            elif tier == "project":
                parts.append("No project overview yet. Use gcc_consolidate(tier='project') to generate one.")

        return "\n\n".join(parts) if parts else "No summaries available."
