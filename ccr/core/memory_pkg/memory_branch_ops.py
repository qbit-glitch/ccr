"""BranchOpsMixin -- create_branch and merge operations."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Template strings duplicated here to avoid cross-importing from memory_types.
# Canonical copies live in memory_types.py; keep in sync if templates change.
COMMITS_TEMPLATE = """# Branch: {branch}

## Purpose
{purpose}

## Hypothesis
{hypothesis}

## Conclusion
(Fill in at merge time — success/failure/partial)

## Rolling Summary
(none yet)

---

# Milestone Journal

"""

SUMMARY_TEMPLATE = """# Branch: {branch}

## Purpose
{purpose}

## Status
active

## Parent
{parent}

## Created
{date}

## Key Decisions
(none yet)
"""


class BranchOpsMixin:
    """Branch create/merge operations for MemoryManager.

    Expects the composite class to provide:
        self.branches_dir: str
        self.ccr_root: str
        self.config: CCRConfig
        self._locks: dict
        self.get_active_branch() -> str
        self.get_context(level, branch) -> str
        self._read_file(path) -> str
        self._write_file(path, content)
        self._read_file_unlocked(path) -> str
        self._write_file_unlocked(path, content)
        self._get_commits_path(branch) -> str
        self._get_log_path(branch) -> str
        self._get_rolling_summary(branch) -> str
        self._write_rolling_summary(branch, text)
        self._structured_truncate_summary(text) -> str
        self._load_metadata() -> dict
        self._update_registry_active_branch(branch)
        self._add_branch_to_registry(name, date)
        self._update_registry_branch_status(name, status)
        self._add_branch_to_main_md(name)
        self._remove_branch_from_main_md(name)
        self._update_metadata_branch(name, status, date, parent)
        self._update_metadata_branch_status(name, status)
        self._update_summary_status(branch, status)
        self._append_log(branch, line)
        self._format_ota_log(...) -> str
        self._git_commit(message) -> bool
        self._prepend_commit(branch, entry)
        self._get_next_commit_id(branch) -> str
        self._update_main_milestones(date, branch, title)
        self._update_branch_conclusion(branch, outcome, conclusion)
        self._parse_recent_commit_data(branch, k) -> list[dict]
        self._get_branch_header(branch) -> str
        self.generate_phase_summary(branch_name, trigger) -> str
        COMMITS_TEMPLATE: str
        SUMMARY_TEMPLATE: str
    """

    def create_branch(
        self,
        name: str,
        purpose: str,
        hypothesis: str,
        linked_issue: str = "",
        team_owner: str = "",
        priority: str = "",
    ) -> str:
        """Create a new exploration branch. Must be on main."""
        # Validate kebab-case
        if not re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", name):
            raise ValueError(f"Branch name must be kebab-case: {name}")

        active = self.get_active_branch()
        if active != "main":
            raise ValueError(f"Must be on main to create branch. Currently on: {active}")

        branch_dir = os.path.join(self.branches_dir, name)
        # Allow re-creating a branch that was previously merged
        if os.path.isdir(branch_dir):
            meta = self._load_metadata()
            for b in meta.get("branches", []):
                if b.get("name") == name and b.get("status") == "merged":
                    break
            else:
                raise ValueError(f"Branch already exists: {name}")

        os.makedirs(branch_dir, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_full = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        # Write commits.md with header
        self._write_file(
            os.path.join(branch_dir, "commits.md"),
            COMMITS_TEMPLATE.format(branch=name, purpose=purpose, hypothesis=hypothesis),
        )
        self._write_file(os.path.join(branch_dir, "log.md"), "")

        # Write summary.md (GCC paper requirement)
        self._write_file(
            os.path.join(branch_dir, "summary.md"),
            SUMMARY_TEMPLATE.format(branch=name, purpose=purpose, parent="main", date=now),
        )

        # Update registry
        self._update_registry_active_branch(name)
        self._add_branch_to_registry(name, now)

        # Update main.md open branches
        self._add_branch_to_main_md(name)

        # Update metadata.yaml
        self._update_metadata_branch(name, "active", now, "main")

        # Store optional metadata fields (linked_issue, team_owner, priority)
        if linked_issue or team_owner or priority:
            meta = self._load_metadata()
            for b in meta.get("branches", []):
                if b.get("name") == name:
                    if linked_issue:
                        b["linked_issue"] = linked_issue
                    if team_owner:
                        b["team_owner"] = team_owner
                    if priority:
                        b["priority"] = priority
                    break
            meta_path = os.path.join(self.ccr_root, "metadata.yaml")
            try:
                import yaml as _yaml
                self._write_file(meta_path, _yaml.dump(meta, default_flow_style=False))
            except Exception:
                pass  # Metadata enrichment is supplementary

        # Log with OTA format
        self._append_log(name, self._format_ota_log(
            "branch-create", name, "OK",
            observation=f"Creating exploration branch '{name}'",
            thought=f"Hypothesis: {hypothesis}",
            action=f"Created branch with purpose: {purpose}",
        ))

        return f"Created branch '{name}' — purpose: {purpose}"

    def merge(
        self,
        branch_name: str,
        outcome: str,
        conclusion: str,
        allow_custom_outcome: bool = False,
    ) -> str:
        """Merge a branch back into main."""
        if branch_name == "main":
            raise ValueError("Cannot merge main into itself.")
        if not allow_custom_outcome and outcome not in ("success", "failure", "partial"):
            raise ValueError(f"Outcome must be success/failure/partial, got: {outcome}")

        active = self.get_active_branch()
        if active != branch_name:
            raise ValueError(f"Must be on branch '{branch_name}' to merge. Currently on: {active}")

        # Auto-CONTEXT before merge (GCC paper requirement)
        _pre_merge_context = self.get_context(level=3, branch=branch_name)

        # Update summary.md status
        self._update_summary_status(branch_name, f"merged ({outcome})")

        # Update branch conclusion
        self._update_branch_conclusion(branch_name, outcome, conclusion)

        # Create merge commit on main
        commit_id = self._get_next_commit_id("main")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        merge_entry = (
            f"## [{commit_id}] {now} | branch:main | Merge: {branch_name} ({outcome})\n"
            f"**What**: Merged exploration branch '{branch_name}'. {conclusion}\n"
            f"**Why**: Consolidate findings from exploration.\n"
            f"**Files**: .ccr/branches/{branch_name}/\n"
            f"**Next**: Continue on main with findings applied.\n\n---\n\n"
        )
        self._prepend_commit("main", merge_entry)
        self._update_main_milestones(now, "main", f"Merge: {branch_name} ({outcome})")

        # Integrate branch summary into main (per GCC paper F_merge)
        branch_summary = self._get_rolling_summary(branch_name)
        if branch_summary:
            main_summary = self._get_rolling_summary("main")
            merged = f"{main_summary}; [From {branch_name}]: {branch_summary}" if main_summary else f"[From {branch_name}]: {branch_summary}"
            if len(merged) > 1500:
                merged = self._structured_truncate_summary(merged)
            self._write_rolling_summary("main", merged)

        # Copy branch log to main with provenance (per GCC paper: H_{t+1} = H_t union H_t^(b))
        branch_log = self._read_file(self._get_log_path(branch_name))
        if branch_log and branch_log.strip():
            provenance_header = f"\n---\n# [Merged from {branch_name}] ---\n"
            self._append_log("main", provenance_header + branch_log.strip())

        # Switch back to main
        self._update_registry_active_branch("main")
        self._update_registry_branch_status(branch_name, "merged")
        self._remove_branch_from_main_md(branch_name)

        # Update metadata.yaml
        self._update_metadata_branch_status(branch_name, "merged")

        # Log to both with OTA format
        self._append_log(branch_name, self._format_ota_log(
            "merge", f"\u2192 main ({outcome})", "OK",
            observation=f"Branch '{branch_name}' ready for merge",
            thought=f"Outcome: {outcome}. {conclusion}",
            action=f"Merged into main, switching back",
        ))
        self._append_log("main", self._format_ota_log(
            "merge", f"\u2190 {branch_name} ({outcome})", "OK",
            observation=f"Merging exploration results from '{branch_name}'",
            thought=f"Outcome: {outcome}",
            action=f"Incorporated findings from {branch_name}",
        ))

        # Git commit integration
        self._git_commit(f"ccr: Merge {branch_name} ({outcome})")

        # TiMem §3.2: Generate phase summary on merge (L3-L4 consolidation)
        try:
            self.generate_phase_summary(branch_name=branch_name, trigger="merge")
        except Exception:
            pass  # Phase summary is supplementary, don't fail the merge

        return f"Merged '{branch_name}' into main ({outcome})"
