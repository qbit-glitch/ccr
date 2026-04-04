"""GCC-style version-controlled memory management.

Port of open-gcc (TypeScript) context management to Python.
Manages the .ccr/ directory with COMMIT, BRANCH, MERGE, CONTEXT operations.

GCC paper features:
- COMMIT/BRANCH/MERGE/CONTEXT operations
- metadata.yaml (file trees, deps, config)
- summary.md per branch
- OTA logging (Observation-Thought-Action triples)
- Context windowing (scrollable K-window)
- Auto-CONTEXT before MERGE
- Log rotation
"""

from __future__ import annotations

import fcntl
import heapq
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

import yaml

from ccr.context.embeddings import get_embedding_model, load_embeddings, quick_cosine, save_embeddings
from ccr.core.types import CCRConfig, CommitLink
from ccr.utils.parsing import extract_json_string


@dataclass
class EvolvedSummary:
    """Mutable overlay for a commit's summary (A-MEM §3.3 Eq.7)."""

    commit_id: str
    evolved_what: str        # LLM-rewritten version of original what
    evolution_reason: str    # why the summary was evolved
    evolved_at: str          # ISO timestamp
    source_commit_id: str    # the new commit that triggered evolution
    original_what: str       # preserved original

# --- Templates (from open-gcc bootstrap.ts) ---

MAIN_MD_TEMPLATE = """# Project Context

## Current Focus
(Auto-populated after first commit)

## Recent Milestones
(none yet)

## Open Branches
(none)
"""

REGISTRY_TEMPLATE = """# Branch Registry

## Active Branch
main

## Branch History
| Branch | Created | Status |
|--------|---------|--------|
| main | {date} | active |
"""

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

MAIN_COMMITS_TEMPLATE = """# Branch: main

## Rolling Summary
(none yet)

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

METADATA_TEMPLATE = {
    "version": 1,
    "created": "",
    "proactive_commits": True,
    "branches": [],
    "file_tree": [],
    "dependencies": [],
    "config": {
        "language": "unknown",
        "framework": "unknown",
    },
}


class MemoryManager:
    """Manages the .ccr/ directory for a project.

    All file I/O is synchronous. Thread-safe via per-file locks.
    """

    def __init__(self, project_root: str, config: CCRConfig | None = None):
        self.project_root = os.path.abspath(project_root)
        self.ccr_root = os.path.join(self.project_root, ".ccr")
        self.branches_dir = os.path.join(self.ccr_root, "branches")
        self.config = config or CCRConfig()
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.sub_client = None
        self._evolved_summaries: dict[str, EvolvedSummary] = {}  # commit_id → EvolvedSummary
        # ALMA-inspired: optional schema override for retrieval params
        self._schema_overrides: dict[str, Any] = {}
        # In-memory commit index: branch -> {commit_id -> text_block} for O(1) lookups
        self._commit_index: dict[str, dict[str, str]] = {}

    def set_sub_client(self, client) -> None:
        """Set the sub-model client for LLM-regenerated rolling summaries."""
        self.sub_client = client

    def set_schema_overrides(self, overrides: dict[str, Any]) -> None:
        """Set schema-driven parameter overrides (ALMA-inspired).

        Called by MCP server when PlaybookSchema has non-default retrieval params.
        Supported keys: link_scan_window, link_semantic_threshold.
        """
        self._schema_overrides = dict(overrides)

    @property
    def effective_link_scan_window(self) -> int:
        """Return link_scan_window from schema override or config fallback."""
        return self._schema_overrides.get("link_scan_window", self.config.link_scan_window)

    @property
    def effective_link_semantic_threshold(self) -> float:
        """Return link_semantic_threshold from schema override or config fallback."""
        return self._schema_overrides.get("link_semantic_threshold", self.config.link_semantic_threshold)

    # --- A-MEM Evolved Summary Storage (§3.3 Eq.7) ---

    @property
    def _evolved_path(self) -> str:
        return os.path.join(self.ccr_root, "evolved_summaries.json")

    def _load_evolved_summaries(self) -> None:
        """Load evolved summary overlays from JSON. Non-destructive if missing."""
        path = self._evolved_path
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
        if not raw:
            return
        try:
            data = json.loads(raw)
            entries = data.get("evolved", {})
            for commit_id, ev in entries.items():
                self._evolved_summaries[commit_id] = EvolvedSummary(
                    commit_id=ev["commit_id"],
                    evolved_what=ev["evolved_what"],
                    evolution_reason=ev["evolution_reason"],
                    evolved_at=ev["evolved_at"],
                    source_commit_id=ev["source_commit_id"],
                    original_what=ev["original_what"],
                )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    def _save_evolved_summaries(self) -> None:
        """Persist the evolved_summaries dict to JSON."""
        path = self._evolved_path
        data = {
            "version": 1,
            "evolved": {
                cid: {
                    "commit_id": ev.commit_id,
                    "evolved_what": ev.evolved_what,
                    "evolution_reason": ev.evolution_reason,
                    "evolved_at": ev.evolved_at,
                    "source_commit_id": ev.source_commit_id,
                    "original_what": ev.original_what,
                }
                for cid, ev in self._evolved_summaries.items()
            },
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def get_evolved_what(self, commit_id: str) -> str | None:
        """Return evolved summary if available, else None (A-MEM §3.3 Eq.7)."""
        ev = self._evolved_summaries.get(commit_id)
        return ev.evolved_what if ev else None

    def _evolve_commit_summary(
        self, existing_commit: dict, new_commit: dict
    ) -> EvolvedSummary | None:
        """Rewrite existing commit's 'what' to incorporate context from new_commit.

        Implements A-MEM §3.3 Eq.7: m̃_i = f_LLM(m_i, m')
        Returns None when sub_client is unavailable or an error occurs.
        """
        if self.sub_client is None:
            return None
        try:
            prompt = (
                "You are updating a project memory entry based on new related work.\n\n"
                f"Original entry:\n"
                f"Title: {existing_commit.get('title', '')}\n"
                f"What: {existing_commit.get('what', '')}\n"
                f"Why: {existing_commit.get('why', '')}\n\n"
                f"New related work arrived:\n"
                f"Title: {new_commit.get('title', '')}\n"
                f"What: {new_commit.get('what', '')}\n"
                f"Why: {new_commit.get('why', '')}\n\n"
                "Rewrite the original \"What\" field to incorporate relevant context "
                "from the new work.\n"
                "Keep it concise (1-3 sentences). Only update if the new work adds "
                "meaningful context.\n"
                'Respond with a JSON object: {"evolved_what": "...", "evolution_reason": "..."}'
            )
            response = self.sub_client.completion([{"role": "user", "content": prompt}])
            raw_json = extract_json_string(response)
            parsed = json.loads(raw_json)
            evolved_what = parsed.get("evolved_what", "").strip()
            evolution_reason = parsed.get("evolution_reason", "").strip()
            if not evolved_what:
                return None
            now_iso = datetime.now(timezone.utc).isoformat()
            existing_id = existing_commit.get("id", "")
            new_id = new_commit.get("id", "")
            return EvolvedSummary(
                commit_id=existing_id,
                evolved_what=evolved_what,
                evolution_reason=evolution_reason,
                evolved_at=now_iso,
                source_commit_id=new_id,
                original_what=existing_commit.get("what", ""),
            )
        except Exception:
            return None  # Never fail — evolution is supplementary

    def _trigger_memory_evolution(self, new_commit_id: str, links: list) -> None:
        """Evolve related commit summaries when a new commit arrives.

        Implements A-MEM §3.3: fires on semantic/supersession links with score > 0.5.
        Caps at 3 evolutions per commit to avoid LLM overuse.
        """
        try:
            branch = self.get_active_branch()
            new_block = self._find_commit_by_id(branch, new_commit_id)
            if not new_block:
                return
            new_commit = self._parse_commit_block(new_block)
            new_commit["id"] = new_commit_id

            evolution_count = 0
            for link in links:
                if evolution_count >= 3:
                    break
                # Only evolve on semantic or supersession links above threshold
                link_type = link.link_type if hasattr(link, "link_type") else link.get("link_type", "")
                score = link.score if hasattr(link, "score") else link.get("score", 0.0)
                if link_type not in ("semantic", "supersession"):
                    continue
                if score <= 0.5:
                    continue
                existing_id = link.target if hasattr(link, "target") else link.get("target", "")
                if not existing_id:
                    continue
                existing_block = self._find_commit_by_id(branch, existing_id)
                if not existing_block:
                    continue
                existing_commit = self._parse_commit_block(existing_block)
                existing_commit["id"] = existing_id

                result = self._evolve_commit_summary(existing_commit, new_commit)
                if result is not None:
                    self._evolved_summaries[existing_id] = result
                    self._save_evolved_summaries()
                    evolution_count += 1
        except Exception:
            pass  # Evolution is supplementary — never fail

    # --- Bootstrap ---

    def ensure_structure(self) -> bool:
        """Create .ccr/ layout if missing. Idempotent. Returns True if created."""
        created = False
        main_branch_dir = os.path.join(self.branches_dir, "main")

        if not os.path.isdir(self.ccr_root):
            os.makedirs(self.ccr_root, exist_ok=True)
            created = True

        os.makedirs(main_branch_dir, exist_ok=True)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self._write_if_missing(
            os.path.join(self.ccr_root, "main.md"),
            MAIN_MD_TEMPLATE,
        )
        self._write_if_missing(
            os.path.join(self.branches_dir, "_registry.md"),
            REGISTRY_TEMPLATE.format(date=now),
        )
        self._write_if_missing(
            os.path.join(main_branch_dir, "commits.md"),
            MAIN_COMMITS_TEMPLATE,
        )
        self._write_if_missing(
            os.path.join(main_branch_dir, "log.md"),
            "",
        )

        # metadata.yaml
        metadata_path = self._get_metadata_path()
        if not os.path.isfile(metadata_path):
            meta = dict(METADATA_TEMPLATE)
            meta["created"] = now
            meta["branches"] = [{"name": "main", "status": "active", "created": now, "parent": None}]
            self._save_metadata(meta)

        # Hierarchical summaries directory (TiMem §3.1)
        summaries_dir = os.path.join(self.ccr_root, "summaries")
        os.makedirs(summaries_dir, exist_ok=True)

        # Initialize summary_meta.yaml if missing
        summary_meta_path = self._get_summary_meta_path()
        if not os.path.isfile(summary_meta_path):
            self._save_summary_meta(self._default_summary_meta())

        # Add .ccr to .gitignore if in a git repo
        gitignore = os.path.join(self.project_root, ".gitignore")
        if os.path.isdir(os.path.join(self.project_root, ".git")):
            self._add_to_gitignore(gitignore)

        # Load A-MEM evolved summaries overlay
        self._load_evolved_summaries()

        return created

    # --- Branch Operations ---

    def get_active_branch(self) -> str:
        registry = self._read_file(os.path.join(self.branches_dir, "_registry.md"))
        if not registry:
            return "main"
        match = re.search(r"## Active Branch\s*\n(\S+)", registry)
        return match.group(1) if match else "main"

    def create_branch(self, name: str, purpose: str, hypothesis: str) -> str:
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

        # Log with OTA format
        self._append_log(name, self._format_ota_log(
            "branch-create", name, "OK",
            observation=f"Creating exploration branch '{name}'",
            thought=f"Hypothesis: {hypothesis}",
            action=f"Created branch with purpose: {purpose}",
        ))

        return f"Created branch '{name}' — purpose: {purpose}"

    # --- Commit Operations ---

    def commit(
        self,
        title: str,
        what: str,
        why: str,
        files_changed: list[str],
        next_step: str,
        patterns_learned: list[str] | None = None,
        admission_threshold: float = 0.85,
        rejection_threshold: float = 0.0,
        compressed_summary: str | None = None,
    ) -> str:
        """Create a structured commit on the active branch.

        Per the GCC paper (§2.2): Each COMMIT produces a structured record
        M_t = (I_t, S_t, D_t) where S_t is a rolling summary regenerated
        from S_{t-1} + new contribution. This creates a progressively
        refined understanding without needing to re-read all commits.

        Per A-MAC (arXiv:2603.04549) Algorithm 1 (correct polarity):
        1. Compute S(m) and S(m_conflict) — admission scores (higher = more valuable)
        2. Compute similarity — max recency-weighted Jaccard to existing commits
        3. If S(m) < rejection_threshold → Reject (low-value, per Alg. 1 line 11)
        4. FindConflict: if similarity >= admission_threshold → conflict found
        5. If conflict AND S(m) > S(m_conflict) → REPLACE old with Merge(m, m_conflict)
           (per Alg. 1 lines 6-7: new outranks, so merge replaces old)
        6. If conflict AND S(m) <= S(m_conflict) → ADD new alongside existing
           (per Alg. 1 lines 8-9: existing outranks, both coexist)
        7. No conflict → Create new commit

        Args:
            admission_threshold: Similarity score (0-1) above which a conflict
                is detected (FindConflict). Default 0.85 per paper §3.3.
                Set to 1.0 to disable merging.
            rejection_threshold: Admission score below which commits are rejected.
                Uses correct polarity: low score = low value = reject.
                Set to 0.0 to disable rejection (default).
        """
        branch = self.get_active_branch()
        admission_score_value = None  # Will be set if admission control runs

        # --- Admission Control (A-MAC Algorithm 1) ---
        if admission_threshold < 1.0 or rejection_threshold > 0.0:
            score = self.compute_admission_score(
                branch, title, what, why, files_changed, next_step,
            )
            admission_score_value = score["score"]

            # Step 3: Rejection — low admission score = low value (Alg. 1 line 11)
            # Correct polarity (G9): reject when score is LOW (not valuable enough)
            if rejection_threshold > 0.0 and score["score"] < rejection_threshold:
                reason = score.get("reason", "low value")
                self._append_log(branch, self._format_ota_log(
                    "commit-reject", f"Rejected: {title}", "REJECTED",
                    observation=f"Admission control: score={score['score']:.2f}, sim={score['similarity']:.2f} ({reason})",
                    thought=f"Score {score['score']:.2f} below rejection threshold {rejection_threshold:.2f} — low value",
                    action=f"Commit rejected, not stored",
                ))
                return f"[REJECTED] {title} (score={score['score']:.2f}, below threshold {rejection_threshold:.2f})"

            # Step 4-6: FindConflict + Score comparison (Alg. 1 lines 4-9)
            # similarity >= admission_threshold → conflict detected (paper: sim > 0.85)
            if score["similarity"] >= admission_threshold and score["conflict_id"]:
                # Score comparison (Alg. 1 lines 6-9):
                # If S(m) > S(m_conflict) → REPLACE old with Merge(m, m_conflict)
                #   (new is more valuable, replaces old via merge — Alg. 1 lines 6-7)
                # If S(m) <= S(m_conflict) → ADD new alongside existing
                #   (existing is more valuable, new simply coexists — Alg. 1 lines 8-9)
                new_score = score["score"]
                conflict_score = score.get("conflict_score", 0.0)
                if new_score > conflict_score:
                    # Replace: new outranks old → merge new into conflict target
                    merged_id = self._merge_into_last_commit(
                        branch, score["conflict_id"],
                        title, what, why, files_changed, next_step,
                        patterns_learned,
                    )
                    self._update_rolling_summary(branch, what, why, next_step, compressed_summary)
                    self._update_current_focus(title, next_step)

                    self._append_log(branch, self._format_ota_log(
                        "commit-merge", f"{merged_id}: +{title}", "OK",
                        observation=f"Admission control: sim={score['similarity']:.2f}, S(m)={new_score:.2f} vs S(m')={conflict_score:.2f}",
                        thought=f"S(m) > S(m_conflict): replacing {merged_id} with merged version",
                        action=f"Replaced {merged_id} with merged info",
                    ))

                    self._git_commit(f"ccr: +{title} (merged into {merged_id})")
                    return f"[{merged_id}+] {title} (merged, sim={score['similarity']:.2f})"
                else:
                    pass  # Fall through: existing is more valuable, add new alongside

        # --- Normal commit path ---
        commit_id = self._get_next_commit_id(branch)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        files_str = ", ".join(files_changed) if files_changed else "(none)"

        # Compute admission score for storage if not already computed.
        # This allows future S(m_conflict) comparisons to use the real score.
        if admission_score_value is None:
            score = self.compute_admission_score(
                branch, title, what, why, files_changed, next_step,
            )
            admission_score_value = score["score"]

        patterns_str = ""
        if patterns_learned:
            patterns_str = f"**Patterns**: {' | '.join(patterns_learned)}\n"

        entry = (
            f"## [{commit_id}] {now} | branch:{branch} | {title}\n"
            f"**What**: {what}\n"
            f"**Why**: {why}\n"
            f"**Files**: {files_str}\n"
            f"**Next**: {next_step}\n"
            f"{patterns_str}"
            f"**Score**: {admission_score_value:.2f}\n\n---\n\n"
        )

        # Reference OTA log slice in commit (per GCC paper)
        ota_slice = self._get_ota_slice_since_last_commit(branch)
        if ota_slice:
            entry = entry.rstrip("---\n\n") + f"**OTA Trace**: {ota_slice}\n\n---\n\n"

        self._prepend_commit(branch, entry)

        # Track commit count (ALMA-inspired retrieval parameter evolution)
        try:
            self._increment_memory_metric("total_commits")
        except Exception:
            pass  # Metrics are supplementary — never fail the commit

        # GCC paper: regenerate rolling summary S_t = f(S_{t-1}, D_t)
        # Capture pre-update summary length for the compression warning (G4).
        # Strategy 2.5 may auto-compress the summary during _update_rolling_summary(),
        # so reading after the call would miss the original length.
        _pre_update_summary_len = len(self._get_rolling_summary(branch))
        self._update_rolling_summary(branch, what, why, next_step, compressed_summary)

        if branch == "main":
            self._update_main_milestones(now, branch, title)

        # Update Current Focus in main.md
        self._update_current_focus(title, next_step)

        self._append_log(branch, self._format_ota_log(
            "commit", f"{commit_id}: {title}", "OK",
            observation=f"Committing: {what}",
            thought=f"Reason: {why}",
            action=f"Created commit {commit_id} with {len(files_changed)} file(s) changed",
        ))

        # Git commit integration
        self._git_commit(f"ccr: {title}")

        # Heuristic commit cross-linking (A-MEM/MAGMA inspired taxonomy)
        # _embed_commit is called here (normal path only) and its vector is
        # passed to _compute_links to avoid a second inference pass.
        commit_links: list = []
        try:
            new_vec = self._embed_commit(
                commit_id, f"{title} {what} {why} {next_step}"
            )
            commit_links = self._compute_links(
                branch, commit_id, title, what, why, files_changed, next_step,
                new_vec=new_vec,
            )
            if commit_links:
                self._update_links(commit_id, commit_links)
        except Exception:
            pass  # Linking is supplementary — never fail the commit

        # A-MEM §3.3: evolve related commit summaries if sub-model available
        if self.sub_client is not None and commit_links:
            try:
                self._trigger_memory_evolution(commit_id, commit_links)
            except Exception:
                pass  # Evolution is supplementary — never fail the commit

        # CER-inspired pattern buffer management (arXiv:2506.06698 §3.1)
        promotion_suggestions: list[dict] = []
        if patterns_learned:
            try:
                promotion_suggestions = self._process_patterns(
                    commit_id, patterns_learned, now,
                )
            except Exception:
                pass  # Pattern management is supplementary — never fail the commit

        if not promotion_suggestions:
            # Auto-scan: surface patterns that crossed threshold without new patterns_learned.
            # Even commits with no patterns_learned will bubble up ready-to-promote patterns.
            try:
                promotion_suggestions = self._scan_pending_promotions()
            except Exception:
                pass  # Pattern scanning is supplementary — never fail the commit

        # TiMem §3.2.2: Check if session summary should be generated
        self._maybe_generate_session_summary(branch)

        result = f"[{commit_id}] {title} (branch: {branch})"
        if promotion_suggestions:
            result += (
                f"\n\n**Pattern promotion suggestions** "
                f"(appeared in {self.config.pattern_promotion_count}+ commits):"
            )
            for ps in promotion_suggestions:
                result += (
                    f"\n  - \"{ps['text']}\" "
                    f"(seen in {ps['count']} commits: {', '.join(ps['commit_ids'])})"
                )
            result += "\n\nConsider calling ace_apply_delta with ADD to promote these to the playbook."

        # GCC paper G4: Check if rolling summary needs LLM compression.
        # Strategy 2.5 auto-compressed the summary if it exceeded 1200 chars, so the
        # warning now fires based on the PRE-UPDATE length to remain a "could do better"
        # hint (LLM compression is always higher quality than mechanical compression).
        # The current (post-update) summary is shown so Claude Code can act immediately.
        summary_compression_threshold = 1200
        if _pre_update_summary_len > summary_compression_threshold and compressed_summary is None:
            current_summary = self._get_rolling_summary(branch)
            result += (
                f"\n\n\u26a0\ufe0f Rolling summary is getting long ({_pre_update_summary_len} chars). "
                f"Call gcc_commit again with compressed_summary='<your 2-3 sentence synthesis>'. "
                f"Current summary to compress:\n\n---\n{current_summary}\n---\n\n"
                f"Write a concise synthesis capturing key decisions and current direction, "
                f"then pass it as compressed_summary= in your next gcc_commit call. "
                f"Alternatively, call gcc_consolidate to compress project memory."
            )

        return result

    # --- Admission Control (A-MAC inspired) ---

    def _parse_recent_commit_data(self, branch: str, k: int = 3) -> list[dict[str, Any]]:
        """Parse last k commits from commits.md into structured dicts.

        Returns list of dicts, each with:
            id, timestamp, title, what, why, files (list[str]), next
        """
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return []
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        commit_parts = [p for p in parts if re.match(r"## \[C\d{3,}\]", p.strip())]

        results = []
        for part in commit_parts[:k]:
            data: dict[str, Any] = {}
            # Parse header: ## [C021] 2026-03-10 22:25 | branch:main | Title
            header_match = re.match(
                r"## \[(C\d{3,})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|[^|]*\|\s*(.*)",
                part.strip(),
            )
            if header_match:
                data["id"] = header_match.group(1)
                data["timestamp"] = header_match.group(2)
                data["title"] = header_match.group(3).strip()
            else:
                continue

            # Parse fields
            what_match = re.search(r"\*\*What\*\*:\s*(.*)", part)
            why_match = re.search(r"\*\*Why\*\*:\s*(.*)", part)
            files_match = re.search(r"\*\*Files\*\*:\s*(.*)", part)
            next_match = re.search(r"\*\*Next\*\*:\s*(.*)", part)
            score_match = re.search(r"\*\*Score\*\*:\s*([\d.]+)", part)

            data["what"] = what_match.group(1).strip() if what_match else ""
            data["why"] = why_match.group(1).strip() if why_match else ""
            data["next"] = next_match.group(1).strip() if next_match else ""

            # Stored admission score (None if not present — backward compat)
            if score_match:
                try:
                    data["stored_score"] = float(score_match.group(1))
                except (ValueError, TypeError):
                    data["stored_score"] = None
            else:
                data["stored_score"] = None

            files_str = files_match.group(1).strip() if files_match else ""
            if files_str and files_str != "(none)":
                data["files"] = [f.strip() for f in files_str.split(",") if f.strip()]
            else:
                data["files"] = []

            # CER patterns (backward compatible — empty list if absent)
            patterns_match = re.search(r"\*\*Patterns\*\*:\s*(.*)", part)
            if patterns_match:
                raw_patterns = patterns_match.group(1).strip()
                data["patterns"] = [p.strip() for p in raw_patterns.split("|") if p.strip()]
            else:
                data["patterns"] = []

            results.append(data)
        return results

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Jaccard similarity between two sets."""
        if not a and not b:
            return 0.0
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    @staticmethod
    def _commit_text(title: str, what: str, why: str) -> str:
        """Canonical text representation of a commit for ONNX embedding.

        A-MAC §3.2 Eq. 3: φ(m) computed on commit content.
        """
        return f"{title} {what} {why}".strip()

    # --- Heuristic Commit Cross-Linking (A-MEM/MAGMA inspired taxonomy) ---
    # Uses mechanical heuristics (file overlap, regex, word Jaccard) instead of
    # the papers' LLM inference (A-MEM Eq. 6, MAGMA Eq. 8) and dense vector
    # embeddings (A-MEM Eq. 4, MAGMA semantic graph). MAGMA's temporal graph
    # (immutable chronological chain) is implicit in sequential commit IDs.
    # MAGMA's adaptive beam search (Alg. 1) is replaced with plain BFS.
    # A-MEM's memory evolution (Eq. 7) is not implemented.

    _LINK_TYPES = ("entity", "causal", "supersession", "semantic")
    _STOP_WORDS = frozenset({
        # Articles & determiners
        "the", "a", "an", "this", "that", "these", "those", "some", "any",
        "each", "every", "all", "both", "few", "more", "most", "other",
        # Prepositions
        "to", "for", "of", "in", "on", "at", "by", "from", "into", "about",
        "between", "through", "during", "before", "after", "above", "below",
        "up", "out", "off", "over", "under", "again", "further", "then",
        # Conjunctions
        "and", "or", "but", "nor", "yet", "so", "if", "when", "while",
        "because", "although", "than",
        # Pronouns
        "it", "its", "they", "them", "their", "we", "our", "you", "your",
        "he", "she", "his", "her", "who", "which", "what", "how",
        # Be/have/do
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "having",
        "do", "does", "did", "doing",
        # Modals
        "will", "would", "can", "could", "shall", "should", "may", "might",
        "must",
        # Common adverbs/adjectives
        "not", "no", "just", "also", "very", "only", "now", "here", "there",
        "where", "still", "already",
        # Common verbs (too generic for keywords)
        "get", "got", "set", "use", "used", "using", "make", "made",
    })
    _SUPERSESSION_KEYWORDS = re.compile(
        r"(?:replaced|superseded|reverted|refactored\s+from|deprecated|reworked|improved\s+upon)",
        re.IGNORECASE,
    )
    _COMMIT_ID_RE = re.compile(r"\b(C\d{3,})\b")

    # --- Memory Metrics (ALMA-inspired retrieval parameter evolution) ---

    def _get_memory_metrics_path(self) -> str:
        """Return path to .ccr/memory_metrics.json."""
        return os.path.join(self.ccr_root, "memory_metrics.json")

    def _increment_memory_metric(self, key: str, amount: int = 1) -> None:
        """Thread-safe increment of a memory metric counter."""
        path = self._get_memory_metrics_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
            if raw:
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Failed to load %s: %s", path, exc)
                    data = {}
            else:
                data = {}
            # Ensure defaults
            data.setdefault("search_calls", 0)
            data.setdefault("search_zero_results", 0)
            data.setdefault("link_creations", 0)
            data.setdefault("total_commits", 0)
            data[key] = data.get(key, 0) + amount
            data["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._write_file_unlocked(path, json.dumps(data, indent=2, ensure_ascii=False))

    def get_memory_metrics(self) -> dict:
        """Load and return current memory metrics. Returns zeros if no file."""
        path = self._get_memory_metrics_path()
        raw = self._read_file(path)
        if not raw:
            return {
                "search_calls": 0,
                "search_zero_results": 0,
                "link_creations": 0,
                "total_commits": 0,
                "last_updated": "",
            }
        try:
            data = json.loads(raw)
            # Ensure all expected keys exist
            data.setdefault("search_calls", 0)
            data.setdefault("search_zero_results", 0)
            data.setdefault("link_creations", 0)
            data.setdefault("total_commits", 0)
            data.setdefault("last_updated", "")
            return data
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return {
                "search_calls": 0,
                "search_zero_results": 0,
                "link_creations": 0,
                "total_commits": 0,
                "last_updated": "",
            }

    def _get_links_path(self) -> str:
        return os.path.join(self.ccr_root, "commit_links.json")

    def _get_commit_embeddings_path(self) -> str:
        return os.path.join(self.ccr_root, "commit_embeddings.json.gz")

    def _embed_commit(self, commit_id: str, text: str) -> "np.ndarray | None":
        """Embed commit text and persist to cache. Returns vector or None.

        Tries sqlite-vec first (persistent KNN store at .ccr/embeddings.db).
        Falls back to .ccr/commit_embeddings.json.gz (capped at
        link_scan_window * 2 entries, oldest evicted). Returns the computed
        (384,) float32 L2-normalized vector so the caller can reuse it
        without a second inference pass. Returns None if ONNX unavailable.
        """
        model = get_embedding_model()
        if model is None:
            return None
        import numpy as np  # soft dep -- only reachable when ONNX available
        try:
            # embed_query is expensive ONNX inference — run outside the lock
            vec = model.embed_query(text)

            # Try sqlite-vec first (persistent vector store)
            from ccr.context.vec_store import get_vec_store
            db_path = os.path.join(self.ccr_root, "embeddings.db")
            store = get_vec_store(db_path)
            if store is not None:
                store.upsert(commit_id, vec.tolist(), namespace="commit")
                return vec

            # Fallback: gzip JSON (existing behavior)
            path = self._get_commit_embeddings_path()
            with self._locks[path], self._file_lock(path):
                cache = load_embeddings(path)
                cache[commit_id] = vec.tolist()
                cap = self.effective_link_scan_window * 2
                if len(cache) > cap:
                    for old_id in sorted(
                        cache.keys(),
                        key=lambda c: int(re.search(r"\d+$", c).group()) if re.search(r"\d+$", c) else 0,
                    )[: len(cache) - cap]:
                        del cache[old_id]
                save_embeddings(cache, path)
            return vec
        except Exception as exc:
            logger.warning("Failed to embed/persist commit %s: %s", commit_id, exc)
            return None

    def _load_commit_embeddings(self, commit_ids: list) -> dict:
        """Load cached embeddings for given commit IDs as numpy arrays.

        Tries sqlite-vec first, falls back to gzip JSON.
        Returns dict[str, np.ndarray] with only IDs present in cache.
        Silently omits missing IDs. Returns empty dict on any error.
        """
        try:
            import numpy as np  # soft dep

            # Try sqlite-vec first
            from ccr.context.vec_store import get_vec_store
            db_path = os.path.join(self.ccr_root, "embeddings.db")
            store = get_vec_store(db_path)
            if store is not None:
                batch = store.get_batch(commit_ids)
                if batch:
                    return {
                        cid: np.array(vec, dtype=np.float32)
                        for cid, vec in batch.items()
                    }

            # Fallback: gzip JSON
            raw = load_embeddings(self._get_commit_embeddings_path())
            return {
                cid: np.array(raw[cid], dtype=np.float32)
                for cid in commit_ids
                if cid in raw
            }
        except Exception as exc:
            logger.warning("Failed to load commit embeddings: %s", exc)
            return {}

    def _load_all_commit_embeddings(self) -> dict:
        """Load ALL cached commit embeddings as numpy arrays.

        Unlike _load_commit_embeddings(ids) which filters by ID,
        this returns the entire cache for search operations.
        Returns dict[str, np.ndarray]. Empty dict on error.
        """
        try:
            import numpy as np
            raw = load_embeddings(self._get_commit_embeddings_path())
            return {
                cid: np.array(vec, dtype=np.float32)
                for cid, vec in raw.items()
            }
        except Exception as exc:
            logger.warning("Failed to load all commit embeddings: %s", exc)
            return {}

    def _load_links(self) -> dict:
        """Load the commit link graph from JSON. Returns default if missing/corrupt."""
        path = self._get_links_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
        if not raw:
            return {"version": 1, "links": {}}
        try:
            data = json.loads(raw)
            if not isinstance(data.get("links"), dict):
                return {"version": 1, "links": {}}
            return data
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return {"version": 1, "links": {}}

    def _save_links(self, data: dict) -> None:
        """Atomically save the commit link graph."""
        path = self._get_links_path()
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def _update_links(self, commit_id: str, links: list[CommitLink]) -> None:
        """Load, modify, and save links under a single lock (avoids TOCTOU)."""
        path = self._get_links_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
            if not raw:
                data: dict = {"version": 1, "links": {}}
            else:
                try:
                    data = json.loads(raw)
                    if not isinstance(data.get("links"), dict):
                        data = {"version": 1, "links": {}}
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Failed to load %s: %s", path, exc)
                    data = {"version": 1, "links": {}}
            for cl in links:
                self._add_link(data, commit_id, cl.target, cl)
            content = json.dumps(data, indent=2, ensure_ascii=False)
            self._write_file_unlocked(path, content)
        # Track link creation metrics (ALMA-inspired retrieval parameter evolution)
        try:
            self._increment_memory_metric("link_creations", len(links))
        except Exception:
            pass  # Metrics are supplementary — never fail the link update

    @staticmethod
    def _add_link(data: dict, source: str, target: str, link: CommitLink) -> None:
        """Add a bidirectional link (A-MEM Zettelkasten). Deduplicates by higher score."""
        for src, tgt in [(source, target), (target, source)]:
            node = data["links"].setdefault(src, {})
            bucket = node.setdefault(link.link_type, [])
            # Dedup: if same target exists, keep higher score
            for existing in bucket:
                if existing["target"] == tgt:
                    if link.score > existing.get("score", 0.0):
                        update: dict[str, Any] = {"target": tgt, "score": link.score}
                        if link.shared_files:
                            update["shared_files"] = link.shared_files
                        if link.snippet:
                            update["snippet"] = link.snippet
                        existing.update(update)
                    break
            else:
                entry = link.to_dict()
                entry["target"] = tgt
                bucket.append(entry)

    @staticmethod
    def _extract_commit_references(text: str) -> list[str]:
        """Extract all commit ID references (C###) from text."""
        return re.findall(r"\b(C\d{3,})\b", text)

    @classmethod
    def _detect_supersession(cls, text: str) -> list[tuple[str, str]]:
        """Detect replacement language near commit IDs.

        Returns list of (commit_id, snippet) tuples.
        """
        results = []
        for m in cls._COMMIT_ID_RE.finditer(text):
            cid = m.group(1)
            # Check within 120 chars on either side for replacement keywords
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 120)
            window = text[start:end]
            if cls._SUPERSESSION_KEYWORDS.search(window):
                # Extract a concise snippet around the match
                snippet = text[max(0, m.start() - 40):min(len(text), m.end() + 40)].strip()
                results.append((cid, snippet))
        return results

    @classmethod
    def _extract_keywords(cls, text: str) -> set[str]:
        """Extract keywords from text, filtering stop words, short tokens, and pure digits."""
        return {w for w in re.findall(r"\w+", text.lower())
                if w not in cls._STOP_WORDS and len(w) > 2 and not w.isdigit()}

    def _compute_links(
        self,
        branch: str,
        commit_id: str,
        title: str,
        what: str,
        why: str,
        files_changed: list[str],
        next_step: str,
        new_vec=None,  # (384,) float32 L2-normalized ndarray, or None
    ) -> list[CommitLink]:
        """Compute heuristic cross-links for a new commit against recent history.

        All linking is mechanical (zero LLM calls). Scans the last k commits
        (config.link_scan_window, default 20) — NOT global retrieval across
        all history. Commits older than the window are never scanned.

        Link types:
        1. Entity links: file-set Jaccard > threshold (cf. MAGMA entity graph
           which uses LLM-extracted abstract entity nodes — we use file paths)
        2. Causal links: regex detection of C### IDs in text, validated against
           existing commits (cf. MAGMA causal graph which uses LLM inference
           for implicit causality — we only detect explicit references)
        3. Supersession links: replacement language + C### (heuristic, no paper analog)
        4. Semantic links: dense cosine similarity when ONNX embedding available
           for both commits; word Jaccard fallback otherwise (per-commit fallback).

        MAGMA's temporal graph (immutable chronological chain) is implicit in
        sequential commit IDs and not stored as explicit links.
        """
        recent = self._parse_recent_commit_data(branch, k=self.effective_link_scan_window)
        if not recent:
            return []

        new_files = {f.strip().lower() for f in files_changed if f.strip()}
        combined_text = f"{title} {what} {why} {next_step}"
        new_keywords = self._extract_keywords(combined_text)

        # Pre-compute causal and supersession references from the new commit's text
        all_refs = set(self._extract_commit_references(combined_text))
        supersession_hits = {cid: snip for cid, snip in self._detect_supersession(combined_text)}

        # Validate: only keep references to commits that actually exist
        existing_ids = {c.get("id", "") for c in recent if c.get("id")}
        all_refs = all_refs & existing_ids
        supersession_hits = {k: v for k, v in supersession_hits.items() if k in existing_ids}

        links: list[CommitLink] = []

        # Hoist embedding load outside the per-commit loop: one gzip read for all recent
        # commits instead of N reads (O(1) I/O vs O(N) I/O per commit() call).
        all_cached = self._load_commit_embeddings([c.get("id", "") for c in recent if c.get("id")])

        for commit in recent:
            cid = commit.get("id", "")
            if not cid or cid == commit_id:
                continue

            has_typed_link = False  # Track if entity/causal/supersession already found

            # 1. Entity links (shared files)
            old_files = {f.strip().lower() for f in commit.get("files", []) if f.strip()}
            file_sim = self._jaccard(new_files, old_files)
            if file_sim > self.config.link_entity_threshold:
                shared = sorted(new_files & old_files)
                links.append(CommitLink(
                    target=cid, link_type="entity", score=round(file_sim, 3),
                    shared_files=shared,
                ))
                has_typed_link = True

            # 2 & 3. Causal / supersession links
            if cid in supersession_hits:
                # Supersession subsumes causal
                links.append(CommitLink(
                    target=cid, link_type="supersession", score=1.0,
                    snippet=supersession_hits[cid],
                ))
                has_typed_link = True
            elif cid in all_refs:
                # Pure causal reference
                # Extract snippet around the reference
                idx = combined_text.find(cid)
                snippet = combined_text[max(0, idx - 40):min(len(combined_text), idx + len(cid) + 40)].strip()
                links.append(CommitLink(
                    target=cid, link_type="causal", score=1.0,
                    snippet=snippet,
                ))
                has_typed_link = True

            # 4. Semantic links (only if no other link type to this target)
            if not has_typed_link:
                if new_vec is not None and cid in all_cached:
                    cached_vec = all_cached[cid]
                    if cached_vec.shape == new_vec.shape:
                        # Dense cosine: dot product of L2-normalized vectors
                        score = float(cached_vec @ new_vec)
                    else:
                        # Shape mismatch (stale cache from different model dim) — fall back
                        old_text = f"{commit.get('title', '')} {commit.get('what', '')} {commit.get('why', '')}".lower()
                        old_keywords = self._extract_keywords(old_text)
                        score = self._jaccard(new_keywords, old_keywords)
                else:
                    old_text = f"{commit.get('title', '')} {commit.get('what', '')} {commit.get('why', '')}".lower()
                    old_keywords = self._extract_keywords(old_text)
                    score = self._jaccard(new_keywords, old_keywords)
                if score > self.effective_link_semantic_threshold:
                    links.append(CommitLink(
                        target=cid, link_type="semantic", score=round(score, 3),
                    ))

        return links

    def get_commit_links(self, commit_id: str) -> dict[str, list[dict]]:
        """Retrieve all cross-links for a commit.

        Returns dict with keys per link type, each containing a list of link dicts.
        """
        data = self._load_links()
        node = data.get("links", {}).get(commit_id, {})
        result: dict[str, list[dict]] = {}
        for lt in self._LINK_TYPES:
            result[lt] = node.get(lt, [])
        return result

    def get_linked_commits(
        self,
        commit_id: str,
        link_types: list[str] | None = None,
        max_hops: int = 1,
        query: str | None = None,
        max_results: int | None = None,
    ) -> list[dict]:
        """Priority-queue traversal of commit links (MAGMA-inspired adaptive).

        Uses a max-heap to explore the most relevant linked commits first.
        When ``query`` is provided and ONNX embeddings are available, edge
        scores are computed via ``quick_cosine(query, commit_what)`` so
        traversal is intent-aware (MAGMA Alg. 1 Eq. 5-6).  When ONNX is
        unavailable or ``query`` is empty, falls back to plain BFS using
        stored heuristic link scores.

        Returns list of dicts: {id, link_type, score, hop, title, what}.

        When ONNX embeddings are available for the source commit and candidate
        commits, results are re-ranked by dense cosine similarity (dot product
        of L2-normalised vectors) so the most semantically relevant linked
        commits appear first.  An ``embedding_score`` key (float) is added to
        each result dict when ONNX is available; it is absent on graceful
        degradation.

        Args:
            commit_id: Starting commit for traversal.
            link_types: Restrict to these link types (default: all).
            max_hops: Maximum BFS depth (default 1).
            query: Natural language query for intent-aware traversal.
                   When provided and ONNX available, edges are weighted by
                   cosine(query, commit) so the heap explores the most
                   query-relevant commits first.  Results include
                   ``query_score`` field when query is used.
            max_results: Maximum number of results to return.
                         Defaults to ``config.link_max_results`` (10).
        """
        data = self._load_links()
        branch = self.get_active_branch()
        types = set(link_types) if link_types else set(self._LINK_TYPES)
        visited: set[str] = {commit_id}
        results: list[dict] = []
        effective_limit = max_results if max_results is not None else self.config.link_max_results
        # Collect up to 2x the final limit so re-ranking has enough candidates
        collect_limit = effective_limit * 2

        # --- Determine whether ONNX-based scoring is available ---
        # Two scoring paths for query-aware traversal:
        #   1. quick_cosine (text-to-text): preferred, computes cosine on the fly
        #   2. cached vector dot product: fallback when quick_cosine unavailable
        # Either path produces query_score in results.
        use_onnx_scoring = False
        if query:
            # Probe quick_cosine to check if ONNX backend is operational
            try:
                probe = quick_cosine("a", "b")
                if probe is not None:
                    use_onnx_scoring = True
            except Exception:
                pass  # Graceful fallback to cached vectors or plain BFS

        # --- ONNX embeddings: load cached vectors for edge scoring + post-traversal re-ranking ---
        all_ids_in_graph = list(data.get("links", {}).keys())
        all_cached = self._load_commit_embeddings(all_ids_in_graph) if all_ids_in_graph else {}

        # Pre-embed query vector for cached-vector scoring fallback
        query_vec = None
        if query:
            try:
                _qemb = get_embedding_model()
                if _qemb is not None:
                    query_vec = _qemb.embed_query(query)
            except Exception:
                pass

        # query_scoring_active: True when either quick_cosine or cached vectors can score
        query_scoring_active = use_onnx_scoring or (query_vec is not None)

        # --- Priority-queue traversal (max-heap via negated scores) ---
        # Heap entries: (-edge_score, tie_breaker, hop, target_id, link_type, link_entry)
        # The tie_breaker ensures stable ordering when scores are equal.
        heap: list[tuple[float, int, int, str, str, dict]] = []
        tie_counter = 0

        # Seed the heap with neighbors of the starting commit
        self._heap_push_neighbors(
            data, commit_id, types, visited, heap, tie_counter,
            hop=1, query=query, use_onnx_scoring=use_onnx_scoring,
            branch=branch, all_cached=all_cached, query_vec=query_vec,
        )
        tie_counter += len(heap)

        while heap and len(results) < collect_limit:
            neg_score, _tie, hop, tgt, lt, link_entry = heapq.heappop(heap)
            if tgt in visited:
                continue
            visited.add(tgt)
            edge_score = -neg_score

            # Fetch commit data for context
            commit_text = self._find_commit_by_id(branch, tgt)
            parsed = self._parse_commit_block(commit_text) if commit_text else {}
            result: dict = {
                "id": tgt,
                "link_type": lt,
                "score": link_entry.get("score", 0.0),
                "hop": hop,
                "title": parsed.get("title", ""),
                "what": parsed.get("what", ""),
                **({k: link_entry[k] for k in ("shared_files", "snippet") if k in link_entry}),
            }
            if query_scoring_active and query:
                result["query_score"] = edge_score
            results.append(result)

            # Expand next hop if within depth limit
            if hop < max_hops:
                added = self._heap_push_neighbors(
                    data, tgt, types, visited, heap, tie_counter,
                    hop=hop + 1, query=query, use_onnx_scoring=use_onnx_scoring,
                    branch=branch, all_cached=all_cached, query_vec=query_vec,
                )
                tie_counter += added

        # --- ONNX re-ranking (graceful degradation when unavailable) ---
        all_result_ids = [commit_id] + [r["id"] for r in results]
        cached = self._load_commit_embeddings(all_result_ids)
        src_vec = cached.get(commit_id)

        if src_vec is not None:
            try:
                import numpy as np
                for r in results:
                    tgt_vec = cached.get(r["id"])
                    if tgt_vec is not None:
                        r["embedding_score"] = float(np.dot(src_vec, tgt_vec))
                    else:
                        r["embedding_score"] = r.get("score", 0.0)
            except ImportError:
                pass

        # Post-traversal sort: prefer query_score when available, else embedding_score
        if query_scoring_active and any("query_score" in r for r in results):
            results.sort(key=lambda r: r.get("query_score", 0.0), reverse=True)
        elif src_vec is not None:
            results.sort(key=lambda r: r.get("embedding_score", 0.0), reverse=True)

        # Truncate to effective limit AFTER re-ranking
        return results[:effective_limit]

    def _heap_push_neighbors(
        self,
        data: dict,
        src: str,
        types: set[str],
        visited: set[str],
        heap: list[tuple[float, int, int, str, str, dict]],
        tie_counter: int,
        *,
        hop: int,
        query: str | None,
        use_onnx_scoring: bool,
        branch: str,
        all_cached: dict,
        query_vec: Any,
    ) -> int:
        """Push neighbors of ``src`` onto the priority heap.

        Computes edge scores using ``quick_cosine(query, commit_what)`` when
        ONNX is available, otherwise uses stored heuristic link scores.

        Args:
            data: Full link graph data.
            src: Source commit ID to expand.
            types: Allowed link types.
            visited: Already-visited commit IDs.
            heap: The priority heap (mutated in place).
            tie_counter: Starting counter for tie-breaking.
            hop: Current hop depth for newly pushed entries.
            query: Natural language query (may be None).
            use_onnx_scoring: Whether ONNX quick_cosine is available.
            branch: Active branch name for commit text lookup.
            all_cached: Pre-loaded commit embedding vectors.
            query_vec: Pre-embedded query vector (numpy array or None).

        Returns:
            Number of entries pushed onto the heap.
        """
        node = data.get("links", {}).get(src, {})
        pushed = 0
        for lt in types:
            for link_entry in node.get(lt, []):
                tgt = link_entry.get("target", "")
                if tgt in visited:
                    continue

                edge_score: float
                if use_onnx_scoring and query:
                    # Primary: quick_cosine(query, target commit text)
                    commit_text = self._find_commit_by_id(branch, tgt)
                    parsed = self._parse_commit_block(commit_text) if commit_text else {}
                    tgt_what = parsed.get("what", parsed.get("title", ""))
                    cosine = quick_cosine(query, tgt_what) if tgt_what else None
                    if cosine is not None:
                        edge_score = cosine
                    else:
                        # Fallback to cached vector dot product
                        tgt_vec = all_cached.get(tgt)
                        if query_vec is not None and tgt_vec is not None:
                            try:
                                import numpy as np
                                edge_score = float(np.dot(query_vec, tgt_vec))
                            except Exception:
                                edge_score = link_entry.get("score", 0.0)
                        else:
                            edge_score = link_entry.get("score", 0.0)
                else:
                    edge_score = link_entry.get("score", 0.0)

                # Max-heap: negate score for heapq (min-heap)
                heapq.heappush(
                    heap,
                    (-edge_score, tie_counter + pushed, hop, tgt, lt, link_entry),
                )
                pushed += 1
        return pushed

    @staticmethod
    def _parse_commit_block(text: str) -> dict:
        """Parse a single commit block (Markdown) into a dict with title/what/why."""
        result: dict[str, Any] = {}
        title_match = re.search(r"\[C\d{3,}\]\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*\|\s*branch:[\w-]+\s*\|\s*(.*)", text)
        if title_match:
            result["title"] = title_match.group(1).strip()
        for field in ("What", "Why", "Files", "Next"):
            m = re.search(rf"\*\*{field}\*\*:\s*(.*?)(?=\n\*\*(?:What|Why|Files|Next|Patterns|Score|OTA)\*\*|\n##|\Z)", text, re.DOTALL)
            if m:
                result[field.lower()] = m.group(1).strip()
        # CER patterns (backward compatible)
        patterns_m = re.search(r"\*\*Patterns\*\*:\s*(.*)", text)
        if patterns_m:
            result["patterns"] = [p.strip() for p in patterns_m.group(1).strip().split("|") if p.strip()]
        return result

    def _format_links_for_context(self, commit_id: str, links: dict[str, list[dict]]) -> str:
        """Format commit links as Markdown for gcc_context output."""
        lines = [f"# Links for {commit_id}"]
        for lt in self._LINK_TYPES:
            entries = links.get(lt, [])
            if not entries:
                continue
            parts = []
            for e in entries:
                tgt = e.get("target", "?")
                if lt == "entity":
                    shared = ", ".join(e.get("shared_files", []))
                    parts.append(f"{tgt} (shared: {shared})" if shared else tgt)
                elif lt in ("causal", "supersession"):
                    snippet = e.get("snippet", "")
                    parts.append(f'{tgt} ("{snippet}")' if snippet else tgt)
                else:  # semantic
                    emb = e.get("embedding_score")
                    score_val = emb if emb is not None else e.get("score", 0.0)
                    tag = "emb" if emb is not None else "score"
                    parts.append(f"{tgt} ({tag}: {score_val:.2f})")
            lines.append(f"- **{lt.capitalize()}**: {', '.join(parts)}")
        return "\n".join(lines)

    # --- EverMemOS-Inspired Thematic Clustering (arXiv:2601.02163) ---

    def compute_clusters(
        self, min_cluster_size: int = 2, link_score_threshold: float = 0.3
    ) -> list[dict]:
        """Compute thematic clusters from the cross-link graph.

        Uses connected components (BFS) on entity + semantic links.
        Inspired by EverMemOS (arXiv:2601.02163) MemScene clustering.

        Args:
            min_cluster_size: Minimum commits to form a cluster.
            link_score_threshold: Minimum link score to consider.

        Returns:
            List of cluster dicts: {id, name, commit_ids, top_keywords}
        """
        links_data = self._load_links()
        all_links = links_data.get("links", {})

        # Build adjacency from entity + semantic links
        adjacency: dict[str, set[str]] = {}
        for src, typed_links in all_links.items():
            if src not in adjacency:
                adjacency[src] = set()
            for link_type in ("entity", "semantic"):
                for entry in typed_links.get(link_type, []):
                    tgt = entry.get("target", "")
                    score = entry.get("score", 0.0)
                    if tgt and score >= link_score_threshold:
                        adjacency.setdefault(src, set()).add(tgt)
                        adjacency.setdefault(tgt, set()).add(src)

        # BFS connected components
        visited: set[str] = set()
        components: list[list[str]] = []
        for node in adjacency:
            if node in visited:
                continue
            component: list[str] = []
            queue: deque[str] = deque([node])
            visited.add(node)  # Mark at enqueue — prevents duplicate queue entries
            while queue:
                n = queue.popleft()
                component.append(n)
                for neighbor in adjacency.get(n, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)  # Mark at enqueue, not at pop
                        queue.append(neighbor)
            if len(component) >= min_cluster_size:
                components.append(sorted(component))

        # Name clusters by top keywords
        branch = self.get_active_branch()
        clusters: list[dict] = []
        for i, commit_ids in enumerate(components):
            keywords = self._extract_cluster_keywords(branch, commit_ids)
            name = (
                " ".join(w.title() for w in keywords[:4])
                if keywords
                else f"Cluster {i + 1}"
            )
            clusters.append({
                "id": f"CL{i + 1:03d}",
                "name": name,
                "commit_ids": commit_ids,
                "top_keywords": keywords[:6],
            })

        # Save clusters
        self._save_clusters(clusters)
        return clusters

    def _extract_cluster_keywords(
        self, branch: str, commit_ids: list[str]
    ) -> list[str]:
        """Extract top keywords from a set of commits."""
        from collections import Counter

        _cluster_stop_words = frozenset({
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "has", "have", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "that", "this",
            "it", "not", "no", "if", "when", "then", "than", "so", "as", "up",
            "out", "about", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "only", "own", "same", "into",
            "over", "after",
        })

        word_counts: Counter = Counter()
        for cid in commit_ids:
            block = self._find_commit_by_id(branch, cid)
            if not block:
                continue
            # Extract text from What and Title lines
            for line in block.split("\n"):
                if line.startswith("**What**:") or line.startswith("## [C"):
                    text = line.split("|")[-1] if "|" in line else line
                    text = re.sub(r"\*\*\w+\*\*:", "", text)  # Remove **Field**: markers
                    text = re.sub(r"\[C\d+\]", "", text)  # Remove commit refs
                    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
                    for w in words:
                        if w not in _cluster_stop_words:
                            word_counts[w] += 1

        return [w for w, _ in word_counts.most_common(10)]

    def _get_clusters_path(self) -> str:
        return os.path.join(self.ccr_root, "commit_clusters.json")

    def _save_clusters(self, clusters: list[dict]) -> None:
        """Save clusters to JSON using atomic write with file locking."""
        path = self._get_clusters_path()
        commit_to_cluster: dict[str, str] = {}
        for cl in clusters:
            for cid in cl.get("commit_ids", []):
                commit_to_cluster[cid] = cl["id"]

        data = {
            "version": 1,
            "clusters": clusters,
            "commit_to_cluster": commit_to_cluster,
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def _load_clusters(self) -> dict:
        """Load clusters from JSON. Returns default if missing/corrupt."""
        path = self._get_clusters_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
        if not raw:
            return {"version": 1, "clusters": [], "commit_to_cluster": {}}
        try:
            data = json.loads(raw)
            if not isinstance(data.get("clusters"), list):
                return {"version": 1, "clusters": [], "commit_to_cluster": {}}
            return data
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return {"version": 1, "clusters": [], "commit_to_cluster": {}}

    def format_clusters_for_context(self) -> str:
        """Format clusters for inclusion in gcc_context output."""
        data = self._load_clusters()
        clusters = data.get("clusters", [])
        if not clusters:
            return ""
        lines = ["# Thematic Clusters"]
        for cl in clusters:
            commit_str = ", ".join(cl["commit_ids"][:5])
            if len(cl["commit_ids"]) > 5:
                commit_str += f" (+{len(cl['commit_ids']) - 5} more)"
            lines.append(f"- **{cl['name']}** [{cl['id']}]: {commit_str}")
        return "\n".join(lines)

    # --- CER-Inspired Pattern Buffer (arXiv:2506.06698 §3.1) ---

    def _get_patterns_path(self) -> str:
        return os.path.join(self.ccr_root, "patterns.json")

    def _load_patterns(self) -> dict:
        """Load pattern buffer from JSON. Returns default if missing/corrupt."""
        path = self._get_patterns_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
        if not raw:
            return {"version": 1, "patterns": {}, "next_id": 1}
        try:
            data = json.loads(raw)
            if not isinstance(data.get("patterns"), dict):
                return {"version": 1, "patterns": {}, "next_id": 1}
            return data
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return {"version": 1, "patterns": {}, "next_id": 1}

    def _save_patterns(self, data: dict) -> None:
        """Atomically save the pattern buffer."""
        path = self._get_patterns_path()
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def _find_matching_pattern(self, data: dict, new_text: str) -> str | None:
        """Find existing pattern matching new_text. Primary: ONNX cosine. Fallback: word Jaccard.

        CER §3.1: existing buffer shown to distiller to avoid repetition.
        Returns matching pattern ID or None.
        """
        new_words = {w.lower() for w in new_text.split()
                     if w.lower() not in self._STOP_WORDS and len(w) > 2}
        if len(new_words) < 2:
            return None

        # Try ONNX cosine similarity first
        from ccr.context.embeddings import quick_cosine
        best_id = None
        best_sim = 0.0

        for pid, entry in data.get("patterns", {}).items():
            existing_text = entry["text"]
            existing_words = {w.lower() for w in existing_text.split()
                              if w.lower() not in self._STOP_WORDS and len(w) > 2}
            if len(existing_words) < 2:
                continue

            # ONNX primary, Jaccard fallback
            onnx_sim = quick_cosine(new_text, existing_text)
            sim = onnx_sim if onnx_sim is not None else self._jaccard(new_words, existing_words)

            if sim >= self.config.pattern_dedup_threshold and sim > best_sim:
                best_sim = sim
                best_id = pid

        return best_id

    def _process_patterns(
        self,
        commit_id: str,
        patterns: list[str],
        timestamp: str,
    ) -> list[dict]:
        """Process new patterns: dedup, store, track occurrences, suggest promotions.

        CER §3.1 Dynamic Experience Buffer: new skills are deduped against existing
        buffer (existing experiences shown to distiller to avoid repetition).

        Returns list of promotion suggestion dicts for patterns that crossed
        the promotion threshold.
        """
        path = self._get_patterns_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
            if not raw:
                data: dict = {"version": 1, "patterns": {}, "next_id": 1}
            else:
                try:
                    data = json.loads(raw)
                    if not isinstance(data.get("patterns"), dict):
                        data = {"version": 1, "patterns": {}, "next_id": 1}
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Failed to load %s: %s", path, exc)
                    data = {"version": 1, "patterns": {}, "next_id": 1}

            promotion_suggestions: list[dict] = []

            for pattern_text in patterns:
                pattern_text = pattern_text.strip()
                if not pattern_text:
                    continue

                # Dedup: find existing pattern with word Jaccard >= threshold
                matched_id = self._find_matching_pattern(data, pattern_text)

                if matched_id:
                    # Update existing pattern's occurrence
                    entry = data["patterns"][matched_id]
                    if commit_id not in entry["commit_ids"]:
                        entry["commit_ids"].append(commit_id)
                        entry["occurrence_count"] = len(entry["commit_ids"])
                    # CER recency tracking: update last_seen timestamp
                    entry["last_seen"] = timestamp

                    # Check promotion threshold
                    if (entry["occurrence_count"] >= self.config.pattern_promotion_count
                            and not entry.get("promoted", False)):
                        promotion_suggestions.append({
                            "pattern_id": matched_id,
                            "text": entry["text"],
                            "count": entry["occurrence_count"],
                            "commit_ids": entry["commit_ids"],
                        })
                else:
                    # New pattern — add to buffer
                    pid = f"P{data['next_id']:03d}"
                    data["next_id"] = data["next_id"] + 1
                    data["patterns"][pid] = {
                        "text": pattern_text,
                        "first_seen": commit_id,
                        "commit_ids": [commit_id],
                        "occurrence_count": 1,
                        "created_at": timestamp,
                        "promoted": False,
                        "success_count": 0,
                        "failure_count": 0,
                        "quality_score": 0.5,
                        "last_quality_update": "",
                    }

            # Buffer size enforcement (CER §3.1 Dynamic Buffer)
            self._enforce_pattern_buffer_size(data)

            content = json.dumps(data, indent=2, ensure_ascii=False)
            self._write_file_unlocked(path, content)

        return promotion_suggestions

    def _enforce_pattern_buffer_size(self, data: dict) -> None:
        """Evict lowest-value patterns if buffer exceeds max size."""
        patterns = data.get("patterns", {})
        max_size = self.config.pattern_max_buffer_size
        if len(patterns) <= max_size:
            return

        # Sort by (quality_score ASC, occurrence_count ASC, created_at ASC)
        # Evict lowest quality + least frequent + oldest first
        sorted_ids = sorted(
            patterns.keys(),
            key=lambda pid: (
                patterns[pid].get("quality_score", 0.5),
                patterns[pid].get("occurrence_count", 1),
                patterns[pid].get("created_at", ""),
            ),
        )

        # Evict from the front (lowest value) until within budget
        evict_count = len(patterns) - max_size
        for pid in sorted_ids[:evict_count]:
            del patterns[pid]

    def _scan_pending_promotions(self) -> list[dict]:
        """Scan the pattern buffer for patterns that crossed the promotion threshold.

        CER §3.1 (CCR extension): surfaces ready-to-promote patterns even when no
        new patterns_learned are passed to commit(). This ensures promotable patterns
        are not silently ignored when the caller omits patterns_learned.

        Returns a list of promotion suggestion dicts (same shape as _process_patterns):
            {pattern_id, text, count, commit_ids}

        Capped at 5 results to avoid flooding the commit response.
        Does NOT mark patterns as promoted — that only happens when the user
        explicitly calls ace_apply_delta.
        """
        data = self._load_patterns()
        threshold = self.config.pattern_promotion_count

        suggestions: list[dict] = []
        for pid, entry in data.get("patterns", {}).items():
            if (entry.get("occurrence_count", 0) >= threshold
                    and not entry.get("promoted", False)):
                suggestions.append({
                    "pattern_id": pid,
                    "text": entry["text"],
                    "count": entry["occurrence_count"],
                    "commit_ids": entry.get("commit_ids", []),
                })

        # Sort by occurrence_count DESC for deterministic ordering
        suggestions.sort(key=lambda x: -x["count"])

        return suggestions[:5]

    def mark_pattern_promoted_by_content(self, text: str) -> int:
        """Mark patterns matching `text` as promoted (CER buffer management).

        Called after ace_apply_delta ADD to close the loop: if the user added
        a bullet matching a pending pattern, that pattern is now promoted.

        Returns: number of patterns marked promoted.
        """
        path = self._get_patterns_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
            if not raw:
                return 0
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to load %s: %s", path, exc)
                return 0

            matched_id = self._find_matching_pattern(data, text)
            if matched_id is None:
                return 0

            entry = data["patterns"][matched_id]
            if entry.get("promoted", False):
                return 0  # Already promoted

            entry["promoted"] = True
            self._write_file_unlocked(path, json.dumps(data, indent=2))
            return 1

    def update_pattern_quality(self, pattern_text: str, success: bool) -> bool:
        """Update quality score for a pattern based on its promoted bullet's performance.

        Called when ace_update_counters tags a bullet that was promoted from a pattern.
        Uses Bayesian average: (success+1) / (success+failure+2).

        Inspired by EvolveR (arXiv:2510.16079) quality-scored pattern buffer.

        Args:
            pattern_text: The pattern text to find (uses word Jaccard matching).
            success: True if bullet was tagged helpful, False if harmful.

        Returns:
            True if a matching pattern was found and updated.
        """
        from datetime import datetime, timezone

        path = self._get_patterns_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
            if not raw:
                return False
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to load %s: %s", path, exc)
                return False

            # Find matching pattern by word Jaccard
            match_pid = self._find_matching_pattern(data, pattern_text)
            if match_pid is None:
                return False

            p = data["patterns"][match_pid]
            # Ensure quality fields exist (backward compat with old format)
            p.setdefault("success_count", 0)
            p.setdefault("failure_count", 0)
            if success:
                p["success_count"] += 1
            else:
                p["failure_count"] += 1

            # Recompute Bayesian quality score
            sc = p["success_count"]
            fc = p["failure_count"]
            p["quality_score"] = (sc + 1) / (sc + fc + 2)
            p["last_quality_update"] = datetime.now(timezone.utc).isoformat()

            self._write_file_unlocked(
                path, json.dumps(data, indent=2, ensure_ascii=False)
            )
            return True

    def get_patterns(
        self,
        min_occurrences: int = 1,
        include_promoted: bool = True,
        search_term: str | None = None,
        max_age_hours: int | None = None,
    ) -> dict:
        """Query the pattern buffer. Returns dict for MCP tool formatting.

        CER-inspired recency-weighted retrieval: patterns seen recently
        are ranked higher. Combines quality_score with temporal decay
        on last_seen timestamp (λ=0.005/hour, half-life ~139h).
        """
        data = self._load_patterns()
        results = []
        # Snapshot current time once to avoid floating point drift between entries
        now = datetime.now(timezone.utc)
        age_cutoff = (
            now - timedelta(hours=max_age_hours) if max_age_hours is not None else None
        )

        for pid, entry in data.get("patterns", {}).items():
            if entry.get("occurrence_count", 1) < min_occurrences:
                continue
            if not include_promoted and entry.get("promoted", False):
                continue
            if search_term:
                if search_term.lower() not in entry["text"].lower():
                    continue
            if age_cutoff is not None:
                last_seen = entry.get("last_seen", entry.get("created_at", ""))
                if last_seen:
                    try:
                        ts = datetime.fromisoformat(last_seen.strip().replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts < age_cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass  # Skip age check for unparseable timestamps
            # CER recency-weighted retrieval: compute effective score
            quality = entry.get("quality_score", 0.5)
            last_seen = entry.get("last_seen", entry.get("created_at", ""))
            recency_weight = self._pattern_recency_weight_at(last_seen, now)
            entry["effective_score"] = quality * recency_weight
            results.append({"id": pid, **entry})

        # Sort by effective_score DESC (quality * recency), then occurrence_count DESC
        # (EvolveR + CER: high-quality recent patterns surface first)
        results.sort(key=lambda x: (
            -x.get("effective_score", 0.5),
            -x.get("occurrence_count", 1),
        ))

        return {
            "total": len(data.get("patterns", {})),
            "matching": len(results),
            "patterns": results,
        }

    @staticmethod
    def _pattern_recency_weight_at(timestamp_str: str, now: datetime) -> float:
        """Compute recency weight for pattern retrieval (CER-inspired).

        Uses decay formula: exp(-0.005 * hours). Half-life ~139h.
        Returns 1.0 for recent patterns, decaying toward 0 for old ones.
        `now` is passed in to avoid floating point drift between entries.
        """
        if not timestamp_str:
            return 0.5  # Default moderate weight for undated patterns
        # Try ISO-8601 first (modern format used by datetime.now().isoformat())
        # Replace 'Z' suffix for Python < 3.11 compat (fromisoformat rejects 'Z' before 3.11)
        try:
            ts = datetime.fromisoformat(timestamp_str.strip().replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            hours = (now - ts).total_seconds() / 3600.0
            return math.exp(-0.005 * max(0.0, hours))
        except (ValueError, TypeError):
            pass
        # Legacy fallback: "%Y-%m-%d %H:%M" format
        try:
            ts = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            hours = (now - ts).total_seconds() / 3600.0
            return math.exp(-0.005 * max(0.0, hours))
        except (ValueError, TypeError):
            return 0.5

    # --- Commit Type Classification (A-MAC §3.2 Factor 5: Type Prior T(m)) ---

    @staticmethod
    def _classify_commit_type(title: str, what: str, files_changed: list[str]) -> str:
        """Classify commit type for Type Prior scoring.

        A-MAC Table 2 ablation shows Type Prior is the most impactful factor
        (ΔF1 = -0.107). We use a lightweight rule-based classifier since CCR
        has no sub-model. Types ranked by admission priority:

        - "merge": branch merge operations → always admit (structural)
        - "branch": branch create/switch → always admit (structural)
        - "structural": architecture, refactor, new module → high priority
        - "milestone": feature complete, bug fixed → high priority
        - "progress": incremental work → normal priority
        - "continuation": "still working on", "continued" → low priority (likely redundant)
        """
        title_lower = title.lower()
        what_lower = what.lower()
        combined = f"{title_lower} {what_lower}"

        # Structural operations (always admit)
        if title_lower.startswith("merge:") or "merged" in title_lower:
            return "merge"
        if "branch" in title_lower and ("create" in combined or "switch" in combined):
            return "branch"

        # Continuation signals (low priority — likely redundant)
        continuation_signals = [
            "still working", "continued", "continuing", "more work on",
            "progress on", "wip", "work in progress", "incremental",
        ]
        if any(sig in combined for sig in continuation_signals):
            return "continuation"

        # Milestone signals (high priority)
        milestone_signals = [
            "complete", "finished", "done", "fixed", "resolved",
            "implemented", "deployed", "released", "shipped",
        ]
        if any(sig in combined for sig in milestone_signals):
            return "milestone"

        # Structural signals (high priority)
        structural_signals = [
            "refactor", "architect", "restructur", "new module",
            "migration", "schema", "breaking change", "api change",
        ]
        if any(sig in combined for sig in structural_signals):
            return "structural"

        # Default: progress
        return "progress"

    @staticmethod
    def _type_prior(commit_type: str) -> float:
        """Return Type Prior T(m) — admission bias by commit type.

        Per A-MAC §3.2 Factor 5 and Table 2: Type Prior is the most impactful
        factor. Structural operations always get high priority (T=1.0),
        continuations get low priority (T=0.2) making them easier to merge.

        Returns value in [0, 1] where higher = more likely to be admitted as new.
        """
        priors = {
            "merge": 1.0,       # Always admit — structural operations
            "branch": 1.0,      # Always admit — structural operations
            "structural": 0.9,  # High priority — architecture changes
            "milestone": 0.85,  # High priority — completed work
            "progress": 0.5,    # Normal priority — default
            "continuation": 0.2,  # Low priority — likely redundant
        }
        return priors.get(commit_type, 0.5)

    def compute_admission_score(
        self,
        branch: str,
        title: str,
        what: str,
        why: str,
        files_changed: list[str],
        next_step: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """Compute admission score S(m) for a new commit per A-MAC Algorithm 1.

        Implements A-MAC (arXiv:2603.04549) with correct polarity:
        **Higher score = more valuable = more likely to admit as new.**

        A-MAC Eq. 1 (adapted — 4 of 5 factors):
            S(m) = w_T · T(m) + w_N · N(m) + w_R · R(m) + w_U · U(m)

        Factors implemented:
            N(m) = 1 − max_{m' ∈ M} sim(φ(m), φ(m'))    [Eq. 3, Novelty]
            R(m) = exp(−λ · τ(m)), λ=0.01/hour           [Eq. 4, Recency]
            T(m) = type_prior(classify(m))                 [§3.2 Factor 5]
            U(m) = utility_heuristic(commit richness)      [§3.2 Factor 3, rule-based proxy]

        Factor NOT implemented:
            C(m) — Confidence: requires ROUGE-L against source turns

        Similarity φ(m): ONNX cosine (preferred, A-MAC Eq. 3) or word Jaccard
        (fallback when ONNX unavailable).

        When the ONNX embedding model is available (ccr[semantic] extras),
        raw_sim uses cosine similarity on all-MiniLM-L6-v2 vectors — matching
        the Sentence-BERT spirit of A-MAC Eq. 3. When ONNX is unavailable,
        falls back to:
            sim(m, m') = 0.50 · Jaccard(files) + 0.50 · Jaccard(keywords)
            Word Jaccard substitutes for Sentence-BERT cosine.
            Lower discriminative power but zero cost.

            Two similarity signals (separated per paper):
            - Raw sim: pure content similarity, used for Novelty N(m) = 1 - max_sim
            - Effective sim: raw_sim · R_conflict(m'), used for FindConflict
              threshold checking only. Old conflicts are dampened per Eq. 4.

        Weight rationale (per Table 2 ablation — T is most impactful, ΔF1=-0.107):
            w_T = 0.40: Type Prior dominates — most impactful factor per ablation.
            w_N = 0.30: Novelty — "does this add new information?"
            w_U = 0.20: Utility — "how rich and useful is this commit?"
            w_R = 0.10: Recency — included per Eq. 1 (for new commits R=1.0,
                but recomputed for S(m_conflict) to decay old scores properly).
            R also modulates FindConflict similarity (old conflicts dampened).

        Returns dict with:
            score: float — S(m) ∈ [0,1], higher = more valuable (paper Eq. 1)
            similarity: float — max recency-weighted sim (for FindConflict threshold)
            novelty: float — N(m) = 1 - max_raw_sim (Eq. 3, pure content, no recency)
            conflict_id: str|None — most similar commit ID (FindConflict target)
            conflict_recency: float — R of the conflicting commit (Eq. 4)
            commit_type: str — classified type of the new commit
            type_prior: float — T(m) (§3.2 Factor 5)
            file_similarity: float — file Jaccard with conflict target
            keyword_similarity: float — keyword Jaccard with conflict target
            reason: str — human-readable explanation
        """
        recent = self._parse_recent_commit_data(branch, k)
        if not recent:
            return {
                "score": 1.0, "similarity": 0.0, "novelty": 1.0,
                "conflict_id": None, "conflict_recency": 0.0,
                "conflict_score": 0.0,
                "commit_type": "progress", "type_prior": 0.5,
                "file_similarity": 0.0, "keyword_similarity": 0.0,
                "reason": "first commit",
            }

        # --- Type Prior T(m) (A-MAC §3.2 Factor 5) ---
        commit_type = self._classify_commit_type(title, what, files_changed)
        tp = self._type_prior(commit_type)

        # Structural operations always admitted (merge/branch)
        if commit_type in ("merge", "branch"):
            return {
                "score": 1.0, "similarity": 0.0, "novelty": 1.0,
                "conflict_id": None, "conflict_recency": 0.0,
                "conflict_score": 0.0,
                "commit_type": commit_type, "type_prior": tp,
                "file_similarity": 0.0, "keyword_similarity": 0.0,
                "reason": f"structural ({commit_type})",
            }

        # --- Similarity computation: max over all k commits (Eq. 3) ---
        new_files = {f.strip().lower() for f in files_changed if f.strip()}
        new_words = self._extract_keywords(f"{title} {what} {why}")

        # A-MAC §3.2 Eq. 3: ONNX cosine similarity φ(m) when available.
        # Compute the new commit's embedding once, outside the per-commit loop.
        new_vec_for_admission = None
        try:
            from ccr.context.embeddings import get_embedding_model
            _emb_model = get_embedding_model()
            if _emb_model is not None:
                new_vec_for_admission = _emb_model.embed_query(
                    self._commit_text(title, what, why)
                )
        except Exception:
            pass  # Fall back to Jaccard on any error

        # Pre-load all recent commit embeddings once (not per-commit) — avoids
        # repeated file I/O inside the loop.
        if new_vec_for_admission is not None and recent:
            recent_ids = [c.get("id", "") for c in recent if c.get("id")]
            all_cached_vecs = self._load_commit_embeddings(recent_ids)
        else:
            all_cached_vecs = {}

        best_similarity = 0.0       # Recency-modulated sim (for FindConflict threshold)
        best_raw_similarity = 0.0   # Pure content sim (for Novelty N(m) per Eq. 3)
        best_file_sim = 0.0
        best_keyword_sim = 0.0
        best_conflict_id = None
        best_conflict_recency = 0.0
        best_conflict_score = 0.0  # S(m_conflict) for Alg. 1 line 6

        for commit in recent:
            old_files = {f.strip().lower() for f in commit.get("files", []) if f.strip()}
            file_sim = self._jaccard(new_files, old_files)

            old_words = self._extract_keywords(
                f"{commit.get('title', '')} {commit.get('what', '')} {commit.get('why', '')}"
            )
            keyword_sim = self._jaccard(new_words, old_words)

            # Raw content similarity (pure, no recency — used for Novelty per Eq. 3).
            # Try ONNX cosine similarity first (A-MAC Eq. 3); fall back to Jaccard.
            commit_id = commit.get("id", "")
            old_vec = all_cached_vecs.get(commit_id) if all_cached_vecs else None
            if new_vec_for_admission is not None and old_vec is not None:
                try:
                    import numpy as np
                    raw_sim = float(np.dot(new_vec_for_admission, old_vec))
                    # Note: both vectors are L2-normalized, so dot product = cosine similarity
                except Exception:
                    raw_sim = 0.50 * file_sim + 0.50 * keyword_sim  # Jaccard fallback
            else:
                raw_sim = 0.50 * file_sim + 0.50 * keyword_sim  # Jaccard fallback

            # Track best raw similarity across all commits for Novelty computation
            if raw_sim > best_raw_similarity:
                best_raw_similarity = raw_sim

            # Recency of existing commit (Eq. 4: λ=0.01/hour, half-life ~69 hours)
            hours_since = self._hours_since_commit(commit.get("timestamp", ""))
            conflict_recency = math.exp(-0.01 * hours_since) if hours_since >= 0 else 0.0

            # Recency-modulated similarity for FindConflict threshold checking.
            # Old conflicts are dampened — a stale duplicate is less of a conflict.
            # NOTE: This is used ONLY for FindConflict (whether a conflict exists),
            # NOT for Novelty N(m). The paper's Eq. 3 uses pure content similarity.
            effective_sim = raw_sim * conflict_recency

            if effective_sim > best_similarity:
                best_similarity = effective_sim
                best_file_sim = file_sim
                best_keyword_sim = keyword_sim
                best_conflict_id = commit.get("id")
                best_conflict_recency = conflict_recency

                # --- Compute S(m_conflict) per Alg. 1 line 6 ---
                # Recompute with CURRENT recency R(m') per Eq. 4, so old
                # commits properly decay. Stored scores were computed at
                # creation time (R=1.0) and don't reflect temporal decay.
                conflict_type = self._classify_commit_type(
                    commit.get("title", ""), commit.get("what", ""), commit.get("files", []),
                )
                conflict_tp = self._type_prior(conflict_type)
                # Use stored novelty proxy or estimate from recency
                stored = commit.get("stored_score")
                if stored is not None:
                    # Extract approximate novelty from stored score:
                    # stored = 0.50*T + 0.35*N + 0.15*1.0 → N ≈ (stored - 0.50*T - 0.15) / 0.35
                    conflict_novelty = max(0.0, min(1.0, (stored - 0.50 * conflict_tp - 0.15) / 0.35)) if stored > 0.15 else 0.5
                else:
                    conflict_novelty = 0.7  # Assume moderate novelty for old commits
                # Recompute S(m') with current R(m') per Eq. 1
                # Conflict utility: estimate from stored commit richness
                conflict_utility = 0.5  # Default moderate utility for existing commits
                best_conflict_score = 0.40 * conflict_tp + 0.30 * conflict_novelty + 0.10 * conflict_recency + 0.20 * conflict_utility

        # Novelty N(m) = 1 - max_similarity (Eq. 3, pure content similarity)
        # Uses best_raw_similarity (no recency modulation) to separate Novelty
        # from Recency as the paper intends. Recency modulation is only used
        # for FindConflict threshold checking (whether a conflict exists).
        novelty = 1.0 - best_raw_similarity

        # Recency R(m) per Eq. 4: for new commits being created now, R=1.0
        # But R is properly included per Eq. 1 so S(m_conflict) can be
        # recomputed with decayed R for old commits.
        recency = 1.0  # New commit — τ(m)=0, so R(m)=exp(0)=1.0

        # --- Utility heuristic U(m) (A-MAC §3.2 Factor 3, rule-based proxy) ---
        # Paper uses LLM to rate future usefulness on 1-5 scale.
        # CCR proxy: score based on commit richness (more detail = higher utility).
        utility = self._utility_heuristic(what, why, files_changed, next_step)

        # Admission score S(m) = w_T*T + w_N*N + w_R*R + w_U*U (Eq. 1, 4 factors)
        # Weights rebalanced from 3-factor (0.50/0.35/0.15) to include utility:
        # T remains most impactful (Table 2 ablation); U gets moderate weight.
        admission_score = 0.40 * tp + 0.30 * novelty + 0.10 * recency + 0.20 * utility

        reason_parts = []
        if best_similarity > 0.5:
            reason_parts.append(f"sim={best_similarity:.2f}")
        if novelty > 0.7:
            reason_parts.append("novel")
        if utility > 0.7:
            reason_parts.append(f"utility={utility:.2f}")
        if tp >= 0.85:
            reason_parts.append(f"type={commit_type}")
        if tp <= 0.3:
            reason_parts.append(f"type={commit_type}")

        return {
            "score": admission_score,
            "similarity": best_similarity,
            "novelty": novelty,
            "utility": utility,
            "conflict_id": best_conflict_id,
            "conflict_recency": best_conflict_recency,
            "conflict_score": best_conflict_score,
            "commit_type": commit_type,
            "type_prior": tp,
            "file_similarity": best_file_sim,
            "keyword_similarity": best_keyword_sim,
            "reason": ", ".join(reason_parts) if reason_parts else "moderate",
        }

    @staticmethod
    def _utility_heuristic(
        what: str, why: str, files_changed: list[str], next_step: str,
    ) -> float:
        """Rule-based proxy for A-MAC Utility U(m) (§3.2 Factor 3).

        Paper uses LLM to rate future usefulness on 1-5 scale.
        CCR proxy: scores commit richness on [0, 1]. More detailed commits
        with explicit reasoning, multiple files, and forward planning are
        rated as higher utility.

        Scoring signals (each contributes up to ~0.2):
        - what length: longer descriptions → higher utility
        - why present and non-trivial: reasoning → higher utility
        - files_changed count: more files → broader impact
        - next_step present: forward planning → higher utility
        - what contains patterns/learnings: meta-cognitive → higher utility
        """
        score = 0.0

        # What description richness (up to 0.25)
        what_len = len(what.strip())
        if what_len > 200:
            score += 0.25
        elif what_len > 100:
            score += 0.20
        elif what_len > 50:
            score += 0.15
        elif what_len > 20:
            score += 0.10

        # Why reasoning present (up to 0.20)
        why_len = len(why.strip())
        if why_len > 50:
            score += 0.20
        elif why_len > 20:
            score += 0.15
        elif why_len > 0:
            score += 0.10

        # File breadth (up to 0.20)
        n_files = len(files_changed)
        if n_files >= 5:
            score += 0.20
        elif n_files >= 3:
            score += 0.15
        elif n_files >= 1:
            score += 0.10

        # Forward planning (up to 0.15)
        if next_step and len(next_step.strip()) > 10:
            score += 0.15
        elif next_step and next_step.strip():
            score += 0.10

        # Meta-cognitive signals (up to 0.20)
        what_lower = what.lower()
        meta_signals = ["pattern", "learn", "insight", "principle", "realized", "discovered"]
        if any(s in what_lower for s in meta_signals):
            score += 0.20

        return min(1.0, score)

    def _hours_since_commit(self, timestamp_str: str) -> float:
        """Parse commit timestamp and return hours elapsed.

        Per A-MAC Eq. 4: τ(m) is measured in hours (λ=0.01/hour).
        Handles both timezone-aware and timezone-naive timestamps
        for backward compatibility with existing commits.
        """
        if not timestamp_str:
            return 999.0  # treat missing as very old
        try:
            commit_time = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M")
            # Assume naive timestamps are UTC (backward compat)
            if commit_time.tzinfo is None:
                commit_time = commit_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - commit_time
            return delta.total_seconds() / 3600.0
        except (ValueError, TypeError):
            return 999.0

    def _merge_into_last_commit(
        self,
        branch: str,
        commit_id: str,
        title: str,
        what: str,
        why: str,
        files_changed: list[str],
        next_step: str,
        patterns_learned: list[str] | None = None,
    ) -> str:
        """Merge new commit data into the most recent commit (admission control).

        Updates the last commit in-place: appends what/why, unions files,
        replaces next_step with the newer one.

        Returns the merged commit ID.
        """
        path = self._get_commits_path(branch)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        files_str = ", ".join(files_changed) if files_changed else ""

        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""

            # Find the target commit block
            pattern = rf"(## \[{re.escape(commit_id)}\].*?)(?=## \[C\d{{3,}}\]|---\n\n|\Z)"
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                return commit_id  # fallback: can't find commit

            old_block = match.group(1)

            # Update timestamp to now (H5: use lambda to avoid backreference interpretation)
            new_block = re.sub(
                r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})",
                lambda m: now,
                old_block,
                count=1,
            )

            # Append to title (keep original, note merge)
            # Use lambda to avoid regex backreference interpretation of title
            new_block = re.sub(
                rf"(\[{re.escape(commit_id)}\]\s+{re.escape(now)}\s*\|[^|]*\|\s*)(.*)",
                lambda m: f"{m.group(1)}{m.group(2)} + {title}",
                new_block,
            )

            # Append to What
            what_match = re.search(r"(\*\*What\*\*:\s*)(.*)", new_block)
            if what_match:
                merged_what = f"{what_match.group(2).strip()}; {what}"
                # Cap length
                if len(merged_what) > 500:
                    merged_what = merged_what[-500:]
                new_block = new_block.replace(
                    what_match.group(0),
                    f"{what_match.group(1)}{merged_what}",
                )

            # Append to Why
            why_match = re.search(r"(\*\*Why\*\*:\s*)(.*)", new_block)
            if why_match:
                old_why = why_match.group(2).strip()
                # Only append if substantially different
                if why.lower().strip() not in old_why.lower():
                    merged_why = f"{old_why}; {why}"
                    if len(merged_why) > 300:
                        merged_why = merged_why[-300:]
                    new_block = new_block.replace(
                        why_match.group(0),
                        f"{why_match.group(1)}{merged_why}",
                    )

            # Union files
            files_match_re = re.search(r"(\*\*Files\*\*:\s*)(.*)", new_block)
            if files_match_re and files_str:
                old_files_str = files_match_re.group(2).strip()
                old_set = {f.strip() for f in old_files_str.split(",") if f.strip() and f.strip() != "(none)"}
                new_set = {f.strip() for f in files_changed if f.strip()}
                all_files = sorted(old_set | new_set)
                new_block = new_block.replace(
                    files_match_re.group(0),
                    f"{files_match_re.group(1)}{', '.join(all_files)}",
                )

            # Union patterns (CER-inspired)
            if patterns_learned:
                patterns_match_re = re.search(r"(\*\*Patterns\*\*:\s*)(.*)", new_block)
                if patterns_match_re:
                    old_patterns = [p.strip() for p in patterns_match_re.group(2).split("|") if p.strip()]
                    merged_patterns = list(dict.fromkeys(old_patterns + patterns_learned))
                    new_block = new_block.replace(
                        patterns_match_re.group(0),
                        f"{patterns_match_re.group(1)}{' | '.join(merged_patterns)}",
                    )
                else:
                    # Insert patterns line before **Next** or **Score**
                    insert_match = re.search(r"(\*\*(?:Next|Score)\*\*:)", new_block)
                    if insert_match:
                        insert = f"**Patterns**: {' | '.join(patterns_learned)}\n"
                        new_block = new_block[:insert_match.start()] + insert + new_block[insert_match.start():]

            # Replace Next with newer value
            next_match = re.search(r"(\*\*Next\*\*:\s*)(.*)", new_block)
            if next_match:
                new_block = new_block.replace(
                    next_match.group(0),
                    f"{next_match.group(1)}{next_step}",
                )

            content = content[:match.start()] + new_block + content[match.end():]
            self._write_file_unlocked(path, content)
            self._invalidate_commit_index(branch)

        return commit_id

    # --- Rolling Summary (GCC paper §2.2) ---

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
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                r"(## Rolling Summary\n).*?(?=\n---|\n# |\Z)",
                lambda m: f"{m.group(1)}{summary}\n",
                content,
                count=1,
                flags=re.DOTALL,
            )
            self._write_file_unlocked(path, content)

    # --- Merge Operations ---

    def merge(self, branch_name: str, outcome: str, conclusion: str) -> str:
        """Merge a branch back into main."""
        if branch_name == "main":
            raise ValueError("Cannot merge main into itself.")
        if outcome not in ("success", "failure", "partial"):
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
            "merge", f"→ main ({outcome})", "OK",
            observation=f"Branch '{branch_name}' ready for merge",
            thought=f"Outcome: {outcome}. {conclusion}",
            action=f"Merged into main, switching back",
        ))
        self._append_log("main", self._format_ota_log(
            "merge", f"← {branch_name} ({outcome})", "OK",
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

    # --- Context Retrieval ---

    def get_context(
        self,
        level: int = 1,
        branch: str | None = None,
        commit_id: str | None = None,
        search_term: str | None = None,
        offset: int = 0,
        log_window: int = 0,
        metadata_segment: str | None = None,
        follow_links: bool = False,
    ) -> str:
        """Multi-level context retrieval with windowing support.

        Level 1: main.md only (~200 tokens)
        Level 2: + last 3 commits from active branch (windowed by offset)
        Level 3: + branch summary.md (purpose/hypothesis/conclusion)
        Level 4: + last 10 commits (windowed by offset)
        Level 5: + specific commit by ID or keyword search + cross-links

        Args:
            offset: Scroll position for commit window (0 = most recent)
            log_window: Number of recent OTA log entries to include (0 = none)
            metadata_segment: Metadata key to include (e.g. "file_tree", "dependencies")
            follow_links: If True and level >= 5, include linked commit summaries (BFS 1-hop).
                When ONNX embeddings are available, each linked commit entry includes an
                ``embedding_score`` tag (e.g. ``[emb: 0.920]``) showing dense cosine similarity.
        """
        branch = branch or self.get_active_branch()
        parts = []

        # Level 1: main.md + project overview (TiMem L5)
        main_content = self._read_file(os.path.join(self.ccr_root, "main.md"))
        if main_content:
            parts.append(f"# Project Overview\n{main_content}")
        # Include generated overview if available (TiMem L5 profile)
        overview = self._read_file(self._get_overview_path())
        if overview:
            parts.append(f"# Project Summary\n{overview}")

        if level >= 2:
            # Level 2: rolling summary + session summary + recent commits
            rolling = self._get_rolling_summary(branch)
            if rolling:
                parts.append(f"# Progress Summary ({branch})\n{rolling}")

            # TiMem L2: include most recent session summary (replaces ~5 raw commits)
            sessions = self._read_session_summaries(branch, count=1)
            if sessions:
                s = sessions[0]
                parts.append(
                    f"# Session Summary ({s.get('id', '?')})\n"
                    f"Commits: {s.get('commits', '?')} | {s.get('start_date', '?')} - {s.get('end_date', '?')}\n"
                    f"Accomplished: {s.get('accomplished', '?')}\n"
                    f"Direction: {s.get('direction', '?')}"
                )

            count = 3 if level < 4 else 10
            recent = self._read_commits_window(branch, offset, count)
            if recent:
                # A-MEM §3.3: inject [evolved] tag for commits with evolved summaries
                try:
                    recent = self._inject_evolved_tags(recent)
                except Exception:
                    pass  # Evolution display is supplementary
                parts.append(f"# Recent Commits ({branch}, offset={offset})\n{recent}")

            # CER pattern buffer: show high-occurrence patterns (arXiv:2506.06698)
            try:
                pattern_data = self._load_patterns()
                recurring = [
                    (pid, entry) for pid, entry in pattern_data.get("patterns", {}).items()
                    if entry.get("occurrence_count", 1) >= 2
                ]
                if recurring:
                    recurring.sort(key=lambda x: -x[1]["occurrence_count"])
                    pattern_lines = ["# Recurring Patterns"]
                    for pid, entry in recurring[:10]:
                        promoted_tag = " [promoted]" if entry.get("promoted") else ""
                        pattern_lines.append(
                            f"- [{pid}] ({entry['occurrence_count']}x){promoted_tag} "
                            f"{entry['text'][:150]}"
                        )
                    parts.append("\n".join(pattern_lines))
            except Exception:
                pass  # Pattern display is supplementary

        if level >= 3:
            # Level 3: branch summary + phase summary
            if branch != "main":
                summary = self._read_branch_summary(branch)
                if summary:
                    parts.append(f"# Branch: {branch}\n{summary}")
                else:
                    header = self._get_branch_header(branch)
                    if header:
                        parts.append(f"# Branch: {branch}\n{header}")
                # Include branch session summaries
                branch_sessions = self._read_session_summaries(branch, count=3)
                if branch_sessions:
                    sess_lines = [f"# Branch Sessions ({branch})"]
                    for s in branch_sessions:
                        sess_lines.append(f"- [{s['id']}] {s.get('accomplished', '')[:150]}")
                    parts.append("\n".join(sess_lines))
            # Include recent phase summaries
            phases = self._read_phase_summaries(count=2)
            if phases:
                phase_lines = ["# Phase History"]
                for p in phases:
                    phase_lines.append(
                        f"**[{p['id']}]** {p.get('scope', '?')}: {p.get('outcome', '?')}\n"
                        f"  {p.get('accomplishments', '')[:200]}"
                    )
                parts.append("\n".join(phase_lines))

        if level >= 5:
            # Level 5: specific commit or search + cross-links
            if commit_id:
                found = self._find_commit_by_id(branch, commit_id)
                if found:
                    parts.append(f"# Commit {commit_id}\n{found}")
                    # Show heuristic cross-links
                    commit_links_data = self.get_commit_links(commit_id)
                    if any(v for v in commit_links_data.values()):
                        parts.append(self._format_links_for_context(commit_id, commit_links_data))
                    # Optionally include linked commit summaries
                    if follow_links:
                        linked = self.get_linked_commits(commit_id, max_hops=1)
                        for lc in linked[:5]:
                            emb_tag = f" [emb: {lc['embedding_score']:.3f}]" if "embedding_score" in lc else ""
                            parts.append(
                                f"## Linked: [{lc['id']}] {lc.get('title', '')} ({lc['link_type']}){emb_tag}\n"
                                f"**What**: {lc.get('what', '')}"
                            )
            elif search_term:
                results = self._search_commits(branch, search_term)
                if results:
                    parts.append(f"# Search: '{search_term}'\n{results}")

        # CONTEXT --log: windowed OTA log retrieval (independent of level)
        if log_window > 0:
            log_content = self._read_log_window(branch, log_window)
            if log_content:
                parts.append(f"# Execution Log ({branch}, last {log_window} entries)\n{log_content}")

        # CONTEXT --metadata: metadata segment retrieval (independent of level)
        if metadata_segment:
            import json
            meta = self._load_metadata()
            if metadata_segment in meta:
                segment_data = meta[metadata_segment]
                if isinstance(segment_data, (list, dict)):
                    parts.append(f"# Metadata: {metadata_segment}\n{json.dumps(segment_data, indent=2)}")
                else:
                    parts.append(f"# Metadata: {metadata_segment}\n{segment_data}")

        return "\n\n".join(parts)

    # --- OTA Logging (Observation-Thought-Action triples per GCC paper) ---

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

    # --- Session Context (for gateway injection) ---

    def get_session_context(self) -> str:
        """Return compact level-1 context for system prompt injection."""
        return self.get_context(level=1)

    # --- Index Cache ---

    def get_index_path(self) -> str:
        return os.path.join(self.ccr_root, "index.json")

    def save_index(self, index_json: str) -> None:
        self._write_file(self.get_index_path(), index_json)

    def load_index(self) -> str | None:
        path = self.get_index_path()
        if os.path.isfile(path):
            return self._read_file(path)
        return None

    # --- Hierarchical Summary Helpers (TiMem-inspired, §3.1-3.2) ---
    #
    # NOTE: This is a TiMem-INSPIRED aggregation, not a full TiMem implementation.
    # TiMem's two core contributions — the Temporal Memory Tree (TMT, §3.1) and
    # the relevance-scored recall pipeline (§3.3) — are NOT implemented here.
    # We use a flat three-tier summary system (session/phase/overview) with
    # mechanical consolidation (template extraction + Jaccard dedup) instead of
    # TiMem's hierarchical tree with consolidation functions Phi_i.
    # This avoids the sub-model dependency that TiMem's Phi functions require.

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
        path = self._get_summary_meta_path()
        with self._locks[path]:
            if not os.path.isfile(path):
                return self._default_summary_meta()
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data else self._default_summary_meta()

    def _save_summary_meta(self, data: dict) -> None:
        path = self._get_summary_meta_path()
        with self._locks[path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            content = yaml.dump(data, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper)
            tmp_path = path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def _get_commit_count(self, branch: str) -> int:
        """Count total commits on a branch."""
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return 0
        return len(re.findall(r"## \[C\d{3,}\]", content))

    def _get_next_session_summary_id(self, branch: str) -> str:
        """Get next S### ID for session summaries on a branch."""
        content = self._read_file(self._get_summaries_path(branch))
        if not content:
            return "S001"
        matches = re.findall(r"\[S(\d{3,})\]", content)
        if not matches:
            return "S001"
        return f"S{max(int(m) for m in matches) + 1:03d}"

    def _get_next_phase_summary_id(self) -> str:
        """Get next P### ID for phase summaries."""
        content = self._read_file(self._get_phases_path())
        if not content:
            return "P001"
        matches = re.findall(r"\[P(\d{3,})\]", content)
        if not matches:
            return "P001"
        return f"P{max(int(m) for m in matches) + 1:03d}"

    def _read_session_summaries(self, branch: str, count: int = 3) -> list[dict]:
        """Parse last `count` session summaries from summaries.md.

        Returns list of dicts with: id, start_date, end_date, branch,
        commits, accomplished, files, direction.
        """
        content = self._read_file(self._get_summaries_path(branch))
        if not content:
            return []
        parts = re.split(r"(?=## \[S\d{3,}\])", content)
        summary_parts = [p for p in parts if re.match(r"## \[S\d{3,}\]", p.strip())]
        results = []
        for part in summary_parts[:count]:
            data: dict[str, Any] = {}
            header = re.match(
                r"## \[(S\d{3,})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*-\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|\s*(\S+)\s*\|\s*Session Summary",
                part.strip(),
            )
            if header:
                data["id"] = header.group(1)
                data["start_date"] = header.group(2).strip()
                data["end_date"] = header.group(3).strip()
                data["branch"] = header.group(4).strip()
            else:
                continue
            commits_m = re.search(r"\*\*Commits\*\*:\s*(.*)", part)
            accomplished_m = re.search(r"\*\*Accomplished\*\*:\s*(.*)", part)
            files_m = re.search(r"\*\*Files touched\*\*:\s*(.*)", part)
            direction_m = re.search(r"\*\*Direction\*\*:\s*(.*)", part)
            data["commits"] = commits_m.group(1).strip() if commits_m else ""
            data["accomplished"] = accomplished_m.group(1).strip() if accomplished_m else ""
            data["files"] = files_m.group(1).strip() if files_m else ""
            data["direction"] = direction_m.group(1).strip() if direction_m else ""
            results.append(data)
        return results

    def _read_phase_summaries(self, count: int = 3) -> list[dict]:
        """Parse last `count` phase summaries from phases.md."""
        content = self._read_file(self._get_phases_path())
        if not content:
            return []
        parts = re.split(r"(?=## \[P\d{3,}\])", content)
        summary_parts = [p for p in parts if re.match(r"## \[P\d{3,}\]", p.strip())]
        results = []
        for part in summary_parts[:count]:
            data: dict[str, Any] = {}
            header = re.match(
                r"## \[(P\d{3,})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*-\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|\s*Phase Summary",
                part.strip(),
            )
            if header:
                data["id"] = header.group(1)
                data["start_date"] = header.group(2).strip()
                data["end_date"] = header.group(3).strip()
            else:
                continue
            scope_m = re.search(r"\*\*Scope\*\*:\s*(.*)", part)
            goal_m = re.search(r"\*\*Goal\*\*:\s*(.*)", part)
            outcome_m = re.search(r"\*\*Outcome\*\*:\s*(.*)", part)
            accomplishments_m = re.search(r"\*\*Key accomplishments\*\*:\s*(.*?)(?=\n\*\*|\Z)", part, re.DOTALL)
            data["scope"] = scope_m.group(1).strip() if scope_m else ""
            data["goal"] = goal_m.group(1).strip() if goal_m else ""
            data["outcome"] = outcome_m.group(1).strip() if outcome_m else ""
            data["accomplishments"] = accomplishments_m.group(1).strip() if accomplishments_m else ""
            results.append(data)
        return results

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

        # Prepend to summaries.md
        path = self._get_summaries_path(branch)
        with self._locks[path], self._file_lock(path):
            existing = self._read_file_unlocked(path) or ""
            self._write_file_unlocked(path, entry + existing)

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

        # Prepend to phases.md
        path = self._get_phases_path()
        with self._locks[path], self._file_lock(path):
            existing = self._read_file_unlocked(path) or ""
            self._write_file_unlocked(path, entry + existing)

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
        overview_path = self._get_overview_path()
        existing_overview = self._read_file(overview_path) or "(none)"

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
        path = self._get_overview_path()
        self._write_file(path, content)
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
            overview = self._read_file(self._get_overview_path())
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

    # --- Internal Helpers ---

    def _get_branch_dir(self, branch: str) -> str:
        # H3: Validate branch name to prevent path traversal
        if branch != "main" and not re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", branch):
            raise ValueError(f"Invalid branch name: {branch}")
        return os.path.join(self.branches_dir, branch)

    def _get_commits_path(self, branch: str) -> str:
        return os.path.join(self._get_branch_dir(branch), "commits.md")

    def _get_log_path(self, branch: str) -> str:
        return os.path.join(self._get_branch_dir(branch), "log.md")

    def _get_next_commit_id(self, branch: str) -> str:
        """Parse latest C### from commits.md and increment."""
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return "C001"
        matches = re.findall(r"\[C(\d{3,})\]", content)
        if not matches:
            return "C001"
        latest = max(int(m) for m in matches)
        return f"C{latest + 1:03d}"

    def _prepend_commit(self, branch: str, entry: str) -> None:
        """Insert commit at top of Milestone Journal section."""
        path = self._get_commits_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            anchor = "# Milestone Journal\n\n"
            idx = content.find(anchor)
            if idx >= 0:
                insert_at = idx + len(anchor)
                content = content[:insert_at] + entry + content[insert_at:]
            else:
                # Fallback: append
                content = content + "\n" + entry
            self._write_file_unlocked(path, content)
            self._invalidate_commit_index(branch)

    def _update_main_milestones(self, date: str, branch: str, title: str) -> None:
        """Update Recent Milestones in main.md."""
        path = os.path.join(self.ccr_root, "main.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            new_entry = f"- [{date}] ({branch}) {title}"

            section_match = re.search(
                r"(## Recent Milestones\n)(.*?)(\n## |\Z)",
                content,
                re.DOTALL,
            )
            if section_match:
                existing = section_match.group(2).strip()
                lines = [l for l in existing.split("\n") if l.strip() and l.strip() != "(none yet)"]
                lines.insert(0, new_entry)
                lines = lines[: self.config.milestones_kept]
                new_section = section_match.group(1) + "\n".join(lines) + "\n"
                end = section_match.group(3)
                content = content[: section_match.start()] + new_section + end
            else:
                content += f"\n## Recent Milestones\n{new_entry}\n"

            self._write_file_unlocked(path, content)

    def _read_recent_commits(self, branch: str, count: int) -> str:
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return ""
        # Split by ## [C markers
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        commit_parts = [p for p in parts if re.match(r"## \[C\d{3,}\]", p.strip())]
        return "\n".join(commit_parts[:count])

    def _get_branch_header(self, branch: str) -> str:
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return ""
        # Everything before first ## [C commit marker
        match = re.search(r"## \[C\d{3,}\]", content)
        if match:
            return content[: match.start()].strip()
        return content.strip()

    def _update_branch_conclusion(self, branch: str, outcome: str, conclusion: str) -> None:
        path = self._get_commits_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                r"\(Fill in at merge time — success/failure/partial\)",
                lambda m: f"{outcome}: {conclusion}",
                content,
            )
            self._write_file_unlocked(path, content)
            self._invalidate_commit_index(branch)

    def _build_commit_index(self, branch: str) -> dict[str, str]:
        """Parse commits.md once and build {commit_id: text_block} index.

        Lazily called by _find_commit_by_id on cache miss. The index is
        invalidated by _invalidate_commit_index whenever commits.md is mutated
        (_prepend_commit, _merge_into_last_commit, _update_branch_conclusion).
        """
        content = self._read_file(self._get_commits_path(branch))
        index: dict[str, str] = {}
        if not content:
            self._commit_index[branch] = index
            return index
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        for part in parts:
            id_match = re.match(r"## \[(C\d{3,})\]", part.strip())
            if id_match:
                index[id_match.group(1)] = part.strip()
        self._commit_index[branch] = index
        return index

    def _invalidate_commit_index(self, branch: str) -> None:
        """Drop cached commit index for a branch so next lookup rebuilds it."""
        self._commit_index.pop(branch, None)

    def _find_commit_by_id(self, branch: str, commit_id: str) -> str:
        """O(1) commit lookup using in-memory index (lazy-built, auto-invalidated)."""
        if branch not in self._commit_index:
            self._build_commit_index(branch)
        result = self._commit_index[branch].get(commit_id, "")
        if not result:
            # Cache miss after index exists — rebuild once in case file changed externally
            self._build_commit_index(branch)
            result = self._commit_index[branch].get(commit_id, "")
        return result

    def _bm25_search_commits(
        self, parts: list[str], query: str, max_results: int
    ) -> list[tuple[float, str]]:
        """BM25-scored search over commit text blocks (zero-dep fallback).

        Inspired by ExpRAG (arXiv:2603.18272) experience retrieval.
        Uses simplified Okapi BM25 with k1=1.5, b=0.75.
        Returns list of (score, commit_block) tuples sorted by score descending.
        """
        query_terms = [w.lower() for w in query.split() if len(w) > 2]
        if not query_terms:
            return []

        # Compute average document length
        non_empty = [p for p in parts if p.strip()]
        doc_lengths = [len(p.split()) for p in non_empty]
        avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)

        # Document frequency for IDF
        n_docs = len(non_empty)
        df: dict[str, int] = {}
        for term in query_terms:
            df[term] = sum(1 for p in non_empty if term in p.lower())

        k1 = 1.5
        b = 0.75

        scored: list[tuple[float, str]] = []
        for p in non_empty:
            p_lower = p.lower()
            dl = len(p.split())
            score = 0.0
            for term in query_terms:
                tf = p_lower.count(term)
                if tf == 0:
                    continue
                # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                idf = math.log(
                    (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1
                )
                # BM25 TF normalization
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm
            if score > 0:
                scored.append((score, p.strip()))

        scored.sort(key=lambda x: -x[0])
        return scored[:max_results]

    def _search_commits(self, branch: str, term: str, max_results: int = 5) -> str:
        """Search commits using 3-phase strategy: exact -> semantic -> BM25.

        Phase 1: Exact substring match (existing behavior, always first).
        Phase 2: ONNX embedding cosine similarity (when available).
        Phase 3: BM25 fallback (when ONNX unavailable and exact finds nothing).

        Inspired by ExpRAG (arXiv:2603.18272) embedding-based experience retrieval.
        """
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return ""
        parts = re.split(r"(?=## \[C\d{3,}\])", content)

        # Phase 1: Exact substring match
        exact_matches = [p.strip() for p in parts if term.lower() in p.lower()]

        # Extract commit IDs from exact matches to avoid duplicates
        exact_ids: set[str] = set()
        for m in exact_matches:
            id_match = re.search(r"\[C(\d{3,})\]", m)
            if id_match:
                exact_ids.add(f"C{id_match.group(1)}")

        remaining = max_results - len(exact_matches)
        semantic_matches: list[str] = []

        if remaining > 0:
            # Phase 2: Semantic search via embeddings
            model = get_embedding_model()
            if model is not None:
                try:
                    import numpy as np

                    query_vec = model.embed_query(term)
                    all_embeddings = self._load_all_commit_embeddings()
                    if all_embeddings:
                        ids = list(all_embeddings.keys())
                        vecs = np.stack([all_embeddings[cid] for cid in ids])
                        scores = vecs @ query_vec  # cosine (L2-normalized)
                        ranked = sorted(zip(ids, scores), key=lambda x: -x[1])
                        for cid, score in ranked:
                            if score < 0.3 or cid in exact_ids:
                                continue
                            block = self._find_commit_by_id(branch, cid)
                            if block:
                                semantic_matches.append(block.strip())
                            if len(semantic_matches) >= remaining:
                                break
                except Exception:
                    pass  # Fall through to BM25

            # Phase 3: BM25 fallback when no ONNX and exact found nothing
            if not semantic_matches and not exact_matches:
                bm25_results = self._bm25_search_commits(parts, term, remaining)
                semantic_matches = [block for _, block in bm25_results]

        # Merge: exact first, then semantic/BM25
        combined = exact_matches[:max_results]
        remaining = max_results - len(combined)
        if remaining > 0 and semantic_matches:
            combined.extend(semantic_matches[:remaining])

        # Track memory metrics (ALMA-inspired retrieval parameter evolution)
        result = "\n\n".join(combined[:max_results])
        try:
            self._increment_memory_metric("search_calls")
            if not result.strip():
                self._increment_memory_metric("search_zero_results")
        except Exception:
            pass  # Metrics are supplementary — never fail the search
        return result

    def _update_registry_active_branch(self, branch: str) -> None:
        path = os.path.join(self.branches_dir, "_registry.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                r"(## Active Branch\s*\n)\S+",
                lambda m: f"{m.group(1)}{branch}",
                content,
            )
            self._write_file_unlocked(path, content)

    def _add_branch_to_registry(self, name: str, date: str) -> None:
        path = os.path.join(self.branches_dir, "_registry.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            row = f"| {name} | {date} | active |"
            content = content.rstrip() + "\n" + row + "\n"
            self._write_file_unlocked(path, content)

    def _update_registry_branch_status(self, name: str, status: str) -> None:
        path = os.path.join(self.branches_dir, "_registry.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                rf"(\| {re.escape(name)} \| [^|]+ \| )\w+( \|)",
                lambda m: f"{m.group(1)}{status}{m.group(2)}",
                content,
            )
            self._write_file_unlocked(path, content)

    def _add_branch_to_main_md(self, name: str) -> None:
        path = os.path.join(self.ccr_root, "main.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = content.replace("(none)", "")
            section_match = re.search(r"(## Open Branches\n)", content)
            if section_match:
                insert_at = section_match.end()
                content = content[:insert_at] + f"- {name}\n" + content[insert_at:]
            else:
                content += f"\n## Open Branches\n- {name}\n"
            self._write_file_unlocked(path, content)

    def _remove_branch_from_main_md(self, name: str) -> None:
        path = os.path.join(self.ccr_root, "main.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(rf"- {re.escape(name)}\n?", "", content)
            # If open branches section is now empty, add placeholder
            if "## Open Branches\n" in content:
                after = content.split("## Open Branches\n", 1)[1]
                next_section = re.search(r"\n## ", after)
                branch_text = after[: next_section.start()] if next_section else after
                if not branch_text.strip():
                    content = content.replace(
                        "## Open Branches\n" + branch_text,
                        "## Open Branches\n(none)\n",
                    )
            self._write_file_unlocked(path, content)

    # --- metadata.yaml ---

    def _get_metadata_path(self) -> str:
        return os.path.join(self.ccr_root, "metadata.yaml")

    def _load_metadata(self) -> dict:
        path = self._get_metadata_path()
        with self._locks[path]:
            if not os.path.isfile(path):
                return dict(METADATA_TEMPLATE)
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or dict(METADATA_TEMPLATE)

    def _save_metadata(self, data: dict) -> None:
        """Save metadata atomically via tmp + fsync + os.replace (H4)."""
        path = self._get_metadata_path()
        with self._locks[path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            content = yaml.dump(data, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper)
            tmp_path = path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def _update_metadata_branch(self, name: str, status: str, created: str, parent: str) -> None:
        meta = self._load_metadata()
        meta.setdefault("branches", [])
        meta["branches"].append({
            "name": name, "status": status, "created": created, "parent": parent,
        })
        meta["version"] = meta.get("version", 0) + 1
        self._save_metadata(meta)

    def _update_metadata_branch_status(self, name: str, status: str) -> None:
        meta = self._load_metadata()
        for b in meta.get("branches", []):
            if b.get("name") == name:
                b["status"] = status
                break
        meta["version"] = meta.get("version", 0) + 1
        self._save_metadata(meta)

    def update_metadata_file_tree(self, file_list: list[str]) -> None:
        """Update metadata.yaml with the current file tree."""
        meta = self._load_metadata()
        meta["file_tree"] = file_list[:500]  # cap at 500 files
        self._save_metadata(meta)

    def update_metadata_dependencies(self, deps: list[str]) -> None:
        """Update metadata.yaml with project dependencies."""
        meta = self._load_metadata()
        meta["dependencies"] = deps
        self._save_metadata(meta)

    def update_metadata_config(self, language: str = "", framework: str = "") -> None:
        """Update metadata.yaml with project config."""
        meta = self._load_metadata()
        if language:
            meta.setdefault("config", {})["language"] = language
        if framework:
            meta.setdefault("config", {})["framework"] = framework
        self._save_metadata(meta)

    # --- summary.md per branch ---

    def _get_summary_path(self, branch: str) -> str:
        return os.path.join(self._get_branch_dir(branch), "summary.md")

    def _read_branch_summary(self, branch: str) -> str:
        return self._read_file(self._get_summary_path(branch))

    def _update_summary_status(self, branch: str, status: str) -> None:
        path = self._get_summary_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path)
            if content:
                content = re.sub(r"(## Status\n)\S+.*", lambda m: f"{m.group(1)}{status}", content)
                self._write_file_unlocked(path, content)

    def _add_key_decision_to_summary(self, branch: str, decision: str) -> None:
        path = self._get_summary_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = content.replace("(none yet)", "")
            section_match = re.search(r"(## Key Decisions\n)", content)
            if section_match:
                insert_at = section_match.end()
                content = content[:insert_at] + f"- {decision}\n" + content[insert_at:]
            self._write_file_unlocked(path, content)

    # --- Context windowing ---

    def _read_commits_window(self, branch: str, offset: int, count: int) -> str:
        """Read a window of commits: [offset:offset+count] from most recent."""
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return ""
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        commit_parts = [p for p in parts if re.match(r"## \[C\d{3,}\]", p.strip())]
        windowed = commit_parts[offset:offset + count]
        return "\n".join(windowed)

    def _inject_evolved_tags(self, commits_text: str) -> str:
        """Replace **What**: <original> with **What** [evolved]: <evolved> for evolved commits.

        A-MEM §3.3 Eq.7: shows evolved summaries in context output.
        """
        if not self._evolved_summaries:
            return commits_text

        def _evolve_block(block: str) -> str:
            """Inject evolved tag into a single commit block if applicable."""
            id_match = re.search(r"\[(C\d{3,})\]", block)
            if not id_match:
                return block
            cid = id_match.group(1)
            evolved_what = self.get_evolved_what(cid)
            if not evolved_what:
                return block
            return re.sub(
                r"\*\*What\*\*: .+",
                f"**What** [evolved]: {evolved_what}",
                block,
                count=1,
            )

        # Split on commit header boundaries, evolve each block, then rejoin
        parts = re.split(r"(?=## \[C\d{3,}\])", commits_text)
        processed = [
            _evolve_block(p) if re.match(r"## \[C\d{3,}\]", p.strip()) else p
            for p in parts
        ]
        return "".join(processed)

    def _read_log_window(self, branch: str, count: int) -> str:
        """Read last N OTA entries from the branch log."""
        content = self._read_file(self._get_log_path(branch))
        if not content:
            return ""
        # Split by OTA entry markers
        entries = re.split(r"(?=---\n\*\*\[OTA-)", content)
        entries = [e for e in entries if e.strip()]
        # Also handle table-format entries
        if not entries:
            lines = content.strip().split("\n")
            return "\n".join(lines[-count:])
        return "\n".join(entries[-count:])

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

    def _update_current_focus(self, title: str, next_step: str) -> None:
        """Update the Current Focus section in main.md."""
        path = os.path.join(self.ccr_root, "main.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            new_focus = f"{title}. Next: {next_step}"
            content = re.sub(
                r"(## Current Focus\n).*?(?=\n## )",
                lambda m: f"{m.group(1)}{new_focus}\n",
                content,
                count=1,
                flags=re.DOTALL,
            )
            self._write_file_unlocked(path, content)

    def _git_commit(self, message: str) -> bool:
        """Create a git commit if in a git repo. Returns True if committed."""
        git_dir = os.path.join(self.project_root, ".git")
        if not os.path.isdir(git_dir):
            return False
        try:
            # Stage .ccr/ changes
            subprocess.run(
                ["git", "add", ".ccr/"],
                cwd=self.project_root, capture_output=True, timeout=10,
            )
            result = subprocess.run(
                ["git", "commit", "-m", message, "--allow-empty"],
                cwd=self.project_root, capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _append_log(self, branch: str, line: str) -> None:
        path = self._get_log_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            lines = content.strip().split("\n") if content.strip() else []
            lines.append(line)
            # Rotate
            if len(lines) > self.config.log_max_lines:
                lines = lines[-200:]
            self._write_file_unlocked(path, "\n".join(lines) + "\n")

    def _add_to_gitignore(self, gitignore_path: str) -> None:
        content = ""
        if os.path.isfile(gitignore_path):
            content = self._read_file(gitignore_path) or ""
        if ".ccr/" not in content and ".ccr" not in content:
            with open(gitignore_path, "a") as f:
                f.write("\n# CCR context directory\n.ccr/\n")

    # --- File I/O (thread-safe + cross-process wrappers) ---

    @contextmanager
    def _file_lock(self, path: str):
        """Cross-process file lock using fcntl.flock.

        Creates a .lock file alongside the target file. Uses LOCK_EX
        (exclusive) to prevent concurrent writes from multiple MCP server
        instances sharing the same .ccr/ directory.
        """
        lock_path = path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
            # M5: Clean up .lock files
            try:
                os.unlink(lock_path)
            except OSError:
                pass  # Race condition: another process may have already removed it

    def _read_file(self, path: str) -> str:
        with self._locks[path], self._file_lock(path):
            return self._read_file_unlocked(path)

    def _read_file_unlocked(self, path: str) -> str:
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_file(self, path: str, content: str) -> None:
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def _write_file_unlocked(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Sanitize surrogates that can't be encoded in UTF-8
        content = content.encode("utf-8", errors="replace").decode("utf-8")
        # Atomic write: write to tmp file, fsync, then os.replace
        dir_name = os.path.dirname(path) or "."
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _write_if_missing(self, path: str, content: str) -> None:
        if not os.path.isfile(path):
            self._write_file(path, content)
