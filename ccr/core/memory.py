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

import math
import os
import re
import subprocess
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any

import yaml

from ccr.core.types import CCRConfig

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

    def set_sub_client(self, client) -> None:
        """Set the sub-model client for LLM-regenerated rolling summaries."""
        self.sub_client = client

    # --- Bootstrap ---

    def ensure_structure(self) -> bool:
        """Create .ccr/ layout if missing. Idempotent. Returns True if created."""
        created = False
        main_branch_dir = os.path.join(self.branches_dir, "main")

        if not os.path.isdir(self.ccr_root):
            os.makedirs(self.ccr_root, exist_ok=True)
            created = True

        os.makedirs(main_branch_dir, exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d")

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

        # Add .ccr to .gitignore if in a git repo
        gitignore = os.path.join(self.project_root, ".gitignore")
        if os.path.isdir(os.path.join(self.project_root, ".git")):
            self._add_to_gitignore(gitignore)

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
        now = datetime.now().strftime("%Y-%m-%d")
        now_full = datetime.now().strftime("%Y-%m-%d %H:%M")

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
        admission_threshold: float = 0.85,
        rejection_threshold: float = 0.0,
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
        5. If conflict AND S(m) > S(m_conflict) → Create new (new outranks existing,
           per Alg. 1 line 6: both coexist)
        6. If conflict AND S(m) <= S(m_conflict) → Merge into conflict target
           (per Alg. 1 line 7: fold new into existing)
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

        # --- Admission Control (A-MAC Algorithm 1) ---
        if admission_threshold < 1.0 or rejection_threshold > 0.0:
            score = self.compute_admission_score(
                branch, title, what, why, files_changed, next_step,
            )

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

            # Step 4-5: FindConflict + Merge (Alg. 1 lines 4-7)
            # similarity >= admission_threshold → conflict detected (paper: sim > 0.85)
            if score["similarity"] >= admission_threshold and score["conflict_id"]:
                # Step 6: Score comparison (Alg. 1 line 6: S(m) > S(m_conflict))
                # Compare new commit's score against the conflicting commit's score.
                # If new outranks existing → create new (both coexist per Alg. 1 line 9).
                # If existing outranks new → merge new into existing (Alg. 1 line 7).
                new_score = score["score"]
                conflict_score = score.get("conflict_score", 0.0)
                if new_score > conflict_score:
                    pass  # Fall through: new is more valuable, add alongside existing
                else:
                    # Merge: existing outranks new → fold new into conflict target
                    merged_id = self._merge_into_last_commit(
                        branch, score["conflict_id"],
                        title, what, why, files_changed, next_step,
                    )
                    self._update_rolling_summary(branch, what, why, next_step)
                    self._update_current_focus(title, next_step)

                    self._append_log(branch, self._format_ota_log(
                        "commit-merge", f"{merged_id}: +{title}", "OK",
                        observation=f"Admission control: sim={score['similarity']:.2f}, S(m)={new_score:.2f} vs S(m')={conflict_score:.2f}",
                        thought=f"S(m) <= S(m_conflict): merging into {merged_id}",
                        action=f"Updated {merged_id} with new info",
                    ))

                    self._git_commit(f"ccr: +{title} (merged into {merged_id})")
                    return f"[{merged_id}+] {title} (merged, sim={score['similarity']:.2f})"

        # --- Normal commit path ---
        commit_id = self._get_next_commit_id(branch)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        files_str = ", ".join(files_changed) if files_changed else "(none)"

        entry = (
            f"## [{commit_id}] {now} | branch:{branch} | {title}\n"
            f"**What**: {what}\n"
            f"**Why**: {why}\n"
            f"**Files**: {files_str}\n"
            f"**Next**: {next_step}\n\n---\n\n"
        )

        # Reference OTA log slice in commit (per GCC paper)
        ota_slice = self._get_ota_slice_since_last_commit(branch)
        if ota_slice:
            entry = entry.rstrip("---\n\n") + f"**OTA Trace**: {ota_slice}\n\n---\n\n"

        self._prepend_commit(branch, entry)

        # GCC paper: regenerate rolling summary S_t = f(S_{t-1}, D_t)
        self._update_rolling_summary(branch, what, why, next_step)

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

        return f"[{commit_id}] {title} (branch: {branch})"

    # --- Admission Control (A-MAC inspired) ---

    def _parse_recent_commit_data(self, branch: str, k: int = 3) -> list[dict[str, Any]]:
        """Parse last k commits from commits.md into structured dicts.

        Returns list of dicts, each with:
            id, timestamp, title, what, why, files (list[str]), next
        """
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return []
        parts = re.split(r"(?=## \[C\d{3}\])", content)
        commit_parts = [p for p in parts if re.match(r"## \[C\d{3}\]", p.strip())]

        results = []
        for part in commit_parts[:k]:
            data: dict[str, Any] = {}
            # Parse header: ## [C021] 2026-03-10 22:25 | branch:main | Title
            header_match = re.match(
                r"## \[(C\d{3})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|[^|]*\|\s*(.*)",
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

            data["what"] = what_match.group(1).strip() if what_match else ""
            data["why"] = why_match.group(1).strip() if why_match else ""
            data["next"] = next_match.group(1).strip() if next_match else ""

            files_str = files_match.group(1).strip() if files_match else ""
            if files_str and files_str != "(none)":
                data["files"] = [f.strip() for f in files_str.split(",") if f.strip()]
            else:
                data["files"] = []

            results.append(data)
        return results

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Jaccard similarity between two sets."""
        if not a and not b:
            return 0.0
        union = a | b
        return len(a & b) / len(union) if union else 0.0

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

        A-MAC Eq. 1 (adapted — 3 of 5 factors, no LLM):
            S(m) = w_N · N(m) + w_R · R(m) + w_T · T(m)

        Factors implemented:
            N(m) = 1 − max_{m' ∈ M} sim(φ(m), φ(m'))    [Eq. 3, Novelty]
            R(m) = exp(−λ · τ(m)), λ=0.01/hour           [Eq. 4, Recency]
            T(m) = type_prior(classify(m))                 [§3.2 Factor 5]

        Factors NOT implemented (no LLM available):
            U(m) — Utility: requires LLM to rate future usefulness
            C(m) — Confidence: requires ROUGE-L against source turns

        Similarity computation (proxy for Eq. 3):
            sim(m, m') = 0.50 · Jaccard(files) + 0.50 · Jaccard(keywords)
            Word Jaccard substitutes for Sentence-BERT cosine since CCR
            has no embedding model. Lower discriminative power but zero cost.
            Recency-modulated: effective_sim = raw_sim · R_conflict(m')
            where R_conflict = exp(−0.01 · hours_since_m'). Old conflicts
            are dampened per Eq. 4 (conflicts from 3 days ago are weaker).

        Weight rationale:
            w_N = 0.60: Novelty dominates — the primary question is "does this
                add new information?" Without U and C, novelty carries more weight.
            w_T = 0.40: Type Prior is most impactful per Table 2 ablation
                (ΔF1 = -0.107). High weight ensures structural commits are
                admitted while continuations are easy to merge.
            w_R is folded into similarity (modulates conflict strength)
            rather than the admission score directly, because for new commits
            being created NOW, R(m)=1.0 always (τ=0). Instead, R modulates
            how strongly an old commit "pulls" the new one toward a merge.

        Returns dict with:
            score: float — S(m) ∈ [0,1], higher = more valuable (paper Eq. 1)
            similarity: float — max recency-weighted sim to any existing commit
            novelty: float — N(m) = 1 - similarity (Eq. 3)
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
        stop_words = {"the", "a", "an", "is", "was", "to", "for", "of", "in", "on", "with", "and", "or", "but"}
        new_files = {f.strip().lower() for f in files_changed if f.strip()}
        new_text = f"{title} {what} {why}".lower()
        new_words = {w for w in re.findall(r"\w+", new_text) if w not in stop_words and len(w) > 2}

        best_similarity = 0.0
        best_file_sim = 0.0
        best_keyword_sim = 0.0
        best_conflict_id = None
        best_conflict_recency = 0.0
        best_conflict_score = 0.0  # S(m_conflict) for Alg. 1 line 6

        for commit in recent:
            old_files = {f.strip().lower() for f in commit.get("files", []) if f.strip()}
            file_sim = self._jaccard(new_files, old_files)

            old_text = f"{commit.get('title', '')} {commit.get('what', '')} {commit.get('why', '')}".lower()
            old_words = {w for w in re.findall(r"\w+", old_text) if w not in stop_words and len(w) > 2}
            keyword_sim = self._jaccard(new_words, old_words)

            # Raw content similarity
            raw_sim = 0.50 * file_sim + 0.50 * keyword_sim

            # Recency of existing commit (Eq. 4: λ=0.01/hour, half-life ~69 hours)
            hours_since = self._hours_since_commit(commit.get("timestamp", ""))
            conflict_recency = math.exp(-0.01 * hours_since) if hours_since >= 0 else 0.0

            # Recency-modulated similarity: old conflicts are dampened
            effective_sim = raw_sim * conflict_recency

            if effective_sim > best_similarity:
                best_similarity = effective_sim
                best_file_sim = file_sim
                best_keyword_sim = keyword_sim
                best_conflict_id = commit.get("id")
                best_conflict_recency = conflict_recency

                # --- G1-NEW: Compute S(m_conflict) per Alg. 1 line 6 ---
                # The existing commit's score mirrors S(m) = w_N * N + w_T * T.
                # N(m'): We can't retroactively compute novelty, so we assume
                # the commit WAS novel when stored (N≈0.7, a reasonable prior
                # for commits that passed admission). Recency decays this:
                # old commits lose assumed novelty over time.
                # T(m'): Classify the existing commit's type for its type prior.
                conflict_type = self._classify_commit_type(
                    commit.get("title", ""), commit.get("what", ""), commit.get("files", []),
                )
                conflict_tp = self._type_prior(conflict_type)
                # S(m_conflict) = w_N * (assumed_novelty * recency) + w_T * T(m')
                assumed_novelty = 0.7 * conflict_recency  # decays with age
                best_conflict_score = 0.60 * assumed_novelty + 0.40 * conflict_tp

        # Novelty N(m) = 1 - max_similarity (Eq. 3)
        novelty = 1.0 - best_similarity

        # Admission score S(m) = w_N * N + w_T * T (Eq. 1, adapted)
        # Higher = more valuable = more likely to admit as new
        admission_score = 0.60 * novelty + 0.40 * tp

        reason_parts = []
        if best_similarity > 0.5:
            reason_parts.append(f"sim={best_similarity:.2f}")
        if novelty > 0.7:
            reason_parts.append("novel")
        if tp >= 0.85:
            reason_parts.append(f"type={commit_type}")
        if tp <= 0.3:
            reason_parts.append(f"type={commit_type}")

        return {
            "score": admission_score,
            "similarity": best_similarity,
            "novelty": novelty,
            "conflict_id": best_conflict_id,
            "conflict_recency": best_conflict_recency,
            "conflict_score": best_conflict_score,
            "commit_type": commit_type,
            "type_prior": tp,
            "file_similarity": best_file_sim,
            "keyword_similarity": best_keyword_sim,
            "reason": ", ".join(reason_parts) if reason_parts else "moderate",
        }

    def _hours_since_commit(self, timestamp_str: str) -> float:
        """Parse commit timestamp and return hours elapsed.

        Per A-MAC Eq. 4: τ(m) is measured in hours (λ=0.01/hour).
        """
        if not timestamp_str:
            return 999.0  # treat missing as very old
        try:
            commit_time = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M")
            delta = datetime.now() - commit_time
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
    ) -> str:
        """Merge new commit data into the most recent commit (admission control).

        Updates the last commit in-place: appends what/why, unions files,
        replaces next_step with the newer one.

        Returns the merged commit ID.
        """
        path = self._get_commits_path(branch)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        files_str = ", ".join(files_changed) if files_changed else ""

        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""

            # Find the target commit block
            pattern = rf"(## \[{re.escape(commit_id)}\].*?)(?=## \[C\d{{3}}\]|---\n\n|\Z)"
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                return commit_id  # fallback: can't find commit

            old_block = match.group(1)

            # Update timestamp to now
            new_block = re.sub(
                r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})",
                now,
                old_block,
                count=1,
            )

            # Append to title (keep original, note merge)
            new_block = re.sub(
                rf"(\[{re.escape(commit_id)}\]\s+{re.escape(now)}\s*\|[^|]*\|\s*)(.*)",
                rf"\g<1>\2 + {title}",
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

            # Replace Next with newer value
            next_match = re.search(r"(\*\*Next\*\*:\s*)(.*)", new_block)
            if next_match:
                new_block = new_block.replace(
                    next_match.group(0),
                    f"{next_match.group(1)}{next_step}",
                )

            content = content[:match.start()] + new_block + content[match.end():]
            self._write_file_unlocked(path, content)

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
    ) -> None:
        """Regenerate rolling summary: S_t = f(S_{t-1}, D_t).

        Per the GCC paper: each commit regenerates a coarse-grained summary
        combining the previous summary with the new contribution. This creates
        a progressively refined chain that captures the full branch history
        in a compact form — no need to re-read all individual commits.

        When a sub_client is available, uses LLM to compress the summary
        (per GCC paper section 2.2). Falls back to concatenation otherwise.
        """
        previous_summary = self._get_rolling_summary(branch)
        new_contribution = f"{what} (because: {why}). Next: {next_step}"

        if self.sub_client is not None:
            # LLM-regenerated summary (per GCC paper section 2.2)
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
                    # Use LLM summary
                    self._write_rolling_summary(branch, new_summary.strip()[:1500])
                    return
            except Exception:
                pass  # Fall through to concatenation

        # Fallback: concatenation
        if previous_summary:
            new_summary = f"{previous_summary}; {new_contribution}"
        else:
            new_summary = new_contribution

        # Cap rolling summary length to stay compact
        if len(new_summary) > 1500:
            # Keep the most recent ~1200 chars + ellipsis for older context
            new_summary = "..." + new_summary[-1200:]

        self._write_rolling_summary(branch, new_summary)

    def _write_rolling_summary(self, branch: str, summary: str) -> None:
        """Write the rolling summary into the commits.md file."""
        path = self._get_commits_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                r"(## Rolling Summary\n).*?(?=\n---|\n# |\Z)",
                rf"\g<1>{summary}\n",
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
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

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
                merged = "..." + merged[-1200:]
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
    ) -> str:
        """Multi-level context retrieval with windowing support.

        Level 1: main.md only (~200 tokens)
        Level 2: + last 3 commits from active branch (windowed by offset)
        Level 3: + branch summary.md (purpose/hypothesis/conclusion)
        Level 4: + last 10 commits (windowed by offset)
        Level 5: + specific commit by ID or keyword search

        Args:
            offset: Scroll position for commit window (0 = most recent)
            log_window: Number of recent OTA log entries to include (0 = none)
            metadata_segment: Metadata key to include (e.g. "file_tree", "dependencies")
        """
        branch = branch or self.get_active_branch()
        parts = []

        # Level 1: main.md
        main_content = self._read_file(os.path.join(self.ccr_root, "main.md"))
        if main_content:
            parts.append(f"# Project Overview\n{main_content}")

        if level >= 2:
            # Level 2: rolling summary + recent commits (windowed)
            rolling = self._get_rolling_summary(branch)
            if rolling:
                parts.append(f"# Progress Summary ({branch})\n{rolling}")

            count = 3 if level < 4 else 10
            recent = self._read_commits_window(branch, offset, count)
            if recent:
                parts.append(f"# Recent Commits ({branch}, offset={offset})\n{recent}")

        if level >= 3:
            # Level 3: branch summary (prefer summary.md, fallback to header)
            if branch != "main":
                summary = self._read_branch_summary(branch)
                if summary:
                    parts.append(f"# Branch: {branch}\n{summary}")
                else:
                    header = self._get_branch_header(branch)
                    if header:
                        parts.append(f"# Branch: {branch}\n{header}")

        if level >= 5:
            # Level 5: specific commit or search
            if commit_id:
                found = self._find_commit_by_id(branch, commit_id)
                if found:
                    parts.append(f"# Commit {commit_id}\n{found}")
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
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
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
        """Get the next OTA sequential ID for a branch."""
        content = self._read_file(self._get_log_path(branch))
        if not content:
            return "OTA-001"
        matches = re.findall(r"\[OTA-(\d{3})\]", content)
        if not matches:
            return "OTA-001"
        latest = max(int(m) for m in matches)
        return f"OTA-{latest + 1:03d}"

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

    # --- Internal Helpers ---

    def _get_branch_dir(self, branch: str) -> str:
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
        matches = re.findall(r"\[C(\d{3})\]", content)
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
        parts = re.split(r"(?=## \[C\d{3}\])", content)
        commit_parts = [p for p in parts if re.match(r"## \[C\d{3}\]", p.strip())]
        return "\n".join(commit_parts[:count])

    def _get_branch_header(self, branch: str) -> str:
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return ""
        # Everything before first ## [C commit marker
        match = re.search(r"## \[C\d{3}\]", content)
        if match:
            return content[: match.start()].strip()
        return content.strip()

    def _update_branch_conclusion(self, branch: str, outcome: str, conclusion: str) -> None:
        path = self._get_commits_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                r"\(Fill in at merge time — success/failure/partial\)",
                f"{outcome}: {conclusion}",
                content,
            )
            self._write_file_unlocked(path, content)

    def _find_commit_by_id(self, branch: str, commit_id: str) -> str:
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return ""
        parts = re.split(r"(?=## \[C\d{3}\])", content)
        for part in parts:
            if f"[{commit_id}]" in part:
                return part.strip()
        return ""

    def _search_commits(self, branch: str, term: str, max_results: int = 5) -> str:
        content = self._read_file(self._get_commits_path(branch))
        if not content:
            return ""
        parts = re.split(r"(?=## \[C\d{3}\])", content)
        matches = [p.strip() for p in parts if term.lower() in p.lower()]
        return "\n\n".join(matches[:max_results])

    def _update_registry_active_branch(self, branch: str) -> None:
        path = os.path.join(self.branches_dir, "_registry.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                r"(## Active Branch\s*\n)\S+",
                rf"\g<1>{branch}",
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
                rf"\g<1>{status}\2",
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
        path = self._get_metadata_path()
        with self._locks[path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

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
                content = re.sub(r"(## Status\n)\S+.*", rf"\g<1>{status}", content)
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
        parts = re.split(r"(?=## \[C\d{3}\])", content)
        commit_parts = [p for p in parts if re.match(r"## \[C\d{3}\]", p.strip())]
        windowed = commit_parts[offset:offset + count]
        return "\n".join(windowed)

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
            id_match = re.search(r"\[OTA-(\d{3})\]", entry)
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
                rf"\g<1>{new_focus}\n",
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

    # --- File I/O (thread-safe wrappers) ---

    def _read_file(self, path: str) -> str:
        with self._locks[path]:
            return self._read_file_unlocked(path)

    def _read_file_unlocked(self, path: str) -> str:
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_file(self, path: str, content: str) -> None:
        with self._locks[path]:
            self._write_file_unlocked(path, content)

    def _write_file_unlocked(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Sanitize surrogates that can't be encoded in UTF-8
        content = content.encode("utf-8", errors="replace").decode("utf-8")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _write_if_missing(self, path: str, content: str) -> None:
        if not os.path.isfile(path):
            self._write_file(path, content)
