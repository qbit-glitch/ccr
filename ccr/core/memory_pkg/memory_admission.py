"""AdmissionMixin — A-MAC admission scoring for MemoryManager.

Methods:
    _classify_commit_type (staticmethod)
    _type_prior (staticmethod)
    compute_admission_score
    _utility_heuristic (staticmethod)
    _hours_since_commit
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Any

from ccr.context.embeddings import quick_cosine
from ccr.core.types import CCRConfig

logger = logging.getLogger(__name__)


class AdmissionMixin:
    """A-MAC admission scoring methods for MemoryManager."""

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
