"""CommitMixin -- commit, _parse_recent_commit_data, _merge_into_last_commit, and helpers."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from ccr.core.types import CommitLink

logger = logging.getLogger(__name__)


class CommitMixin:
    """Commit operations for MemoryManager.

    Expects the composite class to provide:
        self.ccr_root: str
        self.config: CCRConfig
        self.sub_client: Any | None
        self._locks: dict
        self._evolved_summaries: dict
        self.get_active_branch() -> str
        self.compute_admission_score(...) -> dict
        self._append_log(branch, line)
        self._format_ota_log(...) -> str
        self._get_ota_slice_since_last_commit(branch) -> str
        self._update_rolling_summary(branch, what, why, next_step, compressed)
        self._get_rolling_summary(branch) -> str
        self._write_rolling_summary(branch, text)
        self._read_file(path) -> str
        self._read_file_unlocked(path) -> str
        self._write_file(path, content)
        self._write_file_unlocked(path, content)
        self._get_commits_path(branch) -> str
        self._git_commit(message) -> bool
        self._embed_commit(commit_id, text) -> Any
        self._compute_links(branch, commit_id, ...) -> list
        self._update_links(commit_id, links)
        self._trigger_memory_evolution(commit_id, links)
        self._process_patterns(commit_id, patterns, now) -> list[dict]
        self._scan_pending_promotions() -> list[dict]
        self._load_patterns() -> dict
        self._increment_memory_metric(name)
        self._invalidate_commit_index(branch)
        self._file_lock(path)
        self._maybe_generate_session_summary(branch)
    """

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
        author: str = "",
        ci_context: dict | None = None,
        experiment: dict | None = None,
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

        author_str = f"**Author**: {author}\n" if author else ""
        ci_str = f"**CI**: {json.dumps(ci_context)}\n" if ci_context else ""

        exp_str = ""
        if experiment and isinstance(experiment, dict):
            exp_lines = ["**Experiment**:"]
            if experiment.get("id"):
                exp_lines.append(f"  - ID: {experiment['id']}")
            if experiment.get("hypothesis"):
                exp_lines.append(f"  - Hypothesis: {experiment['hypothesis']}")
            if experiment.get("metrics") and isinstance(experiment["metrics"], dict):
                metrics_parts = ", ".join(
                    f"{k}={v}" for k, v in experiment["metrics"].items()
                )
                exp_lines.append(f"  - Metrics: {metrics_parts}")
            if experiment.get("conclusion"):
                exp_lines.append(f"  - Conclusion: {experiment['conclusion']}")
            exp_str = "\n".join(exp_lines) + "\n"

        entry = (
            f"## [{commit_id}] {now} | branch:{branch} | {title}\n"
            f"**What**: {what}\n"
            f"**Why**: {why}\n"
            f"**Files**: {files_str}\n"
            f"**Next**: {next_step}\n"
            f"{patterns_str}"
            f"{author_str}"
            f"{ci_str}"
            f"{exp_str}"
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
                f"\n\n\u26a0\ufe0f Rolling summary at {_pre_update_summary_len}/1500 chars. "
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
