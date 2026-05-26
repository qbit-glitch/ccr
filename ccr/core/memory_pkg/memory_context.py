"""ContextMixin -- get_context, search, index, and windowed retrieval."""

from __future__ import annotations

import json
import logging
import math
import os
import re

from ccr.context.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class ContextMixin:
    """Context retrieval and search operations for MemoryManager.

    Expects the composite class to provide:
        self.ccr_root: str
        self.branches_dir: str
        self.config: CCRConfig
        self._locks: dict
        self._commit_index: dict[str, dict[str, str]]
        self._evolved_summaries: dict
        self.get_active_branch() -> str
        self._read_file(path) -> str
        self._read_file_unlocked(path) -> str
        self._write_file(path, content)
        self._write_file_unlocked(path, content)
        self._get_branch_dir(branch) -> str
        self._get_commits_path(branch) -> str
        self._get_log_path(branch) -> str
        self._get_rolling_summary(branch) -> str
        self._get_overview_path() -> str
        self._read_branch_summary(branch) -> str
        self._read_session_summaries(branch, count) -> list[dict]
        self._read_phase_summaries(count) -> list[dict]
        self._load_metadata() -> dict
        self._load_patterns() -> dict
        self.get_commit_links(commit_id) -> dict
        self.get_linked_commits(commit_id, max_hops) -> list[dict]
        self._format_links_for_context(commit_id, links_data) -> str
        self._load_all_commit_embeddings() -> dict
        self._increment_memory_metric(name)
        self.get_evolved_what(cid) -> str | None
    """

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
        overview = self._storage.project_state_get("overview")
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
            meta = self._load_metadata()
            if metadata_segment in meta:
                segment_data = meta[metadata_segment]
                if isinstance(segment_data, (list, dict)):
                    parts.append(f"# Metadata: {metadata_segment}\n{json.dumps(segment_data, indent=2)}")
                else:
                    parts.append(f"# Metadata: {metadata_segment}\n{segment_data}")

        return "\n\n".join(parts)

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

    # --- Branch header / conclusion helpers ---

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
        with self._locks[path], self._file_lock(path):
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                r"\(Fill in at merge time — success/failure/partial\)",
                lambda m: f"{outcome}: {conclusion}",
                content,
            )
            self._write_file_unlocked(path, content)
            self._invalidate_commit_index(branch)

    # --- Commit index (lazy-built, auto-invalidated) ---

    def _build_commit_index(self, branch: str) -> dict[str, str]:
        """Build {commit_id: text_block} index via storage backend.

        Lazily called by _find_commit_by_id on cache miss. The index is
        invalidated by _invalidate_commit_index whenever commits are mutated.
        """
        records = self._storage.commit_list(branch, limit=9999)
        index = {r["id"]: r.get("raw_block", "") for r in records}
        self._commit_index[branch] = index
        return index

    def _invalidate_commit_index(self, branch: str) -> None:
        """Drop cached commit index for a branch so next lookup rebuilds it."""
        self._commit_index.pop(branch, None)

    def _find_commit_by_id(self, branch: str, commit_id: str) -> str:
        """O(1) commit lookup using in-memory index, with backend fallback."""
        if branch not in self._commit_index:
            self._build_commit_index(branch)
        result = self._commit_index[branch].get(commit_id, "")
        if not result:
            record = self._storage.commit_get(branch, commit_id)
            if record:
                result = record.get("raw_block", "")
        return result

    # --- Search ---

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

        Phase 1: Exact text search via storage backend.
        Phase 2: ONNX embedding cosine similarity (when available).
        Phase 3: BM25 fallback (when ONNX unavailable and exact finds nothing).
        """
        # Phase 1: Text search via backend
        text_results = self._storage.commit_search_text(branch, term, max_results)
        exact_matches = [r.get("raw_block", "").strip() for r in text_results if r.get("raw_block")]

        exact_ids: set[str] = {r["id"] for r in text_results}

        remaining = max_results - len(exact_matches)
        semantic_matches: list[str] = []

        if remaining > 0:
            model = get_embedding_model()
            if model is not None:
                try:
                    import numpy as np

                    query_vec = model.embed_query(term)

                    # Phase 4: prefer backend KNN when sqlite-vec is wired.
                    if getattr(self._storage, "vec_available", False):
                        hits = self._storage.commit_semantic_search(
                            branch, query_vec.tolist(), remaining + len(exact_ids),
                        )
                        for h in hits:
                            cid = h["id"]
                            if cid in exact_ids:
                                continue
                            block = (h.get("raw_block") or "").strip()
                            if block:
                                semantic_matches.append(block)
                            if len(semantic_matches) >= remaining:
                                break
                    else:
                        # Legacy path: ONNX-cosine-in-Python over gzip JSON cache.
                        all_embeddings = self._load_all_commit_embeddings()
                        if all_embeddings:
                            ids = list(all_embeddings.keys())
                            vecs = np.stack([all_embeddings[cid] for cid in ids])
                            scores = vecs @ query_vec
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
                    pass

            if not semantic_matches and not exact_matches:
                all_records = self._storage.commit_list(branch, limit=9999)
                parts = [r.get("raw_block", "") for r in all_records if r.get("raw_block")]
                bm25_results = self._bm25_search_commits(parts, term, remaining)
                semantic_matches = [block for _, block in bm25_results]

        combined = exact_matches[:max_results]
        remaining = max_results - len(combined)
        if remaining > 0 and semantic_matches:
            combined.extend(semantic_matches[:remaining])

        result = "\n\n".join(combined[:max_results])
        try:
            self._increment_memory_metric("search_calls")
            if not result.strip():
                self._increment_memory_metric("search_zero_results")
        except Exception:
            pass
        return result

    # --- Windowed retrieval helpers ---

    def _read_commits_window(self, branch: str, offset: int, count: int) -> str:
        """Read a window of commits via storage backend."""
        records = self._storage.commit_list(branch, limit=count, offset=offset)
        return "\n".join(r.get("raw_block", "") for r in records if r.get("raw_block"))

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
        """Read last N OTA entries from the branch log via storage backend."""
        content = self._storage.log_read(branch, count * 10)
        if not content:
            return ""
        entries = re.split(r"(?=---\n\*\*\[OTA-)", content)
        entries = [e for e in entries if e.strip()]
        if not entries:
            lines = content.strip().split("\n")
            return "\n".join(lines[-count:])
        return "\n".join(entries[-count:])
