"""LinkTraversalMixin — Priority-queue BFS traversal for MemoryManager.

MAGMA-inspired adaptive traversal (arXiv:2601.03236): max-heap exploration
with optional query-aware scoring (ONNX cosine or word Jaccard fallback),
temporal age decay (F2), and adaptive beam pruning (F3).
"""

from __future__ import annotations

import heapq
import re
from typing import Any

from ccr.context.embeddings import get_embedding_model, quick_cosine
from ccr.core.memory_pkg.memory_links_compute import LINK_TYPES

__all__ = ["LinkTraversalMixin"]


# ---------------------------------------------------------------------------
# F2: Temporal Link Aging — EverMemOS + MAGMA (arXiv:2601.02163 + 2601.03236)
# ---------------------------------------------------------------------------

def _link_age_weight(link: dict) -> float:
    """Softer exponential decay for link age (lambda=0.005/day).

    A link created 30 days ago retains ~86% weight; 60 days -> ~74%.
    When ``created_at`` is absent (legacy links), returns 1.0 (no aging).
    """
    import math
    from datetime import datetime, timezone

    created_at = link.get("created_at", "")
    if not created_at:
        return 1.0
    try:
        dt = datetime.fromisoformat(created_at).replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
        return math.exp(-0.005 * days)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# F3: Adaptive Beam Width — MAGMA Algorithm 1 (arXiv:2601.03236)
# ---------------------------------------------------------------------------

def _prune_frontier(
    candidates: list[tuple[float, str]], adaptive: bool
) -> list[tuple[float, str]]:
    """Prune low-scoring candidates from a BFS frontier (MAGMA Alg. 1).

    Removes candidates whose score falls below ``mean - 0.5 * std``.
    Always keeps at least 2 candidates to avoid empty frontiers.
    When ``adaptive=False`` or fewer than 3 candidates, returns unchanged.
    """
    if not adaptive or len(candidates) < 3:
        return candidates
    scores = [s for s, _ in candidates]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5
    threshold = mean - 0.5 * std
    pruned = [(s, cid) for s, cid in candidates if s >= threshold]
    return pruned if len(pruned) >= 2 else candidates[:2]


class LinkTraversalMixin:
    """BFS link traversal methods for MemoryManager."""

    def get_commit_links(self, commit_id: str) -> dict[str, list[dict]]:
        """Retrieve all cross-links for a commit.

        Returns dict with keys per link type, each containing a list of link dicts.
        """
        data = self._load_links()
        node = data.get("links", {}).get(commit_id, {})
        result: dict[str, list[dict]] = {}
        for lt in LINK_TYPES:
            result[lt] = node.get(lt, [])
        return result

    def get_linked_commits(
        self,
        commit_id: str,
        link_types: list[str] | None = None,
        max_hops: int = 1,
        query: str | None = None,
        max_results: int | None = None,
        adaptive: bool = True,
    ) -> list[dict]:
        """Priority-queue traversal of commit links (MAGMA-inspired adaptive).

        Uses a max-heap to explore the most relevant linked commits first.
        When ``query`` is provided and ONNX embeddings are available, edge
        scores are computed via ``quick_cosine(query, commit_what)`` so
        traversal is intent-aware (MAGMA Alg. 1 Eq. 5-6).

        Returns list of dicts: {id, link_type, score, hop, title, what}.
        """
        data = self._load_links()
        branch = self.get_active_branch()
        types = set(link_types) if link_types else set(LINK_TYPES)
        visited: set[str] = {commit_id}
        results: list[dict] = []
        effective_limit = max_results if max_results is not None else self.config.link_max_results
        collect_limit = effective_limit * 2

        # Determine whether ONNX-based scoring is available
        use_onnx_scoring = False
        if query:
            try:
                probe = quick_cosine("a", "b")
                if probe is not None:
                    use_onnx_scoring = True
            except Exception:
                pass

        # Load cached vectors for edge scoring + post-traversal re-ranking
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

        query_scoring_active = use_onnx_scoring or (query_vec is not None)

        # Priority-queue traversal (max-heap via negated scores)
        heap: list[tuple[float, int, int, str, str, dict]] = []
        tie_counter = 0

        self._heap_push_neighbors(
            data, commit_id, types, visited, heap, tie_counter,
            hop=1, query=query, use_onnx_scoring=use_onnx_scoring,
            branch=branch, all_cached=all_cached, query_vec=query_vec,
            adaptive=adaptive,
        )
        tie_counter += len(heap)

        while heap and len(results) < collect_limit:
            neg_score, _tie, hop, tgt, lt, link_entry = heapq.heappop(heap)
            if tgt in visited:
                continue
            visited.add(tgt)
            edge_score = -neg_score

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

            if hop < max_hops:
                added = self._heap_push_neighbors(
                    data, tgt, types, visited, heap, tie_counter,
                    hop=hop + 1, query=query, use_onnx_scoring=use_onnx_scoring,
                    branch=branch, all_cached=all_cached, query_vec=query_vec,
                    adaptive=adaptive,
                )
                tie_counter += added

        # ONNX re-ranking
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

        if query_scoring_active and any("query_score" in r for r in results):
            results.sort(key=lambda r: r.get("query_score", 0.0), reverse=True)
        elif src_vec is not None:
            results.sort(key=lambda r: r.get("embedding_score", 0.0), reverse=True)

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
        adaptive: bool = True,
    ) -> int:
        """Push neighbors of ``src`` onto the priority heap.

        Returns number of entries pushed onto the heap.
        """
        node = data.get("links", {}).get(src, {})
        candidates: list[tuple[float, str, str, dict]] = []
        for lt in types:
            for link_entry in node.get(lt, []):
                tgt = link_entry.get("target", "")
                if tgt in visited:
                    continue

                edge_score: float
                if use_onnx_scoring and query:
                    commit_text = self._find_commit_by_id(branch, tgt)
                    parsed = self._parse_commit_block(commit_text) if commit_text else {}
                    tgt_what = parsed.get("what", parsed.get("title", ""))
                    cosine = quick_cosine(query, tgt_what) if tgt_what else None
                    if cosine is not None:
                        edge_score = cosine
                    else:
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

                # F2: apply temporal age decay
                edge_score = edge_score * _link_age_weight(link_entry)
                candidates.append((edge_score, tgt, lt, link_entry))

        # F3: adaptive beam pruning
        pruned = _prune_frontier([(s, tgt) for s, tgt, _, _ in candidates], adaptive)
        pruned_tgts = {tgt for _, tgt in pruned}

        pushed = 0
        for edge_score, tgt, lt, link_entry in candidates:
            if tgt not in pruned_tgts:
                continue
            heapq.heappush(
                heap,
                (-edge_score, tie_counter + pushed, hop, tgt, lt, link_entry),
            )
            pushed += 1
        return pushed

    def _format_links_for_context(self, commit_id: str, links: dict[str, list[dict]]) -> str:
        """Format commit links as Markdown for gcc_context output."""
        lines = [f"# Links for {commit_id}"]
        for lt in LINK_TYPES:
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
