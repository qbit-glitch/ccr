"""LinksMixin — Cross-linking, clustering, and embedding methods for MemoryManager.

Heuristic commit cross-linking (A-MEM/MAGMA inspired taxonomy),
EverMemOS-inspired thematic clustering, and commit embedding persistence.
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import os
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from ccr.context.embeddings import get_embedding_model, load_embeddings, quick_cosine, save_embeddings
from ccr.core.types import CommitLink

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F2: Temporal Link Aging — EverMemOS + MAGMA (arXiv:2601.02163 + 2601.03236)
# ---------------------------------------------------------------------------

def _link_age_weight(link: dict) -> float:
    """Softer exponential decay for link age (λ=0.005/day).

    A link created 30 days ago retains ~86% weight; 60 days → ~74%.
    When ``created_at`` is absent (legacy links), returns 1.0 (no aging).
    """
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


class LinksMixin:
    """Cross-linking, clustering, and embedding methods for MemoryManager."""

    # --- Class-level constants (shared with PatternsMixin via self/cls lookup) ---

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

    # --- Static helpers used by both links and admission ---

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

    # --- Commit Embeddings ---

    def _get_links_path(self) -> str:
        return os.path.join(self.ccr_root, "commit_links.json")

    def _get_commit_embeddings_path(self) -> str:
        return os.path.join(self.ccr_root, "commit_embeddings.json.gz")

    def _embed_commit(self, commit_id: str, text: str) -> "Any":
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
        Tries sqlite-vec first, falls back to gzip JSON.
        Returns dict[str, np.ndarray]. Empty dict on error.
        """
        try:
            import numpy as np
            # Try sqlite-vec first (mirrors _load_commit_embeddings pattern)
            from ccr.context.vec_store import get_vec_store
            db_path = os.path.join(self.ccr_root, "embeddings.db")
            store = get_vec_store(db_path)
            if store is not None:
                all_ids = store.list_ids(namespace="commit")
                if all_ids:
                    batch = store.get_batch(all_ids)
                    if batch:
                        return {
                            cid: np.array(vec, dtype=np.float32)
                            for cid, vec in batch.items()
                        }
            # Fallback: gzip JSON
            raw = load_embeddings(self._get_commit_embeddings_path())
            return {
                cid: np.array(vec, dtype=np.float32)
                for cid, vec in raw.items()
            }
        except Exception as exc:
            logger.warning("Failed to load all commit embeddings: %s", exc)
            return {}

    # --- Link Graph Persistence ---

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
                # F2: stamp creation time for temporal aging
                if "created_at" not in entry:
                    entry["created_at"] = datetime.now(timezone.utc).isoformat()
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

    # --- Link Computation ---

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

    # --- Link Retrieval ---

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
        adaptive: bool = True,
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
            adaptive: If True (default), apply MAGMA-style beam pruning to
                      each hop's frontier — removes candidates whose score
                      falls below mean - 0.5*std (F3, arXiv:2601.03236).
                      Set False for exhaustive BFS without pruning.
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
            adaptive=adaptive,
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
                    adaptive=adaptive,
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
        adaptive: bool = True,
    ) -> int:
        """Push neighbors of ``src`` onto the priority heap.

        Computes edge scores using ``quick_cosine(query, commit_what)`` when
        ONNX is available, otherwise uses stored heuristic link scores.
        Applies temporal age decay (F2) and optional adaptive beam pruning (F3).

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
            adaptive: If True, apply _prune_frontier() before pushing (F3).

        Returns:
            Number of entries pushed onto the heap.
        """
        node = data.get("links", {}).get(src, {})
        # Collect all candidates before pruning
        candidates: list[tuple[float, str, str, dict]] = []  # (score, tgt, lt, entry)
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

                # F2: apply temporal age decay to traversal priority
                edge_score = edge_score * _link_age_weight(link_entry)

                candidates.append((edge_score, tgt, lt, link_entry))

        # F3: adaptive beam pruning — remove low-score frontier candidates
        pruned = _prune_frontier([(s, tgt) for s, tgt, _, _ in candidates], adaptive)
        pruned_tgts = {tgt for _, tgt in pruned}

        pushed = 0
        for edge_score, tgt, lt, link_entry in candidates:
            if tgt not in pruned_tgts:
                continue
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
