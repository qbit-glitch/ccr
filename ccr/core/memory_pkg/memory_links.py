"""LinksMixin — Cross-linking, clustering, and embedding methods for MemoryManager.

Composes three sub-mixins:
- LinkComputeMixin: heuristic link computation (entity/causal/supersession/semantic)
- LinkTraversalMixin: priority-queue BFS traversal with ONNX re-ranking
- LinkClusterMixin: EverMemOS-inspired thematic clustering

This module retains embedding I/O, link persistence, graph pruning,
and memory metrics — the shared infrastructure used by all sub-mixins.
"""

from __future__ import annotations

import logging
import os
import re

from ccr.context.embeddings import get_embedding_model, load_embeddings, save_embeddings
from ccr.core.memory_pkg.memory_links_clustering import LinkClusterMixin
from ccr.core.memory_pkg.memory_links_compute import LINK_TYPES, LinkComputeMixin  # noqa: F401
from ccr.core.memory_pkg.memory_links_traversal import LinkTraversalMixin
from ccr.core.types import CommitLink

# Re-export for backward compatibility (tests import these from memory_links)
from ccr.core.memory_pkg.memory_links_traversal import (  # noqa: F401
    _link_age_weight,
    _prune_frontier,
)

logger = logging.getLogger(__name__)


class LinksMixin(LinkComputeMixin, LinkTraversalMixin, LinkClusterMixin):
    """Cross-linking, clustering, and embedding methods for MemoryManager.

    Composes sub-mixins and provides shared infrastructure:
    - Commit embedding persistence (ONNX + sqlite-vec + gzip JSON)
    - Link graph persistence (commit_links.json)
    - Link graph pruning (size cap)
    - Memory metrics (ALMA-inspired retrieval parameter evolution)
    """

    # --- Memory Metrics (ALMA-inspired retrieval parameter evolution) ---

    def _get_memory_metrics_path(self) -> str:
        return os.path.join(self.ccr_root, "memory_metrics.json")

    def _increment_memory_metric(self, key: str, amount: int = 1) -> None:
        """Increment a memory metric counter via storage backend."""
        self._storage.metrics_increment(key, amount)

    def get_memory_metrics(self) -> dict:
        """Load and return current memory metrics."""
        data = self._storage.metrics_get()
        data.setdefault("search_calls", 0)
        data.setdefault("search_zero_results", 0)
        data.setdefault("link_creations", 0)
        data.setdefault("total_commits", 0)
        data.setdefault("last_updated", "")
        return data

    # --- Commit Embeddings ---

    def _get_links_path(self) -> str:
        return os.path.join(self.ccr_root, "commit_links.json")

    def _get_commit_embeddings_path(self) -> str:
        return os.path.join(self.ccr_root, "commit_embeddings.json.gz")

    def _embed_commit(self, commit_id: str, text: str) -> "object":
        """Embed commit text and persist to cache. Returns vector or None.

        Tries sqlite-vec first (persistent KNN store at .ccr/embeddings.db).
        Falls back to .ccr/commit_embeddings.json.gz (capped at
        link_scan_window * 2 entries, oldest evicted).
        """
        model = get_embedding_model()
        if model is None:
            return None
        try:
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
        """
        try:
            import numpy as np

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

        Tries sqlite-vec first, falls back to gzip JSON.
        Returns dict[str, np.ndarray]. Empty dict on error.
        """
        try:
            import numpy as np
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
        """Load the commit link graph via storage backend."""
        return {"version": 1, "links": self._storage.link_get_all()}

    def _save_links(self, data: dict) -> None:
        """Atomically save the commit link graph (used by tests)."""
        import json as _json
        path = self._get_links_path()
        content = _json.dumps(data, indent=2, ensure_ascii=False)
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def _prune_link_graph(self, data: dict) -> None:
        """Evict oldest commit nodes if graph exceeds link_graph_max_nodes."""
        links = data.get("links", {})
        max_nodes = self.config.link_graph_max_nodes
        if len(links) <= max_nodes:
            return

        def _commit_num(cid: str) -> int:
            m = re.search(r"\d+$", cid)
            return int(m.group()) if m else 0

        sorted_ids = sorted(links.keys(), key=_commit_num)
        evict_count = len(links) - max_nodes
        evict_set = set(sorted_ids[:evict_count])

        for cid in evict_set:
            del links[cid]

        for typed_links in links.values():
            for lt in list(typed_links.keys()):
                typed_links[lt] = [
                    e for e in typed_links[lt]
                    if e.get("target", "") not in evict_set
                ]
                if not typed_links[lt]:
                    del typed_links[lt]

    def _update_links(self, commit_id: str, links: list[CommitLink]) -> None:
        """Insert bidirectional links and prune via storage backend."""
        forward: list[dict] = []
        reverse_by_target: dict[str, list[dict]] = {}
        for cl in links:
            entry = {
                "target": cl.target,
                "link_type": cl.link_type,
                "score": cl.score,
                "shared_files": cl.shared_files,
                "snippet": cl.snippet,
            }
            forward.append(entry)
            reverse_by_target.setdefault(cl.target, []).append({
                "target": commit_id,
                "link_type": cl.link_type,
                "score": cl.score,
                "shared_files": cl.shared_files,
                "snippet": cl.snippet,
            })
        self._storage.link_insert_batch(commit_id, forward)
        for target_id, rev_links in reverse_by_target.items():
            self._storage.link_insert_batch(target_id, rev_links)
        self._storage.link_prune(self.config.link_graph_max_nodes)
        try:
            self._increment_memory_metric("link_creations", len(links))
        except Exception:
            pass
