"""EmbeddingsMixin — commit embedding persistence (sqlite-vec + gzip JSON fallback).

Methods: _get_commit_embeddings_path, _embed_commit, _load_commit_embeddings,
_load_all_commit_embeddings.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from ccr.context.embeddings import get_embedding_model, load_embeddings, save_embeddings

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingsMixin:
    """Commit embedding persistence for MemoryManager.

    Provides methods to embed, cache, and retrieve commit vectors
    via sqlite-vec (preferred) or gzip JSON fallback.
    """

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
