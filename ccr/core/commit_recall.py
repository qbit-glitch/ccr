"""Ranked commit retrieval for ``RecallEngine`` using the hybrid search index.

One ``CommitSearchIndex`` per (MemoryManager, branch) lives on the manager, so the
per-call ``RecallEngine`` objects the MCP tool creates share it and nothing is
re-embedded. The index re-syncs whenever the commit count or newest id changes.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Protocol

from ccr.context.embeddings import get_embedding_model
from ccr.core.commit_search import CommitSearchIndex

logger = logging.getLogger(__name__)

__all__ = ["ranked_commits"]

LINK_SEEDS = 5
LINK_DISCOUNT = 0.5


class SearchOptions(Protocol):
    """The ``RecallOptions`` fields commit ranking reads."""

    commit_search: str
    link_expansion: bool

_CACHE_ATTR = "_commit_search_cache"


def _commit_text(commit: dict[str, Any]) -> str:
    return " ".join(str(commit.get(k) or "") for k in ("title", "what", "why", "next"))


def _vecs_path(mem: Any, branch: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", branch) or "main"
    return os.path.join(mem.ccr_root, f"commit_search_vecs.{safe}.npz")


def _entry(mem: Any, branch: str) -> dict[str, Any]:
    cache = mem.__dict__.setdefault(_CACHE_ATTR, {})
    if branch not in cache:
        index = CommitSearchIndex(embedder=get_embedding_model(),
                                  vecs_path=_vecs_path(mem, branch))
        cache[branch] = {"index": index, "stamp": None, "commits": {}, "adjacency": {}}
    return cache[branch]


def _refresh(mem: Any, branch: str, entry: dict[str, Any]) -> None:
    storage = mem._storage
    count = storage.commit_count(branch)
    newest = storage.commit_list(branch, limit=1)
    stamp = (count, newest[0]["id"] if newest else "")
    if stamp == entry["stamp"]:
        return
    commits = storage.commit_list(branch, limit=max(count, 1))
    entry["commits"] = {str(c["id"]): c for c in commits}
    entry["index"].sync([(str(c["id"]), _commit_text(c)) for c in reversed(commits)])
    entry["index"].save()
    entry["adjacency"] = _adjacency(storage.link_get_all())
    entry["stamp"] = stamp


def _adjacency(all_links: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    """Undirected commit graph from the stored A-MEM style links (all link types)."""
    graph: dict[str, list[tuple[str, float]]] = {}
    for source, node in all_links.items():
        for links in node.values():
            if not isinstance(links, list):
                continue
            for link in links:
                target = str(link.get("target", ""))
                if target:
                    score = float(link.get("score", 0.0))
                    graph.setdefault(source, []).append((target, score))
                    graph.setdefault(target, []).append((source, score))
    return graph


def _expand_links(
    ranked: list[tuple[str, float]], adjacency: dict[str, list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """Add commits linked to the top hits, scored below the hit that led to them."""
    best = dict(ranked)
    for seed, seed_score in ranked[:LINK_SEEDS]:
        for neighbor, _ in adjacency.get(seed, []):
            candidate = seed_score * LINK_DISCOUNT
            if candidate > best.get(neighbor, 0.0):
                best[neighbor] = candidate
    return sorted(best.items(), key=lambda kv: -kv[1])


def ranked_commits(
    mem: Any, branch: str, query: str, limit: int, options: SearchOptions,
) -> list[tuple[dict[str, Any], float]]:
    """Top ``limit`` commits for ``query`` as ``(commit, score)``, score scaled to (0, 1]."""
    entry = _entry(mem, branch)
    try:
        _refresh(mem, branch, entry)
        hits = entry["index"].search(query, limit, mode=options.commit_search)
    except Exception as exc:  # storage or embedding failure must not break recall
        logger.warning("hybrid commit search failed: %s", exc)
        return []
    if not hits:
        return []
    top = hits[0][1] or 1.0
    scaled = [(cid, score / top) for cid, score in hits]
    if options.link_expansion:
        scaled = _expand_links(scaled, entry["adjacency"])[:limit]
    return [(entry["commits"][cid], score) for cid, score in scaled if cid in entry["commits"]]
