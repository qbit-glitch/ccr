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
SCENE_DISCOUNT = 0.6
SCENE_PER_SEED = 3
SCENE_FLOOR = 0.5  # a scene mate with no query overlap still counts for half
SCENE_MIN_LINK = 0.3
SCENE_TYPES = ("entity", "semantic")


class SearchOptions(Protocol):
    """The ``RecallOptions`` fields commit ranking reads."""

    commit_search: str
    link_expansion: bool
    scene_expansion: bool
    dense_weight: float

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
        cache[branch] = {"index": index, "stamp": None, "commits": {}, "adjacency": {},
                        "scenes": {}}
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
    entry["scenes"] = build_scenes(entry["adjacency"])
    entry["stamp"] = stamp


def _adjacency(all_links: dict[str, Any]) -> dict[str, list[tuple[str, float, str]]]:
    """Undirected commit graph from the stored A-MEM style links (all link types)."""
    graph: dict[str, list[tuple[str, float, str]]] = {}
    for source, node in all_links.items():
        for link_type, links in node.items():
            if not isinstance(links, list):
                continue
            for link in links:
                target = str(link.get("target", ""))
                if target:
                    score = float(link.get("score", 0.0))
                    graph.setdefault(source, []).append((target, score, link_type))
                    graph.setdefault(target, []).append((source, score, link_type))
    return graph


def _expand_links(
    ranked: list[tuple[str, float]], adjacency: dict[str, list[tuple[str, float, str]]],
) -> list[tuple[str, float]]:
    """Add commits linked to the top hits, scored below the hit that led to them."""
    best = dict(ranked)
    for seed, seed_score in ranked[:LINK_SEEDS]:
        for neighbor, _, _ in adjacency.get(seed, []):
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
        hits = entry["index"].search(query, limit, mode=options.commit_search,
                                     dense_weight=options.dense_weight)
    except Exception as exc:  # storage or embedding failure must not break recall
        logger.warning("hybrid commit search failed: %s", exc)
        return []
    if not hits:
        return []
    top = hits[0][1] or 1.0
    scaled = [(cid, score / top) for cid, score in hits]
    if options.link_expansion:
        scaled = _expand_links(scaled, entry["adjacency"])[:limit]
    if options.scene_expansion:
        everything = entry["index"].search(query, len(entry["index"]), mode=options.commit_search,
                                         dense_weight=options.dense_weight)
        peak = everything[0][1] if everything else 1.0
        relevance = {cid: score / (peak or 1.0) for cid, score in everything}
        scaled = expand_scenes(scaled, entry["scenes"], relevance)[:limit]
    return [(entry["commits"][cid], score) for cid, score in scaled if cid in entry["commits"]]


def build_scenes(
    adjacency: dict[str, list[tuple[str, float, str]]], min_score: float = SCENE_MIN_LINK,
) -> dict[str, frozenset[str]]:
    """EverMemOS-style scenes: connected components over entity + semantic links."""
    near: dict[str, set[str]] = {}
    for source, links in adjacency.items():
        for target, score, link_type in links:
            if link_type in SCENE_TYPES and score >= min_score:
                near.setdefault(source, set()).add(target)
                near.setdefault(target, set()).add(source)
    scenes: dict[str, frozenset[str]] = {}
    for start in near:
        if start in scenes:
            continue
        members, stack = {start}, [start]
        while stack:
            for neighbor in near[stack.pop()]:
                if neighbor not in members:
                    members.add(neighbor)
                    stack.append(neighbor)
        scene = frozenset(members)
        scenes.update({member: scene for member in members})
    return scenes


def expand_scenes(
    ranked: list[tuple[str, float]], scenes: dict[str, frozenset[str]],
    relevance: dict[str, float], per_seed: int = SCENE_PER_SEED,
    discount: float = SCENE_DISCOUNT,
) -> list[tuple[str, float]]:
    """Add each top hit's most query-relevant scene mates, scored below the hit."""
    best = dict(ranked)
    for seed, seed_score in ranked[:LINK_SEEDS]:
        mates = sorted(scenes.get(seed, frozenset()) - {seed},
                       key=lambda m: -relevance.get(m, 0.0))[:per_seed]
        for mate in mates:
            weight = SCENE_FLOOR + (1 - SCENE_FLOOR) * relevance.get(mate, 0.0)
            candidate = discount * seed_score * weight
            if candidate > best.get(mate, 0.0):
                best[mate] = candidate
    return sorted(best.items(), key=lambda kv: -kv[1])
