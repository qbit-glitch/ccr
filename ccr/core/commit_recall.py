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
from ccr.core.commit_search import CommitSearchIndex, tokenize
from ccr.core.triples import candidate_triples

logger = logging.getLogger(__name__)

__all__ = ["ranked_commits"]

LINK_SEEDS = 5
LINK_DISCOUNT = 0.9  # fused scores are flat, so 0.5 would sink neighbours below the pool
SCENE_DISCOUNT = 0.6
SCENE_PER_SEED = 3
SCENE_FLOOR = 0.5  # a scene mate with no query overlap still counts for half
SCENE_MIN_LINK = 0.3
SCENE_TYPES = ("entity", "semantic")
TRIPLE_BOOST = 0.5  # a commit whose triples cover every query term scores 1.5x
CHUNK_WORDS = 160  # about one MiniLM window (256 word pieces)


class SearchOptions(Protocol):
    """The ``RecallOptions`` fields commit ranking reads."""

    commit_search: str
    link_expansion: bool
    scene_expansion: bool
    triple_boost: bool
    chunk_dense: bool
    dense_weight: float

_CACHE_ATTR = "_commit_search_cache"


_WHAT_BLOCK = re.compile(r"\*\*What\*\*:(.*?)\n\*\*Why\*\*:", re.DOTALL)


def full_what(commit: dict[str, Any]) -> str:
    """The whole ``What`` body; the parsed ``what`` field keeps only its first line."""
    match = _WHAT_BLOCK.search(str(commit.get("raw_block") or ""))
    return match.group(1).strip() if match else str(commit.get("what") or "")


def _commit_text(commit: dict[str, Any]) -> str:
    return " ".join([str(commit.get("title") or ""), full_what(commit),
                     str(commit.get("why") or ""), str(commit.get("next") or "")])


def _vecs_path(mem: Any, branch: str, chunk_words: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", branch) or "main"
    suffix = f".c{chunk_words}" if chunk_words else ""
    return os.path.join(mem.ccr_root, f"commit_search_vecs.{safe}{suffix}.npz")


def _entry(mem: Any, branch: str, chunk_words: int = 0) -> dict[str, Any]:
    cache = mem.__dict__.setdefault(_CACHE_ATTR, {})
    key = (branch, chunk_words)
    if key not in cache:
        index = CommitSearchIndex(embedder=get_embedding_model(), chunk_words=chunk_words,
                                  vecs_path=_vecs_path(mem, branch, chunk_words))
        cache[key] = {"index": index, "stamp": None, "commits": {}, "adjacency": {},
                      "scenes": {}, "entities": {}}
    return cache[key]


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
    entry["entities"] = {cid: _triple_terms(c) for cid, c in entry["commits"].items()}
    entry["adjacency"] = _adjacency(storage.link_get_all())
    entry["scenes"] = build_scenes(entry["adjacency"])
    entry["stamp"] = stamp


def _triple_terms(commit: dict[str, Any]) -> frozenset[str]:
    """Words of the entities in the commit's regex triples (file triples excluded)."""
    triples = candidate_triples(str(commit.get("id", "")), str(commit.get("title", "")),
                                full_what(commit), str(commit.get("why", "")), [])
    return frozenset(t for tr in triples for t in tokenize(f"{tr.subject} {tr.object}"))


def _boost_by_triples(
    ranked: list[tuple[str, float]], entities: dict[str, frozenset[str]], query: str,
) -> list[tuple[str, float]]:
    """Memori: scale a commit by the share of query terms its triples mention."""
    terms = set(tokenize(query))
    if not terms:
        return ranked
    boosted = [(cid, score * (1 + TRIPLE_BOOST * len(terms & entities.get(cid, frozenset()))
                              / len(terms))) for cid, score in ranked]
    return sorted(boosted, key=lambda kv: -kv[1])


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
        for neighbor, strength, _ in adjacency.get(seed, []):
            candidate = seed_score * LINK_DISCOUNT * min(1.0, max(strength, 0.0))
            if candidate > best.get(neighbor, 0.0):
                best[neighbor] = candidate
    return sorted(best.items(), key=lambda kv: -kv[1])


def ranked_commits(
    mem: Any, branch: str, query: str, limit: int, options: SearchOptions,
) -> list[tuple[dict[str, Any], float]]:
    """Top ``limit`` commits for ``query`` as ``(commit, score)``, score scaled to (0, 1]."""
    entry = _entry(mem, branch, CHUNK_WORDS if options.chunk_dense else 0)
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
    if options.triple_boost:
        scaled = _boost_by_triples(scaled, entry["entities"], query)
    if options.link_expansion:
        scaled = _expand_links(scaled, entry["adjacency"])[:limit]
    if options.scene_expansion:
        everything = entry["index"].search(query, len(entry["index"]), mode=options.commit_search,
                                         dense_weight=options.dense_weight)
        peak = everything[0][1] if everything else 1.0
        relevance = {cid: score / (peak or 1.0) for cid, score in everything}
        scaled = expand_scenes(scaled, entry["scenes"], relevance)[:limit]
    return [({**entry["commits"][cid], "what": full_what(entry["commits"][cid])}, score)
            for cid, score in scaled if cid in entry["commits"]]


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
