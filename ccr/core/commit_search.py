"""Hybrid commit search: BM25 (OR semantics) + dense vectors fused by RRF.

Adapts the retrieval ideas CCR is built on: A-RAG's keyword / semantic / hybrid
modes and ExpRAG's embedding-based experience retrieval. Pure Python plus numpy;
the dense half switches on only when an embedder is supplied (ONNX MiniLM).
There is no result cap and no size cap on stored vectors.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CommitSearchIndex", "rrf", "tokenize"]

K1 = 1.5
B = 0.75
RRF_K = 60
POOL_FLOOR = 100
BATCH = 32
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a about above after again all also am an and any are as at be because been before being "
    "below between both but by can could did do does doing down during each few for from further "
    "had has have having he her here hers him his how i if in into is it its just me more most my "
    "no nor not of off on once only or other our out over own same she should so some such than "
    "that the their them then there these they this those through to too under until up very was "
    "we were what when where which while who whom why will with would you your".split()
)


class Embedder(Protocol):
    """The slice of ``EmbeddingModel`` this index needs."""

    def embed_batch(self, texts: list[str]) -> Any: ...

    def embed_query(self, text: str) -> Any: ...


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without stop words and single characters."""
    return [t for t in _WORD.findall(text.lower()) if len(t) > 1 and t not in _STOP]


def rrf(
    rankings: Sequence[Sequence[str]], k: int = RRF_K, weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Weighted reciprocal-rank fusion; ties keep the order of first appearance."""
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for n, ranking in enumerate(rankings):
        weight = weights[n] if weights else 1.0
        for rank, item in enumerate(ranking, 1):
            scores[item] += weight / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


class CommitSearchIndex:
    """Searchable view over ``(commit_id, text)`` pairs, kept in sync incrementally."""

    def __init__(
        self, embedder: Embedder | None = None, vecs_path: str | None = None, chunk_words: int = 0,
    ) -> None:
        """``chunk_words > 0`` embeds each commit as chunks and scores it by its best one (A-RAG)."""
        self._embedder = embedder
        self._vecs_path = vecs_path
        self._chunk_words = max(0, chunk_words)
        self._order: list[str] = []
        self._hash: dict[str, str] = {}
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._lengths: list[int] = []
        self._vec: dict[str, tuple[str, np.ndarray]] = {}  # id -> (key, rows of shape n x dim)
        self._matrix: np.ndarray | None = None
        self._owner: np.ndarray | None = None  # matrix row -> position in ``_order``
        self._dirty = False
        if embedder is not None and vecs_path:
            self._load()

    def __len__(self) -> int:
        return len(self._order)

    @property
    def dense_available(self) -> bool:
        return self._embedder is not None

    def sync(self, commits: Sequence[tuple[str, str]]) -> None:
        """Bring the index in line with ``commits``; only new or edited text is processed."""
        wanted = {cid: _digest(text) for cid, text in commits}
        stale = any(cid not in wanted or self._hash[cid] != wanted[cid] for cid in self._order)
        if stale:
            self._reset_lexical()
        known = set(self._order)
        fresh = [(cid, text) for cid, text in commits if cid not in known]
        for cid, text in fresh:
            self._add_lexical(cid, wanted[cid], text)
        if self._embedder is not None:
            self._embed_missing(commits, wanted)
        if fresh or stale:
            self._rebuild_matrix()

    def search(
        self, query: str, k: int, mode: str = "hybrid", dense_weight: float = 1.0,
    ) -> list[tuple[str, float]]:
        """Top ``k`` ``(commit_id, score)`` pairs, best first."""
        if k <= 0 or not self._order:
            return []
        pool = max(k * 4, POOL_FLOOR)
        lexical = self._bm25(query, pool) if mode in ("bm25", "hybrid") else []
        dense = self._dense(query, pool) if mode in ("dense", "hybrid") else []
        if mode == "bm25" or (mode == "hybrid" and not dense):
            return lexical[:k]
        if mode == "dense":
            return dense[:k]
        return rrf([[c for c, _ in lexical], [c for c, _ in dense]],
                   weights=[1.0, dense_weight])[:k]

    def save(self) -> None:
        """Persist vectors atomically so a restart does not re-embed."""
        if not (self._vecs_path and self._dirty and self._vec):
            return
        ids = [cid for cid, (_, rows) in self._vec.items() for _ in range(len(rows))]
        directory = os.path.dirname(os.path.abspath(self._vecs_path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".vecs.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, ids=np.array(ids), hashes=np.array([self._vec[i][0] for i in ids]),
                         matrix=np.concatenate([rows for _, rows in self._vec.values()]))
            os.replace(tmp, self._vecs_path)
            self._dirty = False
        except OSError as exc:
            logger.warning("could not persist commit vectors: %s", exc)
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _reset_lexical(self) -> None:
        self._order, self._hash, self._lengths = [], {}, []
        self._postings = defaultdict(dict)

    def _add_lexical(self, cid: str, digest: str, text: str) -> None:
        position = len(self._order)
        tokens = tokenize(text)
        for term in set(tokens):
            self._postings[term][position] = tokens.count(term)
        self._order.append(cid)
        self._hash[cid] = digest
        self._lengths.append(len(tokens))

    def _bm25(self, query: str, pool: int) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        n = len(self._order)
        avg = (sum(self._lengths) / n) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for pos, tf in postings.items():
                norm = tf + K1 * (1 - B + B * self._lengths[pos] / avg)
                scores[pos] += idf * tf * (K1 + 1) / norm
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:pool]
        return [(self._order[pos], score) for pos, score in best]

    def _units(self, text: str) -> list[str]:
        """The pieces of ``text`` that get their own vector."""
        words = text.split()
        if self._chunk_words <= 0 or len(words) <= self._chunk_words:
            return [text]
        step = self._chunk_words
        return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]

    def _embed_missing(self, commits: Sequence[tuple[str, str]], wanted: dict[str, str]) -> None:
        key = {cid: f"{wanted[cid]}:{self._chunk_words}" for cid in wanted}
        todo = [(cid, text) for cid, text in commits
                if cid not in self._vec or self._vec[cid][0] != key[cid]]
        pieces = [(cid, unit) for cid, text in todo for unit in self._units(text)]
        rows: dict[str, list[np.ndarray]] = defaultdict(list)
        for start in range(0, len(pieces), BATCH):
            batch = pieces[start:start + BATCH]
            vectors = np.asarray(self._embedder.embed_batch([u for _, u in batch]),
                                 dtype=np.float32)
            for (cid, _), vec in zip(batch, vectors):
                rows[cid].append(vec)
        for cid, vecs in rows.items():
            self._vec[cid] = (key[cid], np.stack(vecs))
            self._dirty = True
        for gone in set(self._vec) - set(wanted):
            del self._vec[gone]
            self._dirty = True

    def _rebuild_matrix(self) -> None:
        if not self._order or any(cid not in self._vec for cid in self._order):
            self._matrix = self._owner = None
            return
        self._matrix = np.concatenate([self._vec[cid][1] for cid in self._order])
        self._owner = np.concatenate([np.full(len(self._vec[cid][1]), pos, dtype=np.int64)
                                      for pos, cid in enumerate(self._order)])

    def _dense(self, query: str, pool: int) -> list[tuple[str, float]]:
        if self._embedder is None or self._matrix is None or not query.strip():
            return []
        rows = self._matrix @ np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        sims = np.full(len(self._order), -np.inf, dtype=np.float32)
        np.maximum.at(sims, self._owner, rows)
        top = np.argsort(-sims, kind="stable")[:pool]
        return [(self._order[int(i)], float(sims[i])) for i in top]

    def _load(self) -> None:
        try:
            with np.load(self._vecs_path, allow_pickle=False) as data:
                ids, hashes, matrix = data["ids"], data["hashes"], data["matrix"]
            if not (len(ids) == len(hashes) == len(matrix)):
                return
            grouped: dict[str, list[int]] = defaultdict(list)
            for n, cid in enumerate(ids):
                grouped[str(cid)].append(n)
            self._vec = {cid: (str(hashes[at[0]]), matrix[at]) for cid, at in grouped.items()}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            logger.debug("ignoring unreadable vector file %s: %s", self._vecs_path, exc)
