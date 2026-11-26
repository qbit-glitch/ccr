"""BEIR adapter using the plain-file Hugging Face copies ``mteb/<name>``.

One world holding the full corpus and every test query that has relevance
judgments. Graded relevance is kept for nDCG through ``world_gains``. The pilot
samples queries only; the corpus is never shrunk, because a smaller corpus
would make retrieval look easier than it is.
"""

from __future__ import annotations

import csv
import json
import logging
import random

from ccr.bench import datasets
from ccr.bench.download import fetch
from ccr.bench.types import BenchConfig, Doc, Query, World

logger = logging.getLogger(__name__)

BASE = "https://huggingface.co/datasets/mteb/{name}/resolve/main/{path}"
NAMES = {"beir-scifact": "scifact", "beir-nfcorpus": "nfcorpus"}
LOAD_STATS: dict[str, int] = {}
_GAINS: dict[str, dict[str, dict[str, int]]] = {}


def world_gains(world_id: str) -> dict[str, dict[str, int]]:
    """Graded relevance per query id for a loaded BEIR world."""
    return _GAINS.get(world_id, {})


def parse_beir(world_id: str, corpus_path: str, queries_path: str, qrels_path: str) -> World:
    """Build the single BEIR world from the three files."""
    LOAD_STATS.clear()
    LOAD_STATS["dropped_qrels"] = 0
    docs: list[Doc] = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                title = (row.get("title") or "").strip()
                text = row.get("text") or ""
                docs.append(Doc(str(row["_id"]), f"{title}\n{text}" if title else text))
    known = {d.id for d in docs}
    gains: dict[str, dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8", newline="") as fh:
        rows = csv.reader(fh, delimiter="\t")
        next(rows, None)  # header: query-id, corpus-id, score
        for qid, doc_id, score in rows:
            if int(score) <= 0:
                continue
            if doc_id not in known:
                LOAD_STATS["dropped_qrels"] += 1
                continue
            gains.setdefault(qid, {})[doc_id] = int(score)
    queries: list[Query] = []
    with open(queries_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["_id"])
            if qid in gains:
                queries.append(Query(qid, row["text"], frozenset(gains[qid]), None, "beir"))
    _GAINS[world_id] = {q.id: gains[q.id] for q in queries}
    logger.info("%s parsed: %d docs, %d queries, %s", world_id, len(docs), len(queries),
                dict(LOAD_STATS))
    return World(world_id, tuple(docs), tuple(queries), world_id)


def sample_queries(world: World, fraction: float, seed: int) -> World:
    """Seeded query sample that keeps the whole corpus (smaller fraction is a subset)."""
    if fraction >= 1.0 or not world.queries:
        return world
    ordered = sorted(world.queries, key=lambda q: q.id)
    random.Random(seed).shuffle(ordered)
    keep = {q.id for q in ordered[: max(1, round(len(ordered) * fraction))]}
    picked = tuple(q for q in world.queries if q.id in keep)
    return World(world.id, world.docs, picked, world.group)


def _loader(dataset: str):
    name = NAMES[dataset]

    def load(cfg: BenchConfig) -> list[World]:
        paths = [
            fetch(BASE.format(name=name, path=p), cfg.cache_dir)
            for p in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
        ]
        return [parse_beir(dataset, *paths)]

    return load


def _sampler(worlds: list[World], cfg: BenchConfig) -> list[World]:
    return [sample_queries(w, cfg.sample_fraction, cfg.seed) for w in worlds]


for _dataset in NAMES:
    datasets.ADAPTERS[_dataset] = _loader(_dataset)
    datasets.STATS[_dataset] = LOAD_STATS
    datasets.SAMPLERS[_dataset] = _sampler
    datasets.GAINS[_dataset] = world_gains
