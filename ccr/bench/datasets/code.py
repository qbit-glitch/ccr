"""Code-retrieval adapter: Hugging Face ``mteb/CodeSearchNetRetrieval``, python.

Source: https://huggingface.co/datasets/mteb/CodeSearchNetRetrieval
(MIT license, not gated; derived from code-search-net/code_search_net). The
python configs hold 1000 functions, 1000 docstring queries and one gold
function per query, so the whole corpus is used and only queries are sampled
in the pilot.

Used exactly as published, so numbers stay comparable with MTEB's protocol.
Note the queries are docstrings and the corpus functions still contain their
docstring, so lexical overlap between query and gold is high by design.
"""

from __future__ import annotations

import logging
from typing import Any

from ccr.bench import datasets
from ccr.bench.datasets.beir import sample_queries
from ccr.bench.download import fetch
from ccr.bench.types import BenchConfig, Doc, Query, World

logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co/datasets/mteb/CodeSearchNetRetrieval/resolve/main"
LANGUAGE = "python"
NAME = "code"
LOAD_STATS: dict[str, int] = {}
_GAINS: dict[str, dict[str, dict[str, int]]] = {}


def world_gains(world_id: str) -> dict[str, dict[str, int]]:
    """Relevance per query id for a loaded code world."""
    return _GAINS.get(world_id, {})


def parse_code(
    world_id: str, corpus: list[dict[str, Any]], queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> World:
    """Build the single code-retrieval world from the three tables."""
    LOAD_STATS.clear()
    LOAD_STATS["dropped_qrels"] = 0
    docs = tuple(
        Doc(str(r["id"]), f"{r['title']}\n{r['text']}" if r.get("title") else r["text"])
        for r in corpus
    )
    known = {d.id for d in docs}
    gains: dict[str, dict[str, int]] = {}
    for row in qrels:
        score = int(row["score"])
        if score <= 0:
            continue
        if row["corpus-id"] not in known:
            LOAD_STATS["dropped_qrels"] += 1
            continue
        gains.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score
    picked = tuple(
        Query(str(q["id"]), q["text"], frozenset(gains[str(q["id"])]), None, NAME)
        for q in queries if str(q["id"]) in gains
    )
    _GAINS[world_id] = {q.id: gains[q.id] for q in picked}
    logger.info("%s parsed: %d docs, %d queries, %s", world_id, len(docs), len(picked),
                dict(LOAD_STATS))
    return World(world_id, docs, picked, world_id)


def load_code(cfg: BenchConfig, base_url: str = BASE_URL) -> list[World]:
    """Download (cached) and parse the python CodeSearchNet retrieval set."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('code retrieval needs pyarrow: pip install -e ".[bench]"') from exc
    tables = []
    for part in ("corpus", "queries", "qrels"):
        url = f"{base_url}/{LANGUAGE}-{part}/test-00000-of-00001.parquet"
        tables.append(pq.read_table(fetch(url, cfg.cache_dir)).to_pylist())
    return [parse_code(NAME, *tables)]


def _sampler(worlds: list[World], cfg: BenchConfig) -> list[World]:
    return [sample_queries(w, cfg.sample_fraction, cfg.seed) for w in worlds]


datasets.ADAPTERS[NAME] = load_code
datasets.STATS[NAME] = LOAD_STATS
datasets.SAMPLERS[NAME] = _sampler
datasets.GAINS[NAME] = world_gains
