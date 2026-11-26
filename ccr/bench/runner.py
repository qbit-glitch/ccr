"""Resumable benchmark runner: ingest a world, retrieve, optionally read, score."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from typing import Any

from ccr.bench.metrics import (
    bleu1,
    count_tokens,
    exact_match,
    mrr,
    ndcg_at_k,
    recall_at_k,
    sub_exact_match,
    token_f1,
)
from ccr.bench.results import QueryResult, ResultStore
from ccr.bench.types import BenchConfig, Query, World

logger = logging.getLogger(__name__)

_READER_COUNTERS = ("timeout_count", "empty_count", "truncated_count")
_READER_FLAGS = ("reader_timeout", "reader_empty", "reader_truncated")

GainsFor = Callable[[str], dict[str, dict[str, int]]]


def _key(world: World, system: str, query: Query, tag: str) -> str:
    return f"{world.id}|{system}|{query.id}|{tag}"


def _excluded(world: World, system: str, query: Query) -> QueryResult:
    return QueryResult(query.id, world.id, system, query.category, (), 0, None,
                       {"excluded": 1.0}, False, 0.0, 0.0)


def _answer_scores(pred: str, golds: list[str]) -> dict[str, float]:
    """Score against every acceptable gold answer and keep the best per metric."""
    metrics = {"f1": token_f1, "em": exact_match, "sub_em": sub_exact_match, "bleu1": bleu1}
    return {name: max(fn(pred, g) for g in golds) for name, fn in metrics.items()}


def _run_query(
    world: World, retriever: Any, cfg: BenchConfig, reader: Any, query: Query,
    docs_by_id: dict[str, str], gains: dict[str, dict[str, int]],
    context_limit_tokens: int | None, ingest_share: float,
) -> QueryResult:
    started = time.perf_counter()
    ranked = list(dict.fromkeys(retriever.retrieve(query.text, max(cfg.ks))))
    retrieve_s = time.perf_counter() - started
    ranks = getattr(retriever, "ranks_docs", True)
    scores: dict[str, float] = {}
    if ranks and query.gold_ids:
        graded = gains.get(query.id) or {g: 1 for g in query.gold_ids}
        for k in cfg.ks:
            scores[f"recall@{k}"] = recall_at_k(ranked, query.gold_ids, k)
            scores[f"ndcg@{k}"] = ndcg_at_k(ranked, graded, k)
        scores["mrr"] = mrr(ranked, query.gold_ids)
    context_ids = ranked[: cfg.reader_k] if ranks else ranked
    context = "\n\n".join(docs_by_id[i] for i in context_ids if i in docs_by_id)
    tokens = count_tokens(context)
    answer: str | None = None
    if query.answer is not None:
        if context_limit_tokens is not None and tokens > context_limit_tokens:
            scores["not_run"] = 1.0
        elif reader is not None:
            before = tuple(getattr(reader, n, 0) for n in _READER_COUNTERS)
            answer = reader.answer(query.text, context)
            after = tuple(getattr(reader, n, 0) for n in _READER_COUNTERS)
            for flag, b, a in zip(_READER_FLAGS, before, after):
                if a > b:
                    scores[flag] = 1.0
            scores.update(_answer_scores(answer, [query.answer, *query.alt_answers]))
    return QueryResult(
        query.id, world.id, retriever.name, query.category,
        tuple(ranked[: max(cfg.ks)]) if ranks else (), tokens, answer, scores,
        not ranked, ingest_share, retrieve_s,
    )


def run_world(
    world: World, retriever: Any, cfg: BenchConfig, reader: Any, store: ResultStore,
    context_limit_tokens: int | None, gains_for: GainsFor | None = None,
) -> int:
    """Evaluate every pending query of ``world``; return the number of new results."""
    cap = getattr(retriever, "max_k", None)
    if cap is not None and max(cfg.ks) > cap:
        raise ValueError(f"{retriever.name}: k={max(cfg.ks)} exceeds max_k={cap}")
    tag = reader.tag if reader is not None else "-"
    new = 0
    runnable: list[Query] = []
    for query in world.queries:
        key = _key(world, retriever.name, query, tag)
        done = store.get(key)
        if done is not None and "reader_timeout" not in done.scores:
            continue
        if not query.gold_ids and query.answer is None:
            store.add(key, _excluded(world, retriever.name, query))
            new += 1
        else:
            runnable.append(query)
    if not runnable:
        return new
    docs_by_id = {d.id: d.text for d in world.docs}
    gains = gains_for(world.id) if gains_for else {}
    with tempfile.TemporaryDirectory(prefix="ccr-bench-") as workdir:
        try:
            started = time.perf_counter()
            retriever.index(world.docs, workdir)
            share = (time.perf_counter() - started) / len(runnable)
            for query in runnable:
                result = _run_query(world, retriever, cfg, reader, query, docs_by_id,
                                    gains, context_limit_tokens, share)
                store.add(_key(world, retriever.name, query, tag), result)
                new += 1
        finally:
            retriever.close()
    return new
