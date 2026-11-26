"""LongMemEval adapter, retrieval protocol only (judge-based QA accuracy is not run).

Source: Hugging Face ``xiaowu0162/longmemeval-cleaned``, file
``longmemeval_s_cleaned.json`` (about 50 sessions per question). One world per
question instance, one doc per haystack session (id = session id), gold =
``answer_session_ids``. Abstention questions (``question_id`` ending ``_abs``)
have no answer session and are excluded and counted.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ccr.bench import datasets
from ccr.bench.download import fetch
from ccr.bench.sampling import sample_worlds
from ccr.bench.types import BenchConfig, Doc, Query, World

logger = logging.getLogger(__name__)

URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)
LOAD_STATS: dict[str, int] = {}


def _session_text(date: str, turns: list[dict[str, Any]]) -> str:
    body = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    return f"[{date}]\n{body}"


def parse_longmemeval(data: list[dict[str, Any]]) -> list[World]:
    """Convert raw LongMemEval records into worlds and refresh ``LOAD_STATS``."""
    LOAD_STATS.clear()
    LOAD_STATS.update({"abstention_excluded": 0, "dropped_gold_ids": 0, "duplicate_session_ids": 0})
    worlds: list[World] = []
    for item in data:
        docs_list: list[Doc] = []
        seen: set[str] = set()
        for sid, date, turns in zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
        ):
            doc_id = sid
            while doc_id in seen:  # keep the first under its own id; suffix repeats
                LOAD_STATS["duplicate_session_ids"] += 1
                doc_id = f"{doc_id}#dup"
            seen.add(doc_id)
            docs_list.append(Doc(doc_id, _session_text(date, turns)))
        docs = tuple(docs_list)
        known = {d.id for d in docs}
        qid = str(item["question_id"])
        if qid.endswith("_abs"):
            LOAD_STATS["abstention_excluded"] += 1
            gold, answer = frozenset(), None
        else:
            wanted = list(item["answer_session_ids"])
            gold = frozenset(i for i in wanted if i in known)
            LOAD_STATS["dropped_gold_ids"] += len(wanted) - len(gold)
            answer = str(item["answer"])
        query = Query(qid, item["question"], gold, answer, item["question_type"])
        worlds.append(World(qid, docs, (query,), item["question_type"]))
    logger.info("longmemeval parsed: %s", dict(LOAD_STATS))
    return worlds


def load_longmemeval(cfg: BenchConfig) -> list[World]:
    """Download (cached, large) and parse LongMemEval-S."""
    with open(fetch(URL, cfg.cache_dir), encoding="utf-8") as fh:
        return parse_longmemeval(json.load(fh))


def sample_by_type(worlds: list[World], cfg: BenchConfig) -> list[World]:
    """Sample whole question instances, stratified by question type."""
    return sample_worlds(worlds, cfg.sample_fraction, cfg.seed, stratify_by=lambda w: w.group)


datasets.ADAPTERS["longmemeval"] = load_longmemeval
datasets.STATS["longmemeval"] = LOAD_STATS
datasets.SAMPLERS["longmemeval"] = sample_by_type
