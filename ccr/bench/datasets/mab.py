"""MemoryAgentBench adapter, judge-free sub-tasks only (ai-hyz/MemoryAgentBench).

The project's own metric table (its README, "Clarification on Evaluation
Metrics") says these sources score by substring exact match or exact match, so
they need no judge: event_qa, ruler_qa1/2, fact_mh/sh, the ICL series and
detectiveQA. Excluded: longmemeval (LLM judge), infbench summarization (LLM
judge) and recsys (Recall@5 over generated movie lists, not a short answer).

The data has no evidence labels, so queries carry no gold ids: only answer
metrics (exact match, substring match, F1) and tokens apply, and a reader is
required. One world per example; docs are fixed-size chunks of its context.
Chunk size uses 4 characters per token as an approximation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ccr.bench import datasets
from ccr.bench.download import fetch
from ccr.bench.sampling import sample_worlds
from ccr.bench.types import BenchConfig, Doc, Query, World

logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/data"
SPLITS = ("Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding",
          "Conflict_Resolution")
JUDGE_FREE_PREFIXES = ("ruler_qa", "eventqa", "factconsolidation", "icl_", "detective")
CHARS_PER_TOKEN = 4
LOAD_STATS: dict[str, int] = {}


def is_judge_free(source: str) -> bool:
    """True when the source scores by (substring) exact match per the project's table."""
    return source.lower().startswith(JUDGE_FREE_PREFIXES)


def family(source: str) -> str:
    """Stratum used by the pilot sampler: the source without its size suffix."""
    s = source.lower()
    if s.startswith("ruler_qa"):
        return "ruler_qa"
    if s.startswith("factconsolidation"):
        return "factconsolidation_mh" if "_mh" in s else "factconsolidation_sh"
    return re.split(r"[_\d]", s, maxsplit=1)[0]


def _split_long(line: str, budget: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in line.split(" "):
        if current and size + len(word) + 1 > budget:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_context(context: str, chunk_tokens: int) -> list[str]:
    """Split on line boundaries into chunks of about ``chunk_tokens`` tokens."""
    budget = max(1, chunk_tokens * CHARS_PER_TOKEN)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in context.split("\n"):
        for piece in (_split_long(line, budget) if len(line) > budget else [line]):
            if current and size + len(piece) + 1 > budget:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


def parse_mab(rows: list[dict[str, Any]], chunk_tokens: int) -> list[World]:
    """Convert raw rows into worlds, dropping sources that need a judge."""
    LOAD_STATS.clear()
    LOAD_STATS["dropped_examples"] = 0
    counters: dict[str, int] = {}
    worlds: list[World] = []
    for row in rows:
        source = str(row["metadata"]["source"])
        if not is_judge_free(source):
            LOAD_STATS["dropped_examples"] += 1
            continue
        n = counters.get(source, 0)
        counters[source] = n + 1
        wid = f"{source}:{n}"
        docs = tuple(Doc(f"{wid}#{i}", text)
                     for i, text in enumerate(chunk_context(row["context"], chunk_tokens)))
        queries = tuple(
            Query(f"{wid}:q{i}", question, frozenset(), str(ans[0]), source,
                  tuple(str(a) for a in ans[1:]))
            for i, (question, ans) in enumerate(zip(row["questions"], row["answers"]))
            if ans
        )
        worlds.append(World(wid, docs, queries, source))
    logger.info("memoryagentbench parsed: %d worlds, %s", len(worlds), dict(LOAD_STATS))
    return worlds


def load_mab(cfg: BenchConfig, base_url: str = BASE_URL) -> list[World]:
    """Download (cached) every split's parquet and parse the judge-free sources."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            'memoryagentbench needs pyarrow: pip install -e ".[bench]"'
        ) from exc
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = fetch(f"{base_url}/{split}-00000-of-00001.parquet", cfg.cache_dir)
        table = pq.read_table(path, columns=["context", "questions", "answers", "metadata"])
        rows.extend(table.to_pylist())
    return parse_mab(rows, cfg.chunk_tokens)


def sample_by_family(worlds: list[World], cfg: BenchConfig) -> list[World]:
    """10% pilot stratified by task family so every judge-free task is represented."""
    return sample_worlds(worlds, cfg.sample_fraction, cfg.seed,
                         stratify_by=lambda w: family(w.group))


datasets.ADAPTERS["memoryagentbench"] = load_mab
datasets.STATS["memoryagentbench"] = LOAD_STATS
datasets.SAMPLERS["memoryagentbench"] = sample_by_family
datasets.NEEDS_READER.add("memoryagentbench")
