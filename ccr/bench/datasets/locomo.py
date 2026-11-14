"""LoCoMo adapter (snap-research/locomo, ``data/locomo10.json``).

One world per conversation, one doc per dialog turn (id = ``dia_id`` such as
``D1:3``). Session dates are prefixed to each turn because many questions are
temporal, and image captions are appended because some evidence turns carry
their answer only in the caption.

Real-file quirks handled here: some ``evidence`` strings pack several ids
(``"D8:6; D9:17"`` or ``"D9:1 D4:4"``), some ids do not exist in the
conversation, category-5 (adversarial) questions have evidence but no reference
answer, and a few answers are integers.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ccr.bench import datasets
from ccr.bench.download import fetch
from ccr.bench.types import BenchConfig, Doc, Query, World

logger = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
ADVERSARIAL = 5
_SPLIT = re.compile(r"[;,\s]+")
_SESSION = re.compile(r"^session_(\d+)$")

LOAD_STATS: dict[str, int] = {}


def _turn_text(date: str, turn: dict[str, Any]) -> str:
    text = f"[{date}] {turn['speaker']}: {turn['text']}"
    caption = turn.get("blip_caption")
    return f"{text} [shared image: {caption}]" if caption else text


def _docs(conversation: dict[str, Any]) -> tuple[Doc, ...]:
    docs: list[Doc] = []
    sessions = sorted(
        (int(m.group(1)), key) for key in conversation if (m := _SESSION.match(key))
    )
    for _, key in sessions:
        date = conversation.get(f"{key}_date_time", "")
        docs.extend(Doc(t["dia_id"], _turn_text(date, t)) for t in conversation[key])
    return tuple(docs)


def _gold(evidence: list[str], known: set[str]) -> frozenset[str]:
    ids = [p for raw in evidence for p in _SPLIT.split(raw.strip()) if p]
    kept = [i for i in ids if i in known]
    LOAD_STATS["dropped_evidence_ids"] += len(ids) - len(kept)
    return frozenset(kept)


def parse_locomo(data: list[dict[str, Any]]) -> list[World]:
    """Convert the raw LoCoMo JSON into worlds and refresh ``LOAD_STATS``."""
    LOAD_STATS.clear()
    LOAD_STATS.update({"dropped_evidence_ids": 0, "adversarial_excluded": 0})
    worlds: list[World] = []
    for sample in data:
        sid = sample["sample_id"]
        docs = _docs(sample["conversation"])
        known = {d.id for d in docs}
        queries: list[Query] = []
        for n, qa in enumerate(sample["qa"]):
            category = int(qa["category"])
            if category == ADVERSARIAL:
                LOAD_STATS["adversarial_excluded"] += 1
                gold, answer = frozenset(), None
            else:
                gold = _gold(qa.get("evidence", []), known)
                answer = str(qa["answer"]) if "answer" in qa else None
            queries.append(Query(f"{sid}:q{n}", qa["question"], gold, answer, f"cat{category}"))
        worlds.append(World(sid, docs, tuple(queries), sid))
    logger.info("locomo parsed: %s", dict(LOAD_STATS))
    return worlds


def load_locomo(cfg: BenchConfig) -> list[World]:
    """Download (cached) and parse LoCoMo."""
    path = fetch(URL, cfg.cache_dir)
    with open(path, encoding="utf-8") as fh:
        return parse_locomo(json.load(fh))


datasets.ADAPTERS["locomo"] = load_locomo
