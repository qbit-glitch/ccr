"""Per-query result records and an append-only JSONL store (resume support)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QueryResult:
    """Everything recorded for one (world, system, query) evaluation."""

    query_id: str
    world_id: str
    system: str
    category: str
    ranked_ids: tuple[str, ...]
    retrieved_tokens: int
    answer: str | None
    scores: dict[str, float]
    empty: bool
    ingest_s: float
    retrieve_s: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ranked_ids"] = list(self.ranked_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryResult":
        return cls(**{**data, "ranked_ids": tuple(data["ranked_ids"])})


class ResultStore:
    """Append-only JSONL file keyed by a string; reloaded on construction."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._items: dict[str, QueryResult] = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self._items[row["key"]] = QueryResult.from_dict(row["result"])
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def has(self, key: str) -> bool:
        return key in self._items

    def add(self, key: str, result: QueryResult) -> None:
        self._items[key] = result
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "result": result.to_dict()}) + "\n")

    def all(self) -> list[QueryResult]:
        return list(self._items.values())
