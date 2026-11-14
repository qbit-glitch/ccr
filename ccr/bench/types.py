"""Frozen data types shared by every benchmark adapter, retriever and runner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    """One retrievable unit (a turn, session, document or code snippet)."""

    id: str
    text: str


@dataclass(frozen=True)
class Query:
    """One question with its gold evidence ids and optional reference answer."""

    id: str
    text: str
    gold_ids: frozenset[str]
    answer: str | None
    category: str


@dataclass(frozen=True)
class World:
    """A self-contained corpus plus the queries asked against it."""

    id: str
    docs: tuple[Doc, ...]
    queries: tuple[Query, ...]
    group: str


@dataclass(frozen=True)
class BenchConfig:
    """Run-wide settings. Nothing in the harness hardcodes these values."""

    ks: tuple[int, ...]
    seed: int
    sample_fraction: float
    chunk_tokens: int
    cache_dir: str
