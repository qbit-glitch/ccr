"""Retriever protocol and registry shared by every benchmarked system."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from ccr.bench.types import Doc


class RetrieverUnavailable(Exception):
    """Raised when an optional dependency for a system is missing."""


class Retriever(Protocol):
    """One benchmarked system: index a corpus, then rank doc ids for a query."""

    name: str
    max_k: int | None
    empty_count: int

    def index(self, docs: Sequence[Doc], workdir: str) -> None:
        """Build whatever the system needs; ``workdir`` is an isolated temp dir."""

    def retrieve(self, query: str, k: int) -> list[str]:
        """Return ranked doc ids (possibly fewer than ``k``)."""

    def close(self) -> None:
        """Release resources. Always called by the runner."""


REGISTRY: dict[str, Callable[[], Retriever]] = {}

_T = TypeVar("_T")


def register(name: str) -> Callable[[type[_T]], type[_T]]:
    """Class decorator adding a zero-argument factory to ``REGISTRY``."""

    def decorator(cls: type[_T]) -> type[_T]:
        REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return decorator
