"""Dataset adapters. Each registers a loader (and optionally a sampler or gains)."""

from __future__ import annotations

from collections.abc import Callable

from ccr.bench.sampling import sample_worlds
from ccr.bench.types import BenchConfig, World

Adapter = Callable[[BenchConfig], list[World]]
Sampler = Callable[[list[World], BenchConfig], list[World]]
GainsFor = Callable[[str], dict[str, dict[str, int]]]

ADAPTERS: dict[str, Adapter] = {}
SAMPLERS: dict[str, Sampler] = {}
GAINS: dict[str, GainsFor] = {}
# Datasets with no evidence labels: only answer metrics exist, so a reader is required.
NEEDS_READER: set[str] = set()


def default_sampler(worlds: list[World], cfg: BenchConfig) -> list[World]:
    """Sample whole worlds with the config's seed and fraction."""
    return sample_worlds(worlds, cfg.sample_fraction, cfg.seed)


# Importing each adapter registers it in ADAPTERS (and SAMPLERS/GAINS if needed).
from ccr.bench.datasets import locomo as _locomo  # noqa: E402,F401
from ccr.bench.datasets import longmemeval as _longmemeval  # noqa: E402,F401
from ccr.bench.datasets import beir as _beir  # noqa: E402,F401
from ccr.bench.datasets import mab as _mab  # noqa: E402,F401

__all__ = ["ADAPTERS", "GAINS", "NEEDS_READER", "SAMPLERS", "Adapter", "GainsFor", "Sampler", "default_sampler"]
