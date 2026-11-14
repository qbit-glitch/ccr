"""Seeded, whole-world sampling so pilots are repeatable subsets of full runs."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from ccr.bench.types import World


def _take(worlds: list[World], fraction: float, rng: random.Random) -> set[str]:
    ordered = sorted(worlds, key=lambda w: w.id)
    rng.shuffle(ordered)
    count = max(1, round(len(ordered) * fraction))
    return {w.id for w in ordered[:count]}


def sample_worlds(
    worlds: Sequence[World],
    fraction: float,
    seed: int,
    stratify_by: Callable[[World], str] | None = None,
) -> list[World]:
    """Pick ``fraction`` of whole worlds, at least one overall or per stratum.

    The same seed always yields the same ids, and a smaller fraction yields a
    subset of a larger one, so a 10% pilot is contained in the full run.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0 or not worlds:
        return list(worlds)
    chosen: set[str] = set()
    if stratify_by is None:
        chosen = _take(list(worlds), fraction, random.Random(seed))
    else:
        strata: dict[str, list[World]] = {}
        for world in worlds:
            strata.setdefault(stratify_by(world), []).append(world)
        for key in sorted(strata):
            chosen |= _take(strata[key], fraction, random.Random(f"{seed}:{key}"))
    return [w for w in worlds if w.id in chosen]
