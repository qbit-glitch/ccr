"""Tests for seeded world sampling."""

from __future__ import annotations

import pytest

from ccr.bench.sampling import sample_worlds
from ccr.bench.types import World


def _worlds(n: int, group_of=lambda i: "g") -> list[World]:
    return [World(id=f"w{i}", docs=(), queries=(), group=group_of(i)) for i in range(n)]


def test_ten_percent_of_ten_worlds_is_exactly_one() -> None:
    assert len(sample_worlds(_worlds(10), 0.1, seed=1)) == 1


def test_never_returns_zero_for_small_fraction() -> None:
    assert len(sample_worlds(_worlds(3), 0.01, seed=1)) == 1


def test_same_seed_same_ids_different_seed_can_differ() -> None:
    a = [w.id for w in sample_worlds(_worlds(50), 0.2, seed=7)]
    b = [w.id for w in sample_worlds(_worlds(50), 0.2, seed=7)]
    c = [w.id for w in sample_worlds(_worlds(50), 0.2, seed=8)]
    assert a == b
    assert a != c


def test_full_fraction_returns_everything_in_order() -> None:
    worlds = _worlds(5)
    assert sample_worlds(worlds, 1.0, seed=3) == worlds


def test_stratified_sample_covers_every_stratum() -> None:
    worlds = _worlds(40, group_of=lambda i: "rare" if i == 0 else "common")
    picked = sample_worlds(worlds, 0.1, seed=5, stratify_by=lambda w: w.group)
    assert {w.group for w in picked} == {"rare", "common"}


def test_pilot_sample_is_subset_of_larger_sample_with_same_seed() -> None:
    small = {w.id for w in sample_worlds(_worlds(100), 0.1, seed=11)}
    large = {w.id for w in sample_worlds(_worlds(100), 0.3, seed=11)}
    assert small <= large


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_fraction_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        sample_worlds(_worlds(3), bad, seed=1)
