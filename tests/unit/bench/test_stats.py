"""Tests for bootstrap confidence intervals, paired comparison and the ranking table."""

from __future__ import annotations

import pytest

from ccr.bench.results import QueryResult
from ccr.bench.stats import (
    bootstrap_ci,
    metric_values,
    paired_difference,
    ranking_markdown,
)


def _row(system: str, qid: str, scores: dict[str, float], world: str = "w") -> QueryResult:
    return QueryResult(qid, world, system, "cat", (), 10, None, scores, False, 0.0, 0.0)


def test_bootstrap_ci_brackets_the_mean_and_is_seeded() -> None:
    values = [0.0, 1.0] * 50
    mean, lo, hi = bootstrap_ci(values, n_boot=500, seed=3)
    assert mean == pytest.approx(0.5)
    assert lo < 0.5 < hi and 0.35 < lo and hi < 0.65
    assert bootstrap_ci(values, n_boot=500, seed=3) == (mean, lo, hi)


def test_bootstrap_ci_of_constant_values_has_zero_width() -> None:
    # approx: numpy sums differ in the last ulp between macOS ARM and Linux
    assert bootstrap_ci([0.7] * 20, n_boot=200, seed=1) == pytest.approx((0.7, 0.7, 0.7))


def test_bootstrap_ci_of_empty_is_nan_free_zero() -> None:
    assert bootstrap_ci([], n_boot=10, seed=1) == (0.0, 0.0, 0.0)


def test_metric_values_skip_excluded_not_run_and_missing_metric() -> None:
    rows = [
        _row("s", "1", {"f1": 0.5}),
        _row("s", "2", {"excluded": 1.0}),
        _row("s", "3", {"not_run": 1.0}),
        _row("s", "4", {"ndcg@10": 0.9}),
        _row("t", "1", {"f1": 1.0}),
    ]
    assert metric_values(rows, "s", "f1") == {("w", "1"): 0.5}


def test_paired_difference_detects_a_consistent_gap() -> None:
    a = {("w", str(i)): 0.8 for i in range(40)}
    b = {("w", str(i)): 0.2 for i in range(40)}
    n, diff, lo, hi, p = paired_difference(a, b, n_boot=500, seed=1)
    assert n == 40 and diff == pytest.approx(0.6) and lo > 0 and p < 0.01


def test_paired_difference_of_identical_systems_is_not_significant() -> None:
    a = {("w", str(i)): float(i % 2) for i in range(40)}
    n, diff, lo, hi, p = paired_difference(a, dict(a), n_boot=300, seed=1)
    assert diff == 0.0 and lo <= 0.0 <= hi and p == 1.0


def test_paired_difference_uses_only_shared_queries() -> None:
    a = {("w", "1"): 1.0, ("w", "2"): 1.0}
    b = {("w", "2"): 0.0, ("w", "3"): 0.0}
    n, diff, *_ = paired_difference(a, b, n_boot=50, seed=1)
    assert n == 1 and diff == 1.0


def test_ranking_orders_best_first_and_marks_ties_with_the_leader() -> None:
    rows = []
    for i in range(60):
        rows.append(_row("best", str(i), {"f1": 0.9 if i % 10 else 0.0}))
        rows.append(_row("twin", str(i), {"f1": 0.9 if i % 10 else 0.0}))
        rows.append(_row("worst", str(i), {"f1": 0.1}))
    text = ranking_markdown(rows, ["f1"], n_boot=300, seed=1)
    lines = [ln for ln in text.splitlines() if ln.startswith("| ")]
    body = [ln for ln in lines if "worst" in ln or "best" in ln or "twin" in ln]
    assert "worst" in body[-1]
    assert "tie" in text and "significantly below" in text
    assert text.index("best") < text.index("worst")


def test_ranking_reports_the_number_of_scored_queries() -> None:
    rows = [_row("a", str(i), {"ndcg@10": 0.5}) for i in range(7)]
    assert "| a | 7 |" in ranking_markdown(rows, ["ndcg@10"], n_boot=20, seed=1)
