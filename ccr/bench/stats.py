"""Bootstrap confidence intervals, paired comparisons and a ranking table.

Pure Python and seeded, so the same results file always yields the same table.
Queries are resampled independently; queries from one conversation or document
set are not independent, so intervals are a little optimistic on LoCoMo.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from ccr.bench.results import QueryResult

Key = tuple[str, str]
DEFAULT_BOOT = 2000
ALPHA = 0.05


def metric_values(rows: Sequence[QueryResult], system: str, metric: str) -> dict[Key, float]:
    """Per-query values of ``metric`` for ``system``, skipping excluded and not-run rows."""
    out: dict[Key, float] = {}
    for row in rows:
        if row.system != system or "excluded" in row.scores or "not_run" in row.scores:
            continue
        if metric in row.scores:
            out[(row.world_id, row.query_id)] = row.scores[metric]
    return out


def _boot_means(values: Sequence[float], n_boot: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    return sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    index = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values))))
    return sorted_values[index]


def bootstrap_ci(
    values: Sequence[float], n_boot: int = DEFAULT_BOOT, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, low, high) with a 95% percentile bootstrap interval."""
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    boots = _boot_means(list(values), n_boot, seed)
    return mean, _percentile(boots, ALPHA / 2), _percentile(boots, 1 - ALPHA / 2)


def paired_difference(
    a: dict[Key, float], b: dict[Key, float], n_boot: int = DEFAULT_BOOT, seed: int = 0
) -> tuple[int, float, float, float, float]:
    """Mean of a-b over shared queries: (n, diff, low, high, two-sided p)."""
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return 0, 0.0, 0.0, 0.0, 1.0
    diffs = [a[k] - b[k] for k in shared]
    mean = sum(diffs) / len(diffs)
    boots = _boot_means(diffs, n_boot, seed)
    below = sum(1 for d in boots if d <= 0)
    above = sum(1 for d in boots if d >= 0)
    p = min(1.0, 2 * (min(below, above) + 1) / (n_boot + 1))
    return len(diffs), mean, _percentile(boots, ALPHA / 2), _percentile(boots, 1 - ALPHA / 2), p


def _verdict(diff: float, lo: float, p: float) -> str:
    if p < ALPHA and diff > 0 and lo > 0:
        return "significantly below"
    return "tie"


def ranking_markdown(
    rows: Sequence[QueryResult], metrics: Sequence[str],
    n_boot: int = DEFAULT_BOOT, seed: int = 0,
) -> str:
    """One ranked table per metric, with 95% CIs and a paired test against the leader."""
    systems = sorted({r.system for r in rows})
    lines: list[str] = []
    for metric in metrics:
        per_system = {s: metric_values(rows, s, metric) for s in systems}
        per_system = {s: v for s, v in per_system.items() if v}
        if not per_system:
            continue
        stats = {s: bootstrap_ci(list(v.values()), n_boot, seed) for s, v in per_system.items()}
        order = sorted(per_system, key=lambda s: stats[s][0], reverse=True)
        leader = order[0]
        lines += [f"### {metric}", "",
                  "| rank | system | n | mean | 95% CI | vs leader |", "|---|---|---|---|---|---|"]
        for rank, system in enumerate(order, 1):
            mean, lo, hi = stats[system]
            if system == leader:
                verdict = "leader"
            else:
                _, diff, dlo, _, p = paired_difference(
                    per_system[leader], per_system[system], n_boot, seed)
                verdict = f"{_verdict(diff, dlo, p)} (p={p:.3f})"
            lines.append(f"| {rank} | {system} | {len(per_system[system])} | {mean:.3f} | "
                         f"[{lo:.3f}, {hi:.3f}] | {verdict} |")
        lines.append("")
    return "\n".join(lines)


def metrics_present(rows: Sequence[QueryResult]) -> list[str]:
    """Headline metrics found in the results, answer metrics first when a reader ran."""
    seen: dict[str, int] = defaultdict(int)
    for row in rows:
        for key in row.scores:
            seen[key] += 1
    wanted = ("f1", "em", "sub_em", "ndcg@10", "recall@10", "recall@5", "mrr")
    return [m for m in wanted if seen.get(m)]
