"""Paired bootstrap of each system against a base system in one bench output directory.

Usage: python scripts/bench_paired.py DIR BASE [SYSTEM ...] [--metric ndcg@10]
"""

from __future__ import annotations

import argparse
import os

from ccr.bench.results import ResultStore
from ccr.bench.stats import metric_values, paired_difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("base")
    parser.add_argument("systems", nargs="*")
    parser.add_argument("--metric", default="ndcg@10")
    args = parser.parse_args()
    rows = ResultStore(os.path.join(args.directory, "results.jsonl")).all()
    base = metric_values(rows, args.base, args.metric)
    names = args.systems or sorted({r.system for r in rows} - {args.base})
    print(f"{args.metric} vs {args.base} ({args.directory})")
    for name in names:
        n, diff, low, high, p = paired_difference(metric_values(rows, name, args.metric), base)
        print(f"  {name:<22} n={n:<4} diff={diff:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]  p={p:.3f}")


if __name__ == "__main__":
    main()
