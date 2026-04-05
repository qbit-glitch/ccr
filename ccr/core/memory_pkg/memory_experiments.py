"""ExperimentsMixin — query and filter experiment records stored in commits.

Parses **Experiment**: blocks (written by gcc_commit with experiment= param)
from the commits file and exposes filter/compare/rank operations.

Experiment block format (written by CommitMixin):
    **Experiment**:
      - ID: exp-042
      - Hypothesis: LoRA r=16 matches full fine-tuning
      - Metrics: val_loss=0.23, accuracy=0.87
      - Conclusion: Confirmed — 98% perf at 12% params

All methods are read-only and operate on the existing commits.md file.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for parsing experiment blocks from commits
# ---------------------------------------------------------------------------

# Commit header: ## [C042] 2026-04-05 10:30 | branch:main | title
_COMMIT_HEADER = re.compile(
    r"^## \[(C\d{3,})\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}).*?$",
    re.MULTILINE,
)

# Experiment field extractors (tolerant of optional leading whitespace)
_EXP_BLOCK_START = re.compile(r"\*\*Experiment\*\*:")
_EXP_ID = re.compile(r"[-•]\s*ID:\s*(.+)")
_EXP_HYPOTHESIS = re.compile(r"[-•]\s*Hypothesis:\s*(.+)")
_EXP_METRICS = re.compile(r"[-•]\s*Metrics:\s*(.+)")
_EXP_CONCLUSION = re.compile(r"[-•]\s*Conclusion:\s*(.+)")

# A metric value like "val_loss=0.23"
_METRIC_KV = re.compile(r"(\w+)=([^\s,]+)")


def _parse_metric_value(raw: str) -> float | str:
    """Try to parse a metric value as float; return string on failure."""
    try:
        return float(raw)
    except (ValueError, TypeError):
        return raw


def _parse_experiment_block(block_text: str) -> dict[str, Any]:
    """Extract fields from the text of one **Experiment**: block."""
    result: dict[str, Any] = {}
    m = _EXP_ID.search(block_text)
    if m:
        result["id"] = m.group(1).strip()
    m = _EXP_HYPOTHESIS.search(block_text)
    if m:
        result["hypothesis"] = m.group(1).strip()
    m = _EXP_METRICS.search(block_text)
    if m:
        metrics_str = m.group(1).strip()
        metrics: dict[str, float | str] = {}
        for km in _METRIC_KV.finditer(metrics_str):
            metrics[km.group(1)] = _parse_metric_value(km.group(2))
        result["metrics"] = metrics
    m = _EXP_CONCLUSION.search(block_text)
    if m:
        result["conclusion"] = m.group(1).strip()
    return result


def _parse_commit_blocks(commits_text: str) -> list[dict[str, Any]]:
    """Parse commits.md text and return a list of commit dicts with experiment data.

    Only commits that contain a **Experiment**: block are returned.
    """
    # Split on commit headers
    parts = _COMMIT_HEADER.split(commits_text)
    # parts: [pre, id1, date1, body1, id2, date2, body2, ...]
    results = []
    idx = 1  # skip pre-header text
    while idx + 2 < len(parts):
        commit_id = parts[idx].strip()
        commit_date = parts[idx + 1].strip()
        body = parts[idx + 2]
        idx += 3

        if not _EXP_BLOCK_START.search(body):
            continue  # No experiment block — skip

        # Extract title from header (first line of body before \n or **What**)
        title_match = re.search(r"\*\*What\*\*:\s*(.+)", body)
        title = title_match.group(1).strip()[:80] if title_match else commit_id

        # Slice out just the experiment block (up to next **-field or ---)
        exp_match = _EXP_BLOCK_START.search(body)
        if exp_match:
            exp_start = exp_match.start()
            # Find end: next ** field or --- separator
            end_match = re.search(r"\n\*\*\w|^---", body[exp_start:], re.MULTILINE)
            exp_end = exp_start + end_match.start() if end_match else len(body)
            exp_block = body[exp_start:exp_end]
        else:
            exp_block = body

        exp_data = _parse_experiment_block(exp_block)
        if not exp_data:
            continue

        results.append({
            "commit_id": commit_id,
            "date": commit_date,
            "title": title,
            "experiment": exp_data,
        })

    return results


def _apply_metric_filter(
    records: list[dict],
    metric_filter: dict[str, dict[str, float | int]],
) -> list[dict]:
    """Filter records by metric conditions.

    metric_filter format: {"val_loss": {"lt": 0.3}, "accuracy": {"gte": 0.8}}
    Supported ops: lt, lte, gt, gte, eq.
    """
    def _matches(metrics: dict, key: str, cond: dict) -> bool:
        val = metrics.get(key)
        if val is None:
            return False
        try:
            val = float(val)
        except (TypeError, ValueError):
            return False
        for op, threshold in cond.items():
            threshold = float(threshold)
            if op == "lt" and not (val < threshold):
                return False
            if op == "lte" and not (val <= threshold):
                return False
            if op == "gt" and not (val > threshold):
                return False
            if op == "gte" and not (val >= threshold):
                return False
            if op == "eq" and not (val == threshold):
                return False
        return True

    out = []
    for rec in records:
        metrics = rec["experiment"].get("metrics", {})
        if all(_matches(metrics, k, v) for k, v in metric_filter.items()):
            out.append(rec)
    return out


def _apply_top_n(
    records: list[dict],
    top_n: str,
) -> list[dict]:
    """Sort and slice by 'metric:order'. E.g. 'val_loss:asc' or 'accuracy:desc'."""
    parts = top_n.rsplit(":", 1)
    key = parts[0].strip()
    order = parts[1].strip().lower() if len(parts) == 2 else "asc"
    reverse = order == "desc"

    def _sort_key(rec: dict) -> float:
        val = rec["experiment"].get("metrics", {}).get(key)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("inf") if not reverse else float("-inf")

    return sorted(records, key=_sort_key, reverse=reverse)


def _format_experiment_table(records: list[dict]) -> str:
    """Format a list of experiment records as a markdown table."""
    if not records:
        return "No experiments found."

    lines = [
        "| Commit | Date | ID | Hypothesis | Metrics | Conclusion |",
        "|--------|------|----|------------|---------|------------|",
    ]
    for rec in records:
        exp = rec["experiment"]
        commit_id = rec["commit_id"]
        date = rec["date"]
        eid = exp.get("id", "—")
        hyp = (exp.get("hypothesis") or "—")[:50]
        metrics = ", ".join(
            f"{k}={v}" for k, v in (exp.get("metrics") or {}).items()
        ) or "—"
        conclusion = (exp.get("conclusion") or "—")[:60]
        lines.append(f"| {commit_id} | {date} | {eid} | {hyp} | {metrics} | {conclusion} |")

    return "\n".join(lines)


def _format_comparison_table(records: list[dict]) -> str:
    """Format two experiment records as a side-by-side comparison."""
    if len(records) < 2:
        return _format_experiment_table(records)

    a, b = records[0], records[1]
    exp_a = a["experiment"]
    exp_b = b["experiment"]

    # Collect all metric keys
    all_metrics = sorted(
        set(list(exp_a.get("metrics", {}).keys()) + list(exp_b.get("metrics", {}).keys()))
    )

    lines = [
        f"## Experiment Comparison",
        f"",
        f"| Field | {a['commit_id']} ({exp_a.get('id', '?')}) | {b['commit_id']} ({exp_b.get('id', '?')}) |",
        f"|-------|{'---|' * 2}",
        f"| Date  | {a['date']} | {b['date']} |",
        f"| Hypothesis | {(exp_a.get('hypothesis') or '—')[:50]} | {(exp_b.get('hypothesis') or '—')[:50]} |",
    ]
    for k in all_metrics:
        va = exp_a.get("metrics", {}).get(k, "—")
        vb = exp_b.get("metrics", {}).get(k, "—")
        try:
            diff = float(vb) - float(va)
            diff_str = f" (Δ={diff:+.3f})"
        except (TypeError, ValueError):
            diff_str = ""
        lines.append(f"| {k} | {va} | {vb}{diff_str} |")

    lines += [
        f"| Conclusion | {(exp_a.get('conclusion') or '—')[:60]} | {(exp_b.get('conclusion') or '—')[:60]} |",
    ]
    return "\n".join(lines)


class ExperimentsMixin:
    """Query and filter experiment records from commits.

    Parses **Experiment**: blocks written by gcc_commit(experiment={...}).
    """

    def get_experiments(
        self,
        experiment_id: str | None = None,
        hypothesis_contains: str | None = None,
        metric_filter: dict | None = None,
        date_range: list[str] | None = None,
        top_n: str | None = None,
        compare: list[str] | None = None,
    ) -> dict[str, Any]:
        """Query experiment records from commit history.

        Args:
            experiment_id: Filter to this experiment ID exactly.
            hypothesis_contains: Substring match on hypothesis text.
            metric_filter: Dict of {metric_name: {op: value}} conditions.
                           Supported ops: lt, lte, gt, gte, eq.
                           Example: {"val_loss": {"lt": 0.3}}
            date_range: [start_date, end_date] as ISO-format strings "YYYY-MM-DD".
            top_n: "metric_name:asc|desc" — sort by metric and return all (sorting only).
            compare: List of two commit IDs (e.g. ["C041", "C053"]) for side-by-side.

        Returns:
            dict with keys: count, records (list), message (formatted markdown).
        """
        branch = self.get_active_branch()
        commits_text = self._read_commits_window(branch, 0, 10000)
        all_records = _parse_commit_blocks(commits_text)

        # Filter by experiment_id
        if experiment_id:
            all_records = [
                r for r in all_records
                if r["experiment"].get("id", "").lower() == experiment_id.lower()
            ]

        # Filter by hypothesis substring
        if hypothesis_contains:
            needle = hypothesis_contains.lower()
            all_records = [
                r for r in all_records
                if needle in (r["experiment"].get("hypothesis") or "").lower()
            ]

        # Filter by date range
        if date_range and len(date_range) >= 2:
            try:
                start = datetime.fromisoformat(date_range[0]).replace(tzinfo=timezone.utc)
                end = datetime.fromisoformat(date_range[1]).replace(tzinfo=timezone.utc)
                filtered = []
                for r in all_records:
                    try:
                        dt = datetime.fromisoformat(r["date"].replace(" ", "T") + ":00+00:00")
                        if start <= dt <= end:
                            filtered.append(r)
                    except (ValueError, TypeError):
                        filtered.append(r)  # Include if date unparseable
                all_records = filtered
            except (ValueError, TypeError) as exc:
                logger.warning("date_range parse failed: %s", exc)

        # Filter by metric conditions
        if metric_filter and isinstance(metric_filter, dict):
            all_records = _apply_metric_filter(all_records, metric_filter)

        # Handle compare mode (two specific commits)
        if compare and isinstance(compare, list):
            compare_ids = {cid.upper() for cid in compare}
            compare_records = [r for r in all_records if r["commit_id"].upper() in compare_ids]
            if len(compare_records) >= 2:
                message = _format_comparison_table(compare_records[:2])
            else:
                message = _format_experiment_table(compare_records)
            return {"count": len(compare_records), "records": compare_records, "message": message}

        # Sort by metric
        if top_n:
            all_records = _apply_top_n(all_records, top_n)

        message = _format_experiment_table(all_records)
        return {"count": len(all_records), "records": all_records, "message": message}
