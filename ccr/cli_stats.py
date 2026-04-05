"""CCR stats command — extracted from cli.py to satisfy 800-line limit."""
from __future__ import annotations

import os
import sys

import click


@click.command()
@click.argument("project", default=".")
@click.option("--multiplier", "-m", default=4.0, type=float,
              help="Token re-typing multiplier (default 4). Use 2 for conservative, 6 for optimistic.")
@click.option("--last", default=30, type=int,
              help="Number of recent sessions to include in stats (default 30).")
def stats(project: str, multiplier: float, last: int) -> None:
    """Show CCR ROI dashboard: token savings, memory health, session history."""
    import json as _json
    from datetime import datetime

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    project = os.path.abspath(project)
    ccr_dir = os.path.join(project, ".ccr")

    if not os.path.isdir(ccr_dir):
        click.echo(f"No .ccr/ directory found in {project}. Run 'ccr init' first.", err=True)
        return

    # --- Load session records ---
    jsonl_path = os.path.join(ccr_dir, "metrics", "sessions.jsonl")
    sessions = []
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                    if rec.get("context_tokens", 0) > 0:
                        sessions.append(rec)
                except _json.JSONDecodeError:
                    continue  # Skip corrupt lines
    sessions = sessions[-last:]  # Most recent N sessions

    # --- Load commit count from GCC ---
    try:
        from ccr.core.memory import MemoryManager
        from ccr.core.types import CCRConfig
        mem = MemoryManager(project, CCRConfig())
        branch = mem.get_active_branch()
        commit_index = mem._build_commit_index(branch)
        commit_count = len(commit_index)
        summary_path = os.path.join(mem.ccr_root, "branches", branch, "commits.md")
        summary_len = 0
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                content = f.read()
            import re
            m = re.search(r"## Rolling Summary\s*\n(.*?)\n---", content, re.DOTALL)
            if m:
                summary_len = len(m.group(1).strip())
    except Exception:
        commit_count = 0
        branch = "main"
        summary_len = 0

    # --- Load playbook ---
    try:
        playbook_path = os.path.join(ccr_dir, "playbook.txt")
        playbook_text = ""
        if os.path.isfile(playbook_path):
            with open(playbook_path, "r", encoding="utf-8") as f:
                playbook_text = f.read()
        playbook_tokens = max(1, len(playbook_text) // 4)
        bullet_count = playbook_text.count("] helpful=")
    except Exception:
        playbook_tokens = 0
        bullet_count = 0

    click.echo("\n=== CCR Stats Dashboard ===\n")

    click.echo(f"Project memory ({branch})")
    click.echo(f"  Commits:          {commit_count}")
    if summary_len:
        click.echo(f"  Rolling summary:  {summary_len}/1500 chars ({summary_len * 100 // 1500}%)")
        if summary_len > 1200:
            click.echo(f"  Summary is getting long — run gcc_consolidate() or gcc_context(level=5)")
    if bullet_count:
        click.echo(f"  Playbook:         {playbook_tokens:,} tokens ({bullet_count} bullets)")

    click.echo()

    if not sessions:
        click.echo("Session history")
        click.echo("  No session history yet.")
        click.echo("  (CCR stats are recorded after each session ends with 'ccr install' hooks active.)")
        return

    # --- Compute aggregates ---
    token_list = [s["context_tokens"] for s in sessions]
    dur_list = [s.get("duration_min", 0) for s in sessions]
    total_sessions = len(sessions)
    avg_tokens = sum(token_list) / total_sessions
    total_tokens = sum(token_list)
    avoided = total_tokens * multiplier
    avg_dur = sum(dur_list) / total_sessions if dur_list else 0
    breakeven = int(1 / multiplier * 10) + 1 if multiplier else 6

    click.echo(f"Session history (last {total_sessions} sessions)")
    click.echo(f"  Avg context injected:  {avg_tokens:,.0f} tokens/session")
    click.echo(f"  Total injected:        {total_tokens:,.0f} tokens")
    click.echo(f"  Est. tokens avoided:   {avoided:,.0f} tokens (\u00d7{multiplier:.0f} re-typing heuristic*)")
    if avg_dur:
        click.echo(f"  Avg session duration:  {avg_dur:.0f} min")

    click.echo()
    click.echo(f"30-session projection")
    projected_avoided = avg_tokens * 30 * multiplier
    click.echo(f"  Est. savings:  ~{projected_avoided:,.0f} tokens")
    if total_sessions >= breakeven:
        click.echo(f"  Break-even:    {breakeven} sessions (already past)")
    else:
        remaining = breakeven - total_sessions
        click.echo(f"  Break-even:    {breakeven} sessions ({remaining} more to go)")

    click.echo()
    click.echo(f"Recent sessions")
    click.echo(f"  {'Date':<12} {'Context':>8}  {'Avoided':>9}  {'Duration':>10}")
    click.echo(f"  {'-'*12} {'-'*8}  {'-'*9}  {'-'*10}")
    for s in reversed(sessions[-10:]):
        try:
            dt = datetime.fromisoformat(s["start"]).strftime("%Y-%m-%d")
        except Exception:
            dt = "unknown"
        ctx = s["context_tokens"]
        avd = int(ctx * multiplier)
        dur = s.get("duration_min", 0)
        dur_str = f"{dur:.0f} min" if dur else "  --  "
        click.echo(f"  {dt:<12} {ctx:>8,}  {avd:>9,}  {dur_str:>10}")

    click.echo()
    click.echo(
        f"* Rough estimate: assumes you'd re-type ~{multiplier:.0f}\u00d7 the context without CCR."
    )
    click.echo(
        f"  Pass --multiplier N to adjust (e.g., --multiplier 2 for conservative estimate)."
    )
