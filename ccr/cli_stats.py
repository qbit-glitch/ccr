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

    # --- Load commit count, links, triples, patterns ---
    branch = "main"
    commit_count = 0
    summary_len = 0
    link_count = 0
    triple_count = 0
    pattern_count = 0
    bullet_count = 0
    playbook_tokens = 0
    discussion_count = 0
    backend_type = "files"

    db_path = os.path.join(ccr_dir, "memory.db")
    if os.path.isfile(db_path):
        try:
            from ccr.core.storage.sqlite_backend import SqliteStorageBackend
            backend = SqliteStorageBackend(ccr_dir)
            backend_type = "sqlite"
            commit_count = backend.commit_count(branch)
            rs = backend.rolling_summary_get(branch)
            summary_len = len(rs) if rs else 0
            triple_count = backend.triple_count()
            all_links = backend.link_get_all()
            link_count = sum(
                sum(len(entries) for entries in type_map.values())
                for type_map in all_links.values()
            )
            pats = backend.pattern_load_all()
            pattern_count = len(pats.get("patterns", {}))
            bullets = backend.bullet_list(scope="project")
            bullet_count = len(bullets)
            playbook_tokens = max(1, sum(len(b.get("content", "")) for b in bullets) // 4)
            discussion_count = len(backend.discussion_list(branch))
            backend.close()
        except Exception:
            backend_type = "files"

    if backend_type == "files":
        try:
            from ccr.core.memory import MemoryManager
            from ccr.core.types import CCRConfig
            mem = MemoryManager(project, CCRConfig())
            branch = mem.get_active_branch()
            commit_index = mem._build_commit_index(branch)
            commit_count = len(commit_index)
            summary_path = os.path.join(mem.ccr_root, "branches", branch, "commits.md")
            if os.path.isfile(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    content = f.read()
                import re
                m = re.search(r"## Rolling Summary\s*\n(.*?)\n---", content, re.DOTALL)
                if m:
                    summary_len = len(m.group(1).strip())
        except Exception:
            pass
        try:
            playbook_path = os.path.join(ccr_dir, "playbook.txt")
            if os.path.isfile(playbook_path):
                with open(playbook_path, "r", encoding="utf-8") as f:
                    playbook_text = f.read()
                playbook_tokens = max(1, len(playbook_text) // 4)
                bullet_count = playbook_text.count("] helpful=")
        except Exception:
            pass

    click.echo(f"\n=== CCR Stats Dashboard === [backend: {backend_type}]\n")

    click.echo(f"Project memory ({branch})")
    click.echo(f"  Commits:          {commit_count}")
    if summary_len:
        pct = summary_len * 100 // 1500
        click.echo(f"  Rolling summary:  {summary_len}/1500 chars ({pct}%)")
        if summary_len >= 1350:
            click.echo("  [!!] Summary near capacity — call gcc_consolidate() in your next session")
        elif summary_len >= 1200:
            click.echo("  [--] Summary at 80%+ — consider calling gcc_consolidate() soon")
    if bullet_count:
        click.echo(f"  Playbook:         {playbook_tokens:,} tokens ({bullet_count} bullets)")
    if triple_count:
        click.echo(f"  Triples:          {triple_count}")
    if link_count:
        click.echo(f"  Links:            {link_count}")
    if pattern_count:
        click.echo(f"  Patterns:         {pattern_count}")
    if discussion_count:
        click.echo(f"  Discussions:      {discussion_count}")

    click.echo()

    if not sessions:
        click.echo("Session history")
        # Distinguish: no sessions.jsonl at all vs file exists but empty/unqualified
        def _has_ccr_hooks(path: str) -> bool:
            if not os.path.isfile(path):
                return False
            try:
                import json as _json2
                with open(path, encoding="utf-8") as _f:
                    _s = _json2.load(_f)
                hooks = _s.get("hooks", {})
                return "Stop" in hooks or "UserPromptSubmit" in hooks
            except Exception:
                return False

        hooks_dir = os.path.join(project, ".claude", "settings.local.json")
        global_hooks = os.path.join(os.path.expanduser("~"), ".claude", "settings.local.json")
        hooks_installed = _has_ccr_hooks(hooks_dir) or _has_ccr_hooks(global_hooks)
        if not os.path.isfile(jsonl_path):
            if hooks_installed:
                click.echo("  Hooks are active — stats will appear after your first session ends.")
                click.echo("  Open Claude Code in this project and run at least one prompt.")
            else:
                click.echo("  No session history yet.")
                click.echo("  Run 'ccr install' to enable automatic session tracking.")
        else:
            click.echo("  Sessions logged but no context-injection records found.")
            click.echo("  This usually means sessions started before 'ccr install' was run.")
            click.echo("  Stats will populate from the next session onward.")
        return

    # --- Compute aggregates ---
    token_list = [s["context_tokens"] for s in sessions]
    dur_list = [s.get("duration_min", 0) for s in sessions]
    overhead_list = [s.get("reminder_overhead_tokens", 0) for s in sessions]
    total_sessions = len(sessions)
    avg_tokens = sum(token_list) / total_sessions
    total_tokens = sum(token_list)
    gross_avoided = total_tokens * multiplier
    total_overhead = sum(overhead_list)
    net_saved = max(0, gross_avoided - total_overhead)
    avg_dur = sum(dur_list) / total_sessions if dur_list else 0
    # breakeven removed — replaced with honest token summary below
    has_overhead_data = any(s.get("reminder_overhead_tokens", 0) > 0 for s in sessions)

    click.echo(f"Session history (last {total_sessions} sessions)")
    click.echo(f"  Est. context injected:  {total_tokens:,.0f} tokens "
               f"({avg_tokens:,.0f}/session avg, {total_sessions} sessions)")
    click.echo(f"  Gross savings (estimated):  {gross_avoided:,.0f} tokens (context injected \u00d7{multiplier:.0f} re-typing heuristic)")
    if total_overhead > 0:
        click.echo(f"  Overhead:              {total_overhead:,.0f} tokens (per-turn reminders)")
        click.echo(f"  Net savings:           {net_saved:,.0f} tokens")
    else:
        # Older session records don't have reminder_overhead_tokens — note it
        click.echo(f"  Net savings:           {net_saved:,.0f} tokens (overhead: n/a for older records)")
    if avg_dur:
        click.echo(f"  Avg session duration:  {avg_dur:.0f} min")

    click.echo()
    click.echo("30-session projection")
    avg_overhead_per_session = total_overhead / total_sessions if total_sessions else 0
    projected_gross = avg_tokens * 30 * multiplier
    projected_overhead = avg_overhead_per_session * 30
    projected_net = max(0, projected_gross - projected_overhead)
    click.echo(f"  Est. gross savings:  ~{projected_gross:,.0f} tokens")
    if total_overhead > 0:
        click.echo(f"  Est. overhead:       ~{projected_overhead:,.0f} tokens")
        click.echo(f"  Est. net savings:    ~{projected_net:,.0f} tokens")
    click.echo(f"  Note: 'gross savings' = tokens injected × {multiplier:.0f} re-typing heuristic "
               f"(assumes you'd manually re-explain that much context). Use --multiplier 1 "
               f"to see injected tokens only.")

    click.echo()
    # Show overhead column only when at least one session has reminder data
    if has_overhead_data:
        click.echo("Recent sessions")
        click.echo(f"  {'Date':<12} {'Context':>8}  {'Overhead':>9}  {'Net saved':>10}  {'Duration':>10}")
        click.echo(f"  {'-'*12} {'-'*8}  {'-'*9}  {'-'*10}  {'-'*10}")
        for s in reversed(sessions[-10:]):
            try:
                dt = datetime.fromisoformat(s["start"]).strftime("%Y-%m-%d")
            except Exception:
                dt = "unknown"
            ctx = s["context_tokens"]
            ovhd = s.get("reminder_overhead_tokens", 0)
            net = max(0, int(ctx * multiplier) - ovhd)
            dur = s.get("duration_min", 0)
            dur_str = f"{dur:.0f} min" if dur else "  --  "
            click.echo(f"  {dt:<12} {ctx:>8,}  {ovhd:>9,}  {net:>10,}  {dur_str:>10}")
    else:
        click.echo("Recent sessions")
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
        "* Est. context injected = hook output length \u00f7 4 chars/token (heuristic estimate)."
    )
    click.echo(
        f"  Gross savings = context injected \u00d7{multiplier:.0f} (speculative: assumes you'd "
        f"re-type that much context without CCR — use --multiplier 1 to see measured only)."
    )
    click.echo(
        "  Net savings = gross savings \u2212 per-turn reminder overhead (~32 tokens/turn on tool-use turns; 0 on Q&A-only turns)."
    )
    click.echo(
        "  Pass --multiplier N to adjust (--multiplier 1 = measured only, no speculation)."
    )
