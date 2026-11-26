"""`ccr plan` — shell entry points for a harness that drives the closed model via tmux.

Typical loop: `ccr plan escalate` (exit 3 = cap hit) -> launch tmux session with the
packet -> session ends -> `ccr plan accept <file|->` (accepts raw pane text too).
"""
from __future__ import annotations

import os
import sys

import click

from ccr.core.plan import PlanStore, parse_plan, parse_usage


def _store(project: str) -> PlanStore:
    return PlanStore(os.path.join(os.path.abspath(project), ".ccr"))


@click.group()
@click.option("--project", default=".", help="Project root containing .ccr/")
@click.pass_context
def plan(ctx: click.Context, project: str) -> None:
    """Plan store for the local-agent / closed-planner loop."""
    ctx.obj = _store(project)


@plan.command()
@click.pass_obj
def brief(st: PlanStore) -> None:
    """Print the you-are-here brief."""
    click.echo(st.brief())


@plan.command()
@click.pass_obj
def escalate(st: PlanStore) -> None:
    """Count one closed-model session and print the planner packet (exit 3 if cap hit)."""
    try:
        click.echo(st.escalate())
    except PermissionError as e:
        click.echo(str(e), err=True)
        sys.exit(3)


@plan.command()
@click.argument("source", default="-")
@click.option("--parent", default="", help="Task id being re-planned")
@click.pass_obj
def accept(st: PlanStore, source: str, parent: str) -> None:
    """Store the planner's plan from a file or stdin ('-'); raw pane text is fine."""
    text = sys.stdin.read() if source == "-" else open(source, encoding="utf-8").read()
    try:
        ids = st.accept_plan(parse_plan(text), parent)
    except (ValueError, KeyError) as e:
        click.echo(f"rejected: {e}", err=True)
        sys.exit(2)
    click.echo(f"accepted {len(ids)} tasks: {' '.join(ids)}")


@plan.command()
@click.argument("task_id")
@click.argument("outcome", type=click.Choice(["pass", "fail", "note"]))
@click.option("--detail", default="")
@click.option("--metric", type=float, default=None)
@click.option("--tokens", type=int, default=0)
@click.pass_obj
def log(st: PlanStore, task_id: str, outcome: str, detail: str, metric: float | None, tokens: int) -> None:
    """Journal a test result; the harness knows pass/fail and can report tokens."""
    st.log(task_id, outcome, detail, metric, tokens)
    click.echo("stuck" if st.stuck() else "ok")


@plan.command()
@click.argument("task_id")
@click.option("--timeout", default=600, help="Seconds before the acceptance command is failed")
@click.pass_obj
def verify(st: PlanStore, task_id: str, timeout: int) -> None:
    """Run a task's acceptance command, journal the result; exit 0 pass, 1 fail."""
    try:
        t = st.verify(task_id, timeout)
    except (KeyError, ValueError) as e:
        click.echo(str(e), err=True)
        sys.exit(2)
    click.echo(f"{task_id} {t['status']} fails={t['fails']}" + (" stuck" if st.stuck() else ""))
    sys.exit(0 if t["status"] == "done" else 1)


@plan.command()
@click.argument("task_id")
@click.argument("source", default="-")
@click.option("--closed", is_flag=True, help="Count against the closed-model cap (default: local tokens)")
@click.pass_obj
def ingest(st: PlanStore, task_id: str, source: str, closed: bool) -> None:
    """Record token usage from an `opencode run --format json` log (file or '-')."""
    text = sys.stdin.read() if source == "-" else open(source, encoding="utf-8").read()
    n = parse_usage(text)
    st.log(task_id, "note", f"agent run: {n} tokens", tokens=n, local=not closed)
    click.echo(n)


@plan.command()
@click.option("--max-escalations", type=int, default=None)
@click.option("--max-tokens", type=int, default=None)
@click.pass_obj
def budget(st: PlanStore, max_escalations: int | None, max_tokens: int | None) -> None:
    """Show (and optionally set) the closed-model caps."""
    if max_escalations is not None or max_tokens is not None:
        st.set_budget(max_tokens, max_escalations)
    click.echo(st.budget())
