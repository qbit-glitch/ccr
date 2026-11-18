"""Plan tools — hierarchical plan, journal, stuck detection, planner packet, budget."""

from __future__ import annotations

import os

from mcp.types import ToolAnnotations

from ccr.core.plan import PlanStore
from ccr.mcp.server import mcp
import ccr.mcp.server as _srv


def _store() -> PlanStore:
    return PlanStore(os.path.join(_srv._project_root, ".ccr"))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def plan_update(
    goal: str = "",
    tasks: list[dict] | None = None,
    parent_id: str = "",
    from_planner: bool = False,
    max_tokens: int | None = None,
    max_escalations: int | None = None,
) -> dict:
    """Set the goal, add tasks, or store a planner's returned plan.

    tasks: [{"title", "acceptance" (e.g. "tests/test_x.py passes"), "children": [...]}].
    from_planner=True, with parent_id, replaces that task's open sub-tasks.
    max_* set the closed-model session/token caps (0 = unlimited). Escalations are
    counted by `ccr plan escalate`, not here.
    """
    with _srv._state_lock:
        st = _store()
        if goal:
            st.set_goal(goal)
        ids = (st.accept_plan if from_planner else st.add_tasks)(tasks, parent_id) if tasks else []
        if max_tokens is not None or max_escalations is not None:
            st.set_budget(max_tokens, max_escalations)
    return {"added": ids, "brief": st.brief()}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def plan_log(task_id: str, outcome: str, detail: str = "", metric: float | None = None, tokens: int = 0, local: bool = False) -> dict:
    """Append to the progress journal. outcome: pass (marks task done), fail, or note.

    metric: optional score (e.g. tests passing) used for no-improvement detection.
    tokens: tokens spent. local=False counts against the closed-model cap; local=True is tracked separately.
    """
    with _srv._state_lock:
        st = _store()
        task = st.log(task_id, outcome, detail, metric, tokens, local)
    return {"task": task, "stuck": bool(st.stuck())}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def plan_view(view: str = "brief") -> dict:
    """view: brief (you-are-here + next), packet (planner brief), stuck, budget, tree, journal."""
    st = _store()
    views = {"brief": st.brief, "packet": st.packet, "stuck": st.stuck,
             "budget": st.budget, "tree": st.tree, "journal": st.journal}
    if view not in views:
        raise ValueError(f"view must be one of {sorted(views)}")
    return {"view": view, "result": views[view]()}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def plan_escalate() -> dict:
    """Count one closed-model session and return the planner packet.

    Call right before the harness launches the closed model (same as `ccr plan escalate`).
    Returns allowed=False with no packet when the budget cap is hit; stay on the local model.
    """
    with _srv._state_lock:
        st = _store()
        try:
            return {"allowed": True, "packet": st.escalate()}
        except PermissionError as e:
            return {"allowed": False, "reason": str(e)}
