"""PlanStore: hierarchical plan, append-only journal, stuck detection, planner packet, budget.

Pure logic, no model calls. State is `.ccr/plan.json` (atomic replace); the journal is
`.ccr/plan_journal.jsonl` (append-only). shortcut: whole-file JSON rewrite, fine for
plans under a few thousand tasks; move to SQLite if that is exceeded.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ERRLINE = re.compile(r"^(E\s+|.*\b(Error|Exception|FAILED)\b)")
OPEN = ("todo", "in_progress", "blocked")
OUTCOMES = ("pass", "fail", "note")


def _summarize(out: str) -> str:
    """One-line failure reason: first error-looking line of the output, else its last line."""
    lines = [ln.strip() for ln in _ANSI.sub("", out).splitlines() if ln.strip()]
    pick = next((ln for ln in lines if _ERRLINE.match(ln)), lines[-1] if lines else "")
    return pick[:160]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PlanStore:
    def __init__(self, ccr_dir: str, stuck_after: int = 3):
        self._state_path = os.path.join(ccr_dir, "plan.json")
        self._journal_path = os.path.join(ccr_dir, "plan_journal.jsonl")
        self.stuck_after = stuck_after

    # -- persistence -------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            with open(self._state_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"goal": "", "tasks": {}, "seq": 0,
                    "budget": {"tokens": 0, "escalations": 0,
                               "max_tokens": 0, "max_escalations": 0}}

    def _save(self, s: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        tmp = self._state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f)
        os.replace(tmp, self._state_path)

    def _journal(self, **rec: Any) -> None:
        os.makedirs(os.path.dirname(self._journal_path), exist_ok=True)
        with open(self._journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), **rec}) + "\n")

    def journal(self, task_id: str | None = None) -> list[dict[str, Any]]:
        try:
            with open(self._journal_path, encoding="utf-8") as f:
                recs = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []
        return [r for r in recs if task_id is None or r.get("task") == task_id]

    # -- plan store --------------------------------------------------------
    def set_goal(self, goal: str) -> None:
        s = self._load()
        s["goal"] = goal
        self._save(s)

    def _add(self, s: dict[str, Any], spec: dict[str, Any], parent: str) -> str:
        s["seq"] += 1
        tid = f"T{s['seq']}"
        s["tasks"][tid] = {
            "id": tid, "title": str(spec["title"]), "parent": parent,
            "acceptance": str(spec.get("acceptance", "")), "status": "todo",
            "fails": 0, "last_error": "",
        }
        for child in spec.get("children", []):
            self._add(s, child, tid)
        return tid

    def add_tasks(self, specs: list[dict[str, Any]], parent: str = "") -> list[str]:
        """Add tasks ({title, acceptance?, children?}) under `parent`; returns new top-level ids."""
        s = self._load()
        if parent and parent not in s["tasks"]:
            raise KeyError(f"unknown parent task {parent}")
        ids = [self._add(s, sp, parent) for sp in specs]
        self._save(s)
        return ids

    def accept_plan(self, specs: list[dict[str, Any]], parent: str = "") -> list[str]:
        """Store a plan returned by the planner (the escalation was counted by escalate()).

        With `parent`, the planner's tasks replace that task's open children (a re-plan);
        done children are kept, and the parent's failure count is reset.
        """
        s = self._load()
        if parent:
            if parent not in s["tasks"]:
                raise KeyError(f"unknown parent task {parent}")
            drop = {t for t, v in s["tasks"].items() if v["parent"] == parent and v["status"] in OPEN}
            for t in list(s["tasks"]):  # also drop descendants of dropped tasks
                p = s["tasks"][t]["parent"]
                while p and p not in drop:
                    p = s["tasks"].get(p, {}).get("parent", "")
                if p in drop:
                    drop.add(t)
            for t in drop:
                del s["tasks"][t]
            s["tasks"][parent].update(fails=0, last_error="", status="todo")
        ids = [self._add(s, sp, parent) for sp in specs]
        self._save(s)
        self._journal(task=parent or None, outcome="note", detail=f"plan accepted: {len(ids)} tasks")
        return ids

    def escalate(self) -> str:
        """Count one closed-model session and return its packet; raise if the cap is hit.

        Call when the harness launches the session, not when the plan returns, so a
        session that dies without a plan still counts.
        """
        if not self.budget()["escalation_allowed"]:
            raise PermissionError("closed-model budget exhausted")
        s = self._load()
        s["budget"]["escalations"] += 1
        self._save(s)
        self._journal(task=None, outcome="note", detail="escalated to planner")
        return self.packet()

    def set_budget(self, max_tokens: int | None = None, max_escalations: int | None = None) -> None:
        s = self._load()
        if max_tokens is not None:
            s["budget"]["max_tokens"] = max_tokens
        if max_escalations is not None:
            s["budget"]["max_escalations"] = max_escalations
        self._save(s)

    # -- journal + status --------------------------------------------------
    def log(self, task_id: str, outcome: str, detail: str = "",
            metric: float | None = None, tokens: int = 0, local: bool = False) -> dict[str, Any]:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        s = self._load()
        t = s["tasks"].get(task_id)
        if t is None:
            raise KeyError(f"unknown task {task_id}")
        if outcome == "pass":
            t["status"] = "done"
            p = t["parent"]  # a parent is done once all its children are
            while p and all(c["status"] == "done" for c in s["tasks"].values() if c["parent"] == p):
                s["tasks"][p]["status"] = "done"
                p = s["tasks"][p]["parent"]
        elif outcome == "fail":
            detail = _summarize(detail)
            t["fails"] += 1
            t["last_error"] = detail
            t["status"] = "in_progress"
        elif t["status"] == "todo":
            t["status"] = "in_progress"
        key = "local_tokens" if local else "tokens"  # only closed-model tokens count against the cap
        s["budget"][key] = s["budget"].get(key, 0) + max(0, tokens)
        self._save(s)
        self._journal(task=task_id, outcome=outcome, detail=detail[:500], metric=metric, tokens=tokens, local=local)
        return t

    def verify(self, task_id: str, timeout: int = 600) -> dict[str, Any]:
        """Run the task's acceptance command from the project root and journal pass/fail.

        shortcut: acceptance is a shell command run as-is; it comes from the planner, so only
        the harness CLI calls this (no MCP tool). Sandbox it if plans ever come from untrusted sources.
        """
        t = self._load()["tasks"].get(task_id)
        if t is None:
            raise KeyError(f"unknown task {task_id}")
        if not t["acceptance"]:
            raise ValueError(f"{task_id} has no acceptance command")
        try:
            r = subprocess.run(t["acceptance"], shell=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(self._state_path))),
                               capture_output=True, text=True, timeout=timeout)
            ok, out = r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            ok, out = False, f"timed out after {timeout}s"
        return self.log(task_id, "pass" if ok else "fail", out)

    # -- stuck detection ---------------------------------------------------
    def stuck(self) -> list[dict[str, Any]]:
        """Open tasks that failed >= stuck_after times, or whose last N metrics never beat the earlier best."""
        s = self._load()
        n = self.stuck_after
        out = []
        for t in s["tasks"].values():
            if t["status"] == "done":
                continue
            metrics = [r["metric"] for r in self.journal(t["id"]) if r.get("metric") is not None]
            flat = len(metrics) > n and max(metrics[-n:]) <= max(metrics[:-n])
            if t["fails"] >= n or flat:
                out.append({**t, "reason": "repeated failures" if t["fails"] >= n else "no metric improvement"})
        return out

    # -- brief / packet ----------------------------------------------------
    def _current(self, s: dict[str, Any]) -> dict[str, Any] | None:
        """First open leaf in creation order, preferring in_progress."""
        tasks = s["tasks"]
        parents = {v["parent"] for v in tasks.values()}
        leaves = [t for t in tasks.values() if t["id"] not in parents and t["status"] in OPEN]
        return next((t for t in leaves if t["status"] == "in_progress"), leaves[0] if leaves else None)

    def brief(self) -> str:
        """Short 'you are here, do this next' brief for a fresh context."""
        s = self._load()
        total = len(s["tasks"])
        done = sum(t["status"] == "done" for t in s["tasks"].values())
        cur = self._current(s)
        lines = [f"GOAL: {s['goal'] or '(unset)'}", f"PROGRESS: {done}/{total} tasks done"]
        if cur:
            lines += [f"NEXT [{cur['id']}]: {cur['title']}",
                      f"ACCEPT: {cur['acceptance'] or '(none)'}",
                      f"FAILS: {cur['fails']}" + (f" last: {cur['last_error']}" if cur["last_error"] else "")]
        else:
            lines.append("NEXT: none open" if total else "NEXT: no plan yet; request one")
        if self.stuck():
            lines.append("STUCK: escalate to planner (plan_view view=packet)")
        return "\n".join(lines)

    def packet(self) -> str:
        """Compact brief for the closed-model planner. Accepts back via accept_plan."""
        s = self._load()
        done = [t["title"] for t in s["tasks"].values() if t["status"] == "done"]
        stuck = self.stuck() or ([c] if (c := self._current(s)) and c["fails"] else [])
        lines = [f"GOAL: {s['goal']}", f"DONE ({len(done)}): " + "; ".join(done[-10:])]
        for t in stuck:
            tried = [f"{r['outcome']}: {r['detail']}" for r in self.journal(t["id"]) if r["outcome"] != "note"]
            lines += [f"STUCK [{t['id']}] {t['title']} (accept: {t['acceptance']})",
                      "TRIED: " + " | ".join(tried[-5:])]
        lines.append("NEED: a revised list of sub-tasks as JSON "
                     '[{"title","acceptance","children?"}] for the stuck task(s).')
        return "\n".join(lines)

    def budget(self) -> dict[str, Any]:
        b = self._load()["budget"]
        over = (b["max_escalations"] and b["escalations"] >= b["max_escalations"]) or \
               (b["max_tokens"] and b["tokens"] >= b["max_tokens"])
        return {**b, "escalation_allowed": not over}

    def tree(self) -> str:
        s = self._load()
        kids: dict[str, list[dict[str, Any]]] = {}
        for t in s["tasks"].values():
            kids.setdefault(t["parent"], []).append(t)

        def walk(pid: str, d: int) -> list[str]:
            out: list[str] = []
            for t in kids.get(pid, []):
                out.append(f"{'  ' * d}[{t['id']}] {t['status']}: {t['title']}")
                out += walk(t["id"], d + 1)
            return out

        return "\n".join(walk("", 0)) or "(empty plan)"


def parse_plan(text: str) -> list[dict[str, Any]]:
    """Extract the JSON task list from raw text (a tmux pane capture may add prose or fences)."""
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j < i:
        raise ValueError("no JSON task list found")
    plan = json.loads(text[i:j + 1])
    if not isinstance(plan, list) or not all(isinstance(t, dict) and "title" in t for t in plan):
        raise ValueError('plan must be a list of {"title", ...} objects')
    return plan


def parse_usage(text: str) -> int:
    """Sum tokens (input+output+reasoning) over opencode `run --format json` step_finish events.

    Input is summed per step because the context is re-sent each step; that is what was processed.
    """
    total = 0
    for line in text.splitlines():
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "step_finish":
            t = ev.get("part", {}).get("tokens", {})
            total += sum(int(t.get(k, 0) or 0) for k in ("input", "output", "reasoning"))
    return total
