"""Tests for durable CCR TODO support."""

from __future__ import annotations

import tempfile

from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig
from ccr.hooks.on_session_start import _codex_session_summary
from ccr.mcp_server import (
    _init,
    gcc_commit,
    gcc_context,
    gcc_status,
    gcc_todo_add,
    gcc_todo_delete,
    gcc_todo_digest,
    gcc_todo_done,
    gcc_todo_update,
    gcc_todos,
)
from ccr.mcp_server import mcp as mcp_instance


def test_todo_round_trip_file_and_sqlite_backends():
    for backend in ("files", "sqlite"):
        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryManager(tmp, CCRConfig(storage_backend=backend))
            mem.ensure_structure()
            todo = mem.todo_add("Add TODO storage", priority="high", labels=["mcp"])

            assert todo["id"] == "T001"
            assert mem.todo_counts()["open"] == 1
            assert "Add TODO storage" in mem.format_todo_digest()

            updated = mem.todo_update(todo["id"], {"status": "in_progress", "percent_complete": 25})
            assert updated is not None
            assert updated["status"] == "in_progress"
            assert updated["percent_complete"] == 25
            assert updated["started_at"]

            done = mem.todo_done(todo["id"], source_commit="C001")
            assert done is not None
            assert done["status"] == "done"
            assert done["percent_complete"] == 100
            assert mem.todo_counts()["open"] == 0
            mem._storage.close()


def test_todo_parent_and_blocker_validation():
    with tempfile.TemporaryDirectory() as tmp:
        mem = MemoryManager(tmp, CCRConfig(storage_backend="sqlite"))
        mem.ensure_structure()
        parent = mem.todo_add("Parent")
        child = mem.todo_add("Child", parent_id=parent["id"], blocked_by=[parent["id"]])

        assert child["parent_id"] == parent["id"]
        assert child["blocked_by"] == [parent["id"]]

        try:
            mem.todo_update(child["id"], {"parent_id": child["id"]})
        except ValueError as exc:
            assert "own parent" in str(exc)
        else:
            raise AssertionError("Expected self-parent validation error")
        mem._storage.close()


def test_mcp_todo_tools_and_digest(tmp_path):
    _init(str(tmp_path))

    added = gcc_todo_add("Implement TODO MCP tools", priority="urgent", labels=["todo"])
    todo_id = added["todos"][0]["id"]

    listed = gcc_todos()
    assert listed["count"] == 1
    assert todo_id in listed["message"]

    updated = gcc_todo_update(todo_id, status="blocked", blocked_by=[])
    assert updated["todos"][0]["status"] == "blocked"

    digest = gcc_todo_digest(limit=3)
    assert digest["count"] == 1
    assert digest["blocked"] == 1
    assert "Implement TODO MCP tools" in digest["message"]

    done = gcc_todo_done(todo_id, source_commit="C001")
    assert done["todos"][0]["status"] == "done"
    assert gcc_todos()["count"] == 0

    deleted = gcc_todo_delete(todo_id, hard=True)
    assert deleted["count"] == 1


def test_context_status_and_commit_surface_todos(tmp_path):
    _init(str(tmp_path))
    added = gcc_todo_add("Wire TODO digest into context", priority="high")
    todo_id = added["todos"][0]["id"]

    context = gcc_context(level=1)
    assert "## Active TODOs" in context["message"]
    assert "Wire TODO digest into context" in context["message"]

    status = gcc_status()
    assert "TODOs:" in status["message"]
    assert "1 todo" in status["message"]

    commit = gcc_commit(
        "Finish digest wiring",
        "Wired TODO digest into memory context",
        "Agents need active work visible",
        ["ccr/core/memory_pkg/memory_context.py"],
        "Add broader tests",
        complete_todos=[todo_id],
        admission_threshold=1.0,
    )
    assert commit["todo_updates"][0]["action"] == "done"
    assert gcc_todo_digest()["count"] == 0


def test_codex_session_summary_includes_todo_count_only():
    output = _codex_session_summary(
        "## Current Focus\nUseful focus\n",
        todo_counts={"open": 2, "blocked": 1, "overdue": 1},
    )
    assert "TODOs: 2 active, 1 blocked, 1 overdue" in output
    assert "## Active TODOs" not in output


def test_todo_tools_registered_with_annotations():
    tools = mcp_instance._tool_manager._tools
    for name in (
        "gcc_todos",
        "gcc_todo_add",
        "gcc_todo_update",
        "gcc_todo_done",
        "gcc_todo_delete",
        "gcc_todo_digest",
    ):
        assert name in tools
        assert tools[name].annotations is not None

    assert tools["gcc_todos"].annotations.readOnlyHint is True
    assert tools["gcc_todo_digest"].annotations.readOnlyHint is True
    assert tools["gcc_todo_add"].annotations.readOnlyHint is False
    assert tools["gcc_todo_delete"].annotations.destructiveHint is True
