"""Tests for evidence-first recall, fact ledger, conflicts, and memory-eval."""

from __future__ import annotations

import os

from click.testing import CliRunner

import ccr.mcp_server as mcp_mod
from ccr.cli import cli
from ccr.mcp_server import (
    _init,
    gcc_commit,
    gcc_conflicts,
    gcc_facts,
    gcc_recall,
)


def _patch_home(tmp_path, monkeypatch):
    fake_home_ccr = tmp_path / "global_ccr"
    fake_home_ccr.mkdir()
    original_expanduser = os.path.expanduser

    def mock_expanduser(path):
        if path.startswith("~/.ccr"):
            return str(fake_home_ccr) + path[5:]
        return original_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)


def _reset_mcp_globals():
    mcp_mod._memory = None
    mcp_mod._playbook = None
    mcp_mod._global_playbook = None
    mcp_mod._repo_index = None
    mcp_mod._repl = None


def test_gcc_recall_returns_commit_evidence(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "hello.py").write_text("def greet(): pass\n")
    _init(str(tmp_path))
    try:
        commit = gcc_commit(
            title="Choose SQLite backend",
            what="Decided SQLite should be the default backend for MCP memory.",
            why="It keeps CCR local-first while supporting FTS5 and WAL.",
            files_changed=["hello.py"],
            next_step="Add evals",
            admission_threshold=1.0,
        )

        result = gcc_recall("Which backend did we choose for MCP memory?")

        assert result["confidence"] > 0
        assert result["evidence"]
        assert commit["commit_id"] in [e["id"] for e in result["evidence"]]
        assert "Evidence" in result["message"]
    finally:
        _reset_mcp_globals()


def test_gcc_facts_and_conflicts(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "hello.py").write_text("def greet(): pass\n")
    _init(str(tmp_path))
    try:
        first = gcc_facts(
            action="add",
            key="windows_support",
            statement="Windows support is implemented.",
            confidence=0.8,
        )
        second = gcc_facts(
            action="add",
            key="windows_support",
            statement="Windows support is not implemented.",
            confidence=0.8,
        )

        listed = gcc_facts(action="list", query="windows support")
        conflicts = gcc_conflicts(query="windows support")

        assert first["facts"][0]["id"] == "F001"
        assert second["facts"][0]["id"] == "F002"
        assert listed["count"] == 2
        assert conflicts["count"] == 1
        assert conflicts["conflicts"][0]["severity"] == "high"
    finally:
        _reset_mcp_globals()


def test_gcc_recall_reports_fact_conflict(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "hello.py").write_text("def greet(): pass\n")
    _init(str(tmp_path))
    try:
        gcc_facts(action="add", key="sync_mode", statement="Sync mode is enabled.")
        gcc_facts(action="add", key="sync_mode", statement="Sync mode is not enabled.")

        result = gcc_recall("sync mode")

        assert result["conflict_notes"]
        assert result["confidence"] < 1.0
        assert "Conflict Notes" in result["message"]
    finally:
        _reset_mcp_globals()


def test_gcc_recall_preserves_fact_source_commit(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "hello.py").write_text("def greet(): pass\n")
    _init(str(tmp_path))
    try:
        fact = gcc_facts(
            action="add",
            key="memory_backend",
            statement="CCR memory backend is sqlite for Codex wrapper runs.",
            source_commit="C002",
            source_file="ccr/cli_codex.py",
            confidence=0.9,
        )

        result = gcc_recall("What is the memory backend for Codex wrapper runs?")
        evidence = next(e for e in result["evidence"] if e["id"] == fact["facts"][0]["id"])

        assert evidence["metadata"]["source_commit"] == "C002"
        assert evidence["metadata"]["source_file"] == "ccr/cli_codex.py"
        assert "commit C002" in result["message"]
        assert "ccr/cli_codex.py" in result["message"]
    finally:
        _reset_mcp_globals()


def test_gcc_recall_filters_weak_generic_fact_matches(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "hello.py").write_text("def greet(): pass\n")
    _init(str(tmp_path))
    try:
        backend = gcc_facts(
            action="add",
            key="memory_backend",
            statement="SQLite is the chosen MCP memory backend.",
        )
        stale = gcc_facts(
            action="add",
            key="windows_support",
            statement="Windows support is implemented.",
            confidence=0.5,
        )
        current = gcc_facts(
            action="add",
            key="windows_support",
            statement="Windows support is not implemented.",
            confidence=0.9,
        )
        gcc_facts(
            action="supersede",
            fact_id=stale["facts"][0]["id"],
            superseded_by=current["facts"][0]["id"],
        )

        result = gcc_recall("What is the current memory for windows_support?")
        evidence_ids = [e["id"] for e in result["evidence"]]

        assert backend["facts"][0]["id"] not in evidence_ids
        assert current["facts"][0]["id"] in evidence_ids
        assert stale["facts"][0]["id"] in evidence_ids
        assert result["stale_notes"]
    finally:
        _reset_mcp_globals()


def test_memory_eval_cli_text_report(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    monkeypatch.setenv("CCR_STORAGE_BACKEND", "sqlite")
    (tmp_path / "hello.py").write_text("def greet(): pass\n")
    _init(str(tmp_path))
    try:
        gcc_commit(
            title="Add evidence recall",
            what="Added evidence-backed recall for CCR memory.",
            why="Industry users need proof for remembered facts.",
            files_changed=["hello.py"],
            next_step="Run memory eval",
            admission_threshold=1.0,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["memory-eval", "--project", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "CCR Memory Eval" in result.output
        assert "commit-citation" in result.output
    finally:
        _reset_mcp_globals()
