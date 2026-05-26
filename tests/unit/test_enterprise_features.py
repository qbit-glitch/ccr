"""Tests for industry-adoption CCR surfaces."""

from __future__ import annotations

import os
import shutil
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import ccr.mcp_server as mcp_mod
from ccr.cli import cli
from ccr.core.enterprise_gateway import EnterprisePolicyGateway
from ccr.core.events import AgentEvent, append_agent_event, validate_event_file
from ccr.core.governance import append_audit, redact_text, scan_text
from ccr.core.memory import MemoryManager
from ccr.core.memory_tiers import MemoryTierInspector
from ccr.core.ops import backup_project, restore_backup, verify_project
from ccr.core.rerankers import get_reranker
from ccr.core.sync import GitMemorySync
from ccr.mcp_server import _init, gcc_commit, gcc_facts


def _patch_home(tmp_path, monkeypatch):
    fake_home_ccr = tmp_path / "global_ccr"
    fake_home_ccr.mkdir()
    original_expanduser = os.path.expanduser

    def mock_expanduser(path):
        if path.startswith("~/.ccr"):
            return str(fake_home_ccr) + path[5:]
        return original_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)
    monkeypatch.setenv("CCR_STORAGE_BACKEND", "sqlite")


def _reset_mcp_globals():
    mcp_mod._memory = None
    mcp_mod._playbook = None
    mcp_mod._global_playbook = None
    mcp_mod._repo_index = None
    mcp_mod._repl = None


def test_governance_scan_redact_and_audit(tmp_path):
    text = "AWS_SECRET_ACCESS_KEY=abcdefghijklmnopqrstuvwxyz123456 and user alice@example.com"
    findings = scan_text(text, "memory.md")
    redacted = redact_text(text)

    assert len(findings) >= 2
    assert "alice@example.com" not in redacted
    assert "[REDACTED:email]" in redacted

    ccr_root = tmp_path / ".ccr"
    ccr_root.mkdir()
    first = append_audit(str(ccr_root), "scan", "tester", {"findings": 1})
    second = append_audit(str(ccr_root), "scan", "tester", {"findings": 0})

    assert first["hash"]
    assert second["prev_hash"] == first["hash"]


def test_append_audit_skips_mock_roots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = append_audit(MagicMock(), "memory_read", "tester", {})

    assert result["skipped"] is True
    assert result["reason"] == "non_ccr_root"
    assert not (tmp_path / "MagicMock").exists()


def test_agent_event_schema_validation(tmp_path):
    ccr_root = tmp_path / ".ccr"
    ccr_root.mkdir()
    append_agent_event(str(ccr_root), AgentEvent(
        agent="codex",
        session_id="s1",
        project=str(tmp_path),
        event_type="tool_use",
        tools_used=["gcc_commit"],
        outcome="committed",
    ))

    valid, errors = validate_event_file(str(ccr_root / "agent_events.jsonl"))

    assert valid == 1
    assert errors == []


def test_memory_tiers_browser_and_cli_commands(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "app.py").write_text("def hello(): pass\n")
    _init(str(tmp_path))
    try:
        commit = gcc_commit(
            title="Choose SQLite backend",
            what="Decided SQLite is the memory backend.",
            why="It is local-first.",
            files_changed=["app.py"],
            next_step="Record fact",
            admission_threshold=1.0,
        )
        gcc_facts(
            action="add",
            key="memory_backend",
            statement="SQLite is the chosen backend.",
            source_commit=commit["commit_id"],
        )

        mem = MemoryManager(str(tmp_path))
        tiers = MemoryTierInspector(mem).snapshot()

        assert [t.tier for t in tiers] == [
            "scratchpad", "session", "commit", "fact", "pattern", "playbook",
        ]
        assert next(t for t in tiers if t.tier == "fact").count == 1

        runner = CliRunner()
        browser = runner.invoke(cli, ["browser", str(tmp_path)])
        tiers_cli = runner.invoke(cli, ["tiers", str(tmp_path)])

        assert browser.exit_code == 0, browser.output
        assert (tmp_path / ".ccr" / "browser" / "index.html").is_file()
        assert "CCR Memory Browser" in (tmp_path / ".ccr" / "browser" / "index.html").read_text()
        assert tiers_cli.exit_code == 0, tiers_cli.output
        assert "scratchpad -> session -> commit -> fact -> pattern -> playbook" in tiers_cli.output
    finally:
        _reset_mcp_globals()


def test_ops_backup_restore_verify_and_corrupt_edge(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    runner = CliRunner()
    init = runner.invoke(cli, ["init", str(tmp_path)])
    assert init.exit_code == 0, init.output

    (tmp_path / ".ccr" / "facts.json").write_text('{"version": 1, "facts": []}\n')
    report = verify_project(str(tmp_path))
    assert report.ok

    backup = backup_project(str(tmp_path))
    assert backup.ok
    backup_path = backup.artifacts[0]
    assert os.path.isfile(backup_path + ".manifest.json")

    restore_target = tmp_path / "restore-target"
    restored = restore_backup(str(restore_target), backup_path)
    assert restored.ok
    assert (restore_target / ".ccr").is_dir()

    (tmp_path / ".ccr" / "facts.json").write_text("{bad json")
    corrupt = verify_project(str(tmp_path))
    assert not corrupt.ok
    assert any("facts.json parse failed" in issue for issue in corrupt.issues)


def test_export_redacted_and_gateway_policy(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "app.py").write_text("def hello(): pass\n")
    _init(str(tmp_path))
    try:
        gcc_commit(
            title="Store secret fixture",
            what="Recorded OPENAI_API_KEY=sk-abc12345678901234567890 for redaction testing.",
            why="Exercise redacted export.",
            files_changed=["app.py"],
            next_step="Export",
            admission_threshold=1.0,
        )
        runner = CliRunner()
        out = tmp_path / "commits.json"
        result = runner.invoke(
            cli,
            ["export", "commits", str(tmp_path), "--redacted", "--output", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert "sk-abc" not in out.read_text()
        assert "[REDACTED:" in out.read_text()

        gate = EnterprisePolicyGateway(str(tmp_path))
        gate.init_policy()
        denied = gate.check_memory_write("token=supersecretvalue", actor_role="writer")
        allowed = gate.check_tool("gcc_commit", actor_role="writer")

        assert not denied.allowed
        assert allowed.allowed
    finally:
        _reset_mcp_globals()


def test_gateway_enforces_per_operation_role_grants(tmp_path):
    """check_tool must deny a role lacking the operation a write tool requires."""
    from ccr.core.memory import MemoryManager

    MemoryManager(str(tmp_path)).ensure_structure()
    gate = EnterprisePolicyGateway(str(tmp_path))
    gate.init_policy()

    # reader grants = recall/context/export_redacted — no commit/fact_write/sync_push
    assert not gate.check_tool("gcc_commit", actor_role="reader").allowed
    assert not gate.check_tool("gcc_facts", actor_role="reader").allowed
    # reader may still use read-only tools
    assert gate.check_tool("gcc_recall", actor_role="reader").allowed
    # writer holds commit; admin holds "*"
    assert gate.check_tool("gcc_commit", actor_role="writer").allowed
    assert gate.check_tool("gcc_commit", actor_role="admin").allowed


def test_reranker_falls_back_for_unknown_provider(monkeypatch):
    monkeypatch.setenv("CCR_RERANKER", "not-a-real-provider")
    assert get_reranker().provider == "lexical"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required for sync tests")
def test_git_sync_push_and_resolve(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    runner = CliRunner()
    assert runner.invoke(cli, ["init", str(tmp_path)]).exit_code == 0

    sync = GitMemorySync(str(tmp_path))
    init = sync.init()
    push = sync.push(message="test sync")
    resolve = sync.resolve()

    assert init.ok
    assert push.ok
    assert resolve.ok
    assert (tmp_path / ".ccr" / "sync" / "repo" / ".git").is_dir()
