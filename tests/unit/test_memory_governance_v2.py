"""Tests for CCR memory governance v2 surfaces."""

from __future__ import annotations

import os

from click.testing import CliRunner

import ccr.mcp_server as mcp_mod
from ccr.cli import cli
from ccr.core.browser import MemoryBrowserBuilder
from ccr.core.episodes import EpisodeStore
from ccr.core.memory import MemoryManager
from ccr.core.memory_eval import MemoryEvalRunner
from ccr.core.quarantine import MemoryQuarantine
from ccr.core.replay import RecallTraceStore
from ccr.core.temporal_graph import TemporalGraphStore
from ccr.mcp_server import (
    _init,
    gcc_conflicts,
    gcc_conflicts_resolve,
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
    monkeypatch.setenv("CCR_STORAGE_BACKEND", "sqlite")


def _reset_mcp_globals():
    mcp_mod._memory = None
    mcp_mod._playbook = None
    mcp_mod._global_playbook = None
    mcp_mod._repo_index = None
    mcp_mod._repl = None


def test_episode_store_hash_chain_and_search(tmp_path):
    ccr_root = tmp_path / ".ccr"
    ccr_root.mkdir()
    store = EpisodeStore(str(ccr_root))

    first = store.append_episode(
        "tool_observed",
        summary="Observed SQLite backend",
        content="SQLite is active",
        source_ids=["C001"],
    )
    second = store.append_episode(
        "fact_added",
        summary="Recorded backend fact",
        content="SQLite is the backend",
        source_ids=["F001", "C001"],
    )

    assert first.id == "E001"
    assert second.prev_hash == first.hash
    assert store.verify_chain().ok
    assert {episode.id for episode in store.list_episodes(query="backend")} == {"E001", "E002"}


def test_quarantine_speculative_fact_then_promote(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "app.py").write_text("print('hi')\n")
    _init(str(tmp_path))
    try:
        quarantined = gcc_facts(
            action="add",
            key="release_status",
            statement="The product is probably release-ready.",
            classification="speculative",
            confidence=0.4,
        )

        assert quarantined["count"] == 0
        assert "Quarantined speculative memory Q001" in quarantined["message"]
        queue = MemoryQuarantine(str(tmp_path / ".ccr")).list_items()
        assert queue[0].id == "Q001"
        assert gcc_facts(action="list", query="release_status")["count"] == 0

        runner = CliRunner()
        promoted = runner.invoke(cli, ["quarantine", "promote", str(tmp_path), "Q001"])
        assert promoted.exit_code == 0, promoted.output

        listed = gcc_facts(action="list", query="release_status")
        assert listed["count"] == 1
        assert listed["facts"][0]["classification"] == "speculative"
    finally:
        _reset_mcp_globals()


def test_temporal_graph_recall_trace_and_browser(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "app.py").write_text("print('hi')\n")
    _init(str(tmp_path))
    try:
        old = gcc_facts(
            action="add",
            key="windows_support",
            statement="Windows support is implemented.",
            classification="confirmed",
            confidence=0.7,
        )
        new = gcc_facts(
            action="add",
            key="windows_support",
            statement="Windows support is not implemented.",
            classification="tool_observed",
            confidence=0.9,
        )
        gcc_facts(
            action="supersede",
            fact_id=old["facts"][0]["id"],
            superseded_by=new["facts"][0]["id"],
        )

        result = gcc_recall("What is the current memory for windows_support?")
        graph = TemporalGraphStore(str(tmp_path / ".ccr")).load()
        traces = RecallTraceStore(str(tmp_path / ".ccr")).list_traces()
        html = MemoryBrowserBuilder(MemoryManager(str(tmp_path))).build_html()

        assert result["plan"]["intent"] == "temporal"
        assert result["trace_id"] == "R001"
        assert any(ev["source"] == "graph" for ev in result["evidence"])
        assert result["stale_notes"]
        assert graph["nodes"]
        assert traces[0].plan["strategy"] == "temporal-graph-first"
        assert "Recall Replay: Why Was This Recalled?" in html
    finally:
        _reset_mcp_globals()


def test_conflicts_resolve_cli_and_mcp(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "app.py").write_text("print('hi')\n")
    _init(str(tmp_path))
    try:
        first = gcc_facts(
            action="add",
            key="sync_status",
            statement="Sync is enabled.",
        )
        second = gcc_facts(
            action="add",
            key="sync_status",
            statement="Sync is not enabled.",
        )
        assert gcc_conflicts(query="sync_status")["count"] == 1

        resolved = gcc_conflicts_resolve(
            winner_fact_id=second["facts"][0]["id"],
            loser_fact_id=first["facts"][0]["id"],
            reason="Manual test resolution",
        )
        assert resolved["count"] == 0
        assert "Resolved conflict" in resolved["message"]

        active = gcc_facts(action="list", query="sync_status")
        assert active["count"] == 1
        assert active["facts"][0]["id"] == second["facts"][0]["id"]

        third = gcc_facts(
            action="add",
            key="sync_status",
            statement="Sync is enabled.",
        )
        runner = CliRunner()
        cli_resolve = runner.invoke(
            cli,
            [
                "conflicts",
                "resolve",
                str(tmp_path),
                "--winner",
                active["facts"][0]["id"],
                "--loser",
                third["facts"][0]["id"],
                "--reason",
                "CLI resolution",
            ],
        )
        assert cli_resolve.exit_code == 0, cli_resolve.output
        assert gcc_conflicts(query="sync_status")["count"] == 0
    finally:
        _reset_mcp_globals()


def test_memory_eval_v2_thresholds_and_ci_cli(tmp_path, monkeypatch):
    _patch_home(tmp_path, monkeypatch)
    (tmp_path / "app.py").write_text("print('hi')\n")
    _init(str(tmp_path))
    try:
        gcc_facts(
            action="add",
            key="memory_backend",
            statement="SQLite is the memory backend.",
        )

        report = MemoryEvalRunner(str(tmp_path)).run(
            suite="v2",
            thresholds={"accuracy": 0.5, "fact-citation": 0.5},
        )
        assert report.meets_thresholds
        assert "fact-citation" in report.category_metrics

        runner = CliRunner()
        passing = runner.invoke(
            cli,
            [
                "memory-eval",
                "--project",
                str(tmp_path),
                "--suite",
                "v2",
                "--ci",
                "--min-accuracy",
                "0.5",
            ],
        )
        failing = runner.invoke(
            cli,
            [
                "memory-eval",
                "--project",
                str(tmp_path),
                "--suite",
                "v2",
                "--ci",
                "--min-conflict-accuracy",
                "1.0",
            ],
        )

        assert passing.exit_code == 0, passing.output
        assert "Thresholds: PASS" in passing.output
        assert failing.exit_code == 1
        assert "Thresholds: FAIL" in failing.output
    finally:
        _reset_mcp_globals()
