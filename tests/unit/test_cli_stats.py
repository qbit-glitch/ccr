"""Unit tests for the ccr stats CLI command and reminder-overhead accounting."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from ccr.cli import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(tmp_path, sessions: list[dict] | None = None) -> str:
    """Create a minimal project directory with optional session records."""
    ccr_dir = tmp_path / ".ccr"
    ccr_dir.mkdir()
    if sessions is not None:
        metrics_dir = ccr_dir / "metrics"
        metrics_dir.mkdir()
        jsonl = metrics_dir / "sessions.jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            for rec in sessions:
                f.write(json.dumps(rec) + "\n")
    return str(tmp_path)


def _sample_record(
    context_tokens: int = 1000,
    duration_min: float = 10.0,
    reminder_overhead_tokens: int | None = None,
    start: str = "2026-04-01T10:00:00+00:00",
) -> dict:
    rec: dict = {
        "session_id": "ses_test",
        "start": start,
        "end": "2026-04-01T10:10:00+00:00",
        "context_tokens": context_tokens,
        "duration_min": duration_min,
        "branch": "/tmp/proj/.ccr",
    }
    if reminder_overhead_tokens is not None:
        rec["reminder_overhead_tokens"] = reminder_overhead_tokens
    return rec


# ---------------------------------------------------------------------------
# Tests: reminder_overhead_tokens field handling in cli_stats
# ---------------------------------------------------------------------------


class TestStatsOverheadDisplay:
    """ccr stats correctly shows gross/overhead/net when overhead data is present."""

    def test_gross_and_net_shown_when_overhead_present(self, tmp_path):
        """Sessions with reminder_overhead_tokens > 0 show gross, overhead, and net."""
        sessions = [
            _sample_record(context_tokens=2000, reminder_overhead_tokens=180),
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project, "--multiplier", "4"])
        assert result.exit_code == 0, result.output
        assert "Gross savings" in result.output
        assert "Overhead" in result.output
        assert "Net savings" in result.output
        # Gross = 2000 * 4 = 8000; overhead = 180; net = 7820
        assert "8,000" in result.output
        assert "180" in result.output
        assert "7,820" in result.output

    def test_net_savings_equals_gross_minus_overhead(self, tmp_path):
        """Net savings = gross_avoided - reminder_overhead across all sessions."""
        sessions = [
            _sample_record(context_tokens=1000, reminder_overhead_tokens=90),
            _sample_record(context_tokens=1000, reminder_overhead_tokens=90,
                           start="2026-04-02T10:00:00+00:00"),
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project, "--multiplier", "4"])
        assert result.exit_code == 0, result.output
        # gross = 2000 * 4 = 8000; total overhead = 180; net = 7820
        assert "8,000" in result.output
        assert "180" in result.output
        assert "7,820" in result.output

    def test_no_overhead_column_for_old_records(self, tmp_path):
        """Sessions without reminder_overhead_tokens show legacy Avoided column."""
        sessions = [
            _sample_record(context_tokens=500),  # no reminder_overhead_tokens key
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project])
        assert result.exit_code == 0, result.output
        # Aggregate line says "Net savings" with an n/a note for older records
        assert "Net savings" in result.output
        assert "n/a" in result.output
        # Session table uses legacy "Avoided" column (no overhead data)
        assert "Avoided" in result.output
        # No "Overhead" aggregate label (only shown when total_overhead > 0)
        assert "Overhead:" not in result.output

    def test_overhead_column_shown_in_session_table(self, tmp_path):
        """Per-session table gains Overhead and Net saved columns when overhead data exists."""
        sessions = [
            _sample_record(context_tokens=1000, reminder_overhead_tokens=45),
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project])
        assert result.exit_code == 0, result.output
        assert "Overhead" in result.output
        assert "Net saved" in result.output

    def test_zero_overhead_field_treated_as_no_data(self, tmp_path):
        """reminder_overhead_tokens = 0 means no overhead column (single-turn sessions)."""
        sessions = [
            _sample_record(context_tokens=800, reminder_overhead_tokens=0),
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project])
        assert result.exit_code == 0, result.output
        # With 0 overhead and no session having >0, the legacy table path is shown
        assert "Avoided" in result.output

    def test_net_never_negative(self, tmp_path):
        """Net savings is clamped to 0 even if overhead somehow exceeds gross."""
        sessions = [
            _sample_record(context_tokens=10, reminder_overhead_tokens=9999),
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project, "--multiplier", "4"])
        assert result.exit_code == 0, result.output
        assert "Net savings:           0" in result.output

    def test_disclaimer_references_net_savings(self, tmp_path):
        """Disclaimer at bottom explains gross vs net savings formula."""
        sessions = [
            _sample_record(context_tokens=1000, reminder_overhead_tokens=45),
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project])
        assert result.exit_code == 0, result.output
        assert "Net savings" in result.output
        assert "per-turn reminder overhead" in result.output
        assert "32 tokens/turn" in result.output
        assert "tool-use turns" in result.output

    def test_30_session_projection_shows_net(self, tmp_path):
        """30-session projection shows Est. net savings when overhead data is present."""
        sessions = [
            _sample_record(context_tokens=1000, reminder_overhead_tokens=90),
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project])
        assert result.exit_code == 0, result.output
        assert "Est. net savings" in result.output

    def test_30_session_projection_no_net_without_overhead_data(self, tmp_path):
        """30-session projection omits Net line when no overhead data."""
        sessions = [
            _sample_record(context_tokens=1000),  # no overhead field
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project])
        assert result.exit_code == 0, result.output
        assert "Est. net savings" not in result.output

    def test_mixed_records_old_and_new(self, tmp_path):
        """Mix of old records (no overhead field) and new ones (with overhead) is handled."""
        sessions = [
            _sample_record(context_tokens=1000),  # old record — no overhead
            _sample_record(context_tokens=1000, reminder_overhead_tokens=90,
                           start="2026-04-02T10:00:00+00:00"),
        ]
        project = _make_project(tmp_path, sessions)
        runner = CliRunner()
        result = runner.invoke(cli, ["stats", project])
        assert result.exit_code == 0, result.output
        # Total overhead = 90; gross = 2000*4 = 8000; net = 7910
        assert "Overhead" in result.output
        assert "Net savings" in result.output


# ---------------------------------------------------------------------------
# Tests: _write_session_metrics in on_stop.py emits reminder_overhead_tokens
# ---------------------------------------------------------------------------


class TestWriteSessionMetricsOverhead:
    """_write_session_metrics includes reminder_overhead_tokens field."""

    def test_overhead_field_present_in_metrics_record(self, tmp_path):
        """Calling _write_session_metrics writes reminder_overhead_tokens to JSONL."""
        from ccr.hooks.on_stop import _write_session_metrics

        class FakeState:
            context_tokens = 500
            session_id = "ses_test"
            start_time = 0
            tool_calls = 4  # all 4 subsequent turns had tool use

        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        _write_session_metrics(ccr_root, FakeState(), turn_count=5)

        jsonl = os.path.join(ccr_root, "metrics", "sessions.jsonl")
        assert os.path.isfile(jsonl)
        with open(jsonl) as f:
            rec = json.loads(f.read().strip())
        assert "reminder_overhead_tokens" in rec
        # 5 turns, tool_calls=4 → min(4, 4) * 32 = 128
        assert rec["reminder_overhead_tokens"] == 128

    def test_overhead_zero_for_single_turn(self, tmp_path):
        """Single-turn session (no subsequent prompts) has overhead = 0."""
        from ccr.hooks.on_stop import _write_session_metrics

        class FakeState:
            context_tokens = 200
            session_id = "ses_test"
            start_time = 0
            tool_calls = 0

        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        _write_session_metrics(ccr_root, FakeState(), turn_count=1)

        jsonl = os.path.join(ccr_root, "metrics", "sessions.jsonl")
        with open(jsonl) as f:
            rec = json.loads(f.read().strip())
        assert rec["reminder_overhead_tokens"] == 0

    def test_overhead_zero_when_turn_count_zero(self, tmp_path):
        """Zero turn_count (no session store) yields overhead = 0 (safe default)."""
        from ccr.hooks.on_stop import _write_session_metrics

        class FakeState:
            context_tokens = 300
            session_id = ""
            start_time = 0
            tool_calls = 5  # tool_calls > 0 but turn_count caps it to 0

        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        _write_session_metrics(ccr_root, FakeState(), turn_count=0)

        jsonl = os.path.join(ccr_root, "metrics", "sessions.jsonl")
        with open(jsonl) as f:
            rec = json.loads(f.read().strip())
        assert rec["reminder_overhead_tokens"] == 0

    def test_overhead_formula_max_zero_subturn(self, tmp_path):
        """Overhead formula: min(max(0, turns - 1), tool_calls) * 32 — never negative."""
        from ccr.hooks.on_stop import _write_session_metrics

        class FakeState:
            context_tokens = 100
            session_id = ""
            start_time = 0
            tool_calls = 0

        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        # turn_count = 0 → min(max(0, -1), 0) * 32 = 0
        _write_session_metrics(ccr_root, FakeState(), turn_count=0)
        jsonl = os.path.join(ccr_root, "metrics", "sessions.jsonl")
        with open(jsonl) as f:
            rec = json.loads(f.read().strip())
        assert rec["reminder_overhead_tokens"] == 0

    def test_overhead_default_is_zero(self, tmp_path):
        """Calling _write_session_metrics without turn_count defaults to 0."""
        from ccr.hooks.on_stop import _write_session_metrics

        class FakeState:
            context_tokens = 400
            session_id = ""
            start_time = 0
            tool_calls = 3

        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        _write_session_metrics(ccr_root, FakeState())  # no turn_count arg

        jsonl = os.path.join(ccr_root, "metrics", "sessions.jsonl")
        with open(jsonl) as f:
            rec = json.loads(f.read().strip())
        assert rec["reminder_overhead_tokens"] == 0

    def test_overhead_10_turn_session(self, tmp_path):
        """10-turn session with all turns having tool use → 9 × 32 = 288 tokens overhead."""
        from ccr.hooks.on_stop import _write_session_metrics

        class FakeState:
            context_tokens = 1000
            session_id = ""
            start_time = 0
            tool_calls = 9  # all 9 subsequent turns had tool use

        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        _write_session_metrics(ccr_root, FakeState(), turn_count=10)

        jsonl = os.path.join(ccr_root, "metrics", "sessions.jsonl")
        with open(jsonl) as f:
            rec = json.loads(f.read().strip())
        # min(9, 9) * 32 = 288
        assert rec["reminder_overhead_tokens"] == 288

    def test_skips_zero_context_tokens(self, tmp_path):
        """Sessions with context_tokens = 0 are still skipped (unchanged gate)."""
        from ccr.hooks.on_stop import _write_session_metrics

        class FakeState:
            context_tokens = 0
            session_id = ""
            start_time = 0
            tool_calls = 4

        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        _write_session_metrics(ccr_root, FakeState(), turn_count=5)

        jsonl = os.path.join(ccr_root, "metrics", "sessions.jsonl")
        assert not os.path.isfile(jsonl)

    def test_overhead_zero_for_pure_qa_session(self, tmp_path):
        """Pure Q&A session (tool_calls=0) has overhead=0 even with many turns."""
        from ccr.hooks.on_stop import _write_session_metrics

        class FakeState:
            context_tokens = 800
            session_id = ""
            start_time = 0
            tool_calls = 0  # no tool use — conditional reminder never fired

        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        _write_session_metrics(ccr_root, FakeState(), turn_count=15)

        jsonl = os.path.join(ccr_root, "metrics", "sessions.jsonl")
        with open(jsonl) as f:
            rec = json.loads(f.read().strip())
        # min(14, 0) * 32 = 0
        assert rec["reminder_overhead_tokens"] == 0

    def test_overhead_partial_tool_use_caps_at_tool_calls(self, tmp_path):
        """Session with fewer tool-use turns than total turns is capped at tool_calls."""
        from ccr.hooks.on_stop import _write_session_metrics

        class FakeState:
            context_tokens = 600
            session_id = ""
            start_time = 0
            tool_calls = 3  # only 3 of 10 turns had tool use

        ccr_root = str(tmp_path / ".ccr")
        os.makedirs(ccr_root)
        _write_session_metrics(ccr_root, FakeState(), turn_count=10)

        jsonl = os.path.join(ccr_root, "metrics", "sessions.jsonl")
        with open(jsonl) as f:
            rec = json.loads(f.read().strip())
        # min(9, 3) * 32 = 96 (not 9*45=405)
        assert rec["reminder_overhead_tokens"] == 96
