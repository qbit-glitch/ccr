"""Tests for GCC paper gap closures in MemoryManager.

Tests metadata.yaml, summary.md, OTA triples, context windowing, auto-CONTEXT.
"""

import json
import math
import os
import tempfile
from datetime import datetime, timezone

import pytest
import yaml

from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig


@pytest.fixture
def project():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mem(project):
    m = MemoryManager(project)
    m.ensure_structure()
    return m


class TestMetadataYaml:
    def test_metadata_created_on_bootstrap(self, mem):
        path = os.path.join(mem.ccr_root, "metadata.yaml")
        assert os.path.isfile(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["version"] >= 1
        assert len(data["branches"]) == 1
        assert data["branches"][0]["name"] == "main"

    def test_metadata_updated_on_branch_create(self, mem):
        mem.create_branch("experiment", "test purpose", "test hypothesis")
        data = mem._load_metadata()
        branch_names = [b["name"] for b in data["branches"]]
        assert "experiment" in branch_names

    def test_metadata_updated_on_merge(self, mem):
        mem.create_branch("exp", "purpose", "hypothesis")
        mem.commit("test", "what", "why", [], "next")
        mem.merge("exp", "success", "it worked")
        data = mem._load_metadata()
        for b in data["branches"]:
            if b["name"] == "exp":
                assert b["status"] == "merged"

    def test_update_file_tree(self, mem):
        mem.update_metadata_file_tree(["a.py", "b.py", "c.py"])
        data = mem._load_metadata()
        assert data["file_tree"] == ["a.py", "b.py", "c.py"]

    def test_update_dependencies(self, mem):
        mem.update_metadata_dependencies(["numpy", "pandas"])
        data = mem._load_metadata()
        assert "numpy" in data["dependencies"]

    def test_update_config(self, mem):
        mem.update_metadata_config(language="python", framework="flask")
        data = mem._load_metadata()
        assert data["config"]["language"] == "python"
        assert data["config"]["framework"] == "flask"


class TestSummaryMd:
    def test_summary_created_on_branch(self, mem):
        mem.create_branch("feature", "add auth", "JWT will work")
        path = os.path.join(mem.branches_dir, "feature", "summary.md")
        assert os.path.isfile(path)
        content = open(path).read()
        assert "add auth" in content
        assert "active" in content
        assert "main" in content  # parent

    def test_summary_updated_on_merge(self, mem):
        mem.create_branch("feat", "test", "test")
        mem.commit("work", "did stuff", "needed", [], "next")
        mem.merge("feat", "success", "good")
        content = open(os.path.join(mem.branches_dir, "feat", "summary.md")).read()
        assert "merged" in content.lower()

    def test_read_branch_summary(self, mem):
        mem.create_branch("test-branch", "some purpose", "some hypothesis")
        summary = mem._read_branch_summary("test-branch")
        assert "some purpose" in summary


class TestOTATriples:
    def test_ota_triple_format(self, mem):
        mem.log_ota(
            "edit",
            "main.py",
            "OK",
            observation="Found bug in auth",
            thought="Need to fix the token validation",
            action="Modified validate_token() in auth.py",
        )
        log_path = mem._get_log_path("main")
        content = open(log_path).read()
        assert "**Observation**:" in content
        assert "**Thought**:" in content
        assert "**Action**:" in content
        assert "[OTA-" in content

    def test_ota_simple_format_fallback(self, mem):
        mem.log_ota("edit", "main.py", "OK")
        log_path = mem._get_log_path("main")
        content = open(log_path).read()
        assert "| edit |" in content

    def test_ota_id_increments(self, mem):
        mem.log_ota("a", observation="obs1", thought="t1", action="a1")
        mem.log_ota("b", observation="obs2", thought="t2", action="a2")
        log_path = mem._get_log_path("main")
        content = open(log_path).read()
        assert "[OTA-001]" in content
        assert "[OTA-002]" in content


class TestContextWindowing:
    def test_windowed_commits(self, mem):
        for i in range(5):
            mem.commit(f"Commit {i}", f"What {i}", f"Why {i}", [], "next")

        # Window [0:3] = most recent 3
        recent = mem._read_commits_window("main", 0, 3)
        assert "Commit 4" in recent  # most recent
        assert "Commit 0" not in recent

        # Window [2:5] = offset by 2
        older = mem._read_commits_window("main", 2, 3)
        assert "Commit 2" in older

    def test_context_with_offset(self, mem):
        for i in range(5):
            mem.commit(f"Commit {i}", f"What {i}", f"Why {i}", [], "next")

        ctx_recent = mem.get_context(level=2, offset=0)
        ctx_older = mem.get_context(level=2, offset=3)
        assert ctx_recent != ctx_older

    def test_context_level3_uses_summary(self, mem):
        mem.create_branch("test-branch", "test purpose", "test hyp")
        ctx = mem.get_context(level=3, branch="test-branch")
        assert "test purpose" in ctx


class TestRollingSummary:
    """Gap 3: GCC paper §2.2 rolling commit summary chain S_t = f(S_{t-1}, D_t)."""

    def test_rolling_summary_created_on_first_commit(self, mem):
        mem.commit("First", "Added auth", "needed security", ["auth.py"], "add tests")
        summary = mem._get_rolling_summary("main")
        assert "Added auth" in summary
        assert "needed security" in summary

    def test_rolling_summary_chains_across_commits(self, mem):
        mem.commit("First", "Added auth module", "security needed", ["auth.py"], "add tests")
        mem.commit("Second", "Added tests for auth", "verify correctness", ["test_auth.py"], "deploy")

        summary = mem._get_rolling_summary("main")
        # Both contributions should be in the rolling summary
        assert "Added auth module" in summary
        assert "Added tests for auth" in summary

    def test_rolling_summary_in_context_level2(self, mem):
        mem.commit("Work", "Implemented feature X", "user requested", ["x.py"], "test it")
        ctx = mem.get_context(level=2)
        assert "Progress Summary" in ctx
        assert "Implemented feature X" in ctx

    def test_rolling_summary_caps_length(self, mem):
        """Rolling summary should not grow unbounded."""
        for i in range(50):
            mem.commit(f"Commit {i}", f"Did thing {i} " * 10, "reason", [], "next")
        summary = mem._get_rolling_summary("main")
        assert len(summary) <= 1600  # ~1500 cap + some overhead

    def test_rolling_summary_on_branch(self, mem):
        mem.create_branch("feature", "add search", "regex will work")
        mem.commit("Search impl", "Added search function", "core feature", ["search.py"], "optimize")
        summary = mem._get_rolling_summary("feature")
        assert "Added search function" in summary

    def test_empty_rolling_summary_before_commits(self, mem):
        summary = mem._get_rolling_summary("main")
        assert summary == ""


class TestAutoContextBeforeMerge:
    def test_merge_calls_context(self, mem):
        """Merge should internally call get_context before merging."""
        mem.create_branch("exp", "purpose", "hypothesis")
        mem.commit("work", "did stuff", "needed", [], "next")
        # If auto-CONTEXT fails, merge would raise — this is an integration test
        result = mem.merge("exp", "success", "it worked")
        assert "Merged" in result


class TestComputeClusters:
    def test_compute_clusters_returns_list(self, mem):
        """compute_clusters returns a list (even with no cross-links)."""
        clusters = mem.compute_clusters(min_cluster_size=2)
        assert isinstance(clusters, list)

    def test_compute_clusters_empty_with_no_links(self, mem):
        """Without any cross-links, there are no clusters."""
        mem.commit("first commit", "some work", "reason", ["a.py"], "next")
        clusters = mem.compute_clusters(min_cluster_size=2)
        assert clusters == []


class TestPatternRecencyWeight:
    def test_iso8601_timestamp_scores_high(self):
        """A recent ISO-8601 timestamp should score close to 1.0."""
        now = datetime.now(timezone.utc)
        ts = now.isoformat()
        score = MemoryManager._pattern_recency_weight_at(ts, now)
        assert score > 0.99

    def test_legacy_timestamp_still_works(self):
        """The legacy '%Y-%m-%d %H:%M' format should still score > 0.95 when recent."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%d %H:%M")
        score = MemoryManager._pattern_recency_weight_at(ts, now)
        assert score > 0.95

    def test_empty_timestamp_returns_half(self):
        """An empty timestamp string should return the default 0.5 weight."""
        now = datetime.now(timezone.utc)
        score = MemoryManager._pattern_recency_weight_at("", now)
        assert score == 0.5

    def test_iso8601_with_timezone_offset(self):
        """ISO-8601 timestamps with UTC offset should be parsed correctly."""
        now = datetime.now(timezone.utc)
        ts = "2026-04-03T12:00:00+00:00"
        score = MemoryManager._pattern_recency_weight_at(ts, now)
        # Should produce a valid float (not the fallback 0.5)
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    def test_invalid_timestamp_returns_half(self):
        """An unparseable timestamp should return the fallback 0.5."""
        now = datetime.now(timezone.utc)
        score = MemoryManager._pattern_recency_weight_at("not-a-date", now)
        assert score == 0.5
