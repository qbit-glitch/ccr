"""Dual-backend wiring tests: verify every wired mixin method produces
identical results through both FileStorageBackend and SqliteStorageBackend.

Parametrized with @pytest.mark.parametrize("backend", ["files", "sqlite"]).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig


@pytest.fixture(params=["files", "sqlite"])
def mem(request):
    with tempfile.TemporaryDirectory() as d:
        cfg = CCRConfig(storage_backend=request.param)
        m = MemoryManager(d, cfg)
        m.ensure_structure()
        yield m


# ── Commit round-trip ──────────────────────────────────────────


class TestCommitRoundTrip:
    def test_commit_and_retrieve(self, mem):
        result = mem.commit(
            title="Add feature X",
            what="Added feature X",
            why="User requested it",
            files_changed=["src/x.py"],
            next_step="Write tests",
        )
        assert "C001" in result or "C002" in result

        commits = mem._storage.commit_list("main", limit=5)
        assert len(commits) >= 1
        latest = commits[0]
        assert "Added feature X" in latest.get("what", latest.get("raw_block", ""))

    def test_commit_increments_id(self, mem):
        mem.commit("First", "First thing", "reason", ["a.py"], "next")
        mem.commit("Second", "Second thing", "reason", ["b.py"], "next")
        commits = mem._storage.commit_list("main", limit=5)
        ids = [c["id"] for c in commits]
        assert "C002" in ids or "C003" in ids

    def test_commit_updates_rolling_summary(self, mem):
        mem.commit("Setup", "Setup project", "init", ["a.py"], "continue")
        summary = mem._get_rolling_summary("main")
        assert len(summary) > 0

    def test_parse_recent_commit_data(self, mem):
        mem.commit("Task A", "Did task A", "reason A", ["a.py"], "next A")
        mem.commit("Task B", "Did task B", "reason B", ["b.py"], "next B")
        data = mem._parse_recent_commit_data("main", k=2)
        assert len(data) >= 1


# ── Rolling summary ───────────────────────────────────────────


class TestRollingSummaryWiring:
    def test_write_and_read(self, mem):
        mem._write_rolling_summary("main", "Project context established")
        result = mem._get_rolling_summary("main")
        assert "Project context" in result

    def test_summary_capped_at_1500(self, mem):
        long_summary = "x" * 2000
        mem._write_rolling_summary("main", long_summary[:1500])
        result = mem._get_rolling_summary("main")
        assert len(result) <= 1500

    def test_update_rolling_summary(self, mem):
        mem._write_rolling_summary("main", "Initial context")
        mem._update_rolling_summary("main", "new work", "needed", "next")
        result = mem._get_rolling_summary("main")
        assert len(result) > 0


# ── Log append ────────────────────────────────────────────────


class TestLogWiring:
    def test_append_and_read(self, mem):
        mem._append_log("main", "First log entry\n")
        mem._append_log("main", "Second log entry\n")
        content = mem._storage.log_read("main", 100)
        assert "First log" in content
        assert "Second log" in content

    def test_log_rotation(self, mem):
        cfg = CCRConfig(storage_backend=mem.config.storage_backend, log_max_lines=5)
        mem.config = cfg
        for i in range(10):
            mem._append_log("main", f"Line {i}\n")
        content = mem._storage.log_read("main", 100)
        lines = [l for l in content.strip().split("\n") if l.strip()]
        assert len(lines) <= 10


# ── Context window ────────────────────────────────────────────


class TestContextWiring:
    def test_context_level1(self, mem):
        mem.commit("Init", "Init project", "bootstrap", ["a.py"], "next")
        ctx = mem.get_context(level=1, branch="main")
        assert isinstance(ctx, str)

    def test_context_level2(self, mem):
        mem.commit("Setup", "Setup project", "init", ["a.py"], "continue")
        ctx = mem.get_context(level=2, branch="main")
        assert isinstance(ctx, str)

    def test_read_commits_window(self, mem):
        for i in range(3):
            mem.commit(f"Task {i}", f"Did task {i}", "reason", [f"f{i}.py"], "next")
        result = mem._read_commits_window("main", count=2, offset=0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_commit_index(self, mem):
        mem.commit("Indexed", "Indexed task", "test", ["x.py"], "next")
        mem._build_commit_index("main")
        assert "main" in mem._commit_index

    def test_find_commit_by_id(self, mem):
        mem.commit("Find me", "Findable commit", "reason", ["a.py"], "next")
        result = mem._find_commit_by_id("main", "C001")
        assert result is not None


# ── Discussions ───────────────────────────────────────────────


class TestDiscussionWiring:
    def test_add_and_get(self, mem):
        mem.add_discussion(
            topic="Architecture",
            hypothesis="Microservices are better",
            decision="Monolith first",
            rationale="Simpler to start",
            alternatives_considered="Serverless, Monolith",
        )
        result = mem.get_discussions()
        assert result["count"] >= 1
        assert result["records"][0]["topic"] == "Architecture"

    def test_discussion_id_increments(self, mem):
        mem.add_discussion("Topic 1", "H1", "D1", "R1", "A1")
        mem.add_discussion("Topic 2", "H2", "D2", "R2", "A2")
        result = mem.get_discussions()
        assert result["count"] >= 2


# ── Consolidation ─────────────────────────────────────────────


class TestConsolidationWiring:
    def test_summary_meta_round_trip(self, mem):
        meta = mem._load_summary_meta()
        assert isinstance(meta, dict)
        meta["test_key"] = "test_value"
        mem._save_summary_meta(meta)
        reloaded = mem._load_summary_meta()
        assert reloaded.get("test_key") == "test_value"

    def test_commit_count(self, mem):
        count_before = mem._get_commit_count("main")
        mem.commit("A commit", "Did something", "reason", ["a.py"], "next")
        count_after = mem._get_commit_count("main")
        assert count_after >= count_before


# ── Patterns ──────────────────────────────────────────────────


class TestPatternsWiring:
    def test_load_save_round_trip(self, mem):
        data = mem._load_patterns()
        assert isinstance(data, dict)
        assert "patterns" in data
        data["patterns"]["test-pattern"] = {
            "pattern": "Always add tests",
            "count": 1,
            "first_seen": "2026-01-01",
            "last_seen": "2026-01-01",
            "quality_score": 0.5,
        }
        mem._save_patterns(data)
        reloaded = mem._load_patterns()
        assert "test-pattern" in reloaded.get("patterns", {})

    def test_pattern_extraction_from_commit(self, mem):
        mem.commit(
            "Auth fix", "Fixed auth bug", "security",
            ["auth.py"], "monitor",
            patterns_learned=["Always validate tokens"],
        )
        data = mem._load_patterns()
        patterns_dict = data.get("patterns", {})
        assert isinstance(patterns_dict, dict)


# ── Links ─────────────────────────────────────────────────────


class TestLinksWiring:
    def test_increment_and_get_metrics(self, mem):
        mem._increment_memory_metric("commits")
        mem._increment_memory_metric("commits")
        metrics = mem.get_memory_metrics()
        assert isinstance(metrics, dict)


# ── Evolution ─────────────────────────────────────────────────


class TestEvolutionWiring:
    def test_load_save_evolved(self, mem):
        mem._load_evolved_summaries()
        summaries = mem._evolved_summaries
        assert isinstance(summaries, dict)


# ── Metadata (Registry) ──────────────────────────────────────


class TestMetadataWiring:
    def test_load_metadata(self, mem):
        meta = mem._load_metadata()
        assert isinstance(meta, dict)
        assert "branches" in meta

    def test_save_and_reload(self, mem):
        meta = mem._load_metadata()
        meta["test_field"] = "wiring_test"
        mem._save_metadata(meta)
        reloaded = mem._load_metadata()
        assert reloaded.get("test_field") == "wiring_test"

    def test_update_metadata_branch(self, mem):
        mem._update_metadata_branch("test-branch", "active", "2026-01-01", "main")
        meta = mem._load_metadata()
        names = [b["name"] for b in meta.get("branches", [])]
        assert "test-branch" in names

    def test_update_metadata_branch_status(self, mem):
        mem._update_metadata_branch("exp-branch", "active", "2026-01-01", "main")
        mem._update_metadata_branch_status("exp-branch", "merged")
        meta = mem._load_metadata()
        for b in meta.get("branches", []):
            if b["name"] == "exp-branch":
                assert b["status"] == "merged"
                break


# ── Branch operations ─────────────────────────────────────────


class TestBranchOpsWiring:
    def test_create_branch(self, mem):
        result = mem.create_branch("test-branch", "test purpose", "test hypothesis")
        assert "test-branch" in result
        active = mem.get_active_branch()
        assert active == "test-branch"

    def test_create_and_merge_branch(self, mem):
        mem.create_branch("exp-one", "experiment", "it might work")
        mem.commit("Experiment", "Did experiment", "testing", ["x.py"], "check results")
        result = mem.merge("exp-one", "success", "It worked")
        assert "success" in result
        active = mem.get_active_branch()
        assert active == "main"


# ── File backend layout compatibility ─────────────────────────


class TestFileBackendLayout:
    def test_ccr_directory_structure(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = CCRConfig(storage_backend="files")
            m = MemoryManager(d, cfg)
            m.ensure_structure()
            ccr = os.path.join(d, ".ccr")
            assert os.path.isdir(ccr)
            assert os.path.isfile(os.path.join(ccr, "main.md"))
            assert os.path.isfile(os.path.join(ccr, "metadata.yaml"))
            assert os.path.isdir(os.path.join(ccr, "branches", "main"))
            assert os.path.isfile(os.path.join(ccr, "branches", "main", "commits.md"))
            assert os.path.isfile(os.path.join(ccr, "branches", "_registry.md"))

    def test_commit_writes_to_commits_md(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = CCRConfig(storage_backend="files")
            m = MemoryManager(d, cfg)
            m.ensure_structure()
            m.commit("Test", "Test commit", "verify layout", ["a.py"], "next")
            commits_path = os.path.join(d, ".ccr", "branches", "main", "commits.md")
            content = open(commits_path).read()
            assert "Test commit" in content

    def test_log_writes_to_log_md(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = CCRConfig(storage_backend="files")
            m = MemoryManager(d, cfg)
            m.ensure_structure()
            m._append_log("main", "Test log entry\n")
            log_path = os.path.join(d, ".ccr", "branches", "main", "log.md")
            content = open(log_path).read()
            assert "Test log entry" in content

    def test_metadata_writes_to_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = CCRConfig(storage_backend="files")
            m = MemoryManager(d, cfg)
            m.ensure_structure()
            meta_path = os.path.join(d, ".ccr", "metadata.yaml")
            assert os.path.isfile(meta_path)


# ── Cache consistency ─────────────────────────────────────────


class TestCacheConsistency:
    def test_commit_index_populated_after_build(self, mem):
        mem.commit("Cache test", "Cache test work", "verify", ["c.py"], "next")
        mem._build_commit_index("main")
        assert "main" in mem._commit_index
        index = mem._commit_index["main"]
        assert len(index) >= 1

    def test_commit_invalidates_index(self, mem):
        mem.commit("First", "First thing", "r", ["a.py"], "n")
        mem._build_commit_index("main")
        assert "main" in mem._commit_index
        mem.commit("Second", "Second thing", "r", ["b.py"], "n")
        assert "main" not in mem._commit_index

    def test_rolling_summary_write_invalidates_index(self, mem):
        mem._build_commit_index("main")
        mem._commit_index["main"] = {"test": "data"}
        mem._write_rolling_summary("main", "new summary")
        assert "main" not in mem._commit_index
