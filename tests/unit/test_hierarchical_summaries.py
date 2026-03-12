"""Tests for Multi-Level Hierarchical Summaries (TiMem-adapted).

Tests session summaries (TiMem L2), phase summaries (TiMem L3-4),
project overview (TiMem L5), context integration, and edge cases.
"""

import os
import re
import tempfile

import pytest

from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig


@pytest.fixture
def project_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def memory(project_dir):
    mem = MemoryManager(project_dir, CCRConfig())
    mem.ensure_structure()
    return mem


@pytest.fixture
def memory_interval3(project_dir):
    """Memory with session_summary_interval=3 for faster trigger."""
    config = CCRConfig(session_summary_interval=3)
    mem = MemoryManager(project_dir, config)
    mem.ensure_structure()
    return mem


def _make_commits(memory, n, prefix="Task"):
    """Helper: create n commits with varied content."""
    for i in range(1, n + 1):
        memory.commit(
            title=f"{prefix} {i}",
            what=f"Implemented feature {i} for {prefix}",
            why=f"Required for milestone {i}",
            files_changed=[f"src/mod{i}.py", f"tests/test_mod{i}.py"],
            next_step=f"Continue to {prefix} {i + 1}",
            admission_threshold=1.0,  # Disable A-MAC for test predictability
        )


# ===========================================================================
# Bootstrap Tests
# ===========================================================================


class TestHierarchicalBootstrap:
    def test_ensure_structure_creates_summaries_dir(self, project_dir):
        mem = MemoryManager(project_dir)
        mem.ensure_structure()
        assert os.path.isdir(os.path.join(project_dir, ".ccr", "summaries"))

    def test_ensure_structure_creates_summary_meta(self, project_dir):
        mem = MemoryManager(project_dir)
        mem.ensure_structure()
        assert os.path.isfile(os.path.join(project_dir, ".ccr", "summary_meta.yaml"))

    def test_summary_meta_default_content(self, memory):
        meta = memory._load_summary_meta()
        assert meta["version"] == 1
        assert "session" in meta
        assert "phase" in meta
        assert "overview" in meta


# ===========================================================================
# Session Summary Tests (TiMem L2)
# ===========================================================================


class TestSessionSummary:
    def test_session_summary_generated_after_n_commits(self, memory):
        """5 commits (default interval) should trigger session summary."""
        _make_commits(memory, 5)
        summaries = memory._read_session_summaries("main", count=5)
        assert len(summaries) >= 1
        assert summaries[0]["id"] == "S001"

    def test_session_summary_not_generated_before_threshold(self, memory):
        """3 commits should not trigger session summary (default interval=5)."""
        _make_commits(memory, 3)
        summaries = memory._read_session_summaries("main", count=5)
        assert len(summaries) == 0

    def test_session_summary_captures_all_commits(self, memory):
        """Session summary should reference all commits in the window."""
        _make_commits(memory, 5)
        summaries = memory._read_session_summaries("main", count=1)
        assert len(summaries) == 1
        s = summaries[0]
        # Should have commit range
        assert s["commits"]  # e.g., "C001-C005"
        # Should have files
        assert s["files"]
        # Should have accomplished text
        assert s["accomplished"]

    def test_session_summary_deduplication(self, memory):
        """Nearly identical commits should produce compact summary."""
        for i in range(5):
            memory.commit(
                title=f"Fix auth bug iteration {i}",
                what="Fixed the authentication bypass vulnerability",
                why="Security patch required",
                files_changed=["auth.py"],
                next_step="Deploy to production",
                admission_threshold=1.0,
            )
        summaries = memory._read_session_summaries("main", count=1)
        assert len(summaries) >= 1
        # Accomplished should be compact due to dedup
        accomplished = summaries[0]["accomplished"]
        # Should not repeat 5 times
        assert accomplished.count("authentication") <= 2

    def test_session_summary_sequential(self, memory):
        """10 commits should generate 2 session summaries."""
        _make_commits(memory, 10)
        summaries = memory._read_session_summaries("main", count=10)
        assert len(summaries) >= 2

    def test_session_summary_on_branch(self, memory):
        """Session summaries on a branch go to that branch's summaries.md."""
        memory.create_branch("try-feature", "Test branch summaries", "They should work")
        _make_commits(memory, 5, prefix="Branch work")
        branch_summaries = memory._read_session_summaries("try-feature", count=5)
        assert len(branch_summaries) >= 1
        assert "try-feature" in branch_summaries[0].get("branch", "")

    def test_session_summary_respects_config(self, memory_interval3):
        """Session summary should respect custom interval."""
        _make_commits(memory_interval3, 3)
        summaries = memory_interval3._read_session_summaries("main", count=5)
        assert len(summaries) >= 1

    def test_session_summary_meta_updated(self, memory):
        """Summary meta should track last commit count after generation."""
        _make_commits(memory, 5)
        meta = memory._load_summary_meta()
        session_meta = meta.get("session", {}).get("main", {})
        assert session_meta.get("last_commit_count", 0) >= 5

    def test_session_summary_empty_commits(self, memory):
        """Commits with minimal content should still produce valid summary."""
        for i in range(5):
            memory.commit(
                title=f"C{i}",
                what="",
                why="",
                files_changed=[],
                next_step="",
                admission_threshold=1.0,
            )
        # Should not crash, may or may not generate summary
        summaries = memory._read_session_summaries("main", count=5)
        # If generated, it should be valid
        if summaries:
            assert summaries[0]["id"].startswith("S")

    def test_generate_session_summary_directly(self, memory):
        """Direct call to _generate_session_summary should work."""
        _make_commits(memory, 5)
        result = memory._generate_session_summary("main")
        assert "[S" in result
        assert "Session Summary" in result

    def test_generate_session_summary_insufficient_commits(self, memory):
        """Should return empty string with fewer than 3 commits."""
        _make_commits(memory, 2)
        result = memory._generate_session_summary("main")
        assert result == ""


# ===========================================================================
# Phase Summary Tests (TiMem L3-L4)
# ===========================================================================


class TestPhaseSummary:
    def test_phase_summary_on_merge(self, memory):
        """Merging a branch should generate a phase summary."""
        _make_commits(memory, 2)  # Some main commits first
        memory.create_branch("try-parser", "New parser", "Should be faster")
        _make_commits(memory, 5, prefix="Parser")
        memory.merge("try-parser", "success", "Parser is 2x faster")
        phases = memory._read_phase_summaries(count=5)
        assert len(phases) >= 1
        assert phases[0]["id"] == "P001"

    def test_phase_summary_includes_branch_metadata(self, memory):
        """Phase summary should contain branch purpose and conclusion."""
        memory.create_branch("try-cache", "Add caching layer", "Should reduce latency")
        _make_commits(memory, 3, prefix="Cache")
        memory.merge("try-cache", "success", "Reduced latency by 40%")
        phases = memory._read_phase_summaries(count=1)
        assert len(phases) >= 1
        p = phases[0]
        assert p["scope"] == "try-cache"
        assert "success" in p["outcome"] or "Reduced" in p["outcome"]

    def test_phase_summary_includes_session_summaries(self, memory):
        """Phase summary should aggregate session summary data."""
        memory.create_branch("big-feature", "Large feature", "Multi-session work")
        _make_commits(memory, 10, prefix="Feature")  # Generates 2 session summaries
        memory.merge("big-feature", "success", "Feature complete")
        phases = memory._read_phase_summaries(count=1)
        assert len(phases) >= 1

    def test_phase_summary_manual_trigger(self, memory):
        """Manual phase summary generation should work."""
        _make_commits(memory, 8)
        result = memory.generate_phase_summary(trigger="manual")
        assert "Phase Summary" in result
        assert "[P001]" in result

    def test_phase_summary_sequential(self, memory):
        """Multiple merges produce sequential phase summaries."""
        memory.create_branch("branch-a", "First feature", "Test A")
        _make_commits(memory, 3, prefix="A")
        memory.merge("branch-a", "success", "A done")

        memory.create_branch("branch-b", "Second feature", "Test B")
        _make_commits(memory, 3, prefix="B")
        memory.merge("branch-b", "success", "B done")

        phases = memory._read_phase_summaries(count=10)
        assert len(phases) >= 2
        ids = [p["id"] for p in phases]
        assert "P001" in ids or "P002" in ids

    def test_phase_summary_no_duplicate_commit(self, memory):
        """Phase summary should not create duplicate commits in commits.md."""
        memory.create_branch("test-branch", "Test", "Test")
        _make_commits(memory, 3, prefix="Test")
        commit_count_before = memory._get_commit_count("test-branch")
        memory.merge("test-branch", "success", "Done")
        # Commits on the branch should not increase from phase summary
        # (phase summary goes to phases.md, not commits.md)
        commit_count_after = memory._get_commit_count("test-branch")
        assert commit_count_after == commit_count_before


# ===========================================================================
# Project Overview Tests (TiMem L5)
# ===========================================================================


class TestProjectOverview:
    def test_consolidation_prompt_returned(self, memory):
        """get_consolidation_prompt should return a prompt string."""
        _make_commits(memory, 5)
        memory.generate_phase_summary(trigger="manual")
        prompt = memory.get_consolidation_prompt(tier="project")
        assert "Please generate a project overview" in prompt
        assert "Phase Summaries" in prompt

    def test_overview_save(self, memory):
        """save_overview should persist to overview.md."""
        memory.save_overview("This is a test project about memory management.")
        content = memory._read_file(memory._get_overview_path())
        assert "test project" in content

    def test_overview_in_context_level_1(self, memory):
        """Saved overview should appear in get_context(level=1)."""
        memory.save_overview("CCR is a memory system for Claude Code.")
        ctx = memory.get_context(level=1)
        assert "CCR is a memory system" in ctx

    def test_overview_backward_compat(self, memory):
        """get_context should work when overview.md does not exist."""
        ctx = memory.get_context(level=1)
        assert "Project Overview" in ctx  # main.md still included

    def test_consolidation_session_tier(self, memory):
        """tier='session' should generate mechanically."""
        _make_commits(memory, 5)
        result = memory.get_consolidation_prompt(tier="session")
        assert "Session Summary" in result

    def test_consolidation_phase_tier(self, memory):
        """tier='phase' should generate mechanically."""
        _make_commits(memory, 5)
        result = memory.get_consolidation_prompt(tier="phase")
        assert "Phase Summary" in result

    def test_overview_meta_updated(self, memory):
        """Saving overview should update summary_meta.yaml."""
        memory.save_overview("Test overview.")
        meta = memory._load_summary_meta()
        assert meta["overview"]["last_generated"] is not None


# ===========================================================================
# Context Integration Tests
# ===========================================================================


class TestContextIntegration:
    def test_context_level_2_includes_session_summary(self, memory):
        """Level 2 context should include session summary if available."""
        _make_commits(memory, 5)
        ctx = memory.get_context(level=2)
        assert "Session Summary" in ctx

    def test_context_level_3_includes_phase_summary(self, memory):
        """Level 3 should include phase history."""
        _make_commits(memory, 5)
        memory.generate_phase_summary(trigger="manual")
        ctx = memory.get_context(level=3)
        assert "Phase" in ctx

    def test_context_level_4_includes_multiple_summaries(self, memory):
        """Level 4 should include sessions + phases + more raw commits."""
        _make_commits(memory, 10)
        memory.generate_phase_summary(trigger="manual")
        ctx = memory.get_context(level=4)
        # Should have commits and summaries
        assert "Recent Commits" in ctx

    def test_context_without_summaries_unchanged(self, memory):
        """Fresh project with 2 commits should still work (backward compat)."""
        _make_commits(memory, 2)
        ctx = memory.get_context(level=2)
        assert "Project Overview" in ctx
        assert "Recent Commits" in ctx

    def test_get_summaries_all(self, memory):
        """get_summaries(tier='all') should return structured output."""
        _make_commits(memory, 5)
        memory.generate_phase_summary(trigger="manual")
        memory.save_overview("Test overview content.")
        result = memory.get_summaries(tier="all", count=5)
        assert "Session Summaries" in result
        assert "Phase Summaries" in result
        assert "Project Overview" in result

    def test_get_summaries_empty(self, memory):
        """get_summaries should handle empty state gracefully."""
        result = memory.get_summaries(tier="all", count=5)
        assert "No summaries" in result or "No session" in result or "No phase" in result


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    def test_single_commit_no_crash(self, memory):
        """One commit should not crash anything."""
        _make_commits(memory, 1)
        summaries = memory._read_session_summaries("main", count=5)
        assert len(summaries) == 0

    def test_empty_branch_merge(self, memory):
        """Branch with zero commits should merge without crashing."""
        memory.create_branch("empty-branch", "Empty test", "Nothing to do")
        memory.merge("empty-branch", "failure", "Nothing was done")
        # Phase summary might be minimal but shouldn't crash
        phases = memory._read_phase_summaries(count=5)
        # May or may not generate a phase depending on data

    def test_branch_with_no_session_summaries(self, memory):
        """Branch with <5 commits should still merge and create phase summary."""
        memory.create_branch("small-branch", "Quick fix", "One change")
        _make_commits(memory, 2, prefix="Fix")
        memory.merge("small-branch", "success", "Fixed it")
        phases = memory._read_phase_summaries(count=5)
        assert len(phases) >= 1  # Phase from raw commits fallback

    def test_summary_meta_missing(self, memory):
        """Deleting summary_meta.yaml should be handled gracefully."""
        meta_path = memory._get_summary_meta_path()
        if os.path.exists(meta_path):
            os.remove(meta_path)
        # Should recreate with defaults
        meta = memory._load_summary_meta()
        assert meta["version"] == 1

    def test_summaries_md_corruption(self, memory):
        """Corrupt summaries.md should not break commit()."""
        path = memory._get_summaries_path("main")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("CORRUPT DATA\n###NOT VALID###")
        # This should not crash
        _make_commits(memory, 5)

    def test_commit_count_accuracy(self, memory):
        """_get_commit_count should return accurate count."""
        assert memory._get_commit_count("main") == 0
        _make_commits(memory, 7)
        assert memory._get_commit_count("main") == 7

    def test_session_summary_id_sequence(self, memory):
        """Session summary IDs should increment correctly."""
        _make_commits(memory, 10)
        summaries = memory._read_session_summaries("main", count=10)
        if len(summaries) >= 2:
            ids = [s["id"] for s in summaries]
            # Should be in descending order (most recent first)
            nums = [int(re.search(r"\d+", id).group()) for id in ids]
            assert nums == sorted(nums, reverse=True)

    def test_phase_summary_id_sequence(self, memory):
        """Phase summary IDs should increment correctly."""
        memory.create_branch("br-a", "A", "A")
        _make_commits(memory, 3, prefix="A")
        memory.merge("br-a", "success", "Done A")

        memory.create_branch("br-b", "B", "B")
        _make_commits(memory, 3, prefix="B")
        memory.merge("br-b", "success", "Done B")

        phases = memory._read_phase_summaries(count=10)
        if len(phases) >= 2:
            ids = [p["id"] for p in phases]
            nums = [int(re.search(r"\d+", id).group()) for id in ids]
            assert nums == sorted(nums, reverse=True)


# ===========================================================================
# MCP Tool Integration Tests
# ===========================================================================


class TestMCPToolIntegration:
    """Test the MCP tool functions via memory manager methods directly."""

    def test_consolidation_prompt_session(self, memory):
        _make_commits(memory, 5)
        result = memory.get_consolidation_prompt(tier="session")
        assert result  # Should return something

    def test_consolidation_prompt_phase(self, memory):
        _make_commits(memory, 5)
        result = memory.get_consolidation_prompt(tier="phase")
        assert result

    def test_consolidation_prompt_project(self, memory):
        result = memory.get_consolidation_prompt(tier="project")
        assert "Please generate a project overview" in result

    def test_save_and_retrieve_overview(self, memory):
        memory.save_overview("Test project: builds rockets")
        result = memory.get_summaries(tier="project")
        assert "builds rockets" in result

    def test_summaries_session_tier(self, memory):
        _make_commits(memory, 5)
        result = memory.get_summaries(tier="session", count=3)
        assert "Session" in result

    def test_summaries_phase_tier(self, memory):
        _make_commits(memory, 5)
        memory.generate_phase_summary(trigger="manual")
        result = memory.get_summaries(tier="phase", count=3)
        assert "Phase" in result
