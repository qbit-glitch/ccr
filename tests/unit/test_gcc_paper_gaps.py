"""Tests for GCC paper gap closures — all missing/partial features.

Covers:
1. LLM-based rolling summary (mock sub_client)
2. Fallback to concatenation when no sub_client
3. CONTEXT --log (log_window parameter)
4. CONTEXT --metadata (metadata_segment parameter)
5. OTA slice reference in commits
6. F_merge integrates branch summary into main
7. Execution trace union on merge
8. Git commit integration (mock subprocess)
9. Current Focus update on commit
10. _read_log_window with OTA entries
11. _get_ota_slice_since_last_commit
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

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


class TestLLMRollingSummary:
    """1. LLM-based rolling summary with mock sub_client."""

    def test_llm_summary_used_when_sub_client_set(self, mem):
        mock_client = MagicMock()
        mock_client.completion.return_value = "Compressed summary: auth module added with tests planned."
        mem.set_sub_client(mock_client)

        mem.commit("Add auth", "Added auth module", "security", ["auth.py"], "add tests")

        summary = mem._get_rolling_summary("main")
        assert "Compressed summary" in summary
        mock_client.completion.assert_called_once()

    def test_llm_summary_prompt_contains_previous_and_new(self, mem):
        mock_client = MagicMock()
        mock_client.completion.return_value = "Updated summary with both contributions."
        mem.set_sub_client(mock_client)

        mem.commit("First", "Did A", "reason A", [], "next A")
        # Reset mock to check second call
        mock_client.reset_mock()
        mock_client.completion.return_value = "Combined summary of A and B."

        mem.commit("Second", "Did B", "reason B", [], "next B")

        call_args = mock_client.completion.call_args[0][0]
        prompt = call_args[0]["content"]
        assert "Previous summary" in prompt
        assert "New contribution" in prompt

    def test_llm_summary_truncated_to_1500(self, mem):
        mock_client = MagicMock()
        mock_client.completion.return_value = "x" * 2000
        mem.set_sub_client(mock_client)

        mem.commit("Long", "stuff", "reason", [], "next")
        summary = mem._get_rolling_summary("main")
        assert len(summary) <= 1500

    def test_llm_summary_fallback_on_short_response(self, mem):
        mock_client = MagicMock()
        mock_client.completion.return_value = "short"  # <= 10 chars
        mem.set_sub_client(mock_client)

        mem.commit("Test", "Did something", "reason", [], "next")
        summary = mem._get_rolling_summary("main")
        # Should fall back to concatenation
        assert "Did something" in summary
        assert "(because: reason)" in summary

    def test_llm_summary_fallback_on_exception(self, mem):
        mock_client = MagicMock()
        mock_client.completion.side_effect = RuntimeError("API error")
        mem.set_sub_client(mock_client)

        mem.commit("Test", "Did something", "reason", [], "next")
        summary = mem._get_rolling_summary("main")
        # Should fall back to concatenation
        assert "Did something" in summary


class TestConcatenationFallback:
    """2. Fallback to concatenation when no sub_client."""

    def test_no_sub_client_uses_concatenation(self, mem):
        assert mem.sub_client is None
        mem.commit("First", "Did A", "reason A", [], "next A")
        mem.commit("Second", "Did B", "reason B", [], "next B")

        summary = mem._get_rolling_summary("main")
        assert "Did A" in summary
        assert "Did B" in summary
        assert ";" in summary  # concatenation marker

    def test_concatenation_caps_at_1500(self, mem):
        for i in range(50):
            mem.commit(f"Commit {i}", f"Did thing {i} " * 10, "reason", [], "next")
        summary = mem._get_rolling_summary("main")
        assert len(summary) <= 1600  # small overhead allowed
        # Structured truncation preserves first entry (project context)
        # and last 3 entries — no longer starts with "..."
        assert "Did thing 0" in summary  # first entry preserved


class TestContextLogWindow:
    """3. CONTEXT --log (log_window parameter)."""

    def test_log_window_returns_recent_entries(self, mem):
        # Create some OTA log entries
        for i in range(5):
            mem.log_ota(
                f"tool-{i}",
                observation=f"Observed {i}",
                thought=f"Thought {i}",
                action=f"Action {i}",
            )

        ctx = mem.get_context(level=1, log_window=3)
        assert "Execution Log" in ctx
        assert "last 3 entries" in ctx

    def test_log_window_zero_no_log(self, mem):
        mem.log_ota("tool", observation="obs", thought="t", action="a")
        ctx = mem.get_context(level=1, log_window=0)
        assert "Execution Log" not in ctx

    def test_log_window_empty_log(self, mem):
        ctx = mem.get_context(level=1, log_window=5)
        # Empty log should not add section
        assert "Execution Log" not in ctx


class TestContextMetadataSegment:
    """4. CONTEXT --metadata (metadata_segment parameter)."""

    def test_metadata_segment_file_tree(self, mem):
        mem.update_metadata_file_tree(["src/main.py", "src/utils.py"])
        ctx = mem.get_context(level=1, metadata_segment="file_tree")
        assert "Metadata: file_tree" in ctx
        assert "src/main.py" in ctx

    def test_metadata_segment_dependencies(self, mem):
        mem.update_metadata_dependencies(["numpy", "pandas"])
        ctx = mem.get_context(level=1, metadata_segment="dependencies")
        assert "Metadata: dependencies" in ctx
        assert "numpy" in ctx

    def test_metadata_segment_config(self, mem):
        mem.update_metadata_config(language="python")
        ctx = mem.get_context(level=1, metadata_segment="config")
        assert "Metadata: config" in ctx
        assert "python" in ctx

    def test_metadata_segment_nonexistent(self, mem):
        ctx = mem.get_context(level=1, metadata_segment="nonexistent_key")
        assert "Metadata: nonexistent_key" not in ctx

    def test_metadata_segment_scalar_value(self, mem):
        ctx = mem.get_context(level=1, metadata_segment="version")
        assert "Metadata: version" in ctx


class TestOTASliceInCommits:
    """5. OTA slice reference in commits."""

    def test_commit_references_ota_entries(self, mem):
        # Create OTA entries first
        mem.log_ota(
            "edit",
            observation="Found bug in auth",
            thought="Need fix",
            action="Fixed auth.py",
        )
        mem.log_ota(
            "test",
            observation="Running tests",
            thought="Verify fix",
            action="All tests pass",
        )

        mem.commit("Fix auth", "Fixed auth bug", "security", ["auth.py"], "deploy")

        commits_content = mem._read_file(mem._get_commits_path("main"))
        assert "OTA Trace" in commits_content
        assert "OTA-001" in commits_content

    def test_commit_no_ota_trace_when_no_entries(self, mem):
        mem.commit("First", "Did something", "reason", [], "next")
        commits_content = mem._read_file(mem._get_commits_path("main"))
        assert "OTA Trace" not in commits_content


class TestFMergeIntegration:
    """6. F_merge integrates branch summary into main."""

    def test_merge_integrates_branch_summary_into_main(self, mem):
        mem.create_branch("feature", "add search", "regex works")
        mem.commit("Search impl", "Added search function", "core feature", ["search.py"], "optimize")

        branch_summary_before = mem._get_rolling_summary("feature")
        assert "Added search function" in branch_summary_before

        mem.merge("feature", "success", "It worked")

        main_summary = mem._get_rolling_summary("main")
        assert "[From feature]" in main_summary
        assert "Added search function" in main_summary

    def test_merge_no_branch_summary_skips_integration(self, mem):
        mem.create_branch("empty-branch", "test", "test")
        # No commits on branch, so no rolling summary
        mem.merge("empty-branch", "failure", "Nothing done")

        main_summary = mem._get_rolling_summary("main")
        # Main summary should not have [From empty-branch] since branch had no summary
        assert "[From empty-branch]" not in main_summary

    def test_merge_caps_merged_summary_length(self, mem):
        mem.create_branch("big-branch", "test", "test")
        for i in range(30):
            mem.commit(f"Commit {i}", f"Long content {i} " * 10, "reason", [], "next")
        mem.merge("big-branch", "success", "done")

        main_summary = mem._get_rolling_summary("main")
        assert len(main_summary) <= 1600


class TestExecutionTraceUnion:
    """7. Execution trace union on merge."""

    def test_merge_copies_branch_log_to_main(self, mem):
        mem.create_branch("feature", "add auth", "JWT works")
        mem.log_ota(
            "edit",
            observation="Editing auth.py",
            thought="Adding JWT",
            action="Modified auth.py",
        )
        mem.commit("Auth", "Added JWT", "security", ["auth.py"], "test")

        mem.merge("feature", "success", "JWT works")

        main_log = mem._read_file(mem._get_log_path("main"))
        assert "[Merged from feature]" in main_log
        assert "Editing auth.py" in main_log

    def test_merge_empty_branch_log_no_provenance(self, mem):
        mem.create_branch("empty-log", "test", "test")
        # Write empty log explicitly
        mem._write_file(mem._get_log_path("empty-log"), "")

        mem.merge("empty-log", "partial", "nothing")

        main_log = mem._read_file(mem._get_log_path("main"))
        assert "[Merged from empty-log]" not in main_log


class TestGitCommitIntegration:
    """8. Git commit integration (mock subprocess)."""

    def test_git_commit_called_on_commit(self, mem):
        with patch("ccr.core.memory.subprocess") as mock_sp:
            # Simulate not a git repo (no .git dir)
            result = mem.commit("Test", "what", "why", [], "next")
            # Should not call subprocess since no .git dir
            mock_sp.run.assert_not_called()

    def test_git_commit_in_git_repo(self, mem):
        # Create fake .git directory
        os.makedirs(os.path.join(mem.project_root, ".git"), exist_ok=True)

        with patch("ccr.core.memory.subprocess") as mock_sp:
            mock_sp.run.return_value = MagicMock(returncode=0)
            mem.commit("Test", "what", "why", [], "next")
            # Should have called git add and git commit
            assert mock_sp.run.call_count == 2

    def test_git_commit_on_merge(self, mem):
        os.makedirs(os.path.join(mem.project_root, ".git"), exist_ok=True)

        mem.create_branch("feature", "test", "test")
        with patch("ccr.core.memory.subprocess") as mock_sp:
            mock_sp.run.return_value = MagicMock(returncode=0)
            mem.merge("feature", "success", "done")
            # Should have called git add + git commit
            assert mock_sp.run.call_count >= 2

    def test_git_commit_returns_false_on_failure(self, mem):
        os.makedirs(os.path.join(mem.project_root, ".git"), exist_ok=True)
        with patch("ccr.core.memory.subprocess") as mock_sp:
            mock_sp.run.return_value = MagicMock(returncode=1)
            result = mem._git_commit("test message")
            assert result is False

    def test_git_commit_returns_false_no_git_dir(self, mem):
        result = mem._git_commit("test message")
        assert result is False


class TestCurrentFocusUpdate:
    """9. Current Focus update on commit."""

    def test_commit_updates_current_focus(self, mem):
        mem.commit("Add auth", "Built auth module", "security", ["auth.py"], "write tests")

        main_md = mem._read_file(os.path.join(mem.ccr_root, "main.md"))
        assert "Add auth. Next: write tests" in main_md

    def test_current_focus_overwritten_on_second_commit(self, mem):
        mem.commit("First", "Did A", "reason", [], "do B")
        mem.commit("Second", "Did B", "reason", [], "do C")

        main_md = mem._read_file(os.path.join(mem.ccr_root, "main.md"))
        assert "Second. Next: do C" in main_md
        # First focus should be overwritten
        assert "First. Next: do B" not in main_md


class TestReadLogWindow:
    """10. _read_log_window with OTA entries."""

    def test_read_log_window_ota_entries(self, mem):
        for i in range(5):
            mem.log_ota(
                f"tool-{i}",
                observation=f"Obs {i}",
                thought=f"Think {i}",
                action=f"Act {i}",
            )

        result = mem._read_log_window("main", 2)
        assert result  # Should have content
        # Should contain entries from the end
        assert "Obs 4" in result or "Obs 3" in result

    def test_read_log_window_empty_log(self, mem):
        result = mem._read_log_window("main", 5)
        assert result == ""

    def test_read_log_window_table_format_fallback(self, mem):
        # Write plain table-format entries (no OTA markers)
        log_path = mem._get_log_path("main")
        lines = "\n".join([f"| 2024-01-01 | tool{i} | file{i} | OK |" for i in range(5)])
        mem._write_file(log_path, lines)

        result = mem._read_log_window("main", 2)
        assert "tool4" in result
        assert "tool3" in result


class TestGetOTASliceSinceLastCommit:
    """11. _get_ota_slice_since_last_commit."""

    def test_returns_recent_ota_refs(self, mem):
        mem.log_ota("edit", observation="Edit file A", thought="t", action="a")
        mem.log_ota("test", observation="Run tests", thought="t", action="a")

        result = mem._get_ota_slice_since_last_commit("main")
        assert "OTA-001" in result
        assert "OTA-002" in result
        assert "Edit file A" in result

    def test_returns_empty_when_no_log(self, mem):
        result = mem._get_ota_slice_since_last_commit("main")
        assert result == ""

    def test_caps_at_max_entries(self, mem):
        for i in range(10):
            mem.log_ota(f"tool-{i}", observation=f"Obs {i}", thought="t", action="a")

        result = mem._get_ota_slice_since_last_commit("main", max_entries=3)
        # Should only have last 3
        assert "OTA-010" in result
        assert "OTA-009" in result
        assert "OTA-008" in result
        assert "OTA-001" not in result

    def test_truncates_long_observations(self, mem):
        long_obs = "A" * 200
        mem.log_ota("tool", observation=long_obs, thought="t", action="a")

        result = mem._get_ota_slice_since_last_commit("main")
        # Observation should be truncated to 80 chars
        assert len(result.split(": ", 1)[1]) <= 80
