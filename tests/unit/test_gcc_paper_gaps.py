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


class TestONNXAdmissionSimilarity:
    """Tests for ONNX cosine similarity in A-MAC admission control."""

    def test_onnx_path_uses_dot_product_not_jaccard(self, mem):
        """When ONNX available, raw_sim uses cosine not Jaccard."""
        import numpy as np

        # Pre-create a commit so there is a recent commit to compare against
        mem.commit("First commit", "Did initial work", "bootstrap", ["a.py"], "next")
        commits = mem._parse_recent_commit_data("main", k=5)
        assert len(commits) >= 1
        first_id = commits[0].get("id")

        # Two orthogonal unit vectors — cosine = 0.0
        vec_new = np.zeros(384, dtype=np.float32)
        vec_new[0] = 1.0
        vec_old = np.zeros(384, dtype=np.float32)
        vec_old[1] = 1.0

        mock_model = MagicMock()
        mock_model.embed_query.return_value = vec_new

        with patch("ccr.core.memory.MemoryManager._load_commit_embeddings") as mock_load, \
             patch("ccr.context.embeddings.get_embedding_model", return_value=mock_model):
            mock_load.return_value = {first_id: vec_old} if first_id else {}
            result = mem.compute_admission_score(
                "main", "Second commit", "More work", "same reason",
                ["a.py"], "continue"
            )

        # With orthogonal vectors cosine=0 → high novelty; Jaccard with shared
        # file "a.py" would give file_sim=1.0 → raw_sim≥0.5 → novelty≤0.5.
        # ONNX path should yield novelty close to 1.0.
        assert result["novelty"] > 0.9

    def test_jaccard_fallback_when_onnx_unavailable(self, mem):
        """When get_embedding_model() returns None, Jaccard fallback runs without crash."""
        mem.commit("First", "Did A", "reason A", ["x.py"], "next")

        with patch("ccr.context.embeddings.get_embedding_model", return_value=None):
            result = mem.compute_admission_score(
                "main", "Second", "Did B", "reason B", ["y.py"], "continue"
            )

        # Must not crash and must return a valid score
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_onnx_fallback_on_exception(self, mem):
        """When embed_query() raises, Jaccard fallback runs silently."""
        mem.commit("First", "Did A", "reason A", ["x.py"], "next")

        mock_model = MagicMock()
        mock_model.embed_query.side_effect = RuntimeError("ONNX inference error")

        with patch("ccr.context.embeddings.get_embedding_model", return_value=mock_model):
            result = mem.compute_admission_score(
                "main", "Second", "Did B", "reason B", ["y.py"], "continue"
            )

        # Exception must be swallowed — score returned normally
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_commit_embeddings_preloaded_once(self, mem):
        """_load_commit_embeddings called at most once per admission check, not per commit."""
        import numpy as np

        # Create two prior commits so the loop iterates more than once
        mem.commit("C1", "Work A", "reason", ["a.py"], "next")
        mem.commit("C2", "Work B", "reason", ["b.py"], "next")

        vec_new = np.zeros(384, dtype=np.float32)
        vec_new[0] = 1.0

        mock_model = MagicMock()
        mock_model.embed_query.return_value = vec_new

        with patch("ccr.context.embeddings.get_embedding_model", return_value=mock_model), \
             patch.object(mem, "_load_commit_embeddings", wraps=mem._load_commit_embeddings) as mock_load:
            mem.compute_admission_score(
                "main", "C3", "Work C", "reason", ["c.py"], "next"
            )

        # Must be called at most once regardless of how many recent commits exist
        assert mock_load.call_count <= 1

    def test_cosine_sim_reduces_false_positives(self, mem):
        """ONNX path can detect commits as different even when they share filenames."""
        import numpy as np

        mem.commit("First", "Add feature", "need it", ["shared.py"], "next")
        commits = mem._parse_recent_commit_data("main", k=5)
        first_id = commits[0].get("id")

        # Orthogonal vectors → cosine = 0.0 even though both commits touch shared.py
        vec_new = np.zeros(384, dtype=np.float32)
        vec_new[0] = 1.0
        vec_old = np.zeros(384, dtype=np.float32)
        vec_old[1] = 1.0

        mock_model = MagicMock()
        mock_model.embed_query.return_value = vec_new

        with patch("ccr.core.memory.MemoryManager._load_commit_embeddings") as mock_load, \
             patch("ccr.context.embeddings.get_embedding_model", return_value=mock_model):
            mock_load.return_value = {first_id: vec_old} if first_id else {}
            result = mem.compute_admission_score(
                "main", "Second", "Refactor feature", "clean up", ["shared.py"], "done"
            )

        # Pure Jaccard on the shared file would give file_sim=1.0 → raw_sim=0.5+
        # ONNX cosine of orthogonal vectors = 0.0 → novelty near 1.0
        assert result["novelty"] > 0.9


class TestAMEMMemoryEvolution:
    """A-MEM §3.3 Eq.7 — mutable memory evolution (EvolvedSummary overlay)."""

    def _make_commit_dict(self, cid: str, title: str, what: str, why: str) -> dict:
        return {"id": cid, "title": title, "what": what, "why": why}

    def test_evolve_commit_summary_calls_llm_and_returns_evolved(self, mem):
        """With sub_client set, _evolve_commit_summary returns an EvolvedSummary."""
        mock_client = MagicMock()
        mock_client.completion.return_value = (
            '{"evolved_what": "Auth module now integrates JWT.", '
            '"evolution_reason": "New commit added JWT support."}'
        )
        mem.set_sub_client(mock_client)

        existing = self._make_commit_dict("C001", "Add auth", "Added basic auth", "security")
        new_c = self._make_commit_dict("C002", "Add JWT", "Added JWT tokens", "security")
        result = mem._evolve_commit_summary(existing, new_c)

        assert result is not None
        assert result.commit_id == "C001"
        assert "JWT" in result.evolved_what
        assert result.source_commit_id == "C002"
        assert result.original_what == "Added basic auth"
        mock_client.completion.assert_called_once()

    def test_evolve_commit_summary_returns_none_without_sub_client(self, mem):
        """Without sub_client, _evolve_commit_summary returns None immediately."""
        assert mem.sub_client is None
        existing = self._make_commit_dict("C001", "Title", "What A", "Why A")
        new_c = self._make_commit_dict("C002", "Title2", "What B", "Why B")
        result = mem._evolve_commit_summary(existing, new_c)
        assert result is None

    def test_evolve_commit_summary_returns_none_on_error(self, mem):
        """When sub_client raises, _evolve_commit_summary swallows and returns None."""
        mock_client = MagicMock()
        mock_client.completion.side_effect = RuntimeError("LLM timeout")
        mem.set_sub_client(mock_client)

        existing = self._make_commit_dict("C001", "Title", "What A", "Why A")
        new_c = self._make_commit_dict("C002", "Title2", "What B", "Why B")
        result = mem._evolve_commit_summary(existing, new_c)
        assert result is None

    def test_trigger_memory_evolution_evolves_semantic_links(self, mem):
        """Semantic links with score > 0.5 trigger evolution of linked commit."""
        # Create two real commits so _find_commit_by_id can locate them
        mem.commit("First", "Did initial work", "bootstrap", ["a.py"], "next")
        mem.commit("Second", "Did follow-up work", "extend", ["a.py"], "done")

        recent = mem._parse_recent_commit_data("main", k=5)
        assert len(recent) >= 2
        old_id = recent[1]["id"]   # earlier commit
        new_id = recent[0]["id"]   # latest commit

        from ccr.core.types import CommitLink

        mock_client = MagicMock()
        mock_client.completion.return_value = (
            '{"evolved_what": "Initial work now enriched by follow-up.", '
            '"evolution_reason": "Related work arrived."}'
        )
        mem.set_sub_client(mock_client)

        links = [CommitLink(target=old_id, link_type="semantic", score=0.8)]
        mem._trigger_memory_evolution(new_id, links)

        # The older commit should now have an evolved summary
        evolved = mem.get_evolved_what(old_id)
        assert evolved is not None
        assert "enriched" in evolved or len(evolved) > 0

    def test_trigger_memory_evolution_skips_entity_links(self, mem):
        """Entity and causal links are NOT evolved — only semantic/supersession."""
        mem.commit("Alpha", "Alpha work", "reason", ["x.py"], "next")
        mem.commit("Beta", "Beta work", "reason", ["x.py"], "next")

        recent = mem._parse_recent_commit_data("main", k=5)
        old_id = recent[1]["id"]
        new_id = recent[0]["id"]

        from ccr.core.types import CommitLink

        mock_client = MagicMock()
        mock_client.completion.return_value = '{"evolved_what": "evolved", "evolution_reason": "r"}'
        mem.set_sub_client(mock_client)

        entity_links = [CommitLink(target=old_id, link_type="entity", score=0.9)]
        mem._trigger_memory_evolution(new_id, entity_links)

        # Entity links should NOT trigger evolution
        evolved = mem.get_evolved_what(old_id)
        assert evolved is None
        mock_client.completion.assert_not_called()

    def test_get_context_shows_evolved_summary(self, mem):
        """get_context at level 2+ shows [evolved] tag for commits with evolved summaries."""
        mem.commit("Feature A", "Built feature A", "needed", ["a.py"], "next")
        recent = mem._parse_recent_commit_data("main", k=5)
        commit_id = recent[0]["id"]

        # Manually inject an evolved summary
        from ccr.core.memory import EvolvedSummary
        mem._evolved_summaries[commit_id] = EvolvedSummary(
            commit_id=commit_id,
            evolved_what="Feature A now includes caching layer.",
            evolution_reason="New caching commit arrived.",
            evolved_at="2026-03-23T00:00:00+00:00",
            source_commit_id="C999",
            original_what="Built feature A",
        )

        ctx = mem.get_context(level=2)
        assert "[evolved]" in ctx
        assert "caching layer" in ctx

    def test_gcc_evolve_memory_tool_without_sub_client(self, mem):
        """gcc_evolve_memory returns appropriate message when no sub-model."""
        from unittest.mock import patch

        # Patch only _get_sub_client to return None — function returns early before
        # touching _state_lock or _ensure_memory, so no other patching needed.
        with patch("ccr.mcp_server._get_sub_client", return_value=None):
            import ccr.mcp_server as srv
            result = srv.gcc_evolve_memory(commit_id=None)

        msg = result["message"]
        assert "Sub-model not available" in msg or "not available" in msg.lower()
