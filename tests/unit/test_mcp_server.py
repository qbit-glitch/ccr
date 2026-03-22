"""Tests for the CCR MCP server tools."""

import json
import os
import tempfile

import pytest

from ccr.mcp_server import (
    _init,
    _save_playbook,
    ace_apply_delta,
    ace_evolve_from_failures,
    ace_evolve_schema,
    ace_find_similar,
    ace_get_playbook,
    ace_get_stats,
    ace_prune,
    ace_update_counters,
    gcc_branch,
    gcc_commit,
    gcc_consolidate,
    gcc_context,
    gcc_links,
    gcc_log_ota,
    gcc_merge,
    gcc_patterns,
    gcc_status,
    gcc_summaries,
    index_build,
    index_search,
    rlm_execute,
    rlm_finalize,
    rlm_init,
)
import ccr.mcp_server as mcp_mod
from ccr.mcp_server import mcp as mcp_instance


@pytest.fixture(autouse=True)
def setup_project(tmp_path, monkeypatch):
    """Initialize CCR in a temp directory for each test."""
    # Redirect ~/.ccr/ to temp dir so tests don't touch real global playbook
    fake_home_ccr = tmp_path / "global_ccr"
    fake_home_ccr.mkdir()
    original_expanduser = os.path.expanduser
    def mock_expanduser(path):
        if path.startswith("~/.ccr"):
            return str(fake_home_ccr) + path[5:]  # Replace ~/.ccr with fake
        return original_expanduser(path)
    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

    # Create a sample file for indexing
    src = tmp_path / "hello.py"
    src.write_text("def greet(name):\n    return f'Hello, {name}!'\n\nclass Greeter:\n    pass\n")

    (tmp_path / "utils.py").write_text("def helper():\n    pass\n")

    _init(str(tmp_path))
    yield tmp_path

    # Cleanup globals
    mcp_mod._memory = None
    mcp_mod._playbook = None
    mcp_mod._global_playbook = None
    mcp_mod._repo_index = None
    mcp_mod._repl = None


# ===========================================================================
# GCC Memory Tools
# ===========================================================================


class TestGCCCommit:
    def test_basic_commit(self):
        result = gcc_commit(
            title="Add greeting",
            what="Added greet function",
            why="User needs greeting",
            files_changed=["hello.py"],
            next_step="Add tests",
        )
        assert "[C001]" in result
        assert "Add greeting" in result

    def test_sequential_commits(self):
        gcc_commit("First", "did A", "because A", [], "do B")
        result = gcc_commit("Second", "did B", "because B", [], "do C")
        assert "[C002]" in result

    def test_commit_appears_in_context(self):
        gcc_commit("Add greeting", "Added greet", "needed", ["hello.py"], "test")
        ctx = gcc_context(level=2)
        assert "Add greeting" in ctx


class TestGCCBranch:
    def test_create_branch(self):
        result = gcc_branch("try-refactor", "Explore refactoring", "Will simplify code")
        assert "try-refactor" in result

    def test_branch_shows_in_status(self):
        gcc_branch("experiment", "Testing", "Hypothesis")
        status = gcc_status()
        assert "experiment" in status

    def test_invalid_branch_name(self):
        with pytest.raises(ValueError, match="kebab-case"):
            gcc_branch("InvalidName", "purpose", "hypothesis")

    def test_cannot_branch_from_branch(self):
        gcc_branch("first-branch", "purpose", "hypothesis")
        with pytest.raises(ValueError, match="Must be on main"):
            gcc_branch("second-branch", "purpose", "hypothesis")


class TestGCCMerge:
    def test_merge_branch(self):
        gcc_branch("test-branch", "Testing merge", "Should work")
        gcc_commit("Work on branch", "did stuff", "testing", [], "merge")
        result = gcc_merge("test-branch", "success", "It worked")
        assert "success" in result

    def test_merge_invalid_outcome(self):
        gcc_branch("bad-merge", "test", "test")
        with pytest.raises(ValueError, match="success/failure/partial"):
            gcc_merge("bad-merge", "invalid", "nope")


class TestGCCContext:
    def test_level_1(self):
        ctx = gcc_context(level=1)
        assert "Project" in ctx

    def test_level_2_with_commits(self):
        gcc_commit("First", "did A", "because", [], "next")
        ctx = gcc_context(level=2)
        assert "First" in ctx

    def test_level_5_search(self):
        gcc_commit("Fix parser bug", "Fixed regex", "was broken", ["parser.py"], "test")
        ctx = gcc_context(level=5, search_term="parser")
        assert "parser" in ctx.lower()

    def test_log_window(self):
        gcc_log_ota("Found issue", "Need to investigate", "Reading code")
        ctx = gcc_context(level=1, log_window=5)
        assert "Found issue" in ctx


class TestGCCLogOTA:
    def test_log_ota(self):
        result = gcc_log_ota("Test failed", "Bug in logic", "Fixed condition")
        assert result == "OTA logged."


class TestGCCStatus:
    def test_status_shows_branch(self):
        status = gcc_status()
        assert "main" in status


# ===========================================================================
# ACE Playbook Tools
# ===========================================================================


class TestACEGetPlaybook:
    def test_empty_playbook(self):
        result = ace_get_playbook()
        assert "empty" in result.lower() or "STRATEGIES" in result

    def test_playbook_with_bullets(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Always test first"}
        ])
        result = ace_get_playbook()
        assert "Always test first" in result


class TestACEApplyDelta:
    def test_add_bullet(self):
        result = ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Check edge cases"}
        ])
        assert "Applied 1" in result

    def test_add_multiple(self):
        result = ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Strategy A"},
            {"type": "ADD", "section": "COMMON MISTAKES TO AVOID", "content": "Mistake A"},
        ])
        assert "Applied 2" in result

    def test_update_bullet(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Original content"}
        ])
        pb = ace_get_playbook()
        # Find the bullet ID
        import re
        match = re.search(r"\[(str-\d+)\]", pb)
        assert match
        bullet_id = match.group(1)

        result = ace_apply_delta([
            {"type": "UPDATE", "bullet_id": bullet_id, "content": "Updated content"}
        ])
        assert "Applied 1" in result
        assert "Updated content" in ace_get_playbook()

    def test_remove_bullet(self):
        ace_apply_delta([
            {"type": "ADD", "section": "OTHERS", "content": "Remove me"}
        ])
        pb = ace_get_playbook()
        import re
        match = re.search(r"\[(\w+-\d+)\]", pb)
        bullet_id = match.group(1)

        result = ace_apply_delta([
            {"type": "REMOVE", "bullet_id": bullet_id}
        ])
        assert "Applied 1" in result
        assert "Remove me" not in ace_get_playbook()

    def test_persists_to_disk(self, setup_project):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Persisted"}
        ])
        # Reload from disk
        playbook_path = os.path.join(str(setup_project), ".ccr", "playbook.txt")
        assert os.path.isfile(playbook_path)
        with open(playbook_path) as f:
            assert "Persisted" in f.read()


class TestACEUpdateCounters:
    def test_update_helpful(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Good strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook())
        bullet_id = match.group(1)

        result = ace_update_counters([{"id": bullet_id, "tag": "helpful"}])
        assert "Updated 1" in result

        pb = ace_get_playbook()
        assert "helpful=1" in pb

    def test_update_harmful(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Bad strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook())
        bullet_id = match.group(1)

        ace_update_counters([{"id": bullet_id, "tag": "harmful"}])
        assert "harmful=1" in ace_get_playbook()


class TestACEUpdateCountersWithFailureLessons:
    def test_harmful_with_failure_lesson(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Test strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook())
        bullet_id = match.group(1)

        result = ace_update_counters([{
            "id": bullet_id,
            "tag": "harmful",
            "failure_lesson": {
                "failure_point": "Strategy failed on binary files",
                "flawed_reasoning": "Assumed text input only",
                "counterfactual": "Check file type before processing",
                "prevention_principle": "Validate input format at boundaries",
            },
        }])
        assert "Updated 1" in result
        assert "1 structured failure lesson" in result

    def test_harmful_without_lesson_backward_compat(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Test strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook())
        bullet_id = match.group(1)

        result = ace_update_counters([{"id": bullet_id, "tag": "harmful"}])
        assert "Updated 1" in result
        assert "failure lesson" not in result

    def test_failure_lesson_appears_in_playbook(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Fragile strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook())
        bullet_id = match.group(1)

        ace_update_counters([{
            "id": bullet_id,
            "tag": "harmful",
            "failure_lesson": {
                "failure_point": "Broke on empty input",
                "flawed_reasoning": "Assumed non-empty",
                "counterfactual": "Guard against empty",
                "prevention_principle": "Always handle empty inputs",
            },
        }])

        pb = ace_get_playbook()
        assert "FAILURE: Broke on empty input" in pb
        assert "PRINCIPLE: Always handle empty inputs" in pb

    def test_failure_lesson_persists_to_disk(self, setup_project):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Persisted strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook())
        bullet_id = match.group(1)

        ace_update_counters([{
            "id": bullet_id,
            "tag": "harmful",
            "failure_lesson": {
                "failure_point": "Persistent failure",
                "flawed_reasoning": "Bad logic",
                "counterfactual": "Better approach",
                "prevention_principle": "Saved principle",
            },
        }])

        lessons_path = os.path.join(str(setup_project), ".ccr", "failure_lessons.json")
        assert os.path.isfile(lessons_path)
        with open(lessons_path) as f:
            data = json.load(f)
        assert bullet_id in data
        assert data[bullet_id]["lessons"][0]["failure_point"] == "Persistent failure"

    def test_failure_lessons_in_stats(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Tested strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook())
        bullet_id = match.group(1)

        ace_update_counters([{
            "id": bullet_id,
            "tag": "harmful",
            "failure_lesson": {
                "failure_point": "X", "flawed_reasoning": "Y",
                "counterfactual": "Z", "prevention_principle": "W",
            },
        }])

        stats = json.loads(ace_get_stats())
        assert stats["project"]["total_failure_lessons"] == 1
        assert stats["project"]["harmful_with_lessons"] == 1


class TestACEGetStats:
    def test_stats_empty(self):
        result = ace_get_stats()
        stats = json.loads(result)
        assert stats["project"]["total_bullets"] == 0

    def test_stats_with_bullets(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "A"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "B"},
        ])
        result = ace_get_stats()
        stats = json.loads(result)
        assert stats["project"]["total_bullets"] == 2


class TestACEFindSimilar:
    def test_no_similar(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Alpha beta gamma"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Completely different topic here"},
        ])
        result = ace_find_similar(threshold=0.9)
        assert "No similar" in result

    def test_find_similar_pair(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS",
             "content": "Always check edge cases in loop bounds and iterations"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS",
             "content": "Always check edge cases in loop bounds and ranges"},
        ])
        result = ace_find_similar(threshold=0.5)
        assert "similarity=" in result


class TestACEPrune:
    def test_prune_no_problematic(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Good one"}
        ])
        result = ace_prune()
        assert "1 bullets" in result  # still has 1

    def test_prune_harmful(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Bad strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook())
        bullet_id = match.group(1)

        # Mark harmful 3 times
        for _ in range(3):
            ace_update_counters([{"id": bullet_id, "tag": "harmful"}])

        result = ace_prune()
        assert "Pruned 1" in result


class TestPrunePreservesLessons:
    """Tests that ace_prune evolves failure lessons before removing bullets."""

    def _add_harmful_bullets_with_lessons(self, n=3):
        """Helper: add n harmful bullets each with a failure lesson."""
        import re
        for i in range(n):
            ace_apply_delta([{
                "type": "ADD",
                "section": "STRATEGIES & INSIGHTS",
                "content": f"Strategy that fails {i}",
            }])
        pb_text = ace_get_playbook()
        ids = re.findall(r"\[(str-\d+)\]", pb_text)
        for i, bid in enumerate(ids[:n]):
            # Mark harmful enough times to trigger pruning (>=3)
            for _ in range(3):
                ace_update_counters([{"id": bid, "tag": "harmful"}])
            # Add failure lesson on last harmful call is already done above;
            # we need to add the lesson separately
            ace_update_counters([{
                "id": bid,
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Failure {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Fix {i}",
                    "prevention_principle": f"Prevention rule {i}",
                    "task_context": f"Task context {i}",
                },
            }])
        return ids[:n]

    def test_prune_evolves_before_removing(self):
        """Pruning harmful bullets with lessons first evolves them into new skills."""
        self._add_harmful_bullets_with_lessons(3)
        # Prune — should auto-evolve first
        result = ace_prune()
        # Harmful source bullets should be removed
        pb = ace_get_playbook()
        assert "Strategy that fails" not in pb
        # But prevention principles should survive as new heuristic bullets
        assert "Prevention rule 0" in pb
        assert "Prevention rule 1" in pb
        assert "Prevention rule 2" in pb

    def test_prune_message_includes_evolved_count(self):
        """Return message mentions how many skills were evolved."""
        self._add_harmful_bullets_with_lessons(3)
        result = ace_prune()
        assert "Evolved 3 new skill" in result
        assert "Pruned" in result

    def test_prune_with_no_lessons_works_normally(self):
        """Pruning bullets without failure lessons still works fine."""
        import re
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Bad no-lesson strategy"}
        ])
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook())
        bid = match.group(1)
        for _ in range(3):
            ace_update_counters([{"id": bid, "tag": "harmful"}])
        result = ace_prune()
        assert "Pruned 1" in result
        assert "Evolved" not in result
        assert "Bad no-lesson strategy" not in ace_get_playbook()


class TestACEEvolveFromFailures:
    def _add_harmful_bullets_with_lessons(self, n=3):
        """Helper: add n bullets and mark each harmful with a failure lesson."""
        import re
        for i in range(n):
            ace_apply_delta([{
                "type": "ADD",
                "section": "STRATEGIES & INSIGHTS",
                "content": f"Strategy that fails {i}",
            }])
        pb_text = ace_get_playbook()
        ids = re.findall(r"\[(str-\d+)\]", pb_text)
        for i, bid in enumerate(ids[:n]):
            ace_update_counters([{
                "id": bid,
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Failure {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Fix {i}",
                    "prevention_principle": f"Prevention rule {i}",
                    "task_context": f"Task context {i}",
                },
            }])
        return ids[:n]

    def test_evolve_not_triggered_below_threshold(self):
        self._add_harmful_bullets_with_lessons(2)
        result = ace_evolve_from_failures(threshold=3)
        assert "not triggered" in result.lower()

    def test_evolve_creates_new_skills(self):
        self._add_harmful_bullets_with_lessons(3)
        result = ace_evolve_from_failures(threshold=3)
        assert "Evolved 3 new skill" in result
        # Verify skills appear in playbook
        pb = ace_get_playbook()
        assert "Prevention rule 0" in pb
        assert "Prevention rule 1" in pb
        assert "Prevention rule 2" in pb

    def test_evolve_skills_persisted(self):
        self._add_harmful_bullets_with_lessons(3)
        ace_evolve_from_failures(threshold=3)
        # Reload playbook from disk
        mcp_mod._playbook = None
        mcp_mod._playbook = mcp_mod._load_playbook()
        pb = ace_get_playbook()
        assert "Prevention rule 0" in pb

    def test_stats_show_evolution_needed(self):
        self._add_harmful_bullets_with_lessons(3)
        stats = json.loads(ace_get_stats())
        assert stats["project"]["evolution_needed"] is True
        assert stats["project"]["evolution_candidates"] == 3


# ===========================================================================
# RLM Sandbox Tools
# ===========================================================================


class TestRLMInit:
    def test_init(self):
        result = rlm_init("Analyze the code")
        assert "REPL initialized" in result
        assert "task_prompt" in result

    def test_init_shows_file_count(self):
        result = rlm_init("Test")
        assert "files indexed" in result


class TestRLMExecute:
    def test_basic_execution(self):
        rlm_init("Test")
        result = rlm_execute("x = 2 + 3\nprint(x)")
        assert "5" in result

    def test_variable_persistence(self):
        rlm_init("Test")
        rlm_execute("my_var = 42")
        result = rlm_execute("print(my_var * 2)")
        assert "84" in result

    def test_access_task_prompt(self):
        rlm_init("Find all classes")
        result = rlm_execute("print(task_prompt)")
        assert "Find all classes" in result

    def test_search_repo(self):
        rlm_init("Search test")
        result = rlm_execute("results = search_repo('greet')\nprint(len(results))")
        assert "stdout" in result or "1" in result

    def test_get_file(self):
        rlm_init("Read test")
        result = rlm_execute("content = get_file('hello.py')\nprint(content[:20])")
        assert "greet" in result.lower() or "def" in result

    def test_show_vars(self):
        rlm_init("Test")
        rlm_execute("x = 42\ny = 'hello'")
        result = rlm_execute("print(SHOW_VARS())")
        assert "x" in result

    def test_error_handling(self):
        rlm_init("Test")
        result = rlm_execute("1/0")
        assert "error" in result.lower() or "ZeroDivision" in result

    def test_no_init_error(self):
        # Reset REPL
        mcp_mod._repl = None
        result = rlm_execute("print(1)")
        assert "not initialized" in result.lower()


class TestRLMExecuteMetadataOnly:
    """Tests for R5 audit fix: metadata-only stdout enforcement (RLM paper Section 3)."""

    def test_short_stdout_unchanged(self):
        """Stdout under the 1000-char threshold is returned as-is."""
        rlm_init("Test")
        result = rlm_execute("print('hello world')")
        assert "hello world" in result
        # Should NOT contain truncation markers
        assert "[stdout truncated:" not in result

    def test_long_stdout_summarized(self):
        """Stdout over the 1000-char threshold gets a metadata summary."""
        rlm_init("Test")
        # Generate output well over 1000 chars (50 lines x ~30 chars each = ~1500 chars)
        code = "for i in range(50): print(f'line {i}: ' + 'x' * 20)"
        result = rlm_execute(code)
        assert "[stdout truncated:" in result
        assert "lines" in result
        assert "chars" in result
        # First lines should be present
        assert "line 0:" in result
        # Last lines should be present
        assert "line 49:" in result
        # Middle lines should NOT be present (they were truncated)
        assert "line 25:" not in result

    def test_metadata_only_false_returns_full(self):
        """Setting metadata_only=False returns full stdout regardless of length."""
        rlm_init("Test")
        code = "for i in range(50): print(f'line {i}: ' + 'x' * 20)"
        result = rlm_execute(code, metadata_only=False)
        # Should NOT be truncated
        assert "[stdout truncated:" not in result
        # All lines should be present
        assert "line 0:" in result
        assert "line 25:" in result
        assert "line 49:" in result


class TestRLMFinalize:
    def test_finalize(self):
        rlm_init("Test")
        rlm_execute("answer = 'The result is 42'")
        result = rlm_finalize("answer")
        assert "42" in result

    def test_finalize_dict(self):
        rlm_init("Test")
        rlm_execute("data = {'key': 'value', 'count': 3}")
        result = rlm_finalize("data")
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_finalize_missing_var(self):
        rlm_init("Test")
        result = rlm_finalize("nonexistent")
        assert "Error" in result or "not found" in result.lower()

    def test_no_init_error(self):
        mcp_mod._repl = None
        result = rlm_finalize("x")
        assert "not initialized" in result.lower()


# ===========================================================================
# Repo Index Tools
# ===========================================================================


class TestIndexBuild:
    def test_build(self):
        result = index_build()
        assert "Files:" in result
        assert "2" in result  # hello.py + utils.py


class TestIndexSearch:
    def test_search_by_symbol(self):
        result = index_search("greet")
        assert "hello.py" in result

    def test_search_by_class(self):
        result = index_search("Greeter")
        assert "hello.py" in result

    def test_search_no_results(self):
        result = index_search("nonexistent_symbol_xyz")
        assert "No files" in result

    def test_search_top_k(self):
        result = index_search("py", top_k=1)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        # 1 header line + 1 result line
        assert len(lines) == 2


class TestIndexSearchModes:
    """Tests for the A-RAG-inspired search mode parameter."""

    def test_keyword_mode(self):
        result = index_search("greet", mode="keyword")
        assert "keyword search" in result
        assert "hello.py" in result

    def test_semantic_mode_bm25_fallback(self):
        result = index_search("greet", mode="semantic")
        assert "semantic search" in result
        # BM25 fallback since no ONNX
        assert "BM25 fallback" in result

    def test_hybrid_mode_default(self):
        result = index_search("greet")
        assert "hybrid search" in result

    def test_invalid_mode_error(self):
        result = index_search("greet", mode="invalid")
        assert "Error" in result
        assert "invalid" in result

    def test_build_shows_embedding_status(self):
        result = index_build()
        assert "Embedding" in result or "embedding" in result


# ===========================================================================
# Integration: Cross-tool workflows
# ===========================================================================


class TestWorkflows:
    def test_commit_then_context(self):
        """GCC workflow: commit → retrieve context."""
        gcc_commit("Setup", "Created project", "starting out", ["hello.py"], "Add features")
        ctx = gcc_context(level=2)
        assert "Setup" in ctx
        assert "Created project" in ctx

    def test_branch_commit_merge(self):
        """GCC workflow: branch → commit → merge."""
        gcc_branch("experiment", "Try new approach", "Might work")
        gcc_commit("Experiment work", "tried stuff", "exploration", [], "evaluate")
        result = gcc_merge("experiment", "success", "Approach works")
        assert "success" in result

        # Back on main, context includes merge info
        ctx = gcc_context(level=2)
        assert "experiment" in ctx.lower() or "Merge" in ctx

    def test_ace_full_cycle(self):
        """ACE workflow: add → tag → prune."""
        # Add strategies
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Good one"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Bad one"},
        ])

        # Tag them
        import re
        pb = ace_get_playbook()
        ids = re.findall(r"\[(str-\d+)\]", pb)
        assert len(ids) == 2

        # Tag good one as helpful (2x)
        ace_update_counters([{"id": ids[0], "tag": "helpful"}])
        ace_update_counters([{"id": ids[0], "tag": "helpful"}])
        # Tag bad one as harmful (3x) — each call increments once
        ace_update_counters([{"id": ids[1], "tag": "harmful"}])
        ace_update_counters([{"id": ids[1], "tag": "harmful"}])
        ace_update_counters([{"id": ids[1], "tag": "harmful"}])

        # Prune
        ace_prune()
        pb = ace_get_playbook()
        assert "Good one" in pb
        assert "Bad one" not in pb

    def test_rlm_analysis_workflow(self):
        """RLM workflow: init → explore → finalize."""
        rlm_init("Find all function definitions")
        rlm_execute("results = search_repo('greet')")
        rlm_execute("file_content = get_file(results[0]['path'])")
        rlm_execute(
            "import re\n"
            "functions = re.findall(r'def (\\w+)', file_content)\n"
            "answer = {'file': results[0]['path'], 'functions': functions}"
        )
        result = rlm_finalize("answer")
        data = json.loads(result)
        assert "hello.py" in data["file"]
        assert "greet" in data["functions"]


# ===========================================================================
# Two-Tier Playbook (Global + Project)
# ===========================================================================


class TestTwoTierPlaybook:
    def test_apply_delta_global_scope(self):
        """Operations with scope='global' go to global playbook."""
        result = ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Global strategy"}],
            scope="global",
        )
        assert "global" in result
        gpb = mcp_mod._global_playbook
        assert len(gpb.bullets) == 1
        assert gpb.bullets[0].content == "Global strategy"

    def test_apply_delta_project_scope_default(self):
        """Default scope is 'project'."""
        ace_apply_delta([{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Project strategy"}])
        ppb = mcp_mod._playbook
        assert any(b.content == "Project strategy" for b in ppb.bullets)
        # Global should be empty
        gpb = mcp_mod._global_playbook
        assert not any(b.content == "Project strategy" for b in gpb.bullets)

    def test_get_playbook_shows_both(self):
        """ace_get_playbook returns both global and project sections."""
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "G1"}],
            scope="global",
        )
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "P1"}],
            scope="project",
        )
        result = ace_get_playbook()
        assert "GLOBAL PLAYBOOK" in result
        assert "PROJECT PLAYBOOK" in result
        assert "G1" in result
        assert "P1" in result

    def test_get_playbook_both_sections_present(self):
        """Both global and project sections are always present."""
        result = ace_get_playbook()
        assert "GLOBAL PLAYBOOK" in result
        assert "PROJECT PLAYBOOK" in result

    def test_update_counters_global(self):
        """Update counters in global playbook."""
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Global insight"}],
            scope="global",
        )
        gpb = mcp_mod._global_playbook
        bid = gpb.bullets[0].id
        ace_update_counters([{"id": bid, "tag": "helpful"}], scope="global")
        assert gpb.bullets[0].helpful == 1

    def test_stats_both_tiers(self):
        """ace_get_stats returns stats for both global and project."""
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "G"}],
            scope="global",
        )
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "P"}],
        )
        stats = json.loads(ace_get_stats())
        assert "global" in stats
        assert "project" in stats
        assert stats["global"]["total_bullets"] == 1
        assert stats["project"]["total_bullets"] == 1

    def test_find_similar_with_scope(self):
        """ace_find_similar respects scope."""
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Alpha beta gamma delta epsilon"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Alpha beta gamma delta zeta"},
        ], scope="global")
        result = ace_find_similar(threshold=0.5, scope="global")
        assert "similarity" in result

    def test_find_similar_cross_scope(self):
        """Cross-tier similarity detection."""
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Always validate user input before processing"}],
            scope="global",
        )
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Always validate user input before processing"}],
            scope="project",
        )
        result = ace_find_similar(threshold=0.5, scope="cross")
        assert "global" in result.lower() and "project" in result.lower()

    def test_prune_with_scope(self):
        """ace_prune respects scope."""
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Bad global"}],
            scope="global",
        )
        gpb = mcp_mod._global_playbook
        bid = gpb.bullets[0].id
        for _ in range(4):
            ace_update_counters([{"id": bid, "tag": "harmful"}], scope="global")
        result = ace_prune(scope="global")
        assert "1" in result  # pruned 1
        assert len(gpb.bullets) == 0

    def test_global_playbook_persistence(self, tmp_path):
        """Global playbook persists to disk."""
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Persistent global"}],
            scope="global",
        )
        # Check file exists
        assert os.path.isfile(mcp_mod._global_playbook_path)
        with open(mcp_mod._global_playbook_path) as f:
            text = f.read()
        assert "Persistent global" in text

    def test_tiers_are_independent(self):
        """Adding to one tier doesn't affect the other."""
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Only global"}],
            scope="global",
        )
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Only project"}],
        )
        gpb = mcp_mod._global_playbook
        ppb = mcp_mod._playbook
        g_contents = {b.content for b in gpb.bullets}
        p_contents = {b.content for b in ppb.bullets}
        assert "Only global" in g_contents
        assert "Only global" not in p_contents
        assert "Only project" in p_contents
        assert "Only project" not in g_contents


# ===========================================================================
# Tool Annotations (MCP spec 2025-11-25)
# ===========================================================================


def _get_tool_annotations(tool_name: str):
    """Helper to retrieve annotations for a registered MCP tool."""
    tool = mcp_instance._tool_manager._tools[tool_name]
    return tool.annotations


class TestToolAnnotations:
    """Verify readOnlyHint, destructiveHint, idempotentHint on all 18 tools."""

    # -- Read-only tools --

    @pytest.mark.parametrize("tool_name", [
        "gcc_context", "gcc_status", "gcc_patterns",
        "ace_get_playbook", "ace_get_stats", "ace_find_similar",
        "index_search",
    ])
    def test_read_only_tools(self, tool_name):
        ann = _get_tool_annotations(tool_name)
        assert ann.readOnlyHint is True, f"{tool_name} should be readOnlyHint=True"
        assert ann.destructiveHint is False, f"{tool_name} should be destructiveHint=False"
        assert ann.idempotentHint is True, f"{tool_name} should be idempotentHint=True"

    # -- gcc_log_ota: writes to OTA log, not idempotent (each call appends) --

    def test_gcc_log_ota_annotations(self):
        ann = _get_tool_annotations("gcc_log_ota")
        assert ann.readOnlyHint is False, "gcc_log_ota writes to OTA log"
        assert ann.destructiveHint is False, "gcc_log_ota appends, doesn't destroy"
        assert ann.idempotentHint is False, "gcc_log_ota creates new entry each call"

    # -- Destructive tools --

    @pytest.mark.parametrize("tool_name", [
        "gcc_merge", "ace_prune",
    ])
    def test_destructive_tools(self, tool_name):
        ann = _get_tool_annotations(tool_name)
        assert ann.destructiveHint is True, f"{tool_name} should be destructiveHint=True"
        assert ann.readOnlyHint is False, f"{tool_name} should be readOnlyHint=False"

    # -- gcc_branch: creates branches, not destructive --

    def test_gcc_branch_annotations(self):
        ann = _get_tool_annotations("gcc_branch")
        assert ann.readOnlyHint is False, "gcc_branch creates state"
        assert ann.destructiveHint is False, "gcc_branch only creates, doesn't destroy"
        assert ann.idempotentHint is False, "gcc_branch fails if branch exists"

    # -- Mutating non-destructive tools --

    @pytest.mark.parametrize("tool_name", [
        "gcc_commit", "ace_apply_delta", "ace_update_counters",
        "rlm_init", "rlm_execute",
    ])
    def test_mutating_non_destructive_tools(self, tool_name):
        ann = _get_tool_annotations(tool_name)
        assert ann.readOnlyHint is False, f"{tool_name} should be readOnlyHint=False"
        assert ann.destructiveHint is False, f"{tool_name} should be destructiveHint=False"
        assert ann.idempotentHint is False, f"{tool_name} should be idempotentHint=False"

    def test_rlm_finalize_annotations(self):
        """rlm_finalize is destructive — it cleans up the REPL and destroys session state."""
        ann = _get_tool_annotations("rlm_finalize")
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is True, "rlm_finalize destroys REPL session state"
        assert ann.idempotentHint is False

    # -- ace_evolve_from_failures: idempotent due to evolved flag --

    def test_ace_evolve_from_failures_annotations(self):
        ann = _get_tool_annotations("ace_evolve_from_failures")
        assert ann.readOnlyHint is False, "ace_evolve_from_failures may create bullets"
        assert ann.destructiveHint is False, "ace_evolve_from_failures adds, doesn't destroy"
        assert ann.idempotentHint is True, "ace_evolve_from_failures is idempotent (evolved flag)"

    # -- Idempotent tools --

    @pytest.mark.parametrize("tool_name", [
        "gcc_context", "gcc_status",
        "ace_get_playbook", "ace_get_stats", "ace_find_similar",
        "index_search", "index_build",
        "ace_evolve_from_failures",
    ])
    def test_idempotent_tools(self, tool_name):
        ann = _get_tool_annotations(tool_name)
        assert ann.idempotentHint is True, f"{tool_name} should be idempotentHint=True"

    # -- index_build: idempotent but not read-only --

    def test_index_build_annotations(self):
        ann = _get_tool_annotations("index_build")
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.idempotentHint is True

    # -- gcc_consolidate: not read-only, not destructive, not idempotent --

    def test_gcc_consolidate_annotations(self):
        ann = _get_tool_annotations("gcc_consolidate")
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.idempotentHint is False

    # -- gcc_summaries: read-only, not destructive, idempotent --

    def test_gcc_summaries_annotations(self):
        ann = _get_tool_annotations("gcc_summaries")
        assert ann.readOnlyHint is True
        assert ann.destructiveHint is False
        assert ann.idempotentHint is True

    # -- All 23 tools have annotations --

    def test_all_tools_have_annotations(self):
        all_tools = mcp_instance._tool_manager._tools
        assert len(all_tools) == 23, f"Expected 23 tools, got {len(all_tools)}"
        for name, tool in all_tools.items():
            assert tool.annotations is not None, f"{name} missing annotations"


# ===========================================================================
# GCC Commit Links MCP Tool Tests
# ===========================================================================


class TestGccLinks:
    """Tests for gcc_links MCP tool and gcc_context follow_links."""

    def test_gcc_links_basic(self):
        gcc_commit("Setup", "Created server", "Need infra",
                   ["mcp_server.py"], "Next", admission_threshold=1.0)
        gcc_commit("Fix server", "Fixed bug in server",
                   "Bug discovered in C001", ["mcp_server.py"], "Test",
                   admission_threshold=1.0)
        result = gcc_links("C002")
        assert "C001" in result
        assert "Links for C002" in result

    def test_gcc_links_no_links(self):
        result = gcc_links("C999")
        assert "No links found" in result

    def test_gcc_links_filtered_types(self):
        gcc_commit("A", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        gcc_commit("B", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        result = gcc_links("C002", link_types="entity")
        assert "Entity" in result

    def test_gcc_links_multi_hop(self):
        gcc_commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        gcc_commit("B", "Refs C001", "See C001", ["b.py"], "N",
                   admission_threshold=1.0)
        gcc_commit("C", "Refs C002", "See C002", ["c.py"], "N",
                   admission_threshold=1.0)
        result = gcc_links("C001", max_hops=2)
        # Should reach C003 via C002 -> C003 (hop 2)
        assert "C002" in result

    def test_gcc_context_follow_links(self):
        gcc_commit("A", "Created server module", "Foundation",
                   ["server.py"], "Next", admission_threshold=1.0)
        gcc_commit("B", "Extended server module", "Build on A",
                   ["server.py"], "Test", admission_threshold=1.0)
        result = gcc_context(level=5, commit_id="C001", follow_links=True)
        assert "Linked:" in result


# ===========================================================================
# GCC Hierarchical Summary MCP Tool Tests
# ===========================================================================


class TestGCCHierarchicalSummaryTools:
    """Tests for gcc_consolidate and gcc_summaries MCP tools."""

    def _make_commits(self, n):
        for i in range(1, n + 1):
            gcc_commit(
                title=f"Task {i}",
                what=f"Implemented feature {i}",
                why=f"Milestone {i}",
                files_changed=[f"src/mod{i}.py"],
                next_step=f"Continue {i + 1}",
                admission_threshold=1.0,
            )

    def test_gcc_consolidate_session(self):
        self._make_commits(5)
        result = gcc_consolidate(tier="session")
        assert "Session Summary" in result

    def test_gcc_consolidate_phase(self):
        self._make_commits(5)
        result = gcc_consolidate(tier="phase")
        assert "Phase Summary" in result

    def test_gcc_consolidate_project_prompt(self):
        result = gcc_consolidate(tier="project")
        assert "Please generate a project overview" in result

    def test_gcc_consolidate_project_save(self):
        result = gcc_consolidate(tier="project", content="Test project builds AI tools.")
        assert "saved" in result.lower()

    def test_gcc_summaries_all(self):
        self._make_commits(5)
        gcc_consolidate(tier="phase")
        gcc_consolidate(tier="project", content="AI project overview")
        result = gcc_summaries(tier="all", count=5)
        assert "Session" in result or "Phase" in result or "Project" in result

    def test_gcc_summaries_empty(self):
        result = gcc_summaries(tier="all", count=5)
        assert result  # Should return something, not crash

    def test_gcc_summaries_count_bounded(self):
        result = gcc_summaries(tier="all", count=0)
        assert result  # count=0 should be bounded to 1


# ===========================================================================
# CER Pattern Buffer MCP Tests
# ===========================================================================


class TestGCCCommitPatterns:
    def test_gcc_commit_with_patterns(self):
        result = gcc_commit(
            title="Add feature",
            what="Added new feature",
            why="User request",
            files_changed=["feature.py"],
            next_step="Add tests",
            patterns_learned=["When adding {feature}, update tests and docs together"],
        )
        assert "C001" in result

    def test_gcc_commit_patterns_optional(self):
        """gcc_commit works without patterns_learned (backward compat)."""
        result = gcc_commit(
            title="Fix bug",
            what="Fixed the bug",
            why="Bug report",
            files_changed=["fix.py"],
            next_step="Verify",
        )
        assert "C001" in result

    def test_gcc_commit_promotion_suggestion(self):
        """Promotion suggestion appears after 3 commits with same pattern."""
        pat = "Always validate input parameters before processing data"
        for i in range(2):
            gcc_commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                       patterns_learned=[pat])
        result = gcc_commit("T3", "did 2", "reason", ["2.py"], "next",
                            patterns_learned=[pat])
        assert "Pattern promotion suggestions" in result
        assert "ace_apply_delta" in result


class TestGCCPatterns:
    def test_gcc_patterns_empty(self):
        result = gcc_patterns()
        assert "No patterns found" in result

    def test_gcc_patterns_basic(self):
        gcc_commit("T1", "did A", "reason", ["a.py"], "next",
                   patterns_learned=["Use structured logging for debugging"])
        result = gcc_patterns()
        assert "Pattern Buffer" in result
        assert "structured logging" in result

    def test_gcc_patterns_min_occurrences_filter(self):
        gcc_commit("T1", "did A", "reason", ["a.py"], "next",
                   patterns_learned=["Frequent pattern observed multiple times here",
                                      "Rare single occurrence pattern for test"])
        gcc_commit("T2", "did B", "reason", ["b.py"], "next",
                   patterns_learned=["Frequent pattern observed multiple times here"])
        result = gcc_patterns(min_occurrences=2)
        assert "Frequent pattern" in result
        assert "Rare single" not in result

    def test_gcc_patterns_search_term(self):
        gcc_commit("T1", "did A", "reason", ["a.py"], "next",
                   patterns_learned=["Use sandbox for REPL execution",
                                      "Always update documentation after changes"])
        result = gcc_patterns(search_term="sandbox")
        assert "sandbox" in result
        assert "documentation" not in result

    def test_gcc_patterns_annotations(self):
        """gcc_patterns has correct MCP tool annotations."""
        ann = _get_tool_annotations("gcc_patterns")
        assert ann.readOnlyHint is True
        assert ann.destructiveHint is False
        assert ann.idempotentHint is True


# ===========================================================================
# ACE Schema Evolution MCP Tool Tests (MCE arXiv:2601.21557)
# ===========================================================================


class TestACEEvolveSchema:
    """Test ace_evolve_schema tool — MCE (1+1)-ES schema evolution."""

    def test_evolve_schema_evaluation_mode(self):
        """Default call returns metrics and schema info."""
        result = ace_evolve_schema()
        assert "Schema Health Report" in result
        assert "Overall Health" in result
        assert "Section Balance" in result

    def test_evolve_schema_apply_proposal(self, tmp_path):
        """Apply a proposal updates schema version."""
        # Set up a playbook with conditions that trigger a proposal
        ace_apply_delta(operations=[
            {"type": "ADD", "section": "OTHERS", "content": "Database query optimization techniques"},
            {"type": "ADD", "section": "OTHERS", "content": "Database indexing optimization strategies"},
            {"type": "ADD", "section": "OTHERS", "content": "Database performance optimization tuning"},
        ])
        result = ace_evolve_schema()
        # Might or might not have proposals depending on exact metrics
        if "apply_proposal" in result:
            result2 = ace_evolve_schema(apply_proposal=1)
            assert "schema v" in result2.lower() or "Applied" in result2

    def test_evolve_schema_invalid_proposal_index(self):
        """Invalid proposal index returns error."""
        result = ace_evolve_schema(apply_proposal=99)
        assert "Invalid" in result or "Error" in result

    def test_evolve_schema_rollback_no_parent(self):
        """Rollback with no parent returns error."""
        result = ace_evolve_schema(rollback=True)
        assert "Cannot rollback" in result

    def test_evolve_schema_annotations(self):
        """ace_evolve_schema has correct MCP tool annotations."""
        ann = _get_tool_annotations("ace_evolve_schema")
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.idempotentHint is True


# ===========================================================================
# G4 Fix: Rolling Summary Compression via gcc_commit
# ===========================================================================


class TestGCCCommitRollingSummaryCompression:
    """Tests for the rolling summary compression feature (G4 audit fix)."""

    def test_gcc_commit_with_compressed_summary(self):
        """gcc_commit accepts compressed_summary and uses it."""
        gcc_commit("T1", "did A", "reason A", ["a.py"], "next A")
        result = gcc_commit("T2", "did B", "reason B", ["b.py"], "next B",
                            compressed_summary="Project: started A, added B. Next: C.")
        assert "[C002]" in result
        ctx = gcc_context(level=2)
        # The compressed summary should appear in context
        assert "started A, added B" in ctx

    def test_gcc_commit_compressed_summary_optional(self):
        """gcc_commit works without compressed_summary (backward compat)."""
        result = gcc_commit("T1", "did A", "reason A", ["a.py"], "next A")
        assert "[C001]" in result

    def test_gcc_commit_short_summary_no_warning(self):
        """Short summaries don't produce compression warning."""
        result = gcc_commit("T1", "did A", "reason A", ["a.py"], "next A")
        assert "Rolling summary is getting long" not in result

    def test_gcc_commit_long_summary_triggers_warning(self):
        """Long summaries produce compression warning in return value."""
        # Access memory to write a long summary directly
        import ccr.mcp_server as mod
        mem = mod._memory
        mem._write_rolling_summary("main", "x" * 1300)
        result = gcc_commit("Next", "added more stuff", "reason", ["f.py"], "done")
        assert "Rolling summary is getting long" in result
        assert "compressed_summary" in result

    def test_gcc_commit_compressed_summary_suppresses_warning(self):
        """When compressed_summary is provided, no warning even if summary was long."""
        import ccr.mcp_server as mod
        mem = mod._memory
        mem._write_rolling_summary("main", "x" * 1300)
        result = gcc_commit("Next", "added more", "reason", ["f.py"], "done",
                            compressed_summary="Clean compressed summary here")
        assert "Rolling summary is getting long" not in result
