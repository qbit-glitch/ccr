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
    ace_find_similar,
    ace_get_playbook,
    ace_get_stats,
    ace_prune,
    ace_update_counters,
    gcc_branch,
    gcc_commit,
    gcc_context,
    gcc_log_ota,
    gcc_merge,
    gcc_status,
    index_build,
    index_search,
    rlm_execute,
    rlm_finalize,
    rlm_init,
)
import ccr.mcp_server as mcp_mod


@pytest.fixture(autouse=True)
def setup_project(tmp_path):
    """Initialize CCR in a temp directory for each test."""
    # Create a sample file for indexing
    src = tmp_path / "hello.py"
    src.write_text("def greet(name):\n    return f'Hello, {name}!'\n\nclass Greeter:\n    pass\n")

    (tmp_path / "utils.py").write_text("def helper():\n    pass\n")

    _init(str(tmp_path))
    yield tmp_path

    # Cleanup globals
    mcp_mod._memory = None
    mcp_mod._playbook = None
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
        assert stats["total_failure_lessons"] == 1
        assert stats["harmful_with_lessons"] == 1


class TestACEGetStats:
    def test_stats_empty(self):
        result = ace_get_stats()
        stats = json.loads(result)
        assert stats["total_bullets"] == 0

    def test_stats_with_bullets(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "A"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "B"},
        ])
        result = ace_get_stats()
        stats = json.loads(result)
        assert stats["total_bullets"] == 2


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
        assert stats["evolution_needed"] is True
        assert stats["evolution_candidates"] == 3


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
        assert len(lines) == 1


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
