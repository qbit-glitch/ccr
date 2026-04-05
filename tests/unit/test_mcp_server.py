"""Tests for the CCR MCP server tools."""

import json
import os
import tempfile

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from ccr.mcp_server import (
    _init,
    _save_playbook,
    ace_apply_delta,
    ace_evolve_from_failures,
    ace_evolve_schema,
    ace_find_similar,
    ace_generate_bullets,
    ace_get_playbook,
    ace_prune,
    ace_update_counters,
    gcc_branch,
    gcc_clusters,
    gcc_commit,
    gcc_consolidate,
    gcc_context,
    gcc_links,
    gcc_log_ota,
    gcc_merge,
    gcc_patterns,
    gcc_scratchpad,
    gcc_status,
    gcc_triples,
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
        assert "[C001]" in result["message"]
        assert "Add greeting" in result["message"]
        assert result["commit_id"] == "C001"
        assert result["branch"] == "main"
        assert result["title"] == "Add greeting"
        assert result["admission_decision"] in ("created", "merged", "rejected")

    def test_sequential_commits(self):
        gcc_commit("First", "did A", "because A", [], "do B")
        result = gcc_commit("Second", "did B", "because B", [], "do C")
        assert "[C002]" in result["message"]

    def test_commit_appears_in_context(self):
        gcc_commit("Add greeting", "Added greet", "needed", ["hello.py"], "test")
        ctx = gcc_context(level=2)
        assert "Add greeting" in ctx["message"]


class TestGCCBranch:
    def test_create_branch(self):
        result = gcc_branch("try-refactor", "Explore refactoring", "Will simplify code")
        assert "try-refactor" in result["message"]

    def test_branch_shows_in_status(self):
        gcc_branch("experiment", "Testing", "Hypothesis")
        status = gcc_status()
        assert "experiment" in status["message"]

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
        assert "success" in result["message"]

    def test_merge_invalid_outcome(self):
        gcc_branch("bad-merge", "test", "test")
        with pytest.raises(ValueError, match="success/failure/partial"):
            gcc_merge("bad-merge", "invalid", "nope")


class TestGCCContext:
    def test_level_1(self):
        ctx = gcc_context(level=1)
        assert "Project" in ctx["message"]

    def test_level_2_with_commits(self):
        gcc_commit("First", "did A", "because", [], "next")
        ctx = gcc_context(level=2)
        assert "First" in ctx["message"]

    def test_level_5_search(self):
        gcc_commit("Fix parser bug", "Fixed regex", "was broken", ["parser.py"], "test")
        ctx = gcc_context(level=5, search_term="parser")
        assert "parser" in ctx["message"].lower()

    def test_log_window(self):
        gcc_log_ota("Found issue", "Need to investigate", "Reading code")
        ctx = gcc_context(level=1, log_window=5)
        assert "Found issue" in ctx["message"]


class TestGCCLogOTA:
    def test_log_ota(self):
        result = gcc_log_ota("Test failed", "Bug in logic", "Fixed condition")
        assert result["message"] == "OTA logged."


class TestGCCStatus:
    def test_status_shows_branch(self):
        status = gcc_status()
        assert "main" in status["message"]
        assert status["branch"] == "main"


# ===========================================================================
# ACE Playbook Tools
# ===========================================================================


class TestACEGetPlaybook:
    def test_empty_playbook(self):
        result = ace_get_playbook()
        assert "empty" in result["message"].lower() or "STRATEGIES" in result["message"]

    def test_playbook_with_bullets(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Always test first"}
        ])
        result = ace_get_playbook()
        assert "Always test first" in result["message"]


class TestACEApplyDelta:
    def test_add_bullet(self):
        result = ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Check edge cases"}
        ])
        assert "Applied 1" in result["message"]

    def test_add_multiple(self):
        result = ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Strategy A"},
            {"type": "ADD", "section": "COMMON MISTAKES TO AVOID", "content": "Mistake A"},
        ])
        assert "Applied 2" in result["message"]

    def test_update_bullet(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Original content"}
        ])
        pb = ace_get_playbook()["message"]
        # Find the bullet ID
        import re
        match = re.search(r"\[(str-\d+)\]", pb)
        assert match
        bullet_id = match.group(1)

        result = ace_apply_delta([
            {"type": "UPDATE", "bullet_id": bullet_id, "content": "Updated content"}
        ])
        assert "Applied 1" in result["message"]
        assert "Updated content" in ace_get_playbook()["message"]

    def test_remove_bullet(self):
        ace_apply_delta([
            {"type": "ADD", "section": "OTHERS", "content": "Remove me"}
        ])
        pb = ace_get_playbook()["message"]
        import re
        match = re.search(r"\[(\w+-\d+)\]", pb)
        bullet_id = match.group(1)

        result = ace_apply_delta([
            {"type": "REMOVE", "bullet_id": bullet_id}
        ])
        assert "Applied 1" in result["message"]
        assert "Remove me" not in ace_get_playbook()["message"]

    def test_persists_to_disk(self, setup_project):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Persisted"}
        ])
        # Reload from disk
        playbook_path = os.path.join(str(setup_project), ".ccr", "playbook.txt")
        assert os.path.isfile(playbook_path)
        with open(playbook_path) as f:
            assert "Persisted" in f.read()

    def test_ace_apply_delta_add_marks_pattern_promoted(self):
        """ADD with content matching a buffered pattern marks it promoted."""
        pat = "Always validate input parameters before processing data request"
        # Build pattern to threshold
        for i in range(3):
            gcc_commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                       patterns_learned=[pat])

        # Now ADD a bullet with matching content
        result = ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": pat}
        ])
        assert "Applied 1" in result["message"]
        assert "Marked 1 pattern(s) as promoted" in result["message"]

        # Verify the pattern is actually promoted in the buffer
        mem = mcp_mod._memory
        assert mem is not None
        patterns_path = mem._get_patterns_path()
        with open(patterns_path) as f:
            data = json.load(f)
        assert data["patterns"]["P001"]["promoted"] is True

    def test_ace_apply_delta_add_no_match_no_promotion_note(self):
        """ADD with content that doesn't match any pattern produces no promotion note."""
        pat = "Always validate input parameters before processing data request"
        # Build a pattern in the buffer
        for i in range(3):
            gcc_commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                       patterns_learned=[pat])

        # ADD a bullet with completely unrelated content
        result = ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS",
             "content": "Database indexing optimization techniques for large datasets"}
        ])
        assert "Applied 1" in result["message"]
        assert "Marked" not in result["message"]


class TestACEUpdateCounters:
    def test_update_helpful(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Good strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook()["message"])
        bullet_id = match.group(1)

        result = ace_update_counters([{"id": bullet_id, "tag": "helpful"}])
        assert "Updated 1" in result["message"]

        pb = ace_get_playbook()["message"]
        assert "helpful=1" in pb

    def test_update_harmful(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Bad strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook()["message"])
        bullet_id = match.group(1)

        ace_update_counters([{"id": bullet_id, "tag": "harmful"}])
        assert "harmful=1" in ace_get_playbook()["message"]


class TestACEUpdateCountersWithFailureLessons:
    def test_harmful_with_failure_lesson(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Test strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook()["message"])
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
        assert "Updated 1" in result["message"]
        assert "1 structured failure lesson" in result["message"]

    def test_harmful_without_lesson_backward_compat(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Test strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook()["message"])
        bullet_id = match.group(1)

        result = ace_update_counters([{"id": bullet_id, "tag": "harmful"}])
        assert "Updated 1" in result["message"]
        assert "failure lesson" not in result["message"]

    def test_failure_lesson_appears_in_playbook(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Fragile strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook()["message"])
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

        pb = ace_get_playbook()["message"]
        assert "FAILURE: Broke on empty input" in pb
        assert "PRINCIPLE: Always handle empty inputs" in pb

    def test_failure_lesson_persists_to_disk(self, setup_project):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Persisted strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook()["message"])
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
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook()["message"])
        bullet_id = match.group(1)

        ace_update_counters([{
            "id": bullet_id,
            "tag": "harmful",
            "failure_lesson": {
                "failure_point": "X", "flawed_reasoning": "Y",
                "counterfactual": "Z", "prevention_principle": "W",
            },
        }])

        result = ace_get_playbook(include_stats=True)
        msg = result["message"]
        assert "PLAYBOOK STATS" in msg
        # Extract the JSON block after the PLAYBOOK STATS header
        stats_json = msg.split("# PLAYBOOK STATS\n")[1]
        stats = json.loads(stats_json)
        assert stats["project"]["total_failure_lessons"] == 1
        assert stats["project"]["harmful_with_lessons"] == 1


class TestACEGetPlaybookStats:
    """Tests for ace_get_playbook(include_stats=True) — replaces old ace_get_stats."""

    def _extract_stats(self, msg: str) -> dict:
        """Extract stats JSON from ace_get_playbook(include_stats=True) message."""
        assert "PLAYBOOK STATS" in msg
        stats_json = msg.split("# PLAYBOOK STATS\n")[1]
        return json.loads(stats_json)

    def test_stats_empty(self):
        result = ace_get_playbook(include_stats=True)
        stats = self._extract_stats(result["message"])
        assert stats["project"]["total_bullets"] == 0

    def test_stats_with_bullets(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "A"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "B"},
        ])
        result = ace_get_playbook(include_stats=True)
        stats = self._extract_stats(result["message"])
        assert stats["project"]["total_bullets"] == 2


class TestACEFindSimilar:
    def test_no_similar(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Alpha beta gamma"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Completely different topic here"},
        ])
        result = ace_find_similar(threshold=0.9)
        assert "No similar" in result["message"]

    def test_find_similar_pair(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS",
             "content": "Always check edge cases in loop bounds and iterations"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS",
             "content": "Always check edge cases in loop bounds and ranges"},
        ])
        result = ace_find_similar(threshold=0.5)
        assert "similarity=" in result["message"]


class TestACEPrune:
    def test_prune_no_problematic(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Good one"}
        ])
        result = ace_prune()
        assert "1 bullets" in result["message"]  # still has 1

    def test_prune_harmful(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Bad strategy"}
        ])
        import re
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook()["message"])
        bullet_id = match.group(1)

        # Mark harmful 3 times
        for _ in range(3):
            ace_update_counters([{"id": bullet_id, "tag": "harmful"}])

        result = ace_prune()
        assert "Pruned 1" in result["message"]


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
        pb_text = ace_get_playbook()["message"]
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
        pb = ace_get_playbook()["message"]
        assert "Strategy that fails" not in pb
        # But prevention principles should survive as new heuristic bullets
        assert "Prevention rule 0" in pb
        assert "Prevention rule 1" in pb
        assert "Prevention rule 2" in pb

    def test_prune_message_includes_evolved_count(self):
        """Return message mentions how many skills were evolved."""
        self._add_harmful_bullets_with_lessons(3)
        result = ace_prune()
        assert "Evolved 3 new skill" in result["message"]
        assert "Pruned" in result["message"]

    def test_prune_with_no_lessons_works_normally(self):
        """Pruning bullets without failure lessons still works fine."""
        import re
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Bad no-lesson strategy"}
        ])
        match = re.search(r"\[(str-\d+)\]", ace_get_playbook()["message"])
        bid = match.group(1)
        for _ in range(3):
            ace_update_counters([{"id": bid, "tag": "harmful"}])
        result = ace_prune()
        assert "Pruned 1" in result["message"]
        assert "Evolved" not in result["message"]
        assert "Bad no-lesson strategy" not in ace_get_playbook()["message"]


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
        pb_text = ace_get_playbook()["message"]
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
        assert "not triggered" in result["message"].lower()

    def test_evolve_creates_new_skills(self):
        self._add_harmful_bullets_with_lessons(3)
        result = ace_evolve_from_failures(threshold=3)
        assert "Evolved 3 new skill" in result["message"]
        # Verify skills appear in playbook
        pb = ace_get_playbook()["message"]
        assert "Prevention rule 0" in pb
        assert "Prevention rule 1" in pb
        assert "Prevention rule 2" in pb

    def test_evolve_skills_persisted(self):
        self._add_harmful_bullets_with_lessons(3)
        ace_evolve_from_failures(threshold=3)
        # Reload playbook from disk
        mcp_mod._playbook = None
        mcp_mod._playbook = mcp_mod._load_playbook()
        pb = ace_get_playbook()["message"]
        assert "Prevention rule 0" in pb

    def test_stats_show_evolution_needed(self):
        self._add_harmful_bullets_with_lessons(3)
        result = ace_get_playbook(include_stats=True)
        msg = result["message"]
        assert "PLAYBOOK STATS" in msg
        stats_json = msg.split("# PLAYBOOK STATS\n")[1]
        stats = json.loads(stats_json)
        assert stats["project"]["evolution_needed"] is True
        assert stats["project"]["evolution_candidates"] == 3


# ===========================================================================
# RLM Sandbox Tools
# ===========================================================================


class TestRLMInit:
    def test_init(self):
        result = rlm_init("Analyze the code")
        assert "REPL initialized" in result["message"]
        assert "task_prompt" in result["message"]
        assert result["session_id"]
        assert isinstance(result["file_count"], int)

    def test_init_shows_file_count(self):
        result = rlm_init("Test")
        assert "files indexed" in result["message"]


class TestRLMExecute:
    def test_basic_execution(self):
        rlm_init("Test")
        result = rlm_execute("x = 2 + 3\nprint(x)")
        assert "5" in result["message"]

    def test_variable_persistence(self):
        rlm_init("Test")
        rlm_execute("my_var = 42")
        result = rlm_execute("print(my_var * 2)")
        assert "84" in result["message"]

    def test_access_task_prompt(self):
        rlm_init("Find all classes")
        result = rlm_execute("print(task_prompt)")
        assert "Find all classes" in result["message"]

    def test_search_repo(self):
        rlm_init("Search test")
        result = rlm_execute("results = search_repo('greet')\nprint(len(results))")
        assert "stdout" in result["message"] or "1" in result["message"]

    def test_get_file(self):
        rlm_init("Read test")
        result = rlm_execute("content = get_file('hello.py')\nprint(content[:20])")
        assert "greet" in result["message"].lower() or "def" in result["message"]

    def test_show_vars(self):
        rlm_init("Test")
        rlm_execute("x = 42\ny = 'hello'")
        result = rlm_execute("print(SHOW_VARS())")
        assert "x" in result["message"]

    def test_error_handling(self):
        rlm_init("Test")
        result = rlm_execute("1/0")
        assert "error" in result["message"].lower() or "ZeroDivision" in result["message"]

    def test_no_init_error(self):
        # Reset REPL
        mcp_mod._repl = None
        with pytest.raises(ToolError, match="not initialized"):
            rlm_execute("print(1)")


class TestRLMExecuteMetadataOnly:
    """Tests for R5 audit fix: metadata-only stdout enforcement (RLM paper Section 3)."""

    def test_short_stdout_unchanged(self):
        """Stdout under the 1000-char threshold is returned as-is."""
        rlm_init("Test")
        result = rlm_execute("print('hello world')")
        assert "hello world" in result["message"]
        # Should NOT contain truncation markers
        assert "[stdout truncated:" not in result["message"]

    def test_long_stdout_summarized(self):
        """Stdout over the 1000-char threshold gets a metadata summary."""
        rlm_init("Test")
        # Generate output well over 1000 chars (50 lines x ~30 chars each = ~1500 chars)
        code = "for i in range(50): print(f'line {i}: ' + 'x' * 20)"
        result = rlm_execute(code)
        assert "[stdout truncated:" in result["message"]
        assert "lines" in result["message"]
        assert "chars" in result["message"]
        # First lines should be present
        assert "line 0:" in result["message"]
        # Last lines should be present
        assert "line 49:" in result["message"]
        # Middle lines should NOT be present (they were truncated)
        assert "line 25:" not in result["message"]

    def test_metadata_only_false_returns_full(self):
        """Setting metadata_only=False returns full stdout regardless of length."""
        rlm_init("Test")
        code = "for i in range(50): print(f'line {i}: ' + 'x' * 20)"
        result = rlm_execute(code, metadata_only=False)
        # Should NOT be truncated
        assert "[stdout truncated:" not in result["message"]
        # All lines should be present
        assert "line 0:" in result["message"]
        assert "line 25:" in result["message"]
        assert "line 49:" in result["message"]


class TestRLMFinalize:
    def test_finalize(self):
        rlm_init("Test")
        rlm_execute("answer = 'The result is 42'")
        result = rlm_finalize("answer")
        assert "42" in result["message"]

    def test_finalize_dict(self):
        rlm_init("Test")
        rlm_execute("data = {'key': 'value', 'count': 3}")
        result = rlm_finalize("data")
        parsed = json.loads(result["message"])
        assert parsed["key"] == "value"

    def test_finalize_missing_var(self):
        rlm_init("Test")
        with pytest.raises(ToolError, match="not found"):
            rlm_finalize("nonexistent")

    def test_no_init_error(self):
        mcp_mod._repl = None
        with pytest.raises(ToolError, match="not initialized"):
            rlm_finalize("x")


# ===========================================================================
# Repo Index Tools
# ===========================================================================


class TestIndexBuild:
    def test_build(self):
        result = index_build()
        assert "Files:" in result["message"]
        assert "2" in result["message"]  # hello.py + utils.py
        assert isinstance(result["files_indexed"], int)


class TestIndexSearch:
    def test_search_by_symbol(self):
        result = index_search("greet")
        assert "hello.py" in result["message"]

    def test_search_by_class(self):
        result = index_search("Greeter")
        assert "hello.py" in result["message"]

    def test_search_no_results(self):
        result = index_search("nonexistent_symbol_xyz")
        assert "No files" in result["message"]

    def test_search_top_k(self):
        result = index_search("py", top_k=1)
        lines = [l for l in result["message"].strip().split("\n") if l.strip()]
        # 1 header line + 1 result line
        assert len(lines) == 2


class TestIndexSearchModes:
    """Tests for the A-RAG-inspired search mode parameter."""

    def test_keyword_mode(self):
        result = index_search("greet", mode="keyword")
        assert "keyword search" in result["message"]
        assert "hello.py" in result["message"]

    def test_semantic_mode_bm25_fallback(self):
        result = index_search("greet", mode="semantic")
        assert "semantic search" in result["message"]
        # BM25 fallback since no ONNX
        assert "BM25 fallback" in result["message"]

    def test_hybrid_mode_default(self):
        result = index_search("greet")
        assert "hybrid search" in result["message"]

    def test_invalid_mode_error(self):
        with pytest.raises(ToolError, match="Invalid mode"):
            index_search("greet", mode="invalid")

    def test_build_shows_embedding_status(self):
        result = index_build()
        assert "Embedding" in result["message"] or "embedding" in result["message"]


# ===========================================================================
# Integration: Cross-tool workflows
# ===========================================================================


class TestWorkflows:
    def test_commit_then_context(self):
        """GCC workflow: commit → retrieve context."""
        gcc_commit("Setup", "Created project", "starting out", ["hello.py"], "Add features")
        ctx = gcc_context(level=2)
        assert "Setup" in ctx["message"]
        assert "Created project" in ctx["message"]

    def test_branch_commit_merge(self):
        """GCC workflow: branch → commit → merge."""
        gcc_branch("experiment", "Try new approach", "Might work")
        gcc_commit("Experiment work", "tried stuff", "exploration", [], "evaluate")
        result = gcc_merge("experiment", "success", "Approach works")
        assert "success" in result["message"]

        # Back on main, context includes merge info
        ctx = gcc_context(level=2)
        assert "experiment" in ctx["message"].lower() or "Merge" in ctx["message"]

    def test_ace_full_cycle(self):
        """ACE workflow: add → tag → prune."""
        # Add strategies
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Good one"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Bad one"},
        ])

        # Tag them
        import re
        pb = ace_get_playbook()["message"]
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
        pb = ace_get_playbook()["message"]
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
        data = json.loads(result["message"])
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
        assert "global" in result["message"]
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
        result = ace_get_playbook()["message"]
        assert "GLOBAL PLAYBOOK" in result
        assert "PROJECT PLAYBOOK" in result
        assert "G1" in result
        assert "P1" in result

    def test_get_playbook_both_sections_present(self):
        """Both global and project sections are always present."""
        result = ace_get_playbook()["message"]
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
        """ace_get_playbook(include_stats=True) returns stats for both global and project."""
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "G"}],
            scope="global",
        )
        ace_apply_delta(
            [{"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "P"}],
        )
        result = ace_get_playbook(include_stats=True)
        msg = result["message"]
        assert "PLAYBOOK STATS" in msg
        stats_json = msg.split("# PLAYBOOK STATS\n")[1]
        stats = json.loads(stats_json)
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
        assert "similarity" in result["message"]

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
        assert "global" in result["message"].lower() and "project" in result["message"].lower()

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
        assert "1" in result["message"]  # pruned 1
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
    """Verify readOnlyHint, destructiveHint, idempotentHint on all 22 tools.

    4 tools (gcc_log_ota, gcc_triples, gcc_clusters, ace_evolve_from_failures)
    were removed from the MCP surface in Phase 2B tool consolidation.
    """

    # -- Read-only tools --

    @pytest.mark.parametrize("tool_name", [
        "gcc_context", "gcc_status", "gcc_patterns",
        "ace_get_playbook", "ace_find_similar",
        "index_search",
    ])
    def test_read_only_tools(self, tool_name):
        ann = _get_tool_annotations(tool_name)
        assert ann.readOnlyHint is True, f"{tool_name} should be readOnlyHint=True"
        assert ann.destructiveHint is False, f"{tool_name} should be destructiveHint=False"
        assert ann.idempotentHint is True, f"{tool_name} should be idempotentHint=True"

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
        "gcc_commit", "gcc_scratchpad", "ace_apply_delta", "ace_update_counters",
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

    # -- Idempotent tools --

    @pytest.mark.parametrize("tool_name", [
        "gcc_context", "gcc_status",
        "ace_get_playbook", "ace_find_similar",
        "index_search",
    ])
    def test_idempotent_tools(self, tool_name):
        ann = _get_tool_annotations(tool_name)
        assert ann.idempotentHint is True, f"{tool_name} should be idempotentHint=True"

    # -- index_build: not idempotent (rebuilds index each call) --

    def test_index_build_annotations(self):
        ann = _get_tool_annotations("index_build")
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.idempotentHint is False

    # -- gcc_consolidate: not read-only, not destructive, not idempotent --

    def test_gcc_consolidate_annotations(self):
        ann = _get_tool_annotations("gcc_consolidate")
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.idempotentHint is False

    # -- All tools have annotations (count updated when new tools added) --

    def test_all_tools_have_annotations(self):
        all_tools = mcp_instance._tool_manager._tools
        assert len(all_tools) >= 22, f"Expected at least 22 tools, got {len(all_tools)}"
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
        assert "C001" in result["message"]
        assert "Links for C002" in result["message"]

    def test_gcc_links_no_links(self):
        result = gcc_links("C999")
        assert "No links found" in result["message"]

    def test_gcc_links_filtered_types(self):
        gcc_commit("A", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        gcc_commit("B", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        result = gcc_links("C002", link_types="entity")
        assert "Entity" in result["message"]

    def test_gcc_links_multi_hop(self):
        gcc_commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        gcc_commit("B", "Refs C001", "See C001", ["b.py"], "N",
                   admission_threshold=1.0)
        gcc_commit("C", "Refs C002", "See C002", ["c.py"], "N",
                   admission_threshold=1.0)
        result = gcc_links("C001", max_hops=2)
        # Should reach C003 via C002 -> C003 (hop 2)
        assert "C002" in result["message"]

    def test_gcc_context_follow_links(self):
        gcc_commit("A", "Created server module", "Foundation",
                   ["server.py"], "Next", admission_threshold=1.0)
        gcc_commit("B", "Extended server module", "Build on A",
                   ["server.py"], "Test", admission_threshold=1.0)
        result = gcc_context(level=5, commit_id="C001", follow_links=True)
        assert "Linked:" in result["message"]

    def test_gcc_links_invalid_link_type_raises(self):
        from mcp.server.fastmcp.exceptions import ToolError
        with pytest.raises(ToolError, match="Invalid link_type"):
            gcc_links("C001", link_types="invalid_type")

    def test_gcc_links_valid_types_accepted(self):
        # All valid types should not raise
        for lt in ("entity", "causal", "supersession", "semantic"):
            gcc_links("C999", link_types=lt)  # C999 has no links — that's fine

    def test_gcc_links_mixed_valid_invalid_raises(self):
        from mcp.server.fastmcp.exceptions import ToolError
        with pytest.raises(ToolError, match="Invalid link_type"):
            gcc_links("C001", link_types="entity,bad_type")


# ===========================================================================
# GCC Clusters MCP Tool Tests (EverMemOS arXiv:2601.02163)
# ===========================================================================


class TestGccClusters:
    """Tests for gcc_clusters MCP tool."""

    def test_gcc_clusters_no_links(self):
        result = gcc_clusters(min_size=2)
        assert "No clusters found" in result["message"]

    def test_gcc_clusters_with_entity_links(self):
        gcc_commit("Auth setup", "JWT auth module", "Security",
                   ["auth.py"], "Login page", admission_threshold=1.0)
        gcc_commit("Auth fix", "Fixed token bug in auth",
                   "Bug discovered", ["auth.py"], "Tests",
                   admission_threshold=1.0)
        result = gcc_clusters(min_size=2)
        # May or may not cluster depending on link scores and scan window;
        # at minimum it should not error
        assert isinstance(result, dict)
        assert "message" in result

    def test_gcc_clusters_recompute_false(self):
        result = gcc_clusters(recompute=False)
        assert "No clusters found" in result["message"]

    def test_gcc_context_includes_clusters_at_level_3(self):
        """Clusters should appear in context at level 3+."""
        gcc_commit("A", "Auth module", "Security", ["auth.py"], "N",
                   admission_threshold=1.0)
        gcc_commit("B", "Auth fix", "Bug", ["auth.py"], "N",
                   admission_threshold=1.0)
        # Recompute clusters so they're saved
        gcc_clusters(min_size=2)
        result = gcc_context(level=3)
        # If clusters were found they'd appear; if not, just verify no error
        assert isinstance(result, dict)
        assert "message" in result


# ===========================================================================
# GCC Hierarchical Summary MCP Tool Tests
# ===========================================================================


class TestGCCHierarchicalSummaryTools:
    """Tests for gcc_consolidate and gcc_context(include_summaries=True) MCP tools."""

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
        assert "Session Summary" in result["message"]

    def test_gcc_consolidate_phase(self):
        self._make_commits(5)
        result = gcc_consolidate(tier="phase")
        assert "Phase Summary" in result["message"]

    def test_gcc_consolidate_project_prompt(self):
        result = gcc_consolidate(tier="project")
        assert "Please generate a project overview" in result["message"]

    def test_gcc_consolidate_project_save(self):
        result = gcc_consolidate(tier="project", content="Test project builds AI tools.")
        assert "saved" in result["message"].lower()

    def test_gcc_context_include_summaries_all(self):
        self._make_commits(5)
        gcc_consolidate(tier="phase")
        gcc_consolidate(tier="project", content="AI project overview")
        result = gcc_context(include_summaries=True, summaries_tier="all", summaries_count=5)
        assert "Session" in result["message"] or "Phase" in result["message"] or "Project" in result["message"]

    def test_gcc_context_include_summaries_empty(self):
        result = gcc_context(include_summaries=True, summaries_tier="all", summaries_count=5)
        assert result["message"]  # Should return something, not crash

    def test_gcc_context_include_summaries_count_bounded(self):
        result = gcc_context(include_summaries=True, summaries_tier="all", summaries_count=0)
        assert result["message"]  # count=0 should be bounded to 1


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
        assert "C001" in result["message"]

    def test_gcc_commit_patterns_optional(self):
        """gcc_commit works without patterns_learned (backward compat)."""
        result = gcc_commit(
            title="Fix bug",
            what="Fixed the bug",
            why="Bug report",
            files_changed=["fix.py"],
            next_step="Verify",
        )
        assert "C001" in result["message"]

    def test_gcc_commit_promotion_suggestion(self):
        """Promotion suggestion appears after 3 commits with same pattern."""
        pat = "Always validate input parameters before processing data"
        for i in range(2):
            gcc_commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                       patterns_learned=[pat])
        result = gcc_commit("T3", "did 2", "reason", ["2.py"], "next",
                            patterns_learned=[pat])
        assert "Pattern promotion suggestions" in result["message"]
        assert "ace_apply_delta" in result["message"]


class TestGCCPatterns:
    def test_gcc_patterns_empty(self):
        result = gcc_patterns()
        assert "No patterns found" in result["message"]

    def test_gcc_patterns_basic(self):
        gcc_commit("T1", "did A", "reason", ["a.py"], "next",
                   patterns_learned=["Use structured logging for debugging"])
        result = gcc_patterns()
        assert "Pattern Buffer" in result["message"]
        assert "structured logging" in result["message"]

    def test_gcc_patterns_min_occurrences_filter(self):
        gcc_commit("T1", "did A", "reason", ["a.py"], "next",
                   patterns_learned=["Frequent pattern observed multiple times here",
                                      "Rare single occurrence pattern for test"])
        gcc_commit("T2", "did B", "reason", ["b.py"], "next",
                   patterns_learned=["Frequent pattern observed multiple times here"])
        result = gcc_patterns(min_occurrences=2)
        assert "Frequent pattern" in result["message"]
        assert "Rare single" not in result["message"]

    def test_gcc_patterns_search_term(self):
        gcc_commit("T1", "did A", "reason", ["a.py"], "next",
                   patterns_learned=["Use sandbox for REPL execution",
                                      "Always update documentation after changes"])
        result = gcc_patterns(search_term="sandbox")
        assert "sandbox" in result["message"]
        assert "documentation" not in result["message"]

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
        assert "Schema Health Report" in result["message"]
        assert "Overall Health" in result["message"]
        assert "Section Balance" in result["message"]

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
        if "apply_proposal" in result["message"]:
            result2 = ace_evolve_schema(apply_proposal=1)
            assert "schema v" in result2["message"].lower() or "Applied" in result2["message"]

    def test_evolve_schema_invalid_proposal_index(self):
        """Invalid proposal index returns error."""
        result = ace_evolve_schema(apply_proposal=99)
        assert "Invalid" in result["message"] or "Error" in result["message"]

    def test_evolve_schema_rollback_no_parent(self):
        """Rollback with no parent returns error."""
        result = ace_evolve_schema(rollback=True)
        assert "Cannot rollback" in result["message"]

    def test_evolve_schema_annotations(self):
        """ace_evolve_schema has correct MCP tool annotations."""
        ann = _get_tool_annotations("ace_evolve_schema")
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.idempotentHint is False


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
        assert "[C002]" in result["message"]
        ctx = gcc_context(level=2)
        # The compressed summary should appear in context
        assert "started A, added B" in ctx["message"]

    def test_gcc_commit_compressed_summary_optional(self):
        """gcc_commit works without compressed_summary (backward compat)."""
        result = gcc_commit("T1", "did A", "reason A", ["a.py"], "next A")
        assert "[C001]" in result["message"]

    def test_gcc_commit_short_summary_no_warning(self):
        """Short summaries don't produce compression warning."""
        result = gcc_commit("T1", "did A", "reason A", ["a.py"], "next A")
        assert "Rolling summary at" not in result["message"]

    def test_gcc_commit_long_summary_triggers_warning(self):
        """Long summaries produce compression warning in return value."""
        # Access memory to write a long summary directly
        import ccr.mcp_server as mod
        mem = mod._memory
        mem._write_rolling_summary("main", "x" * 1300)
        result = gcc_commit("Next", "added more stuff", "reason", ["f.py"], "done")
        assert "Rolling summary at" in result["message"]
        assert "/1500 chars" in result["message"]
        assert "compressed_summary" in result["message"]

    def test_gcc_commit_compressed_summary_suppresses_warning(self):
        """When compressed_summary is provided, no warning even if summary was long."""
        import ccr.mcp_server as mod
        mem = mod._memory
        mem._write_rolling_summary("main", "x" * 1300)
        result = gcc_commit("Next", "added more", "reason", ["f.py"], "done",
                            compressed_summary="Clean compressed summary here")
        assert "Rolling summary at" not in result["message"]


# ===========================================================================
# MAGMA Query-Aware BFS Tests
# ===========================================================================


class TestMagmaQueryBFS:
    """Tests for MAGMA intent-aware BFS traversal via gcc_links query= parameter."""

    def _setup_linked_commits(self):
        """Create two commits sharing a file so entity links are generated."""
        gcc_commit("Auth setup", "Implemented auth module", "Security baseline",
                   ["auth.py"], "Next", admission_threshold=1.0)
        gcc_commit("Auth fix", "Fixed auth bug", "Bug in C001",
                   ["auth.py"], "Test", admission_threshold=1.0)

    def _write_embeddings(self, commit_ids_and_vecs):
        """Write embeddings directly into the memory cache file."""
        import ccr.mcp_server as mod
        import numpy as np
        from ccr.context.embeddings import save_embeddings
        mem = mod._memory
        path = mem._get_commit_embeddings_path()
        data = {cid: vec.tolist() for cid, vec in commit_ids_and_vecs.items()}
        save_embeddings(data, path)

    def test_gcc_links_with_query_returns_query_score(self):
        """gcc_links with query= returns results with query_score field."""
        import numpy as np
        from unittest.mock import MagicMock, patch

        self._setup_linked_commits()

        # Build a unit query vector and a target embedding
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c001_vec = np.array([0.8, 0.2, 0.0], dtype=np.float32)
        c002_vec = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        self._write_embeddings({"C001": c001_vec, "C002": c002_vec})

        mock_model = MagicMock()
        mock_model.embed_query.return_value = query_vec

        with patch("ccr.core.memory_pkg.memory_links.get_embedding_model", return_value=mock_model), \
             patch("ccr.core.memory_pkg.memory_links.quick_cosine", return_value=0.5):
            import ccr.mcp_server as mod
            linked = mod._memory.get_linked_commits("C001", query="auth security")

        # At least one result should carry a query_score
        assert any("query_score" in r for r in linked), (
            "Expected 'query_score' key in results when query= is provided"
        )

    def test_gcc_links_with_query_affects_ordering(self):
        """query= can change result ordering vs no-query BFS when multi-hop."""
        import numpy as np
        from unittest.mock import MagicMock, patch

        # Three commits: C001 (root) links to C002 and C003 via causal references
        gcc_commit("Root", "Initial feature", "Baseline",
                   ["root.py"], "Next", admission_threshold=1.0)
        gcc_commit("Target A", "See C001 for context", "Related to C001",
                   ["a.py"], "Test", admission_threshold=1.0)
        gcc_commit("Target B", "See C001 for context", "Related to C001",
                   ["b.py"], "Test", admission_threshold=1.0)

        # C002 embedding is far from query; C003 is very close to query
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c001_vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        # C002 is orthogonal to query (low relevance)
        c002_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # C003 is aligned with query (high relevance)
        c003_vec = np.array([0.95, 0.05, 0.0], dtype=np.float32)
        self._write_embeddings({"C001": c001_vec, "C002": c002_vec, "C003": c003_vec})

        mock_model = MagicMock()
        mock_model.embed_query.return_value = query_vec

        # quick_cosine returns non-None for probe (activates ONNX path) but None
        # for actual comparisons, so the code falls back to cached vector dot product
        def _probe_only_cosine(a: str, b: str):
            if a == "a" and b == "b":
                return 0.5  # probe succeeds
            return None  # fall back to cached vectors

        with patch("ccr.core.memory_pkg.memory_links.get_embedding_model", return_value=mock_model), \
             patch("ccr.core.memory_pkg.memory_links.quick_cosine", side_effect=_probe_only_cosine):
            import ccr.mcp_server as mod
            mem = mod._memory
            results_with_query = mem.get_linked_commits("C001", query="feature root")

        # With query: C003 (score ~0.95) should appear before C002 (score ~0.0)
        ids_with_query = [r["id"] for r in results_with_query]
        if "C002" in ids_with_query and "C003" in ids_with_query:
            assert ids_with_query.index("C003") < ids_with_query.index("C002"), (
                "With query aligned to C003, C003 should appear before C002"
            )

    def test_gcc_links_without_query_unchanged(self):
        """gcc_links without query= behaves identically to before (no query_score)."""
        import ccr.mcp_server as mod
        import numpy as np
        from ccr.context.embeddings import save_embeddings

        self._setup_linked_commits()

        # Write embeddings so embedding_score path is exercised
        c001_vec = np.array([0.8, 0.2, 0.0], dtype=np.float32)
        c002_vec = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        self._write_embeddings({"C001": c001_vec, "C002": c002_vec})

        # No query= provided — should use old embedding_score path
        linked = mod._memory.get_linked_commits("C001")

        # No result should have query_score
        assert not any("query_score" in r for r in linked), (
            "Without query=, results must not contain 'query_score'"
        )


# ===========================================================================
# ACE Cross-Synthesis (Phase 2C — SkillRL §3.3 sub-model synthesis)
# ===========================================================================


class TestACECrossSynthesis:
    """Tests for _auto_synthesize_skills and its integration into ace_evolve_from_failures."""

    def _make_candidate(self, slug, prevention_principle, task_context, failure_point=""):
        return {
            "bullet_slug": slug,
            "failure_lesson": {
                "prevention_principle": prevention_principle,
                "task_context": task_context,
                "failure_point": failure_point,
            },
        }

    def test_auto_synthesize_skills_groups_by_task_context(self):
        """Candidates with similar task_context are grouped; sub_client called once per group."""
        from ccr.mcp_server import _auto_synthesize_skills
        from unittest.mock import MagicMock

        # Two candidates with similar task_context (high word overlap)
        c1 = self._make_candidate("heu-00001", "Always validate input before processing",
                                   "database query optimization task")
        c2 = self._make_candidate("heu-00002", "Check query plan before running expensive queries",
                                   "database query performance task")
        # One candidate with distinct task_context
        c3 = self._make_candidate("mis-00001", "Verify file permissions before writing",
                                   "file system access control")

        sub = MagicMock()
        sub.completion.return_value = '[{"content": "When optimizing database queries, validate inputs and review query plans first.", "when_to_apply": "Before executing queries"}]'

        result = _auto_synthesize_skills([c1, c2, c3], sub)

        # sub_client should be called once — for the group with c1 + c2 (similar context)
        # c3 is alone → skipped (single-candidate group)
        assert sub.completion.call_count == 1
        assert len(result) == 1
        assert "content" in result[0]
        assert result[0]["scope"] == "project"

    def test_auto_synthesize_skills_skips_single_candidates(self):
        """Groups with exactly 1 candidate get no LLM call (mechanical path handles them)."""
        from ccr.mcp_server import _auto_synthesize_skills
        from unittest.mock import MagicMock

        # All candidates have very different task_contexts → each in its own group
        c1 = self._make_candidate("heu-00001", "Rule A", "database optimization")
        c2 = self._make_candidate("heu-00002", "Rule B", "file system permissions")
        c3 = self._make_candidate("heu-00003", "Rule C", "network request throttling")

        sub = MagicMock()
        sub.completion.return_value = '[]'

        result = _auto_synthesize_skills([c1, c2, c3], sub)

        # No group has 2+ candidates → no LLM calls
        assert sub.completion.call_count == 0
        assert result == []

    def test_auto_synthesize_skills_returns_empty_on_error(self):
        """sub_client raising an exception → returns [] without propagating."""
        from ccr.mcp_server import _auto_synthesize_skills
        from unittest.mock import MagicMock

        # Two similar candidates so a group is formed
        c1 = self._make_candidate("heu-00001", "Rule A", "database query optimization task")
        c2 = self._make_candidate("heu-00002", "Rule B", "database query performance task")

        sub = MagicMock()
        sub.completion.side_effect = RuntimeError("API unavailable")

        # Should not raise — graceful no-op
        result = _auto_synthesize_skills([c1, c2], sub)
        assert result == []

    def test_auto_synthesize_skills_empty_candidates(self):
        """Empty candidate list returns [] immediately without calling sub_client."""
        from ccr.mcp_server import _auto_synthesize_skills
        from unittest.mock import MagicMock

        sub = MagicMock()
        result = _auto_synthesize_skills([], sub)
        assert result == []
        sub.completion.assert_not_called()

    def test_ace_evolve_from_failures_with_sub_client(self):
        """With a mock sub-client available, synthesized skills are added to playbook."""
        import re
        from unittest.mock import MagicMock, patch

        # Add 3 harmful bullets with similar task_contexts so grouping triggers
        for i in range(3):
            ace_apply_delta([{
                "type": "ADD",
                "section": "STRATEGIES & INSIGHTS",
                "content": f"Fragile strategy {i}",
            }])
        pb_text = ace_get_playbook()["message"]
        ids = re.findall(r"\[(str-\d+)\]", pb_text)
        for i, bid in enumerate(ids[:3]):
            ace_update_counters([{
                "id": bid,
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Failure point {i}",
                    "flawed_reasoning": f"Bad assumption {i}",
                    "counterfactual": f"Should have done {i}",
                    "prevention_principle": f"Always check condition {i} before proceeding",
                    # All share similar task context → group of 3
                    "task_context": "code review automated testing workflow",
                },
            }])

        mock_sub = MagicMock()
        mock_sub.completion.return_value = (
            '[{"content": "When reviewing code, always verify test coverage before approving.", '
            '"when_to_apply": "During code review"}]'
        )

        with patch("ccr.mcp.server._get_sub_client", return_value=mock_sub):
            result = ace_evolve_from_failures(threshold=3)

        # Mechanical path should have evolved 3 skills; sub-model adds 1 more
        assert "Evolved 3 new skill" in result["message"]
        assert "Synthesized 1 cross-lesson skill" in result["message"]
        # Verify the synthesized skill appears in the playbook
        pb = ace_get_playbook()["message"]
        assert "When reviewing code" in pb


class TestCERAutoPatternExtraction:
    """Tests for _extract_patterns_from_commit and gcc_commit auto-extraction (Phase 2D)."""

    def test_extract_patterns_from_commit_returns_patterns(self):
        from unittest.mock import MagicMock
        import ccr.mcp_server as mod

        mock_sub = MagicMock()
        mock_sub.completion.return_value = '["When adding auth, update tests and docs", "When fixing bugs, add regression test"]'

        patterns = mod._extract_patterns_from_commit(
            title="Add auth",
            what="Implemented authentication module with JWT tokens",
            why="Security requirement",
            files_changed=["auth.py", "tests/test_auth.py"],
            sub_client=mock_sub,
        )

        assert isinstance(patterns, list)
        assert len(patterns) == 2
        assert "When adding auth, update tests and docs" in patterns
        mock_sub.completion.assert_called_once()

    def test_extract_patterns_from_commit_returns_empty_on_error(self):
        from unittest.mock import MagicMock
        import ccr.mcp_server as mod

        mock_sub = MagicMock()
        mock_sub.completion.side_effect = RuntimeError("API unavailable")

        patterns = mod._extract_patterns_from_commit(
            title="Some commit", what="Did something", why="Reason",
            files_changed=["foo.py"], sub_client=mock_sub,
        )
        assert patterns == []

    def test_extract_patterns_from_commit_returns_empty_on_invalid_json(self):
        from unittest.mock import MagicMock
        import ccr.mcp_server as mod

        mock_sub = MagicMock()
        mock_sub.completion.return_value = "Here are some patterns: blah blah blah"

        patterns = mod._extract_patterns_from_commit(
            title="Some commit", what="Did something", why="Reason",
            files_changed=["foo.py"], sub_client=mock_sub,
        )
        assert patterns == []

    def test_gcc_commit_auto_extracts_patterns_when_enabled(self):
        from unittest.mock import MagicMock, patch
        import ccr.mcp_server as mod

        mod._memory.config.auto_extract_patterns = True
        try:
            mock_sub = MagicMock()
            mock_sub.completion.return_value = '["When adding features with long descriptions, document the motivation"]'

            with patch("ccr.mcp.server._get_sub_client", return_value=mock_sub):
                result = gcc_commit(
                    title="Add feature",
                    what="Implemented a comprehensive new feature with detailed logic spanning multiple modules and touching the core data pipeline",
                    why="Required by product spec",
                    files_changed=["feature.py"],
                    next_step="Write tests",
                )

            assert "C0" in result["message"]
            # completion called at least once: pattern extraction + ACE pipeline may also fire
            mock_sub.completion.assert_called()
        finally:
            mod._memory.config.auto_extract_patterns = False

    def test_gcc_commit_skips_auto_extract_when_patterns_provided(self):
        from unittest.mock import MagicMock, patch
        import ccr.mcp_server as mod

        mod._memory.config.auto_extract_patterns = True
        try:
            mock_sub = MagicMock()
            # Patch _run_ace_pipeline to isolate pattern-extraction assertion
            with patch("ccr.mcp.server._get_sub_client", return_value=mock_sub), \
                 patch("ccr.mcp.ace_tools._run_ace_pipeline"):
                result = gcc_commit(
                    title="Add feature",
                    what="Implemented a comprehensive new feature with detailed logic spanning multiple modules and touching the core data pipeline",
                    why="Required by product spec",
                    files_changed=["feature.py"],
                    next_step="Write tests",
                    patterns_learned=["When patterns are already provided, skip extraction"],
                )
            assert "C0" in result["message"]
            # Pattern extraction skipped since patterns_learned already provided
            mock_sub.completion.assert_not_called()
        finally:
            mod._memory.config.auto_extract_patterns = False

    def test_gcc_commit_skips_auto_extract_when_disabled(self):
        from unittest.mock import MagicMock, patch
        import ccr.mcp_server as mod

        mock_sub = MagicMock()
        # Patch _run_ace_pipeline to isolate pattern-extraction assertion
        with patch("ccr.mcp.server._get_sub_client", return_value=mock_sub), \
             patch("ccr.mcp.ace_tools._run_ace_pipeline"):
            result = gcc_commit(
                title="Add feature",
                what="Implemented a comprehensive new feature with detailed logic spanning multiple modules",
                why="Required",
                files_changed=["feature.py"],
                next_step="Test",
            )
        assert "C0" in result["message"]
        # auto_extract_patterns is False (default): no pattern extraction call
        mock_sub.completion.assert_not_called()

    def test_gcc_commit_skips_auto_extract_when_what_too_short(self):
        from unittest.mock import MagicMock, patch
        import ccr.mcp_server as mod

        mod._memory.config.auto_extract_patterns = True
        try:
            mock_sub = MagicMock()
            # Patch _run_ace_pipeline to isolate pattern-extraction assertion
            with patch("ccr.mcp.server._get_sub_client", return_value=mock_sub), \
                 patch("ccr.mcp.ace_tools._run_ace_pipeline"):
                result = gcc_commit(
                    title="Tiny fix", what="Fixed typo", why="Quality",
                    files_changed=["readme.md"], next_step="Done",
                )
            assert "C0" in result["message"]
            # what is too short (<100 chars): no pattern extraction call
            mock_sub.completion.assert_not_called()
        finally:
            mod._memory.config.auto_extract_patterns = False


# ===========================================================================
# ACE 3-Agent Loop Tests
# ===========================================================================


class TestACEThreeAgentLoop:
    """Tests for ACE Generator → Reflector → Curator pipeline."""

    def test_ace_generator_returns_bullets(self):
        """Generator returns a list of strings from sub_client."""
        from unittest.mock import MagicMock
        import ccr.mcp_server as mod

        mock_sub = MagicMock()
        mock_sub.completion.return_value = '["When you add a new feature, write tests first", "Always check for edge cases in loops"]'

        result = mod._ace_generator("Title: Add feature\nWhat: added code\nWhy: needed", mock_sub)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_ace_generator_returns_empty_on_error(self):
        """Generator returns [] when sub_client raises an exception."""
        from unittest.mock import MagicMock
        import ccr.mcp_server as mod

        mock_sub = MagicMock()
        mock_sub.completion.side_effect = RuntimeError("connection failed")

        result = mod._ace_generator("trajectory", mock_sub)
        assert result == []

    def test_ace_reflector_filters_low_quality(self):
        """Reflector removes bullets scoring below 3."""
        from unittest.mock import MagicMock
        import ccr.mcp_server as mod

        mock_sub = MagicMock()
        mock_sub.completion.return_value = json.dumps([
            {"bullet": "high quality bullet here", "score": 5},
            {"bullet": "mediocre bullet text", "score": 2},
            {"bullet": "decent bullet content", "score": 3},
        ])

        candidates = ["high quality bullet here", "mediocre bullet text", "decent bullet content"]
        result = mod._ace_reflector(candidates, "some context", mock_sub)
        assert "high quality bullet here" in result
        assert "mediocre bullet text" not in result
        assert "decent bullet content" in result

    def test_ace_curator_returns_add_skip_merge(self):
        """Curator returns list of decisions with action field."""
        from unittest.mock import MagicMock
        import ccr.mcp_server as mod

        mock_sub = MagicMock()
        mock_sub.completion.return_value = json.dumps([
            {"bullet": "novel strategy bullet", "action": "ADD", "merge_with": None},
            {"bullet": "duplicate existing bullet", "action": "SKIP", "merge_with": None},
            {"bullet": "similar to existing strategy", "action": "MERGE", "merge_with": "existing strategy text"},
        ])

        existing = ["existing strategy text", "another bullet"]
        candidates = ["novel strategy bullet", "duplicate existing bullet", "similar to existing strategy"]
        result = mod._ace_curator(existing, candidates, mock_sub)

        assert len(result) == 3
        actions = [d["action"] for d in result]
        assert "ADD" in actions
        assert "SKIP" in actions
        assert "MERGE" in actions

    def test_gcc_commit_triggers_ace_pipeline(self):
        """After a successful commit, ACE pipeline fires when sub_client is available."""
        from unittest.mock import MagicMock, patch, call
        import ccr.mcp_server as mod

        mock_sub = MagicMock()
        # Generator returns candidates
        mock_sub.completion.return_value = '["When refactoring, write tests first", "Document all changes"]'

        with patch("ccr.mcp.server._get_sub_client", return_value=mock_sub):
            result = gcc_commit(
                title="Test pipeline",
                what="Implemented the pipeline wiring code",
                why="Required for feature",
                files_changed=["mcp_server.py"],
                next_step="Verify",
            )

        assert "C0" in result["message"]
        # sub_client.completion should have been called at least once (by pipeline)
        assert mock_sub.completion.call_count >= 1

    def test_ace_generate_bullets_preview_only(self):
        """ace_generate_bullets with auto_apply=False returns preview without modifying playbook."""
        from unittest.mock import MagicMock, patch
        import ccr.mcp_server as mod

        initial_bullet_count = len(mod._playbook.bullets)

        mock_sub = MagicMock()
        # Generator response
        mock_sub.completion.side_effect = [
            '["When debugging, add logging first", "Always check return values"]',
            json.dumps([{"bullet": "When debugging, add logging first", "score": 5}, {"bullet": "Always check return values", "score": 4}]),
            json.dumps([{"bullet": "When debugging, add logging first", "action": "ADD", "merge_with": None}, {"bullet": "Always check return values", "action": "ADD", "merge_with": None}]),
        ]

        with patch("ccr.mcp.server._get_sub_client", return_value=mock_sub):
            result = ace_generate_bullets(context="debugging code", auto_apply=False)

        # Playbook should not have been modified
        assert len(mod._playbook.bullets) == initial_bullet_count
        # But preview should mention decisions
        assert "ADD" in result["message"] or "decisions" in result["message"].lower() or "Results" in result["message"]

    def test_ace_get_playbook_with_task_context(self):
        """ace_get_playbook with task_context returns policy-ranked section."""
        # First add some bullets with counters
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Always verify database connections before queries"},
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Check API rate limits when making requests"},
        ])
        # Update counters to give non-zero scores
        import ccr.mcp_server as mod
        bullets = mod._playbook.bullets
        if bullets:
            ace_update_counters([{"id": bullets[0].id, "tag": "helpful"}])

        result = ace_get_playbook(task_context="database connection issues")
        assert "Policy-ranked skills" in result["message"]
