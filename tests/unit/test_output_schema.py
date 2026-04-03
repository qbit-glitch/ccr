"""Tests verifying MCP outputSchema compliance — every tool returns a TypedDict, not a str.

Each tool must:
  1. Return a dict (not a plain string)
  2. Include all expected keys per its TypedDict definition
  3. Have a 'message' field (str) for backward compatibility
  4. Raise ToolError (not return "Error: ...") on genuine errors
"""

import json
import os

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
    fake_home_ccr = tmp_path / "global_ccr"
    fake_home_ccr.mkdir()
    original_expanduser = os.path.expanduser

    def mock_expanduser(path):
        if path.startswith("~/.ccr"):
            return str(fake_home_ccr) + path[5:]
        return original_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

    src = tmp_path / "hello.py"
    src.write_text("def greet(name):\n    return f'Hello, {name}!'\n\nclass Greeter:\n    pass\n")
    (tmp_path / "utils.py").write_text("def helper():\n    pass\n")

    _init(str(tmp_path))
    yield


# ===========================================================================
# Tool count verification
# ===========================================================================


class TestToolCount:
    def test_exactly_22_tools_registered(self):
        """MCP server exposes at least 22 tools (index_status added in v4)."""
        all_tools = mcp_instance._tool_manager._tools
        assert len(all_tools) >= 22, f"Expected at least 22 tools, got {len(all_tools)}"


# ===========================================================================
# GCC tools return dicts with expected keys
# ===========================================================================


class TestGccCommitSchema:
    def test_returns_dict(self):
        result = gcc_commit("Title", "What", "Why", ["f.py"], "Next")
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = gcc_commit("Title", "What", "Why", ["f.py"], "Next")
        assert "commit_id" in result
        assert "branch" in result
        assert "title" in result
        assert "admission_decision" in result
        assert "message" in result

    def test_message_is_str(self):
        result = gcc_commit("Title", "What", "Why", ["f.py"], "Next")
        assert isinstance(result["message"], str)

    def test_commit_id_populated(self):
        result = gcc_commit("Title", "What", "Why", ["f.py"], "Next")
        assert result["commit_id"].startswith("C")

    def test_admission_decision_valid(self):
        result = gcc_commit("Title", "What", "Why", ["f.py"], "Next")
        assert result["admission_decision"] in ("created", "merged", "rejected")

    def test_error_raises_tool_error(self):
        """Genuine errors raise ToolError, not return error strings."""
        from unittest.mock import patch
        with patch.object(mcp_mod, "_ensure_memory", side_effect=RuntimeError("boom")):
            with pytest.raises(ToolError, match="boom"):
                gcc_commit("T", "W", "W", ["f"], "N")


class TestGccBranchSchema:
    def test_returns_dict_with_keys(self):
        result = gcc_branch("exp", "Try experiment", "Might work")
        assert isinstance(result, dict)
        assert "branch" in result
        assert "message" in result
        assert result["branch"] == "exp"


class TestGccMergeSchema:
    def test_returns_dict_with_keys(self):
        gcc_branch("feat", "Feature branch", "Testing")
        result = gcc_merge("feat", "success", "Completed")
        assert isinstance(result, dict)
        assert "source" in result
        assert "target" in result
        assert "message" in result
        assert result["source"] == "feat"


class TestGccContextSchema:
    def test_returns_dict_with_keys(self):
        gcc_commit("T", "W", "R", ["f.py"], "N")
        result = gcc_context(level=2)
        assert isinstance(result, dict)
        assert "level" in result
        assert "branch" in result
        assert "message" in result
        assert result["level"] == 2


class TestGccLinksSchema:
    def test_returns_dict_with_keys(self):
        result = gcc_links("C999")
        assert isinstance(result, dict)
        assert "commit_id" in result
        assert "links_found" in result
        assert "message" in result
        assert result["commit_id"] == "C999"
        assert result["links_found"] == 0


class TestGccEvolveMemorySchema:
    def test_returns_dict_with_keys(self):
        gcc_commit("T", "W", "R", ["f.py"], "N")
        result = mcp_mod.gcc_evolve_memory()
        assert isinstance(result, dict)
        assert "evolutions" in result
        assert "message" in result


class TestGccLogOtaSchema:
    def test_returns_dict_with_keys(self):
        result = gcc_log_ota("test observation", "testing", "action taken")
        assert isinstance(result, dict)
        assert "message" in result


class TestGccStatusSchema:
    def test_returns_dict_with_keys(self):
        result = gcc_status()
        assert isinstance(result, dict)
        assert "branch" in result
        assert "total_commits" in result
        assert "message" in result
        assert isinstance(result["total_commits"], int)


class TestGccConsolidateSchema:
    def test_returns_dict_with_keys(self):
        for i in range(5):
            gcc_commit(f"T{i}", f"W{i}", "R", [f"{i}.py"], "N", admission_threshold=1.0)
        result = gcc_consolidate(tier="session")
        assert isinstance(result, dict)
        assert "tier" in result
        assert "message" in result
        assert result["tier"] == "session"


class TestGccContextSummariesSchema:
    def test_returns_dict_with_keys(self):
        result = gcc_context(include_summaries=True, summaries_tier="all", summaries_count=5)
        assert isinstance(result, dict)
        assert "level" in result
        assert "message" in result


class TestGccPatternsSchema:
    def test_returns_dict_with_keys_empty(self):
        result = gcc_patterns()
        assert isinstance(result, dict)
        assert "total" in result
        assert "matching" in result
        assert "message" in result
        assert result["total"] == 0

    def test_returns_dict_with_patterns(self):
        gcc_commit("T", "W", "R", ["f.py"], "N",
                   patterns_learned=["Always test before committing"])
        result = gcc_patterns()
        assert result["total"] > 0


class TestGccClustersSchema:
    def test_returns_dict_with_keys_empty(self):
        result = gcc_clusters(min_size=2, recompute=True)
        assert isinstance(result, dict)
        assert "cluster_count" in result
        assert "message" in result
        assert result["cluster_count"] == 0

    def test_message_is_str(self):
        result = gcc_clusters()
        assert isinstance(result["message"], str)


class TestGccTriplesSchema:
    def test_returns_dict_with_keys_empty(self):
        result = gcc_triples()
        assert isinstance(result, dict)
        assert "count" in result
        assert "message" in result

    def test_returns_dict_with_query(self):
        gcc_commit("Add feature", "Added new feature to memory", "Needed it", ["f.py"], "Next")
        result = gcc_triples(query="feature")
        assert isinstance(result, dict)
        assert "count" in result
        assert "message" in result

    def test_message_is_str(self):
        result = gcc_triples()
        assert isinstance(result["message"], str)


class TestGccScratchpadSchema:
    def test_set_returns_dict_with_keys(self):
        result = gcc_scratchpad(mode="set", key="test_key", value="test_value")
        assert isinstance(result, dict)
        assert "mode" in result
        assert "key" in result
        assert "message" in result
        assert result["mode"] == "set"
        assert result["key"] == "test_key"

    def test_set_message_is_str(self):
        result = gcc_scratchpad(mode="set", key="k", value="v")
        assert isinstance(result["message"], str)

    def test_get_returns_dict_with_keys_existing(self):
        gcc_scratchpad(mode="set", key="mykey", value="myval")
        result = gcc_scratchpad(mode="get", key="mykey")
        assert isinstance(result, dict)
        assert "mode" in result
        assert "key" in result
        assert "message" in result
        assert result["mode"] == "get"
        assert result["key"] == "mykey"

    def test_get_returns_dict_with_keys_missing(self):
        result = gcc_scratchpad(mode="get", key="nonexistent")
        assert isinstance(result, dict)
        assert "mode" in result
        assert "key" in result
        assert "message" in result
        assert result["mode"] == "get"
        assert result["key"] == "nonexistent"

    def test_get_returns_dict_all_entries(self):
        result = gcc_scratchpad(mode="get")
        assert isinstance(result, dict)
        assert "mode" in result
        assert "message" in result
        assert result["mode"] == "get"

    def test_get_message_is_str(self):
        result = gcc_scratchpad(mode="get")
        assert isinstance(result["message"], str)

    def test_clear_returns_dict_with_keys_clear_all(self):
        gcc_scratchpad(mode="set", key="a", value="1")
        result = gcc_scratchpad(mode="clear")
        assert isinstance(result, dict)
        assert "mode" in result
        assert "cleared" in result
        assert "message" in result
        assert result["mode"] == "clear"
        assert isinstance(result["cleared"], int)

    def test_clear_returns_dict_with_keys_clear_specific(self):
        gcc_scratchpad(mode="set", key="b", value="2")
        result = gcc_scratchpad(mode="clear", key="b")
        assert isinstance(result, dict)
        assert "mode" in result
        assert "cleared" in result
        assert "message" in result
        assert result["mode"] == "clear"
        assert result["cleared"] == 1

    def test_clear_returns_dict_with_keys_clear_missing(self):
        result = gcc_scratchpad(mode="clear", key="nonexistent")
        assert isinstance(result, dict)
        assert result["mode"] == "clear"
        assert result["cleared"] == 0

    def test_clear_message_is_str(self):
        result = gcc_scratchpad(mode="clear")
        assert isinstance(result["message"], str)


# ===========================================================================
# ACE tools return dicts with expected keys
# ===========================================================================


class TestAcePlaybookSchema:
    def test_returns_dict_with_keys(self):
        result = ace_get_playbook()
        assert isinstance(result, dict)
        assert "global_bullet_count" in result
        assert "project_bullet_count" in result
        assert "message" in result
        assert isinstance(result["global_bullet_count"], int)
        assert isinstance(result["project_bullet_count"], int)


class TestAceApplyDeltaSchema:
    def test_returns_dict_with_keys(self):
        result = ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Test strategy"},
        ])
        assert isinstance(result, dict)
        assert "applied" in result
        assert "scope" in result
        assert "message" in result
        assert result["applied"] == 1


class TestAceUpdateCountersSchema:
    def test_returns_dict_with_keys(self):
        ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Count me"},
        ])
        import re
        pb = ace_get_playbook()["message"]
        ids = re.findall(r"\[(str-\d+)\]", pb)
        result = ace_update_counters([{"id": ids[0], "tag": "helpful"}])
        assert isinstance(result, dict)
        assert "updated" in result
        assert "scope" in result
        assert "message" in result


class TestAceGetPlaybookWithStatsSchema:
    def test_returns_dict_with_keys(self):
        result = ace_get_playbook(include_stats=True)
        assert isinstance(result, dict)
        assert "global_bullet_count" in result
        assert "project_bullet_count" in result
        assert "message" in result
        # message contains playbook text with appended stats
        assert isinstance(result["message"], str)


class TestAceFindSimilarSchema:
    def test_returns_dict_with_keys(self):
        result = ace_find_similar()
        assert isinstance(result, dict)
        assert "pairs_found" in result
        assert "scope" in result
        assert "message" in result


class TestAcePruneSchema:
    def test_returns_dict_with_keys(self):
        result = ace_prune()
        assert isinstance(result, dict)
        assert "removed" in result
        assert "evolved" in result
        assert "scope" in result
        assert "message" in result


class TestAceGenerateBulletsSchema:
    def test_returns_dict_with_keys(self):
        from unittest.mock import MagicMock, patch

        mock_sub = MagicMock()
        mock_sub.completion.side_effect = [
            '["Test bullet one", "Test bullet two"]',
            json.dumps([
                {"bullet": "Test bullet one", "score": 5},
                {"bullet": "Test bullet two", "score": 4},
            ]),
            json.dumps([
                {"bullet": "Test bullet one", "action": "ADD", "merge_with": None},
                {"bullet": "Test bullet two", "action": "ADD", "merge_with": None},
            ]),
        ]

        with patch.object(mcp_mod, "_get_sub_client", return_value=mock_sub):
            result = ace_generate_bullets(context="testing", auto_apply=False)

        assert isinstance(result, dict)
        assert "decisions" in result
        assert "applied" in result
        assert "message" in result


class TestAceEvolveFromFailuresSchema:
    def test_returns_dict_with_keys(self):
        result = ace_evolve_from_failures()
        assert isinstance(result, dict)
        assert "evolved" in result
        assert "synthesized" in result
        assert "message" in result


class TestAceEvolveSchemaSchema:
    def test_returns_dict_with_keys(self):
        result = ace_evolve_schema()
        assert isinstance(result, dict)
        assert "version" in result
        assert "message" in result


# ===========================================================================
# RLM tools return dicts with expected keys
# ===========================================================================


class TestRlmInitSchema:
    def test_returns_dict_with_keys(self):
        result = rlm_init("Test problem")
        assert isinstance(result, dict)
        assert "session_id" in result
        assert "file_count" in result
        assert "message" in result
        assert isinstance(result["file_count"], int)


class TestRlmExecuteSchema:
    def test_returns_dict_with_keys(self):
        rlm_init("Test")
        result = rlm_execute("x = 1 + 1")
        assert isinstance(result, dict)
        assert "has_error" in result
        assert "has_final_answer" in result
        assert "message" in result
        assert result["has_error"] is False

    def test_not_initialized_raises_tool_error(self):
        mcp_mod._repl = None
        with pytest.raises(ToolError, match="not initialized"):
            rlm_execute("x = 1")


class TestRlmFinalizeSchema:
    def test_returns_dict_with_keys(self):
        rlm_init("Test")
        rlm_execute("answer = 42")
        result = rlm_finalize("answer")
        assert isinstance(result, dict)
        assert "variable_name" in result
        assert "message" in result
        assert result["variable_name"] == "answer"

    def test_not_initialized_raises_tool_error(self):
        mcp_mod._repl = None
        with pytest.raises(ToolError, match="not initialized"):
            rlm_finalize("x")

    def test_missing_var_raises_tool_error(self):
        rlm_init("Test")
        with pytest.raises(ToolError, match="not found"):
            rlm_finalize("nonexistent")


# ===========================================================================
# Index tools return dicts with expected keys
# ===========================================================================


class TestIndexBuildSchema:
    def test_returns_dict_with_keys(self):
        result = index_build()
        assert isinstance(result, dict)
        assert "files_indexed" in result
        assert "message" in result
        assert isinstance(result["files_indexed"], int)
        assert result["files_indexed"] > 0


class TestIndexSearchSchema:
    def test_returns_dict_with_keys(self):
        result = index_search("greet")
        assert isinstance(result, dict)
        assert "result_count" in result
        assert "mode" in result
        assert "message" in result

    def test_invalid_mode_raises_tool_error(self):
        with pytest.raises(ToolError, match="Invalid mode"):
            index_search("greet", mode="invalid")


# ===========================================================================
# Cross-cutting: no tool returns a plain string
# ===========================================================================


class TestNoToolReturnsString:
    """Verify that calling each tool returns a dict, never a bare string."""

    def test_gcc_commit_not_str(self):
        r = gcc_commit("T", "W", "R", ["f.py"], "N")
        assert not isinstance(r, str)

    def test_gcc_branch_not_str(self):
        r = gcc_branch("b", "purpose", "ctx")
        assert not isinstance(r, str)

    def test_gcc_context_not_str(self):
        r = gcc_context(level=1)
        assert not isinstance(r, str)

    def test_gcc_status_not_str(self):
        r = gcc_status()
        assert not isinstance(r, str)

    def test_gcc_log_ota_not_str(self):
        r = gcc_log_ota("obs", "thought", "action")
        assert not isinstance(r, str)

    def test_gcc_patterns_not_str(self):
        r = gcc_patterns()
        assert not isinstance(r, str)

    def test_gcc_links_not_str(self):
        r = gcc_links("C999")
        assert not isinstance(r, str)

    def test_gcc_context_with_summaries_not_str(self):
        r = gcc_context(include_summaries=True, summaries_tier="all", summaries_count=5)
        assert not isinstance(r, str)

    def test_ace_get_playbook_not_str(self):
        r = ace_get_playbook()
        assert not isinstance(r, str)

    def test_ace_apply_delta_not_str(self):
        r = ace_apply_delta([
            {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "X"},
        ])
        assert not isinstance(r, str)

    def test_ace_get_playbook_with_stats_not_str(self):
        r = ace_get_playbook(include_stats=True)
        assert not isinstance(r, str)

    def test_ace_find_similar_not_str(self):
        r = ace_find_similar()
        assert not isinstance(r, str)

    def test_ace_prune_not_str(self):
        r = ace_prune()
        assert not isinstance(r, str)

    def test_ace_evolve_from_failures_not_str(self):
        r = ace_evolve_from_failures()
        assert not isinstance(r, str)

    def test_ace_evolve_schema_not_str(self):
        r = ace_evolve_schema()
        assert not isinstance(r, str)

    def test_rlm_init_not_str(self):
        r = rlm_init("Test")
        assert not isinstance(r, str)

    def test_rlm_execute_not_str(self):
        rlm_init("Test")
        r = rlm_execute("x = 1")
        assert not isinstance(r, str)

    def test_rlm_finalize_not_str(self):
        rlm_init("Test")
        rlm_execute("ans = 'ok'")
        r = rlm_finalize("ans")
        assert not isinstance(r, str)

    def test_index_build_not_str(self):
        r = index_build()
        assert not isinstance(r, str)

    def test_index_search_not_str(self):
        r = index_search("greet")
        assert not isinstance(r, str)

    def test_gcc_clusters_not_str(self):
        r = gcc_clusters()
        assert not isinstance(r, str)

    def test_gcc_triples_not_str(self):
        r = gcc_triples()
        assert not isinstance(r, str)

    def test_gcc_scratchpad_set_not_str(self):
        r = gcc_scratchpad(mode="set", key="k", value="v")
        assert not isinstance(r, str)

    def test_gcc_scratchpad_get_not_str(self):
        r = gcc_scratchpad(mode="get")
        assert not isinstance(r, str)

    def test_gcc_scratchpad_clear_not_str(self):
        r = gcc_scratchpad(mode="clear")
        assert not isinstance(r, str)
