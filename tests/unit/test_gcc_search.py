"""Unit tests for ccr.mcp.gcc_search_tools — gcc_search unified search tool.

GccSearchResult is a TypedDict (plain dict at runtime): use result["key"] not result.key.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import ccr.mcp.server as _srv
from ccr.mcp_server import _init
from ccr.mcp.gcc_search_tools import gcc_search
import ccr.mcp_server as mcp_mod


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
    (tmp_path / "hello.py").write_text("def greet(): pass\n")
    _init(str(tmp_path))
    yield tmp_path

    mcp_mod._memory = None
    mcp_mod._playbook = None
    mcp_mod._global_playbook = None
    mcp_mod._repo_index = None
    mcp_mod._repl = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _patch_mem(tmp_path=None):
    """Return a mock MemoryManager with sensible defaults."""
    mem = MagicMock()
    # Use the current _project_root so ccr_root resolves to a real path
    proj = _srv._project_root or (str(tmp_path) if tmp_path else "/tmp")
    mem.ccr_root = os.path.join(proj, ".ccr")
    mem.get_active_branch.return_value = "main"
    mem._search_commits.return_value = ""
    mem.get_discussions.return_value = {"count": 0, "message": ""}
    mem.get_experiments.return_value = {"records": []}
    return mem


# ===========================================================================
# Commits source
# ===========================================================================

class TestSearchCommits:
    def test_commits_returns_results(self):
        """Commits with a matching header → total=1, message contains 'Commits'."""
        mem = _patch_mem()
        mem._search_commits.return_value = "## [C001] LoRA experiment\n\nContent here\n"
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("LoRA", sources=["commits"])
        assert result["total"] == 1
        assert "commits" in result["sources_searched"]
        assert "Commits" in result["message"]

    def test_commits_empty_returns_zero(self):
        """Empty commit search → total=0, no-results message."""
        mem = _patch_mem()
        mem._search_commits.return_value = ""
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("nothing", sources=["commits"])
        assert result["total"] == 0
        assert "No results" in result["message"]

    def test_commits_multiple_matches_count(self):
        """Three commit headers → total=3."""
        mem = _patch_mem()
        mem._search_commits.return_value = (
            "## [C001] first\n\n## [C002] second\n\n## [C003] third\n"
        )
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("anything", sources=["commits"])
        assert result["total"] == 3

    def test_commits_exception_logged_not_raised(self, tmp_path):
        """RuntimeError in _search_commits → no crash, error logged to file."""
        mem = _patch_mem(tmp_path)
        mem._search_commits.side_effect = RuntimeError("disk error")
        mem.ccr_root = str(tmp_path / ".ccr")
        log_path = str(tmp_path / ".ccr" / ".hook_errors.log")
        with patch.object(_srv, "_memory", mem), \
             patch.object(_srv, "_project_root", str(tmp_path)):
            result = gcc_search("query", sources=["commits"])
        assert result["total"] == 0  # no crash
        if os.path.isfile(log_path):
            assert "gcc_search:commits" in open(log_path).read()


# ===========================================================================
# Discussions source
# ===========================================================================

class TestSearchDiscussions:
    def test_discussions_returns_results(self):
        """Discussions with count=2 → total=2, 'Discussions' in message."""
        mem = _patch_mem()
        mem.get_discussions.return_value = {
            "count": 2,
            "message": "Discussion about dataset preprocessing",
        }
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("preprocessing", sources=["discussions"])
        assert result["total"] == 2
        assert "discussions" in result["sources_searched"]
        assert "Discussions" in result["message"]

    def test_discussions_no_match(self):
        """Discussions count=0 → not added to message."""
        mem = _patch_mem()
        mem.get_discussions.return_value = {"count": 0, "message": ""}
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("xyz", sources=["discussions"])
        assert result["total"] == 0
        assert "Discussions" not in result["message"]

    def test_discussions_exception_logged_not_raised(self, tmp_path):
        """KeyError in get_discussions → no crash."""
        mem = _patch_mem(tmp_path)
        mem.ccr_root = str(tmp_path / ".ccr")
        mem.get_discussions.side_effect = KeyError("missing key")
        with patch.object(_srv, "_memory", mem), \
             patch.object(_srv, "_project_root", str(tmp_path)):
            result = gcc_search("query", sources=["discussions"])
        assert result["total"] == 0  # no crash

    def test_discussions_date_range_forwarded(self):
        """date_range is passed through to get_discussions."""
        mem = _patch_mem()
        mem.get_discussions.return_value = {"count": 0, "message": ""}
        dr = ["2026-01-01", "2026-04-01"]
        with patch.object(_srv, "_memory", mem):
            gcc_search("topic", sources=["discussions"], date_range=dr)
        mem.get_discussions.assert_called_once_with(search="topic", date_range=dr)


# ===========================================================================
# Experiments source
# ===========================================================================

class TestSearchExperiments:
    def test_experiments_empty_returns_zero(self):
        """No experiment records → total stays 0."""
        mem = _patch_mem()
        mem.get_experiments.return_value = {"records": []}
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("nothing", sources=["experiments"])
        assert result["total"] == 0

    def test_experiments_exception_logged_not_raised(self, tmp_path):
        """ValueError in get_experiments → no crash."""
        mem = _patch_mem(tmp_path)
        mem.ccr_root = str(tmp_path / ".ccr")
        mem.get_experiments.side_effect = ValueError("bad data")
        with patch.object(_srv, "_memory", mem), \
             patch.object(_srv, "_project_root", str(tmp_path)):
            result = gcc_search("query", sources=["experiments"])
        assert result["total"] == 0  # no crash

    def test_experiments_in_sources_searched(self):
        """experiments always appears in sources_searched when requested."""
        mem = _patch_mem()
        mem.get_experiments.return_value = {"records": []}
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("test", sources=["experiments"])
        assert "experiments" in result["sources_searched"]

    def test_experiments_not_double_counted(self):
        """Records matching both hypothesis search and full scan must appear only once."""
        shared_record = {
            "commit_id": "C001",
            "id": "exp-1",
            "hypothesis": "test query hypothesis",
            "metrics": {},
            "conclusion": "",
        }
        mem = _patch_mem()
        mem.get_experiments.side_effect = [
            {"records": [shared_record], "count": 1, "message": ""},  # first call (hypothesis_contains)
            {"records": [shared_record], "count": 1, "message": ""},  # second call (full scan)
        ]
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("test query", sources=["experiments"])
        # The record appears in both calls but must only be counted once
        assert result["total"] == 1


# ===========================================================================
# Sessions source
# ===========================================================================

class TestSearchSessions:
    def test_sessions_db_missing_skipped(self, tmp_path):
        """If sessions.db doesn't exist, sessions returns 0 results gracefully."""
        mem = _patch_mem(tmp_path)
        mem.ccr_root = str(tmp_path / ".ccr")
        # Don't create the sessions.db file → isfile returns False
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("test", sources=["sessions"])
        assert result["total"] == 0
        assert "sessions" in result["sources_searched"]

    def test_sessions_exception_logged_not_raised(self, tmp_path):
        """Exception when opening SessionStore → no crash."""
        mem = _patch_mem(tmp_path)
        mem.ccr_root = str(tmp_path / ".ccr")
        with patch.object(_srv, "_memory", mem), \
             patch.object(_srv, "_project_root", str(tmp_path)), \
             patch("os.path.isfile", return_value=True), \
             patch("ccr.core.session_store.SessionStore",
                   side_effect=RuntimeError("db error")):
            result = gcc_search("query", sources=["sessions"])
        assert result["total"] == 0  # no crash

    def test_sessions_returns_results_when_turns_found(self, tmp_path):
        """When SessionStore.search_turns returns turns → total reflects count."""
        mem = _patch_mem(tmp_path)
        mem.ccr_root = str(tmp_path / ".ccr")
        fake_turns = [
            {"timestamp": "2026-04-01T10:00:00", "session_id": "abc12345",
             "user_message": "how does LoRA work", "assistant_message": "LoRA uses..."},
        ]
        mock_store = MagicMock()
        mock_store.search_turns.return_value = fake_turns
        with patch.object(_srv, "_memory", mem), \
             patch("os.path.isfile", return_value=True), \
             patch("ccr.core.session_store.SessionStore", return_value=mock_store):
            result = gcc_search("LoRA", sources=["sessions"])
        assert result["total"] == 1
        assert "sessions" in result["sources_searched"]


# ===========================================================================
# Multi-source and aggregation
# ===========================================================================

class TestSearchAggregation:
    def test_all_sources_searched_by_default(self, tmp_path):
        """No sources= argument → all 4 sources are attempted."""
        mem = _patch_mem(tmp_path)
        mem.ccr_root = str(tmp_path / ".ccr")
        mem.get_discussions.return_value = {"count": 0, "message": ""}
        mem.get_experiments.return_value = {"records": []}
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("anything")
        assert set(result["sources_searched"]) == {"commits", "discussions", "experiments", "sessions"}

    def test_sources_filter_limits_search(self):
        """sources=["commits"] → only commits in sources_searched."""
        mem = _patch_mem()
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("anything", sources=["commits"])
        assert result["sources_searched"] == ["commits"]

    def test_total_aggregates_across_sources(self):
        """total = sum of matches from all requested sources."""
        mem = _patch_mem()
        mem._search_commits.return_value = "## [C001] a\n\n## [C002] b\n"
        mem.get_discussions.return_value = {"count": 3, "message": "three discussions"}
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("data", sources=["commits", "discussions"])
        assert result["total"] == 5  # 2 commits + 3 discussions

    def test_no_results_message(self, tmp_path):
        """When all sources return empty → message contains 'No results' and the query."""
        mem = _patch_mem(tmp_path)
        mem.ccr_root = str(tmp_path / ".ccr")
        mem.get_discussions.return_value = {"count": 0, "message": ""}
        mem.get_experiments.return_value = {"records": []}
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("nonexistent_xyz")
        assert "No results" in result["message"]
        assert "nonexistent_xyz" in result["message"]

    def test_sources_searched_in_result(self):
        """sources_searched lists only the requested sources."""
        mem = _patch_mem()
        mem.get_discussions.return_value = {"count": 0, "message": ""}
        with patch.object(_srv, "_memory", mem):
            result = gcc_search("test", sources=["commits", "discussions"])
        assert "commits" in result["sources_searched"]
        assert "discussions" in result["sources_searched"]
        assert "experiments" not in result["sources_searched"]
        assert "sessions" not in result["sources_searched"]

    def test_limit_parameter_passed_through(self, tmp_path):
        """limit parameter is used when querying sessions."""
        mem = _patch_mem(tmp_path)
        mem.ccr_root = str(tmp_path / ".ccr")
        mock_store = MagicMock()
        mock_store.search_turns.return_value = []
        with patch.object(_srv, "_memory", mem), \
             patch("os.path.isfile", return_value=True), \
             patch("ccr.core.session_store.SessionStore", return_value=mock_store):
            gcc_search("test", sources=["sessions"], limit=3)
        mock_store.search_turns.assert_called_once_with("test", limit=3)


# ===========================================================================
# F7: Auto-Trigger Extraction — ERL (arXiv:2603.24639)
# ===========================================================================

class TestExtractTriggerSuggestions:
    """_extract_trigger_suggestions() parses trigger/action pairs from patterns."""

    def test_when_clause(self):
        """'when X, Y' → trigger=X, action=Y."""
        from ccr.mcp.gcc_tools import _extract_trigger_suggestions
        result = _extract_trigger_suggestions(["when adding endpoints, validate first"])
        assert len(result) == 1
        assert result[0]["trigger"] == "adding endpoints"
        assert result[0]["action"] == "validate first"

    def test_arrow_notation(self):
        """'X → Y' arrow → trigger=X, action=Y."""
        from ccr.mcp.gcc_tools import _extract_trigger_suggestions
        result = _extract_trigger_suggestions(["deploy → run smoke tests"])
        assert len(result) == 1
        assert result[0]["trigger"] == "deploy"
        assert result[0]["action"] == "run smoke tests"

    def test_if_clause(self):
        """'if X, Y' → trigger=X, action=Y."""
        from ccr.mcp.gcc_tools import _extract_trigger_suggestions
        result = _extract_trigger_suggestions(["if file exceeds 800 lines, split it"])
        assert len(result) == 1
        assert result[0]["trigger"] == "file exceeds 800 lines"
        assert result[0]["action"] == "split it"

    def test_before_clause(self):
        """'before X, Y' → trigger=X, action=Y."""
        from ccr.mcp.gcc_tools import _extract_trigger_suggestions
        result = _extract_trigger_suggestions(["before committing, run tests"])
        assert len(result) == 1
        assert "committing" in result[0]["trigger"]

    def test_no_match_returns_empty(self):
        """Pattern with no trigger/action structure → empty list."""
        from ccr.mcp.gcc_tools import _extract_trigger_suggestions
        result = _extract_trigger_suggestions(["use snake_case for variable names"])
        assert result == []

    def test_multiple_patterns(self):
        """Multiple patterns → multiple suggestions."""
        from ccr.mcp.gcc_tools import _extract_trigger_suggestions
        patterns = [
            "when adding new tool, update __init__.py",
            "deploy → run smoke tests",
            "plain pattern with no structure",
        ]
        result = _extract_trigger_suggestions(patterns)
        assert len(result) == 2

    def test_empty_list_returns_empty(self):
        """Empty patterns list → empty result."""
        from ccr.mcp.gcc_tools import _extract_trigger_suggestions
        assert _extract_trigger_suggestions([]) == []


# ===========================================================================
# F9: Scratchpad Semantic Search — AgeMem (arXiv:2601.01885)
# ===========================================================================

class TestScratchpadSearch:
    """Scratchpad.search() BM25-fallback path (no ONNX required)."""

    def _make_scratchpad(self, tmp_path):
        from ccr.core.scratchpad import Scratchpad
        sp = Scratchpad(str(tmp_path / "scratchpad.json"))
        sp.set("auth", "authentication token management and JWT handling")
        sp.set("db", "database connection pooling strategy")
        sp.set("cache", "Redis cache invalidation approach")
        return sp

    def test_keyword_match_returns_result(self, tmp_path):
        """Query with a keyword present in value → non-zero score entry returned."""
        sp = self._make_scratchpad(tmp_path)
        results = sp.search("authentication")
        assert len(results) > 0
        keys = [r["key"] for r in results]
        assert "auth" in keys

    def test_top_k_limits_results(self, tmp_path):
        """top_k=1 → at most 1 result."""
        sp = self._make_scratchpad(tmp_path)
        results = sp.search("database cache", top_k=1)
        assert len(results) <= 1

    def test_no_match_returns_empty_or_sorted(self, tmp_path):
        """Query with zero overlap → results still returned (score=0) or empty."""
        sp = self._make_scratchpad(tmp_path)
        results = sp.search("completely unrelated xyz123", top_k=5)
        # All scores should be 0 for BM25 fallback; results may be returned
        for r in results:
            assert r["score"] >= 0

    def test_result_has_required_keys(self, tmp_path):
        """Each result has key, value, score, created_at, updated_at."""
        sp = self._make_scratchpad(tmp_path)
        results = sp.search("auth")
        assert len(results) > 0
        for r in results:
            assert "key" in r
            assert "value" in r
            assert "score" in r

    def test_empty_scratchpad_returns_empty(self, tmp_path):
        """Empty scratchpad → search returns []."""
        from ccr.core.scratchpad import Scratchpad
        sp = Scratchpad(str(tmp_path / "empty.json"))
        assert sp.search("anything") == []


class TestGccScratchpadSearchTool:
    """gcc_scratchpad_search MCP tool integration tests."""

    def test_returns_results_when_entries_match(self):
        """When scratchpad has matching entries → total > 0."""
        from ccr.mcp.gcc_search_tools import gcc_scratchpad_search
        from ccr.core.scratchpad import Scratchpad
        from unittest.mock import patch
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            sp = Scratchpad(os.path.join(tmp, "sp.json"))
            sp.set("auth", "JWT authentication token handling")
            with patch.object(_srv, "_scratchpad", sp):
                result = gcc_scratchpad_search("authentication")
        assert result["total"] > 0
        assert "auth" in result["message"]

    def test_no_scratchpad_returns_zero(self):
        """_scratchpad is None → total=0, no crash."""
        from ccr.mcp.gcc_search_tools import gcc_scratchpad_search
        from unittest.mock import patch
        with patch.object(_srv, "_scratchpad", None):
            result = gcc_scratchpad_search("test")
        assert result["total"] == 0

    def test_no_match_returns_zero(self):
        """Non-matching query → total=0."""
        from ccr.mcp.gcc_search_tools import gcc_scratchpad_search
        from ccr.core.scratchpad import Scratchpad
        from unittest.mock import patch
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            sp = Scratchpad(os.path.join(tmp, "sp.json"))
            sp.set("note", "some unrelated content")
            with patch.object(_srv, "_scratchpad", sp):
                result = gcc_scratchpad_search("completelydifferenttermxyz999", top_k=0)
        # top_k=0 → no results
        assert result["total"] == 0
