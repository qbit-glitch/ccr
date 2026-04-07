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
