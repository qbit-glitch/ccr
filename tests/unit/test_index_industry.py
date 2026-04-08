"""Industry-standard index improvements: SHA-256 hash cache, progress logging,
score normalization, and mode_hint for index_search.

Tests: D1 (hash cache + progress), D2 (score normalization + mode_hint), D3 (progress_interval wiring).
"""

from __future__ import annotations

import logging
import os

import pytest

from ccr.context.indexer import RepoIndex
from ccr.mcp_server import (
    _init,
    index_build,
    index_search,
)
import ccr.mcp_server as mcp_mod


# ---------------------------------------------------------------------------
# Shared fixture (mirrors test_mcp_server.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_project(tmp_path, monkeypatch):
    """Initialize CCR in a temp directory for each test."""
    fake_home_ccr = tmp_path / "global_ccr"
    fake_home_ccr.mkdir()
    original_expanduser = os.path.expanduser

    def mock_expanduser(path: str) -> str:
        if path.startswith("~/.ccr"):
            return str(fake_home_ccr) + path[5:]
        return original_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

    # Create sample files for indexing
    (tmp_path / "hello.py").write_text(
        "def greet(name):\n    return f'Hello, {name}!'\n\nclass Greeter:\n    pass\n"
    )
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
# D1: SHA-256 hash cache
# ===========================================================================


class TestIndexBuildSha256Cache:
    """RepoIndex.build() stores SHA-256 hashes (16-char hex) in _file_hashes."""

    def test_file_hashes_populated(self, tmp_path):
        """_file_hashes is populated after build."""
        idx = RepoIndex.build(str(tmp_path))
        assert len(idx._file_hashes) > 0, "_file_hashes must be non-empty after build"

    def test_file_hashes_are_16_char_hex(self, tmp_path):
        """Each hash is a 16-character lowercase hex string."""
        idx = RepoIndex.build(str(tmp_path))
        for path, sha in idx._file_hashes.items():
            assert len(sha) == 16, f"Hash for {path!r} must be 16 chars, got {len(sha)}"
            assert all(c in "0123456789abcdef" for c in sha), (
                f"Hash for {path!r} must be hex, got {sha!r}"
            )

    def test_file_hashes_keys_match_indexed_files(self, tmp_path):
        """Keys in _file_hashes correspond to indexed relative paths."""
        idx = RepoIndex.build(str(tmp_path))
        for path in idx._file_hashes:
            assert path in idx.files, f"Hash key {path!r} not in index.files"

    def test_file_hashes_serialized_in_to_json(self, tmp_path):
        """to_json() includes file_hashes."""
        import json

        idx = RepoIndex.build(str(tmp_path))
        data = json.loads(idx.to_json())
        assert "file_hashes" in data, "to_json() must include 'file_hashes' key"
        assert isinstance(data["file_hashes"], dict)

    def test_file_hashes_loaded_from_cache(self, tmp_path):
        """from_cache() restores _file_hashes from JSON."""
        idx = RepoIndex.build(str(tmp_path))
        cached_json = idx.to_json()
        idx2 = RepoIndex.from_cache(str(tmp_path), cached_json)
        assert idx2 is not None
        assert idx2._file_hashes == idx._file_hashes

    def test_from_cache_backward_compat_missing_hashes(self, tmp_path):
        """from_cache() returns empty dict when file_hashes key is absent (old cache)."""
        import json

        idx = RepoIndex.build(str(tmp_path))
        data = json.loads(idx.to_json())
        del data["file_hashes"]  # Simulate old cache format
        old_cache = json.dumps(data)
        idx2 = RepoIndex.from_cache(str(tmp_path), old_cache)
        assert idx2 is not None
        assert idx2._file_hashes == {}


# ===========================================================================
# D1: Progress logging
# ===========================================================================


class TestIndexBuildProgress:
    """RepoIndex.build() emits progress log messages at configured intervals."""

    def test_progress_logged_at_interval(self, tmp_path, caplog):
        """With progress_interval=1, every file triggers a log message."""
        with caplog.at_level(logging.INFO, logger="ccr.context.indexer"):
            idx = RepoIndex.build(str(tmp_path), progress_interval=1)

        progress_msgs = [r for r in caplog.records if "Index progress:" in r.message]
        # We have at least 2 Python files, so expect at least 1 progress log
        assert len(progress_msgs) >= 1, (
            f"Expected at least 1 progress log with interval=1; got {len(progress_msgs)}"
        )

    def test_progress_message_format(self, tmp_path, caplog):
        """Progress messages include the file count."""
        with caplog.at_level(logging.INFO, logger="ccr.context.indexer"):
            RepoIndex.build(str(tmp_path), progress_interval=1)

        progress_msgs = [r for r in caplog.records if "Index progress:" in r.message]
        for msg in progress_msgs:
            assert "files indexed" in msg.message, (
                f"Progress message format unexpected: {msg.message!r}"
            )

    def test_no_progress_at_large_interval(self, tmp_path, caplog):
        """With a very large interval, no progress messages are emitted for small repos."""
        with caplog.at_level(logging.INFO, logger="ccr.context.indexer"):
            RepoIndex.build(str(tmp_path), progress_interval=10000)

        progress_msgs = [r for r in caplog.records if "Index progress:" in r.message]
        assert len(progress_msgs) == 0, (
            "No progress messages expected for large interval on a tiny repo"
        )

    def test_progress_interval_default_no_crash(self, tmp_path):
        """Default progress_interval=100 builds without error."""
        idx = RepoIndex.build(str(tmp_path))
        assert len(idx.files) > 0


# ===========================================================================
# D2: Score normalization
# ===========================================================================


class TestIndexSearchNormalized:
    """index_search normalizes scores so the top result is always 1.0."""

    def test_top_result_score_is_one(self, tmp_path):
        """After normalization the highest-scored result has score == 1.0."""
        index_build()
        result = index_search("greet", mode="keyword")
        msg = result["message"]
        # Parse scores from output lines like "[0.5] path ..."
        score_lines = [l for l in msg.split("\n") if l.startswith("[")]
        assert score_lines, f"Expected scored lines in result: {msg!r}"
        scores = []
        for line in score_lines:
            score_str = line.split("]")[0].lstrip("[")
            scores.append(float(score_str))
        assert max(scores) == pytest.approx(1.0, abs=1e-4), (
            f"Top score should be 1.0 after normalization, got {max(scores)}"
        )

    def test_scores_in_zero_to_one_range(self, tmp_path):
        """All scores are in [0.0, 1.0] after normalization."""
        index_build()
        result = index_search("helper", mode="keyword")
        msg = result["message"]
        score_lines = [l for l in msg.split("\n") if l.startswith("[")]
        for line in score_lines:
            score_str = line.split("]")[0].lstrip("[")
            score = float(score_str)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range"

    def test_normalization_with_multiple_results(self, tmp_path):
        """When multiple results exist, scores are proportionally normalized."""
        index_build()
        # Search for 'py' which matches both files via path
        result = index_search("py", mode="keyword", top_k=10)
        msg = result["message"]
        score_lines = [l for l in msg.split("\n") if l.startswith("[")]
        if len(score_lines) >= 2:
            scores = []
            for line in score_lines:
                score_str = line.split("]")[0].lstrip("[")
                scores.append(float(score_str))
            # Top score must be 1.0
            assert max(scores) == pytest.approx(1.0, abs=1e-4)
            # All others <= 1.0
            assert all(s <= 1.0 for s in scores)


# ===========================================================================
# D2: mode_hint
# ===========================================================================


class TestIndexSearchModeHintLong:
    """Question-form queries get 'semantic' suggestion via _detect_mode."""

    def test_long_query_suggests_semantic(self, tmp_path):
        # _detect_mode: question-word prefix → semantic
        index_build()
        long_query = "how does the authentication flow work in this system"
        result = index_search(long_query, mode_hint=True)
        msg = result["message"]
        assert "mode_hint" in msg, f"Expected mode_hint in result: {msg!r}"
        assert "semantic" in msg, f"Expected 'semantic' suggestion: {msg!r}"

    def test_long_query_exactly_51_chars(self, tmp_path):
        # _detect_mode: ends with '?' → semantic
        index_build()
        query = "what is the purpose of the main entry point here?"
        result = index_search(query, mode_hint=True)
        assert "semantic" in result["message"]

    def test_long_query_no_hint_when_disabled(self, tmp_path):
        index_build()
        long_query = "b" * 51
        result = index_search(long_query, mode_hint=False)
        assert "mode_hint" not in result["message"]


class TestIndexSearchModeHintPath:
    """Queries with '/' or file extensions get 'keyword' suggestion."""

    def test_path_with_slash_suggests_keyword(self, tmp_path):
        index_build()
        result = index_search("src/utils.py", mode_hint=True)
        msg = result["message"]
        assert "mode_hint" in msg
        assert "keyword" in msg

    def test_py_extension_suggests_keyword(self, tmp_path):
        index_build()
        result = index_search("utils.py", mode_hint=True)
        assert "keyword" in result["message"]

    def test_ts_extension_suggests_keyword(self, tmp_path):
        index_build()
        result = index_search("app.ts", mode_hint=True)
        assert "keyword" in result["message"]

    def test_js_extension_suggests_keyword(self, tmp_path):
        index_build()
        result = index_search("bundle.js", mode_hint=True)
        assert "keyword" in result["message"]

    def test_go_extension_suggests_keyword(self, tmp_path):
        index_build()
        result = index_search("server.go", mode_hint=True)
        assert "keyword" in result["message"]

    def test_rs_extension_suggests_keyword(self, tmp_path):
        index_build()
        result = index_search("main.rs", mode_hint=True)
        assert "keyword" in result["message"]

    def test_yaml_extension_suggests_keyword(self, tmp_path):
        index_build()
        result = index_search("config.yaml", mode_hint=True)
        assert "keyword" in result["message"]

    def test_json_extension_suggests_keyword(self, tmp_path):
        index_build()
        result = index_search("schema.json", mode_hint=True)
        assert "keyword" in result["message"]

    def test_md_extension_suggests_keyword(self, tmp_path):
        index_build()
        result = index_search("README.md", mode_hint=True)
        assert "keyword" in result["message"]


class TestIndexSearchModeHintElse:
    """Short plain-word queries get 'hybrid' suggestion."""

    def test_plain_word_suggests_hybrid(self, tmp_path):
        index_build()
        result = index_search("authentication", mode_hint=True)
        msg = result["message"]
        assert "mode_hint" in msg
        assert "hybrid" in msg

    def test_short_query_suggests_hybrid(self, tmp_path):
        index_build()
        result = index_search("greet", mode_hint=True)
        assert "hybrid" in result["message"]

    def test_hint_absent_by_default(self, tmp_path):
        """mode_hint defaults to False — no hint appended."""
        index_build()
        result = index_search("greet")
        assert "mode_hint" not in result["message"]


# ===========================================================================
# D3: progress_interval wired into index_build MCP tool
# ===========================================================================


class TestIndexBuildProgressInterval:
    """index_build MCP tool passes progress_interval to RepoIndex.build()."""

    def test_index_build_accepts_progress_interval(self, tmp_path):
        """index_build(progress_interval=N) runs without error."""
        result = index_build(progress_interval=1)
        assert "Files:" in result["message"]

    def test_index_build_default_progress_interval(self, tmp_path):
        """index_build() with default interval runs without error."""
        result = index_build()
        assert "Files:" in result["message"]

    def test_index_build_progress_interval_one_logs(self, tmp_path, caplog):
        """index_build(progress_interval=1) triggers progress logging for files."""
        with caplog.at_level(logging.INFO, logger="ccr.context.indexer"):
            index_build(progress_interval=1)

        progress_msgs = [r for r in caplog.records if "Index progress:" in r.message]
        # At least the two sample files should trigger messages with interval=1
        assert len(progress_msgs) >= 1


# ===========================================================================
# F1: mode="auto" + resolved_mode (A-RAG §3.2)
# ===========================================================================


class TestAutoMode:
    """index_search with mode='auto' resolves to keyword/semantic/hybrid."""

    def test_symbol_query_resolves_keyword(self, tmp_path):
        """Short symbol-like query → auto resolves to keyword."""
        index_build()
        result = index_search("greet", mode="auto")
        assert result["resolved_mode"] == "keyword"
        assert "auto→keyword" in result["message"]

    def test_question_query_resolves_semantic(self, tmp_path):
        """Question-form query → auto resolves to semantic."""
        index_build()
        result = index_search("how does greet function work", mode="auto")
        assert result["resolved_mode"] == "semantic"
        assert "auto→semantic" in result["message"]

    def test_general_query_resolves_hybrid(self, tmp_path):
        """Multi-token general query (>3 tokens) → auto resolves to hybrid."""
        index_build()
        result = index_search("helper utility function module", mode="auto")
        assert result["resolved_mode"] == "hybrid"
        assert "auto→hybrid" in result["message"]

    def test_non_auto_mode_resolved_mode_matches(self, tmp_path):
        """Explicit mode → resolved_mode equals mode."""
        index_build()
        result = index_search("greet", mode="keyword")
        assert result["resolved_mode"] == "keyword"
        assert result["mode"] == "keyword"

    def test_invalid_mode_raises(self, tmp_path):
        """Unknown mode string → ToolError."""
        from mcp.server.fastmcp.exceptions import ToolError
        index_build()
        with pytest.raises(ToolError):
            index_search("greet", mode="fuzzy")


# ===========================================================================
# F5: exclude_paths filter (A-RAG C^read)
# ===========================================================================


class TestExcludePaths:
    """index_search exclude_paths removes specified paths from results."""

    def test_excluded_path_absent_from_results(self, tmp_path):
        """Path in exclude_paths does not appear in results."""
        index_build()
        # First search to find what paths exist
        all_results = index_search("helper", mode="keyword")
        paths = [line.split("] ", 1)[-1].split(" (")[0]
                 for line in all_results["message"].splitlines()
                 if line.startswith("[")]
        if not paths:
            pytest.skip("No results to exclude")
        exclude = [paths[0]]
        filtered = index_search("helper", mode="keyword", exclude_paths=exclude)
        filtered_paths = [line.split("] ", 1)[-1].split(" (")[0]
                          for line in filtered["message"].splitlines()
                          if line.startswith("[")]
        assert paths[0] not in filtered_paths

    def test_exclude_empty_list_no_effect(self, tmp_path):
        """Empty exclude_paths → same results as no filter."""
        index_build()
        r1 = index_search("greet", mode="keyword")
        r2 = index_search("greet", mode="keyword", exclude_paths=[])
        assert r1["result_count"] == r2["result_count"]

    def test_exclude_all_paths_returns_zero(self, tmp_path):
        """Excluding all result paths → result_count=0."""
        index_build()
        all_results = index_search("helper", mode="keyword")
        paths = [line.split("] ", 1)[-1].split(" (")[0]
                 for line in all_results["message"].splitlines()
                 if line.startswith("[")]
        if not paths:
            pytest.skip("No results to exclude")
        result = index_search("helper", mode="keyword", exclude_paths=paths)
        assert result["result_count"] == 0
