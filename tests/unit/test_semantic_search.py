"""Tests for semantic commit search (ExpRAG-inspired).

Tests the 3-phase search in MemoryManager._search_commits:
  Phase 1: Exact substring match
  Phase 2: ONNX embedding cosine similarity
  Phase 3: BM25 fallback
"""

import gzip
import json
import os
import re
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig


@pytest.fixture
def ccr_dir(tmp_path):
    """Create a minimal .ccr directory with sample commits."""
    ccr = tmp_path / ".ccr"
    ccr.mkdir()
    branches = ccr / "branches" / "main"
    branches.mkdir(parents=True)

    commits = """## [C001] 2026-03-01 10:00 | branch:main | Add authentication module
**What**: Implemented JWT-based authentication with login and signup endpoints.
**Why**: Users need to securely authenticate.
**Files**: auth.py, routes/login.py
**Next**: Add role-based access control.

---

## [C002] 2026-03-02 11:00 | branch:main | Refactor database layer
**What**: Restructured the database access layer to use repository pattern.
**Why**: Reduce coupling between business logic and data access.
**Files**: db/repository.py, db/models.py
**Next**: Add connection pooling.

---

## [C003] 2026-03-03 12:00 | branch:main | Fix login rate limiting
**What**: Added rate limiting to the login endpoint to prevent brute force attacks.
**Why**: Security audit flagged missing rate limiting.
**Files**: auth.py, middleware/rate_limit.py
**Next**: Add account lockout after N failures.

---

## [C004] 2026-03-04 13:00 | branch:main | Add user profile API
**What**: Created REST endpoints for user profile management (CRUD operations).
**Why**: Frontend team needs profile editing capability.
**Files**: routes/profile.py, models/user.py
**Next**: Add avatar upload.

---

## [C005] 2026-03-05 14:00 | branch:main | Optimize query performance
**What**: Added database indexes and query optimization for the user search feature.
**Why**: Search was slow with >10k users.
**Files**: db/migrations/add_indexes.py, db/repository.py
**Next**: Add caching layer.

---
"""
    (branches / "commits.md").write_text(commits)

    # Create metadata
    metadata = ccr / "metadata.yaml"
    metadata.write_text("project: test\ncreated: 2026-03-01\n")

    # Create registry
    registry = branches / "_registry.md"
    registry.write_text(
        "## Active Branch\nmain\n\n## Branches\n"
        "| Name | Created | Status |\n| main | 2026-03-01 | active |\n"
    )

    return str(ccr)


@pytest.fixture
def mem(ccr_dir, tmp_path):
    """Create a MemoryManager with the test .ccr directory."""
    config = CCRConfig()
    manager = MemoryManager(os.path.dirname(ccr_dir), config)
    return manager


class TestSubstringSearch:
    """Phase 1: Exact substring matching (existing behavior)."""

    def test_exact_match_by_keyword(self, mem):
        result = mem._search_commits("main", "authentication")
        assert "C001" in result
        assert "authentication" in result.lower()

    def test_exact_match_case_insensitive(self, mem):
        result = mem._search_commits("main", "JWT")
        assert "C001" in result

    def test_exact_match_multiple(self, mem):
        result = mem._search_commits("main", "auth.py")
        assert "C001" in result
        assert "C003" in result

    def test_no_match_returns_empty(self, mem):
        result = mem._search_commits("main", "nonexistent_xyz_123")
        assert result == ""

    def test_max_results_respected(self, mem):
        result = mem._search_commits("main", "2026", max_results=2)
        blocks = [b for b in result.split("\n\n") if b.strip() and "[C0" in b]
        assert len(blocks) <= 2

    def test_empty_branch_returns_empty(self, mem):
        result = mem._search_commits("no-such-branch", "test")
        assert result == ""


class TestBM25Search:
    """Phase 3: BM25 fallback when ONNX unavailable."""

    def test_bm25_scores_relevant_higher(self, mem):
        content = mem._read_file(mem._get_commits_path("main"))
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        results = mem._bm25_search_commits(parts, "database repository pattern", 5)
        assert len(results) > 0
        # C002 (database refactor) should score highest
        top_block = results[0][1]
        assert "C002" in top_block

    def test_bm25_empty_query(self, mem):
        content = mem._read_file(mem._get_commits_path("main"))
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        results = mem._bm25_search_commits(parts, "", 5)
        assert results == []

    def test_bm25_short_terms_filtered(self, mem):
        content = mem._read_file(mem._get_commits_path("main"))
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        results = mem._bm25_search_commits(parts, "a b c", 5)
        assert results == []  # all terms <= 2 chars

    def test_bm25_max_results(self, mem):
        content = mem._read_file(mem._get_commits_path("main"))
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        results = mem._bm25_search_commits(parts, "user", 2)
        assert len(results) <= 2

    def test_bm25_returns_scores(self, mem):
        content = mem._read_file(mem._get_commits_path("main"))
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        results = mem._bm25_search_commits(parts, "authentication login", 5)
        for score, block in results:
            assert isinstance(score, float)
            assert score > 0


class TestSemanticSearch:
    """Phase 2: ONNX embedding cosine similarity."""

    def test_search_falls_through_to_bm25_without_onnx(self, mem):
        """When ONNX unavailable and no exact match, BM25 kicks in."""
        with patch("ccr.core.memory_pkg.memory_context.get_embedding_model", return_value=None):
            # "database indexes optimization queries" won't exact-match a single
            # commit but BM25 should find C005 (indexes + optimization + query)
            result = mem._search_commits(
                "main", "database indexes optimization queries", max_results=3
            )
            # BM25 should find C005 (indexes + optimization + query)
            assert "C005" in result

    def test_exact_match_takes_priority(self, mem):
        """Exact matches come before semantic matches."""
        result = mem._search_commits("main", "authentication", max_results=5)
        # First block should be exact match
        first_block = result.split("---")[0] if "---" in result else result
        assert "authentication" in first_block.lower()

    def test_semantic_with_mock_embeddings(self, mem, ccr_dir):
        """Semantic search with mocked ONNX model."""
        import numpy as np

        # Create fake embeddings cache
        embeddings = {
            "C001": np.random.randn(384).astype(np.float32).tolist(),
            "C002": np.random.randn(384).astype(np.float32).tolist(),
            "C003": np.random.randn(384).astype(np.float32).tolist(),
            "C004": np.random.randn(384).astype(np.float32).tolist(),
            "C005": np.random.randn(384).astype(np.float32).tolist(),
        }
        # Make C002 embedding very similar to our query
        query_vec = np.random.randn(384).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        embeddings["C002"] = (
            query_vec + np.random.randn(384).astype(np.float32) * 0.1
        ).tolist()
        # Normalize all
        for k in embeddings:
            v = np.array(embeddings[k], dtype=np.float32)
            embeddings[k] = (v / np.linalg.norm(v)).tolist()

        # Save to cache
        embed_path = os.path.join(ccr_dir, "commit_embeddings.json.gz")
        with gzip.open(embed_path, "wt", encoding="utf-8") as f:
            json.dump(embeddings, f)

        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.embed_query.return_value = query_vec

        with patch("ccr.core.memory_pkg.memory_context.get_embedding_model", return_value=mock_model):
            # Use a term that won't substring-match anything
            result = mem._search_commits(
                "main", "zzz_no_substring_match_zzz", max_results=3
            )
            # Should find C002 via semantic similarity
            assert "C002" in result

    def test_semantic_skips_exact_duplicates(self, mem, ccr_dir):
        """Semantic results don't duplicate commits already found by exact match."""
        import numpy as np

        # Create embeddings where C001 is most similar to query
        query_vec = np.random.randn(384).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        embeddings = {}
        for cid in ["C001", "C002", "C003", "C004", "C005"]:
            v = np.random.randn(384).astype(np.float32)
            embeddings[cid] = (v / np.linalg.norm(v)).tolist()
        # Make C001 very similar to query
        embeddings["C001"] = (
            query_vec + np.random.randn(384).astype(np.float32) * 0.05
        ).tolist()
        v = np.array(embeddings["C001"], dtype=np.float32)
        embeddings["C001"] = (v / np.linalg.norm(v)).tolist()

        embed_path = os.path.join(ccr_dir, "commit_embeddings.json.gz")
        with gzip.open(embed_path, "wt", encoding="utf-8") as f:
            json.dump(embeddings, f)

        mock_model = MagicMock()
        mock_model.embed_query.return_value = query_vec

        with patch("ccr.core.memory_pkg.memory_context.get_embedding_model", return_value=mock_model):
            # "authentication" exact-matches C001; semantic should NOT duplicate it
            result = mem._search_commits("main", "authentication", max_results=5)
            # Count how many times C001 appears
            c001_count = len(re.findall(r"\[C001\]", result))
            assert c001_count == 1

    def test_bm25_only_when_no_exact_and_no_semantic(self, mem):
        """BM25 fires only when both exact and semantic find nothing."""
        with patch("ccr.core.memory_pkg.memory_context.get_embedding_model", return_value=None):
            # "authentication" has exact matches, so BM25 should NOT run
            result = mem._search_commits("main", "authentication", max_results=5)
            assert "C001" in result
            # Confirm it's exact match (authentication appears literally)
            assert "authentication" in result.lower()


class TestLoadAllCommitEmbeddings:
    def test_empty_when_no_cache(self, mem):
        result = mem._load_all_commit_embeddings()
        assert result == {}

    def test_loads_all_embeddings(self, mem, ccr_dir):
        import numpy as np

        embeddings = {
            "C001": np.random.randn(384).astype(np.float32).tolist(),
            "C002": np.random.randn(384).astype(np.float32).tolist(),
        }
        embed_path = os.path.join(ccr_dir, "commit_embeddings.json.gz")
        with gzip.open(embed_path, "wt", encoding="utf-8") as f:
            json.dump(embeddings, f)

        result = mem._load_all_commit_embeddings()
        assert len(result) == 2
        assert "C001" in result
        assert "C002" in result

    def test_returns_numpy_arrays(self, mem, ccr_dir):
        import numpy as np

        embeddings = {
            "C001": np.random.randn(384).astype(np.float32).tolist(),
        }
        embed_path = os.path.join(ccr_dir, "commit_embeddings.json.gz")
        with gzip.open(embed_path, "wt", encoding="utf-8") as f:
            json.dump(embeddings, f)

        result = mem._load_all_commit_embeddings()
        assert isinstance(result["C001"], np.ndarray)
        assert result["C001"].dtype == np.float32
        assert result["C001"].shape == (384,)
