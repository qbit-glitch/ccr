"""Tests for EverMemOS-inspired thematic commit clustering (arXiv:2601.02163).

Tests compute_clusters (connected components over cross-link graph),
keyword extraction, persistence (save/load), and context formatting.
"""

import json
import os
import tempfile

import pytest

from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def memory(project_dir):
    mem = MemoryManager(project_dir, CCRConfig())
    mem.ensure_structure()
    return mem


@pytest.fixture
def memory_with_commits(memory):
    """Pre-populate with commits that will produce entity cross-links."""
    # Auth cluster: C001 and C002 share auth.py
    memory.commit(
        "Add authentication",
        "Implemented JWT authentication module",
        "Security requirement",
        ["auth.py", "config.py"],
        "Add login page",
        admission_threshold=1.0,
    )
    memory.commit(
        "Fix auth bug",
        "Fixed token expiration in authentication module",
        "Tokens were not expiring",
        ["auth.py"],
        "Add refresh tokens",
        admission_threshold=1.0,
    )
    # DB cluster: C003 and C004 share db/repository.py
    memory.commit(
        "Add database layer",
        "Created repository pattern for database access",
        "Better separation of concerns",
        ["db/repository.py", "db/models.py"],
        "Add models",
        admission_threshold=1.0,
    )
    memory.commit(
        "Fix database queries",
        "Optimized slow database queries in repository",
        "Performance issues in database layer",
        ["db/repository.py"],
        "Add caching",
        admission_threshold=1.0,
    )
    # Isolated commit: C005 has no shared files with others
    memory.commit(
        "Add user profiles",
        "Implemented user profile endpoints",
        "Frontend needs profile data",
        ["routes/profile.py"],
        "Add avatar",
        admission_threshold=1.0,
    )
    return memory


@pytest.fixture
def memory_with_manual_links(memory):
    """Pre-populate with commits and manually set up cross-links for precise control."""
    # Create commits
    memory.commit("Add auth", "JWT auth", "Security", ["auth.py"], "Next",
                  admission_threshold=1.0)
    memory.commit("Fix auth", "Token fix", "Bug", ["auth.py"], "Next",
                  admission_threshold=1.0)
    memory.commit("Add DB", "Database layer", "Infra", ["db.py"], "Next",
                  admission_threshold=1.0)
    memory.commit("Fix DB", "Query optimization", "Perf", ["db.py"], "Next",
                  admission_threshold=1.0)
    memory.commit("Add docs", "Documentation update", "Docs", ["docs.md"], "Next",
                  admission_threshold=1.0)

    # Manually write clean cross-links
    links = {
        "version": 1,
        "links": {
            "C001": {
                "entity": [{"target": "C002", "score": 1.0, "shared_files": ["auth.py"]}],
                "causal": [], "supersession": [], "semantic": [],
            },
            "C002": {
                "entity": [{"target": "C001", "score": 1.0, "shared_files": ["auth.py"]}],
                "causal": [], "supersession": [], "semantic": [],
            },
            "C003": {
                "entity": [{"target": "C004", "score": 1.0, "shared_files": ["db.py"]}],
                "causal": [], "supersession": [], "semantic": [],
            },
            "C004": {
                "entity": [{"target": "C003", "score": 1.0, "shared_files": ["db.py"]}],
                "causal": [], "supersession": [], "semantic": [],
            },
            "C005": {
                "entity": [], "causal": [], "supersession": [], "semantic": [],
            },
        },
    }
    links_path = memory._get_links_path()
    with open(links_path, "w") as f:
        json.dump(links, f)

    return memory


# ---------------------------------------------------------------------------
# compute_clusters
# ---------------------------------------------------------------------------

class TestComputeClusters:
    def test_finds_clusters_from_entity_links(self, memory_with_manual_links):
        mem = memory_with_manual_links
        clusters = mem.compute_clusters(min_cluster_size=2)
        assert len(clusters) == 2
        all_ids = [cid for cl in clusters for cid in cl["commit_ids"]]
        assert "C001" in all_ids
        assert "C002" in all_ids
        assert "C003" in all_ids
        assert "C004" in all_ids
        # C005 is isolated — not in any cluster
        assert "C005" not in all_ids

    def test_min_cluster_size_filters(self, memory_with_manual_links):
        mem = memory_with_manual_links
        clusters = mem.compute_clusters(min_cluster_size=3)
        # No cluster has 3+ members in this data
        assert len(clusters) == 0

    def test_cluster_has_id_and_name(self, memory_with_manual_links):
        mem = memory_with_manual_links
        clusters = mem.compute_clusters(min_cluster_size=2)
        for cl in clusters:
            assert cl["id"].startswith("CL")
            assert cl["name"] != ""

    def test_cluster_has_keywords(self, memory_with_manual_links):
        mem = memory_with_manual_links
        clusters = mem.compute_clusters(min_cluster_size=2)
        for cl in clusters:
            assert isinstance(cl["top_keywords"], list)

    def test_empty_links_no_clusters(self, memory):
        """No links at all should produce no clusters."""
        clusters = memory.compute_clusters()
        assert clusters == []

    def test_low_score_links_filtered(self, memory):
        """Links below threshold should not form clusters."""
        # Create commits first
        memory.commit("A", "Something", "Why", ["a.py"], "Next",
                      admission_threshold=1.0)
        memory.commit("B", "Something else", "Why", ["b.py"], "Next",
                      admission_threshold=1.0)
        # Write low-score links
        links = {
            "version": 1,
            "links": {
                "C001": {
                    "entity": [],
                    "causal": [],
                    "supersession": [],
                    "semantic": [{"target": "C002", "score": 0.1}],
                },
                "C002": {
                    "entity": [],
                    "causal": [],
                    "supersession": [],
                    "semantic": [{"target": "C001", "score": 0.1}],
                },
            },
        }
        with open(memory._get_links_path(), "w") as f:
            json.dump(links, f)
        clusters = memory.compute_clusters(link_score_threshold=0.3)
        assert len(clusters) == 0

    def test_semantic_links_form_clusters(self, memory):
        """Semantic links (not just entity) should also form clusters."""
        memory.commit("Refactor auth", "Restructured", "Clean up", ["auth.py"], "N",
                      admission_threshold=1.0)
        memory.commit("Auth tests", "Added test coverage", "Quality", ["test_auth.py"], "N",
                      admission_threshold=1.0)
        links = {
            "version": 1,
            "links": {
                "C001": {
                    "entity": [],
                    "causal": [],
                    "supersession": [],
                    "semantic": [{"target": "C002", "score": 0.6}],
                },
                "C002": {
                    "entity": [],
                    "causal": [],
                    "supersession": [],
                    "semantic": [{"target": "C001", "score": 0.6}],
                },
            },
        }
        with open(memory._get_links_path(), "w") as f:
            json.dump(links, f)
        clusters = memory.compute_clusters(link_score_threshold=0.3)
        assert len(clusters) == 1
        assert sorted(clusters[0]["commit_ids"]) == ["C001", "C002"]

    def test_causal_and_supersession_links_ignored(self, memory):
        """Only entity and semantic links are used for clustering."""
        memory.commit("A", "Something", "Why", ["a.py"], "Next",
                      admission_threshold=1.0)
        memory.commit("B", "Something else", "Why", ["b.py"], "Next",
                      admission_threshold=1.0)
        links = {
            "version": 1,
            "links": {
                "C001": {
                    "entity": [],
                    "causal": [{"target": "C002", "score": 1.0}],
                    "supersession": [{"target": "C002", "score": 1.0}],
                    "semantic": [],
                },
                "C002": {
                    "entity": [],
                    "causal": [{"target": "C001", "score": 1.0}],
                    "supersession": [{"target": "C001", "score": 1.0}],
                    "semantic": [],
                },
            },
        }
        with open(memory._get_links_path(), "w") as f:
            json.dump(links, f)
        clusters = memory.compute_clusters()
        assert len(clusters) == 0

    def test_transitive_clustering(self, memory):
        """A-B and B-C should produce one cluster {A, B, C}."""
        memory.commit("A", "First", "Why", ["a.py"], "N", admission_threshold=1.0)
        memory.commit("B", "Second", "Why", ["a.py", "c.py"], "N", admission_threshold=1.0)
        memory.commit("C", "Third", "Why", ["c.py"], "N", admission_threshold=1.0)
        links = {
            "version": 1,
            "links": {
                "C001": {
                    "entity": [{"target": "C002", "score": 0.5, "shared_files": ["a.py"]}],
                    "causal": [], "supersession": [], "semantic": [],
                },
                "C002": {
                    "entity": [
                        {"target": "C001", "score": 0.5, "shared_files": ["a.py"]},
                        {"target": "C003", "score": 0.5, "shared_files": ["c.py"]},
                    ],
                    "causal": [], "supersession": [], "semantic": [],
                },
                "C003": {
                    "entity": [{"target": "C002", "score": 0.5, "shared_files": ["c.py"]}],
                    "causal": [], "supersession": [], "semantic": [],
                },
            },
        }
        with open(memory._get_links_path(), "w") as f:
            json.dump(links, f)
        clusters = memory.compute_clusters(min_cluster_size=2)
        assert len(clusters) == 1
        assert sorted(clusters[0]["commit_ids"]) == ["C001", "C002", "C003"]

    def test_commit_ids_sorted(self, memory_with_manual_links):
        """Commit IDs within each cluster should be sorted."""
        clusters = memory_with_manual_links.compute_clusters(min_cluster_size=2)
        for cl in clusters:
            assert cl["commit_ids"] == sorted(cl["commit_ids"])


# ---------------------------------------------------------------------------
# Cluster persistence
# ---------------------------------------------------------------------------

class TestClusterPersistence:
    def test_save_and_load(self, memory_with_manual_links):
        mem = memory_with_manual_links
        clusters = mem.compute_clusters(min_cluster_size=2)
        assert len(clusters) > 0

        # Load from file
        data = mem._load_clusters()
        assert len(data["clusters"]) == len(clusters)
        assert "commit_to_cluster" in data

        # Verify commit-to-cluster mapping
        for cl in clusters:
            for cid in cl["commit_ids"]:
                assert data["commit_to_cluster"][cid] == cl["id"]

    def test_load_missing_file(self, memory):
        data = memory._load_clusters()
        assert data["clusters"] == []
        assert data["commit_to_cluster"] == {}
        assert data["version"] == 1

    def test_load_corrupt_file(self, memory):
        path = memory._get_clusters_path()
        with open(path, "w") as f:
            f.write("not json")
        data = memory._load_clusters()
        assert data["clusters"] == []

    def test_load_invalid_structure(self, memory):
        path = memory._get_clusters_path()
        with open(path, "w") as f:
            json.dump({"version": 1, "clusters": "not a list"}, f)
        data = memory._load_clusters()
        assert data["clusters"] == []

    def test_clusters_path(self, memory):
        path = memory._get_clusters_path()
        assert path.endswith("commit_clusters.json")


# ---------------------------------------------------------------------------
# format_clusters_for_context
# ---------------------------------------------------------------------------

class TestFormatClusters:
    def test_format_empty(self, memory):
        assert memory.format_clusters_for_context() == ""

    def test_format_with_clusters(self, memory_with_manual_links):
        mem = memory_with_manual_links
        mem.compute_clusters(min_cluster_size=2)
        text = mem.format_clusters_for_context()
        assert "Thematic Clusters" in text
        assert "CL" in text

    def test_format_truncates_long_commit_lists(self, memory):
        """Clusters with 6+ commits should show (+N more)."""
        # Manually save a cluster with many commits
        clusters = [{
            "id": "CL001",
            "name": "Big Cluster",
            "commit_ids": [f"C{i:03d}" for i in range(1, 9)],
            "top_keywords": ["big"],
        }]
        memory._save_clusters(clusters)
        text = memory.format_clusters_for_context()
        assert "(+3 more)" in text


# ---------------------------------------------------------------------------
# _extract_cluster_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_extracts_keywords(self, memory_with_manual_links):
        mem = memory_with_manual_links
        keywords = mem._extract_cluster_keywords("main", ["C001", "C002"])
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should find "auth" related words
        combined = " ".join(keywords)
        assert "auth" in combined or "token" in combined or "jwt" in combined

    def test_empty_commits(self, memory):
        keywords = memory._extract_cluster_keywords("main", [])
        assert keywords == []

    def test_missing_commits(self, memory):
        keywords = memory._extract_cluster_keywords("main", ["C999"])
        assert keywords == []

    def test_filters_stop_words(self, memory_with_manual_links):
        mem = memory_with_manual_links
        keywords = mem._extract_cluster_keywords("main", ["C001", "C002"])
        stop_words = {"the", "and", "for", "was", "not"}
        for kw in keywords:
            assert kw not in stop_words

    def test_max_10_keywords(self, memory_with_manual_links):
        mem = memory_with_manual_links
        # Use all commits to get a richer keyword set
        keywords = mem._extract_cluster_keywords(
            "main", ["C001", "C002", "C003", "C004", "C005"]
        )
        assert len(keywords) <= 10


# ---------------------------------------------------------------------------
# Integration with real commits (using compute_links)
# ---------------------------------------------------------------------------

class TestClusteringWithRealCommits:
    def test_clusters_from_real_entity_links(self, memory_with_commits):
        """Commits with shared files should cluster via real entity links."""
        mem = memory_with_commits
        # Verify links exist
        links_data = mem._load_links()
        all_links = links_data.get("links", {})
        # There should be some entity links from shared files
        has_entity = any(
            typed.get("entity")
            for typed in all_links.values()
        )
        if not has_entity:
            pytest.skip("No entity links generated (link_scan_window might be too small)")

        clusters = mem.compute_clusters(min_cluster_size=2)
        # Should have at least one cluster
        assert len(clusters) >= 1
