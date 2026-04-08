"""Tests for retroactive commit linking (A-MEM + MAGMA inspired).

Tests the link graph I/O, link computation heuristics (entity, causal,
supersession, semantic), bidirectional storage, BFS retrieval, and
integration with gcc_commit and gcc_context.
"""

import json
import os
import tempfile

import pytest

from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig, CommitLink


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
    """Pre-populate with diverse commits for link testing."""
    memory.commit("Setup MCP server", "Created mcp_server.py with 5 tools",
                  "Need tool infrastructure", ["ccr/mcp_server.py", "ccr/core/types.py"],
                  "Add more tools", admission_threshold=1.0)
    memory.commit("Add memory module", "Implemented MemoryManager class",
                  "Core persistence layer needed", ["ccr/core/memory.py"],
                  "Wire into MCP", admission_threshold=1.0)
    memory.commit("Fix MCP bug", "Fixed exception handling in gcc_commit",
                  "Bug discovered in C001", ["ccr/mcp_server.py"],
                  "Add tests", admission_threshold=1.0)
    memory.commit("Add playbook", "Implemented ACE playbook data structure",
                  "Need self-evolving strategies", ["ccr/ace/playbook.py", "ccr/core/types.py"],
                  "Wire ACE tools", admission_threshold=1.0)
    memory.commit("Refactor types", "Replaced the approach from C001 for type definitions",
                  "Old types were too rigid", ["ccr/core/types.py"],
                  "Update tests", admission_threshold=1.0)
    return memory


# ---------------------------------------------------------------------------
# CommitLink dataclass
# ---------------------------------------------------------------------------

class TestCommitLink:
    def test_to_dict_minimal(self):
        link = CommitLink(target="C005", link_type="semantic", score=0.42)
        d = link.to_dict()
        assert d == {"target": "C005", "score": 0.42}

    def test_to_dict_with_shared_files(self):
        link = CommitLink(target="C005", link_type="entity", score=0.67,
                          shared_files=["mcp_server.py"])
        d = link.to_dict()
        assert d["shared_files"] == ["mcp_server.py"]

    def test_to_dict_with_snippet(self):
        link = CommitLink(target="C008", link_type="causal", score=1.0,
                          snippet="fixing the bug from C008")
        d = link.to_dict()
        assert d["snippet"] == "fixing the bug from C008"

    def test_from_dict(self):
        d = {"target": "C005", "score": 0.67, "shared_files": ["a.py"]}
        link = CommitLink.from_dict("entity", d)
        assert link.target == "C005"
        assert link.link_type == "entity"
        assert link.score == 0.67
        assert link.shared_files == ["a.py"]

    def test_from_dict_defaults(self):
        d = {"target": "C005"}
        link = CommitLink.from_dict("semantic", d)
        assert link.score == 0.0
        assert link.shared_files == []
        assert link.snippet == ""


# ---------------------------------------------------------------------------
# Link Storage (I/O)
# ---------------------------------------------------------------------------

class TestLinkStorage:
    def test_links_path_location(self, memory):
        path = memory._get_links_path()
        assert path.endswith("commit_links.json")
        assert ".ccr" in path

    def test_load_links_missing_file_returns_default(self, memory):
        data = memory._load_links()
        assert data == {"version": 1, "links": {}}

    def test_load_links_corrupt_json_returns_default(self, memory):
        path = memory._get_links_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("not valid json {{{")
        data = memory._load_links()
        assert data == {"version": 1, "links": {}}

    def test_load_links_invalid_structure_returns_default(self, memory):
        path = memory._get_links_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"version": 1, "links": "not a dict"}, f)
        data = memory._load_links()
        assert data == {"version": 1, "links": {}}

    def test_save_and_load_roundtrip(self, memory):
        data = {"version": 1, "links": {
            "C001": {"entity": [{"target": "C002", "score": 0.5}]}
        }}
        memory._save_links(data)
        loaded = memory._load_links()
        assert loaded == data

    def test_atomic_write(self, memory):
        """Save creates the file even if it didn't exist before."""
        path = memory._get_links_path()
        assert not os.path.exists(path)
        memory._save_links({"version": 1, "links": {}})
        assert os.path.exists(path)

    def test_empty_file_returns_default(self, memory):
        path = memory._get_links_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("")
        data = memory._load_links()
        assert data == {"version": 1, "links": {}}


# ---------------------------------------------------------------------------
# _add_link (bidirectional)
# ---------------------------------------------------------------------------

class TestAddLink:
    def test_adds_both_directions(self):
        data = {"version": 1, "links": {}}
        link = CommitLink(target="C005", link_type="entity", score=0.6,
                          shared_files=["a.py"])
        MemoryManager._add_link(data, "C010", "C005", link)
        assert len(data["links"]["C010"]["entity"]) == 1
        assert data["links"]["C010"]["entity"][0]["target"] == "C005"
        assert len(data["links"]["C005"]["entity"]) == 1
        assert data["links"]["C005"]["entity"][0]["target"] == "C010"

    def test_dedup_keeps_higher_score(self):
        data = {"version": 1, "links": {}}
        link1 = CommitLink(target="C005", link_type="entity", score=0.3)
        link2 = CommitLink(target="C005", link_type="entity", score=0.8)
        MemoryManager._add_link(data, "C010", "C005", link1)
        MemoryManager._add_link(data, "C010", "C005", link2)
        # Forward: should still be 1 entry, with score 0.8
        assert len(data["links"]["C010"]["entity"]) == 1
        assert data["links"]["C010"]["entity"][0]["score"] == 0.8
        # Reverse: should also be 1 entry with correct target and score
        assert len(data["links"]["C005"]["entity"]) == 1
        assert data["links"]["C005"]["entity"][0]["score"] == 0.8
        assert data["links"]["C005"]["entity"][0]["target"] == "C010"

    def test_dedup_ignores_lower_score(self):
        data = {"version": 1, "links": {}}
        link1 = CommitLink(target="C005", link_type="entity", score=0.9)
        link2 = CommitLink(target="C005", link_type="entity", score=0.4)
        MemoryManager._add_link(data, "C010", "C005", link1)
        MemoryManager._add_link(data, "C010", "C005", link2)
        assert data["links"]["C010"]["entity"][0]["score"] == 0.9
        # Reverse: should keep original higher score, count stays 1
        assert len(data["links"]["C005"]["entity"]) == 1
        assert data["links"]["C005"]["entity"][0]["score"] == 0.9
        assert data["links"]["C005"]["entity"][0]["target"] == "C010"

    def test_different_types_not_deduped(self):
        data = {"version": 1, "links": {}}
        link1 = CommitLink(target="C005", link_type="entity", score=0.6)
        link2 = CommitLink(target="C005", link_type="causal", score=1.0,
                           snippet="refs C005")
        MemoryManager._add_link(data, "C010", "C005", link1)
        MemoryManager._add_link(data, "C010", "C005", link2)
        assert len(data["links"]["C010"]["entity"]) == 1
        assert len(data["links"]["C010"]["causal"]) == 1


# ---------------------------------------------------------------------------
# Commit Reference Extraction
# ---------------------------------------------------------------------------

class TestExtractCommitReferences:
    def test_single_reference(self):
        refs = MemoryManager._extract_commit_references("Fixed bug from C008")
        assert refs == ["C008"]

    def test_multiple_references(self):
        refs = MemoryManager._extract_commit_references("Merged C005 and C012 changes")
        assert refs == ["C005", "C012"]

    def test_no_references(self):
        refs = MemoryManager._extract_commit_references("Fixed a regular bug")
        assert refs == []

    def test_word_boundary(self):
        """Should not match partial IDs like 'C12' (less than 3 digits)."""
        refs = MemoryManager._extract_commit_references("Variable C12 is fine")
        assert refs == []

    def test_four_digit_id(self):
        refs = MemoryManager._extract_commit_references("See C1234 for details")
        assert refs == ["C1234"]

    # --- False positive tests (F11) ---

    def test_no_match_inside_hex(self):
        """Should not match inside hex addresses like 0xC0001234."""
        refs = MemoryManager._extract_commit_references("Address 0xC0001234")
        assert refs == []  # No word boundary before C

    def test_no_match_lowercase(self):
        """Commit IDs are uppercase C only."""
        refs = MemoryManager._extract_commit_references("See c001 for context")
        assert refs == []

    def test_no_match_in_url(self):
        """Should not match inside URLs or paths."""
        refs = MemoryManager._extract_commit_references("See /path/C001/file.txt")
        # /C001/ has word boundaries around C001 so this WILL match
        # This is a known limitation — documenting expected behavior
        assert refs == ["C001"]

    def test_no_match_config_key(self):
        """Underscored config keys like C001_TIMEOUT should NOT match (no word boundary)."""
        refs = MemoryManager._extract_commit_references("Set C001_TIMEOUT=30")
        # _ is a word character, so \b does NOT fire between C001 and _
        assert refs == []

    def test_no_match_two_digit(self):
        refs = MemoryManager._extract_commit_references("C99 standard")
        assert refs == []


# ---------------------------------------------------------------------------
# Supersession Detection
# ---------------------------------------------------------------------------

class TestDetectSupersession:
    def test_replaced_keyword(self):
        text = "Replaced the approach from C003 with a new design"
        results = MemoryManager._detect_supersession(text)
        assert len(results) == 1
        assert results[0][0] == "C003"

    def test_superseded_keyword(self):
        text = "This superseded C012 entirely"
        results = MemoryManager._detect_supersession(text)
        assert len(results) == 1
        assert results[0][0] == "C012"

    def test_reverted_keyword(self):
        text = "Reverted C005 due to regressions"
        results = MemoryManager._detect_supersession(text)
        assert len(results) == 1
        assert results[0][0] == "C005"

    def test_refactored_from_keyword(self):
        text = "Refactored from C008 to use cleaner patterns"
        results = MemoryManager._detect_supersession(text)
        assert len(results) == 1
        assert results[0][0] == "C008"

    def test_no_supersession_without_keyword(self):
        text = "See C008 for context"
        results = MemoryManager._detect_supersession(text)
        assert results == []

    def test_keyword_far_from_commit_id(self):
        """Keyword must be within 120 chars of commit ID."""
        text = "Replaced the old code. " + "x" * 150 + " C003 is unrelated"
        results = MemoryManager._detect_supersession(text)
        assert results == []


# ---------------------------------------------------------------------------
# Entity Links
# ---------------------------------------------------------------------------

class TestEntityLinks:
    def test_shared_file_creates_entity_link(self, memory_with_commits):
        # C001 touched mcp_server.py, C003 also touched mcp_server.py
        links = memory_with_commits.get_commit_links("C003")
        entity = links.get("entity", [])
        targets = {e["target"] for e in entity}
        assert "C001" in targets

    def test_no_shared_files_no_entity_link(self, memory):
        memory.commit("Commit A", "Did A", "Reason A", ["a.py"], "Next",
                      admission_threshold=1.0)
        memory.commit("Commit B", "Did B", "Reason B", ["b.py"], "Next",
                      admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        entity = links.get("entity", [])
        assert len(entity) == 0

    def test_entity_link_score_is_jaccard(self, memory):
        memory.commit("A", "X", "Y", ["a.py", "b.py"], "N", admission_threshold=1.0)
        memory.commit("B", "X", "Y", ["b.py", "c.py"], "N", admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        entity = links.get("entity", [])
        assert len(entity) == 1
        # Jaccard({a,b}, {b,c}) = 1/3 ≈ 0.333
        assert abs(entity[0]["score"] - 0.333) < 0.01

    def test_shared_files_recorded(self, memory):
        memory.commit("A", "X", "Y", ["a.py", "b.py"], "N", admission_threshold=1.0)
        memory.commit("B", "X", "Y", ["b.py", "c.py"], "N", admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        entity = links["entity"]
        assert "b.py" in entity[0]["shared_files"]

    def test_multiple_shared_files(self, memory):
        memory.commit("A", "X", "Y", ["a.py", "b.py", "c.py"], "N",
                      admission_threshold=1.0)
        memory.commit("B", "X", "Y", ["a.py", "b.py", "d.py"], "N",
                      admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        entity = links["entity"]
        assert set(entity[0]["shared_files"]) == {"a.py", "b.py"}


# ---------------------------------------------------------------------------
# Causal Links
# ---------------------------------------------------------------------------

class TestCausalLinks:
    def test_commit_id_in_why_creates_causal_link(self, memory_with_commits):
        # C003 has "Bug discovered in C001" in why field
        links = memory_with_commits.get_commit_links("C003")
        causal = links.get("causal", [])
        targets = {e["target"] for e in causal}
        assert "C001" in targets

    def test_causal_link_has_snippet(self, memory_with_commits):
        links = memory_with_commits.get_commit_links("C003")
        causal = [e for e in links.get("causal", []) if e["target"] == "C001"]
        assert len(causal) == 1
        assert "C001" in causal[0].get("snippet", "")

    def test_no_commit_id_no_causal_link(self, memory):
        memory.commit("A", "Did stuff", "No reason", ["a.py"], "N",
                      admission_threshold=1.0)
        memory.commit("B", "More stuff", "Also no reason", ["b.py"], "N",
                      admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        causal = links.get("causal", [])
        assert len(causal) == 0

    def test_causal_link_score_is_one(self, memory):
        memory.commit("A", "Did stuff", "Reason", ["a.py"], "N",
                      admission_threshold=1.0)
        memory.commit("B", "Fixed bug from C001", "See C001", ["b.py"], "N",
                      admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        causal = links.get("causal", [])
        assert len(causal) >= 1
        assert causal[0]["score"] == 1.0

    def test_multiple_causal_refs(self, memory):
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        memory.commit("B", "X", "Y", ["b.py"], "N", admission_threshold=1.0)
        memory.commit("C", "Combined fixes from C001 and C002", "Merge work",
                      ["c.py"], "N", admission_threshold=1.0)
        links = memory.get_commit_links("C003")
        causal = links.get("causal", [])
        targets = {e["target"] for e in causal}
        assert "C001" in targets
        assert "C002" in targets


# ---------------------------------------------------------------------------
# Supersession Links
# ---------------------------------------------------------------------------

class TestSupersessionLinks:
    def test_supersession_creates_link(self, memory_with_commits):
        # C005 has "Replaced the approach from C001"
        links = memory_with_commits.get_commit_links("C005")
        supersession = links.get("supersession", [])
        targets = {e["target"] for e in supersession}
        assert "C001" in targets

    def test_supersession_subsumes_causal(self, memory):
        """When supersession detected, should NOT also create causal for same target."""
        memory.commit("A", "Original approach", "First try", ["a.py"], "N",
                      admission_threshold=1.0)
        memory.commit("B", "Replaced the approach from C001",
                      "C001 was wrong", ["a.py"], "N", admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        supersession = links.get("supersession", [])
        causal = links.get("causal", [])
        sup_targets = {e["target"] for e in supersession}
        causal_targets = {e["target"] for e in causal}
        assert "C001" in sup_targets
        # C001 should NOT also appear as causal
        assert "C001" not in causal_targets

    def test_supersession_has_snippet(self, memory_with_commits):
        links = memory_with_commits.get_commit_links("C005")
        supersession = [e for e in links.get("supersession", []) if e["target"] == "C001"]
        assert len(supersession) == 1
        assert "C001" in supersession[0].get("snippet", "")

    def test_no_supersession_without_replacement_language(self, memory):
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        memory.commit("B", "See C001 for context", "Referenced C001",
                      ["b.py"], "N", admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        supersession = links.get("supersession", [])
        assert len(supersession) == 0


# ---------------------------------------------------------------------------
# Semantic Links
# ---------------------------------------------------------------------------

class TestSemanticLinks:
    def test_high_keyword_overlap_creates_link(self, memory):
        memory.commit("Implement authentication module with JWT tokens",
                      "Added JWT authentication with token refresh and validation",
                      "Security requirements for authentication", ["auth.py"], "Add tests",
                      admission_threshold=1.0)
        memory.commit("Fix authentication module JWT token handling",
                      "Fixed JWT authentication token refresh and validation bugs",
                      "Security requirements for authentication", ["security.py"], "Deploy",
                      admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        semantic = links.get("semantic", [])
        # High keyword overlap (authentication, JWT, token, security)
        assert len(semantic) >= 1

    def test_low_keyword_overlap_no_link(self, memory):
        memory.commit("Setup database", "Created PostgreSQL schema",
                      "Need data persistence", ["db.py"], "N",
                      admission_threshold=1.0)
        memory.commit("Fix CSS layout", "Adjusted flexbox alignment",
                      "UI was broken on mobile", ["styles.css"], "N",
                      admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        semantic = links.get("semantic", [])
        assert len(semantic) == 0

    def test_semantic_threshold_configurable(self, memory):
        """With threshold=0.0, even low overlap creates semantic links."""
        memory.config.link_semantic_threshold = 0.0
        memory.commit("A unique topic", "Something about cats",
                      "Cat reason", ["cat.py"], "N", admission_threshold=1.0)
        memory.commit("Another unique topic", "Something about cats",
                      "Cat reason too", ["dog.py"], "N", admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        semantic = links.get("semantic", [])
        # With threshold=0.0, any keyword overlap should create a link
        assert len(semantic) >= 1

    def test_semantic_not_created_when_entity_exists(self, memory):
        """Entity links suppress semantic links to the same target."""
        memory.commit("Setup module A", "Created module with many shared concepts",
                      "Foundation work", ["shared.py"], "N",
                      admission_threshold=1.0)
        memory.commit("Extend module A", "Extended module with same concepts",
                      "Building on foundation", ["shared.py"], "N",
                      admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        entity = links.get("entity", [])
        semantic = links.get("semantic", [])
        # Should have entity link (shared.py), NOT semantic
        assert len(entity) >= 1
        entity_targets = {e["target"] for e in entity}
        semantic_targets = {e["target"] for e in semantic}
        assert not (entity_targets & semantic_targets), "Entity and semantic should not overlap"


# ---------------------------------------------------------------------------
# Bidirectional
# ---------------------------------------------------------------------------

class TestBidirectional:
    def test_link_stored_both_directions(self, memory):
        memory.commit("A", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        memory.commit("B", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        links_c2 = memory.get_commit_links("C002")
        links_c1 = memory.get_commit_links("C001")
        # C002 should link to C001 and vice versa
        c2_entity_targets = {e["target"] for e in links_c2.get("entity", [])}
        c1_entity_targets = {e["target"] for e in links_c1.get("entity", [])}
        assert "C001" in c2_entity_targets
        assert "C002" in c1_entity_targets

    def test_reverse_link_has_same_score(self, memory):
        memory.commit("A", "X", "Y", ["a.py", "b.py"], "N", admission_threshold=1.0)
        memory.commit("B", "X", "Y", ["b.py", "c.py"], "N", admission_threshold=1.0)
        links_c2 = memory.get_commit_links("C002")
        links_c1 = memory.get_commit_links("C001")
        score_fwd = links_c2["entity"][0]["score"]
        score_rev = links_c1["entity"][0]["score"]
        assert score_fwd == score_rev

    def test_causal_link_bidirectional(self, memory):
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        memory.commit("B", "Fixed bug from C001", "See C001", ["b.py"], "N",
                      admission_threshold=1.0)
        links_c1 = memory.get_commit_links("C001")
        causal = links_c1.get("causal", [])
        targets = {e["target"] for e in causal}
        assert "C002" in targets


# ---------------------------------------------------------------------------
# _compute_links
# ---------------------------------------------------------------------------

class TestComputeLinks:
    def test_no_existing_commits_returns_empty(self, memory):
        links = memory._compute_links("main", "C001", "Title", "What",
                                       "Why", ["a.py"], "Next")
        assert links == []

    def test_scan_window_config(self, memory):
        """Only scans the configured window of recent commits."""
        memory.config.link_scan_window = 2
        for i in range(5):
            memory.commit(f"Commit {i}", f"Did {i}", f"Why {i}",
                         ["shared.py"], "N", admission_threshold=1.0)
        # C005 should only scan C004 and C003, not C001/C002
        # (Though it will still have links via bidirectional from earlier commits)
        links = memory._compute_links("main", "C006", "New commit", "X",
                                       "Y", ["shared.py"], "N")
        targets = {l.target for l in links}
        # Should see recent commits within window, but not more than window size
        assert 1 <= len(targets) <= 2

    def test_mixed_link_types(self, memory):
        memory.commit("A", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        links = memory._compute_links(
            "main", "C002", "Fix bug",
            "Fixed bug from C001", "See C001", ["shared.py"], "Next",
        )
        types = {l.link_type for l in links}
        # Should have entity (shared.py) and causal (C001 reference)
        assert "entity" in types
        assert "causal" in types

    def test_self_reference_excluded(self, memory):
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        links = memory._compute_links("main", "C001", "X", "Y", "Z",
                                       ["a.py"], "N")
        targets = {l.target for l in links}
        assert "C001" not in targets


# ---------------------------------------------------------------------------
# Link Retrieval (get_commit_links, get_linked_commits)
# ---------------------------------------------------------------------------

class TestLinkRetrieval:
    def test_get_commit_links_empty(self, memory):
        links = memory.get_commit_links("C999")
        assert links == {"entity": [], "causal": [], "supersession": [], "semantic": []}

    def test_get_commit_links_returns_all_types(self, memory_with_commits):
        links = memory_with_commits.get_commit_links("C001")
        assert "entity" in links
        assert "causal" in links
        assert "supersession" in links
        assert "semantic" in links

    def test_get_linked_commits_one_hop(self, memory_with_commits):
        linked = memory_with_commits.get_linked_commits("C001")
        assert len(linked) > 0
        assert all(e["hop"] == 1 for e in linked)

    def test_get_linked_commits_two_hops(self, memory_with_commits):
        linked = memory_with_commits.get_linked_commits("C001", max_hops=2)
        hops = {e["hop"] for e in linked}
        # Should have at least hop-1 entries
        assert 1 in hops

    def test_get_linked_commits_no_cycles(self, memory):
        """BFS should not revisit the source commit."""
        memory.commit("A", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        memory.commit("B", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        linked = memory.get_linked_commits("C001", max_hops=3)
        ids = [e["id"] for e in linked]
        assert "C001" not in ids
        assert len(ids) == len(set(ids))  # No duplicates

    def test_get_linked_commits_caps_at_config_max(self, memory):
        """Results should be capped at config.link_max_results."""
        for i in range(15):
            memory.commit(f"Commit {i}", "X", "Y", ["shared.py"], "N",
                         admission_threshold=1.0)
        linked = memory.get_linked_commits("C001", max_hops=3)
        assert len(linked) <= memory.config.link_max_results

    def test_get_linked_commits_custom_cap(self, memory):
        """Custom link_max_results should be respected."""
        memory.config.link_max_results = 3
        for i in range(10):
            memory.commit(f"Commit {i}", "X", "Y", ["shared.py"], "N",
                         admission_threshold=1.0)
        linked = memory.get_linked_commits("C001", max_hops=3)
        assert len(linked) <= 3

    def test_get_linked_commits_filtered_by_type(self, memory_with_commits):
        linked = memory_with_commits.get_linked_commits("C001", link_types=["entity"])
        for e in linked:
            assert e["link_type"] == "entity"


# ---------------------------------------------------------------------------
# Context Integration
# ---------------------------------------------------------------------------

class TestContextIntegration:
    def test_context_level5_shows_links(self, memory_with_commits):
        ctx = memory_with_commits.get_context(level=5, commit_id="C001")
        assert "Links for C001" in ctx

    def test_context_level5_no_links_no_section(self, memory):
        memory.commit("Solo commit", "X", "Y", ["a.py"], "N",
                      admission_threshold=1.0)
        ctx = memory.get_context(level=5, commit_id="C001")
        assert "Links for" not in ctx

    def test_context_follow_links(self, memory_with_commits):
        ctx = memory_with_commits.get_context(
            level=5, commit_id="C001", follow_links=True,
        )
        assert "Linked:" in ctx

    def test_context_without_follow_links_no_expansion(self, memory_with_commits):
        ctx = memory_with_commits.get_context(
            level=5, commit_id="C001", follow_links=False,
        )
        # Should have links section but not expanded linked commits
        assert "Links for C001" in ctx
        # The "Linked:" sections should not appear
        assert "Linked: [" not in ctx


# ---------------------------------------------------------------------------
# Commit Integration
# ---------------------------------------------------------------------------

class TestCommitIntegration:
    def test_commit_creates_links_automatically(self, memory):
        memory.commit("A", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        memory.commit("B", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)
        # Links should have been created automatically during commit
        links = memory.get_commit_links("C002")
        entity = links.get("entity", [])
        assert len(entity) >= 1

    def test_links_file_created_after_linked_commit(self, memory):
        path = memory._get_links_path()
        assert not os.path.exists(path)
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        # First commit has nothing to link to
        memory.commit("B", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        # Second commit should have created the file
        assert os.path.exists(path)

    def test_commit_succeeds_even_if_linking_fails(self, memory):
        """Linking errors should not prevent the commit from succeeding."""
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        # Corrupt the links file
        path = memory._get_links_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.makedirs(path, exist_ok=True)  # Make it a directory to cause write failure
        result = memory.commit("B", "X", "Y", ["a.py"], "N",
                               admission_threshold=1.0)
        assert result.startswith("[C002]")
        # Cleanup: remove the directory so other tests aren't affected
        os.rmdir(path)

    def test_first_commit_no_links(self, memory):
        """First commit should not error even with no prior commits."""
        result = memory.commit("First", "X", "Y", ["a.py"], "N",
                               admission_threshold=1.0)
        assert result.startswith("[C001]")
        links = memory.get_commit_links("C001")
        assert all(len(v) == 0 for v in links.values())


# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_old_ccr_dir_without_links_file(self, memory):
        """Existing .ccr/ directories without commit_links.json should work."""
        links = memory.get_commit_links("C001")
        assert links == {"entity": [], "causal": [], "supersession": [], "semantic": []}

    def test_get_context_no_links_file(self, memory):
        """Context retrieval should work without links file."""
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        ctx = memory.get_context(level=5, commit_id="C001")
        assert "C001" in ctx


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_entity_links(self, memory):
        links = {
            "entity": [{"target": "C005", "score": 0.67,
                         "shared_files": ["mcp_server.py", "memory.py"]}],
            "causal": [], "supersession": [], "semantic": [],
        }
        out = memory._format_links_for_context("C012", links)
        assert "Links for C012" in out
        assert "Entity" in out
        assert "mcp_server.py" in out

    def test_format_causal_links(self, memory):
        links = {
            "entity": [],
            "causal": [{"target": "C008", "score": 1.0,
                         "snippet": "fixing the bug from C008"}],
            "supersession": [], "semantic": [],
        }
        out = memory._format_links_for_context("C012", links)
        assert "Causal" in out
        assert "C008" in out
        assert "fixing the bug from C008" in out

    def test_format_empty_links(self, memory):
        links = {"entity": [], "causal": [], "supersession": [], "semantic": []}
        out = memory._format_links_for_context("C012", links)
        assert "Links for C012" in out
        # Should only have the header, no type sections
        assert "Entity" not in out


# ---------------------------------------------------------------------------
# _parse_commit_block
# ---------------------------------------------------------------------------

class TestParseCommitBlock:
    def test_parse_typical_block(self):
        block = (
            "## [C012] 2026-03-12 17:55 | branch:main | Fix MCP bug\n"
            "**What**: Fixed exception handling\n"
            "**Why**: Bug in production\n"
            "**Files**: mcp_server.py\n"
            "**Next**: Add tests\n"
        )
        parsed = MemoryManager._parse_commit_block(block)
        assert parsed["title"] == "Fix MCP bug"
        assert parsed["what"] == "Fixed exception handling"
        assert parsed["why"] == "Bug in production"

    def test_parse_hyphenated_branch(self):
        block = (
            "## [C012] 2026-03-12 17:55 | branch:fix-auth-bug | Fix token refresh\n"
            "**What**: Fixed JWT token refresh logic\n"
            "**Why**: Tokens were expiring prematurely\n"
            "**Files**: auth.py\n"
            "**Next**: Add tests\n"
        )
        parsed = MemoryManager._parse_commit_block(block)
        assert parsed["title"] == "Fix token refresh"
        assert parsed["what"] == "Fixed JWT token refresh logic"

    def test_parse_empty_block(self):
        parsed = MemoryManager._parse_commit_block("")
        assert parsed == {}


# ---------------------------------------------------------------------------
# Causal Link False Positives (non-existent commit IDs)
# ---------------------------------------------------------------------------

class TestCausalFalsePositives:
    def test_reference_to_nonexistent_commit_ignored(self, memory):
        """C### reference to a commit that doesn't exist should NOT create a link."""
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        # C002 references C999 which doesn't exist
        memory.commit("B", "Fixed bug from C999", "See C999", ["b.py"], "N",
                      admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        causal = links.get("causal", [])
        targets = {e["target"] for e in causal}
        assert "C999" not in targets

    def test_reference_to_future_commit_ignored(self, memory):
        """Reference to a commit ID not yet created should NOT create a link."""
        memory.commit("A", "Prep for C005", "Will be needed by C005",
                      ["a.py"], "N", admission_threshold=1.0)
        links = memory.get_commit_links("C001")
        causal = links.get("causal", [])
        targets = {e["target"] for e in causal}
        assert "C005" not in targets

    def test_supersession_to_nonexistent_commit_ignored(self, memory):
        """Supersession referencing a non-existent commit should NOT create a link."""
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        memory.commit("B", "Replaced the approach from C999",
                      "C999 was wrong", ["a.py"], "N", admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        supersession = links.get("supersession", [])
        targets = {e["target"] for e in supersession}
        assert "C999" not in targets

    def test_valid_and_invalid_refs_mixed(self, memory):
        """Only valid references should create links; invalid ones filtered out."""
        memory.commit("A", "X", "Y", ["a.py"], "N", admission_threshold=1.0)
        memory.commit("B", "Combined fixes from C001 and C999",
                      "Both referenced", ["b.py"], "N", admission_threshold=1.0)
        links = memory.get_commit_links("C002")
        causal = links.get("causal", [])
        targets = {e["target"] for e in causal}
        assert "C001" in targets
        assert "C999" not in targets


# ---------------------------------------------------------------------------
# Thread Safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_link_updates(self, memory):
        """Multiple threads writing links simultaneously should not corrupt data."""
        import threading

        memory.commit("A", "X", "Y", ["shared.py"], "N", admission_threshold=1.0)

        errors = []

        def make_commit(idx):
            try:
                memory.commit(f"Thread {idx}", f"Work {idx}", f"Why {idx}",
                              ["shared.py"], "Next", admission_threshold=1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_commit, args=(i,))
                   for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        # All commits should have created links
        data = memory._load_links()
        assert len(data["links"]) > 0

    def test_concurrent_link_reads(self, memory_with_commits):
        """Concurrent reads should not error."""
        import threading

        errors = []

        def read_links(cid):
            try:
                memory_with_commits.get_commit_links(cid)
                memory_with_commits.get_linked_commits(cid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_links, args=(f"C00{i+1}",))
                   for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Read errors: {errors}"


# ---------------------------------------------------------------------------
# Large History Scaling
# ---------------------------------------------------------------------------

class TestLargeHistory:
    def test_large_scan_window_performance(self, memory):
        """Creating links with many prior commits should complete without error."""
        memory.config.link_scan_window = 50
        for i in range(30):
            memory.commit(f"Commit {i}", f"Did {i}", f"Why {i}",
                         ["shared.py", f"file_{i}.py"], "N",
                         admission_threshold=1.0)
        # Last commit should have links to many prior commits via shared.py
        links = memory.get_commit_links(f"C{30:03d}")
        entity = links.get("entity", [])
        assert len(entity) > 0

    def test_bfs_with_dense_graph(self, memory):
        """BFS should respect max_results even with a highly connected graph."""
        memory.config.link_max_results = 5
        for i in range(20):
            memory.commit(f"Commit {i}", f"Did {i}", f"Why {i}",
                         ["shared.py"], "N", admission_threshold=1.0)
        linked = memory.get_linked_commits("C001", max_hops=5)
        assert len(linked) <= 5


# ===========================================================================
# F2: Temporal Link Aging — EverMemOS + MAGMA
# ===========================================================================


class TestLinkAgeWeight:
    """_link_age_weight() applies exponential decay based on link created_at."""

    def test_no_created_at_returns_one(self):
        """Missing created_at → weight = 1.0 (no decay)."""
        from ccr.core.memory_pkg.memory_links import _link_age_weight
        assert _link_age_weight({}) == 1.0
        assert _link_age_weight({"score": 0.9}) == 1.0

    def test_fresh_link_near_one(self):
        """Link created just now → weight ≈ 1.0."""
        from ccr.core.memory_pkg.memory_links import _link_age_weight
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        w = _link_age_weight({"created_at": now})
        assert 0.999 <= w <= 1.001

    def test_old_link_lower_weight(self):
        """60-day-old link has lower weight than 1-day-old link."""
        from ccr.core.memory_pkg.memory_links import _link_age_weight
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        fresh = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        assert _link_age_weight({"created_at": old}) < _link_age_weight({"created_at": fresh})

    def test_invalid_created_at_returns_one(self):
        """Malformed created_at → weight = 1.0 (graceful)."""
        from ccr.core.memory_pkg.memory_links import _link_age_weight
        assert _link_age_weight({"created_at": "not-a-date"}) == 1.0

    def test_new_link_has_created_at(self, memory):
        """Links stored via memory.commit() get created_at stamped."""
        memory.commit("A", "Did A", "Why A", ["a.py", "common.py"], "N", admission_threshold=1.0)
        memory.commit("B", "Did B", "Why B", ["b.py", "common.py"], "N", admission_threshold=1.0)
        data = memory._load_links()
        all_entries = [
            entry
            for node in data.get("links", {}).values()
            for lt_list in node.values()
            for entry in lt_list
        ]
        assert any("created_at" in e for e in all_entries), "At least one link should have created_at"


# ===========================================================================
# F3: Adaptive Beam Width — MAGMA Algorithm 1
# ===========================================================================


class TestPruneFrontier:
    """_prune_frontier() removes low-score candidates from BFS frontier."""

    def test_adaptive_false_returns_unchanged(self):
        """adaptive=False → candidates unchanged."""
        from ccr.core.memory_pkg.memory_links import _prune_frontier
        candidates = [(0.9, "C001"), (0.8, "C002"), (0.1, "C003"), (0.05, "C004"), (0.07, "C005")]
        result = _prune_frontier(candidates, adaptive=False)
        assert result == candidates

    def test_fewer_than_3_unchanged(self):
        """< 3 candidates → no pruning even when adaptive=True."""
        from ccr.core.memory_pkg.memory_links import _prune_frontier
        two = [(0.9, "C001"), (0.1, "C002")]
        assert _prune_frontier(two, adaptive=True) == two

    def test_low_scores_pruned(self):
        """High/low score mix → very low scores removed."""
        from ccr.core.memory_pkg.memory_links import _prune_frontier
        candidates = [(0.9, "C001"), (0.85, "C002"), (0.88, "C003"), (0.05, "C004"), (0.02, "C005")]
        result = _prune_frontier(candidates, adaptive=True)
        result_ids = [cid for _, cid in result]
        assert "C004" not in result_ids
        assert "C005" not in result_ids

    def test_at_least_two_kept(self):
        """Even with all-low scores, at least 2 are kept."""
        from ccr.core.memory_pkg.memory_links import _prune_frontier
        candidates = [(0.1, "C001"), (0.09, "C002"), (0.08, "C003")]
        result = _prune_frontier(candidates, adaptive=True)
        assert len(result) >= 2

    def test_adaptive_param_forwarded(self, memory):
        """get_linked_commits(adaptive=False) returns same or more results than adaptive=True."""
        memory.commit("A", "Did A", "Why A", ["common.py"], "N", admission_threshold=1.0)
        memory.commit("B", "Did B", "Why B", ["common.py"], "N", admission_threshold=1.0)
        memory.commit("C", "Did C", "Why C", ["common.py"], "N", admission_threshold=1.0)
        r_adaptive = memory.get_linked_commits("C001", adaptive=True)
        r_exhaustive = memory.get_linked_commits("C001", adaptive=False)
        # adaptive=False should return >= results (no pruning)
        assert len(r_exhaustive) >= len(r_adaptive)


# ===========================================================================
# Test 4A — _load_all_commit_embeddings sqlite-vec path
# ===========================================================================


class TestLoadAllCommitEmbeddings:
    """Tests that _load_all_commit_embeddings tries sqlite-vec before gzip JSON."""

    def _make_stub(self, tmp_path):
        """Create a minimal LinksMixin stub with a temp .ccr root."""
        import os
        ccr_root = os.path.join(tmp_path, ".ccr")
        os.makedirs(ccr_root, exist_ok=True)
        from ccr.core.memory_pkg.memory_links import LinksMixin

        class _Stub(LinksMixin):
            def __init__(self):
                self.ccr_root = ccr_root

        return _Stub()

    def test_sqlite_vec_path_used_when_available(self, tmp_path):
        """When sqlite-vec store has IDs, _load_all_commit_embeddings returns them."""
        from unittest.mock import MagicMock, patch

        stub = self._make_stub(str(tmp_path))
        mock_store = MagicMock()
        mock_store.list_ids.return_value = ["C001", "C002"]
        mock_store.get_batch.return_value = {
            "C001": [0.1, 0.2, 0.3],
            "C002": [0.4, 0.5, 0.6],
        }
        with patch("ccr.context.vec_store.get_vec_store", return_value=mock_store):
            result = stub._load_all_commit_embeddings()

        assert "C001" in result
        assert "C002" in result
        assert len(result) == 2

    def test_falls_back_to_gzip_json_when_store_is_none(self, tmp_path):
        """When get_vec_store returns None, falls back to gzip JSON."""
        import gzip
        import json
        import os
        from unittest.mock import patch

        stub = self._make_stub(str(tmp_path))
        emb_path = stub._get_commit_embeddings_path()
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        data = {"C003": [0.7, 0.8]}
        with gzip.open(emb_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        with patch("ccr.context.vec_store.get_vec_store", return_value=None):
            result = stub._load_all_commit_embeddings()

        assert "C003" in result

    def test_falls_back_when_store_returns_empty(self, tmp_path):
        """When sqlite-vec has no IDs, falls back to gzip JSON."""
        import gzip
        import json
        import os
        from unittest.mock import MagicMock, patch

        stub = self._make_stub(str(tmp_path))
        mock_store = MagicMock()
        mock_store.list_ids.return_value = []  # no IDs

        emb_path = stub._get_commit_embeddings_path()
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        data = {"C004": [0.1, 0.2]}
        with gzip.open(emb_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        with patch("ccr.context.vec_store.get_vec_store", return_value=mock_store):
            result = stub._load_all_commit_embeddings()

        assert "C004" in result
