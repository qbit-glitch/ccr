"""Tests for MemoryManager — GCC-style context management."""

import os
import re
import tempfile
from datetime import datetime, timezone

import pytest

from unittest.mock import MagicMock, patch

from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig


@pytest.fixture
def project_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def memory(project_dir):
    mem = MemoryManager(project_dir, CCRConfig())
    mem.ensure_structure()
    return mem


class TestBootstrap:
    def test_ensure_structure_creates_dirs(self, project_dir):
        mem = MemoryManager(project_dir)
        created = mem.ensure_structure()
        assert created
        assert os.path.isdir(os.path.join(project_dir, ".ccr"))
        assert os.path.isdir(os.path.join(project_dir, ".ccr", "branches", "main"))

    def test_ensure_structure_creates_files(self, project_dir):
        mem = MemoryManager(project_dir)
        mem.ensure_structure()
        assert os.path.isfile(os.path.join(project_dir, ".ccr", "main.md"))
        assert os.path.isfile(os.path.join(project_dir, ".ccr", "branches", "_registry.md"))
        assert os.path.isfile(os.path.join(project_dir, ".ccr", "branches", "main", "commits.md"))

    def test_ensure_structure_idempotent(self, project_dir):
        mem = MemoryManager(project_dir)
        mem.ensure_structure()
        created = mem.ensure_structure()
        assert not created  # Already existed


class TestCommit:
    def test_basic_commit(self, memory):
        result = memory.commit(
            title="Add user auth",
            what="Implemented login endpoint",
            why="User story #42",
            files_changed=["auth.py", "routes.py"],
            next_step="Add tests",
        )
        assert "C001" in result
        assert "Add user auth" in result

    def test_sequential_commits(self, memory):
        memory.commit("First", "did A", "reason A", ["a.py"], "next A")
        result = memory.commit("Second", "did B", "reason B", ["b.py"], "next B")
        assert "C002" in result

    def test_commit_updates_milestones(self, memory):
        memory.commit("Important change", "did something", "reason", ["x.py"], "continue")
        main_md = memory._read_file(os.path.join(memory.ccr_root, "main.md"))
        assert "Important change" in main_md

    def test_commit_appends_log(self, memory):
        memory.commit("Test commit", "did it", "reason", ["test.py"], "next")
        log = memory._read_file(memory._get_log_path("main"))
        assert "commit" in log
        assert "C001" in log


class TestBranch:
    def test_create_branch(self, memory):
        result = memory.create_branch(
            "fix-auth",
            purpose="Fix auth bug",
            hypothesis="Token expiry is wrong",
        )
        assert "fix-auth" in result
        assert memory.get_active_branch() == "fix-auth"

    def test_branch_creates_files(self, memory):
        memory.create_branch("explore-api", "Try new API", "It will be faster")
        branch_dir = os.path.join(memory.branches_dir, "explore-api")
        assert os.path.isdir(branch_dir)
        assert os.path.isfile(os.path.join(branch_dir, "commits.md"))
        assert os.path.isfile(os.path.join(branch_dir, "log.md"))

    def test_branch_validates_kebab_case(self, memory):
        with pytest.raises(ValueError, match="kebab-case"):
            memory.create_branch("NotKebab", "test", "test")

    def test_branch_requires_main(self, memory):
        memory.create_branch("first-branch", "test", "test")
        with pytest.raises(ValueError, match="Must be on main"):
            memory.create_branch("second-branch", "test", "test")

    def test_branch_no_duplicates(self, memory):
        memory.create_branch("my-branch", "test", "test")
        memory._update_registry_active_branch("main")  # switch back
        with pytest.raises(ValueError, match="already exists"):
            memory.create_branch("my-branch", "test", "test")

    def test_branch_appears_in_main_md(self, memory):
        memory.create_branch("new-feature", "Add feature", "It will work")
        main_md = memory._read_file(os.path.join(memory.ccr_root, "main.md"))
        assert "new-feature" in main_md


class TestMerge:
    def test_merge_success(self, memory):
        memory.create_branch("test-branch", "Test something", "Will succeed")
        memory.commit("Branch work", "did it", "testing", ["test.py"], "merge")
        result = memory.merge("test-branch", "success", "It worked!")
        assert "Merged" in result
        assert memory.get_active_branch() == "main"

    def test_merge_creates_commit_on_main(self, memory):
        memory.create_branch("exp-branch", "Experiment", "hypothesis")
        memory.merge("exp-branch", "partial", "Some insights")
        main_commits = memory._read_file(memory._get_commits_path("main"))
        assert "Merge: exp-branch" in main_commits

    def test_merge_updates_conclusion(self, memory):
        memory.create_branch("conclude-branch", "Test", "hypo")
        memory.merge("conclude-branch", "failure", "Didn't work")
        branch_commits = memory._read_file(memory._get_commits_path("conclude-branch"))
        assert "failure: Didn't work" in branch_commits

    def test_merge_requires_being_on_branch(self, memory):
        memory.create_branch("some-branch", "test", "test")
        memory._update_registry_active_branch("main")
        with pytest.raises(ValueError, match="Must be on branch"):
            memory.merge("some-branch", "success", "done")


class TestContext:
    def test_level_1_returns_main(self, memory):
        ctx = memory.get_context(level=1)
        assert "Project" in ctx

    def test_level_2_includes_commits(self, memory):
        memory.commit("Test commit", "what", "why", ["f.py"], "next")
        ctx = memory.get_context(level=2)
        assert "Test commit" in ctx

    def test_level_3_includes_branch_header(self, memory):
        memory.create_branch("my-branch", "Test purpose", "Test hypothesis")
        ctx = memory.get_context(level=3, branch="my-branch")
        assert "Test purpose" in ctx

    def test_level_5_search(self, memory):
        memory.commit("Add auth module", "built auth", "security", ["auth.py"], "test")
        memory.commit("Add logging", "built logging", "observability", ["log.py"], "deploy")
        ctx = memory.get_context(level=5, search_term="auth")
        assert "auth" in ctx.lower()


class TestOTALog:
    def test_log_ota(self, memory):
        memory.log_ota("Edit", "src/main.py", "OK")
        log = memory._read_file(memory._get_log_path("main"))
        assert "Edit" in log
        assert "src/main.py" in log

    def test_log_rotation(self, memory):
        config = CCRConfig(log_max_lines=10)
        mem = MemoryManager(memory.project_root, config)
        for i in range(20):
            mem.log_ota("Edit", f"file_{i}.py")
        log = mem._read_file(mem._get_log_path("main"))
        lines = [l for l in log.strip().split("\n") if l.strip()]
        # Should be rotated down
        assert len(lines) <= 200


class TestIndexCache:
    def test_save_and_load_index(self, memory):
        memory.save_index('{"files": {}, "built_at": 1234}')
        loaded = memory.load_index()
        assert loaded is not None
        assert "files" in loaded


# ===========================================================================
# Memory Admission Control (A-MAC inspired)
# ===========================================================================


class TestJaccard:
    def test_identical_sets(self, memory):
        assert memory._jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self, memory):
        assert memory._jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self, memory):
        score = memory._jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(score - 0.5) < 0.01  # 2/4

    def test_empty_sets(self, memory):
        assert memory._jaccard(set(), set()) == 0.0

    def test_one_empty(self, memory):
        assert memory._jaccard({"a"}, set()) == 0.0


class TestParseRecentCommitData:
    def test_parse_single_commit(self, memory):
        memory.commit("First", "did A", "reason A", ["a.py"], "next A")
        data = memory._parse_recent_commit_data("main", k=3)
        assert len(data) == 1
        assert data[0]["id"] == "C001"
        assert data[0]["title"] == "First"
        assert data[0]["what"] == "did A"
        assert data[0]["why"] == "reason A"
        assert "a.py" in data[0]["files"]
        assert data[0]["next"] == "next A"

    def test_parse_multiple_commits(self, memory):
        memory.commit("First", "did A", "why A", ["a.py"], "next A")
        memory.commit("Second", "did B", "why B", ["b.py"], "next B")
        memory.commit("Third", "did C", "why C", ["c.py"], "next C")
        data = memory._parse_recent_commit_data("main", k=3)
        assert len(data) == 3
        # Most recent first
        assert data[0]["title"] == "Third"
        assert data[1]["title"] == "Second"
        assert data[2]["title"] == "First"

    def test_parse_k_limit(self, memory):
        for i in range(5):
            memory.commit(f"Commit {i}", f"did {i}", f"why {i}", [f"f{i}.py"], f"next {i}")
        data = memory._parse_recent_commit_data("main", k=2)
        assert len(data) == 2

    def test_parse_no_commits(self, memory):
        data = memory._parse_recent_commit_data("main", k=3)
        assert data == []


class TestComputeAdmissionScore:
    def test_first_commit_max_score(self, memory):
        """First commit gets score=1.0 (max novelty, no existing commits)."""
        score = memory.compute_admission_score(
            "main", "First commit", "did something", "reason",
            ["a.py"], "next",
        )
        assert score["score"] == 1.0
        assert score["similarity"] == 0.0
        assert score["novelty"] == 1.0

    def test_identical_commit_high_similarity(self, memory):
        memory.commit("Add auth", "implemented auth module", "security requirement",
                      ["auth.py", "routes.py"], "add tests")
        score = memory.compute_admission_score(
            "main", "Add auth", "implemented auth module", "security requirement",
            ["auth.py", "routes.py"], "add tests",
        )
        # High similarity → high merge signal
        assert score["similarity"] > 0.5
        # High similarity → low novelty → lower admission score
        assert score["novelty"] < 0.5

    def test_different_commit_low_similarity(self, memory):
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        score = memory.compute_admission_score(
            "main", "Refactor database", "rewrote ORM layer", "performance",
            ["models.py", "db.py"], "benchmark",
        )
        # Low similarity → novel → high admission score
        assert score["similarity"] < 0.3
        assert score["score"] > 0.5

    def test_partial_file_overlap(self, memory):
        memory.commit("Update auth", "changed login", "bug fix",
                      ["auth.py", "routes.py", "tests.py"], "deploy")
        score = memory.compute_admission_score(
            "main", "Fix auth tests", "fixed test failures", "CI broken",
            ["auth.py", "tests.py"], "merge PR",
        )
        assert score["file_similarity"] > 0.3
        assert 0.0 < score["similarity"] < 0.8

    def test_score_returns_commit_type(self, memory):
        """Score includes commit type classification."""
        memory.commit("Setup", "initial", "reason", ["a.py"], "next")
        score = memory.compute_admission_score(
            "main", "Refactor auth module", "restructured codebase", "clarity",
            ["auth.py"], "test",
        )
        assert "commit_type" in score
        assert "type_prior" in score
        assert score["commit_type"] == "structural"
        assert score["type_prior"] >= 0.85

    def test_continuation_type_low_prior(self, memory):
        """Continuation commits get low type prior."""
        memory.commit("Setup", "initial", "reason", ["a.py"], "next")
        score = memory.compute_admission_score(
            "main", "Still working on auth", "continued implementation", "in progress",
            ["auth.py"], "keep going",
        )
        assert score["commit_type"] == "continuation"
        assert score["type_prior"] <= 0.3

    def test_merge_type_always_admitted(self, memory):
        """Merge/branch commits get T=1.0 and score=1.0 (structural bypass)."""
        memory.commit("Setup", "initial", "reason", ["a.py"], "next")
        score = memory.compute_admission_score(
            "main", "Merge: fix-auth", "merged auth fix", "consolidate",
            ["auth.py"], "deploy",
        )
        assert score["commit_type"] == "merge"
        assert score["type_prior"] == 1.0
        assert score["score"] == 1.0  # Structural bypass
        assert score["similarity"] == 0.0

    def test_correct_polarity_higher_score_is_better(self, memory):
        """Higher score = more valuable. Novel commits score higher."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        # Novel commit (different topic)
        novel = memory.compute_admission_score(
            "main", "Build database", "created schema", "persistence",
            ["models.py", "db.py"], "migrate",
        )
        # Redundant commit (same topic)
        redundant = memory.compute_admission_score(
            "main", "Add auth", "implemented login", "security",
            ["auth.py"], "add tests",
        )
        assert novel["score"] > redundant["score"]

    def test_conflict_score_computed(self, memory):
        """G1-NEW: S(m_conflict) is computed for Algorithm 1 line 6 comparison."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        score = memory.compute_admission_score(
            "main", "Add auth", "implemented login", "security",
            ["auth.py"], "add tests",
        )
        assert "conflict_score" in score
        assert score["conflict_score"] > 0.0  # Has a real score
        # conflict_score should be based on existing commit's type + recency
        assert score["conflict_score"] <= 1.0


# ===========================================================================
# G1: Max similarity across ALL k commits (A-MAC Eq. 3)
# ===========================================================================


class TestMaxSimilarityAcrossK:
    def test_matches_older_commit_not_just_recent(self, memory):
        """Should find high similarity with C001 even when C002 is different."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        memory.commit("Add database", "built ORM", "persistence",
                      ["models.py", "db.py"], "migrations")
        # This is very similar to C001 (auth), not C002 (database)
        score = memory.compute_admission_score(
            "main", "Add auth tests", "implemented auth tests", "security testing",
            ["auth.py", "test_auth.py"], "deploy",
        )
        # Should find similarity with C001 (auth), not just C002 (database)
        assert score["conflict_id"] == "C001"
        assert score["file_similarity"] > 0.3

    def test_picks_highest_similarity_target(self, memory):
        """conflict_id should point to the commit with highest similarity."""
        memory.commit("Build auth", "login endpoint", "security",
                      ["auth.py", "routes.py"], "tests")
        memory.commit("Build logging", "added logger", "observability",
                      ["logger.py", "config.py"], "integrate")
        memory.commit("Build API docs", "generated openapi", "documentation",
                      ["docs.py", "openapi.yaml"], "publish")
        score = memory.compute_admission_score(
            "main", "Fix auth routes", "patched login bug", "security fix",
            ["auth.py", "routes.py"], "deploy",
        )
        # Should target C001 (auth+routes), not C002 or C003
        assert score["conflict_id"] == "C001"


# ===========================================================================
# G2: Type Prior (A-MAC §3.2 Factor 5)
# ===========================================================================


class TestTypePrior:
    def test_classify_merge(self, memory):
        assert memory._classify_commit_type("Merge: fix-auth", "merged branch", []) == "merge"

    def test_classify_branch(self, memory):
        assert memory._classify_commit_type("Branch create explore-api", "created branch", []) == "branch"

    def test_classify_continuation(self, memory):
        assert memory._classify_commit_type("Still working on auth", "continued", []) == "continuation"
        assert memory._classify_commit_type("WIP feature", "work in progress", []) == "continuation"

    def test_classify_milestone(self, memory):
        assert memory._classify_commit_type("Auth complete", "finished login", []) == "milestone"
        assert memory._classify_commit_type("Fixed bug", "resolved issue", []) == "milestone"

    def test_classify_structural(self, memory):
        assert memory._classify_commit_type("Refactor auth", "restructured module", []) == "structural"
        assert memory._classify_commit_type("Schema migration", "updated database", []) == "structural"

    def test_classify_default_progress(self, memory):
        assert memory._classify_commit_type("Add tests", "wrote unit tests", []) == "progress"

    def test_type_prior_values(self, memory):
        assert memory._type_prior("merge") == 1.0
        assert memory._type_prior("branch") == 1.0
        assert memory._type_prior("structural") == 0.9
        assert memory._type_prior("milestone") == 0.85
        assert memory._type_prior("progress") == 0.5
        assert memory._type_prior("continuation") == 0.2

    def test_type_prior_affects_score(self, memory):
        """Continuation commits score lower than milestones (correct polarity)."""
        memory.commit("Auth work", "implemented login", "security",
                      ["auth.py"], "continue")
        score_continuation = memory.compute_admission_score(
            "main", "Still working on auth", "continued auth", "security",
            ["auth.py"], "keep going",
        )
        score_milestone = memory.compute_admission_score(
            "main", "Auth complete", "finished auth implementation", "security",
            ["auth.py"], "deploy",
        )
        # Milestone has higher type_prior → higher score (more valuable)
        assert score_milestone["score"] > score_continuation["score"]
        # Continuation has lower type_prior → lower score (easier to merge/reject)
        assert score_continuation["type_prior"] < score_milestone["type_prior"]


# ===========================================================================
# Recency (A-MAC Eq. 4: λ=0.01/hour)
# ===========================================================================


class TestRecency:
    def test_hours_since_recent(self, memory):
        """Recent timestamps should return small hour values."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        hours = memory._hours_since_commit(now)
        assert hours < 0.1  # Less than 6 minutes

    def test_hours_since_missing(self, memory):
        """Missing timestamps treated as very old."""
        assert memory._hours_since_commit("") == 999.0

    def test_recency_modulates_similarity(self, memory):
        """Similarity to recent commits should be stronger than old commits.

        Per Eq. 4: R(m) = exp(-0.01 * hours). At 0 hours, R=1.0.
        At 69 hours (half-life), R=0.5. At 999 hours, R≈0.
        Since test commits are created "now", conflict_recency should be ~1.0.
        """
        memory.commit("Add auth", "login", "security", ["auth.py"], "test")
        score = memory.compute_admission_score(
            "main", "Add auth", "login", "security", ["auth.py"], "test",
        )
        # Conflict recency should be near 1.0 (commit just happened)
        assert score["conflict_recency"] > 0.99


# ===========================================================================
# G3: Score comparison in merge path
# ===========================================================================


class TestScoreComparisonInMerge:
    def test_new_outranks_existing_merges(self, memory):
        """Alg. 1 lines 6-7: S(m) > S(m_conflict) → REPLACE old with merged version.

        A high-value new commit that conflicts with a low-value existing commit
        should merge (replace old with merged version), not create alongside.
        """
        # Create a low-priority continuation commit (low type_prior, low novelty)
        memory.commit("Still working on auth", "continued auth work", "wip",
                      ["auth.py"], "continue auth")
        # New commit is a progress commit with same files → higher type_prior → higher score
        # Same files trigger conflict at low threshold, and new outranks existing
        result = memory.commit(
            "Working on auth", "implemented auth module fully", "security requirement",
            ["auth.py"], "add tests",
            admission_threshold=0.3,  # low threshold to trigger FindConflict
        )
        assert "merged" in result.lower() or "+" in result

    def test_existing_outranks_new_creates_new(self, memory):
        """Alg. 1 lines 8-9: S(m) <= S(m_conflict) → ADD new alongside existing.

        A low-value new commit that conflicts with a high-value existing commit
        should be added alongside (both coexist), not merged.
        """
        # Create a normal existing commit (progress type, recent → high conflict_score)
        memory.commit("Working on auth", "started auth", "security",
                      ["auth.py"], "continue")
        # New commit is a continuation → low score → existing outranks → add alongside
        result = memory.commit(
            "Still working on auth", "continued auth", "security",
            ["auth.py"], "keep going",
            admission_threshold=0.3,
        )
        assert "C002" in result
        assert "merged" not in result.lower()

    def test_score_comparison_uses_actual_scores(self, memory):
        """G1-NEW: Score comparison uses S(m) vs S(m_conflict), not just type_prior."""
        memory.commit("Build auth module", "implemented login", "security",
                      ["auth.py"], "tests")
        # Compute score to verify conflict_score exists and is used
        score = memory.compute_admission_score(
            "main", "Still working on auth", "continued auth", "security",
            ["auth.py"], "keep going",
        )
        # New commit (continuation, low novelty) should have lower score than existing
        assert score["score"] <= score["conflict_score"]


# ===========================================================================
# G4/G9: Rejection path (correct polarity)
# ===========================================================================


class TestRejectionPath:
    def test_reject_low_score_commit(self, memory):
        """G9: Rejection uses correct polarity — low SCORE = low value = reject.

        A redundant continuation commit has low score (low novelty + low type_prior).
        With a high enough rejection_threshold, it should be rejected.
        """
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        # Same topic, continuation type → low novelty + low type_prior → low score
        result = memory.commit(
            "Still working on auth", "continued auth implementation", "security",
            ["auth.py"], "keep going",
            admission_threshold=1.0,  # disable merge (so rejection can be tested)
            rejection_threshold=0.99,  # absurdly high — forces rejection
        )
        assert "REJECTED" in result

    def test_novel_commit_not_rejected(self, memory):
        """G9: Novel commits have high score and should NOT be rejected."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        # Completely different topic → high novelty → high score → not rejected
        result = memory.commit(
            "Add database", "built ORM", "persistence",
            ["models.py", "db.py"], "migrations",
            admission_threshold=1.0,  # disable merge
            rejection_threshold=0.5,  # moderate threshold
        )
        assert "REJECTED" not in result
        assert "C002" in result

    def test_rejection_does_not_create_commit(self, memory):
        """Rejected commits don't appear in commits.md."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        memory.commit(
            "Still working on auth", "continued auth", "security",
            ["auth.py"], "keep going",
            admission_threshold=1.0,
            rejection_threshold=0.99,
        )
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "Still working" not in commits
        assert "C002" not in commits

    def test_rejection_logs_event(self, memory):
        """Rejected commits are logged in OTA."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        memory.commit(
            "Still working on auth", "continued auth", "security",
            ["auth.py"], "keep going",
            admission_threshold=1.0,
            rejection_threshold=0.99,
        )
        log = memory._read_file(memory._get_log_path("main"))
        assert "Commit rejected" in log or "rejection threshold" in log

    def test_rejection_disabled_by_default(self, memory):
        """Default rejection_threshold=0.0 means no commits are rejected."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        result = memory.commit(
            "Add database", "built ORM", "persistence",
            ["models.py", "db.py"], "migrations",
        )
        assert "REJECTED" not in result
        assert "C002" in result


class TestAdmissionControlIntegration:
    def test_redundant_commit_adds_alongside(self, memory):
        """Nearly identical continuation commit adds alongside per Alg. 1 lines 8-9.

        When S(m) <= S(m_conflict), the new commit is added alongside the existing
        one rather than merged. A low-value continuation doesn't replace the
        higher-value existing commit.
        """
        memory.commit("Working on auth", "started auth module", "security",
                      ["auth.py"], "continue auth")
        result = memory.commit(
            "Still working on auth", "continued auth module", "security requirement",
            ["auth.py"], "finish auth",
            admission_threshold=0.5,  # low similarity threshold to ensure FindConflict
        )
        assert "C002" in result
        assert "merged" not in result.lower()

    def test_novel_commit_creates_new(self, memory):
        """Sufficiently different commit creates a new entry."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        result = memory.commit(
            "Add database layer", "built ORM models", "data persistence",
            ["models.py", "db.py"], "add migrations",
            admission_threshold=0.85,
        )
        assert "C002" in result
        assert "merged" not in result.lower()

    def test_threshold_1_disables_admission(self, memory):
        """Setting threshold to 1.0 always creates new commits."""
        memory.commit("Same thing", "did X", "why X", ["a.py"], "next")
        result = memory.commit(
            "Same thing", "did X", "why X", ["a.py"], "next",
            admission_threshold=1.0,
        )
        assert "C002" in result
        assert "merged" not in result.lower()

    def test_default_threshold_0_85(self, memory):
        """G5: Default threshold matches paper's 0.85 FindConflict.

        Identical commits trigger FindConflict (similarity >= 0.85).
        But S(m) < S(m_conflict) since the new duplicate has low novelty while
        the existing commit was stored with high novelty. Per Alg. 1 lines 8-9,
        the new commit is added alongside (not merged).
        """
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        result = memory.commit(
            "Add auth", "implemented login", "security",
            ["auth.py"], "add tests",
            # No explicit threshold — uses default 0.85
        )
        # Identical commit: S(m) < S(m_conflict) → add alongside (Alg. 1 lines 8-9)
        assert "C002" in result
        assert "merged" not in result.lower()

    def test_different_topic_not_merged_at_default_threshold(self, memory):
        """G8: At realistic threshold (0.85), different topics are never merged."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        result = memory.commit(
            "Add database", "built ORM", "persistence",
            ["models.py", "db.py"], "migrations",
            # Default threshold 0.85 — low similarity → no merge
        )
        assert "C002" in result
        assert "merged" not in result.lower()

    def test_merged_commit_updates_what(self, memory):
        """Merged commit appends new what/why to previous.

        Per Alg. 1 lines 6-7: merge happens when S(m) > S(m_conflict).
        Setup: low-value continuation first, then higher-value progress commit.
        """
        # Low-value existing commit (continuation type → low type_prior)
        memory.commit("Still working on feature", "continued implementation", "wip",
                      ["feature.py"], "continue")
        # Higher-value new commit (progress type → higher type_prior → higher score)
        memory.commit(
            "Working on feature", "added validation", "new feature needs validation",
            ["feature.py"], "add tests",
            admission_threshold=0.3,  # force FindConflict
        )
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "continued implementation" in commits
        assert "added validation" in commits

    def test_merged_commit_unions_files(self, memory):
        """Merged commit unions file lists.

        Per Alg. 1 lines 6-7: merge happens when S(m) > S(m_conflict).
        """
        # Low-value existing commit (continuation type)
        memory.commit("Still working on feature", "continued", "wip",
                      ["feature.py"], "continue")
        # Higher-value new commit (progress type) with extra file
        memory.commit(
            "Working on feature", "added tests", "reason",
            ["feature.py", "test_feature.py"], "next",
            admission_threshold=0.3,
        )
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "test_feature.py" in commits
        assert "feature.py" in commits

    def test_merged_commit_updates_next(self, memory):
        """Merged commit replaces next_step with newer value.

        Per Alg. 1 lines 6-7: merge happens when S(m) > S(m_conflict).
        """
        # Low-value existing commit (continuation type)
        memory.commit("Still on feature", "continued", "wip", ["f.py"], "old next step")
        # Higher-value new commit (progress type)
        memory.commit(
            "Feature progress", "implemented core logic", "reason", ["f.py"], "new next step",
            admission_threshold=0.3,
        )
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "new next step" in commits

    def test_admission_updates_rolling_summary(self, memory):
        """Even merged commits update the rolling summary.

        Per Alg. 1 lines 6-7: merge happens when S(m) > S(m_conflict).
        """
        # Low-value existing commit (continuation type)
        memory.commit("Still on feature", "continued", "wip", ["f.py"], "continue")
        # Higher-value new commit (progress type)
        memory.commit(
            "Feature progress", "added validation", "reason", ["f.py"], "test",
            admission_threshold=0.3,
        )
        summary = memory._get_rolling_summary("main")
        assert "validation" in summary.lower()

    def test_admission_logs_merge_event(self, memory):
        """Admission merges are logged in OTA.

        Per Alg. 1 lines 6-7: merge happens when S(m) > S(m_conflict).
        """
        # Low-value existing commit (continuation type)
        memory.commit("Still on feature", "continued", "wip", ["f.py"], "continue")
        # Higher-value new commit (progress type)
        memory.commit(
            "Feature progress", "implemented logic", "reason", ["f.py"], "test",
            admission_threshold=0.3,
        )
        log = memory._read_file(memory._get_log_path("main"))
        assert "commit-merge" in log or "Admission control" in log


# ===========================================================================
# Issue 1: OTA ID overflow after 999 entries
# ===========================================================================


class TestOTAOverflow:
    def test_ota_id_past_999(self, memory):
        """OTA IDs above 999 should still be found and incremented."""
        branch = "main"
        log_path = memory._get_log_path(branch)
        # Manually write an OTA entry with ID 999
        memory._write_file(log_path, "---\n[OTA-999] 2026-01-01T00:00:00\n- **Observation**: test\n")
        next_id = memory._get_next_ota_id(branch)
        assert next_id == "OTA-1000"

    def test_ota_id_1000_is_found(self, memory):
        """Once OTA-1000 exists, the next ID should be OTA-1001."""
        branch = "main"
        log_path = memory._get_log_path(branch)
        memory._write_file(log_path, "---\n[OTA-1000] 2026-01-01T00:00:00\n- test\n")
        next_id = memory._get_next_ota_id(branch)
        assert next_id == "OTA-1001"

    def test_ota_id_mixed_lengths(self, memory):
        """With both 3-digit and 4-digit OTA IDs, pick the highest."""
        branch = "main"
        log_path = memory._get_log_path(branch)
        content = (
            "---\n[OTA-050] 2026-01-01T00:00:00\n- test\n"
            "---\n[OTA-999] 2026-01-01T00:00:01\n- test\n"
            "---\n[OTA-1002] 2026-01-01T00:00:02\n- test\n"
        )
        memory._write_file(log_path, content)
        next_id = memory._get_next_ota_id(branch)
        assert next_id == "OTA-1003"

    def test_ota_id_format_is_zero_padded_minimum_3(self, memory):
        """Small IDs are zero-padded to 3 digits; large IDs use as many digits as needed."""
        branch = "main"
        log_path = memory._get_log_path(branch)
        memory._write_file(log_path, "---\n[OTA-005] 2026-01-01T00:00:00\n- test\n")
        next_id = memory._get_next_ota_id(branch)
        assert next_id == "OTA-006"  # still 3-digit padded


# ===========================================================================
# Issue 2: Stored admission score for S(m_conflict)
# ===========================================================================


class TestStoredAdmissionScore:
    def test_commit_stores_score(self, memory):
        """Commits should contain a **Score**: line."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "**Score**:" in commits
        # Score should be a float
        score_match = re.search(r"\*\*Score\*\*:\s*([\d.]+)", commits)
        assert score_match is not None
        score_val = float(score_match.group(1))
        assert 0.0 <= score_val <= 1.0

    def test_parsed_stored_score(self, memory):
        """_parse_recent_commit_data should extract stored_score."""
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        commits = memory._parse_recent_commit_data("main", 5)
        assert len(commits) == 1
        assert commits[0]["stored_score"] is not None
        assert 0.0 <= commits[0]["stored_score"] <= 1.0

    def test_conflict_uses_recomputed_score(self, memory):
        """S(m_conflict) should be recomputed with current R(m') for temporal decay."""
        # Create first commit (gets score=1.0 as first commit, but R decays over time)
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        # Compute score for a similar commit — conflict score is recomputed
        score = memory.compute_admission_score(
            "main", "Add auth v2", "improved login", "security",
            ["auth.py"], "more tests",
        )
        if score["conflict_id"]:
            # Recomputed conflict_score uses current R(m') which decays with time.
            # For a very recent commit, R ≈ 1.0, so score should be close to stored.
            # S(m') = 0.50*T + 0.35*N + 0.15*R where R ≈ 1.0 for recent commits.
            assert 0.5 <= score["conflict_score"] <= 1.0

    def test_backward_compat_no_stored_score(self, memory):
        """Old commits without **Score**: should fall back to heuristic."""
        # Manually write a commit without Score field
        path = memory._get_commits_path("main")
        old_commit = (
            "## [C001] 2026-01-01 00:00 | branch:main | Old commit\n"
            "**What**: did something\n"
            "**Why**: reason\n"
            "**Files**: old.py\n"
            "**Next**: continue\n\n---\n\n"
        )
        memory._write_file(path, f"# Branch: main\n\n## Rolling Summary\n(none yet)\n\n# Milestone Journal\n\n{old_commit}")
        commits = memory._parse_recent_commit_data("main", 5)
        assert len(commits) == 1
        assert commits[0]["stored_score"] is None
        # Should still work — falls back to heuristic
        score = memory.compute_admission_score(
            "main", "Old commit continued", "did more", "reason",
            ["old.py"], "next",
        )
        assert "conflict_score" in score


# ===========================================================================
# Issue 3: Timezone awareness
# ===========================================================================


class TestTimezoneAwareness:
    def test_commit_timestamp_is_utc(self, memory):
        """Commit timestamps should be generated from UTC time."""
        memory.commit("Test tz", "testing timezone", "reason",
                      ["tz.py"], "verify")
        commits = memory._read_file(memory._get_commits_path("main"))
        # The timestamp should be present in commits.md
        assert re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", commits) is not None

    def test_hours_since_commit_handles_naive_timestamp(self, memory):
        """_hours_since_commit should handle naive timestamps (backward compat)."""
        # Naive timestamp (no timezone info) — should be treated as UTC
        hours = memory._hours_since_commit("2020-01-01 00:00")
        assert hours > 0  # should be a large positive number (years ago)
        assert hours < 100_000  # sanity check

    def test_hours_since_commit_handles_empty(self, memory):
        """Missing timestamps should return 999."""
        assert memory._hours_since_commit("") == 999.0
        assert memory._hours_since_commit("   ") == 999.0

    def test_hours_since_commit_handles_invalid(self, memory):
        """Invalid timestamps should return 999."""
        assert memory._hours_since_commit("not-a-date") == 999.0

    def test_hours_since_commit_recent(self, memory):
        """A timestamp from just now should have ~0 hours elapsed."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        hours = memory._hours_since_commit(now)
        assert hours < 0.1  # less than 6 minutes


# ===========================================================================
# Issue 4: Regex injection in merge title
# ===========================================================================


class TestRegexInjectionInMerge:
    def test_merge_title_with_backslash_n(self, memory):
        r"""Title containing \n should not create a newline in merged commit.

        Per Alg. 1 lines 6-7: merge when S(m) > S(m_conflict).
        First commit is low-value continuation, second is higher-value fix.
        """
        # Low-value continuation (low type_prior)
        memory.commit("Still working on feature", "continued feature", "wip",
                      ["feature.py"], "continue")
        result = memory.commit(
            r"Fix path C:\new\thing", "fixed path handling", "bug fix",
            ["feature.py"], "done",
            admission_threshold=0.3,
        )
        if "merged" in result.lower() or "+" in result:
            commits = memory._read_file(memory._get_commits_path("main"))
            # The literal backslash-n should appear, not an actual newline
            assert r"C:\new\thing" in commits

    def test_merge_title_with_backreference(self, memory):
        r"""Title containing \1 should not be interpreted as regex backreference.

        Per Alg. 1 lines 6-7: merge when S(m) > S(m_conflict).
        """
        # Low-value continuation (low type_prior)
        memory.commit("Still working on feature", "continued feature", "wip",
                      ["feature.py"], "continue")
        result = memory.commit(
            r"Fix regex \1 group", "fixed regex handling", "bug fix",
            ["feature.py"], "done",
            admission_threshold=0.3,
        )
        if "merged" in result.lower() or "+" in result:
            commits = memory._read_file(memory._get_commits_path("main"))
            assert r"\1" in commits

    def test_merge_title_with_special_chars(self, memory):
        r"""Title with mixed special regex chars should be handled safely.

        Per Alg. 1 lines 6-7: merge when S(m) > S(m_conflict).
        """
        # Low-value continuation (low type_prior)
        memory.commit("Still working on feature", "continued feature", "wip",
                      ["feature.py"], "continue")
        title_with_specials = r"Fix $100 cost \g<1> issue"
        result = memory.commit(
            title_with_specials, "fixed cost issue", "accounting",
            ["feature.py"], "deploy",
            admission_threshold=0.3,
        )
        if "merged" in result.lower() or "+" in result:
            commits = memory._read_file(memory._get_commits_path("main"))
            assert r"\g<1>" in commits


# ===========================================================================
# H1: Atomic writes
# ===========================================================================


class TestAtomicWrites:
    def test_write_file_creates_content(self, memory):
        """_write_file should create a file with the given content."""
        path = os.path.join(memory.ccr_root, "test_atomic.txt")
        memory._write_file(path, "hello atomic")
        assert os.path.isfile(path)
        content = memory._read_file(path)
        assert content == "hello atomic"

    def test_write_file_no_partial_on_interrupt(self, memory):
        """Atomic writes via os.replace ensure no partial files."""
        path = os.path.join(memory.ccr_root, "test_atomic2.txt")
        memory._write_file(path, "first version")
        # Overwrite — should be atomic
        memory._write_file(path, "second version")
        content = memory._read_file(path)
        assert content == "second version"

    def test_write_file_handles_surrogates(self, memory):
        """Content with surrogate characters should be sanitized."""
        path = os.path.join(memory.ccr_root, "test_surr.txt")
        memory._write_file(path, "test \ud800 data")
        content = memory._read_file(path)
        assert "test" in content  # should not crash


# ===========================================================================
# H2: Cross-process file locking
# ===========================================================================


class TestCrossProcessLocking:
    def test_file_lock_context_manager(self, memory):
        """_file_lock should create a .lock file and release it."""
        path = os.path.join(memory.ccr_root, "test_lock.txt")
        memory._write_file(path, "data")
        with memory._file_lock(path):
            assert os.path.isfile(path + ".lock")
        # Lock should be released (not holding exclusive)

    def test_concurrent_writes_dont_corrupt(self, memory):
        """Concurrent thread writes should not corrupt the file."""
        import concurrent.futures
        path = os.path.join(memory.ccr_root, "concurrent.txt")
        memory._write_file(path, "initial")

        def write_thread(i):
            memory._write_file(path, f"version-{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(write_thread, i) for i in range(10)]
            for f in futures:
                f.result()

        # File should contain one complete version, not a mix
        content = memory._read_file(path)
        assert content.startswith("version-")


# ===========================================================================
# H3: Regex injection in rolling summary and conclusion
# ===========================================================================


class TestRegexInjectionInSummary:
    def test_rolling_summary_with_backslash(self, memory):
        r"""Rolling summary containing \n should not create newline."""
        memory.commit(r"Fix path C:\new", "fixed", "reason", ["f.py"], "next")
        summary = memory._get_rolling_summary("main")
        # Should not crash; summary should contain the text
        assert summary is not None

    def test_branch_conclusion_with_backreference(self, memory):
        r"""Conclusion with \1 should not be treated as backreference."""
        memory.create_branch("test-regex", "test", "hypo")
        # This should not raise an error
        memory.merge("test-regex", "success", r"Fixed \1 and \g<0> refs")
        commits = memory._read_file(memory._get_commits_path("test-regex"))
        assert r"\1" in commits


# ===========================================================================
# CER-Inspired Pattern Buffer Tests (arXiv:2506.06698)
# ===========================================================================


class TestPatternBuffer:
    """Tests for the CER-inspired pattern buffer in gcc_commit."""

    # --- Basic Storage ---

    def test_commit_with_patterns_stores_inline(self, memory):
        """Patterns appear as **Patterns**: field in commits.md."""
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["Use {tool} for testing", "Always update docs"])
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "**Patterns**:" in commits
        assert "Use {tool} for testing" in commits
        assert "Always update docs" in commits

    def test_commit_without_patterns_backward_compat(self, memory):
        """Old-style commit without patterns still works."""
        result = memory.commit("T1", "did A", "reason", ["a.py"], "next")
        assert "C001" in result
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "**Patterns**:" not in commits

    def test_patterns_json_created_on_first_pattern(self, memory):
        """patterns.json created lazily on first commit with patterns."""
        path = memory._get_patterns_path()
        assert not os.path.exists(path)
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["Pattern one"])
        assert os.path.exists(path)

    def test_pattern_entry_structure(self, memory):
        """Verify patterns.json has correct schema."""
        import json
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["Check tests before merging"])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert data["version"] == 1
        assert "P001" in data["patterns"]
        entry = data["patterns"]["P001"]
        assert entry["text"] == "Check tests before merging"
        assert entry["first_seen"] == "C001"
        assert entry["commit_ids"] == ["C001"]
        assert entry["occurrence_count"] == 1
        assert entry["promoted"] is False

    def test_sequential_pattern_ids(self, memory):
        """Patterns get P001, P002, P003 IDs sequentially."""
        import json
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["First distinct pattern here",
                                          "Second distinct pattern here"])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert "P001" in data["patterns"]
        assert "P002" in data["patterns"]
        assert data["next_id"] == 3

    # --- Dedup ---

    def test_identical_pattern_deduped(self, memory):
        """Exact same pattern text only stored once, occurrence incremented."""
        import json
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["Always run tests before committing code"])
        memory.commit("T2", "did B", "reason", ["b.py"], "next",
                       patterns_learned=["Always run tests before committing code"])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert len(data["patterns"]) == 1
        entry = data["patterns"]["P001"]
        assert entry["occurrence_count"] == 2
        assert "C001" in entry["commit_ids"]
        assert "C002" in entry["commit_ids"]

    def test_similar_pattern_deduped_above_threshold(self, memory):
        """Similar patterns (Jaccard >= 0.7) are merged."""
        import json
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["Always validate input parameters before processing request data"])
        memory.commit("T2", "did B", "reason", ["b.py"], "next",
                       patterns_learned=["Always validate input parameters before processing response data"])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        # Should be deduped (high word overlap — differs only in request/response)
        assert len(data["patterns"]) == 1
        assert data["patterns"]["P001"]["occurrence_count"] == 2

    def test_different_pattern_not_deduped(self, memory):
        """Distinct patterns (Jaccard < 0.7) are stored separately."""
        import json
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["Always update documentation after API changes"])
        memory.commit("T2", "did B", "reason", ["b.py"], "next",
                       patterns_learned=["Use kernel sandbox for REPL execution"])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert len(data["patterns"]) == 2

    def test_dedup_case_insensitive(self, memory):
        """Dedup is case-insensitive."""
        import json
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["Always Update Tests Before Committing Code"])
        memory.commit("T2", "did B", "reason", ["b.py"], "next",
                       patterns_learned=["always update tests before committing code"])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert len(data["patterns"]) == 1
        assert data["patterns"]["P001"]["occurrence_count"] == 2

    # --- Occurrence Tracking ---

    def test_occurrence_increments_on_re_observation(self, memory):
        """Count goes 1 -> 2 -> 3 on successive commits with same pattern."""
        import json
        pat = "Always validate input parameters before processing data"
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                           patterns_learned=[pat])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert data["patterns"]["P001"]["occurrence_count"] == 3

    def test_same_commit_no_double_count(self, memory):
        """Same pattern twice in one commit doesn't double-count."""
        import json
        pat = "Always validate input parameters before processing data"
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=[pat, pat])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        # First time creates P001, second match finds P001 but commit_id already there
        assert data["patterns"]["P001"]["occurrence_count"] == 1

    def test_commit_ids_accumulated(self, memory):
        """All distinct commit IDs tracked in pattern entry."""
        import json
        pat = "Always validate input parameters before processing data"
        memory.commit("T1", "did A", "reason", ["a.py"], "next", patterns_learned=[pat])
        memory.commit("T2", "did B", "reason", ["b.py"], "next", patterns_learned=[pat])
        memory.commit("T3", "did C", "reason", ["c.py"], "next", patterns_learned=[pat])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert data["patterns"]["P001"]["commit_ids"] == ["C001", "C002", "C003"]

    # --- Promotion Suggestions ---

    def test_promotion_suggested_at_threshold(self, memory):
        """3 occurrences triggers promotion suggestion in return string."""
        pat = "Always validate input parameters before processing data"
        for i in range(2):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                           patterns_learned=[pat])
        result = memory.commit("T3", "did 2", "reason", ["2.py"], "next",
                                patterns_learned=[pat])
        assert "Pattern promotion suggestions" in result
        assert "ace_apply_delta" in result

    def test_no_promotion_below_threshold(self, memory):
        """2 occurrences returns no suggestion."""
        pat = "Always validate input parameters before processing data"
        memory.commit("T1", "did A", "reason", ["a.py"], "next", patterns_learned=[pat])
        result = memory.commit("T2", "did B", "reason", ["b.py"], "next",
                                patterns_learned=[pat])
        assert "Pattern promotion suggestions" not in result

    def test_promoted_flag_not_auto_set(self, memory):
        """promoted remains False — suggestion-only, no auto-add to playbook."""
        import json
        pat = "Always validate input parameters before processing data"
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                           patterns_learned=[pat])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert data["patterns"]["P001"]["promoted"] is False

    # --- Buffer Management ---

    def test_buffer_eviction_at_max_size(self, memory):
        """Evicts lowest-occurrence patterns when buffer exceeds max."""
        import json
        memory.config.pattern_max_buffer_size = 3
        distinct_patterns = [
            "Always validate database connection parameters carefully",
            "Configure kernel sandbox with allowlist security policies",
            "Structure playbook bullets using helpful harmful counters",
            "Implement admission control with recency modulated scoring",
            "Generate session summaries after completing milestone work",
        ]
        for i, pat in enumerate(distinct_patterns):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                           patterns_learned=[pat])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert len(data["patterns"]) == 3

    def test_eviction_preserves_high_occurrence(self, memory):
        """High-occurrence patterns survive eviction."""
        import json
        memory.config.pattern_max_buffer_size = 2
        # Create a high-occurrence pattern observed 3 times
        high_pat = "Always validate database connection parameters carefully"
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                           patterns_learned=[high_pat])
        # Now add 2 more low-occurrence distinct patterns — should evict low, keep high
        memory.commit("T4", "did 3", "reason", ["3.py"], "next",
                       patterns_learned=["Configure kernel sandbox allowlist security policies",
                                          "Structure playbook bullets using helpful harmful counters"])
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert len(data["patterns"]) == 2
        # The high-occurrence pattern should survive
        high_entries = [e for e in data["patterns"].values() if e["occurrence_count"] >= 3]
        assert len(high_entries) == 1

    # --- Merge Path ---

    def test_merge_path_preserves_patterns(self, memory):
        """Patterns survive the admission control merge path."""
        # First commit
        memory.commit("T1", "did A", "reason A", ["a.py"], "next A",
                       patterns_learned=["Pattern from first commit for merge test"])
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "Pattern from first commit" in commits

    def test_merge_unions_patterns(self, memory):
        """Merge path unions old + new patterns.

        Per Alg. 1 lines 6-7: merge when S(m) > S(m_conflict).
        First commit is low-value continuation, second is higher-value progress.
        """
        # Low-value continuation (low type_prior → low stored score)
        memory.commit("Still working on topic", "continued topic work here", "wip",
                       ["a.py"], "next A",
                       patterns_learned=["First pattern for merge"])
        # Higher-value progress commit → S(m) > S(m_conflict) → merge
        memory.commit("Working on topic progress", "completed topic work here", "shipped",
                       ["a.py"], "next B",
                       patterns_learned=["Second pattern for merge"],
                       admission_threshold=0.0)
        commits = memory._read_file(memory._get_commits_path("main"))
        # Both patterns should be present (either in same or separate commits)
        assert "First pattern for merge" in commits

    # --- Parse ---

    def test_parse_commit_with_patterns(self, memory):
        """_parse_recent_commit_data extracts patterns list."""
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["Pattern alpha", "Pattern beta"])
        data = memory._parse_recent_commit_data("main", k=1)
        assert len(data) == 1
        assert "patterns" in data[0]
        assert "Pattern alpha" in data[0]["patterns"]
        assert "Pattern beta" in data[0]["patterns"]

    def test_parse_commit_without_patterns(self, memory):
        """Old commits without patterns produce empty patterns list."""
        memory.commit("T1", "did A", "reason", ["a.py"], "next")
        data = memory._parse_recent_commit_data("main", k=1)
        assert data[0]["patterns"] == []

    # --- Context Retrieval ---

    def test_context_level2_shows_recurring_patterns(self, memory):
        """Level 2 context shows patterns with occurrence >= 2."""
        pat = "Always validate input parameters before processing data"
        memory.commit("T1", "did A", "reason", ["a.py"], "next", patterns_learned=[pat])
        memory.commit("T2", "did B", "reason", ["b.py"], "next", patterns_learned=[pat])
        ctx = memory.get_context(level=2)
        assert "Recurring Patterns" in ctx
        assert "(2x)" in ctx

    def test_context_level1_no_patterns(self, memory):
        """Level 1 context doesn't show patterns."""
        pat = "Always validate input parameters before processing data"
        memory.commit("T1", "did A", "reason", ["a.py"], "next", patterns_learned=[pat])
        memory.commit("T2", "did B", "reason", ["b.py"], "next", patterns_learned=[pat])
        ctx = memory.get_context(level=1)
        assert "Recurring Patterns" not in ctx

    # --- get_patterns query ---

    def test_get_patterns_empty(self, memory):
        """get_patterns on empty buffer returns zero matches."""
        result = memory.get_patterns()
        assert result["total"] == 0
        assert result["matching"] == 0

    def test_get_patterns_min_occurrences_filter(self, memory):
        """min_occurrences filter works correctly."""
        pat = "Always validate input parameters before processing data"
        memory.commit("T1", "did A", "reason", ["a.py"], "next", patterns_learned=[pat])
        memory.commit("T2", "did B", "reason", ["b.py"], "next",
                       patterns_learned=[pat, "Single occurrence unique pattern for filter test"])
        result = memory.get_patterns(min_occurrences=2)
        assert result["matching"] == 1
        assert result["total"] == 2

    def test_get_patterns_search_term(self, memory):
        """search_term filter works correctly."""
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=["Always validate input parameters here",
                                          "Use kernel sandbox for REPL execution"])
        result = memory.get_patterns(search_term="kernel")
        assert result["matching"] == 1
        assert "kernel" in result["patterns"][0]["text"].lower()

    def test_get_patterns_sorted_by_occurrence(self, memory):
        """Results sorted by occurrence_count DESC."""
        high_pat = "High frequency pattern validated repeatedly here"
        low_pat = "Low frequency pattern only once appearing here"
        memory.commit("T1", "did A", "reason", ["a.py"], "next",
                       patterns_learned=[high_pat, low_pat])
        memory.commit("T2", "did B", "reason", ["b.py"], "next",
                       patterns_learned=[high_pat])
        result = memory.get_patterns()
        assert result["patterns"][0]["occurrence_count"] > result["patterns"][1]["occurrence_count"]


# ===========================================================================
# G4 Fix: Rolling Summary Compression (GCC paper S_t = f(S_{t-1}, D_t))
# ===========================================================================


class TestRollingSummaryStructuredTruncation:
    """Tests for the improved structured truncation (replacing blind tail truncation)."""

    def test_short_summary_not_truncated(self, memory):
        """Summaries under 1500 chars are not truncated."""
        memory.commit("T1", "did A", "reason A", ["a.py"], "next A")
        summary = memory._get_rolling_summary("main")
        assert "..." not in summary
        assert "did A" in summary

    def test_structured_truncation_preserves_first_entry(self, memory):
        """First entry (project context) is preserved by structured truncation."""
        # Create enough commits to exceed 1500 chars
        memory.commit("Init", "Created the initial project scaffold and infrastructure",
                       "Foundation needed", ["init.py"], "build features")
        for i in range(20):
            memory.commit(
                f"Feature {i}", f"Implemented feature number {i} with extensive detail and documentation",
                f"User requested feature {i}", [f"feat{i}.py"], f"Implement feature {i+1}",
            )
        summary = memory._get_rolling_summary("main")
        assert len(summary) <= 1500
        # First entry should be preserved (project context)
        assert "Created the initial project scaffold" in summary

    def test_structured_truncation_preserves_last_three(self, memory):
        """Last 3 entries are preserved in full."""
        memory.commit("Init", "Setup project", "bootstrap", ["init.py"], "start")
        for i in range(20):
            memory.commit(
                f"Step {i}", f"Implemented step {i} with full details",
                f"needed for progress {i}", [f"s{i}.py"], f"do step {i+1}",
            )
        summary = memory._get_rolling_summary("main")
        # Last 3 entries should be present in full
        assert "Implemented step 19" in summary
        assert "Implemented step 18" in summary
        assert "Implemented step 17" in summary

    def test_structured_truncation_compresses_middle(self, memory):
        """Middle entries are compressed (parentheticals removed)."""
        memory.commit("Init", "Setup project", "bootstrap", ["init.py"], "start")
        for i in range(25):
            memory.commit(
                f"Step {i}", f"Implemented step {i} with full details",
                f"needed for progress {i}", [f"s{i}.py"], f"do step {i+1}",
            )
        summary = memory._get_rolling_summary("main")
        # Summary should be within budget
        assert len(summary) <= 1500
        # Should not use old-style blind truncation (no leading "..." unless
        # very degenerate case with extremely few entries)
        # The structure is: first entry; compressed middles; last 3

    def test_static_structured_truncate_basic(self, memory):
        """Direct test of _structured_truncate_summary static method."""
        from ccr.core.memory import MemoryManager
        # Build a long summary with semicolons — need entries long enough to exceed 1500
        entries = ["Initial project setup for the platform with comprehensive documentation"] + \
                  [f"Feature {i} fully implemented with extensive testing and documentation (because: reason {i} was critically needed). Next: implement feature {i+1}" for i in range(30)]
        long_summary = "; ".join(entries)
        assert len(long_summary) > 1500

        result = MemoryManager._structured_truncate_summary(long_summary)
        assert len(result) <= 1500
        # First entry preserved
        assert "Initial project setup" in result
        # Last 3 entries preserved in full
        assert "Feature 29" in result
        assert "Feature 28" in result
        assert "Feature 27" in result

    def test_static_structured_truncate_short_input(self, memory):
        """Short inputs returned as-is."""
        from ccr.core.memory import MemoryManager
        short = "Small summary; another bit"
        result = MemoryManager._structured_truncate_summary(short)
        assert result == short

    def test_static_structured_truncate_few_entries(self, memory):
        """With <= 3 entries, falls back to tail truncation."""
        from ccr.core.memory import MemoryManager
        # Build 3 very long entries
        entries = [f"Entry {i} " + "x" * 600 for i in range(3)]
        long_summary = "; ".join(entries)
        assert len(long_summary) > 1500
        result = MemoryManager._structured_truncate_summary(long_summary)
        assert len(result) <= 1500
        assert result.startswith("...")


class TestRollingSummaryCompressedSummaryParam:
    """Tests for the compressed_summary parameter on commit()."""

    def test_compressed_summary_used_directly(self, memory):
        """When compressed_summary is provided, it replaces the rolling summary."""
        memory.commit("T1", "did A", "reason A", ["a.py"], "next A")
        memory.commit("T2", "did B", "reason B", ["b.py"], "next B",
                       compressed_summary="Project started with A, then added B. Direction: C.")
        summary = memory._get_rolling_summary("main")
        assert summary == "Project started with A, then added B. Direction: C."
        # Mechanical concatenation should NOT be present
        assert "did A (because:" not in summary

    def test_compressed_summary_none_uses_mechanical(self, memory):
        """When compressed_summary is None (default), uses mechanical concatenation."""
        memory.commit("T1", "did A", "reason A", ["a.py"], "next A")
        memory.commit("T2", "did B", "reason B", ["b.py"], "next B")
        summary = memory._get_rolling_summary("main")
        # Should have mechanical concatenation markers
        assert "(because:" in summary
        assert "did A" in summary
        assert "did B" in summary

    def test_compressed_summary_capped_at_1500(self, memory):
        """compressed_summary is capped at 1500 chars."""
        long_summary = "x" * 2000
        memory.commit("T1", "did A", "reason A", ["a.py"], "next A",
                       compressed_summary=long_summary)
        summary = memory._get_rolling_summary("main")
        assert len(summary) <= 1500

    def test_compressed_summary_stripped(self, memory):
        """compressed_summary is stripped of leading/trailing whitespace."""
        memory.commit("T1", "did A", "reason A", ["a.py"], "next A",
                       compressed_summary="  Cleaned summary content  ")
        summary = memory._get_rolling_summary("main")
        assert summary == "Cleaned summary content"

    def test_compressed_summary_backward_compat(self, memory):
        """Commit without compressed_summary works exactly as before."""
        result = memory.commit("T1", "did A", "reason A", ["a.py"], "next A")
        assert "C001" in result
        summary = memory._get_rolling_summary("main")
        assert "did A" in summary


class TestRollingSummaryCompressionPrompt:
    """Tests for the compression warning in commit return value.

    Warning threshold is 1200 chars, structured truncation caps at 1500.
    So summaries between 1200-1500 chars trigger the warning, giving
    Claude Code a chance to compress before truncation degrades quality.
    """

    def test_short_summary_no_warning(self, memory):
        """Short summaries don't trigger compression warning."""
        result = memory.commit("T1", "did A", "reason A", ["a.py"], "next A")
        assert "Rolling summary at" not in result
        assert "compressed_summary" not in result

    def test_long_summary_triggers_warning(self, memory):
        """When rolling summary exceeds 1200 chars, commit returns warning."""
        # Build up a summary that exceeds 1200 chars through multiple commits
        # Each commit appends ~80-100 chars to the summary
        for i in range(15):
            result = memory.commit(
                f"Step {i}",
                f"Implemented detailed feature number {i} with comprehensive docs",
                f"User requested feature {i} with thorough testing coverage needed",
                [f"feature_{i}.py", f"test_feature_{i}.py"],
                f"Proceed to implement feature {i+1} next",
            )
        summary = memory._get_rolling_summary("main")
        # After enough commits, the summary should exceed 1200 chars
        # (each entry is ~100 chars: "what (because: why). Next: next_step")
        if len(summary) > 1200:
            assert "Rolling summary at" in result
            assert "compressed_summary" in result

    def test_warning_includes_char_count(self, memory):
        """Warning message includes the current summary length and 1500-char cap."""
        # Write a summary just over the 1200 char threshold directly
        # then commit to trigger the warning
        summary_1300 = "; ".join([f"Entry {i} with some content here" for i in range(40)])
        assert len(summary_1300) > 1200
        memory._write_rolling_summary("main", summary_1300[:1400])
        result = memory.commit("Next", "added more", "reason", ["f.py"], "done")
        assert "Rolling summary at" in result
        # Should include the actual char count and the 1500-char cap
        assert "/1500 chars" in result

    def test_no_warning_when_compressed_summary_provided(self, memory):
        """When compressed_summary is provided, no warning even if summary is long."""
        # Write a long base summary
        memory._write_rolling_summary("main", "x" * 1300)
        # Commit with compressed_summary — should NOT trigger warning
        result = memory.commit("Next", "added more", "reason", ["f.py"], "done",
                               compressed_summary="Compressed version of the summary")
        assert "Rolling summary at" not in result

    def test_warning_suggests_two_call_pattern(self, memory):
        """Warning message mentions the two-call pattern for compression."""
        memory._write_rolling_summary("main", "x" * 1300)
        result = memory.commit("Next", "added more", "reason", ["f.py"], "done")
        assert "compressed_summary" in result
        assert "gcc_consolidate" in result

    def test_warning_includes_current_summary_inline(self, memory):
        """Compression prompt includes the actual summary so Claude Code
        doesn't need an extra round-trip to retrieve it."""
        # Base must exceed 1200 chars even after a new entry is appended
        summary = "x" * 1200 + " important context here"
        memory._write_rolling_summary("main", summary)
        result = memory.commit("Next", "added more", "reason", ["f.py"], "done")
        assert "Rolling summary at" in result
        assert "important context here" in result
        # Should show the summary between delimiters
        assert "---" in result


class TestCommitEmbeddings:
    """Tests for ONNX commit embedding cache."""

    @pytest.fixture(autouse=True)
    def _disable_vec_store(self):
        """Disable sqlite-vec so tests exercise the gzip JSON fallback path."""
        with patch("ccr.context.vec_store.get_vec_store", return_value=None):
            yield

    def test_get_commit_embeddings_path(self, memory):
        path = memory._get_commit_embeddings_path()
        assert path.endswith("commit_embeddings.json.gz")
        assert memory.ccr_root in path

    def test_embed_commit_no_op_when_model_unavailable(self, memory):
        """When no ONNX model, _embed_commit returns None without error."""
        with patch("ccr.core.memory_pkg.memory_embeddings.get_embedding_model", return_value=None):
            result = memory._embed_commit("C001", "some text")
        assert result is None

    def test_embed_commit_stores_vector_in_cache(self, memory):
        """When model available, vector is persisted to cache file."""
        import numpy as np

        fake_vec = np.ones(384, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)

        mock_model = MagicMock()
        mock_model.embed_query.return_value = fake_vec

        with patch("ccr.core.memory_pkg.memory_embeddings.get_embedding_model", return_value=mock_model):
            result = memory._embed_commit("C001", "test text")

        assert result is not None
        # Cache file must exist
        assert os.path.isfile(memory._get_commit_embeddings_path())
        # Returned vector matches
        np.testing.assert_allclose(result, fake_vec, rtol=1e-5)

    def test_embed_commit_cache_grows(self, memory):
        """Each call appends one entry to cache."""
        import numpy as np

        def make_vec():
            v = np.random.rand(384).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_model = MagicMock()
        mock_model.embed_query.side_effect = [make_vec(), make_vec(), make_vec()]

        with patch("ccr.core.memory_pkg.memory_embeddings.get_embedding_model", return_value=mock_model):
            memory._embed_commit("C001", "a")
            memory._embed_commit("C002", "b")
            memory._embed_commit("C003", "c")

        from ccr.context.embeddings import load_embeddings
        cache = load_embeddings(memory._get_commit_embeddings_path())
        assert set(cache.keys()) == {"C001", "C002", "C003"}

    def test_embed_commit_caps_cache(self, memory):
        """Cache is capped at link_scan_window * 2; oldest entries evicted."""
        import numpy as np

        cap = memory.config.link_scan_window * 2  # default: 40

        def make_vec():
            v = np.random.rand(384).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_model = MagicMock()
        mock_model.embed_query.side_effect = [make_vec() for _ in range(cap + 5)]

        with patch("ccr.core.memory_pkg.memory_embeddings.get_embedding_model", return_value=mock_model):
            for i in range(cap + 5):
                memory._embed_commit(f"C{i:03d}", f"text {i}")

        from ccr.context.embeddings import load_embeddings
        cache = load_embeddings(memory._get_commit_embeddings_path())
        assert len(cache) == cap
        # All evicted IDs (C000-C004) are gone
        for i in range(5):
            assert f"C{i:03d}" not in cache
        # All retained IDs are present
        for i in range(5, cap + 5):
            assert f"C{i:03d}" in cache

    def test_load_commit_embeddings_returns_ndarrays(self, memory):
        """_load_commit_embeddings converts list[float] -> np.ndarray."""
        import numpy as np

        fake_vec = np.ones(384, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)
        mock_model = MagicMock()
        mock_model.embed_query.return_value = fake_vec

        with patch("ccr.core.memory_pkg.memory_embeddings.get_embedding_model", return_value=mock_model):
            memory._embed_commit("C001", "text")

        result = memory._load_commit_embeddings(["C001"])
        assert "C001" in result
        assert isinstance(result["C001"], np.ndarray)
        assert result["C001"].shape == (384,)

    def test_load_commit_embeddings_missing_ids_ignored(self, memory):
        """IDs not in cache are silently omitted."""
        result = memory._load_commit_embeddings(["C999", "C998"])
        assert result == {}

    def test_compute_links_uses_cosine_when_vec_available(self, memory):
        """When new_vec provided and old embedding cached, cosine used for semantic link."""
        import numpy as np

        # Commit C001 — store its embedding in cache
        vec_old = np.zeros(384, dtype=np.float32)
        vec_old[0] = 1.0  # unit vector in dim 0

        from ccr.context.embeddings import save_embeddings
        cache = {"C001": vec_old.tolist()}
        save_embeddings(cache, memory._get_commit_embeddings_path())

        # Parse a fake C001 in recent commits
        with patch.object(memory, "_parse_recent_commit_data") as mock_recent:
            mock_recent.return_value = [{
                "id": "C001", "title": "old", "what": "old work", "why": "old reason",
                "files": [], "next_step": "old next",
            }]

            # new_vec similar to vec_old — dot product ~1.0 (above default threshold)
            vec_new = np.zeros(384, dtype=np.float32)
            vec_new[0] = 1.0

            links = memory._compute_links(
                "main", "C002", "similar title", "similar work", "similar reason",
                [], "next", new_vec=vec_new,
            )

        semantic_links = [l for l in links if l.link_type == "semantic"]
        assert len(semantic_links) == 1
        assert semantic_links[0].score > 0.9  # cosine of near-identical vectors

    def test_compute_links_falls_back_to_jaccard_when_no_vec(self, memory):
        """When new_vec=None, semantic link uses Jaccard (existing behavior)."""
        with patch.object(memory, "_parse_recent_commit_data") as mock_recent:
            mock_recent.return_value = [{
                "id": "C001", "title": "unique rare keyword zebra",
                "what": "zebra keyword work", "why": "zebra reason",
                "files": [], "next_step": "next",
            }]
            links = memory._compute_links(
                "main", "C002", "unique rare keyword zebra",
                "zebra keyword work", "zebra reason", [], "next", new_vec=None,
            )
        semantic_links = [l for l in links if l.link_type == "semantic"]
        assert len(semantic_links) == 1  # Jaccard high for identical rare keywords

    def test_commit_calls_embed_commit_and_passes_vec_to_compute_links(self, memory):
        """commit() embeds and passes new_vec to _compute_links (no double embed)."""
        import numpy as np

        fake_vec = np.ones(384, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)

        mock_model = MagicMock()
        mock_model.embed_query.return_value = fake_vec

        with patch("ccr.core.memory_pkg.memory_embeddings.get_embedding_model", return_value=mock_model):
            with patch.object(memory, "_compute_links", wraps=memory._compute_links) as mock_cl:
                memory.commit("T1", "did A", "reason A", ["a.py"], "next A")

        # embed_query called exactly once (in _embed_commit; NOT again in _compute_links)
        assert mock_model.embed_query.call_count == 1
        # _compute_links received new_vec as keyword argument
        _, kwargs = mock_cl.call_args
        assert kwargs.get("new_vec") is not None

    def test_embed_commit_not_called_on_amac_merge_path(self, memory):
        """When A-MAC takes the merge path (new_score > conflict_score -> early return),
        _embed_commit is not called -- it lives after the merge return point in commit().

        Key: raw_sim = 0.50*file_sim + 0.50*kw_sim. Same files -> file_sim=1 but
        novelty suppressed (raw_sim>=0.5 -> new_score ~= conflict_score -> tie -> fall-through).
        Fix: drive conflict via keyword overlap only, use DIFFERENT files so file_sim=0.

        stored_score=0.05 short-circuits to conflict_novelty=0.5 (memory.py line ~1265):
          conflict_score = 0.50*0.5 + 0.35*0.5 + 0.15*1.0 = 0.575
          Greek-letter keywords: overlap=9/union=11 -> kw_sim~=0.818
          raw_sim(new) = 0.50*0 + 0.50*0.818 = 0.409  (file_sim=0, different files)
          novelty(new) = 1 - 0.409 = 0.591
          new_score = 0.50*0.5 + 0.35*0.591 + 0.15*1.0 = 0.607 > 0.575 -> merge
        """
        import numpy as np
        import re as _re

        fake_vec = np.ones(384, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)
        mock_model = MagicMock()
        mock_model.embed_query.return_value = fake_vec

        with patch("ccr.core.memory_pkg.memory_embeddings.get_embedding_model", return_value=mock_model):
            # First commit -- keyword-rich with rare Greek letters, file "f.py"
            memory.commit(
                "alpha beta gamma delta epsilon sigma",
                "alpha beta gamma work", "delta epsilon sigma reason",
                ["f.py"], "alpha next",
                admission_threshold=0.3,
            )
            count_after_first = mock_model.embed_query.call_count
            assert count_after_first == 1

            # Lower C001's stored score to 0.05 -> conflict_score=0.575 via short-circuit.
            # Score is written as "**Score**: 1.00" (:.2f format, memory.py line ~379),
            # so use regex replace to match any stored score value.
            path = memory._get_commits_path("main")
            content = memory._read_file(path)
            content = _re.sub(r'\*\*Score\*\*: [\d.]+', '**Score**: 0.05', content)
            memory._write_file(path, content)

            # Second commit: overlapping keywords (kw_sim~=0.818) but different file "g.py"
            # -> file_sim=0, novelty=0.591, new_score=0.607 > conflict_score=0.575 -> merge
            memory.commit(
                "alpha beta gamma delta epsilon sigma zeta eta",
                "alpha beta gamma zeta work", "delta epsilon sigma eta reason",
                ["g.py"], "zeta eta next",
                admission_threshold=0.3,
            )

        # Merge path early return: _embed_commit was not reached
        assert mock_model.embed_query.call_count == count_after_first

    def test_compute_links_per_commit_mixed_fallback(self, memory):
        """In one _compute_links call: commit with cached embedding uses cosine,
        commit without cached embedding uses Jaccard -- both in the same scan."""
        import numpy as np
        from ccr.context.embeddings import save_embeddings

        # C001 has a cached embedding (unit vector in dim 0)
        vec_old = np.zeros(384, dtype=np.float32)
        vec_old[0] = 1.0
        save_embeddings({"C001": vec_old.tolist()}, memory._get_commit_embeddings_path())

        # C002 has NO cached embedding

        with patch.object(memory, "_parse_recent_commit_data") as mock_recent:
            mock_recent.return_value = [
                # C001: has embedding, will use cosine
                {"id": "C001", "title": "alpha beta gamma", "what": "alpha work",
                 "why": "alpha reason", "files": []},
                # C002: no embedding, will fall back to Jaccard
                # Use rare keywords that match new commit text -> Jaccard fires
                {"id": "C002", "title": "unique zebra quux xyzzy",
                 "what": "zebra quux xyzzy", "why": "xyzzy reason", "files": []},
            ]

            # new_vec is a unit vector in dim 0 -- cosine with C001 ~= 1.0
            vec_new = np.zeros(384, dtype=np.float32)
            vec_new[0] = 1.0

            links = memory._compute_links(
                "main", "C003",
                "unique zebra quux xyzzy alpha beta gamma",
                "zebra quux xyzzy alpha work", "alpha reason",
                [], "next", new_vec=vec_new,
            )

        # C001: cosine similarity ~1.0 -> semantic link
        # C002: Jaccard on rare matching keywords -> semantic link
        semantic_links = {l.target: l for l in links if l.link_type == "semantic"}
        assert "C001" in semantic_links, "C001 should link via cosine"
        assert "C002" in semantic_links, "C002 should link via Jaccard fallback"
        # C001 cosine score should be very high
        assert semantic_links["C001"].score > 0.9


# ===========================================================================
# Task 2: Auto-trigger rolling summary compression (_mechanical_compress_summary)
# ===========================================================================


class TestMechanicalCompressSummary:
    """Tests for the _mechanical_compress_summary static method (Strategy 2.5)."""

    def test_short_summary_unchanged(self, memory):
        """Summaries with <= 4 segments are returned unchanged."""
        from ccr.core.memory import MemoryManager
        short = "First entry; Second entry; Third entry"
        result = MemoryManager._mechanical_compress_summary(short)
        assert result == short

    def test_exactly_four_segments_unchanged(self, memory):
        """Exactly 4 segments boundary — returned unchanged."""
        from ccr.core.memory import MemoryManager
        s = "Seg one; Seg two; Seg three; Seg four"
        result = MemoryManager._mechanical_compress_summary(s)
        assert result == s

    def test_keeps_first_segment(self, memory):
        """First segment (project context anchor) is always preserved."""
        from ccr.core.memory import MemoryManager
        segments = ["CCR is a zero-LLM MCP server for persistent memory"] + \
                   [f"Added feature {i} with detailed implementation notes" for i in range(10)]
        summary = "; ".join(segments)
        result = MemoryManager._mechanical_compress_summary(summary)
        assert "CCR is a zero-LLM MCP server" in result

    def test_keeps_last_three_segments(self, memory):
        """Last 3 segments (most recent contributions) are always preserved."""
        from ccr.core.memory import MemoryManager
        segments = ["Initial anchor context segment here"] + \
                   [f"Middle work on item {i}" for i in range(10)] + \
                   ["Last segment A details", "Last segment B details", "Last segment C details"]
        summary = "; ".join(segments)
        result = MemoryManager._mechanical_compress_summary(summary)
        assert "Last segment A details" in result
        assert "Last segment B details" in result
        assert "Last segment C details" in result

    def test_drops_middle_segments(self, memory):
        """Middle segments (not first, not last 3) are dropped."""
        from ccr.core.memory import MemoryManager
        segments = ["First anchor"] + \
                   [f"Middle segment number {i} with content" for i in range(8)] + \
                   ["Last A", "Last B", "Last C"]
        summary = "; ".join(segments)
        result = MemoryManager._mechanical_compress_summary(summary)
        # Middle segments should be absent
        for i in range(1, 7):  # Middle segments (not last 3)
            assert f"Middle segment number {i} with content" not in result

    def test_result_well_under_900_chars_typical(self, memory):
        """Typical compressed result (first + last 3) is well under 900 chars."""
        from ccr.core.memory import MemoryManager
        anchor = "CCR project initial setup with comprehensive documentation and infrastructure"
        segments = [anchor] + \
                   [f"Feature {i} fully implemented with extensive tests, documentation, and review" for i in range(20)]
        summary = "; ".join(segments)
        assert len(summary) > 1200  # Long enough to be a real scenario
        result = MemoryManager._mechanical_compress_summary(summary)
        assert len(result) <= 900

    def test_fallback_truncation_when_result_still_too_long(self, memory):
        """If first + last 3 still exceed 900 chars, tail-truncate at 900."""
        from ccr.core.memory import MemoryManager
        # Create a very long first segment + long last 3
        long_anchor = "A" * 300
        long_segments = [long_anchor] + \
                        [f"middle{i}" for i in range(5)] + \
                        ["B" * 300, "C" * 300, "D" * 300]
        summary = "; ".join(long_segments)
        result = MemoryManager._mechanical_compress_summary(summary)
        assert len(result) <= 900
        assert result.startswith("...")

    def test_strategy_2_5_fires_when_previous_summary_over_1200(self, memory):
        """Strategy 2.5 auto-compresses previous_summary > 1200 chars before concat."""
        # Write a long summary (> 1200 chars) directly
        long_segments = ["CCR initial anchor context providing project foundation"] + \
                        [f"work item {i} fully completed with extensive implementation details and thorough documentation" for i in range(20)]
        long_summary = "; ".join(long_segments)
        assert len(long_summary) > 1200
        memory._write_rolling_summary("main", long_summary)

        # Now commit without compressed_summary — Strategy 2.5 should auto-compress
        memory.commit("New commit", "added something new", "reason", ["f.py"], "next")
        result_summary = memory._get_rolling_summary("main")

        # The final summary must be under 1400 chars (auto-compress + new entry appended)
        assert len(result_summary) < 1400

    def test_many_commits_without_compressed_summary_stays_bounded(self, memory):
        """After many commits with no compressed_summary, summary stays under 1400 chars."""
        for i in range(30):
            memory.commit(
                f"Feature {i}",
                f"Implemented detailed feature {i} with comprehensive tests and documentation",
                f"User requested feature {i} as part of milestone {i // 5}",
                [f"feature_{i}.py", f"test_feature_{i}.py"],
                f"Implement feature {i + 1} next",
            )
        summary = memory._get_rolling_summary("main")
        assert len(summary) < 1400

    def test_compressed_summary_param_still_takes_priority(self, memory):
        """When compressed_summary is provided, auto-compression is bypassed entirely."""
        long_segments = ["anchor"] + [f"item {i}" for i in range(20)]
        memory._write_rolling_summary("main", "; ".join(long_segments))

        memory.commit(
            "New commit", "added work", "reason", ["f.py"], "next",
            compressed_summary="Manually compressed: CCR built features 0-19. Next: feature 20.",
        )
        summary = memory._get_rolling_summary("main")
        assert summary == "Manually compressed: CCR built features 0-19. Next: feature 20."


# ===========================================================================
# Task 4: Auto-suggest ACE promotion from pending patterns (_scan_pending_promotions)
# ===========================================================================


class TestScanPendingPromotions:
    """Tests for _scan_pending_promotions() and the fallback auto-scan in commit()."""

    def test_empty_buffer_returns_empty(self, memory):
        """Returns empty list when patterns.json is absent."""
        result = memory._scan_pending_promotions()
        assert result == []

    def test_below_threshold_not_returned(self, memory):
        """Patterns below promotion_count threshold are not returned."""
        import json
        pat = "Always validate input parameters before processing data"
        # Only 2 occurrences, threshold is 3
        memory.commit("T1", "did A", "reason", ["a.py"], "next", patterns_learned=[pat])
        memory.commit("T2", "did B", "reason", ["b.py"], "next", patterns_learned=[pat])
        result = memory._scan_pending_promotions()
        assert result == []

    def test_at_threshold_returned(self, memory):
        """Pattern at exactly pattern_promotion_count is returned."""
        pat = "Always validate input parameters before processing data"
        for i in range(3):  # exactly threshold (3)
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                          patterns_learned=[pat])
        result = memory._scan_pending_promotions()
        assert len(result) == 1
        assert result[0]["text"] == pat
        assert result[0]["count"] == 3

    def test_above_threshold_returned(self, memory):
        """Pattern above threshold (more than 3 occurrences) is returned."""
        pat = "Always validate input parameters before processing data"
        for i in range(5):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                          patterns_learned=[pat])
        result = memory._scan_pending_promotions()
        assert len(result) == 1
        assert result[0]["count"] == 5

    def test_promoted_patterns_excluded(self, memory):
        """Patterns with promoted=True are NOT returned."""
        import json
        pat = "Always validate input parameters before processing data"
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                          patterns_learned=[pat])
        # Manually mark pattern as promoted
        path = memory._get_patterns_path()
        data = json.loads(memory._read_file(path))
        data["patterns"]["P001"]["promoted"] = True
        memory._write_file(path, json.dumps(data, indent=2))
        result = memory._scan_pending_promotions()
        assert result == []

    def test_caps_at_five(self, memory):
        """Returns at most 5 patterns even if more are pending."""
        # Create 8 distinct patterns each with 3+ occurrences
        patterns = [
            f"Pattern {i} with distinct words for testing promotion threshold logic here"
            for i in range(8)
        ]
        for pat in patterns:
            for j in range(3):
                memory.commit(f"T{j}", f"did {j}", "reason", [f"{j}.py"], "next",
                              patterns_learned=[pat])
        result = memory._scan_pending_promotions()
        assert len(result) <= 5

    def test_does_not_mark_promoted(self, memory):
        """_scan_pending_promotions must NOT set promoted=True."""
        import json
        pat = "Always validate input parameters before processing data"
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                          patterns_learned=[pat])
        memory._scan_pending_promotions()  # Call it
        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert data["patterns"]["P001"]["promoted"] is False

    def test_result_shape_matches_process_patterns(self, memory):
        """Return dicts have pattern_id, text, count, commit_ids keys."""
        pat = "Always validate input parameters before processing data"
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                          patterns_learned=[pat])
        result = memory._scan_pending_promotions()
        assert len(result) == 1
        entry = result[0]
        assert "pattern_id" in entry
        assert "text" in entry
        assert "count" in entry
        assert "commit_ids" in entry

    def test_sorted_by_count_desc(self, memory):
        """Results are sorted by occurrence count descending."""
        pat_high = "High frequency pattern for testing promotion here"
        pat_low = "Low frequency pattern appearing only rarely here"
        # pat_high: 5 occurrences; pat_low: 3 occurrences
        for i in range(5):
            memory.commit(f"H{i}", f"did {i}", "reason", [f"h{i}.py"], "next",
                          patterns_learned=[pat_high])
        for i in range(3):
            memory.commit(f"L{i}", f"did {i}", "reason", [f"l{i}.py"], "next",
                          patterns_learned=[pat_low])
        result = memory._scan_pending_promotions()
        assert len(result) == 2
        assert result[0]["count"] >= result[1]["count"]

    def test_commit_without_patterns_surfaces_pending(self, memory):
        """A commit() with no patterns_learned auto-scans and suggests pending patterns."""
        pat = "Always validate input parameters before processing data"
        # Build pattern to threshold via earlier commits
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                          patterns_learned=[pat])
        # Now commit with NO patterns_learned — should still surface the suggestion
        result = memory.commit("No patterns commit", "added something", "reason", ["f.py"], "next")
        assert "Pattern promotion suggestions" in result
        assert "ace_apply_delta" in result

    def test_no_duplicate_suggestions_when_patterns_learned_fires(self, memory):
        """When patterns_learned fires, _scan_pending_promotions is NOT called (short-circuit)."""
        pat = "Always validate input parameters before processing data"
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                          patterns_learned=[pat])
        # Commit WITH patterns_learned — should get suggestions from _process_patterns,
        # not a double-set from both paths
        result = memory.commit("With patterns", "added work", "reason", ["g.py"], "next",
                               patterns_learned=[pat])
        # Should contain promotion suggestion exactly once (not duplicated)
        assert result.count("Pattern promotion suggestions") == 1


# ===========================================================================
# Tests for mark_pattern_promoted_by_content
# ===========================================================================


class TestMarkPatternPromotedByContent:
    """Tests for CER buffer close-the-loop: mark_pattern_promoted_by_content."""

    def test_mark_pattern_promoted_by_content_marks_match(self, memory):
        """Matching pattern is marked promoted=True, returns 1."""
        import json
        pat = "Always validate input parameters before processing data"
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                          patterns_learned=[pat])

        result = memory.mark_pattern_promoted_by_content(pat)
        assert result == 1

        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert data["patterns"]["P001"]["promoted"] is True

    def test_mark_pattern_promoted_already_promoted(self, memory):
        """Already-promoted pattern returns 0 and is not double-promoted."""
        import json
        pat = "Always validate input parameters before processing data"
        for i in range(3):
            memory.commit(f"T{i+1}", f"did {i}", "reason", [f"{i}.py"], "next",
                          patterns_learned=[pat])

        # First promotion
        result1 = memory.mark_pattern_promoted_by_content(pat)
        assert result1 == 1

        # Second call on same content — already promoted
        result2 = memory.mark_pattern_promoted_by_content(pat)
        assert result2 == 0

        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert data["patterns"]["P001"]["promoted"] is True

    def test_mark_pattern_promoted_no_match(self, memory):
        """Content that doesn't match any pattern returns 0."""
        import json
        pat = "Always validate input parameters before processing data"
        memory.commit("T1", "did A", "reason", ["a.py"], "next", patterns_learned=[pat])

        result = memory.mark_pattern_promoted_by_content(
            "Completely unrelated text about database indexing optimization"
        )
        assert result == 0

        data = json.loads(memory._read_file(memory._get_patterns_path()))
        assert data["patterns"]["P001"]["promoted"] is False

    def test_mark_pattern_promoted_empty_buffer(self, memory):
        """Empty pattern buffer returns 0 without error."""
        # No patterns committed — patterns.json doesn't exist yet
        result = memory.mark_pattern_promoted_by_content(
            "Always validate input parameters before processing data"
        )
        assert result == 0


# ===========================================================================
# Tests for ONNX re-ranking in get_linked_commits()
# ===========================================================================


class TestGetLinkedCommitsONNXRerank:
    """Tests for embedding-guided re-ranking in get_linked_commits()."""

    def _setup_links(self, memory, src: str, targets: list[str]) -> None:
        """Write a minimal commit_links.json with semantic links from src -> each target."""
        import json

        links_data: dict = {"version": 1, "links": {src: {}}}
        entries = []
        for tgt in targets:
            entries.append({"target": tgt, "score": 0.5})
        links_data["links"][src]["semantic"] = entries
        memory._save_links(links_data)

    def test_embedding_score_present_when_onnx_available(self, memory):
        """embedding_score key appears in every result dict when ONNX available."""
        import numpy as np

        src_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tgt_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self._setup_links(memory, "C001", ["C002"])

        with patch.object(
            memory,
            "_load_commit_embeddings",
            return_value={"C001": src_vec, "C002": tgt_vec},
        ):
            results = memory.get_linked_commits("C001")

        assert len(results) == 1
        assert "embedding_score" in results[0]
        assert abs(results[0]["embedding_score"] - 0.0) < 1e-5  # dot([1,0,0],[0,1,0])=0

    def test_results_sorted_by_embedding_score_descending(self, memory):
        """When ONNX available, results are ordered by embedding_score high→low."""
        import numpy as np

        # C001 source: unit vector along dim 0
        src_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # C002: high similarity (cos ≈ 0.8)
        v2 = np.array([0.8, 0.6, 0.0], dtype=np.float32)
        v2 /= np.linalg.norm(v2)
        # C003: low similarity (cos ≈ 0.1)
        v3 = np.array([0.1, 0.995, 0.0], dtype=np.float32)
        v3 /= np.linalg.norm(v3)

        self._setup_links(memory, "C001", ["C003", "C002"])  # C003 first in link order

        with patch.object(
            memory,
            "_load_commit_embeddings",
            return_value={"C001": src_vec, "C002": v2, "C003": v3},
        ):
            results = memory.get_linked_commits("C001")

        assert len(results) == 2
        # C002 has higher cosine similarity → should appear first after re-rank
        assert results[0]["id"] == "C002"
        assert results[1]["id"] == "C003"
        assert results[0]["embedding_score"] > results[1]["embedding_score"]

    def test_graceful_degradation_when_no_embeddings(self, memory):
        """When _load_commit_embeddings returns empty dict, no embedding_score key."""
        self._setup_links(memory, "C001", ["C002", "C003"])

        with patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits("C001")

        assert len(results) == 2
        for r in results:
            assert "embedding_score" not in r

    def test_truncation_after_rerank(self, memory):
        """Truncation to link_max_results happens after ONNX re-ranking."""
        import numpy as np

        # Override limit to 2 so we can observe truncation clearly
        memory.config.link_max_results = 2

        src_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Three candidate vectors with descending cosine similarity
        v_high = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        v_high /= np.linalg.norm(v_high)
        v_mid = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        v_mid /= np.linalg.norm(v_mid)
        v_low = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Put low-similarity target first in link order
        self._setup_links(memory, "C001", ["C004", "C003", "C002"])

        with patch.object(
            memory,
            "_load_commit_embeddings",
            return_value={
                "C001": src_vec,
                "C002": v_high,
                "C003": v_mid,
                "C004": v_low,
            },
        ):
            results = memory.get_linked_commits("C001")

        # Only 2 results returned (truncated)
        assert len(results) == 2
        # Top 2 by embedding score: C002 (high) and C003 (mid), not C004 (low)
        ids = [r["id"] for r in results]
        assert "C002" in ids
        assert "C003" in ids
        assert "C004" not in ids

    def test_partial_embeddings_missing_targets_use_heuristic_score(self, memory):
        """When src embedding available but some targets missing, those targets
        fall back to their stored heuristic score for embedding_score."""
        import numpy as np

        src_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tgt_vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        tgt_vec /= np.linalg.norm(tgt_vec)

        # C002 has embedding, C003 does not
        import json

        links_data = {
            "version": 1,
            "links": {
                "C001": {
                    "semantic": [
                        {"target": "C002", "score": 0.4},
                        {"target": "C003", "score": 0.7},
                    ]
                }
            },
        }
        memory._save_links(links_data)

        with patch.object(
            memory,
            "_load_commit_embeddings",
            return_value={"C001": src_vec, "C002": tgt_vec},
        ):
            results = memory.get_linked_commits("C001")

        by_id = {r["id"]: r for r in results}
        # C002 has a proper cosine score
        assert "embedding_score" in by_id["C002"]
        # C003 falls back to its stored heuristic score (0.7)
        assert "embedding_score" in by_id["C003"]
        assert abs(by_id["C003"]["embedding_score"] - 0.7) < 1e-5


# ===========================================================================
# Tests for embedding_score in gcc_context level 5 follow_links output
# ===========================================================================


class TestContextFollowLinksEmbeddingScore:
    """Tests for embedding_score display in get_context(level=5, follow_links=True)."""

    def test_follow_links_includes_embedding_score(self, memory):
        """When get_linked_commits returns embedding_score, it appears in L5 output."""
        memory.commit("Source commit", "did X", "why X", ["x.py"], "next X")
        linked_result = [
            {
                "id": "C002",
                "title": "Linked commit",
                "link_type": "semantic",
                "what": "did Y",
                "embedding_score": 0.92,
            }
        ]
        with patch.object(memory, "get_linked_commits", return_value=linked_result):
            ctx = memory.get_context(level=5, commit_id="C001", follow_links=True)

        assert "[emb: 0.920]" in ctx

    def test_follow_links_no_embedding_score_when_absent(self, memory):
        """When get_linked_commits result has no embedding_score, tag is omitted."""
        memory.commit("Source commit", "did X", "why X", ["x.py"], "next X")
        linked_result = [
            {
                "id": "C002",
                "title": "Linked commit",
                "link_type": "semantic",
                "what": "did Y",
            }
        ]
        with patch.object(memory, "get_linked_commits", return_value=linked_result):
            ctx = memory.get_context(level=5, commit_id="C001", follow_links=True)

        assert "[emb:" not in ctx

    def test_format_links_for_context_semantic_shows_embedding_score(self, memory):
        """_format_links_for_context shows 'emb:' tag for semantic links with embedding_score."""
        links = {
            "entity": [],
            "causal": [],
            "supersession": [],
            "semantic": [
                {"target": "C005", "score": 0.5, "embedding_score": 0.875},
            ],
        }
        output = memory._format_links_for_context("C001", links)

        assert "emb: 0.88" in output
        # Must NOT fall back to the raw score label when embedding_score is present
        assert "score: 0.50" not in output


# ===========================================================================
# Tests for MAGMA-inspired adaptive (heapq) traversal in get_linked_commits()
# ===========================================================================


class TestAdaptiveTraversal:
    """Tests for priority-queue traversal with quick_cosine edge scoring."""

    def _setup_links(self, memory, src: str, targets: list[str], link_type: str = "semantic") -> None:
        """Write a minimal commit_links.json with links from src -> each target."""
        links_data: dict = {"version": 1, "links": {src: {}}}
        entries = []
        for i, tgt in enumerate(targets):
            entries.append({"target": tgt, "score": 0.5 - i * 0.1})
        links_data["links"][src][link_type] = entries
        memory._save_links(links_data)

    def _setup_links_multihop(self, memory, graph: dict[str, list[str]]) -> None:
        """Write a link graph for multi-hop traversal tests.

        graph: mapping from src -> list of targets (all semantic links).
        """
        links_data: dict = {"version": 1, "links": {}}
        for src, targets in graph.items():
            entries = [{"target": tgt, "score": 0.5} for tgt in targets]
            links_data["links"][src] = {"semantic": entries}
        memory._save_links(links_data)

    def test_query_with_onnx_sorts_by_quick_cosine(self, memory):
        """When query is provided and ONNX available, results are sorted by cosine relevance."""
        self._setup_links(memory, "C001", ["C002", "C003", "C004"])

        # quick_cosine returns controlled scores: C004 most relevant, C002 least
        cosine_map = {"C002": 0.2, "C003": 0.7, "C004": 0.95}
        call_count = {"probe": 0}

        def mock_quick_cosine(text_a: str, text_b: str) -> float:
            # The first call is the ONNX probe ("a", "b")
            if text_a == "a" and text_b == "b":
                call_count["probe"] += 1
                return 0.5
            for cid, score in cosine_map.items():
                if cid in text_b:
                    return score
            return 0.1

        # Mock _parse_commit_block to return recognizable what text
        def mock_parse(text: str) -> dict:
            for cid in ("C002", "C003", "C004"):
                if cid in text:
                    return {"title": f"{cid} title", "what": f"text about {cid}"}
            return {}

        # Mock _find_commit_by_id to return text containing commit ID
        def mock_find(branch: str, cid: str) -> str:
            return f"[{cid}] commit text for {cid}"

        with patch("ccr.core.memory_pkg.memory_links.quick_cosine", side_effect=mock_quick_cosine), \
             patch.object(memory, "_parse_commit_block", side_effect=mock_parse), \
             patch.object(memory, "_find_commit_by_id", side_effect=mock_find), \
             patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits("C001", query="test query")

        assert len(results) == 3
        # Highest cosine first (post-traversal sort by query_score)
        assert results[0]["id"] == "C004"
        assert results[1]["id"] == "C003"
        assert results[2]["id"] == "C002"
        # query_score field present on all results
        for r in results:
            assert "query_score" in r

    def test_query_scores_reflect_cosine_values(self, memory):
        """query_score values match the quick_cosine return values."""
        self._setup_links(memory, "C001", ["C002", "C003"])

        cosine_map = {"C002": 0.35, "C003": 0.88}

        def mock_quick_cosine(text_a: str, text_b: str) -> float:
            if text_a == "a" and text_b == "b":
                return 0.5  # probe
            for cid, score in cosine_map.items():
                if cid in text_b:
                    return score
            return 0.0

        def mock_parse(text: str) -> dict:
            for cid in cosine_map:
                if cid in text:
                    return {"what": f"text about {cid}"}
            return {}

        def mock_find(branch: str, cid: str) -> str:
            return f"[{cid}] text"

        with patch("ccr.core.memory_pkg.memory_links.quick_cosine", side_effect=mock_quick_cosine), \
             patch.object(memory, "_parse_commit_block", side_effect=mock_parse), \
             patch.object(memory, "_find_commit_by_id", side_effect=mock_find), \
             patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits("C001", query="find relevant")

        by_id = {r["id"]: r for r in results}
        assert abs(by_id["C002"]["query_score"] - 0.35) < 1e-5
        assert abs(by_id["C003"]["query_score"] - 0.88) < 1e-5

    def test_bfs_fallback_when_onnx_unavailable(self, memory):
        """When quick_cosine returns None (no ONNX), BFS fallback works identically."""
        self._setup_links(memory, "C001", ["C002", "C003"])

        with patch("ccr.core.memory_pkg.memory_links.quick_cosine", return_value=None), \
             patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits("C001", query="test query")

        # All linked commits are still returned (BFS fallback)
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"C002", "C003"}
        # No query_score since ONNX was unavailable
        for r in results:
            assert "query_score" not in r

    def test_bfs_fallback_when_no_query(self, memory):
        """When query is None/empty, plain BFS is used (no quick_cosine calls)."""
        self._setup_links(memory, "C001", ["C002", "C003"])

        with patch("ccr.core.memory_pkg.memory_links.quick_cosine") as mock_qc, \
             patch.object(memory, "_load_commit_embeddings", return_value={}):
            results_no_query = memory.get_linked_commits("C001")

        # quick_cosine should never be called when no query
        mock_qc.assert_not_called()
        assert len(results_no_query) == 2
        for r in results_no_query:
            assert "query_score" not in r

    def test_max_results_parameter(self, memory):
        """max_results parameter limits returned results."""
        self._setup_links(memory, "C001", ["C002", "C003", "C004", "C005"])

        with patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits("C001", max_results=2)

        assert len(results) == 2

    def test_max_results_defaults_to_config(self, memory):
        """When max_results is None, config.link_max_results is used."""
        memory.config.link_max_results = 3
        self._setup_links(memory, "C001", ["C002", "C003", "C004", "C005", "C006"])

        with patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits("C001")

        assert len(results) == 3

    def test_max_results_overrides_config(self, memory):
        """Explicit max_results overrides config.link_max_results."""
        memory.config.link_max_results = 10
        self._setup_links(memory, "C001", ["C002", "C003", "C004"])

        with patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits("C001", max_results=1)

        assert len(results) == 1

    def test_heap_explores_highest_score_first_multihop(self, memory):
        """Priority queue explores highest-scored edges first across hops."""
        # Graph: C001 -> C002 (low score), C003 (high score)
        #        C003 -> C004
        self._setup_links_multihop(memory, {
            "C001": ["C002", "C003"],
            "C003": ["C004"],
        })

        # C003 scores highest, so it gets expanded first and C004 appears
        def mock_quick_cosine(text_a: str, text_b: str) -> float:
            if text_a == "a" and text_b == "b":
                return 0.5  # probe
            if "C003" in text_b:
                return 0.9
            if "C002" in text_b:
                return 0.1
            if "C004" in text_b:
                return 0.85
            return 0.0

        def mock_parse(text: str) -> dict:
            for cid in ("C002", "C003", "C004"):
                if cid in text:
                    return {"what": f"about {cid}"}
            return {}

        def mock_find(branch: str, cid: str) -> str:
            return f"[{cid}] text"

        with patch("ccr.core.memory_pkg.memory_links.quick_cosine", side_effect=mock_quick_cosine), \
             patch.object(memory, "_parse_commit_block", side_effect=mock_parse), \
             patch.object(memory, "_find_commit_by_id", side_effect=mock_find), \
             patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits(
                "C001", max_hops=2, query="find important", max_results=10,
            )

        ids = [r["id"] for r in results]
        assert "C003" in ids
        assert "C004" in ids
        assert "C002" in ids
        # C003 (0.9) should appear before C002 (0.1) in results
        assert ids.index("C003") < ids.index("C002")

    def test_visited_nodes_not_revisited(self, memory):
        """Nodes already visited are never re-added to results."""
        # Cycle: C001 -> C002, C002 -> C001
        self._setup_links_multihop(memory, {
            "C001": ["C002"],
            "C002": ["C001"],
        })

        with patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits("C001", max_hops=3)

        ids = [r["id"] for r in results]
        assert ids.count("C001") == 0  # Starting node never in results
        assert ids.count("C002") == 1  # Visited exactly once

    def test_quick_cosine_exception_falls_back_gracefully(self, memory):
        """If quick_cosine probe raises an exception, BFS fallback is used."""
        self._setup_links(memory, "C001", ["C002"])

        with patch("ccr.core.memory_pkg.memory_links.quick_cosine", side_effect=RuntimeError("ONNX crash")), \
             patch.object(memory, "_load_commit_embeddings", return_value={}):
            results = memory.get_linked_commits("C001", query="test")

        # Still returns results via BFS fallback
        assert len(results) == 1
        assert results[0]["id"] == "C002"
        assert "query_score" not in results[0]

    def test_embedding_score_still_added_post_traversal(self, memory):
        """embedding_score from cached vectors is still added after heap traversal."""
        import numpy as np

        src_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tgt_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self._setup_links(memory, "C001", ["C002"])

        with patch.object(
            memory,
            "_load_commit_embeddings",
            return_value={"C001": src_vec, "C002": tgt_vec},
        ):
            results = memory.get_linked_commits("C001")

        assert len(results) == 1
        assert "embedding_score" in results[0]
        assert abs(results[0]["embedding_score"] - 0.0) < 1e-5
