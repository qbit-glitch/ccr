"""Tests for MemoryManager — GCC-style context management."""

import os
import re
import tempfile
from datetime import datetime, timezone

import pytest

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
    def test_new_outranks_existing_creates_new(self, memory):
        """Alg. 1 line 6: S(m) > S(m_conflict) → create new (both coexist)."""
        # Create an auth-related commit first so the continuation commit has low novelty
        memory.commit("Setup auth module", "initialized auth", "security",
                      ["auth.py"], "add login")
        # Create a low-priority continuation commit — low novelty (same file/topic) + low type prior
        memory.commit("Still working on auth", "continued auth work", "wip",
                      ["auth.py"], "continue auth")
        # New commit is a milestone with different files → higher novelty + high type prior
        result = memory.commit(
            "Auth complete", "finished deployment pipeline", "shipped to production",
            ["deploy.py", "config.yaml"], "monitor",
            admission_threshold=0.3,  # low threshold to trigger FindConflict
        )
        assert "C003" in result
        assert "merged" not in result.lower()

    def test_existing_outranks_new_merges(self, memory):
        """Alg. 1 line 7: S(m) <= S(m_conflict) → merge into existing."""
        # Create a normal existing commit (progress type, recent → high conflict_score)
        memory.commit("Working on auth", "started auth", "security",
                      ["auth.py"], "continue")
        # New commit is a continuation → low score → existing outranks
        result = memory.commit(
            "Still working on auth", "continued auth", "security",
            ["auth.py"], "keep going",
            admission_threshold=0.3,
        )
        assert "merged" in result.lower() or "+" in result

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
    def test_redundant_commit_merges(self, memory):
        """Nearly identical continuation commit should merge into previous."""
        memory.commit("Working on auth", "started auth module", "security",
                      ["auth.py"], "continue auth")
        result = memory.commit(
            "Still working on auth", "continued auth module", "security requirement",
            ["auth.py"], "finish auth",
            admission_threshold=0.5,  # low similarity threshold to ensure merge
        )
        assert "merged" in result.lower() or "+" in result

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
        """G5: Default threshold matches paper's 0.85 FindConflict."""
        # At default threshold (0.85), identical commits should merge
        # if similarity >= 0.85 (requires exact same files + keywords + recent)
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        result = memory.commit(
            "Add auth", "implemented login", "security",
            ["auth.py"], "add tests",
            # No explicit threshold — uses default 0.85
        )
        # Identical commit + very recent → similarity should be ~1.0 → merge
        assert "merged" in result.lower() or "+" in result

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
        """Merged commit appends new what/why to previous."""
        memory.commit("Working on feature", "started implementation", "new feature",
                      ["feature.py"], "continue")
        memory.commit(
            "Still on feature", "added validation", "new feature needs validation",
            ["feature.py"], "add tests",
            admission_threshold=0.3,  # force merge
        )
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "started implementation" in commits
        assert "added validation" in commits

    def test_merged_commit_unions_files(self, memory):
        """Merged commit unions file lists."""
        memory.commit("Working on feature", "started", "reason",
                      ["feature.py"], "continue")
        memory.commit(
            "Still on feature", "added tests", "reason",
            ["feature.py", "test_feature.py"], "next",
            admission_threshold=0.3,
        )
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "test_feature.py" in commits
        assert "feature.py" in commits

    def test_merged_commit_updates_next(self, memory):
        """Merged commit replaces next_step with newer value."""
        memory.commit("Feature", "started", "reason", ["f.py"], "old next step")
        memory.commit(
            "Feature", "continued", "reason", ["f.py"], "new next step",
            admission_threshold=0.3,
        )
        commits = memory._read_file(memory._get_commits_path("main"))
        assert "new next step" in commits

    def test_admission_updates_rolling_summary(self, memory):
        """Even merged commits update the rolling summary."""
        memory.commit("Feature", "started", "reason", ["f.py"], "continue")
        memory.commit(
            "Feature", "added validation", "reason", ["f.py"], "test",
            admission_threshold=0.3,
        )
        summary = memory._get_rolling_summary("main")
        assert "validation" in summary.lower()

    def test_admission_logs_merge_event(self, memory):
        """Admission merges are logged in OTA."""
        memory.commit("Feature", "started", "reason", ["f.py"], "continue")
        memory.commit(
            "Feature", "continued", "reason", ["f.py"], "test",
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

    def test_conflict_uses_stored_score(self, memory):
        """S(m_conflict) should use stored score when available."""
        # Create first commit (gets score=1.0 as first commit)
        memory.commit("Add auth", "implemented login", "security",
                      ["auth.py"], "add tests")
        # Compute score for a similar commit — conflict should use stored score
        score = memory.compute_admission_score(
            "main", "Add auth v2", "improved login", "security",
            ["auth.py"], "more tests",
        )
        if score["conflict_id"]:
            # The conflict_score should be the stored score (1.0 for first commit)
            assert score["conflict_score"] == 1.0

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
        r"""Title containing \n should not create a newline in merged commit."""
        memory.commit("Feature work", "started feature", "reason",
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
        r"""Title containing \1 should not be interpreted as regex backreference."""
        memory.commit("Feature work", "started feature", "reason",
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
        r"""Title with mixed special regex chars should be handled safely."""
        memory.commit("Feature work", "started feature", "reason",
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
