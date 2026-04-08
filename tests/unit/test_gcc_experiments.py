"""Tests for B1: gcc_experiments query tool.

Tests ExperimentsMixin.get_experiments() parsing and filtering logic.
"""
import os
import sys
import tempfile
import unittest

import pytest
import ccr.mcp.server as _srv_mod
import ccr.mcp_server as mcp_mod
from ccr.mcp_server import _init
from ccr.mcp.gcc_search_tools import gcc_experiments

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.core.memory_pkg.memory_experiments import (
    ExperimentsMixin,
    _parse_commit_blocks,
    _apply_metric_filter,
    _apply_top_n,
)


# ---------------------------------------------------------------------------
# Sample commits.md content with experiment blocks
# ---------------------------------------------------------------------------

SAMPLE_COMMITS = """
## [C041] 2026-04-01 10:00 | branch:main | LoRA baseline
**What**: Ran LoRA baseline experiment
**Why**: Establish baseline
**Files**: train.py
**Experiment**:
  - ID: exp-001
  - Hypothesis: LoRA r=16 matches full fine-tuning
  - Metrics: val_loss=0.31, accuracy=0.82
  - Conclusion: Underperformed — 95% accuracy gap
**Score**: 0.75

---

## [C042] 2026-04-02 14:30 | branch:main | LoRA improved
**What**: LoRA with larger rank
**Why**: Improve accuracy
**Files**: train.py, configs/lora.yaml
**Experiment**:
  - ID: exp-002
  - Hypothesis: LoRA r=32 closes the gap
  - Metrics: val_loss=0.21, accuracy=0.88
  - Conclusion: Confirmed — 98% perf at 22% params
**Score**: 0.82

---

## [C043] 2026-04-03 09:15 | branch:main | Ablation: no dropout
**What**: Ablation study — removing dropout
**Why**: Diagnose variance
**Files**: train.py
**Experiment**:
  - ID: exp-003
  - Hypothesis: Dropout hurts convergence
  - Metrics: val_loss=0.19, accuracy=0.90
  - Conclusion: Confirmed — dropout was the bottleneck
**Score**: 0.88

---

## [C044] 2026-04-04 11:00 | branch:main | Regular code refactor
**What**: Refactored data pipeline
**Why**: Code cleanup
**Files**: data.py
**Score**: 0.70

---
"""


class TestParseCommitBlocks(unittest.TestCase):
    def test_finds_three_experiment_commits(self):
        """Three commits have experiment blocks; one plain commit is excluded."""
        records = _parse_commit_blocks(SAMPLE_COMMITS)
        self.assertEqual(len(records), 3)

    def test_parses_experiment_id(self):
        records = _parse_commit_blocks(SAMPLE_COMMITS)
        ids = [r["experiment"].get("id") for r in records]
        self.assertIn("exp-001", ids)
        self.assertIn("exp-002", ids)

    def test_parses_metrics_as_dict(self):
        records = _parse_commit_blocks(SAMPLE_COMMITS)
        r = next(r for r in records if r["experiment"].get("id") == "exp-002")
        self.assertAlmostEqual(float(r["experiment"]["metrics"]["val_loss"]), 0.21)
        self.assertAlmostEqual(float(r["experiment"]["metrics"]["accuracy"]), 0.88)

    def test_commit_without_experiment_excluded(self):
        """C044 (no **Experiment**: block) must not appear in results."""
        records = _parse_commit_blocks(SAMPLE_COMMITS)
        commit_ids = [r["commit_id"] for r in records]
        self.assertNotIn("C044", commit_ids)


class TestApplyMetricFilter(unittest.TestCase):
    def _records(self):
        return _parse_commit_blocks(SAMPLE_COMMITS)

    def test_filter_val_loss_lt(self):
        """val_loss < 0.25 should keep exp-002 and exp-003 only."""
        records = self._records()
        filtered = _apply_metric_filter(records, {"val_loss": {"lt": 0.25}})
        ids = [r["experiment"].get("id") for r in filtered]
        self.assertNotIn("exp-001", ids)
        self.assertIn("exp-002", ids)
        self.assertIn("exp-003", ids)

    def test_filter_accuracy_gte(self):
        """accuracy >= 0.89 keeps only exp-003."""
        records = self._records()
        filtered = _apply_metric_filter(records, {"accuracy": {"gte": 0.89}})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["experiment"]["id"], "exp-003")

    def test_no_match_returns_empty(self):
        records = self._records()
        filtered = _apply_metric_filter(records, {"val_loss": {"lt": 0.01}})
        self.assertEqual(len(filtered), 0)


class TestApplyTopN(unittest.TestCase):
    def _records(self):
        return _parse_commit_blocks(SAMPLE_COMMITS)

    def test_sort_val_loss_asc(self):
        """Ascending sort: lowest val_loss first."""
        records = self._records()
        sorted_recs = _apply_top_n(records, "val_loss:asc")
        losses = [r["experiment"]["metrics"]["val_loss"] for r in sorted_recs]
        self.assertEqual(losses, sorted(losses))

    def test_sort_accuracy_desc(self):
        """Descending sort: highest accuracy first."""
        records = self._records()
        sorted_recs = _apply_top_n(records, "accuracy:desc")
        accs = [r["experiment"]["metrics"]["accuracy"] for r in sorted_recs]
        self.assertEqual(accs, sorted(accs, reverse=True))


class TestExperimentsMixinIntegration(unittest.TestCase):
    """Integration test using ExperimentsMixin with a real (stubbed) MemoryManager."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_mixin(self, commits_text: str):
        """Create a minimal ExperimentsMixin instance with injected commits text."""

        class FakeMemory(ExperimentsMixin):
            def __init__(self_inner):
                pass

            def get_active_branch(self_inner):
                return "main"

            def _read_commits_window(self_inner, branch, start, count):
                return commits_text

        return FakeMemory()

    def test_single_result_returned(self):
        mem = self._make_mixin(SAMPLE_COMMITS)
        result = mem.get_experiments(experiment_id="exp-001")
        self.assertEqual(result["count"], 1)
        self.assertIn("exp-001", result["message"])

    def test_no_experiments_returns_zero_count(self):
        mem = self._make_mixin("## [C001] 2026-04-01 10:00 | branch:main | Plain commit\n**What**: nothing\n\n---\n")
        result = mem.get_experiments()
        self.assertEqual(result["count"], 0)
        self.assertIn("No experiments found", result["message"])

    def test_compare_mode_returns_comparison_table(self):
        mem = self._make_mixin(SAMPLE_COMMITS)
        result = mem.get_experiments(compare=["C041", "C042"])
        self.assertIn("Comparison", result["message"])
        self.assertIn("exp-001", result["message"])
        self.assertIn("exp-002", result["message"])

    def test_hypothesis_contains_filter(self):
        mem = self._make_mixin(SAMPLE_COMMITS)
        result = mem.get_experiments(hypothesis_contains="LoRA")
        self.assertEqual(result["count"], 2)  # exp-001 and exp-002 mention LoRA

    def test_metric_filter_via_get_experiments(self):
        mem = self._make_mixin(SAMPLE_COMMITS)
        result = mem.get_experiments(metric_filter={"val_loss": {"lt": 0.25}})
        self.assertEqual(result["count"], 2)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# pytest fixture + MCP-layer integration tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def setup_project(tmp_path, monkeypatch):
    """Initialize CCR in a temp directory."""
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


class TestGccExperimentsToolLayer:
    """Integration tests for the gcc_experiments MCP tool (not just the mixin)."""

    def test_returns_empty_on_fresh_project(self, setup_project):
        """gcc_experiments returns count=0 on a project with no commits."""
        fn = getattr(gcc_experiments, "__wrapped__", gcc_experiments)
        result = fn()
        assert result["count"] == 0
        assert result["records"] == []

    def test_ensure_memory_called_with_state_lock(self, setup_project, monkeypatch):
        """gcc_experiments must call _ensure_memory() while holding _state_lock."""
        lock_was_held = []
        original_ensure = _srv_mod._ensure_memory

        def spy_ensure():
            lock_was_held.append(_srv_mod._state_lock.locked())
            return original_ensure()

        monkeypatch.setattr(_srv_mod, "_ensure_memory", spy_ensure)
        fn = getattr(gcc_experiments, "__wrapped__", gcc_experiments)
        fn()
        assert lock_was_held, "_ensure_memory spy was never called"
        assert lock_was_held[0], "_ensure_memory() was called WITHOUT _state_lock held"
