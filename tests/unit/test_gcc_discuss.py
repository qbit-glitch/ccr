"""Tests for B3: gcc_discuss and gcc_discussions.

Tests DiscussionsMixin add_discussion() and get_discussions() logic.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.core.memory_pkg.memory_discussions import DiscussionsMixin, _parse_discussion_blocks


class FakeMemory(DiscussionsMixin):
    """Minimal MemoryManager stub for discussions testing."""

    def __init__(self, ccr_root: str):
        self.ccr_root = ccr_root
        self._branch = "main"

    def get_active_branch(self) -> str:
        return self._branch


class TestAddDiscussion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "branches", "main"), exist_ok=True)
        self.mem = FakeMemory(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_d001_on_first_call(self):
        result = self.mem.add_discussion(
            topic="dataset preprocessing",
            hypothesis="TorchDataset is faster than pandas",
            alternatives_considered="pandas, HDF5",
            decision="TorchDataset",
            rationale="40% throughput gain in benchmark",
        )
        self.assertEqual(result["id"], "D001")
        self.assertIn("dataset preprocessing", result["topic"])

    def test_creates_d002_on_second_call(self):
        self.mem.add_discussion(
            topic="optimizer choice",
            hypothesis="AdamW is better",
            alternatives_considered="SGD",
            decision="AdamW",
            rationale="lower val_loss",
        )
        result = self.mem.add_discussion(
            topic="learning rate",
            hypothesis="1e-4 is optimal",
            alternatives_considered="1e-3, 1e-5",
            decision="1e-4",
            rationale="grid search results",
        )
        self.assertEqual(result["id"], "D002")

    def test_stored_in_discussions_md(self):
        self.mem.add_discussion(
            topic="loss function",
            hypothesis="CrossEntropy is best",
            alternatives_considered="MSE, focal loss",
            decision="CrossEntropy",
            rationale="standard for classification",
            uncertainty="May need focal for imbalanced classes",
            linked_commit="C041",
        )
        disc_path = os.path.join(self.tmp, "branches", "main", "discussions.md")
        self.assertTrue(os.path.isfile(disc_path))
        content = open(disc_path).read()
        self.assertIn("CrossEntropy", content)
        self.assertIn("Linked Commit**: C041", content)
        self.assertIn("Uncertainty", content)

    def test_no_uncertainty_no_field(self):
        self.mem.add_discussion(
            topic="batch size",
            hypothesis="32 is optimal",
            alternatives_considered="16, 64",
            decision="32",
            rationale="memory fits, speed ok",
        )
        disc_path = os.path.join(self.tmp, "branches", "main", "discussions.md")
        content = open(disc_path).read()
        self.assertNotIn("Uncertainty", content)


class TestGetDiscussions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "branches", "main"), exist_ok=True)
        self.mem = FakeMemory(self.tmp)
        # Add 3 discussions
        self.mem.add_discussion("optimizer choice", "AdamW best", "SGD", "AdamW", "lower loss")
        self.mem.add_discussion("dataset", "TorchDataset fast", "pandas", "TorchDataset", "benchmark")
        self.mem.add_discussion("loss function", "CrossEntropy", "MSE", "CrossEntropy", "standard")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_finds_matching(self):
        result = self.mem.get_discussions(search="AdamW")
        self.assertGreaterEqual(result["count"], 1)
        self.assertIn("AdamW", result["message"])

    def test_empty_project_returns_zero(self):
        tmp2 = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp2, "branches", "main"), exist_ok=True)
        mem2 = FakeMemory(tmp2)
        result = mem2.get_discussions()
        self.assertEqual(result["count"], 0)
        import shutil; shutil.rmtree(tmp2, ignore_errors=True)

    def test_get_all_returns_three(self):
        result = self.mem.get_discussions()
        self.assertEqual(result["count"], 3)

    def test_linked_commit_stored_correctly(self):
        self.mem.add_discussion(
            topic="with link",
            hypothesis="X",
            alternatives_considered="Y",
            decision="X",
            rationale="Z",
            linked_commit="C099",
        )
        disc_path = os.path.join(self.tmp, "branches", "main", "discussions.md")
        content = open(disc_path).read()
        self.assertIn("C099", content)


if __name__ == "__main__":
    unittest.main()
