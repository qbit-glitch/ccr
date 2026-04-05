"""Tests for A5: ccr clean command.

Verifies that clean archives old commits and preserves rolling summary + recent commits.
"""
import os
import sys
import tempfile
import time
import unittest
from click.testing import CliRunner
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.cli import clean


def _make_commits_md(tmp_dir: str, old_dates: list[str], recent_dates: list[str]) -> str:
    """Write a minimal commits.md with old and recent commit blocks."""
    ccr_dir = os.path.join(tmp_dir, ".ccr")
    branch_dir = os.path.join(ccr_dir, "branches", "main")
    os.makedirs(branch_dir, exist_ok=True)

    lines = ["## Rolling Summary\n\nSome project context.\n\n---\n\n"]
    commit_num = 1
    for d in old_dates:
        lines.append(f"## [C{commit_num:03d}] {d} 10:00 | branch:main | Old commit {commit_num}\n**What**: Old work\n**Why**: N/A\n**Files**: old.py\n**Score**: 0.70\n\n---\n\n")
        commit_num += 1
    for d in recent_dates:
        lines.append(f"## [C{commit_num:03d}] {d} 10:00 | branch:main | Recent commit {commit_num}\n**What**: Recent work\n**Why**: N/A\n**Files**: new.py\n**Score**: 0.80\n\n---\n\n")
        commit_num += 1

    commits_path = os.path.join(branch_dir, "commits.md")
    with open(commits_path, "w") as f:
        f.write("".join(lines))

    # Write metadata.yaml for MemoryManager
    meta_path = os.path.join(ccr_dir, "metadata.yaml")
    with open(meta_path, "w") as f:
        f.write("version: 1\nbranch: main\n")

    return tmp_dir


class TestCcrClean(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.runner = CliRunner()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _old_date(self, days_ago: int = 100) -> str:
        dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return dt.strftime("%Y-%m-%d")

    def _recent_date(self, days_ago: int = 5) -> str:
        dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return dt.strftime("%Y-%m-%d")

    def test_dry_run_shows_preview_no_changes(self):
        """--dry-run shows what would be archived without modifying files."""
        _make_commits_md(self.tmp, [self._old_date()], [self._recent_date()])
        result = self.runner.invoke(clean, [self.tmp, "--days", "30", "--dry-run"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("dry-run", result.output)
        # No archive file created
        self.assertFalse(os.path.isdir(os.path.join(self.tmp, ".ccr", "archive")))

    def test_old_commits_archived_recent_kept(self):
        """Old commits move to archive; recent commits stay in commits.md."""
        _make_commits_md(self.tmp, [self._old_date(100)], [self._recent_date(5)])
        result = self.runner.invoke(clean, [self.tmp, "--days", "30", "--yes"])
        self.assertEqual(result.exit_code, 0, result.output)

        # Archive directory created
        archive_dir = os.path.join(self.tmp, ".ccr", "archive")
        self.assertTrue(os.path.isdir(archive_dir))

        # commits.md still exists and contains recent commit
        commits_path = os.path.join(self.tmp, ".ccr", "branches", "main", "commits.md")
        with open(commits_path) as f:
            content = f.read()
        self.assertIn("Recent commit", content)
        self.assertNotIn("Old commit", content)

    def test_rolling_summary_preserved(self):
        """Rolling summary section must survive clean."""
        _make_commits_md(self.tmp, [self._old_date(100)], [self._recent_date(5)])
        result = self.runner.invoke(clean, [self.tmp, "--days", "30", "--yes"])
        self.assertEqual(result.exit_code, 0, result.output)

        commits_path = os.path.join(self.tmp, ".ccr", "branches", "main", "commits.md")
        with open(commits_path) as f:
            content = f.read()
        self.assertIn("Rolling Summary", content)
        self.assertIn("Some project context", content)

    def test_nothing_to_prune_graceful(self):
        """When all commits are recent, clean exits gracefully with no changes."""
        _make_commits_md(self.tmp, [], [self._recent_date(5)])
        result = self.runner.invoke(clean, [self.tmp, "--days", "30", "--yes"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("nothing to archive", result.output.lower())


if __name__ == "__main__":
    unittest.main()
