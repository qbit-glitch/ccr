"""Tests for A2: empty-project first-session UX.

When a project has no commits, _handle_session_start should print
<ccr_ready> and return early — no MANDATORY_CCR_ACTIONS injection.
"""
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ccr.hooks.on_session_start import _handle_session_start, _project_has_commits


class FakeMemNoCommits:
    """Minimal MemoryManager stub with no commits."""

    def __init__(self, ccr_root: str):
        self.ccr_root = ccr_root

    def get_active_branch(self) -> str:
        return "main"

    def _read_commits_window(self, branch: str, start: int, count: int) -> str:
        # No commits — returns template-only content without [C###] markers
        return "# CCR Memory\n\n## Summary\n\nNo commits yet.\n"

    def get_context(self, level: int = 1) -> str:
        return ""

    def log_ota(self, **kwargs):
        pass


class FakeMemWithCommits:
    """Minimal MemoryManager stub with one commit."""

    def __init__(self, ccr_root: str):
        self.ccr_root = ccr_root

    def get_active_branch(self) -> str:
        return "main"

    def _read_commits_window(self, branch: str, start: int, count: int) -> str:
        return "## [C001] 2026-04-05 10:00 | First commit\n**What**: Initial setup\n"

    def get_context(self, level: int = 1) -> str:
        return "Project: My Thesis"

    def log_ota(self, **kwargs):
        pass


class TestProjectHasCommits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_commits_returns_false(self):
        mem = FakeMemNoCommits(self.tmp)
        self.assertFalse(_project_has_commits(mem))

    def test_one_commit_returns_true(self):
        mem = FakeMemWithCommits(self.tmp)
        self.assertTrue(_project_has_commits(mem))

    def test_template_only_no_c_markers_returns_false(self):
        """Content with no [C###] markers is treated as empty."""

        class FakeMemTemplateOnly:
            ccr_root = self.tmp

            def get_active_branch(self):
                return "main"

            def _read_commits_window(self, *_):
                return "# CCR\n\n## Section\nSome text without commit markers.\n"

        self.assertFalse(_project_has_commits(FakeMemTemplateOnly()))


class TestHandleSessionStartEmptyProject(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_session_start(self, mem):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch("ccr.hooks.on_session_start._create_db_session"):
                with patch("ccr.hooks.on_session_start._buffer_user_prompt"):
                    _handle_session_start(mem, project_root=self.tmp)
        return buf.getvalue()

    def test_empty_project_prints_ccr_ready(self):
        mem = FakeMemNoCommits(self.tmp)
        output = self._run_session_start(mem)
        self.assertIn("<ccr_ready>", output)
        self.assertNotIn("MANDATORY_CCR_ACTIONS", output)

    def test_project_with_commits_prints_mandatory_actions(self):
        mem = FakeMemWithCommits(self.tmp)
        output = self._run_session_start(mem)
        self.assertIn("MANDATORY_CCR_ACTIONS", output)
        self.assertNotIn("<ccr_ready>", output)

    def test_empty_project_ccr_ready_mentions_gcc_commit(self):
        """<ccr_ready> message must mention gcc_commit so users know what to do."""
        mem = FakeMemNoCommits(self.tmp)
        output = self._run_session_start(mem)
        self.assertIn("gcc_commit", output)

    def test_codex_project_with_commits_prints_compact_summary(self):
        class FakeMemCodex(FakeMemWithCommits):
            def get_context(self, level: int = 1) -> str:
                return (
                    "## Current Focus\n"
                    "Codex CCR quiet lifecycle policy. Next: restart sessions.\n\n"
                    "## Recent Milestones\n"
                    "- [2026-04-26 16:31] (main) Codex CCR quiet lifecycle policy\n\n"
                    "## Recent Commits\n"
                    "## [C124] 2026-04-26 16:31 | branch:main | Codex CCR quiet lifecycle policy\n"
                    "**What**: This raw context should not be printed.\n"
                )

        mem = FakeMemCodex(self.tmp)
        with patch.dict(os.environ, {"CCR_HOOK_AGENT": "codex"}):
            output = self._run_session_start(mem)

        self.assertLessEqual(len(output.strip().splitlines()), 2)
        self.assertIn("CCR retrieved full memory", output)
        self.assertIn("Codex CCR quiet lifecycle policy", output)
        self.assertNotIn("<gcc_context>", output)
        self.assertNotIn("MANDATORY_CCR_ACTIONS", output)
        self.assertNotIn("This raw context should not be printed", output)

    def test_codex_project_with_noisy_auto_focus_suppresses_it(self):
        class FakeMemNoisyCodex(FakeMemWithCommits):
            def get_context(self, level: int = 1) -> str:
                return (
                    "## Current Focus\n"
                    "[auto] status ??. Next:\n\n"
                    "## Recent Milestones\n"
                    "- [2026-05-26 06:09] (main) [auto] status ??\n"
                    "- [2026-05-26 05:49] (main) Useful milestone\n\n"
                    "## Recent Commits\n"
                    "## [C124] 2026-04-26 16:31 | branch:main | [auto] status ??\n"
                )

        mem = FakeMemNoisyCodex(self.tmp)
        with patch.dict(os.environ, {"CCR_HOOK_AGENT": "codex"}):
            output = self._run_session_start(mem)

        self.assertIn("CCR retrieved full memory", output)
        self.assertIn("Useful milestone", output)
        self.assertNotIn("[auto] status", output)
        self.assertNotIn("Next:Use", output.replace("\n", ""))


if __name__ == "__main__":
    unittest.main()
