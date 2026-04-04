"""FileIOMixin -- path helpers, git commit, log append, gitignore, file lock, read/write."""

from __future__ import annotations

import fcntl
import logging
import os
import re
import subprocess
import tempfile
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FileIOMixin:
    """Low-level file I/O methods for MemoryManager.

    Expects the composite class to provide:
        self.project_root: str
        self.ccr_root: str
        self.branches_dir: str
        self.config: CCRConfig  (with .log_max_lines)
        self._locks: dict[str, threading.Lock]
    """

    # --- Path helpers ---

    def _get_branch_dir(self, branch: str) -> str:
        # H3: Validate branch name to prevent path traversal
        if branch != "main" and not re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", branch):
            raise ValueError(f"Invalid branch name: {branch}")
        return os.path.join(self.branches_dir, branch)

    def _get_commits_path(self, branch: str) -> str:
        return os.path.join(self._get_branch_dir(branch), "commits.md")

    def _get_log_path(self, branch: str) -> str:
        return os.path.join(self._get_branch_dir(branch), "log.md")

    # --- Git ---

    def _git_commit(self, message: str) -> bool:
        """Create a git commit if in a git repo. Returns True if committed."""
        git_dir = os.path.join(self.project_root, ".git")
        if not os.path.isdir(git_dir):
            return False
        try:
            # Stage .ccr/ changes
            subprocess.run(
                ["git", "add", ".ccr/"],
                cwd=self.project_root, capture_output=True, timeout=10,
            )
            result = subprocess.run(
                ["git", "commit", "-m", message, "--allow-empty"],
                cwd=self.project_root, capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    # --- Log ---

    def _append_log(self, branch: str, line: str) -> None:
        path = self._get_log_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            lines = content.strip().split("\n") if content.strip() else []
            lines.append(line)
            # Rotate
            if len(lines) > self.config.log_max_lines:
                lines = lines[-200:]
            self._write_file_unlocked(path, "\n".join(lines) + "\n")

    # --- Gitignore ---

    def _add_to_gitignore(self, gitignore_path: str) -> None:
        content = ""
        if os.path.isfile(gitignore_path):
            content = self._read_file(gitignore_path) or ""
        if ".ccr/" not in content and ".ccr" not in content:
            with open(gitignore_path, "a") as f:
                f.write("\n# CCR context directory\n.ccr/\n")

    # --- File I/O (thread-safe + cross-process wrappers) ---

    @contextmanager
    def _file_lock(self, path: str):
        """Cross-process file lock using fcntl.flock.

        Creates a .lock file alongside the target file. Uses LOCK_EX
        (exclusive) to prevent concurrent writes from multiple MCP server
        instances sharing the same .ccr/ directory.
        """
        lock_path = path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
            # M5: Clean up .lock files
            try:
                os.unlink(lock_path)
            except OSError:
                pass  # Race condition: another process may have already removed it

    def _read_file(self, path: str) -> str:
        with self._locks[path], self._file_lock(path):
            return self._read_file_unlocked(path)

    def _read_file_unlocked(self, path: str) -> str:
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_file(self, path: str, content: str) -> None:
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def _write_file_unlocked(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Sanitize surrogates that can't be encoded in UTF-8
        content = content.encode("utf-8", errors="replace").decode("utf-8")
        # Atomic write: write to tmp file, fsync, then os.replace
        dir_name = os.path.dirname(path) or "."
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _write_if_missing(self, path: str, content: str) -> None:
        if not os.path.isfile(path):
            self._write_file(path, content)
