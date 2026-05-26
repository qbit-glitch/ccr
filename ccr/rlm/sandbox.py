"""Kernel-level sandbox for REPL execution.

Wraps Python subprocess execution in OS-level sandboxing:
- macOS: sandbox-exec with Seatbelt profiles (deny-default, allow-list)
- Linux: Landlock (Linux 5.13+, via ctypes syscalls)
- Fallback: Python-level restrictions only (with warning)

The kernel sandbox is defense-in-depth on top of the existing
Python-level AST validation and restricted builtins.

Performance note: Subprocess spawning adds ~100-300ms overhead per
execution. This is acceptable for the RLM use case (iterative
exploration), but is NOT zero overhead.

Composes sub-modules:
- sandbox_platform: OS detection + Landlock enforcement
- sandbox_seatbelt: macOS Seatbelt profile + subprocess runner script
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

# Re-export platform detection and Seatbelt generation for backward compat
from ccr.rlm.sandbox_platform import (  # noqa: F401
    LANDLOCK_ACCESS_FS_EXECUTE,
    LANDLOCK_ACCESS_FS_MAKE_BLOCK,
    LANDLOCK_ACCESS_FS_MAKE_CHAR,
    LANDLOCK_ACCESS_FS_MAKE_DIR,
    LANDLOCK_ACCESS_FS_MAKE_FIFO,
    LANDLOCK_ACCESS_FS_MAKE_IPC,
    LANDLOCK_ACCESS_FS_MAKE_REG,
    LANDLOCK_ACCESS_FS_MAKE_SOCK,
    LANDLOCK_ACCESS_FS_MAKE_SYM,
    LANDLOCK_ACCESS_FS_READ_DIR,
    LANDLOCK_ACCESS_FS_READ_FILE,
    LANDLOCK_ACCESS_FS_REMOVE_DIR,
    LANDLOCK_ACCESS_FS_REMOVE_FILE,
    LANDLOCK_ACCESS_FS_WRITE_FILE,
    LANDLOCK_RULE_PATH_BENEATH,
    _expand,
    _landlock_probe,
    _landlock_syscall_nr,
    apply_landlock_restrictions,
    get_sandbox_type,
    is_landlock_available,
    is_seatbelt_available,
)
from ccr.rlm.sandbox_seatbelt import (  # noqa: F401
    _RUNNER_SCRIPT,
    _SENSITIVE_DIRS,
    _SENSITIVE_FILES,
    _get_python_exec_paths,
    _get_python_read_paths,
    generate_seatbelt_profile,
)

logger = logging.getLogger(__name__)

_SEATBELT_USABLE: bool | None = None


# ---------------------------------------------------------------------------
# Sandbox executor
# ---------------------------------------------------------------------------

class SandboxResult:
    """Result from sandboxed execution."""

    __slots__ = ("stdout", "stderr", "error", "variables", "sandbox_type", "dropped_vars")

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        error: str | None = None,
        variables: dict[str, Any] | None = None,
        sandbox_type: str = "none",
        dropped_vars: list[str] | None = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.variables = variables or {}
        self.sandbox_type = sandbox_type
        self.dropped_vars = dropped_vars or []

    def __repr__(self) -> str:
        return (
            f"SandboxResult(sandbox_type={self.sandbox_type!r}, "
            f"error={self.error!r}, stdout_len={len(self.stdout)})"
        )


class KernelSandbox:
    """Kernel-level sandbox for executing Python code.

    Uses macOS Seatbelt (sandbox-exec) or Linux Landlock for OS-level
    enforcement, with graceful fallback to subprocess-only isolation.
    """

    def __init__(
        self,
        project_root: str | None = None,
        timeout_seconds: float = 30.0,
        python_executable: str | None = None,
    ):
        self.project_root = os.path.realpath(project_root) if project_root else None
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable
        self.sandbox_type = get_sandbox_type()
        self.temp_dir = tempfile.mkdtemp(prefix="ccr_sandbox_")
        self._profile_path: str | None = None

        if self.sandbox_type == "seatbelt" and self.project_root:
            self._generate_profile()
            if not self._seatbelt_runtime_usable():
                logger.debug(
                    "macOS sandbox-exec is present but blocked by this host. "
                    "Falling back to subprocess-only isolation."
                )
                self.sandbox_type = "none"
        elif self.sandbox_type == "none":
            logger.warning(
                "No kernel sandbox available on this platform. "
                "Falling back to subprocess-only isolation."
            )

    def _generate_profile(self) -> None:
        """Generate and write the Seatbelt profile to a temp file.

        Profile file is written with 0o600 permissions (owner read/write only)
        to prevent other processes from reading or modifying the sandbox rules.
        """
        profile = generate_seatbelt_profile(
            project_root=self.project_root or self.temp_dir,
            temp_dir=self.temp_dir,
            python_executable=self.python_executable,
        )
        self._profile_path = os.path.join(self.temp_dir, "sandbox.sb")
        fd = os.open(self._profile_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, profile.encode("utf-8"))
        finally:
            os.close(fd)

    def _seatbelt_runtime_usable(self) -> bool:
        """Return whether sandbox-exec can actually apply profiles here.

        Some macOS hosts expose sandbox-exec but reject sandbox_apply with
        "Operation not permitted" (exit 71), especially inside nested/sandboxed
        dev environments. Treat that as unavailable so normal REPL execution
        remains functional and tests can still exercise the subprocess fallback.
        """
        global _SEATBELT_USABLE

        if _SEATBELT_USABLE is not None:
            return _SEATBELT_USABLE
        if not self._profile_path:
            _SEATBELT_USABLE = False
            return False

        python = os.path.realpath(self.python_executable)
        try:
            proc = subprocess.run(
                [
                    "sandbox-exec",
                    "-f",
                    self._profile_path,
                    python,
                    "-c",
                    "print('ccr-seatbelt-ok')",
                ],
                capture_output=True,
                text=True,
                timeout=min(max(self.timeout_seconds, 1.0), 5.0),
                env=self._build_env(),
                cwd=self.temp_dir,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            _SEATBELT_USABLE = False
            return False

        _SEATBELT_USABLE = proc.returncode == 0
        if not _SEATBELT_USABLE:
            logger.debug(
                "sandbox-exec probe failed with code %s: %s",
                proc.returncode,
                (proc.stderr or proc.stdout or "")[:300],
            )
        return _SEATBELT_USABLE

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Execute code in a kernel-sandboxed subprocess.

        Args:
            code: Python code to execute.
            variables: Variables to inject (must be JSON-serializable).

        Returns:
            SandboxResult with stdout, stderr, error, and extracted variables.
        """
        # Serialize variables (only JSON-safe ones), log dropped ones
        safe_vars: dict[str, Any] = {}
        dropped_input_vars: list[str] = []
        if variables:
            for k, v in variables.items():
                try:
                    json.dumps(v)
                    safe_vars[k] = v
                except (TypeError, ValueError, OverflowError):
                    dropped_input_vars.append(k)
        if dropped_input_vars:
            logger.warning(
                "Non-serializable variables dropped from sandbox input: %s",
                dropped_input_vars,
            )

        payload = json.dumps({
            "code": code,
            "variables": safe_vars,
            "project_root": self.project_root,
            "temp_dir": self.temp_dir,
            "sandbox_type": self.sandbox_type,
        })

        # Build command
        cmd = self._build_command()

        # Minimal environment
        env = self._build_env()

        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
                cwd=self.temp_dir,
            )

            # Parse result from stdout
            if proc.returncode != 0 and not proc.stdout.strip():
                return SandboxResult(
                    stdout="",
                    stderr=proc.stderr,
                    error=f"Subprocess exited with code {proc.returncode}: {proc.stderr[:500]}",
                    sandbox_type=self.sandbox_type,
                )

            try:
                result = json.loads(proc.stdout)
                # Log any variables that were dropped in the subprocess
                dropped_output = result.get("dropped_vars", [])
                if dropped_output:
                    logger.warning(
                        "Non-serializable variables in sandbox output: %s",
                        dropped_output,
                    )
                return SandboxResult(
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", "") + (proc.stderr or ""),
                    error=result.get("error"),
                    variables=result.get("variables", {}),
                    sandbox_type=self.sandbox_type,
                    dropped_vars=dropped_input_vars + dropped_output,
                )
            except json.JSONDecodeError:
                return SandboxResult(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    error="Failed to parse sandbox result",
                    sandbox_type=self.sandbox_type,
                )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                stderr=f"Execution exceeded {self.timeout_seconds}s limit",
                error=f"TimeoutError: Execution exceeded {self.timeout_seconds}s limit",
                sandbox_type=self.sandbox_type,
            )
        except FileNotFoundError as e:
            return SandboxResult(
                error=f"Sandbox executable not found: {e}",
                sandbox_type="none",
            )
        except OSError as e:
            return SandboxResult(
                error=f"Sandbox execution failed: {e}",
                sandbox_type=self.sandbox_type,
            )

    def _build_command(self) -> list[str]:
        """Build the subprocess command with appropriate sandboxing."""
        python = os.path.realpath(self.python_executable)

        if self.sandbox_type == "seatbelt" and self._profile_path:
            return [
                "sandbox-exec",
                "-f", self._profile_path,
                python, "-c", _RUNNER_SCRIPT,
            ]
        # For landlock or no sandbox, just use subprocess isolation
        return [python, "-c", _RUNNER_SCRIPT]

    def _build_env(self) -> dict[str, str]:
        """Build minimal environment for the subprocess.

        Only passes PATH, TMPDIR, and PYTHONDONTWRITEBYTECODE.
        HOME is set to the temp dir (not the real home, which would expose
        dotfiles). PYTHONPATH is excluded to prevent module injection.
        """
        env: dict[str, str] = {}

        # PATH: needed for Python to find shared libraries
        path = os.environ.get("PATH")
        if path:
            env["PATH"] = path

        # TMPDIR: needed for tempfile module
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir:
            env["TMPDIR"] = tmpdir

        # Set HOME to sandbox temp dir, not real home
        env["HOME"] = self.temp_dir

        # Prevent .pyc file creation in project dir
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        # PYTHONPATH intentionally excluded — prevents module injection

        return env

    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()
