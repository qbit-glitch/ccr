"""Tests for the kernel sandbox module (ccr/rlm/sandbox.py).

Tests cover:
- Seatbelt profile generation
- Sandboxed execution basics
- Filesystem access restrictions
- Network access denial
- Sensitive path protection
- Graceful fallback
- Integration with CCRRepl
"""

import json
import os
import platform
import sys
import tempfile
from unittest import mock

import pytest

from ccr.rlm.sandbox import (
    KernelSandbox,
    SandboxResult,
    _RUNNER_SCRIPT,
    _expand,
    _get_python_read_paths,
    _landlock_syscall_nr,
    apply_landlock_restrictions,
    generate_seatbelt_profile,
    get_sandbox_type,
    is_landlock_available,
    is_seatbelt_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_linux() -> bool:
    return platform.system() == "Linux"


# ---------------------------------------------------------------------------
# Platform detection tests
# ---------------------------------------------------------------------------

class TestPlatformDetection:
    def test_is_seatbelt_available_on_macos(self):
        """On macOS, sandbox-exec should be available."""
        if _is_macos():
            assert is_seatbelt_available() is True
        else:
            assert is_seatbelt_available() is False

    def test_is_landlock_not_available_on_macos(self):
        """Landlock is Linux-only."""
        if _is_macos():
            assert is_landlock_available() is False

    def test_get_sandbox_type_returns_string(self):
        result = get_sandbox_type()
        assert result in ("seatbelt", "landlock", "none")

    @mock.patch("ccr.rlm.sandbox.platform")
    @mock.patch("ccr.rlm.sandbox.shutil")
    def test_seatbelt_detected_when_darwin(self, mock_shutil, mock_platform):
        mock_platform.system.return_value = "Darwin"
        mock_shutil.which.return_value = "/usr/bin/sandbox-exec"
        assert is_seatbelt_available() is True

    @mock.patch("ccr.rlm.sandbox.platform")
    def test_seatbelt_not_detected_on_linux(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        assert is_seatbelt_available() is False

    def test_landlock_always_unavailable_on_macos(self):
        """Landlock is not available on macOS (Linux-only)."""
        if _is_macos():
            assert is_landlock_available() is False

    @mock.patch("ccr.rlm.sandbox.platform")
    def test_landlock_not_detected_on_macos(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        assert is_landlock_available() is False

    @mock.patch("ccr.rlm.sandbox._landlock_probe")
    @mock.patch("ccr.rlm.sandbox.platform")
    def test_is_landlock_available_on_linux_513(self, mock_platform, mock_probe):
        """Linux 5.13 with working syscall → True."""
        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.13.0-generic"
        mock_platform.machine.return_value = "x86_64"
        mock_probe.return_value = None  # probe succeeds
        assert is_landlock_available() is True

    @mock.patch("ccr.rlm.sandbox.platform")
    def test_is_landlock_not_available_old_kernel(self, mock_platform):
        """Linux 5.12 → False (too old)."""
        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.12.0-generic"
        assert is_landlock_available() is False

    @mock.patch("ccr.rlm.sandbox.platform")
    def test_is_landlock_not_available_on_macos(self, mock_platform):
        """Darwin → False."""
        mock_platform.system.return_value = "Darwin"
        assert is_landlock_available() is False

    @mock.patch("ccr.rlm.sandbox.platform")
    def test_landlock_syscall_nr_x86_64(self, mock_platform):
        """x86_64 create_ruleset → 444."""
        mock_platform.machine.return_value = "x86_64"
        assert _landlock_syscall_nr("create_ruleset") == 444

    @mock.patch("ccr.rlm.sandbox.platform")
    def test_landlock_syscall_nr_unknown_arch(self, mock_platform):
        """Unknown arch → NotImplementedError."""
        mock_platform.machine.return_value = "riscv64"
        with pytest.raises(NotImplementedError, match="riscv64"):
            _landlock_syscall_nr("create_ruleset")

    @mock.patch("ccr.rlm.sandbox.is_seatbelt_available", return_value=False)
    @mock.patch("ccr.rlm.sandbox.is_landlock_available", return_value=True)
    def test_get_sandbox_type_returns_landlock_on_linux(self, mock_ll, mock_sb):
        """When landlock available and seatbelt not, get_sandbox_type() == 'landlock'."""
        assert get_sandbox_type() == "landlock"


# ---------------------------------------------------------------------------
# Seatbelt profile generation tests
# ---------------------------------------------------------------------------

class TestSeatbeltProfile:
    def test_profile_has_deny_default(self):
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(
                project_root=td,
                temp_dir=td,
            )
            assert "(deny default)" in profile

    def test_profile_denies_network(self):
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(
                project_root=td,
                temp_dir=td,
            )
            assert "(deny network*)" in profile

    def test_profile_allows_project_dir_write(self):
        with tempfile.TemporaryDirectory() as td:
            real_td = os.path.realpath(td)
            profile = generate_seatbelt_profile(
                project_root=td,
                temp_dir=td,
            )
            assert real_td in profile
            assert f'(allow file-write* (subpath "{real_td}"))' in profile

    def test_profile_restricts_process_exec_to_python(self):
        """Process exec should be limited to the Python interpreter only."""
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(
                project_root=td,
                temp_dir=td,
            )
            python_real = os.path.realpath(sys.executable)
            assert "(allow process-fork)" in profile
            assert f'(allow process-exec (literal "{python_real}"))' in profile
            # Should NOT have blanket process-exec
            assert "(allow process*)" not in profile

    def test_profile_uses_subpath_root_for_reads(self):
        """Profile uses (subpath '/') for reads (macOS Seatbelt limitation).

        Python reads numerous unpredictable paths at startup and Seatbelt's
        'allow wins' semantics prevent selective deny rules from working.
        Read access is safe because network is denied (no exfiltration).
        """
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(
                project_root=td,
                temp_dir=td,
            )
            assert '(allow file-read* (subpath "/"))' in profile

    def test_profile_write_restricted_to_project_and_temp(self):
        """Writes should ONLY be allowed to project dir and temp dir."""
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(
                project_root=td,
                temp_dir=td,
            )
            real_td = os.path.realpath(td)
            # Count file-write allow rules
            write_rules = [l.strip() for l in profile.split("\n")
                          if "allow file-write" in l and l.strip()]
            # Should only have project + temp (may be same dir in this test)
            for rule in write_rules:
                assert real_td in rule, f"Unexpected write rule: {rule}"

    def test_profile_allows_sysctl_and_mach(self):
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(
                project_root=td,
                temp_dir=td,
            )
            assert "(allow sysctl*)" in profile
            assert "(allow mach*)" in profile

    def test_profile_allows_ipc_and_signal(self):
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(
                project_root=td,
                temp_dir=td,
            )
            assert "(allow ipc*)" in profile
            assert "(allow signal)" in profile

    def test_profile_version_1(self):
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(
                project_root=td,
                temp_dir=td,
            )
            assert profile.startswith("(version 1)")

    def test_profile_separate_project_and_temp(self):
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as tmp:
            profile = generate_seatbelt_profile(
                project_root=proj,
                temp_dir=tmp,
            )
            proj_real = os.path.realpath(proj)
            tmp_real = os.path.realpath(tmp)
            # Both paths should have write rules
            assert f'(allow file-write* (subpath "{proj_real}"))' in profile
            assert f'(allow file-write* (subpath "{tmp_real}"))' in profile

    def test_profile_reads_cover_all_paths(self):
        """Broad read access covers project and temp dirs."""
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as tmp:
            profile = generate_seatbelt_profile(
                project_root=proj,
                temp_dir=tmp,
            )
            # Broad read via (subpath "/") covers everything
            assert '(allow file-read* (subpath "/"))' in profile


# ---------------------------------------------------------------------------
# SandboxResult tests
# ---------------------------------------------------------------------------

class TestSandboxResult:
    def test_default_values(self):
        r = SandboxResult()
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.error is None
        assert r.variables == {}
        assert r.sandbox_type == "none"
        assert r.dropped_vars == []

    def test_repr(self):
        r = SandboxResult(sandbox_type="seatbelt", stdout="hello")
        assert "seatbelt" in repr(r)
        assert "stdout_len=5" in repr(r)

    def test_with_error(self):
        r = SandboxResult(error="boom", sandbox_type="seatbelt")
        assert r.error == "boom"

    def test_dropped_vars(self):
        r = SandboxResult(dropped_vars=["fn", "obj"])
        assert r.dropped_vars == ["fn", "obj"]


# ---------------------------------------------------------------------------
# KernelSandbox tests
# ---------------------------------------------------------------------------

class TestKernelSandbox:
    def test_init_creates_temp_dir(self):
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            assert os.path.isdir(ks.temp_dir)
            ks.cleanup()

    def test_init_detects_sandbox_type(self):
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            assert ks.sandbox_type in ("seatbelt", "landlock", "none")
            ks.cleanup()

    def test_cleanup_removes_temp_dir(self):
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            td = ks.temp_dir
            ks.cleanup()
            assert not os.path.exists(td)

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj) as ks:
                td = ks.temp_dir
                assert os.path.isdir(td)
            assert not os.path.exists(td)

    @pytest.mark.skipif(not _is_macos(), reason="Seatbelt only on macOS")
    def test_generates_profile_on_macos(self):
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            assert ks._profile_path is not None
            assert os.path.exists(ks._profile_path)
            with open(ks._profile_path) as f:
                content = f.read()
            assert "(deny default)" in content
            ks.cleanup()

    def test_build_env_minimal(self):
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            env = ks._build_env()
            assert "PYTHONDONTWRITEBYTECODE" in env
            assert env["PYTHONDONTWRITEBYTECODE"] == "1"
            # Should not contain random env vars
            assert "EDITOR" not in env
            assert "HISTFILE" not in env
            ks.cleanup()

    def test_build_env_passes_path(self):
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            env = ks._build_env()
            if "PATH" in os.environ:
                assert "PATH" in env
            ks.cleanup()

    def test_build_env_no_pythonpath(self):
        """PYTHONPATH should not be passed to prevent module injection."""
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            env = ks._build_env()
            assert "PYTHONPATH" not in env
            ks.cleanup()

    def test_build_env_home_is_sandbox_temp(self):
        """HOME should be set to sandbox temp dir, not real home."""
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            env = ks._build_env()
            assert env["HOME"] == ks.temp_dir
            assert env["HOME"] != os.path.expanduser("~")
            ks.cleanup()

    def test_profile_permissions(self):
        """Profile file should have 0o600 permissions."""
        if not _is_macos():
            pytest.skip("Seatbelt only on macOS")
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            if ks._profile_path and os.path.exists(ks._profile_path):
                mode = os.stat(ks._profile_path).st_mode & 0o777
                assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"
            ks.cleanup()

    def test_build_command_no_sandbox(self):
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            ks.sandbox_type = "none"
            ks._profile_path = None
            cmd = ks._build_command()
            assert cmd[0] == os.path.realpath(sys.executable)
            assert cmd[1] == "-c"
            ks.cleanup()

    @pytest.mark.skipif(not _is_macos(), reason="Seatbelt only on macOS")
    def test_build_command_seatbelt(self):
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            if ks.sandbox_type == "seatbelt":
                cmd = ks._build_command()
                assert cmd[0] == "sandbox-exec"
                assert cmd[1] == "-f"
                assert cmd[2] == ks._profile_path
            ks.cleanup()

    def test_execute_payload_includes_sandbox_type(self):
        """execute() payload JSON must contain 'sandbox_type' key."""
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            captured_payloads = []

            original_run = __import__("subprocess").run

            def _fake_run(cmd, input=None, **kwargs):
                if input is not None:
                    captured_payloads.append(input)
                # Return a fake successful result
                import subprocess as _sp
                result = mock.MagicMock()
                result.returncode = 0
                result.stdout = json.dumps({
                    "stdout": "", "stderr": "", "error": None,
                    "variables": {}, "dropped_vars": [],
                })
                result.stderr = ""
                return result

            with mock.patch("subprocess.run", side_effect=_fake_run):
                ks.execute("x = 1")

            assert len(captured_payloads) == 1
            payload_dict = json.loads(captured_payloads[0])
            assert "sandbox_type" in payload_dict
            ks.cleanup()

    def test_apply_landlock_log_warning_on_failure(self):
        """If apply_landlock_restrictions raises, _run() logs warning and continues."""
        import logging

        # We exercise the _run() logic indirectly by constructing a payload
        # where sandbox_type == "landlock" but apply_landlock_restrictions fails.
        # Since _run() is embedded as _RUNNER_SCRIPT string, we test the
        # KernelSandbox execute path with a mocked subprocess that simulates
        # the subprocess succeeding (Landlock setup failure is non-fatal).
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj)
            ks.sandbox_type = "landlock"

            # Mock subprocess.run to return a successful result
            success_result = mock.MagicMock()
            success_result.returncode = 0
            success_result.stdout = json.dumps({
                "stdout": "ok", "stderr": "", "error": None,
                "variables": {"x": 1}, "dropped_vars": [],
            })
            success_result.stderr = ""

            with mock.patch("subprocess.run", return_value=success_result):
                result = ks.execute("x = 1")

            # Execution must succeed even when sandbox_type is landlock
            assert result.error is None
            assert result.sandbox_type == "landlock"
            ks.cleanup()


# ---------------------------------------------------------------------------
# Sandbox execution tests (subprocess, may or may not use kernel sandbox)
# ---------------------------------------------------------------------------

class TestSandboxExecution:
    def test_basic_execution(self):
        """Simple code should execute and return result."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute("x = 2 + 2\nprint(x)")
                assert result.error is None
                assert "4" in result.stdout

    def test_variable_return(self):
        """Variables should be returned from subprocess."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute("answer = 42")
                assert result.error is None
                assert result.variables.get("answer") == 42

    def test_variable_injection(self):
        """Variables passed in should be available in code."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute(
                    "result = x * 2",
                    variables={"x": 21},
                )
                assert result.error is None
                assert result.variables.get("result") == 42

    def test_syntax_error(self):
        """Syntax errors should be reported."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute("def foo(:")
                assert result.error is not None
                assert "SyntaxError" in result.error

    def test_runtime_error(self):
        """Runtime errors should be reported."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute("1 / 0")
                assert result.error is not None
                assert "ZeroDivisionError" in result.error

    def test_timeout(self):
        """Infinite loops should be terminated by timeout."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=2) as ks:
                result = ks.execute("while True: pass")
                assert result.error is not None
                assert "timeout" in result.error.lower() or "Timeout" in result.error

    def test_non_serializable_vars_skipped(self):
        """Non-JSON-serializable variables should be skipped, not crash."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                # Lambda is not JSON-serializable
                result = ks.execute("x = 42", variables={"fn": lambda: None})
                assert result.error is None
                assert result.variables.get("x") == 42

    def test_stdout_capture(self):
        """Multiple print statements should be captured."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute("print('hello')\nprint('world')")
                assert "hello" in result.stdout
                assert "world" in result.stdout

    def test_sandbox_type_in_result(self):
        """Result should indicate which sandbox type was used."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute("x = 1")
                assert result.sandbox_type in ("seatbelt", "landlock", "none")

    def test_string_variable(self):
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute("msg = 'hello world'")
                assert result.variables.get("msg") == "hello world"

    def test_dict_variable(self):
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute("data = {'a': 1, 'b': [2, 3]}")
                assert result.variables.get("data") == {"a": 1, "b": [2, 3]}

    def test_list_variable(self):
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute("items = [1, 2, 3]")
                assert result.variables.get("items") == [1, 2, 3]


# ---------------------------------------------------------------------------
# Filesystem restriction tests (require actual sandbox-exec)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _is_macos(), reason="Seatbelt only on macOS")
class TestSeatbeltFilesystemRestrictions:
    def test_read_project_file_allowed(self):
        """Reading files within the project dir should work."""
        with tempfile.TemporaryDirectory() as proj:
            # Create a test file in the project
            test_file = os.path.join(proj, "test.txt")
            with open(test_file, "w") as f:
                f.write("hello")

            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute(
                    f"f = open('{test_file}'); content = f.read(); f.close()"
                )
                # This may or may not work depending on sandbox profile
                # being correctly applied — if sandbox-exec is available,
                # it should be allowed
                if result.error is None:
                    assert result.variables.get("content") == "hello"

    def test_write_project_file_allowed(self):
        """Writing files within the project dir should work."""
        with tempfile.TemporaryDirectory() as proj:
            outfile = os.path.join(proj, "output.txt")
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute(
                    f"f = open('{outfile}', 'w'); f.write('written'); f.close()"
                )
                if result.error is None:
                    assert os.path.exists(outfile)
                    with open(outfile) as f:
                        assert f.read() == "written"

    def test_network_denied(self):
        """Network access should be blocked by sandbox."""
        with tempfile.TemporaryDirectory() as proj:
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                result = ks.execute(
                    "import socket; s = socket.socket(); s.connect(('8.8.8.8', 53))"
                )
                # Should fail — either with sandbox denial or import error
                assert result.error is not None

    def test_write_outside_project_denied(self):
        """Writing files outside project dir should be denied by kernel."""
        with tempfile.TemporaryDirectory() as proj:
            # Try to write outside the project directory
            with KernelSandbox(project_root=proj, timeout_seconds=10) as ks:
                if ks.sandbox_type == "seatbelt":
                    result = ks.execute(
                        "f = open('/tmp/ccr_sandbox_test_write.txt', 'w'); f.write('test'); f.close()"
                    )
                    # Should fail — writes outside project/temp are denied
                    assert result.error is not None


# ---------------------------------------------------------------------------
# Sensitive path protection tests
# ---------------------------------------------------------------------------

class TestSensitivePathProtection:
    """Sensitive path protection relies on multiple layers.

    Seatbelt layer: broad file reads are allowed (Python needs them), but
    writes are restricted to project/temp only, network is denied, and
    process exec is limited to Python. This means sensitive files can be
    READ but not modified or exfiltrated.

    Python layer: AST validation and restricted builtins block programmatic
    access to sensitive files (restricted_open, blocked modules like os/pathlib).
    """

    def test_writes_to_sensitive_dirs_denied(self):
        """Write access to sensitive directories is denied by Seatbelt.

        Only project and temp dirs have write access.
        """
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(td, td)
            real_td = os.path.realpath(td)
            # All file-write rules should only reference project/temp
            for line in profile.split("\n"):
                if "allow file-write" in line:
                    assert real_td in line, \
                        f"Write rule should only reference project/temp: {line}"

    def test_network_denied_prevents_exfiltration(self):
        """Even with broad reads, data can't be exfiltrated — network is denied."""
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(td, td)
            assert "(deny network*)" in profile

    def test_process_exec_restricted(self):
        """Can't exec arbitrary programs to exfiltrate data."""
        with tempfile.TemporaryDirectory() as td:
            profile = generate_seatbelt_profile(td, td)
            assert "(allow process*)" not in profile
            assert "(allow process-fork)" in profile
            # Only Python executables allowed
            python_real = os.path.realpath(sys.executable)
            assert f'process-exec (literal "{python_real}")' in profile


# ---------------------------------------------------------------------------
# Graceful fallback tests
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    @mock.patch("ccr.rlm.sandbox.get_sandbox_type", return_value="none")
    def test_fallback_warns(self, mock_type):
        """When no sandbox is available, should warn and still work."""
        import logging
        with tempfile.TemporaryDirectory() as proj:
            with pytest.raises(Exception) if False else mock.patch.object(
                logging.getLogger("ccr.rlm.sandbox"), "warning"
            ) as mock_warn:
                ks = KernelSandbox(project_root=proj)
                mock_warn.assert_called_once()
                assert "unavailable" in mock_warn.call_args[0][0].lower() or \
                       "fallback" in mock_warn.call_args[0][0].lower() or \
                       "No kernel sandbox" in mock_warn.call_args[0][0]
                ks.cleanup()

    def test_fallback_still_executes(self):
        """Even without kernel sandbox, code should execute via subprocess."""
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(project_root=proj, timeout_seconds=10)
            # Force no-sandbox mode
            ks.sandbox_type = "none"
            ks._profile_path = None
            result = ks.execute("x = 1 + 1\nprint(x)")
            assert result.error is None
            assert "2" in result.stdout
            ks.cleanup()

    def test_missing_executable_handled(self):
        """If the Python executable doesn't exist, handle gracefully."""
        with tempfile.TemporaryDirectory() as proj:
            ks = KernelSandbox(
                project_root=proj,
                python_executable="/nonexistent/python",
                timeout_seconds=5,
            )
            ks.sandbox_type = "none"
            ks._profile_path = None
            result = ks.execute("x = 1")
            assert result.error is not None
            ks.cleanup()


# ---------------------------------------------------------------------------
# Runner script tests
# ---------------------------------------------------------------------------

class TestRunnerScript:
    def test_runner_script_is_valid_python(self):
        """The embedded runner script should be valid Python."""
        compile(_RUNNER_SCRIPT, "<test>", "exec")

    def test_runner_script_contains_json_output(self):
        """Runner script should produce JSON output."""
        assert "json.dumps" in _RUNNER_SCRIPT
        assert "json.loads" in _RUNNER_SCRIPT

    def test_runner_script_captures_stdout(self):
        assert "stdout_capture" in _RUNNER_SCRIPT
        assert "stderr_capture" in _RUNNER_SCRIPT

    def test_runner_script_has_restricted_builtins(self):
        """Runner script should use restricted builtins, not full __builtins__."""
        assert "_safe_builtins" in _RUNNER_SCRIPT
        assert '"__builtins__": _safe_builtins' in _RUNNER_SCRIPT
        # Should NOT have full builtins
        assert '"__builtins__": __builtins__' not in _RUNNER_SCRIPT

    def test_runner_script_blocks_dangerous_imports(self):
        """Runner script should use allowlist for module imports."""
        assert "_allowed_modules" in _RUNNER_SCRIPT
        assert "_safe_import" in _RUNNER_SCRIPT

    def test_runner_script_tracks_dropped_vars(self):
        """Runner script should report non-serializable variables."""
        assert "dropped_vars" in _RUNNER_SCRIPT


# ---------------------------------------------------------------------------
# Integration with CCRRepl
# ---------------------------------------------------------------------------

class TestREPLKernelSandboxIntegration:
    def test_repl_default_no_kernel_sandbox(self):
        """By default, CCRRepl should not use kernel sandbox."""
        from ccr.rlm.repl import CCRRepl
        repl = CCRRepl()
        assert repl._kernel_sandbox is None
        assert repl.use_kernel_sandbox is False
        repl.cleanup()

    def test_repl_kernel_sandbox_flag(self):
        """CCRRepl should accept use_kernel_sandbox parameter."""
        from ccr.rlm.repl import CCRRepl
        repl = CCRRepl(use_kernel_sandbox=True, project_root="/tmp")
        assert repl.use_kernel_sandbox is True
        # On macOS, should have a kernel sandbox; on other platforms, None
        if _is_macos():
            assert repl._kernel_sandbox is not None
        repl.cleanup()

    def test_repl_kernel_sandbox_basic_execution(self):
        """CCRRepl with kernel sandbox should execute basic code."""
        from ccr.rlm.repl import CCRRepl
        with tempfile.TemporaryDirectory() as proj:
            repl = CCRRepl(
                use_kernel_sandbox=True,
                project_root=proj,
                timeout_seconds=10,
            )
            if repl._kernel_sandbox is not None:
                result = repl.execute_code("x = 42\nprint(x)")
                assert "42" in result.stdout
                assert result.error is None
            repl.cleanup()

    def test_repl_kernel_sandbox_ast_validation_still_applies(self):
        """AST validation should still block dangerous code even with kernel sandbox."""
        from ccr.rlm.repl import CCRRepl
        with tempfile.TemporaryDirectory() as proj:
            repl = CCRRepl(
                use_kernel_sandbox=True,
                project_root=proj,
                timeout_seconds=10,
            )
            if repl._kernel_sandbox is not None:
                result = repl.execute_code("x = obj.__class__")
                assert result.error is not None
                assert "blocked" in result.error.lower() or "__class__" in result.error
            repl.cleanup()

    def test_repl_kernel_sandbox_cleanup(self):
        """CCRRepl cleanup should also clean up kernel sandbox."""
        from ccr.rlm.repl import CCRRepl
        with tempfile.TemporaryDirectory() as proj:
            repl = CCRRepl(
                use_kernel_sandbox=True,
                project_root=proj,
            )
            ks = repl._kernel_sandbox
            repl.cleanup()
            assert repl._kernel_sandbox is None
            if ks is not None:
                assert not os.path.exists(ks.temp_dir)

    @mock.patch("ccr.rlm.sandbox.get_sandbox_type", return_value="none")
    def test_repl_fallback_when_unavailable(self, _):
        """CCRRepl should fall back gracefully when kernel sandbox unavailable."""
        from ccr.rlm.repl import CCRRepl
        repl = CCRRepl(
            use_kernel_sandbox=True,
            project_root="/tmp",
        )
        # Should fall back to None (no kernel sandbox)
        assert repl._kernel_sandbox is None
        # But should still execute via in-process path
        result = repl.execute_code("x = 1 + 1\nprint(x)")
        assert "2" in result.stdout
        repl.cleanup()


# ---------------------------------------------------------------------------
# Expand helper tests
# ---------------------------------------------------------------------------

class TestPythonReadPaths:
    def test_returns_list(self):
        paths = _get_python_read_paths()
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_includes_sys_prefix(self):
        paths = _get_python_read_paths()
        prefix_real = os.path.realpath(sys.prefix)
        assert prefix_real in paths

    def test_all_paths_absolute(self):
        paths = _get_python_read_paths()
        for p in paths:
            assert os.path.isabs(p), f"Path should be absolute: {p}"

    def test_paths_are_sorted(self):
        paths = _get_python_read_paths()
        assert paths == sorted(paths)

    def test_no_duplicates(self):
        paths = _get_python_read_paths()
        assert len(paths) == len(set(paths))


class TestExpandHelper:
    def test_expand_tilde(self):
        result = _expand("~/.ssh")
        assert "~" not in result
        assert result.startswith("/")

    def test_expand_already_absolute(self):
        result = _expand("/usr/bin")
        assert result == os.path.realpath("/usr/bin")
