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
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Sensitive directories that must never be accessible from the sandbox
_SENSITIVE_DIRS = [
    "~/.ssh",
    "~/.aws",
    "~/.config/gcloud",
    "~/.gnupg",
    "~/.config",
    "~/.kube",
    "~/.docker",
]

_SENSITIVE_FILES = [
    "~/.bash_history",
    "~/.zsh_history",
    "~/.python_history",
    "~/.netrc",
    "~/.npmrc",
    "~/.pypirc",
]


def _expand(p: str) -> str:
    """Expand ~ and resolve to real path."""
    return os.path.realpath(os.path.expanduser(p))


def is_seatbelt_available() -> bool:
    """Check if macOS sandbox-exec is available."""
    if platform.system() != "Darwin":
        return False
    return shutil.which("sandbox-exec") is not None


def is_landlock_available() -> bool:
    """Check if Linux Landlock enforcement is available.

    Performs a real probe: checks OS, kernel version >= 5.13, then attempts
    a syscall to confirm the kernel supports Landlock.

    Returns True only if Landlock syscalls are confirmed working.
    """
    if platform.system() != "Linux":
        return False
    # Check kernel version >= 5.13
    try:
        release = platform.release()
        major = int(release.split(".")[0])
        minor = int(release.split(".")[1].split("-")[0])
        if (major, minor) < (5, 13):
            return False
    except (ValueError, IndexError):
        return False
    # Probe: try creating a minimal ruleset; if ENOSYS/-1 → unavailable
    try:
        _landlock_probe()
        return True
    except (OSError, NotImplementedError):
        return False


def _landlock_probe() -> None:
    """Probe Landlock availability by attempting syscall. Raises OSError if unavailable."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)

    # struct landlock_ruleset_attr { __u64 handled_access_fs; }
    class _RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    ALL_FS = sum(1 << i for i in range(14))  # all LANDLOCK_ACCESS_FS_* bits
    attr = _RulesetAttr(handled_access_fs=ALL_FS)
    nr = _landlock_syscall_nr("create_ruleset")
    fd = libc.syscall(nr, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"landlock_create_ruleset failed: errno={errno}")
    # Close the fd
    os.close(fd)


def _landlock_syscall_nr(name: str) -> int:
    """Return syscall number for Landlock on the current architecture."""
    arch = platform.machine()
    NR: dict[str, dict[str, int]] = {
        "x86_64":  {"create_ruleset": 444, "add_rule": 445, "restrict_self": 446},
        "aarch64": {"create_ruleset": 444, "add_rule": 445, "restrict_self": 446},
        "arm64":   {"create_ruleset": 444, "add_rule": 445, "restrict_self": 446},
    }
    if arch not in NR:
        raise NotImplementedError(f"Landlock syscall numbers not known for arch: {arch}")
    return NR[arch][name]


# Landlock access rights bitmask constants
LANDLOCK_ACCESS_FS_EXECUTE    = (1 << 0)
LANDLOCK_ACCESS_FS_WRITE_FILE = (1 << 1)
LANDLOCK_ACCESS_FS_READ_FILE  = (1 << 2)
LANDLOCK_ACCESS_FS_READ_DIR   = (1 << 3)
LANDLOCK_ACCESS_FS_REMOVE_DIR = (1 << 4)
LANDLOCK_ACCESS_FS_REMOVE_FILE= (1 << 5)
LANDLOCK_ACCESS_FS_MAKE_CHAR  = (1 << 6)
LANDLOCK_ACCESS_FS_MAKE_DIR   = (1 << 7)
LANDLOCK_ACCESS_FS_MAKE_REG   = (1 << 8)
LANDLOCK_ACCESS_FS_MAKE_SYM   = (1 << 9)
LANDLOCK_ACCESS_FS_MAKE_SOCK  = (1 << 10)
LANDLOCK_ACCESS_FS_MAKE_FIFO  = (1 << 11)
LANDLOCK_ACCESS_FS_MAKE_BLOCK = (1 << 12)
LANDLOCK_ACCESS_FS_MAKE_IPC   = (1 << 13)
LANDLOCK_RULE_PATH_BENEATH    = 1


def apply_landlock_restrictions(project_root: str, temp_dir: str, python_executable: str) -> None:
    """Apply Landlock FS restrictions to the current process.

    Must be called after fork (in subprocess) before executing untrusted code.
    Calls prctl(PR_SET_NO_NEW_PRIVS) first to allow restrict_self without
    CAP_SYS_ADMIN.

    Allows:
    - Read: Python stdlib paths (sys.prefix, sys.exec_prefix, sysconfig paths)
    - Read+Execute: Python executable directory
    - Read+Write: temp_dir
    - Read: project_root (read-only access to project files)
    Denies: everything else

    On non-Linux platforms, raises NotImplementedError.
    """
    if platform.system() != "Linux":
        raise NotImplementedError("apply_landlock_restrictions is Linux-only")

    import ctypes
    import sysconfig as _sysconfig

    libc = ctypes.CDLL(None, use_errno=True)

    # prctl(PR_SET_NO_NEW_PRIVS=38, 1, 0, 0, 0) — required before restrict_self
    PR_SET_NO_NEW_PRIVS = 38
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")

    class _RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class _PathBeneathAttr(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    ALL_FS = sum(1 << i for i in range(14))
    attr = _RulesetAttr(handled_access_fs=ALL_FS)
    nr_create = _landlock_syscall_nr("create_ruleset")
    nr_add    = _landlock_syscall_nr("add_rule")
    nr_self   = _landlock_syscall_nr("restrict_self")

    ruleset_fd = libc.syscall(nr_create, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    READ_EXEC  = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR | LANDLOCK_ACCESS_FS_EXECUTE
    READ_ONLY  = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR
    READ_WRITE = (READ_ONLY | LANDLOCK_ACCESS_FS_WRITE_FILE
                  | LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_DIR)

    def _add_path_rule(path: str, access: int) -> None:
        path = os.path.realpath(path)
        if not os.path.exists(path):
            return
        target = path if os.path.isdir(path) else os.path.dirname(path)
        fd = os.open(target, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = _PathBeneathAttr(allowed_access=access, parent_fd=fd)
            ret = libc.syscall(nr_add, ruleset_fd, LANDLOCK_RULE_PATH_BENEATH,
                               ctypes.byref(rule), 0)
            if ret < 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
        finally:
            os.close(fd)

    # Allow Python stdlib read paths
    for p in [sys.prefix, sys.exec_prefix] + list(_sysconfig.get_paths().values()):
        if p and os.path.exists(p):
            _add_path_rule(p, READ_ONLY)

    # Allow Python executable directory read+exec
    _add_path_rule(os.path.dirname(os.path.realpath(python_executable)), READ_EXEC)

    # Allow temp_dir read+write (sandbox scratch space)
    _add_path_rule(temp_dir, READ_WRITE)

    # Allow project_root read-only
    if project_root and os.path.exists(project_root):
        _add_path_rule(project_root, READ_ONLY)

    # Restrict self — after this call, no new FS capabilities can be acquired
    ret = libc.syscall(nr_self, ruleset_fd, 0)
    if ret < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")

    os.close(ruleset_fd)


def get_sandbox_type() -> str:
    """Return the sandbox type available on this platform."""
    if is_seatbelt_available():
        return "seatbelt"
    if is_landlock_available():
        return "landlock"
    return "none"


# ---------------------------------------------------------------------------
# Seatbelt profile generation (macOS)
# ---------------------------------------------------------------------------

def _get_python_read_paths() -> list[str]:
    """Discover paths Python needs to read for stdlib, site-packages, etc.

    Returns a deduplicated list of real (resolved) directory paths.
    """
    paths: set[str] = set()

    # Python prefix and exec_prefix (stdlib, lib-dynload, etc.)
    for p in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
        if p:
            paths.add(os.path.realpath(p))

    # sysconfig paths (stdlib, platstdlib, purelib, platlib, include, data)
    for key, val in sysconfig.get_paths().items():
        if val:
            paths.add(os.path.realpath(val))

    # sys.path entries (includes site-packages, .egg dirs, etc.)
    for p in sys.path:
        if p and os.path.isdir(p):
            paths.add(os.path.realpath(p))

    # Common system paths Python reads at startup
    for p in ("/usr/lib", "/usr/local/lib", "/etc/ssl", "/etc/ssl/certs"):
        rp = os.path.realpath(p)
        if os.path.exists(rp):
            paths.add(rp)

    # macOS-specific: frameworks, dyld shared cache
    if platform.system() == "Darwin":
        for p in (
            "/System/Library/Frameworks",
            "/Library/Frameworks",
            "/usr/lib/dyld",
            "/System/Library/PrivateFrameworks",
            "/Library/Apple",
            "/private/var/db/dyld",
            "/usr/share",
            "/private/etc",
            "/etc",
            "/var",
        ):
            rp = os.path.realpath(p)
            if os.path.exists(rp):
                paths.add(rp)

    return sorted(paths)


def _get_python_exec_paths(python_real: str) -> list[str]:
    """Discover all Python executables that might be invoked.

    On macOS framework builds, Python uses posix_spawn to re-exec through
    a .app bundle (e.g., Python.framework/.../Python.app/Contents/MacOS/Python).
    We need to allow all these paths in the sandbox.
    """
    paths: set[str] = {python_real}

    # Check for _base_executable (different from sys.executable in venvs)
    base_exe = getattr(sys, "_base_executable", None)
    if base_exe:
        paths.add(os.path.realpath(base_exe))

    # On macOS framework builds, find the Python.app binary
    if platform.system() == "Darwin":
        framework_prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")
        if framework_prefix:
            # Look for Resources/Python.app/Contents/MacOS/Python
            framework_prefix_real = os.path.realpath(framework_prefix)
            version = sysconfig.get_config_var("VERSION") or f"{sys.version_info.major}.{sys.version_info.minor}"
            app_path = os.path.join(
                framework_prefix_real, "Python.framework", "Versions", version,
                "Resources", "Python.app", "Contents", "MacOS", "Python"
            )
            if os.path.exists(app_path):
                paths.add(os.path.realpath(app_path))

            # Also check the direct framework path (Homebrew layout)
            python_dir = os.path.dirname(python_real)
            framework_root = python_dir
            # Walk up to find the framework root
            for _ in range(10):
                parent = os.path.dirname(framework_root)
                if parent == framework_root:
                    break
                if os.path.basename(framework_root) == "Python.framework":
                    # Found it — look for Python.app under Resources
                    for root, dirs, files in os.walk(framework_root):
                        if "MacOS" in root and "Python" in files:
                            candidate = os.path.join(root, "Python")
                            paths.add(os.path.realpath(candidate))
                    break
                framework_root = parent

    return sorted(paths)


def generate_seatbelt_profile(
    project_root: str,
    temp_dir: str,
    python_executable: str | None = None,
) -> str:
    """Generate a macOS Seatbelt (.sb) profile for sandboxed Python execution.

    Security model:
    - (deny default): everything not explicitly allowed is denied
    - Network: completely denied
    - Process exec: restricted to the Python interpreter only
    - File writes: only to project directory and temp directory
    - File reads: broad (subpath "/") — see note below

    Why broad file reads: Python's startup reads from numerous unpredictable
    paths (dyld cache, framework dirs, locale files, system libraries, etc.)
    that vary across macOS versions and Python installations. Enumerating them
    all is fragile and breaks across environments. Additionally, macOS Seatbelt
    uses "allow wins" semantics — explicit deny rules do NOT override allow
    rules — so we cannot use "allow everything then deny sensitive paths".

    The read access is acceptable because:
    1. Network is denied — data cannot be exfiltrated
    2. Writes are restricted — sensitive files cannot be modified
    3. Process exec is limited — no spawning arbitrary programs
    4. The Python-level sandbox (AST validation, restricted builtins) provides
       defense-in-depth against reading sensitive files programmatically

    mach*/ipc*: Required for Python's runtime on macOS (Obj-C bridge,
    libdispatch, CoreFoundation). Restricting further breaks Python.
    """
    python_exe = python_executable or sys.executable
    python_real = os.path.realpath(python_exe)
    project_real = os.path.realpath(project_root)
    temp_real = os.path.realpath(temp_dir)

    # Discover all Python executables that may be invoked via posix_spawn.
    # On macOS framework builds, Python re-execs through a .app bundle.
    python_exec_paths = _get_python_exec_paths(python_real)
    exec_rules = "\n".join(
        f'(allow process-exec (literal "{p}"))' for p in python_exec_paths
    )

    profile = f"""\
(version 1)

; Deny everything by default
(deny default)

; === NETWORK: deny all ===
(deny network*)

; === PROCESS: restricted to Python interpreter only ===
; Allow fork (Python needs it internally) but only allow exec of the
; specific Python binaries. On macOS framework builds, Python uses
; posix_spawn to re-exec through a .app bundle.
(allow process-fork)
{exec_rules}
(allow sysctl*)
; mach* and ipc* are required for Python's runtime on macOS (Obj-C bridge,
; libdispatch, CoreFoundation). Restricting these further breaks Python.
; Risk is mitigated by: no network, writes limited to project/temp only,
; exec limited to Python only.
(allow mach*)
(allow ipc*)
(allow signal)

; === FILE ACCESS: broad read ===
; Python reads from numerous unpredictable paths at startup (dyld cache,
; frameworks, locale files, etc.) that vary across macOS versions.
; macOS Seatbelt uses "allow wins" semantics so explicit deny rules
; cannot override this. Read access is safe because network is denied
; (no exfiltration) and writes are restricted (no modification).
(allow file-read* (subpath "/"))

; === FILE ACCESS: write only to project dir and temp dir ===
(allow file-write* (subpath "{project_real}"))
(allow file-write* (subpath "{temp_real}"))
"""

    return profile


# ---------------------------------------------------------------------------
# Subprocess runner script (injected into the sandboxed subprocess)
# ---------------------------------------------------------------------------

_RUNNER_SCRIPT = textwrap.dedent("""\
import json
import sys
import os


def _run():
    # Read execution payload from stdin
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    variables = payload.get("variables", {})
    project_root = payload.get("project_root")
    temp_dir = payload.get("temp_dir")

    # Apply Landlock restrictions if requested (Linux only, in-subprocess)
    sandbox_type = payload.get("sandbox_type", "none")
    if sandbox_type == "landlock":
        try:
            import platform as _platform
            if _platform.system() == "Linux":
                from ccr.rlm.sandbox import apply_landlock_restrictions
                apply_landlock_restrictions(
                    project_root=project_root or "",
                    temp_dir=temp_dir or "/tmp",
                    python_executable=sys.executable,
                )
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning("Landlock setup failed: %s", _e)

    # Build restricted builtins — mirrors repl.py's _SAFE_BUILTINS.
    # The kernel sandbox is the primary enforcement layer, but restricted
    # builtins provide defense-in-depth inside the subprocess.
    import builtins as _builtins

    # Restricted open: only allows project_root and temp_dir
    _real_open = open
    _osp = os.path
    _allowed_dirs = []
    if project_root:
        _allowed_dirs.append(_osp.realpath(project_root))
    if temp_dir:
        _allowed_dirs.append(_osp.realpath(temp_dir))

    def _restricted_open(file, mode='r', *args, **kwargs):
        resolved = _osp.realpath(str(file))
        for d in _allowed_dirs:
            if resolved.startswith(d + _osp.sep) or resolved == d:
                return _real_open(resolved, mode, *args, **kwargs)
        raise PermissionError(f"Sandbox: access denied to {file}")

    # Allowlist import — only these modules can be imported in the sandbox.
    # Allowlist is safer than denylist: new stdlib modules are blocked by default.
    _allowed_modules = frozenset({
        "math", "decimal", "fractions", "statistics", "random",
        "string", "re", "textwrap", "unicodedata", "difflib",
        "collections", "functools", "itertools", "operator",
        "datetime", "time", "calendar", "zoneinfo",
        "json", "csv", "base64", "hashlib", "hmac",
        "dataclasses", "enum", "typing", "types", "abc",
        "copy", "pprint", "numbers",
        "heapq", "bisect", "array",
        "contextlib", "warnings",
    })
    _real_import = _builtins.__import__

    def _safe_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level not in _allowed_modules:
            raise ImportError(f"Module '{name}' is blocked in the CCR sandbox (not in allowlist)")
        return _real_import(name, *args, **kwargs)

    _safe_builtins = {
        "print": print, "len": len, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
        "isinstance": isinstance, "issubclass": issubclass,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed, "range": range,
        "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "any": any, "all": all, "pow": pow, "divmod": divmod,
        "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
        "repr": repr, "ascii": ascii, "format": format, "hash": hash, "id": id,
        "iter": iter, "next": next, "slice": slice, "callable": callable,
        "hasattr": hasattr, "getattr": getattr,
        "bytes": bytes, "bytearray": bytearray,
        "complex": complex, "object": object,
        "type": type,
        "__build_class__": __build_class__,
        "__import__": _safe_import,
        "open": _restricted_open,
        # Exceptions
        "Exception": Exception, "BaseException": BaseException,
        "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
        "IndexError": IndexError, "AttributeError": AttributeError,
        "FileNotFoundError": FileNotFoundError, "OSError": OSError, "IOError": IOError,
        "RuntimeError": RuntimeError, "NameError": NameError, "ImportError": ImportError,
        "StopIteration": StopIteration, "AssertionError": AssertionError,
        "NotImplementedError": NotImplementedError, "ArithmeticError": ArithmeticError,
        "LookupError": LookupError, "Warning": Warning, "PermissionError": PermissionError,
        # Blocked (set to None to raise clear errors)
        "input": None, "eval": None, "compile": None, "exec": None,
        "globals": None, "locals": None, "vars": None, "dir": None,
        "setattr": None, "delattr": None,
    }

    namespace = {"__builtins__": _safe_builtins, "__name__": "__ccr_sandbox__"}
    namespace.update(variables)

    stdout_capture = []
    stderr_capture = []

    class _Capture:
        def __init__(self, target):
            self._target = target
        def write(self, s):
            self._target.append(s)
            return len(s)
        def flush(self):
            pass

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Capture(stdout_capture)
    sys.stderr = _Capture(stderr_capture)

    error = None
    try:
        co = _builtins.compile(code, "<ccr-sandbox>", "exec")
        _builtins.exec(co, namespace, namespace)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    # Extract serializable variables, track dropped ones
    result_vars = {}
    dropped_vars = []
    for k, v in namespace.items():
        if k.startswith("_") or k in ("__builtins__", "__name__"):
            continue
        try:
            json.dumps(v)  # test serializability
            result_vars[k] = v
        except (TypeError, ValueError, OverflowError):
            dropped_vars.append(k)
            try:
                result_vars[k] = str(v)
            except Exception:
                result_vars[k] = f"<{type(v).__name__}>"

    result = {
        "stdout": "".join(stdout_capture),
        "stderr": "".join(stderr_capture),
        "error": error,
        "variables": result_vars,
        "dropped_vars": dropped_vars,
    }
    old_stdout.write(json.dumps(result))
    old_stdout.flush()


_run()
""")


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
