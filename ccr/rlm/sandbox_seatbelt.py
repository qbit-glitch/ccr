"""macOS Seatbelt profile generation and subprocess runner script for the kernel sandbox.

Generates deny-default Seatbelt (.sb) profiles for sandboxed Python execution,
and defines the runner script injected into sandboxed subprocesses.
"""

from __future__ import annotations

import os
import platform
import sys
import sysconfig
import textwrap

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
