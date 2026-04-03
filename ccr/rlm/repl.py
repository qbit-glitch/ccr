"""Sandboxed Python REPL for the RLM execution layer.

Adapted from vendor/rlm LocalREPL — simplified for CCR.
Key difference: no TCP sockets. LLM clients injected directly.

Security note: Uses Python's exec() intentionally within a restricted namespace
with _SAFE_BUILTINS. This is the core of the RLM paper — the model writes code
that executes in a sandboxed REPL to programmatically inspect context.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any

from ccr.core.types import REPLResult

# Modules ALLOWED for import inside the REPL sandbox.
# Allowlist is safer than denylist: new stdlib modules are blocked by default,
# and no new dangerous module can slip through by being added to Python.
_ALLOWED_MODULES = frozenset({
    # Math/numeric
    "math", "decimal", "fractions", "statistics", "random",
    # String/text
    "string", "re", "textwrap", "unicodedata", "difflib",
    # Data structures
    "collections", "functools", "itertools", "operator",
    # Date/time
    "datetime", "time", "calendar", "zoneinfo",
    # Serialization (safe subset)
    "json", "csv", "base64", "hashlib", "hmac",
    # Type system
    "dataclasses", "enum", "typing", "types", "abc",
    # Utilities
    "copy", "pprint", "numbers",
    "heapq", "bisect", "array",
    "contextlib", "warnings",
})


# Store the real __import__ once at module load time, before any patching.
_REAL_IMPORT = builtins.__import__


def _safe_import(name: str, *args, **kwargs):
    """Restricted __import__ that only allows safe modules (allowlist)."""
    top_level = name.split(".")[0]
    if top_level not in _ALLOWED_MODULES:
        raise ImportError(
            f"Module '{name}' is blocked in the CCR sandbox (not in allowlist). "
            f"Allowed modules: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    return _REAL_IMPORT(name, *args, **kwargs)


def _make_restricted_open(allowed_dirs: list[str]):
    """Create a restricted open() that only allows access to specific directories.

    Args:
        allowed_dirs: List of directory paths that the sandbox is allowed to access.
                      Paths are resolved to their real (canonical) form for comparison.
    """
    # Use os.path directly (already imported at module level) to avoid
    # going through builtins.__import__ which may be patched during sandbox exec.
    osp = os.path
    _real_open = open

    def restricted_open(file, mode='r', *args, **kwargs):
        resolved = osp.realpath(str(file))
        for d in allowed_dirs:
            if resolved.startswith(osp.realpath(d) + osp.sep) or resolved == osp.realpath(d):
                return _real_open(resolved, mode, *args, **kwargs)  # H1: use resolved, not file
        raise PermissionError(f"REPL sandbox: access denied to {file}")

    restricted_open.__name__ = "restricted_open"
    restricted_open.__doc__ = "Sandbox-restricted open(). Only allows access to project root and temp dir."
    return restricted_open


# --- C1: Safe type() wrapper — blocks 3-arg metaclass form ---
def _safe_type(*args):
    """type(obj) is allowed; type(name, bases, dict) is blocked in the sandbox."""
    if len(args) != 1:
        raise TypeError("type() with multiple arguments is blocked in the sandbox")
    return type(args[0])

_safe_type.__name__ = "type"


# --- C2: Restricted object proxy — blocks __subclasses__ traversal ---
class _RestrictedObject:
    """Proxy for object that blocks __subclasses__ and other dangerous methods."""
    pass


# --- C3: Safe getattr/hasattr — blocks dunder attribute access ---
_DUNDER_RE = re.compile(r'^__.*__$')


def _safe_getattr(obj, name, *default):
    """getattr() that blocks access to dunder attributes in the sandbox."""
    if isinstance(name, str) and _DUNDER_RE.match(name):
        raise AttributeError(f"Access to dunder attribute '{name}' is blocked in the sandbox")
    if default:
        return getattr(obj, name, default[0])
    return getattr(obj, name)


def _safe_hasattr(obj, name):
    """hasattr() that returns False for dunder attributes in the sandbox."""
    if isinstance(name, str) and _DUNDER_RE.match(name):
        return False
    return hasattr(obj, name)


# --- C1-C4: AST-level sandbox hardening ---
# Direct attribute syntax (obj.__class__) bypasses _safe_getattr,
# so we must inspect the AST before execution.

_ALLOWED_DUNDERS = frozenset({
    '__name__', '__doc__', '__str__', '__repr__', '__len__',
    '__init__', '__enter__', '__exit__', '__iter__', '__next__',
    '__getitem__', '__setitem__', '__delitem__', '__contains__',
    '__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__',
    '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
    '__mod__', '__pow__', '__neg__', '__pos__', '__abs__',
    '__and__', '__or__', '__xor__', '__invert__',
    '__radd__', '__rsub__', '__rmul__', '__rtruediv__',
    '__iadd__', '__isub__', '__imul__', '__itruediv__',
    '__hash__', '__bool__', '__int__', '__float__', '__complex__',
    '__index__', '__call__',
})

_DANGEROUS_DUNDERS = frozenset({
    '__class__', '__bases__', '__mro__', '__subclasses__',
    '__globals__', '__code__', '__closure__', '__func__',
    '__self__', '__dict__', '__slots__',
    '__traceback__', '__context__', '__cause__', '__suppress_context__',
    '__builtins__', '__import__', '__loader__', '__spec__',
    '__file__', '__path__', '__package__', '__qualname__',
    '__module__', '__annotations__', '__wrapped__',
    '__init_subclass__', '__set_name__', '__class_getitem__',
    '__getattribute__', '__getattr__', '__setattr__', '__delattr__',
    'gi_frame', 'gi_code', 'gi_yieldfrom',
    'cr_frame', 'cr_code', 'cr_origin',
    'ag_frame', 'ag_code',
    'tb_frame', 'tb_next', 'tb_lineno', 'tb_lasti',
    'f_globals', 'f_locals', 'f_builtins', 'f_code', 'f_back',
    'co_consts', 'co_names', 'co_code',
})

_DANGEROUS_FUNC_DEFS = frozenset({
    '__init_subclass__', '__set_name__', '__del__',
    '__getattr__', '__getattribute__',
})


def _validate_ast(code: str) -> None:
    """Reject code that accesses dangerous dunder attributes.

    Direct attribute syntax (obj.__class__) bypasses _safe_getattr,
    so we must inspect the AST before execution.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # Let exec handle syntax errors naturally

    for node in ast.walk(tree):
        # Block dangerous attribute access: obj.__class__, obj.__globals__, etc.
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in _DANGEROUS_DUNDERS:
                raise PermissionError(
                    f"Access to '{attr}' is blocked in the CCR sandbox"
                )
            # Block any remaining dunder not in allowlist
            if attr.startswith('__') and attr.endswith('__') and attr not in _ALLOWED_DUNDERS:
                raise PermissionError(
                    f"Access to dunder attribute '{attr}' is blocked in the CCR sandbox"
                )

        # Block dangerous function definitions (__init_subclass__, __set_name__, etc.)
        if isinstance(node, ast.FunctionDef) and node.name in _DANGEROUS_FUNC_DEFS:
            raise PermissionError(
                f"Defining '{node.name}' is blocked in the CCR sandbox"
            )


# Safe builtins — blocks eval/exec/compile/input, allows everything else.
# C1: type replaced with _safe_type (blocks 3-arg form)
# C2: object replaced with _RestrictedObject (blocks __subclasses__)
# C3: getattr/hasattr replaced with safe versions (blocks dunder access)
# C4: super, property, staticmethod, classmethod removed (descriptor abuse)
_SAFE_BUILTINS: dict[str, Any] = {
    "print": print, "len": len, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
    "type": _safe_type, "isinstance": isinstance, "issubclass": issubclass,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "range": range,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "any": any, "all": all, "pow": pow, "divmod": divmod,
    "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
    "repr": repr, "ascii": ascii, "format": format, "hash": hash, "id": id,
    "iter": iter, "next": next, "slice": slice, "callable": callable,
    "hasattr": _safe_hasattr, "getattr": _safe_getattr,
    "bytes": bytes, "bytearray": bytearray,
    "complex": complex, "object": _RestrictedObject,
    "__build_class__": __build_class__,  # Required for 'class' statement
    "__import__": _safe_import,
    # Exceptions
    "Exception": Exception, "BaseException": BaseException,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError, "OSError": OSError, "IOError": IOError,
    "RuntimeError": RuntimeError, "NameError": NameError, "ImportError": ImportError,
    "StopIteration": StopIteration, "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError, "ArithmeticError": ArithmeticError,
    "LookupError": LookupError, "Warning": Warning,
    # Blocked (set to None to raise clear errors)
    "input": None, "eval": None, "compile": None, "exec": None,
    "globals": None, "locals": None,
}

class _BoundedStringIO(io.StringIO):
    """StringIO with a maximum size limit to prevent unbounded output DoS (M1).

    Tracks by character count (not bytes) since StringIO operates on str.
    This avoids the char/byte mismatch where s[:remaining] could truncate
    differently than the byte count suggests for multi-byte characters.
    """

    def __init__(self, *args, max_chars: int = 10_000_000, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_chars = max_chars
        self._current_chars = 0

    def write(self, s: str) -> int:
        if self._current_chars + len(s) > self._max_chars:
            remaining = self._max_chars - self._current_chars
            if remaining <= 0:
                return 0
            s = s[:remaining]
        self._current_chars += len(s)
        return super().write(s)


# Default execution timeout in seconds
_DEFAULT_TIMEOUT_SECONDS = 30

# Names that must not be overwritten by user code
_RESERVED_NAMES = {
    "llm_query", "rlm_query", "llm_query_batched", "rlm_query_batched",
    "FINAL_VAR", "SHOW_VARS",
    "get_file", "search_repo", "estimate_tokens", "context",
}


def _run_in_namespace(code: str, namespace: dict) -> None:
    """Execute code string in the given namespace.

    This is the core of the RLM paper's REPL mechanism — the model writes
    Python code that runs against the repo context programmatically,
    rather than loading everything into the LLM's context window.

    The namespace is restricted via _SAFE_BUILTINS (no eval/compile/input).
    """
    _validate_ast(code)  # AST-level security check (C1-C4)
    # Python's built-in code execution in a restricted namespace
    builtins.__dict__  # ensure builtins loaded
    co = builtins.compile(code, "<ccr-repl>", "exec")
    builtins.eval(co, namespace, namespace)  # noqa: S307


class CCRRepl:
    """Sandboxed Python REPL with CCR tools injected.

    Tools available in the REPL:
        context           — repo index dict (file metadata, symbols, imports)
        get_file(path, offset=0, limit=0) — fetch content of any indexed file (paginated)
        search_repo(q, mode="keyword"|"bm25"|"semantic"|"hybrid") — search files
        estimate_tokens(t)— estimate token count of a string
        llm_query(prompt) — one-shot LLM call to the sub-model
        rlm_query(prompt) — recursive RLM sub-call (spawns child orchestrator)
        FINAL_VAR(name)   — signal completion, return variable value
        SHOW_VARS()       — list all user-created variables
    """

    def __init__(
        self,
        sub_client: Any = None,
        repo_index: Any = None,
        subcall_fn: Any = None,
        custom_tools: dict[str, Any] | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        project_root: str | None = None,
        use_kernel_sandbox: bool = False,
    ):
        self.sub_client = sub_client
        self.repo_index = repo_index
        self.subcall_fn = subcall_fn  # Callback for rlm_query (spawns child CCRRlm)
        self.timeout_seconds = timeout_seconds
        self.project_root = project_root

        self._last_final_answer: str | None = None
        self._exec_lock = threading.Lock()
        self._cleaned_up = False
        self.temp_dir = tempfile.mkdtemp(prefix="ccr_repl_")

        # Kernel sandbox (macOS Seatbelt / Linux Landlock)
        self._kernel_sandbox = None
        self.use_kernel_sandbox = use_kernel_sandbox
        if use_kernel_sandbox:
            from ccr.rlm.sandbox import KernelSandbox, get_sandbox_type
            sandbox_type = get_sandbox_type()
            if sandbox_type != "none":
                self._kernel_sandbox = KernelSandbox(
                    project_root=project_root,
                    timeout_seconds=timeout_seconds,
                )
                import logging
                logging.getLogger(__name__).info(
                    "Kernel sandbox enabled: %s", sandbox_type
                )
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "Kernel sandbox requested but unavailable on this platform. "
                    "Using Python-level sandboxing only."
                )

        # Build restricted open with allowed directories
        allowed_dirs = [self.temp_dir]
        if project_root:
            allowed_dirs.append(project_root)
        self._restricted_open = _make_restricted_open(allowed_dirs)

        # Namespace: globals (builtins + tools) and locals (user variables)
        safe_builtins = _SAFE_BUILTINS.copy()
        safe_builtins["open"] = self._restricted_open
        self.globals: dict[str, Any] = {"__builtins__": safe_builtins, "__name__": "__ccr_repl__"}
        # Also set __import__ at globals level to prevent bypass via module __builtins__
        self.globals["__import__"] = _safe_import
        self.locals: dict[str, Any] = {}

        self._setup_tools(custom_tools or {})

    def _setup_tools(self, custom_tools: dict[str, Any]) -> None:
        """Inject CCR tools into the REPL globals."""
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["rlm_query"] = self._rlm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query_batched"] = self._rlm_query_batched
        self.globals["get_file"] = self._get_file
        self.globals["search_repo"] = self._search_repo
        self.globals["estimate_tokens"] = self._estimate_tokens

        # Load repo index as context variable
        if self.repo_index is not None:
            if hasattr(self.repo_index, "to_context_dict"):
                self.locals["context"] = self.repo_index.to_context_dict()
            elif isinstance(self.repo_index, dict):
                self.locals["context"] = self.repo_index
            else:
                self.locals["context"] = str(self.repo_index)

        # Add custom tools
        for name, value in custom_tools.items():
            if name in _RESERVED_NAMES:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                actual_value, description = value
                if callable(actual_value):
                    self.globals[name] = actual_value
                else:
                    self.locals[name] = actual_value
            elif callable(value):
                self.globals[name] = value
            else:
                self.locals[name] = value

    # --- Scaffold functions ---

    def _final_var(self, variable_name: str | Any) -> str:
        """Return the value of a variable as the final answer."""
        if not isinstance(variable_name, str):
            answer = str(variable_name)
            self._last_final_answer = answer
            return answer

        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            val = self.locals[variable_name]
            answer = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
            self._last_final_answer = answer
            return answer

        available = [k for k in self.locals if not k.startswith("_")]
        msg = (
            f"Error: Variable '{variable_name}' not found. "
            f"Available: {available}. Create it before calling FINAL_VAR."
        )
        print(msg)
        return msg

    def _show_vars(self) -> str:
        """Show all user-created variables."""
        available = {
            k: f"{type(v).__name__}({len(v) if hasattr(v, '__len__') else ''})"
            for k, v in self.locals.items()
            if not k.startswith("_")
        }
        if not available:
            return "No variables created yet."
        return f"Variables: {json.dumps(available, indent=2)}"

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """One-shot LLM call to the sub-model. No REPL, no recursion."""
        if self.sub_client is None:
            return "Error: No sub-model client configured"
        try:
            messages = [{"role": "user", "content": prompt}]
            return self.sub_client.completion(messages)
        except Exception as e:
            return f"Error: LLM query failed — {e}"

    def _rlm_query(self, prompt: str, model: str | None = None) -> str:
        """Recursive RLM sub-call. Spawns a child orchestrator if available."""
        if self.subcall_fn is not None:
            try:
                result = self.subcall_fn(prompt, model)
                return result.response if hasattr(result, "response") else str(result)
            except Exception as e:
                return f"Error: RLM sub-call failed — {e}"
        # Fallback to plain LLM call
        return self._llm_query(prompt, model)

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Batch LLM calls — runs sequentially (concurrent mode not yet supported)."""
        return [self._llm_query(p, model) for p in prompts]

    def _rlm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Batch RLM sub-calls — runs sequentially."""
        return [self._rlm_query(p, model) for p in prompts]

    def _get_file(self, path: str, offset: int = 0, limit: int = 0) -> str:
        """Get content of an indexed file, with optional pagination.

        Args:
            path: Path to the file within the repo.
            offset: Number of lines to skip from the start (default 0 — no skip).
            limit: Maximum number of lines to return (default 0 — return all).
        """
        if self.repo_index is None:
            return "Error: No repo index loaded"
        if hasattr(self.repo_index, "get_file"):
            content = self.repo_index.get_file(path)
            if content is None:
                return f"Error: File not found: {path}"
            if offset > 0 or limit > 0:
                lines = content.splitlines(keepends=True)
                if offset > 0:
                    lines = lines[offset:]
                if limit > 0:
                    lines = lines[:limit]
                return "".join(lines)
            return content
        return "Error: Repo index does not support get_file()"

    def _search_repo(self, query: str, file_glob: str = "**/*", mode: str = "keyword") -> list[dict]:
        """Search files by content, symbol, or path pattern.

        Args:
            query: Search query string.
            file_glob: Glob pattern to filter files (default "**/*").
            mode: "keyword" (default), "bm25", "semantic", or "hybrid".
        """
        if self.repo_index is None:
            return []
        idx = self.repo_index
        if mode == "bm25" and hasattr(idx, "bm25_search"):
            return idx.bm25_search(query, file_glob=file_glob)
        if mode == "semantic" and hasattr(idx, "semantic_search"):
            return idx.semantic_search(query, file_glob=file_glob)
        if mode == "hybrid" and hasattr(idx, "hybrid_search"):
            return idx.hybrid_search(query, file_glob=file_glob)
        if hasattr(idx, "search"):
            return idx.search(query, file_glob=file_glob)
        return []

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        from ccr.utils.tokens import count_tokens
        return count_tokens(text)

    # --- Execution ---

    @contextmanager
    def _capture_output(self):
        """Capture stdout and stderr during code execution (M1: bounded to 10M chars)."""
        stdout_buf = _BoundedStringIO(max_chars=10_000_000)
        stderr_buf = _BoundedStringIO(max_chars=10_000_000)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buf, stderr_buf
        try:
            yield stdout_buf, stderr_buf
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _restore_scaffold(self) -> None:
        """Restore reserved names after execution to prevent overwrites."""
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["rlm_query"] = self._rlm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query_batched"] = self._rlm_query_batched
        self.globals["get_file"] = self._get_file
        self.globals["search_repo"] = self._search_repo
        self.globals["estimate_tokens"] = self._estimate_tokens
        # Restore context alias if context_0 exists
        if "context_0" in self.locals:
            self.locals["context"] = self.locals["context_0"]

    def _execute_kernel_sandboxed(self, code: str) -> REPLResult:
        """Execute code via kernel-sandboxed subprocess.

        The kernel sandbox (Seatbelt/Landlock) enforces OS-level restrictions.
        AST validation still runs as defense-in-depth before sending to subprocess.
        """
        start_time = time.perf_counter()

        # AST validation as defense-in-depth (runs in-process, before subprocess)
        try:
            _validate_ast(code)
        except PermissionError as e:
            return REPLResult(
                stdout="",
                stderr=str(e),
                locals_snapshot={},
                execution_time=time.perf_counter() - start_time,
                final_answer=None,
                error=str(e),
            )

        # Serialize only JSON-safe locals for the subprocess
        serializable_locals: dict[str, Any] = {}
        for k, v in self.locals.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                serializable_locals[k] = v
            except (TypeError, ValueError, OverflowError):
                pass

        result = self._kernel_sandbox.execute(
            code=code,
            variables=serializable_locals,
        )

        # Update locals with variables returned from subprocess
        for k, v in result.variables.items():
            if not k.startswith("_"):
                self.locals[k] = v

        # Check for FINAL_VAR in the subprocess output
        # The subprocess can't call our _final_var, so we check if any
        # variable was assigned and FINAL_VAR was called in the code
        if "FINAL_VAR" in code and not result.error:
            # Try to extract the final answer from returned variables
            # by re-running FINAL_VAR logic locally with updated locals
            import re as _re
            match = _re.search(r"FINAL_VAR\(['\"](\w+)['\"]\)", code)
            if match:
                var_name = match.group(1)
                self._final_var(var_name)
            else:
                # Direct value FINAL_VAR(123) — check stdout
                match = _re.search(r"FINAL_VAR\((.+?)\)", code)
                if match:
                    val = match.group(1).strip().strip("'\"")
                    if val in self.locals:
                        self._final_var(val)

        # Build snapshot
        snapshot = {}
        for k, v in self.locals.items():
            if not k.startswith("_") and k not in ("context",):
                try:
                    r = repr(v)
                    snapshot[k] = r[:200] if len(r) > 200 else r
                except Exception:
                    snapshot[k] = f"<{type(v).__name__}>"

        stderr = result.stderr or ""
        if result.error:
            stderr = (stderr + "\n" + result.error).strip()

        return REPLResult(
            stdout=result.stdout,
            stderr=stderr,
            locals_snapshot=snapshot,
            execution_time=time.perf_counter() - start_time,
            final_answer=self._last_final_answer,
            error=result.error,
        )

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return result.

        If kernel sandboxing is enabled and available, executes in a
        sandboxed subprocess. Otherwise uses in-process execution with
        Python-level restrictions (AST validation, restricted builtins).
        """
        if self._cleaned_up:
            return REPLResult(
                stdout="",
                stderr="Error: REPL has been cleaned up. Create a new instance.",
                locals_snapshot={},
                execution_time=0.0,
                final_answer=None,
                error="Error: REPL has been cleaned up. Create a new instance.",
            )

        # Dispatch to kernel sandbox if available
        if self._kernel_sandbox is not None:
            return self._execute_kernel_sandboxed(code)

        start_time = time.perf_counter()
        self._last_final_answer = None

        with self._exec_lock:
            old_cwd = os.getcwd()
            try:
                os.chdir(self.temp_dir)
            except OSError:
                pass  # temp_dir may not exist in edge cases

            try:
                with self._capture_output() as (stdout_buf, stderr_buf):
                    try:
                        combined = {**self.globals, **self.locals}
                        self._run_with_timeout(code, combined)

                        # Update locals with new variables
                        for key, value in combined.items():
                            if key not in self.globals and not key.startswith("_"):
                                self.locals[key] = value

                        self._restore_scaffold()

                        stdout = stdout_buf.getvalue()
                        stderr = stderr_buf.getvalue()
                    except TimeoutError:
                        stdout = stdout_buf.getvalue()
                        stderr = (
                            stderr_buf.getvalue()
                            + f"\nTimeoutError: Execution exceeded {self.timeout_seconds}s limit"
                        )
                    except Exception as e:
                        stdout = stdout_buf.getvalue()
                        stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
            finally:
                try:
                    os.chdir(old_cwd)
                except OSError:
                    pass

        # Build locals snapshot (type + repr for display)
        snapshot = {}
        for k, v in self.locals.items():
            if not k.startswith("_") and k not in ("context",):
                try:
                    r = repr(v)
                    snapshot[k] = r[:200] if len(r) > 200 else r
                except Exception:
                    snapshot[k] = f"<{type(v).__name__}>"

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals_snapshot=snapshot,
            execution_time=time.perf_counter() - start_time,
            final_answer=self._last_final_answer,
            error=stderr.strip() if stderr.strip() else None,
        )

    def add_context(self, payload: dict | list | str, name: str = "context") -> None:
        """Add context variable to REPL. Large payloads use temp file for memory efficiency."""
        text = payload if isinstance(payload, str) else json.dumps(payload)
        if len(text) > 100_000:
            # Write to temp file and load via REPL (per RLM paper: disk-based for large contexts)
            path = os.path.join(self.temp_dir, f"{name}.txt")
            with open(path, "w") as f:
                f.write(text)
            self.locals[name] = text  # still make available directly
            self.locals[f"_{name}_path"] = path  # expose path for programmatic access
        else:
            self.locals[name] = payload

    @staticmethod
    def _scrub_module_builtins(namespace: dict) -> None:
        """Remove user-imported modules from the namespace to prevent escape.

        Previously this replaced __import__ on shared module objects, but that
        contaminates process-global module state (modules are singletons in
        sys.modules). Instead, we remove module references from the namespace
        so they can't be used between REPL calls.
        """
        import types
        to_remove = [
            k for k, v in namespace.items()
            if isinstance(v, types.ModuleType) and not k.startswith("_")
            and k not in ("__builtins__",)
        ]
        for k in to_remove:
            del namespace[k]

    def _run_with_timeout(self, code: str, namespace: dict) -> None:
        """Execute code with a timeout using SIGALRM (Unix) or thread fallback.

        M6: Thread-based fallback cannot kill orphaned threads stuck in tight
        CPU loops. The real fix is subprocess isolation. SIGALRM path is reliable.
        """
        timeout = int(self.timeout_seconds) or _DEFAULT_TIMEOUT_SECONDS

        # Use SIGALRM on Unix for reliable timeout of tight loops
        if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
            def _alarm_handler(signum, frame):
                raise TimeoutError(f"Execution exceeded {timeout}s limit")

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)
            # C1 fix: patch builtins.__import__ during execution so that
            # module.__builtins__['__import__'] also returns the safe version.
            _original_import = builtins.__import__
            builtins.__import__ = _safe_import
            try:
                _run_in_namespace(code, namespace)
            finally:
                builtins.__import__ = _original_import
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                self._scrub_module_builtins(namespace)
        else:
            # Fallback: thread-based timeout (cannot interrupt tight CPU loops)
            result_holder: dict[str, Any] = {"error": None}

            def _target():
                _original_import = builtins.__import__
                builtins.__import__ = _safe_import
                try:
                    _run_in_namespace(code, namespace)
                except Exception as e:
                    result_holder["error"] = e
                finally:
                    builtins.__import__ = _original_import
                    self._scrub_module_builtins(namespace)

            t = threading.Thread(target=_target, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                raise TimeoutError(f"Execution exceeded {timeout}s limit")
            if result_holder["error"] is not None:
                raise result_holder["error"]

    def cleanup(self) -> None:
        """Clean up temp directory, kernel sandbox, and reset state."""
        self._cleaned_up = True
        if self._kernel_sandbox is not None:
            try:
                self._kernel_sandbox.cleanup()
            except Exception:
                pass
            self._kernel_sandbox = None
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
        if hasattr(self, "globals"):
            self.globals.clear()
        if hasattr(self, "locals"):
            self.locals.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()
