"""Sandboxed Python REPL for the RLM execution layer.

Adapted from vendor/rlm LocalREPL — simplified for CCR.
Key difference: no TCP sockets. LLM clients injected directly.

Security note: Uses Python's exec() intentionally within a restricted namespace
with _SAFE_BUILTINS. This is the core of the RLM paper — the model writes code
that executes in a sandboxed REPL to programmatically inspect context.
"""

from __future__ import annotations

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

# Modules blocked from import inside the REPL sandbox.
_BLOCKED_MODULES = frozenset({
    "subprocess", "shutil", "signal", "ctypes", "multiprocessing",
    "pty", "fcntl", "termios", "resource",
})


def _safe_import(name: str, *args, **kwargs):
    """Restricted __import__ that blocks dangerous modules."""
    top_level = name.split(".")[0]
    if top_level in _BLOCKED_MODULES:
        raise ImportError(
            f"Module '{name}' is blocked in the CCR sandbox. "
            f"Blocked modules: {', '.join(sorted(_BLOCKED_MODULES))}"
        )
    return __import__(name, *args, **kwargs)


# Safe builtins — blocks eval/exec/compile/input, allows everything else.
_SAFE_BUILTINS: dict[str, Any] = {
    "print": print, "len": len, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
    "type": type, "isinstance": isinstance, "issubclass": issubclass,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "range": range,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "any": any, "all": all, "pow": pow, "divmod": divmod,
    "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
    "repr": repr, "ascii": ascii, "format": format, "hash": hash, "id": id,
    "iter": iter, "next": next, "slice": slice, "callable": callable,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "delattr": delattr,
    "dir": dir, "vars": vars,
    "bytes": bytes, "bytearray": bytearray, "memoryview": memoryview,
    "complex": complex, "object": object, "super": super,
    "property": property, "staticmethod": staticmethod, "classmethod": classmethod,
    "__import__": _safe_import, "open": open,
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
    # Python's built-in code execution in a restricted namespace
    builtins.__dict__  # ensure builtins loaded
    co = builtins.compile(code, "<ccr-repl>", "exec")
    builtins.eval(co, namespace, namespace)  # noqa: S307


class CCRRepl:
    """Sandboxed Python REPL with CCR tools injected.

    Tools available in the REPL:
        context           — repo index dict (file metadata, symbols, imports)
        get_file(path)    — fetch full content of any indexed file
        search_repo(q)    — search files by content/symbol/path
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
    ):
        self.sub_client = sub_client
        self.repo_index = repo_index
        self.subcall_fn = subcall_fn  # Callback for rlm_query (spawns child CCRRlm)
        self.timeout_seconds = timeout_seconds

        self._last_final_answer: str | None = None
        self._exec_lock = threading.Lock()
        self._cleaned_up = False
        self.temp_dir = tempfile.mkdtemp(prefix="ccr_repl_")

        # Namespace: globals (builtins + tools) and locals (user variables)
        self.globals: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS.copy()}
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

    def _get_file(self, path: str) -> str:
        """Get full content of an indexed file."""
        if self.repo_index is None:
            return "Error: No repo index loaded"
        if hasattr(self.repo_index, "get_file"):
            content = self.repo_index.get_file(path)
            return content if content is not None else f"Error: File not found: {path}"
        return "Error: Repo index does not support get_file()"

    def _search_repo(self, query: str, file_glob: str = "**/*") -> list[dict]:
        """Search files by content, symbol, or path pattern."""
        if self.repo_index is None:
            return []
        if hasattr(self.repo_index, "search"):
            return self.repo_index.search(query)
        return []

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        from ccr.utils.tokens import count_tokens
        return count_tokens(text)

    # --- Execution ---

    @contextmanager
    def _capture_output(self):
        """Capture stdout and stderr during code execution."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
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

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return result."""
        if self._cleaned_up:
            return REPLResult(
                stdout="",
                stderr="Error: REPL has been cleaned up. Create a new instance.",
                locals_snapshot={},
                execution_time=0.0,
                final_answer=None,
                error="Error: REPL has been cleaned up. Create a new instance.",
            )

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

    def _run_with_timeout(self, code: str, namespace: dict) -> None:
        """Execute code with a timeout using SIGALRM (Unix) or thread fallback."""
        timeout = int(self.timeout_seconds) or _DEFAULT_TIMEOUT_SECONDS

        # Use SIGALRM on Unix for reliable timeout of tight loops
        if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
            def _alarm_handler(signum, frame):
                raise TimeoutError(f"Execution exceeded {timeout}s limit")

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)
            try:
                _run_in_namespace(code, namespace)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Fallback: thread-based timeout (cannot interrupt tight CPU loops)
            result_holder: dict[str, Any] = {"error": None}

            def _target():
                try:
                    _run_in_namespace(code, namespace)
                except Exception as e:
                    result_holder["error"] = e

            t = threading.Thread(target=_target, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                raise TimeoutError(f"Execution exceeded {timeout}s limit")
            if result_holder["error"] is not None:
                raise result_holder["error"]

    def cleanup(self) -> None:
        """Clean up temp directory and reset state."""
        self._cleaned_up = True
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
        self.globals.clear()
        self.locals.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()
