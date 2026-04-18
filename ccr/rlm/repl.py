"""Sandboxed Python REPL for the RLM execution layer.

Adapted from vendor/rlm LocalREPL — simplified for CCR.
Key difference: no TCP sockets. LLM clients injected directly.

Security note: Uses Python's exec() intentionally within a restricted namespace
with _SAFE_BUILTINS. This is the core of the RLM paper — the model writes code
that executes in a sandboxed REPL to programmatically inspect context.

Security primitives (AST validation, restricted builtins, safe imports) are in
repl_security.py to keep this file focused on the CCRRepl class.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

from ccr.core.types import REPLResult
from ccr.rlm.repl_security import (
    _BoundedStringIO,
    _DEFAULT_TIMEOUT_SECONDS,
    _RESERVED_NAMES,
    _SAFE_BUILTINS,
    _make_restricted_open,
    _run_in_namespace,
    _safe_import,
    _validate_ast,
)


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
        logger.warning(msg)
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
            try:
                _run_in_namespace(code, namespace)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                self._scrub_module_builtins(namespace)
        else:
            # Fallback: thread-based timeout (cannot interrupt tight CPU loops)
            result_holder: dict[str, Any] = {"error": None}

            def _target():
                try:
                    _run_in_namespace(code, namespace)
                except Exception as e:
                    result_holder["error"] = e
                finally:
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
