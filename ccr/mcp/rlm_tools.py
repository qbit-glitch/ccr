"""RLM Sandbox Tools — rlm_init, rlm_execute, rlm_finalize."""

from __future__ import annotations

import re
import time
import uuid

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.mcp.server import mcp
from ccr.mcp_types import (
    RlmExecuteResult,
    RlmFinalizeResult,
    RlmInitResult,
)
from ccr.rlm.repl import CCRRepl

# Import server module to access _repl via attribute (mutable global)
import ccr.mcp.server as _srv


# ===========================================================================
# Internal helpers
# ===========================================================================


def _get_repl_for_session(session_id: str) -> object:
    """Look up REPL by session_id, falling back to _srv._repl for backward compat.

    Raises ToolError if session_id is non-empty and not found.
    """
    if session_id:
        repl = _srv._repl_sessions.get(session_id)
        if repl is None:
            active = list(_srv._repl_sessions.keys())
            raise ToolError(
                f"Session '{session_id}' not found. "
                f"Active sessions: {active}"
            )
        return repl
    # Backward compat: no session_id → use legacy _repl (always set by rlm_init).
    # This preserves the original single-session contract exactly.
    return _srv._repl


def _get_session_age_warning(session_id: str, threshold_hours: float = 24.0) -> str:
    """Return a warning string if the session TTL is older than threshold_hours, else ''."""
    ttl = _srv._repl_session_ttl.get(session_id)
    if ttl is None:
        return ""
    age_hours = (time.time() - ttl) / 3600.0
    if age_hours > threshold_hours:
        return f"[Warning: session '{session_id}' is {age_hours:.1f}h old]"
    return ""


# ===========================================================================
# RLM Sandbox Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def rlm_init(
    task_prompt: str,
    timeout_seconds: int = 30,
    session_id: str = "",
) -> RlmInitResult:
    """Initialize a sandboxed Python REPL for structured problem-solving.

    Provides the REPL execution component from the RLM paper (arXiv:2512.24601).
    Note: The paper's autonomous generate-execute loop (Algorithm 1) is NOT active
    in MCP mode — Claude Code drives the iteration loop manually via rlm_execute
    calls. This is the REPL substrate only, not the full RLM system.

    Sets up a REPL with these variables and tools pre-loaded:
        - task_prompt: Your problem statement (string)
        - context: Repo index metadata (dict with file paths, symbols, imports)
        - get_file(path): Fetch full content of any indexed file
        - search_repo(query): Search files by content/symbol/path
        - estimate_tokens(text): Estimate token count
        - FINAL_VAR(name): Signal completion and return a variable's value
        - SHOW_VARS(): List all user-created variables

    Multiple named sessions can coexist. Use session_id to route rlm_execute
    and rlm_finalize calls to the correct session.

    Use this for complex analysis that benefits from iterative exploration.

    Args:
        task_prompt: The problem or question to solve.
        timeout_seconds: Maximum execution time per rlm_execute call, in seconds (default 30).
        session_id: Optional session name. Auto-generated as "rlm-<12hex>" when empty.
            Re-using an existing session_id replaces that session.
    """
    try:
        # Auto-generate session_id if not provided
        if not session_id:
            session_id = "rlm-" + uuid.uuid4().hex[:12]

        with _srv._state_lock:
            idx = _srv._ensure_index()

            # Track whether we are replacing an active session
            session_replaced = False

            # Clean up any existing session with the same session_id
            existing = _srv._repl_sessions.get(session_id)
            if existing is not None:
                try:
                    existing.cleanup()
                except Exception:
                    pass
                session_replaced = True

            # Also clean up the legacy _repl slot when we're replacing it
            # (backward compat: single-session callers use _repl directly)
            if _srv._repl is not None and not session_replaced:
                # Only clean up legacy slot if no named sessions exist
                if not _srv._repl_sessions:
                    _srv._repl.cleanup()
                    session_replaced = True

            # Kernel sandbox runs code in a subprocess, which means in-process tools
            # (search_repo, get_file, etc.) are NOT available there. Use Python-level
            # sandboxing (AST validation + restricted builtins) for the RLM REPL,
            # which needs these tools. The kernel sandbox is available for standalone
            # code execution via KernelSandbox directly.
            new_repl = CCRRepl(
                repo_index=idx,
                project_root=_srv._project_root,
                use_kernel_sandbox=False,
                timeout_seconds=float(timeout_seconds),
            )
            new_repl.locals["task_prompt"] = task_prompt

            # Store in named sessions dict
            _srv._repl_sessions[session_id] = new_repl
            _srv._repl_session_ttl[session_id] = time.time()

            # Also update legacy _repl for backward compat (single-session callers)
            _srv._repl = new_repl

        # Load playbook as variable if available
        pb = _srv._ensure_playbook()
        pb_text = pb.serialize()
        if pb_text.strip():
            _srv._repl_sessions[session_id].locals["playbook"] = pb_text

        prompt_preview = task_prompt[:120] + "..." if len(task_prompt) > 120 else task_prompt
        file_count = len(idx.files) if idx.files else 0

        text = (
            f"REPL initialized. Session: {session_id}\n"
            f"- task_prompt: {len(task_prompt)} chars — \"{prompt_preview}\"\n"
            f"- context: repo metadata ({file_count} files indexed)\n"
            f"- Tools: get_file(), search_repo(), estimate_tokens(), FINAL_VAR(), SHOW_VARS()\n"
            f"\nUse rlm_execute(session_id=\"{session_id}\") to run code."
            f" Use rlm_finalize(session_id=\"{session_id}\") when done."
            f" Timeout: {timeout_seconds}s."
        )
        if session_replaced:
            text = "Warning: Previous REPL session was replaced. " + text

        return RlmInitResult(
            session_id=session_id,
            file_count=file_count,
            session_replaced=session_replaced,
            message=text,
        )
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


def _summarize_stdout(stdout: str, threshold: int = 1000) -> str:
    """Summarize long stdout per RLM paper Section 3 ('Metadata-only stdout').

    If stdout exceeds *threshold* chars, replace with a metadata summary
    showing line/char counts plus the first and last 5 lines.  Short output
    (at or below the threshold) is returned unchanged.

    This enforces the paper's principle that stdout should carry metadata
    (lengths, counts, type info) rather than full content, saving tokens.
    """
    if len(stdout) <= threshold:
        return stdout

    lines = stdout.splitlines()
    line_count = len(lines)
    char_count = len(stdout)

    head = "\n".join(lines[:5])
    tail = "\n".join(lines[-5:]) if line_count > 10 else ""

    parts = [f"[stdout truncated: {line_count} lines, {char_count} chars]"]
    parts.append(head)
    if tail:
        parts.append("...")
        parts.append(tail)

    return "\n".join(parts)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def rlm_execute(
    code: str,
    metadata_only: bool = True,
    output_limit: int = 1000,
    session_id: str = "",
) -> RlmExecuteResult:
    """Execute Python code in the sandboxed REPL.

    The REPL persists variables across calls. Use it to:
        - Explore the repo: search_repo("pattern"), get_file("path")
        - Process data: parse, filter, transform results
        - Build answers incrementally across multiple execute calls

    Stdout is captured and returned. Variables persist between calls.
    By default, long stdout is summarized to metadata per the RLM paper
    (Section 3, 'Metadata-only stdout') to save tokens.

    Args:
        code: Python code to execute.
        metadata_only: If True (default), stdout exceeding output_limit chars is
            replaced with a metadata summary (line count, char count, first/last
            5 lines). Set to False to return full stdout regardless of length.
        output_limit: Character threshold for stdout summarization when
            metadata_only is True (default 1000).
        session_id: Session to execute in. If empty, uses the most recent session
            or falls back to the legacy single session.
    """
    try:
        with _srv._state_lock:
            repl_ref = _get_repl_for_session(session_id)
            if repl_ref is None:
                raise ToolError("REPL not initialized. Call rlm_init first.")

        result = repl_ref.execute_code(code)

        parts = []
        if result.stdout.strip():
            stdout = result.stdout.strip()
            if metadata_only:
                stdout = _summarize_stdout(stdout, threshold=output_limit)
            parts.append(f"stdout: {stdout}")

        if result.error:
            parts.append(f"error: {result.error}")

        if result.final_answer is not None:
            parts.append(f"FINAL_ANSWER: {result.final_answer}")

        if result.locals_snapshot:
            var_summary = ", ".join(
                f"{k}: {v[:60]}" for k, v in list(result.locals_snapshot.items())[:20]
            )
            parts.append(f"vars: {var_summary}")

        parts.append(f"time: {result.execution_time:.3f}s")

        # Append age warning if session is old
        if session_id:
            age_warn = _get_session_age_warning(session_id)
            if age_warn:
                parts.append(age_warn)

        text = "\n".join(parts)
        return RlmExecuteResult(
            has_error=bool(result.error),
            has_final_answer=result.final_answer is not None,
            message=text,
        )
    except ValueError:
        raise  # User input validation — let MCP propagate
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def rlm_finalize(
    variable_name: str,
    keep_session: bool = False,
    session_id: str = "",
    variables: list[str] | None = None,
) -> RlmFinalizeResult:
    """Finalize the REPL session and return a variable's value as the result.

    Calls FINAL_VAR internally to extract and serialize the named variable.
    By default, cleans up the REPL after extraction. Set keep_session=True
    to extract a variable without destroying the session.

    For multi-variable extraction, pass variables=["x", "y", ...]. All variables
    are validated before cleanup (atomic: session is only removed when all succeed).

    Args:
        variable_name: Name of the variable to return (single-variable mode).
        keep_session: If True, preserve the REPL session after extraction so
            that rlm_execute or rlm_finalize can be called again (default False).
        session_id: Session to finalize. If empty, uses the most recent session
            or falls back to the legacy single session.
        variables: Optional list of variable names for multi-variable extraction.
            When provided, variable_name is ignored. Returns a JSON dict.
    """
    try:
        import json as _json

        with _srv._state_lock:
            repl_ref = _get_repl_for_session(session_id)
            if repl_ref is None:
                raise ToolError("REPL not initialized. Call rlm_init first.")

        # ----------------------------------------------------------------
        # Multi-variable extraction (atomic: validate all before cleanup)
        # ----------------------------------------------------------------
        if variables is not None:
            # Validate all variable names first
            for var in variables:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
                    raise ToolError(
                        f"Invalid variable name '{var}'. Must be a valid Python identifier."
                    )

            # Extract all values — do NOT cleanup until all succeed
            extracted: dict[str, str] = {}
            for var in variables:
                answer = repl_ref._final_var(var)
                if answer is None or str(answer).startswith("Error:"):
                    # Atomic: leave session intact on partial failure
                    raise ToolError(
                        f"Variable '{var}' not found in REPL session. "
                        f"Session preserved. Available: {[k for k in repl_ref.locals if not k.startswith('_')]}"
                    )
                extracted[var] = answer

            # All variables validated — now optionally cleanup
            if not keep_session:
                _cleanup_session(session_id, repl_ref)

            result_json = _json.dumps(extracted)
            return RlmFinalizeResult(
                variable_name=",".join(variables),
                message=result_json,
            )

        # ----------------------------------------------------------------
        # Single-variable extraction (original behavior)
        # ----------------------------------------------------------------

        # H1: Validate variable_name to prevent code injection
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', variable_name):
            raise ToolError(f"Invalid variable name '{variable_name}'. Must be a valid Python identifier.")

        # Call _final_var directly instead of constructing code string (H1)
        answer = repl_ref._final_var(variable_name)

        # A1 fix: check result BEFORE destroying the session so that a typo in
        # variable_name does not silently wipe all computed state.
        if answer is None or str(answer).startswith("Error:"):
            session_note = "Session preserved." if keep_session else "Call rlm_init to start a new session."
            raise ToolError(f"Variable '{variable_name}' not found in REPL session. {session_note}")

        # Only destroy session after confirming successful retrieval, and only when
        # keep_session is False.
        if not keep_session:
            _cleanup_session(session_id, repl_ref)

        if keep_session:
            answer = answer + " Session preserved — call rlm_finalize again or rlm_execute to continue."

        return RlmFinalizeResult(variable_name=variable_name, message=answer)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


def _cleanup_session(session_id: str, repl_ref: object) -> None:
    """Cleanup a REPL session atomically: remove from sessions dict and cleanup the repl."""
    with _srv._state_lock:
        repl_ref.cleanup()
        if session_id:
            _srv._repl_sessions.pop(session_id, None)
            _srv._repl_session_ttl.pop(session_id, None)
        # Reset legacy _repl if it points to the same repl
        if _srv._repl is repl_ref:
            _srv._repl = None  # M4: Reset _repl after cleanup
