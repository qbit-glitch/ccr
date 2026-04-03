"""Audit logging for CCR MCP tool calls.

Logs every tool invocation to `.ccr/audit.log` with timestamp, tool name,
args summary, and result status. Useful for enterprise compliance and debugging.
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_audit_log_path: str = ""


def configure_audit_log(ccr_root: str) -> None:
    """Set up file-based audit logging to .ccr/audit.log."""
    global _audit_log_path
    _audit_log_path = os.path.join(ccr_root, "audit.log")

    handler = logging.FileHandler(_audit_log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))

    audit_logger = logging.getLogger("ccr.audit")
    audit_logger.setLevel(logging.INFO)
    # Avoid duplicate handlers on re-init
    if not audit_logger.handlers:
        audit_logger.addHandler(handler)


def log_tool_call(tool_name: str, args: dict[str, Any], status: str, duration_ms: float) -> None:
    """Write one audit entry."""
    audit = logging.getLogger("ccr.audit")
    # Truncate large arg values for readability
    summary = {k: _truncate(v) for k, v in args.items()} if args else {}
    audit.info("%s\t%s\t%s\t%.0fms", tool_name, status, summary, duration_ms)


def audit_wrap(func: Callable) -> Callable:
    """Decorator that logs tool calls to the audit log."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            result = func(*args, **kwargs)
            duration = (time.monotonic() - t0) * 1000
            log_tool_call(func.__name__, kwargs, "ok", duration)
            return result
        except Exception as exc:
            duration = (time.monotonic() - t0) * 1000
            log_tool_call(func.__name__, kwargs, f"error:{type(exc).__name__}", duration)
            raise

    return wrapper


def _truncate(value: Any, max_len: int = 100) -> str:
    """Truncate value for audit log readability."""
    s = str(value)
    return s[:max_len] + "..." if len(s) > max_len else s
