"""Optional OpenTelemetry and local JSONL spans for CCR."""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator


def _enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class SpanRecord:
    name: str
    started_at: str
    duration_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""


def _write_local_span(project_root: str, record: SpanRecord) -> None:
    if not project_root or not _enabled(os.environ.get("CCR_TRACE_LOCAL", "")):
        return
    ccr_root = os.path.join(project_root, ".ccr")
    if not os.path.isdir(ccr_root):
        return
    path = os.path.join(ccr_root, "traces.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.__dict__, sort_keys=True, default=str))
        fh.write("\n")


@contextlib.contextmanager
def ccr_span(
    name: str,
    *,
    project_root: str = "",
    attributes: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Create an optional OTel span and optional local JSONL span.

    No dependency is required. If opentelemetry is unavailable or
    CCR_OTEL_ENABLED is not set, this is a no-op except for optional local spans
    when CCR_TRACE_LOCAL=1.
    """
    attrs = dict(attributes or {})
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    status = "ok"
    error = ""
    otel_cm = contextlib.nullcontext()
    if _enabled(os.environ.get("CCR_OTEL_ENABLED", "")):
        try:
            from opentelemetry import trace  # type: ignore
            tracer = trace.get_tracer("ccr")
            otel_cm = tracer.start_as_current_span(name, attributes=attrs)
        except Exception:
            otel_cm = contextlib.nullcontext()
    try:
        with otel_cm:
            yield
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        _write_local_span(project_root, SpanRecord(
            name=name,
            started_at=started_at,
            duration_ms=duration_ms,
            attributes=attrs,
            status=status,
            error=error,
        ))
