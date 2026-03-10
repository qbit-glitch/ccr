"""CCR exception hierarchy — structured errors for every subsystem.

All CCR-specific exceptions inherit from CCRError. Each subsystem has its own
base class so callers can catch at the granularity they need:

    CCRError
    +-- ConfigError              — invalid configuration at startup
    +-- ModelError               — LLM API failures
    |   +-- ModelTimeoutError    — request timed out
    |   +-- ModelRateLimitError  — rate limit / 429
    |   +-- ModelAuthError       — invalid API key / 401/403
    |   +-- ModelConnectionError — can't reach the endpoint
    +-- PackingError             — context packing failures
    +-- RoutingError             — classification / routing failures
    +-- PlaybookError            — ACE playbook operations
    |   +-- PlaybookIOError      — playbook read/write failures
    |   +-- PlaybookParseError   — malformed playbook text
    +-- MemoryError_             — GCC memory operations (underscore to avoid builtin clash)
    +-- HookError                — hook handler failures
"""

from __future__ import annotations


class CCRError(Exception):
    """Base exception for all CCR errors."""

    def __init__(self, message: str, *, recoverable: bool = True, detail: str = ""):
        super().__init__(message)
        self.recoverable = recoverable
        self.detail = detail


# --- Configuration ---

class ConfigError(CCRError):
    """Invalid or missing configuration."""

    def __init__(self, message: str, field: str = "", **kwargs):
        super().__init__(message, recoverable=False, **kwargs)
        self.field = field


# --- Model / LLM ---

class ModelError(CCRError):
    """Base for LLM API errors."""

    def __init__(self, message: str, model: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.model = model


class ModelTimeoutError(ModelError):
    """LLM request timed out."""
    pass


class ModelRateLimitError(ModelError):
    """Rate limit exceeded (HTTP 429)."""

    def __init__(self, message: str, retry_after: float | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ModelAuthError(ModelError):
    """Authentication failed (HTTP 401/403)."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, recoverable=False, **kwargs)


class ModelConnectionError(ModelError):
    """Cannot reach the model endpoint."""
    pass


# --- Context Packing ---

class PackingError(CCRError):
    """Context packing failed."""
    pass


# --- Routing ---

class RoutingError(CCRError):
    """Task classification or routing failed."""
    pass


# --- ACE Playbook ---

class PlaybookError(CCRError):
    """ACE playbook operation failed."""
    pass


class PlaybookIOError(PlaybookError):
    """Playbook file read/write failure."""
    pass


class PlaybookParseError(PlaybookError):
    """Malformed playbook text."""
    pass


# --- GCC Memory ---

class MemoryError_(CCRError):
    """GCC memory operation failed (underscore avoids builtin clash)."""
    pass


# --- Hooks ---

class HookError(CCRError):
    """Hook handler failure."""

    def __init__(self, message: str, event: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.event = event
