"""Filesystem helpers that behave the same on Windows as on POSIX."""

from __future__ import annotations

import os
import sys
import time

# Windows refuses to replace a file while another handle has it open (antivirus,
# the search indexer, or CCR's own background repo indexer reading it). Those
# locks are brief, so a short retry turns a crash into a few milliseconds' wait.
_RETRY_ERRNOS = {5, 32}  # ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION
_ATTEMPTS = 20
_DELAY_S = 0.05


def replace_file(src: str, dst: str) -> None:
    """os.replace(src, dst), retried briefly on Windows sharing violations."""
    if sys.platform != "win32":
        os.replace(src, dst)
        return
    for attempt in range(_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in _RETRY_ERRNOS or attempt == _ATTEMPTS - 1:
                raise
            time.sleep(_DELAY_S * (attempt + 1))
