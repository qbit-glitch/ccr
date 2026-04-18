"""Storage abstraction layer for CCR.

Provides a unified interface for flat-file and SQLite backends,
gated by CCRConfig.storage_backend ("files" or "sqlite").
"""

from __future__ import annotations

from ccr.core.storage.base import StorageBackend
from ccr.core.storage.sqlite_backend import SqliteConnectionManager

__all__ = [
    "StorageBackend",
    "SqliteConnectionManager",
    "get_backend",
]


def get_backend(
    backend_type: str,
    ccr_root: str,
    global_ccr_root: str | None = None,
) -> StorageBackend:
    """Factory: return the correct storage backend based on config.

    Args:
        backend_type: "files" or "sqlite".
        ccr_root: Path to .ccr/ directory.
        global_ccr_root: Path to ~/.ccr/ (for global playbook).
    """
    if backend_type == "sqlite":
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        return SqliteStorageBackend(ccr_root, global_ccr_root)

    from ccr.core.storage.file_backend import FileStorageBackend
    return FileStorageBackend(ccr_root, global_ccr_root)
