"""Shared utilities for migration phases.

Provides _write_sentinel, _backup_file, and common imports
used across all migration phase modules.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _write_sentinel(ccr_root: str) -> None:
    """Write .migrated sentinel after successful full migration."""
    sentinel = os.path.join(ccr_root, ".migrated")
    with open(sentinel, "w", encoding="utf-8") as f:
        from datetime import datetime, timezone
        f.write(datetime.now(timezone.utc).isoformat())


def _backup_file(path: str) -> str | None:
    """Rename a flat file to .bak. Returns new path or None if missing."""
    if not os.path.isfile(path):
        return None
    bak = path + ".bak"
    if os.path.exists(bak):
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        bak = f"{path}.bak.{ts}"
    os.rename(path, bak)
    logger.info("Backed up %s → %s", path, bak)
    return bak
