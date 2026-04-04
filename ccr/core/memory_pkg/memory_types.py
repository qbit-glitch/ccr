"""Shared types, imports, and template constants for MemoryManager mixins.

All imports needed by the full MemoryManager are re-exported here so that
mixin files can import from this single module.  Template strings and the
EvolvedSummary dataclass live at module level (not in a class).
"""

from __future__ import annotations

import fcntl
import heapq
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import yaml

from ccr.context.embeddings import get_embedding_model, load_embeddings, quick_cosine, save_embeddings
from ccr.core.types import CCRConfig, CommitLink
from ccr.utils.parsing import extract_json_string

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvolvedSummary:
    """Mutable overlay for a commit's summary (A-MEM S3.3 Eq.7)."""

    commit_id: str
    evolved_what: str        # LLM-rewritten version of original what
    evolution_reason: str    # why the summary was evolved
    evolved_at: str          # ISO timestamp
    source_commit_id: str    # the new commit that triggered evolution
    original_what: str       # preserved original


# ---------------------------------------------------------------------------
# Templates (from open-gcc bootstrap.ts)
# ---------------------------------------------------------------------------

MAIN_MD_TEMPLATE = """# Project Context

## Current Focus
(Auto-populated after first commit)

## Recent Milestones
(none yet)

## Open Branches
(none)
"""

REGISTRY_TEMPLATE = """# Branch Registry

## Active Branch
main

## Branch History
| Branch | Created | Status |
|--------|---------|--------|
| main | {date} | active |
"""

COMMITS_TEMPLATE = """# Branch: {branch}

## Purpose
{purpose}

## Hypothesis
{hypothesis}

## Conclusion
(Fill in at merge time -- success/failure/partial)

## Rolling Summary
(none yet)

---

# Milestone Journal

"""

MAIN_COMMITS_TEMPLATE = """# Branch: main

## Rolling Summary
(none yet)

# Milestone Journal

"""

SUMMARY_TEMPLATE = """# Branch: {branch}

## Purpose
{purpose}

## Status
active

## Parent
{parent}

## Created
{date}

## Key Decisions
(none yet)
"""

METADATA_TEMPLATE = {
    "version": 1,
    "created": "",
    "proactive_commits": True,
    "branches": [],
    "file_tree": [],
    "dependencies": [],
    "config": {
        "language": "unknown",
        "framework": "unknown",
    },
}
