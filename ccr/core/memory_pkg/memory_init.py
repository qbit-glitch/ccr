"""InitMixin -- constructor state, schema overrides, ensure_structure, get_active_branch."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class InitMixin:
    """Initialisation and bootstrap methods for MemoryManager.

    Expects the composite class to provide (via other mixins or itself):
        self.project_root: str
        self.ccr_root: str
        self.branches_dir: str
        self.config: CCRConfig
        self._locks: dict[str, threading.Lock]
        self._evolved_summaries: dict
        self._schema_overrides: dict
        self._commit_index: dict

    And methods from other mixins:
        self._write_if_missing(path, content)
        self._get_metadata_path() -> str
        self._save_metadata(data)
        self._load_evolved_summaries()
        self._read_file(path) -> str
        self._add_to_gitignore(path)
        self._get_summary_meta_path() -> str
        self._save_summary_meta(data)
        self._default_summary_meta() -> dict
    """

    def _init_state(self, project_root: str, config) -> None:
        """Set all instance variables that __init__ would set.

        Called by the facade __init__ so that mixin code can rely on these
        attributes existing.
        """
        from ccr.core.types import CCRConfig

        self.project_root = os.path.abspath(project_root)
        self.ccr_root = os.path.join(self.project_root, ".ccr")
        self.branches_dir = os.path.join(self.ccr_root, "branches")
        self.config = config or CCRConfig()
        self._locks: dict[str, Any] = defaultdict(lambda: __import__("threading").Lock())
        self.sub_client = None
        self._evolved_summaries: dict = {}
        self._schema_overrides: dict[str, Any] = {}
        self._commit_index: dict[str, dict[str, str]] = {}

        from ccr.core.storage import get_backend
        global_ccr = os.path.expanduser("~/.ccr")
        self._storage = get_backend(
            self.config.storage_backend, self.ccr_root, global_ccr,
        )

    def set_sub_client(self, client) -> None:
        """Set the sub-model client for LLM-regenerated rolling summaries."""
        self.sub_client = client

    def set_schema_overrides(self, overrides: dict[str, Any]) -> None:
        """Set schema-driven parameter overrides (ALMA-inspired).

        Called by MCP server when PlaybookSchema has non-default retrieval params.
        Supported keys: link_scan_window, link_semantic_threshold.
        """
        self._schema_overrides = dict(overrides)

    @property
    def effective_link_scan_window(self) -> int:
        """Return link_scan_window from schema override or config fallback."""
        return self._schema_overrides.get("link_scan_window", self.config.link_scan_window)

    @property
    def effective_link_semantic_threshold(self) -> float:
        """Return link_semantic_threshold from schema override or config fallback."""
        return self._schema_overrides.get("link_semantic_threshold", self.config.link_semantic_threshold)

    # --- Bootstrap ---

    def ensure_structure(self) -> bool:
        """Create .ccr/ layout if missing. Idempotent. Returns True if created."""
        # Import templates at call time to avoid circular imports
        from ccr.core.memory_pkg.memory_types import (
            MAIN_COMMITS_TEMPLATE,
            MAIN_MD_TEMPLATE,
            METADATA_TEMPLATE,
            REGISTRY_TEMPLATE,
        )

        main_branch_dir = os.path.join(self.branches_dir, "main")
        metadata_path = self._get_metadata_path()

        # SQLite backend may create .ccr/ before ensure_structure runs,
        # so check metadata.yaml as the canonical "was initialized" signal
        created = not os.path.isfile(metadata_path)

        if not os.path.isdir(self.ccr_root):
            os.makedirs(self.ccr_root, exist_ok=True)

        os.makedirs(main_branch_dir, exist_ok=True)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self._write_if_missing(
            os.path.join(self.ccr_root, "main.md"),
            MAIN_MD_TEMPLATE,
        )
        self._write_if_missing(
            os.path.join(self.branches_dir, "_registry.md"),
            REGISTRY_TEMPLATE.format(date=now),
        )
        self._write_if_missing(
            os.path.join(main_branch_dir, "commits.md"),
            MAIN_COMMITS_TEMPLATE,
        )
        self._write_if_missing(
            os.path.join(main_branch_dir, "log.md"),
            "",
        )

        # metadata.yaml
        metadata_path = self._get_metadata_path()
        if not os.path.isfile(metadata_path):
            meta = dict(METADATA_TEMPLATE)
            meta["created"] = now
            meta["branches"] = [{"name": "main", "status": "active", "created": now, "parent": None}]
            self._save_metadata(meta)

        # Hierarchical summaries directory (TiMem S3.1)
        summaries_dir = os.path.join(self.ccr_root, "summaries")
        os.makedirs(summaries_dir, exist_ok=True)

        # Initialize summary_meta.yaml if missing
        summary_meta_path = self._get_summary_meta_path()
        if not os.path.isfile(summary_meta_path):
            self._save_summary_meta(self._default_summary_meta())

        # Add .ccr to .gitignore if in a git repo
        gitignore = os.path.join(self.project_root, ".gitignore")
        if os.path.isdir(os.path.join(self.project_root, ".git")):
            self._add_to_gitignore(gitignore)

        # Load A-MEM evolved summaries overlay
        self._load_evolved_summaries()

        # Register in global project registry (global setup awareness)
        try:
            self._register_in_global_registry()
        except Exception:
            pass  # Non-critical — never block init

        return created

    def _register_in_global_registry(self) -> None:
        """Register this project in ~/.ccr/projects.json for global discovery."""
        import json
        from datetime import datetime, timezone

        global_ccr = os.path.expanduser("~/.ccr")
        os.makedirs(global_ccr, exist_ok=True)
        registry_path = os.path.join(global_ccr, "projects.json")

        projects_list: list[dict] = []
        if os.path.isfile(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    projects_list = json.loads(f.read())
            except (json.JSONDecodeError, OSError):
                projects_list = []

        abs_path = self.project_root
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for p in projects_list:
            if p.get("path") == abs_path:
                p["last_used"] = now
                break
        else:
            projects_list.append({
                "path": abs_path,
                "name": os.path.basename(abs_path),
                "last_used": now,
                "commit_count": 0,
            })

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(projects_list, f, indent=2)
            f.write("\n")

    # --- Branch Operations ---

    def get_active_branch(self) -> str:
        registry = self._read_file(os.path.join(self.branches_dir, "_registry.md"))
        if not registry:
            return "main"
        match = re.search(r"## Active Branch\s*\n(\S+)", registry)
        return match.group(1) if match else "main"
