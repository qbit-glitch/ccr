"""RegistryMixin -- branch registry, metadata.yaml, and summary.md helpers."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class RegistryMixin:
    """Branch registry, metadata, and summary methods for MemoryManager.

    Expects the composite class to provide:
        self.ccr_root: str
        self.branches_dir: str
        self._locks: dict[str, threading.Lock]

    And methods from other mixins:
        self._get_branch_dir(branch) -> str
        self._read_file(path) -> str
        self._read_file_unlocked(path) -> str
        self._write_file_unlocked(path, content)
    """

    # --- _registry.md ---

    def _update_registry_active_branch(self, branch: str) -> None:
        path = os.path.join(self.branches_dir, "_registry.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                r"(## Active Branch\s*\n)\S+",
                lambda m: f"{m.group(1)}{branch}",
                content,
            )
            self._write_file_unlocked(path, content)

    def _add_branch_to_registry(self, name: str, date: str) -> None:
        path = os.path.join(self.branches_dir, "_registry.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            row = f"| {name} | {date} | active |"
            content = content.rstrip() + "\n" + row + "\n"
            self._write_file_unlocked(path, content)

    def _update_registry_branch_status(self, name: str, status: str) -> None:
        path = os.path.join(self.branches_dir, "_registry.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(
                rf"(\| {re.escape(name)} \| [^|]+ \| )\w+( \|)",
                lambda m: f"{m.group(1)}{status}{m.group(2)}",
                content,
            )
            self._write_file_unlocked(path, content)

    # --- main.md branch list ---

    def _add_branch_to_main_md(self, name: str) -> None:
        path = os.path.join(self.ccr_root, "main.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = content.replace("(none)", "")
            section_match = re.search(r"(## Open Branches\n)", content)
            if section_match:
                insert_at = section_match.end()
                content = content[:insert_at] + f"- {name}\n" + content[insert_at:]
            else:
                content += f"\n## Open Branches\n- {name}\n"
            self._write_file_unlocked(path, content)

    def _remove_branch_from_main_md(self, name: str) -> None:
        path = os.path.join(self.ccr_root, "main.md")
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = re.sub(rf"- {re.escape(name)}\n?", "", content)
            # If open branches section is now empty, add placeholder
            if "## Open Branches\n" in content:
                after = content.split("## Open Branches\n", 1)[1]
                next_section = re.search(r"\n## ", after)
                branch_text = after[: next_section.start()] if next_section else after
                if not branch_text.strip():
                    content = content.replace(
                        "## Open Branches\n" + branch_text,
                        "## Open Branches\n(none)\n",
                    )
            self._write_file_unlocked(path, content)

    # --- metadata.yaml ---

    def _get_metadata_path(self) -> str:
        return os.path.join(self.ccr_root, "metadata.yaml")

    def _load_metadata(self) -> dict:
        from ccr.core.memory_pkg.memory_types import METADATA_TEMPLATE

        path = self._get_metadata_path()
        with self._locks[path]:
            if not os.path.isfile(path):
                return dict(METADATA_TEMPLATE)
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or dict(METADATA_TEMPLATE)

    def _save_metadata(self, data: dict) -> None:
        """Save metadata atomically via tmp + fsync + os.replace (H4)."""
        path = self._get_metadata_path()
        with self._locks[path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            content = yaml.dump(data, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper)
            tmp_path = path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def _update_metadata_branch(self, name: str, status: str, created: str, parent: str) -> None:
        meta = self._load_metadata()
        meta.setdefault("branches", [])
        meta["branches"].append({
            "name": name, "status": status, "created": created, "parent": parent,
        })
        meta["version"] = meta.get("version", 0) + 1
        self._save_metadata(meta)

    def _update_metadata_branch_status(self, name: str, status: str) -> None:
        meta = self._load_metadata()
        for b in meta.get("branches", []):
            if b.get("name") == name:
                b["status"] = status
                break
        meta["version"] = meta.get("version", 0) + 1
        self._save_metadata(meta)

    def update_metadata_file_tree(self, file_list: list[str]) -> None:
        """Update metadata.yaml with the current file tree."""
        meta = self._load_metadata()
        meta["file_tree"] = file_list[:500]  # cap at 500 files
        self._save_metadata(meta)

    def update_metadata_dependencies(self, deps: list[str]) -> None:
        """Update metadata.yaml with project dependencies."""
        meta = self._load_metadata()
        meta["dependencies"] = deps
        self._save_metadata(meta)

    def update_metadata_config(self, language: str = "", framework: str = "") -> None:
        """Update metadata.yaml with project config."""
        meta = self._load_metadata()
        if language:
            meta.setdefault("config", {})["language"] = language
        if framework:
            meta.setdefault("config", {})["framework"] = framework
        self._save_metadata(meta)

    # --- summary.md per branch ---

    def _get_summary_path(self, branch: str) -> str:
        return os.path.join(self._get_branch_dir(branch), "summary.md")

    def _read_branch_summary(self, branch: str) -> str:
        return self._read_file(self._get_summary_path(branch))

    def _update_summary_status(self, branch: str, status: str) -> None:
        path = self._get_summary_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path)
            if content:
                content = re.sub(r"(## Status\n)\S+.*", lambda m: f"{m.group(1)}{status}", content)
                self._write_file_unlocked(path, content)

    def _add_key_decision_to_summary(self, branch: str, decision: str) -> None:
        path = self._get_summary_path(branch)
        with self._locks[path]:
            content = self._read_file_unlocked(path) or ""
            content = content.replace("(none yet)", "")
            section_match = re.search(r"(## Key Decisions\n)", content)
            if section_match:
                insert_at = section_match.end()
                content = content[:insert_at] + f"- {decision}\n" + content[insert_at:]
            self._write_file_unlocked(path, content)
