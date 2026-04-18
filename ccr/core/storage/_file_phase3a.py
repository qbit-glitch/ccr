"""Phase 3a file-backend mixin: commits, rolling summaries, branches."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_COMMIT_HEADER_RE = re.compile(
    r"## \[(C\d{3,})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|[^|]*\|\s*(.*)",
)


class FilePhase3aMixin:
    """Commits, rolling summaries, and branches methods.

    Requires self.ccr_root and self._lock from FileStorageBackend.
    """

    # ── Commits (Phase 3a) ─────────────────────────────────────

    def _commits_path(self, branch: str) -> str:
        return os.path.join(self.ccr_root, "branches", branch, "commits.md")

    def _parse_all_commits(self, branch: str) -> list[dict]:
        path = self._commits_path(branch)
        if not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if not content:
            return []
        parts = re.split(r"(?=## \[C\d{3,}\])", content)
        results = []
        for part in parts:
            part = part.strip()
            m = _COMMIT_HEADER_RE.match(part)
            if not m:
                continue
            data: dict[str, Any] = {
                "id": m.group(1),
                "timestamp": m.group(2),
                "title": m.group(3).strip(),
                "raw_block": part,
            }
            for field, pattern in [
                ("what", r"\*\*What\*\*(?:\s*\[evolved\])?\s*:\s*(.*)"),
                ("why", r"\*\*Why\*\*:\s*(.*)"),
                ("next_step", r"\*\*Next\*\*:\s*(.*)"),
                ("author", r"\*\*Author\*\*:\s*(.*)"),
                ("ota_trace", r"\*\*OTA Trace\*\*:\s*(.*)"),
            ]:
                fm = re.search(pattern, part)
                data[field] = fm.group(1).strip() if fm else ""
            score_m = re.search(r"\*\*Score\*\*:\s*([\d.]+)", part)
            data["score"] = float(score_m.group(1)) if score_m else None
            files_m = re.search(r"\*\*Files\*\*:\s*(.*)", part)
            files_str = files_m.group(1).strip() if files_m else ""
            data["files"] = [f.strip() for f in files_str.split(",") if f.strip() and f.strip() != "(none)"] if files_str else []
            patterns_m = re.search(r"\*\*Patterns\*\*:\s*(.*)", part)
            data["patterns"] = [p.strip() for p in patterns_m.group(1).split("|") if p.strip()] if patterns_m else []
            data["branch"] = branch
            data.setdefault("next", data.get("next_step", ""))
            data.setdefault("stored_score", data.get("score"))
            results.append(data)
        return results

    def commit_insert(self, branch: str, data: dict) -> None:
        path = self._commits_path(branch)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raw_block = data.get("raw_block", "")
        if not raw_block:
            return

        with self._lock:
            content = ""
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    content = f.read()

            # TOCTOU: re-derive correct ID from content inside lock
            matches = re.findall(r"\[C(\d{3,})\]", content)
            correct_id = f"C{max(int(m) for m in matches) + 1:03d}" if matches else "C001"
            id_match = re.search(r"## \[(C\d{3,})\]", raw_block)
            if id_match:
                stale_id = id_match.group(1)
                if stale_id != correct_id:
                    raw_block = raw_block.replace(f"[{stale_id}]", f"[{correct_id}]", 1)

            anchor = "# Milestone Journal\n\n"
            idx = content.find(anchor)
            if idx >= 0:
                insert_at = idx + len(anchor)
                content = content[:insert_at] + raw_block + content[insert_at:]
            else:
                content = content + "\n" + raw_block

            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    def commit_get(self, branch: str, commit_id: str) -> dict | None:
        for c in self._parse_all_commits(branch):
            if c["id"] == commit_id:
                return c
        return None

    def commit_list(self, branch: str, limit: int = 10, offset: int = 0) -> list[dict]:
        all_commits = self._parse_all_commits(branch)
        return all_commits[offset:offset + limit]

    def commit_get_next_id(self, branch: str) -> str:
        path = self._commits_path(branch)
        if not os.path.isfile(path):
            return "C001"
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if not content:
            return "C001"
        matches = re.findall(r"\[C(\d{3,})\]", content)
        if not matches:
            return "C001"
        latest = max(int(m) for m in matches)
        return f"C{latest + 1:03d}"

    def commit_update(self, branch: str, commit_id: str, updates: dict) -> bool:
        return False

    def commit_search_text(self, branch: str, term: str, max_results: int = 5) -> list[dict]:
        term_lower = term.lower()
        results = []
        for c in self._parse_all_commits(branch):
            text = f"{c.get('title', '')} {c.get('what', '')} {c.get('why', '')} {c.get('next_step', '')}"
            if term_lower in text.lower():
                results.append(c)
                if len(results) >= max_results:
                    break
        return results

    def commit_search_with_snippet(
        self, branch: str, term: str, max_results: int = 5,
    ) -> list[dict]:
        """File backend has no FTS5 — always returns []."""
        return []

    def commit_upsert_vector(self, commit_id: str, vector: list[float]) -> None:
        """File backend keeps using gzip JSON via MemoryManager._embed_commit."""
        return None

    def commit_semantic_search(
        self, branch: str, query_vec: list[float], top_k: int,
    ) -> list[dict]:
        """File backend has no KNN — always returns []."""
        return []

    def commit_count(self, branch: str) -> int:
        path = self._commits_path(branch)
        if not os.path.isfile(path):
            return 0
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return len(re.findall(r"## \[C\d{3,}\]", content))

    # ── Rolling Summaries (Phase 3a) ──────────────────────────

    def rolling_summary_get(self, branch: str) -> str:
        path = self._commits_path(branch)
        if not os.path.isfile(path):
            return ""
        with open(path, encoding="utf-8") as f:
            content = f.read()
        m = re.search(r"## Rolling Summary\n(.*?)(?=\n---|\n# |\Z)", content, re.DOTALL)
        if m:
            summary = m.group(1).strip()
            return "" if summary == "(none yet)" else summary
        return ""

    def rolling_summary_set(self, branch: str, summary: str) -> None:
        path = self._commits_path(branch)
        if not os.path.isfile(path):
            return
        with self._lock:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            content = re.sub(
                r"(## Rolling Summary\n).*?(?=\n---|\n# |\Z)",
                lambda m: f"{m.group(1)}{summary}\n",
                content,
                count=1,
                flags=re.DOTALL,
            )
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    # ── Branches (Phase 3a) ────────────────────────────────────

    def _registry_path(self) -> str:
        return os.path.join(self.ccr_root, "registry.md")

    def branch_create(self, name: str, data: dict) -> None:
        pass

    def branch_get(self, name: str) -> dict | None:
        try:
            import yaml
        except ImportError:
            return None
        meta_path = os.path.join(self.ccr_root, "metadata.yaml")
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        for b in meta.get("branches", []):
            if b.get("name") == name:
                return b
        return None

    def branch_list(self, status: str | None = None) -> list[dict]:
        try:
            import yaml
        except ImportError:
            return []
        meta_path = os.path.join(self.ccr_root, "metadata.yaml")
        if not os.path.isfile(meta_path):
            return []
        with open(meta_path, encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        branches = meta.get("branches", [])
        if status:
            return [b for b in branches if b.get("status") == status]
        return branches

    def branch_update(self, name: str, updates: dict) -> bool:
        return False

    def branch_update_status(self, name: str, status: str) -> bool:
        return False
