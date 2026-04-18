"""Phase 3c file-backend mixin: discussions, session/phase summaries, summary meta, project state."""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


class FilePhase3cMixin:
    """Discussions, session summaries, phase summaries, summary meta, and project state methods.

    Requires self.ccr_root and self._lock from FileStorageBackend.
    """

    # ── Phase 3c: Discussions ─────────────────────────────────────

    def _discussions_path(self, branch: str) -> str:
        return os.path.join(self.ccr_root, "branches", branch, "discussions.md")

    _DISCUSSION_HEADER_RE = re.compile(
        r"^## \[(D\d{3,})\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \| (.+)$",
        re.MULTILINE,
    )

    def _parse_discussions(self, branch: str) -> list[dict]:
        path = self._discussions_path(branch)
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return []

        blocks = re.split(r"\n---\n", content)
        results = []
        for block in blocks:
            block = block.strip()
            m = self._DISCUSSION_HEADER_RE.match(block)
            if not m:
                continue
            rec: dict = {
                "id": m.group(1),
                "timestamp": m.group(2),
                "topic": m.group(3).strip(),
            }
            for field, pat in [
                ("hypothesis", r"\*\*Hypothesis\*\*:\s*(.+)"),
                ("alternatives", r"\*\*Alternatives\*\*:\s*(.+)"),
                ("decision", r"\*\*Decision\*\*:\s*(.+)"),
                ("rationale", r"\*\*Rationale\*\*:\s*(.+)"),
                ("uncertainty", r"\*\*Uncertainty\*\*:\s*(.+)"),
                ("linked_commit", r"\*\*Linked Commit\*\*:\s*(.+)"),
            ]:
                fm = re.search(pat, block)
                rec[field] = fm.group(1).strip() if fm else ""
            results.append(rec)
        return results

    def discussion_insert(self, branch: str, data: dict) -> None:
        path = self._discussions_path(branch)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        block_lines = [
            f"## [{data['id']}] {data.get('timestamp', '')} | {data.get('topic', '')}",
            f"**Hypothesis**: {data.get('hypothesis', '')}",
            f"**Alternatives**: {data.get('alternatives', '')}",
            f"**Decision**: {data.get('decision', '')}",
            f"**Rationale**: {data.get('rationale', '')}",
        ]
        if data.get("uncertainty"):
            block_lines.append(f"**Uncertainty**: {data['uncertainty']}")
        if data.get("linked_commit"):
            block_lines.append(f"**Linked Commit**: {data['linked_commit']}")
        block = "\n".join(block_lines) + "\n"

        with self._lock:
            existing = ""
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        existing = f.read()
                except OSError:
                    pass
            new_content = block + "\n---\n\n" + existing if existing else block
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(new_content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    def discussion_list(
        self, branch: str, search: str | None = None,
        topic: str | None = None, date_range: list[str] | None = None,
    ) -> list[dict]:
        records = self._parse_discussions(branch)
        if search:
            sl = search.lower()
            records = [
                r for r in records
                if sl in r.get("topic", "").lower()
                or sl in r.get("hypothesis", "").lower()
                or sl in r.get("decision", "").lower()
                or sl in r.get("rationale", "").lower()
            ]
        if topic:
            tl = topic.lower()
            records = [r for r in records if tl in r.get("topic", "").lower()]
        if date_range and len(date_range) >= 2:
            records = [
                r for r in records
                if date_range[0] <= r.get("timestamp", "") <= date_range[1]
            ]
        return records

    def discussion_get_next_id(self, branch: str) -> str:
        records = self._parse_discussions(branch)
        if not records:
            return "D001"
        max_num = 0
        for r in records:
            try:
                num = int(r["id"][1:])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass
        return f"D{max_num + 1:03d}"

    def discussion_search_text(
        self, branch: str, term: str, max_results: int = 10,
    ) -> list[dict]:
        """Substring-match discussions by any text field. Case-insensitive."""
        records = self._parse_discussions(branch)
        tl = term.lower()
        hits = [
            r for r in records
            if tl in r.get("topic", "").lower()
            or tl in r.get("hypothesis", "").lower()
            or tl in r.get("alternatives", "").lower()
            or tl in r.get("decision", "").lower()
            or tl in r.get("rationale", "").lower()
            or tl in r.get("uncertainty", "").lower()
        ]
        return hits[:max_results]

    # ── Phase 3c: Session Summaries ───────────────────────────────

    def _summaries_path(self, branch: str) -> str:
        return os.path.join(self.ccr_root, "branches", branch, "summaries.md")

    _SESSION_HEADER_RE = re.compile(
        r"## \[(S\d{3,})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*-\s*"
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|\s*(\S+)\s*\|\s*Session Summary",
    )

    def _parse_session_summaries(self, branch: str) -> list[dict]:
        path = self._summaries_path(branch)
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return []

        blocks = re.split(r"\n---\n", content)
        results = []
        for block in blocks:
            block = block.strip()
            m = self._SESSION_HEADER_RE.search(block)
            if not m:
                continue
            rec: dict = {
                "id": m.group(1),
                "start_date": m.group(2),
                "end_date": m.group(3),
                "branch": m.group(4),
            }
            for field, pat in [
                ("commit_range", r"\*\*Commits\*\*:\s*(.*)"),
                ("accomplished", r"\*\*Accomplished\*\*:\s*(.*)"),
                ("files_touched", r"\*\*Files touched\*\*:\s*(.*)"),
                ("key_decisions", r"\*\*Key decisions\*\*:\s*(.*)"),
                ("direction", r"\*\*Direction\*\*:\s*(.*)"),
            ]:
                fm = re.search(pat, block)
                rec[field] = fm.group(1).strip() if fm else ""
            results.append(rec)
        return results

    def session_summary_insert(self, branch: str, data: dict) -> None:
        path = self._summaries_path(branch)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        block = (
            f"## [{data['id']}] {data.get('start_date', '')} - "
            f"{data.get('end_date', '')} | {branch} | Session Summary\n"
            f"**Commits**: {data.get('commit_range', '')}\n"
            f"**Accomplished**: {data.get('accomplished', '')}\n"
            f"**Files touched**: {data.get('files_touched', '')}\n"
            f"**Key decisions**: {data.get('key_decisions', '')}\n"
            f"**Direction**: {data.get('direction', '')}\n"
        )

        with self._lock:
            existing = ""
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        existing = f.read()
                except OSError:
                    pass
            new_content = block + "\n---\n\n" + existing if existing else block
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(new_content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    def session_summary_list(self, branch: str, count: int = 3) -> list[dict]:
        records = self._parse_session_summaries(branch)
        return records[:count]

    def session_summary_get_next_id(self, branch: str) -> str:
        records = self._parse_session_summaries(branch)
        if not records:
            return "S001"
        max_num = 0
        for r in records:
            try:
                num = int(r["id"][1:])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass
        return f"S{max_num + 1:03d}"

    # ── Phase 3c: Phase Summaries ─────────────────────────────────

    def _phases_path(self) -> str:
        return os.path.join(self.ccr_root, "summaries", "phases.md")

    _PHASE_HEADER_RE = re.compile(
        r"## \[(P\d{3,})\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*-\s*"
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|\s*Phase Summary",
    )

    def _parse_phase_summaries(self) -> list[dict]:
        path = self._phases_path()
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return []

        blocks = re.split(r"\n---\n", content)
        results = []
        for block in blocks:
            block = block.strip()
            m = self._PHASE_HEADER_RE.search(block)
            if not m:
                continue
            rec: dict = {
                "id": m.group(1),
                "start_date": m.group(2),
                "end_date": m.group(3),
            }
            for field, pat in [
                ("scope", r"\*\*Scope\*\*:\s*(.*)"),
                ("goal", r"\*\*Goal\*\*:\s*(.*)"),
                ("outcome", r"\*\*Outcome\*\*:\s*(.*)"),
                ("accomplishments", r"\*\*Key accomplishments\*\*:\s*(.*)"),
                ("files_changed", r"\*\*Files changed\*\*:\s*(.*)"),
                ("branch_summary", r"\*\*Branch summary\*\*:\s*(.*)"),
            ]:
                fm = re.search(pat, block)
                rec[field] = fm.group(1).strip() if fm else ""
            results.append(rec)
        return results

    def phase_summary_insert(self, data: dict) -> None:
        path = self._phases_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        block = (
            f"## [{data['id']}] {data.get('start_date', '')} - "
            f"{data.get('end_date', '')} | Phase Summary\n"
            f"**Scope**: {data.get('scope', '')}\n"
            f"**Goal**: {data.get('goal', '')}\n"
            f"**Outcome**: {data.get('outcome', '')}\n"
            f"**Key accomplishments**: {data.get('accomplishments', '')}\n"
            f"**Files changed**: {data.get('files_changed', '')}\n"
            f"**Branch summary**: {data.get('branch_summary', '')}\n"
        )

        with self._lock:
            existing = ""
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        existing = f.read()
                except OSError:
                    pass
            new_content = block + "\n---\n\n" + existing if existing else block
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(new_content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    def phase_summary_list(self, count: int = 3) -> list[dict]:
        records = self._parse_phase_summaries()
        return records[:count]

    def phase_summary_get_next_id(self) -> str:
        records = self._parse_phase_summaries()
        if not records:
            return "P001"
        max_num = 0
        for r in records:
            try:
                num = int(r["id"][1:])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass
        return f"P{max_num + 1:03d}"

    # ── Phase 3c: Summary Meta ────────────────────────────────────

    def _summary_meta_path(self) -> str:
        return os.path.join(self.ccr_root, "summary_meta.yaml")

    def summary_meta_load(self) -> dict:
        path = self._summary_meta_path()
        if not os.path.isfile(path):
            return {
                "version": 1, "session": {},
                "phase": {"last_commit_id": None, "last_summary_id": None, "last_generated": None},
                "overview": {"last_generated": None, "phase_count_at_generation": 0},
            }
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def summary_meta_save(self, data: dict) -> None:
        path = self._summary_meta_path()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        try:
            import yaml
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            pass

    # ── Phase 3c: Project State ───────────────────────────────────

    def _project_state_path(self, key: str) -> str:
        if key == "overview":
            return os.path.join(self.ccr_root, "overview.md")
        return os.path.join(self.ccr_root, f"{key}.txt")

    def project_state_get(self, key: str) -> str | None:
        path = self._project_state_path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    def project_state_set(self, key: str, value: str) -> None:
        path = self._project_state_path(key)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with self._lock:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(value)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
