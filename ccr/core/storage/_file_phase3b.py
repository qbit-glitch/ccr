"""Phase 3b file-backend mixin: links, patterns, triples, evolved summaries, clusters."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from ccr.core.storage._sqlite_utils import _utcnow

logger = logging.getLogger(__name__)


class FilePhase3bMixin:
    """Links, patterns, triples, evolved summaries, and clusters methods.

    Requires self.ccr_root and self._lock from FileStorageBackend.
    """

    # ── Links (Phase 3b) ──────────────────────────────────────

    def _links_path(self) -> str:
        return os.path.join(self.ccr_root, "commit_links.json")

    def _load_links_json(self) -> dict:
        path = self._links_path()
        if not os.path.isfile(path):
            return {"version": 1, "links": {}}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data.get("links"), dict):
                return {"version": 1, "links": {}}
            return data
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "links": {}}

    def _save_links_json(self, data: dict) -> None:
        path = self._links_path()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def link_insert_batch(self, source_id: str, links: list[dict]) -> None:
        with self._lock:
            data = self._load_links_json()
            if source_id not in data["links"]:
                data["links"][source_id] = {}
            for link in links:
                lt = link["link_type"]
                if lt not in data["links"][source_id]:
                    data["links"][source_id][lt] = []
                entry: dict[str, Any] = {
                    "target": link["target"],
                    "score": link.get("score", 0.0),
                    "created_at": link.get("created_at", _utcnow()),
                }
                if link.get("shared_files"):
                    entry["shared_files"] = link["shared_files"]
                if link.get("snippet"):
                    entry["snippet"] = link["snippet"]
                data["links"][source_id][lt].append(entry)
            self._save_links_json(data)

    def link_get_for_commit(self, commit_id: str) -> dict:
        data = self._load_links_json()
        return data["links"].get(commit_id, {})

    def link_get_all(self) -> dict:
        data = self._load_links_json()
        return data["links"]

    def link_prune(self, max_nodes: int) -> int:
        with self._lock:
            data = self._load_links_json()
            links = data.get("links", {})
            if len(links) <= max_nodes:
                return 0

            def _commit_num(cid: str) -> int:
                m = re.search(r"\d+$", cid)
                return int(m.group()) if m else 0

            sorted_ids = sorted(links.keys(), key=_commit_num)
            evict_count = len(links) - max_nodes
            evict_set = set(sorted_ids[:evict_count])
            for cid in evict_set:
                del links[cid]
            for typed_links in links.values():
                for lt in list(typed_links.keys()):
                    typed_links[lt] = [
                        e for e in typed_links[lt]
                        if e.get("target", "") not in evict_set
                    ]
                    if not typed_links[lt]:
                        del typed_links[lt]
            self._save_links_json(data)
            return evict_count

    # ── Patterns (Phase 3b) ───────────────────────────────────

    def _patterns_path(self) -> str:
        return os.path.join(self.ccr_root, "patterns.json")

    def _load_patterns_json(self) -> dict:
        path = self._patterns_path()
        if not os.path.isfile(path):
            return {"version": 1, "patterns": {}, "next_id": 1}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data.get("patterns"), dict):
                return {"version": 1, "patterns": {}, "next_id": 1}
            return data
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "patterns": {}, "next_id": 1}

    def _save_patterns_json(self, data: dict) -> None:
        path = self._patterns_path()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def pattern_load_all(self) -> dict:
        with self._lock:
            return self._load_patterns_json()

    def pattern_save_all(self, data: dict) -> None:
        with self._lock:
            self._save_patterns_json(data)

    def pattern_get(self, pattern_id: str) -> dict | None:
        data = self._load_patterns_json()
        return data["patterns"].get(pattern_id)

    def pattern_update(self, pattern_id: str, updates: dict) -> bool:
        with self._lock:
            data = self._load_patterns_json()
            if pattern_id not in data["patterns"]:
                return False
            data["patterns"][pattern_id].update(updates)
            self._save_patterns_json(data)
            return True

    def pattern_get_next_id(self) -> int:
        data = self._load_patterns_json()
        return data.get("next_id", 1)

    # ── Triples (Phase 3b) ────────────────────────────────────

    def _triples_path(self) -> str:
        return os.path.join(self.ccr_root, "triples.json")

    def _load_triples_json(self) -> dict:
        path = self._triples_path()
        if not os.path.isfile(path):
            return {"version": 1, "triples": []}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data.get("triples"), list):
                return {"version": 1, "triples": []}
            return data
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "triples": []}

    def _save_triples_json(self, data: dict) -> None:
        path = self._triples_path()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def triple_insert_batch(self, triples: list[dict]) -> int:
        with self._lock:
            data = self._load_triples_json()
            existing_keys = {
                (t["subject"], t["predicate"], t["object"])
                for t in data["triples"]
            }
            added = 0
            for t in triples:
                key = (t["subject"], t["predicate"], t["object"])
                if key not in existing_keys:
                    data["triples"].append(t)
                    existing_keys.add(key)
                    added += 1
            if added:
                self._save_triples_json(data)
            return added

    def triple_list(
        self, top_k: int = 10, commit_id: str | None = None,
        entity: str | None = None,
    ) -> list[dict]:
        data = self._load_triples_json()
        result = data["triples"]
        if commit_id:
            result = [t for t in result if t.get("source_commit") == commit_id]
        if entity:
            result = [
                t for t in result
                if t.get("subject") == entity or t.get("object") == entity
            ]
        return list(reversed(result[-top_k:])) if result else []

    def triple_search(self, query: str, top_k: int = 10) -> list[dict]:
        data = self._load_triples_json()
        q = query.lower()
        matched = [
            t for t in data["triples"]
            if q in t.get("subject", "").lower()
            or q in t.get("predicate", "").lower()
            or q in t.get("object", "").lower()
        ]
        return list(reversed(matched[-top_k:]))

    def triple_count(self) -> int:
        data = self._load_triples_json()
        return len(data["triples"])

    # ── Evolved Summaries (Phase 3b) ──────────────────────────

    def _evolved_path(self) -> str:
        return os.path.join(self.ccr_root, "evolved_summaries.json")

    def _load_evolved_json(self) -> dict:
        path = self._evolved_path()
        if not os.path.isfile(path):
            return {"version": 1, "evolved": {}}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data.get("evolved"), dict):
                return {"version": 1, "evolved": {}}
            return data
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "evolved": {}}

    def _save_evolved_json(self, data: dict) -> None:
        path = self._evolved_path()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def evolved_summary_get(self, commit_id: str) -> dict | None:
        data = self._load_evolved_json()
        return data["evolved"].get(commit_id)

    def evolved_summary_set(self, commit_id: str, data: dict) -> None:
        with self._lock:
            store = self._load_evolved_json()
            store["evolved"][commit_id] = data
            self._save_evolved_json(store)

    def evolved_summary_all(self) -> dict:
        data = self._load_evolved_json()
        return data["evolved"]

    # ── Clusters (Phase 3b) ───────────────────────────────────

    def _clusters_path(self) -> str:
        return os.path.join(self.ccr_root, "commit_clusters.json")

    def _load_clusters_json(self) -> dict:
        path = self._clusters_path()
        if not os.path.isfile(path):
            return {"version": 1, "clusters": [], "commit_to_cluster": {}}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data.get("clusters"), list):
                return {"version": 1, "clusters": [], "commit_to_cluster": {}}
            return data
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "clusters": [], "commit_to_cluster": {}}

    def _save_clusters_json(self, data: dict) -> None:
        path = self._clusters_path()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def cluster_save(self, clusters: list[dict]) -> None:
        with self._lock:
            commit_to_cluster: dict[str, str] = {}
            for cl in clusters:
                for cid in cl.get("commit_ids", []):
                    commit_to_cluster[cid] = cl.get("id", "")
            data = {
                "version": 1,
                "clusters": clusters,
                "commit_to_cluster": commit_to_cluster,
            }
            self._save_clusters_json(data)

    def cluster_load(self) -> dict:
        return self._load_clusters_json()
