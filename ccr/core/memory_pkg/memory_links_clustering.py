"""LinkClusterMixin — EverMemOS-inspired thematic clustering for MemoryManager.

Computes connected components from entity + semantic links to form thematic
clusters. Inspired by EverMemOS MemScene clustering (arXiv:2601.02163).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, deque

__all__ = ["LinkClusterMixin"]

logger = logging.getLogger(__name__)


class LinkClusterMixin:
    """Thematic clustering methods for MemoryManager."""

    def compute_clusters(
        self, min_cluster_size: int = 2, link_score_threshold: float = 0.3
    ) -> list[dict]:
        """Compute thematic clusters from the cross-link graph.

        Uses connected components (BFS) on entity + semantic links.

        Args:
            min_cluster_size: Minimum commits to form a cluster.
            link_score_threshold: Minimum link score to consider.

        Returns:
            List of cluster dicts: {id, name, commit_ids, top_keywords}
        """
        links_data = self._load_links()
        all_links = links_data.get("links", {})

        # Build adjacency from entity + semantic links
        adjacency: dict[str, set[str]] = {}
        for src, typed_links in all_links.items():
            if src not in adjacency:
                adjacency[src] = set()
            for link_type in ("entity", "semantic"):
                for entry in typed_links.get(link_type, []):
                    tgt = entry.get("target", "")
                    score = entry.get("score", 0.0)
                    if tgt and score >= link_score_threshold:
                        adjacency.setdefault(src, set()).add(tgt)
                        adjacency.setdefault(tgt, set()).add(src)

        # BFS connected components
        visited: set[str] = set()
        components: list[list[str]] = []
        for node in adjacency:
            if node in visited:
                continue
            component: list[str] = []
            queue: deque[str] = deque([node])
            visited.add(node)
            while queue:
                n = queue.popleft()
                component.append(n)
                for neighbor in adjacency.get(n, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(component) >= min_cluster_size:
                components.append(sorted(component))

        # Name clusters by top keywords
        branch = self.get_active_branch()
        clusters: list[dict] = []
        for i, commit_ids in enumerate(components):
            keywords = self._extract_cluster_keywords(branch, commit_ids)
            name = (
                " ".join(w.title() for w in keywords[:4])
                if keywords
                else f"Cluster {i + 1}"
            )
            clusters.append({
                "id": f"CL{i + 1:03d}",
                "name": name,
                "commit_ids": commit_ids,
                "top_keywords": keywords[:6],
            })

        self._save_clusters(clusters)
        return clusters

    def _extract_cluster_keywords(
        self, branch: str, commit_ids: list[str]
    ) -> list[str]:
        """Extract top keywords from a set of commits."""
        _cluster_stop_words = frozenset({
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "has", "have", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "that", "this",
            "it", "not", "no", "if", "when", "then", "than", "so", "as", "up",
            "out", "about", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "only", "own", "same", "into",
            "over", "after",
        })

        word_counts: Counter = Counter()
        for cid in commit_ids:
            block = self._find_commit_by_id(branch, cid)
            if not block:
                continue
            for line in block.split("\n"):
                if line.startswith("**What**:") or line.startswith("## [C"):
                    text = line.split("|")[-1] if "|" in line else line
                    text = re.sub(r"\*\*\w+\*\*:", "", text)
                    text = re.sub(r"\[C\d+\]", "", text)
                    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
                    for w in words:
                        if w not in _cluster_stop_words:
                            word_counts[w] += 1

        return [w for w, _ in word_counts.most_common(10)]

    def _get_clusters_path(self) -> str:
        import os
        return os.path.join(self.ccr_root, "commit_clusters.json")

    def _save_clusters(self, clusters: list[dict]) -> None:
        """Save clusters to JSON using atomic write with file locking."""
        path = self._get_clusters_path()
        commit_to_cluster: dict[str, str] = {}
        for cl in clusters:
            for cid in cl.get("commit_ids", []):
                commit_to_cluster[cid] = cl["id"]

        data = {
            "version": 1,
            "clusters": clusters,
            "commit_to_cluster": commit_to_cluster,
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def _load_clusters(self) -> dict:
        """Load clusters from JSON. Returns default if missing/corrupt."""
        path = self._get_clusters_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
        if not raw:
            return {"version": 1, "clusters": [], "commit_to_cluster": {}}
        try:
            data = json.loads(raw)
            if not isinstance(data.get("clusters"), list):
                return {"version": 1, "clusters": [], "commit_to_cluster": {}}
            return data
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return {"version": 1, "clusters": [], "commit_to_cluster": {}}

    def format_clusters_for_context(self) -> str:
        """Format clusters for inclusion in gcc_context output."""
        data = self._load_clusters()
        clusters = data.get("clusters", [])
        if not clusters:
            return ""
        lines = ["# Thematic Clusters"]
        for cl in clusters:
            commit_str = ", ".join(cl["commit_ids"][:5])
            if len(cl["commit_ids"]) > 5:
                commit_str += f" (+{len(cl['commit_ids']) - 5} more)"
            lines.append(f"- **{cl['name']}** [{cl['id']}]: {commit_str}")
        return "\n".join(lines)
