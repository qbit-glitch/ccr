"""LinkComputeMixin — Heuristic link computation for MemoryManager.

Extracts cross-links between commits using mechanical heuristics (zero LLM):
entity (shared files), causal (C### references), supersession (replacement
language), and semantic (dense cosine or word Jaccard fallback).

Inspired by A-MEM/MAGMA taxonomy (arXiv:2502.12110, 2601.03236).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from ccr.core.types import CommitLink

__all__ = ["LinkComputeMixin", "LINK_TYPES"]

# Module-level constant — avoids cross-mixin class attribute references
LINK_TYPES = ("entity", "causal", "supersession", "semantic")


class LinkComputeMixin:
    """Link computation heuristics for MemoryManager."""

    # --- Class-level constants ---

    _LINK_TYPES = LINK_TYPES

    _STOP_WORDS = frozenset({
        # Articles & determiners
        "the", "a", "an", "this", "that", "these", "those", "some", "any",
        "each", "every", "all", "both", "few", "more", "most", "other",
        # Prepositions
        "to", "for", "of", "in", "on", "at", "by", "from", "into", "about",
        "between", "through", "during", "before", "after", "above", "below",
        "up", "out", "off", "over", "under", "again", "further", "then",
        # Conjunctions
        "and", "or", "but", "nor", "yet", "so", "if", "when", "while",
        "because", "although", "than",
        # Pronouns
        "it", "its", "they", "them", "their", "we", "our", "you", "your",
        "he", "she", "his", "her", "who", "which", "what", "how",
        # Be/have/do
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "having",
        "do", "does", "did", "doing",
        # Modals
        "will", "would", "can", "could", "shall", "should", "may", "might",
        "must",
        # Common adverbs/adjectives
        "not", "no", "just", "also", "very", "only", "now", "here", "there",
        "where", "still", "already",
        # Common verbs (too generic for keywords)
        "get", "got", "set", "use", "used", "using", "make", "made",
    })
    _SUPERSESSION_KEYWORDS = re.compile(
        r"(?:replaced|superseded|reverted|refactored\s+from|deprecated|reworked|improved\s+upon)",
        re.IGNORECASE,
    )
    _COMMIT_ID_RE = re.compile(r"\b(C\d{3,})\b")

    # --- Static helpers ---

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Jaccard similarity between two sets."""
        if not a and not b:
            return 0.0
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    @staticmethod
    def _commit_text(title: str, what: str, why: str) -> str:
        """Canonical text representation of a commit for ONNX embedding."""
        return f"{title} {what} {why}".strip()

    @staticmethod
    def _extract_commit_references(text: str) -> list[str]:
        """Extract all commit ID references (C###) from text."""
        return re.findall(r"\b(C\d{3,})\b", text)

    @classmethod
    def _detect_supersession(cls, text: str) -> list[tuple[str, str]]:
        """Detect replacement language near commit IDs.

        Returns list of (commit_id, snippet) tuples.
        """
        results = []
        for m in cls._COMMIT_ID_RE.finditer(text):
            cid = m.group(1)
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 120)
            window = text[start:end]
            if cls._SUPERSESSION_KEYWORDS.search(window):
                snippet = text[max(0, m.start() - 40):min(len(text), m.end() + 40)].strip()
                results.append((cid, snippet))
        return results

    @classmethod
    def _extract_keywords(cls, text: str) -> set[str]:
        """Extract keywords from text, filtering stop words, short tokens, and pure digits."""
        return {w for w in re.findall(r"\w+", text.lower())
                if w not in cls._STOP_WORDS and len(w) > 2 and not w.isdigit()}

    @staticmethod
    def _add_link(data: dict, source: str, target: str, link: CommitLink) -> None:
        """Add a bidirectional link (A-MEM Zettelkasten). Deduplicates by higher score."""
        for src, tgt in [(source, target), (target, source)]:
            node = data["links"].setdefault(src, {})
            bucket = node.setdefault(link.link_type, [])
            for existing in bucket:
                if existing["target"] == tgt:
                    if link.score > existing.get("score", 0.0):
                        update: dict[str, Any] = {"target": tgt, "score": link.score}
                        if link.shared_files:
                            update["shared_files"] = link.shared_files
                        if link.snippet:
                            update["snippet"] = link.snippet
                        existing.update(update)
                    break
            else:
                entry = link.to_dict()
                entry["target"] = tgt
                if "created_at" not in entry:
                    entry["created_at"] = datetime.now(timezone.utc).isoformat()
                bucket.append(entry)

    @staticmethod
    def _parse_commit_block(text: str) -> dict:
        """Parse a single commit block (Markdown) into a dict with title/what/why."""
        result: dict[str, Any] = {}
        title_match = re.search(r"\[C\d{3,}\]\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*\|\s*branch:[\w-]+\s*\|\s*(.*)", text)
        if title_match:
            result["title"] = title_match.group(1).strip()
        for field in ("What", "Why", "Files", "Next"):
            m = re.search(rf"\*\*{field}\*\*:\s*(.*?)(?=\n\*\*(?:What|Why|Files|Next|Patterns|Score|OTA)\*\*|\n##|\Z)", text, re.DOTALL)
            if m:
                result[field.lower()] = m.group(1).strip()
        patterns_m = re.search(r"\*\*Patterns\*\*:\s*(.*)", text)
        if patterns_m:
            result["patterns"] = [p.strip() for p in patterns_m.group(1).strip().split("|") if p.strip()]
        return result

    # --- Link Computation ---

    def _compute_links(
        self,
        branch: str,
        commit_id: str,
        title: str,
        what: str,
        why: str,
        files_changed: list[str],
        next_step: str,
        new_vec=None,
    ) -> list[CommitLink]:
        """Compute heuristic cross-links for a new commit against recent history.

        All linking is mechanical (zero LLM calls). Scans the last k commits
        (config.link_scan_window, default 20).

        Link types:
        1. Entity links: file-set Jaccard > threshold
        2. Causal links: regex detection of C### IDs in text
        3. Supersession links: replacement language + C###
        4. Semantic links: dense cosine or word Jaccard fallback
        """
        recent = self._parse_recent_commit_data(branch, k=self.effective_link_scan_window)
        if not recent:
            return []

        new_files = {f.strip().lower() for f in files_changed if f.strip()}
        combined_text = f"{title} {what} {why} {next_step}"
        new_keywords = self._extract_keywords(combined_text)

        all_refs = set(self._extract_commit_references(combined_text))
        supersession_hits = {cid: snip for cid, snip in self._detect_supersession(combined_text)}

        existing_ids = {c.get("id", "") for c in recent if c.get("id")}
        all_refs = all_refs & existing_ids
        supersession_hits = {k: v for k, v in supersession_hits.items() if k in existing_ids}

        links: list[CommitLink] = []

        all_cached = self._load_commit_embeddings([c.get("id", "") for c in recent if c.get("id")])

        for commit in recent:
            cid = commit.get("id", "")
            if not cid or cid == commit_id:
                continue

            has_typed_link = False

            # 1. Entity links (shared files)
            old_files = {f.strip().lower() for f in commit.get("files", []) if f.strip()}
            file_sim = self._jaccard(new_files, old_files)
            if file_sim > self.config.link_entity_threshold:
                shared = sorted(new_files & old_files)
                links.append(CommitLink(
                    target=cid, link_type="entity", score=round(file_sim, 3),
                    shared_files=shared,
                ))
                has_typed_link = True

            # 2 & 3. Causal / supersession links
            if cid in supersession_hits:
                links.append(CommitLink(
                    target=cid, link_type="supersession", score=1.0,
                    snippet=supersession_hits[cid],
                ))
                has_typed_link = True
            elif cid in all_refs:
                idx = combined_text.find(cid)
                snippet = combined_text[max(0, idx - 40):min(len(combined_text), idx + len(cid) + 40)].strip()
                links.append(CommitLink(
                    target=cid, link_type="causal", score=1.0,
                    snippet=snippet,
                ))
                has_typed_link = True

            # 4. Semantic links
            if not has_typed_link:
                if new_vec is not None and cid in all_cached:
                    cached_vec = all_cached[cid]
                    if cached_vec.shape == new_vec.shape:
                        score = float(cached_vec @ new_vec)
                    else:
                        old_text = f"{commit.get('title', '')} {commit.get('what', '')} {commit.get('why', '')}".lower()
                        old_keywords = self._extract_keywords(old_text)
                        score = self._jaccard(new_keywords, old_keywords)
                else:
                    old_text = f"{commit.get('title', '')} {commit.get('what', '')} {commit.get('why', '')}".lower()
                    old_keywords = self._extract_keywords(old_text)
                    score = self._jaccard(new_keywords, old_keywords)
                if score > self.effective_link_semantic_threshold:
                    links.append(CommitLink(
                        target=cid, link_type="semantic", score=round(score, 3),
                    ))

        return links
