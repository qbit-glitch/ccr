"""Context packer — builds minimal context packs using the RLM REPL pattern.

Orchestrates: repo index + REPL program + LLM-guided ranking → ContextPack
"""

from __future__ import annotations

import json
import logging
import re

from ccr.context.indexer import RepoIndex
from ccr.core.exceptions import ModelError
from ccr.utils.parsing import extract_json_string
from ccr.context.prompts import (
    CONTEXT_PACKING_SYSTEM,
    SYMBOL_EXTRACTION_SYSTEM,
)
from ccr.core.types import ContextPack
from ccr.models.base import BaseLMClient
from ccr.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


class ContextPacker:
    """Builds minimal ContextPacks for a given task.

    Uses a cheap sub-model (Qwen) for semantic ranking.
    Does NOT use the full RLM REPL loop — instead runs a simplified
    programmatic pipeline: extract → search → rank → slice.

    This is the "REPL brain" from the architecture, simplified for v1.
    The full RLM REPL integration can be added in v2.
    """

    def __init__(
        self,
        repo_index: RepoIndex,
        sub_client: BaseLMClient,
        token_budget: int = 8000,
        max_search_candidates: int = 50,
        min_relevance_score: float = 0.3,
    ):
        self.index = repo_index
        self.sub_client = sub_client
        self.token_budget = token_budget
        self.max_search_candidates = max_search_candidates
        self.min_relevance_score = min_relevance_score

    def pack(self, task: str, memory_context: str = "") -> ContextPack:
        """Build a context pack for the given task.

        Pipeline:
        1. Extract symbols/keywords from task (LLM call)
        2. Search repo index programmatically (zero tokens)
        3. Rank candidates by relevance (LLM call)
        4. Slice to fit token budget
        5. Return ContextPack
        """
        logger.info(f"Packing context for: {task[:80]}...")

        # Step 1: Extract search terms
        search_terms = self._extract_search_terms(task)
        logger.info(f"Search terms: {search_terms}")

        # Step 2: Programmatic search (zero LLM tokens)
        candidates = self._search_candidates(search_terms)
        logger.info(f"Found {len(candidates)} candidate files")

        if not candidates:
            return ContextPack(
                task_description=task,
                files=[],
                symbols=[],
                memory_context=memory_context,
                total_tokens=estimate_tokens(memory_context),
            )

        # Step 3: Rank candidates with LLM
        ranked = self._rank_candidates(task, candidates)
        logger.info(f"Ranked {len(ranked)} relevant files")

        # Step 4: Slice to fit budget
        pack = self._slice_to_budget(task, ranked, memory_context)
        logger.info(f"Context pack: {len(pack.files)} files, {pack.total_tokens} tokens")

        return pack

    def _extract_search_terms(self, task: str) -> dict[str, list[str]]:
        """Use sub-model to extract symbols and keywords from task."""
        try:
            response = self.sub_client.completion(
                messages=[
                    {"role": "system", "content": SYMBOL_EXTRACTION_SYSTEM},
                    {"role": "user", "content": task},
                ],
                max_tokens=500,
            )
            return json.loads(extract_json_string(response))
        except ModelError as e:
            logger.warning(f"Symbol extraction model failed: {e}")
            words = re.findall(r"\b\w{3,}\b", task)
            return {"symbols": [], "keywords": words[:10], "file_patterns": []}
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Symbol extraction failed: {e}")
            words = re.findall(r"\b\w{3,}\b", task)
            return {"symbols": [], "keywords": words[:10], "file_patterns": []}

    def _search_candidates(self, terms: dict[str, list[str]]) -> list[dict]:
        """Search repo index programmatically. Zero LLM tokens."""
        all_results: dict[str, dict] = {}

        for symbol in terms.get("symbols", []):
            for result in self.index.search(symbol):
                path = result["path"]
                if path not in all_results or result["score"] > all_results[path]["score"]:
                    all_results[path] = result

        for keyword in terms.get("keywords", []):
            for result in self.index.search(keyword):
                path = result["path"]
                if path not in all_results:
                    all_results[path] = result
                else:
                    all_results[path]["score"] += result["score"]

        for pattern in terms.get("file_patterns", []):
            for result in self.index.search("", file_glob=pattern):
                path = result["path"]
                if path not in all_results:
                    all_results[path] = result

        results = sorted(all_results.values(), key=lambda r: r["score"], reverse=True)
        return results[:self.max_search_candidates]

    def _rank_candidates(self, task: str, candidates: list[dict]) -> list[dict]:
        """Use sub-model to rank candidates by relevance."""
        if len(candidates) <= 5:
            # Too few to bother ranking
            return candidates

        # Build compact file list for ranking
        file_list = "\n".join(
            f"- {c['path']} ({c['language']}, {c['lines']}L) symbols: {', '.join(c.get('symbols', [])[:5])}"
            for c in candidates[:30]  # cap to avoid huge prompts
        )

        prompt = f"Task: {task}\n\nFiles:\n{file_list}"

        try:
            response = self.sub_client.completion(
                messages=[
                    {"role": "system", "content": CONTEXT_PACKING_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
            )
            ranked = json.loads(extract_json_string(response))
            if isinstance(ranked, list):
                # Map back to candidates with content access
                ranked_paths = {r["path"]: r.get("relevance", 0.5) for r in ranked}
                for c in candidates:
                    if c["path"] in ranked_paths:
                        c["relevance"] = ranked_paths[c["path"]]
                    else:
                        c["relevance"] = 0.0
                candidates.sort(key=lambda c: c.get("relevance", 0), reverse=True)
                return [c for c in candidates if c.get("relevance", 0) >= self.min_relevance_score]
        except ModelError as e:
            logger.warning(f"Ranking model failed: {e}")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Ranking failed: {e}")

        return candidates[:15]

    def _slice_to_budget(
        self,
        task: str,
        ranked: list[dict],
        memory_context: str,
    ) -> ContextPack:
        """Select files that fit within token budget."""
        budget = self.token_budget
        used = estimate_tokens(memory_context)
        selected_files: list[tuple[str, str]] = []
        symbols: list[str] = []

        for candidate in ranked:
            path = candidate["path"]
            content = self.index.get_file(path)
            if content is None:
                continue

            file_tokens = estimate_tokens(content)
            if used + file_tokens > budget:
                # Try to include a truncated version
                remaining_chars = (budget - used) * 4  # rough chars
                if remaining_chars > 200:
                    content = content[:remaining_chars] + "\n... [truncated]"
                    file_tokens = estimate_tokens(content)
                else:
                    continue

            selected_files.append((path, content))
            symbols.extend(candidate.get("symbols", [])[:5])
            used += file_tokens

            if used >= budget * 0.95:
                break

        return ContextPack(
            task_description=task,
            files=selected_files,
            symbols=list(dict.fromkeys(symbols)),
            memory_context=memory_context,
            total_tokens=used,
        )

