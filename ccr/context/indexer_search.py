"""Search mixin for RepoIndex — BM25, semantic, hybrid, and chunk search.

Extracted from indexer.py to keep files under 400 lines.
"""

from __future__ import annotations

import fnmatch
import math
import re
import warnings
from collections import Counter
from typing import TYPE_CHECKING

from ccr.context.indexer_types import FileEntry

if TYPE_CHECKING:
    from ccr.context.embeddings import EmbeddingModel


class SearchMixin:
    """Mixin providing BM25, semantic, hybrid, and chunk-level search.

    Must be mixed into a class that has:
      - self.files: dict[str, FileEntry]
      - self._embeddings: dict[str, list[float]]
      - self._bm25_cache: dict | None
      - self._chunk_embeddings: dict[str, list[float]]
      - self.search(pattern, file_glob) method (keyword search)
    """

    # BM25 stop words — common English + Python noise terms
    _BM25_STOP_WORDS: frozenset[str] = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "not", "but", "if", "then", "else", "so",
        "this", "that", "it", "its", "self", "none", "true", "false",
        "import", "return", "def", "class",
    })

    @staticmethod
    def _tokenize_simple(text: str) -> list[str]:
        """Simple word tokenizer for BM25. Lowercases, filters stops."""
        return [
            w for w in re.findall(r"\w{2,}", text.lower())
            if w not in SearchMixin._BM25_STOP_WORDS
        ]

    @staticmethod
    def _file_summary(entry: FileEntry) -> str:
        """Generate embeddable summary for a file (A-RAG S3.1 adaptation).

        Combines path, symbols, and opening lines into a compact
        representation suitable for embedding (< 256 tokens).
        """
        parts = [entry.rel_path]
        if entry.symbols:
            parts.append(" ".join(entry.symbols[:20]))
        if entry._content:
            first_lines = "\n".join(entry._content.split("\n")[:10])
            parts.append(first_lines)
        return "\n".join(parts)

    def _build_bm25_cache(self) -> dict:
        """Pre-compute BM25 data structures from indexed file content.

        Computed lazily on first semantic search to avoid cost for
        keyword-only users. Cached in self._bm25_cache.
        """
        doc_term_counts: dict[str, Counter] = {}
        doc_lengths: dict[str, int] = {}
        doc_freqs: Counter = Counter()
        total_length = 0

        for rel, entry in self.files.items():
            if not entry._content:
                continue
            terms = self._tokenize_simple(entry._content)
            doc_term_counts[rel] = Counter(terms)
            doc_lengths[rel] = len(terms)
            total_length += len(terms)
            for term in set(terms):
                doc_freqs[term] += 1

        n = len(doc_term_counts)
        avg_doc_len = total_length / n if n > 0 else 0.0

        return {
            "doc_term_counts": doc_term_counts,
            "doc_lengths": doc_lengths,
            "doc_freqs": doc_freqs,
            "avg_doc_len": avg_doc_len,
            "N": n,
        }

    def bm25_search(
        self, query: str, top_k: int = 10, file_glob: str = "**/*"
    ) -> list[dict]:
        """BM25 scoring against file content. Zero external deps.

        Uses Okapi BM25 formula (k1=1.5, b=0.75).
        CCR's own zero-dep fallback for semantic search when ONNX is unavailable.
        Not from the A-RAG paper (arXiv:2602.03442), which does not mention BM25.
        """
        if self._bm25_cache is None:
            self._bm25_cache = self._build_bm25_cache()

        cache = self._bm25_cache
        query_terms = self._tokenize_simple(query)
        if not query_terms:
            return []

        n = cache["N"]
        if n == 0:
            return []

        avg_dl = cache["avg_doc_len"]
        df = cache["doc_freqs"]
        k1, b = 1.5, 0.75

        # Pre-compute IDF for query terms
        idf: dict[str, float] = {}
        for term in query_terms:
            d = df.get(term, 0)
            idf[term] = math.log((n - d + 0.5) / (d + 0.5) + 1)

        results = []
        for rel, tf_counter in cache["doc_term_counts"].items():
            if file_glob != "**/*" and not fnmatch.fnmatch(rel, file_glob):
                continue

            score = 0.0
            dl = cache["doc_lengths"][rel]
            for term in query_terms:
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avg_dl)
                score += idf.get(term, 0) * numerator / denominator

            if score > 0:
                entry = self.files[rel]
                results.append({
                    "path": rel,
                    "score": round(score, 4),
                    "language": entry.language,
                    "symbols": entry.symbols[:10],
                    "size": entry.size_bytes,
                    "lines": entry.line_count,
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def fts5_search(
        self, query: str, db, top_k: int = 10,
    ) -> list[dict]:
        """FTS5 search via IndexDB. Falls back to BM25 if FTS5 unavailable."""
        if db is None or not db.fts_available:
            return self.bm25_search(query, top_k=top_k)
        raw = db.fts_search(query, top_k=top_k)
        results = []
        for r in raw:
            entry = self.files.get(r["path"])
            if entry is None:
                continue
            results.append({
                "path": r["path"],
                "score": r.get("score", 0),
                "language": entry.language,
                "symbols": entry.symbols[:10],
                "size": entry.size_bytes,
                "lines": entry.line_count,
            })
        return results

    def build_embeddings(self, model: EmbeddingModel) -> int:
        """Compute embeddings for all indexed files. Returns count.

        A-RAG S3.1: builds dense representations for semantic search.
        """
        summaries = []
        paths = []
        for rel, entry in self.files.items():
            if not entry._content:
                continue
            summaries.append(self._file_summary(entry))
            paths.append(rel)

        if not summaries:
            return 0

        # Batch embed (batch_size=64 for memory efficiency)
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i : i + batch_size]
            vecs = model.embed_batch(batch)
            all_embeddings.append(vecs)

        import numpy as np

        combined = np.vstack(all_embeddings)
        self._embeddings = {
            paths[i]: combined[i].tolist() for i in range(len(paths))
        }
        return len(self._embeddings)

    def semantic_search(
        self,
        query: str,
        model: EmbeddingModel,
        top_k: int = 10,
        file_glob: str = "**/*",
    ) -> list[dict]:
        """Dense embedding cosine similarity search (A-RAG S3.2, Eq 3)."""
        if not self._embeddings:
            return []

        import numpy as np

        query_vec = model.embed_query(query)

        # Build doc matrix from stored embeddings
        paths = []
        vecs = []
        for rel, emb in self._embeddings.items():
            if file_glob != "**/*" and not fnmatch.fnmatch(rel, file_glob):
                continue
            paths.append(rel)
            vecs.append(emb)

        if not vecs:
            return []

        doc_matrix = np.array(vecs, dtype=np.float32)
        scores = model.cosine_similarity(query_vec, doc_matrix)

        # Build results
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])
        results = []
        for idx, score in indexed[:top_k]:
            if score <= 0:
                continue
            rel = paths[idx]
            entry = self.files[rel]
            results.append({
                "path": rel,
                "score": round(float(score), 4),
                "language": entry.language,
                "symbols": entry.symbols[:10],
                "size": entry.size_bytes,
                "lines": entry.line_count,
            })

        return results

    def hybrid_search(
        self,
        query: str,
        model: "EmbeddingModel | None" = None,
        top_k: int = 10,
        file_glob: str = "**/*",
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        return_snippets: bool = False,
    ) -> list[dict]:
        """Combine keyword + semantic results via Reciprocal Rank Fusion (RRF).

        Uses ONNX embeddings if model + embeddings available, else BM25.
        RRF k=60 is the standard constant from Cormack et al. (2009) --
        avoids the normalisation instability of linear score combination.

        When return_snippets=True and chunk embeddings are available, uses
        chunk-level semantic search (A-RAG S3.2) and attaches snippet field
        to results.

        Note: keyword_weight / semantic_weight params are kept for API
        compatibility but are no longer used in scoring (RRF is rank-only).
        """
        if keyword_weight != 0.3 or semantic_weight != 0.7:
            warnings.warn(
                "keyword_weight and semantic_weight are deprecated and ignored; RRF scoring is used.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Keyword ranked list
        kw_results = self.search(query, file_glob=file_glob)

        # Snippet map from chunk search (if requested)
        snippet_map: dict[str, str] = {}

        # Semantic ranked list
        sem_results_ordered: list[dict] = []
        if return_snippets and model is not None and self._chunk_embeddings:
            # Use chunk-level semantic search (A-RAG S3.2)
            chunk_results = self.chunk_semantic_search(query, model, top_k=100)
            seen_paths: dict[str, float] = {}
            for r in chunk_results:
                path = r["path"]
                score = r["score"]
                # Keep best chunk score per file (for de-duplication rank ordering)
                if path not in seen_paths or score > seen_paths[path]:
                    seen_paths[path] = score
                    snippet_map[path] = r["snippet"]
            # Re-order by best chunk score so rank dict is meaningful
            sem_results_ordered = sorted(
                [{"path": p, "score": s} for p, s in seen_paths.items()],
                key=lambda r: r["score"],
                reverse=True,
            )
        elif model is not None and self._embeddings:
            sem_results_ordered = self.semantic_search(
                query, model, top_k=100, file_glob=file_glob
            )
        else:
            # BM25 fallback
            sem_results_ordered = self.bm25_search(query, top_k=100, file_glob=file_glob)

        # Reciprocal Rank Fusion (RRF k=60, Cormack et al. 2009)
        RRF_K = 60
        kw_rank = {r["path"]: i for i, r in enumerate(kw_results)}
        sem_rank = {r["path"]: i for i, r in enumerate(sem_results_ordered)}

        all_paths = set(kw_rank) | set(sem_rank)
        rrf_scores: dict[str, float] = {}
        for path in all_paths:
            score = 0.0
            if path in kw_rank:
                score += 1.0 / (RRF_K + kw_rank[path])
            if path in sem_rank:
                score += 1.0 / (RRF_K + sem_rank[path])
            rrf_scores[path] = score

        # Build merged result list, preserving all metadata
        all_results_meta: dict[str, dict] = {r["path"]: r for r in kw_results}
        for r in sem_results_ordered:
            if r["path"] not in all_results_meta:
                all_results_meta[r["path"]] = r

        combined: list[dict] = []
        for path in all_paths:
            if rrf_scores[path] <= 0:
                continue
            entry = self.files.get(path)
            if entry is None:
                continue
            result: dict = {
                "path": path,
                "score": round(rrf_scores[path], 6),
                "language": entry.language,
                "symbols": entry.symbols[:10],
                "size": entry.size_bytes,
                "lines": entry.line_count,
            }
            if return_snippets and path in snippet_map:
                result["snippet"] = snippet_map[path]
            combined.append(result)

        combined.sort(key=lambda r: r["score"], reverse=True)
        return combined[:top_k]

    def build_chunk_embeddings(self, model: "EmbeddingModel") -> tuple[int, int]:
        """Compute and store embeddings for all file chunks.

        A-RAG S3.1: each chunk gets a dense embedding.

        Returns: (chunks_embedded, files_processed)
        """
        all_texts: list[str] = []
        all_keys: list[str] = []

        for rel, entry in self.files.items():
            for chunk in entry.chunks:
                key = f"{rel}::{chunk.chunk_idx}"
                all_texts.append(chunk.text[:2000])
                all_keys.append(key)

        if not all_texts:
            return 0, 0

        # Batch embed
        vecs = model.embed_batch(all_texts)

        # Store flat dict
        self._chunk_embeddings = {}
        for key, vec in zip(all_keys, vecs):
            self._chunk_embeddings[key] = vec.tolist() if hasattr(vec, "tolist") else list(vec)

        # Restore back into ChunkEntry for convenience
        for rel, entry in self.files.items():
            for chunk in entry.chunks:
                key = f"{rel}::{chunk.chunk_idx}"
                if key in self._chunk_embeddings:
                    chunk.embedding = self._chunk_embeddings[key]

        files = len({k.split("::")[0] for k in all_keys})
        return len(all_texts), files

    @staticmethod
    def extract_snippet(text: str, query: str, max_sentences: int = 3) -> str:
        """Extract sentences from text that contain query keywords.

        A-RAG S3.2 Eq. 2: snippet = sentences containing query terms.

        Returns up to max_sentences sentences containing query keywords,
        joined with " ... ". Returns first 200 chars of text if no match found.
        """
        # Split on sentence boundaries
        parts = re.split(r"(?<=[.!?])\s+|\n", text)
        sentences = [s.strip() for s in parts if s.strip()]

        # Query keywords: words longer than 3 chars, lowercased
        query_words = {w.lower() for w in query.split() if len(w) > 3}

        if not query_words:
            return text[:200] + ("..." if len(text) > 200 else "")

        matching = []
        for sentence in sentences:
            lower_sentence = sentence.lower()
            if any(word in lower_sentence for word in query_words):
                matching.append(sentence)
                if len(matching) >= max_sentences:
                    break

        if matching:
            return " ... ".join(matching)

        return text[:200] + ("..." if len(text) > 200 else "")

    def chunk_semantic_search(
        self,
        query: str,
        model: "EmbeddingModel",
        top_k: int = 10,
    ) -> list[dict]:
        """Semantic search at chunk level with snippet extraction.

        A-RAG S3.2: sentence-level dense retrieval + snippet extraction.

        Returns list of dicts: {path, chunk_idx, start_line, end_line, score, snippet}
        Falls back to empty list if no chunk embeddings built (caller falls back to file-level).
        """
        if not self._chunk_embeddings:
            return []

        try:
            import numpy as np
        except ImportError:
            return []

        query_vec = model.embed_query(query)

        keys = list(self._chunk_embeddings.keys())
        vecs = [self._chunk_embeddings[k] for k in keys]

        scores = [float(np.dot(query_vec, np.array(v, dtype=np.float32))) for v in vecs]
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])

        results = []
        for idx, score in indexed[:top_k]:
            if score <= 0:
                continue
            key = keys[idx]
            rel_path, chunk_idx_str = key.rsplit("::", 1)
            chunk_idx = int(chunk_idx_str)

            entry = self.files.get(rel_path)
            if entry is None:
                continue

            chunk = next((c for c in entry.chunks if c.chunk_idx == chunk_idx), None)
            if chunk is None:
                continue

            snippet = self.extract_snippet(chunk.text, query)
            results.append({
                "path": rel_path,
                "chunk_idx": chunk_idx,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "score": round(score, 4),
                "snippet": snippet,
            })

        return results
