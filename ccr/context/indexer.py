"""Repo indexer — builds an in-memory index of the entire codebase.

Zero LLM tokens. Pure filesystem + regex. Loaded into REPL as a variable.
Semantic search via optional ONNX embeddings (A-RAG §3.2) or BM25 fallback (CCR's own zero-dep alternative).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ccr.context.embeddings import EmbeddingModel

# Symbol extraction patterns per language
SYMBOL_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [
        re.compile(r"^\s*(?:class|def|async\s+def)\s+(\w+)", re.MULTILINE),
    ],
    "typescript": [
        re.compile(r"(?:export\s+)?(?:class|function|const|interface|type|enum)\s+(\w+)", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"(?:export\s+)?(?:class|function|const)\s+(\w+)", re.MULTILINE),
    ],
    "go": [
        re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", re.MULTILINE),
        re.compile(r"^type\s+(\w+)\s+(?:struct|interface)", re.MULTILINE),
    ],
    "rust": [
        re.compile(r"(?:pub\s+)?(?:fn|struct|enum|trait|impl|type|const)\s+(\w+)", re.MULTILINE),
    ],
    "java": [
        re.compile(r"(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|enum)\s+(\w+)", re.MULTILINE),
        re.compile(r"(?:public|private|protected)\s+\w+\s+(\w+)\s*\(", re.MULTILINE),
    ],
}

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".sh": "shell",
}

# Default ignore patterns (gitignore-style)
DEFAULT_IGNORES = [
    ".git", ".ccr", ".gcc", "__pycache__", "node_modules", ".venv", "venv",
    ".env", ".DS_Store", "dist", "build", ".next", ".nuxt", "target",
    "*.pyc", "*.pyo", "*.so", "*.dylib", "*.o", "*.a",
    "*.min.js", "*.min.css", "*.map",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.eot",
    "*.zip", "*.tar", "*.gz", "*.bz2",
    "*.pdf", "*.doc", "*.docx",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Cargo.lock", "go.sum",
]


_AUTO_QUESTION_WORDS = {"how", "why", "what", "when", "where", "explain", "describe"}
_AUTO_SYMBOL_RE = re.compile(r'^[\w./:-]+$')


def _detect_mode(query: str) -> str:
    """A-RAG §3.2: infer best search mode from query shape.

    Rules (in priority order):
    - ≤3 tokens OR looks like a symbol/path → keyword
    - Ends with '?' OR starts with a question word → semantic
    - Otherwise → hybrid
    """
    tokens = query.strip().split()
    if len(tokens) <= 3 or bool(_AUTO_SYMBOL_RE.match(query.strip())):
        return "keyword"
    if query.rstrip().endswith("?") or (tokens and tokens[0].lower() in _AUTO_QUESTION_WORDS):
        return "semantic"
    return "hybrid"


@dataclass
class ChunkEntry:
    """A sentence/paragraph-level chunk of a file (A-RAG §3.1)."""

    file_path: str       # relative path (same as FileEntry.path)
    chunk_idx: int       # 0-based chunk index within the file
    start_line: int      # 1-based line number where chunk starts
    end_line: int        # 1-based line number where chunk ends (inclusive)
    text: str            # raw chunk text
    embedding: list[float] = field(default_factory=list)  # set by build_chunk_embeddings


@dataclass
class FileEntry:
    rel_path: str
    size_bytes: int
    language: str
    symbols: list[str]
    imports: list[str]
    last_modified: float
    line_count: int
    _content: str = field(default="", repr=False)
    chunks: list[ChunkEntry] = field(default_factory=list)

    @property
    def content(self) -> str:
        return self._content


class RepoIndex:
    """In-memory index of a repository. Serializable to JSON for REPL loading.

    Supports three search modes:
      - keyword: Exact substring matching on paths, symbols, content
        (inspired by A-RAG §3.2, but uses substring matching not frequency*length scoring)
      - semantic: Dense embedding cosine via ONNX (A-RAG §3.2 Eq 3 inspired)
        or BM25 zero-dep fallback (CCR's own, not from A-RAG)
      - hybrid: Combines keyword + semantic scores
        (CCR's own design — A-RAG uses agent-driven tool selection, not score fusion)
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

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.files: dict[str, FileEntry] = {}
        self._built_at: float | None = None
        self._mtime_sig: str = ""
        # SHA-256 hash cache for incremental skip (16-char hex prefix per file)
        self._file_hashes: dict[str, str] = {}
        # Semantic search state
        self._embeddings: dict[str, list[float]] = {}
        self._bm25_cache: dict | None = None
        # Chunk-level embeddings (A-RAG §3.1 sentence-level chunks)
        self._chunk_embeddings: dict[str, list[float]] = {}

    @classmethod
    def build(
        cls,
        root: str,
        ignore_patterns: list[str] | None = None,
        max_file_size_kb: int = 500,
        extensions: set[str] | None = None,
        max_files: int = 50000,
        progress_interval: int = 100,
    ) -> RepoIndex:
        """Build index from filesystem. No LLM calls.

        Args:
            root: Project root directory to index.
            ignore_patterns: Additional glob patterns to ignore.
            max_file_size_kb: Max file size in KB to index content (default 500).
            extensions: If set, only index files with these extensions.
            max_files: Max total files to index (default 50000).
            progress_interval: Log a progress message every N files (default 100).
        """
        index = cls(root)
        ignores = (ignore_patterns or []) + DEFAULT_IGNORES
        ignores += cls._load_gitignore(root)
        max_bytes = max_file_size_kb * 1024
        file_count = 0

        for entry in cls._walk(root, ignores):
            if not entry.is_file():
                continue
            # M2: Bound total indexed files to prevent unbounded memory use
            if file_count >= max_files:
                break

            rel = os.path.relpath(entry.path, root)
            ext = os.path.splitext(entry.name)[1].lower()

            if extensions and ext not in extensions:
                continue

            try:
                stat = entry.stat()
            except OSError:
                continue

            if stat.st_size == 0:
                continue
            if stat.st_size > max_bytes:
                # Index metadata for large files but skip content/symbols
                logger.debug("Skipping content for large file %s (%d KB > %d KB)",
                             rel, stat.st_size // 1024, max_file_size_kb)
                language = LANGUAGE_MAP.get(ext, ext.lstrip(".") or "unknown")
                index.files[rel] = FileEntry(
                    rel_path=rel,
                    size_bytes=stat.st_size,
                    language=language,
                    symbols=[],
                    imports=[],
                    last_modified=stat.st_mtime,
                    line_count=0,
                    _content="",
                )
                file_count += 1
                if file_count % progress_interval == 0 and file_count > 0:
                    logger.info("Index progress: %d files indexed", file_count)
                continue

            language = LANGUAGE_MAP.get(ext, ext.lstrip(".") or "unknown")

            try:
                with open(entry.path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError) as e:
                logger.debug("Skipping unreadable file %s: %s", rel, e)
                continue

            # Compute SHA-256 hash (16-char prefix — adequate for cache dedup)
            sha256 = hashlib.sha256(content.encode()).hexdigest()[:16]
            index._file_hashes[rel] = sha256

            symbols = cls._extract_symbols(content, language)
            imports = cls._extract_imports(content, language)
            lines = content.count("\n") + 1

            entry = FileEntry(
                rel_path=rel,
                size_bytes=stat.st_size,
                language=language,
                symbols=symbols,
                imports=imports,
                last_modified=stat.st_mtime,
                line_count=lines,
                _content=content,
            )
            # A-RAG §3.1: chunk content into ~800-token segments
            if content and len(content.encode()) < 50_000:
                entry.chunks = cls._split_into_chunks(content, rel, max_tokens=800)
            index.files[rel] = entry
            file_count += 1
            if file_count % progress_interval == 0 and file_count > 0:
                logger.info("Index progress: %d files indexed", file_count)

        index._built_at = time.time()
        index._mtime_sig = index._compute_mtime_sig()
        return index

    def search(self, pattern: str, file_glob: str = "**/*") -> list[dict]:
        """Search index for files matching pattern in content/symbols/path."""
        pattern_lower = pattern.lower()
        results = []

        for rel, entry in self.files.items():
            if file_glob != "**/*" and not fnmatch.fnmatch(rel, file_glob):
                continue

            score = 0
            # Path match
            if pattern_lower in rel.lower():
                score += 3
            # Symbol match
            for sym in entry.symbols:
                if pattern_lower in sym.lower():
                    score += 5
                    break
            # Content match
            if pattern_lower in entry._content.lower():
                score += 1

            if score > 0:
                results.append({
                    "path": rel,
                    "score": score,
                    "language": entry.language,
                    "symbols": entry.symbols[:10],
                    "size": entry.size_bytes,
                    "lines": entry.line_count,
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def get_file(self, rel_path: str) -> str | None:
        entry = self.files.get(rel_path)
        return entry._content if entry else None

    # --- Semantic search ---

    @staticmethod
    def _tokenize_simple(text: str) -> list[str]:
        """Simple word tokenizer for BM25. Lowercases, filters stops."""
        return [
            w for w in re.findall(r"\w{2,}", text.lower())
            if w not in RepoIndex._BM25_STOP_WORDS
        ]

    @staticmethod
    def _file_summary(entry: FileEntry) -> str:
        """Generate embeddable summary for a file (A-RAG §3.1 adaptation).

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

    @staticmethod
    def _split_into_chunks(content: str, rel_path: str, max_tokens: int = 800) -> list[ChunkEntry]:
        """Split file content into ~max_tokens chunks on natural boundaries.

        Split priority (highest to lowest):
        1. Blank lines between top-level Python def/class blocks
        2. Single blank lines (paragraph breaks) for non-Python files
        3. Hard split at max_tokens on the nearest line boundary

        A-RAG §3.1: corpus is chunked into ~1000-token segments.
        CCR adaptation: 800-token target, split on semantic boundaries.

        Returns [] for empty content.
        """
        if not content:
            return []

        lines = content.split("\n")
        n_lines = len(lines)

        # Collect natural split points (0-based line indices to split BEFORE)
        split_before: list[int] = []

        if rel_path.endswith(".py"):
            # Python: split before top-level def/class that follow a blank line
            for i in range(1, n_lines):
                prev_blank = lines[i - 1].strip() == ""
                curr_toplevel = re.match(r"^(def |class |async def )", lines[i])
                if prev_blank and curr_toplevel:
                    split_before.append(i)
        else:
            # Non-Python: split before a non-blank line that follows a blank line
            for i in range(1, n_lines):
                if lines[i - 1].strip() == "" and lines[i].strip() != "":
                    split_before.append(i)

        # Build list of (start, end) segments from split boundaries
        boundaries = sorted(set(split_before))
        segment_starts = [0] + boundaries
        segment_ends = boundaries + [n_lines]
        raw_segments = [(s, e) for s, e in zip(segment_starts, segment_ends) if s < e]

        chunks: list[ChunkEntry] = []

        for seg_start, seg_end in raw_segments:
            text = "\n".join(lines[seg_start:seg_end])
            if not text.strip():
                continue

            # Count words in this segment
            word_count = sum(len(ln.split()) for ln in lines[seg_start:seg_end])

            if word_count <= max_tokens:
                # Segment fits — emit as-is
                chunks.append(ChunkEntry(
                    file_path=rel_path,
                    chunk_idx=len(chunks),
                    start_line=seg_start + 1,
                    end_line=seg_end,
                    text=text,
                ))
            else:
                # Hard split: walk line by line until max_tokens
                pos = seg_start
                while pos < seg_end:
                    ptr = pos
                    wc = 0
                    while ptr < seg_end:
                        wc += len(lines[ptr].split())
                        if wc > max_tokens:
                            break
                        ptr += 1
                    if ptr == pos:
                        ptr = pos + 1  # always advance at least one line
                    seg_text = "\n".join(lines[pos:ptr])
                    if seg_text.strip():
                        chunks.append(ChunkEntry(
                            file_path=rel_path,
                            chunk_idx=len(chunks),
                            start_line=pos + 1,
                            end_line=ptr,
                            text=seg_text,
                        ))
                    pos = ptr

        # Reindex chunk_idx sequentially
        for i, chunk in enumerate(chunks):
            chunk.chunk_idx = i

        return chunks

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

    def build_embeddings(self, model: EmbeddingModel) -> int:
        """Compute embeddings for all indexed files. Returns count.

        A-RAG §3.1: builds dense representations for semantic search.
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
        """Dense embedding cosine similarity search (A-RAG §3.2, Eq 3)."""
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
        RRF k=60 is the standard constant from Cormack et al. (2009) —
        avoids the normalisation instability of linear score combination.

        When return_snippets=True and chunk embeddings are available, uses
        chunk-level semantic search (A-RAG §3.2) and attaches snippet field
        to results.

        Note: keyword_weight / semantic_weight params are kept for API
        compatibility but are no longer used in scoring (RRF is rank-only).
        """
        # Keyword ranked list
        kw_results = self.search(query, file_glob=file_glob)

        # Snippet map from chunk search (if requested)
        snippet_map: dict[str, str] = {}

        # Semantic ranked list
        sem_results_ordered: list[dict] = []
        if return_snippets and model is not None and self._chunk_embeddings:
            # Use chunk-level semantic search (A-RAG §3.2)
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

    def save_embeddings(self, path: str) -> None:
        """Save embeddings to gzip-compressed JSON."""
        from ccr.context.embeddings import save_embeddings

        save_embeddings(self._embeddings, path)

    def load_embeddings(self, path: str) -> bool:
        """Load pre-computed embeddings. Returns True if loaded."""
        from ccr.context.embeddings import load_embeddings

        loaded = load_embeddings(path)
        if loaded:
            self._embeddings = loaded
            return True
        return False

    def build_chunk_embeddings(self, model: "EmbeddingModel") -> tuple[int, int]:
        """Compute and store embeddings for all file chunks.

        A-RAG §3.1: each chunk gets a dense embedding.

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

    def save_chunk_embeddings(self, path: str) -> None:
        """Save chunk embeddings to gzip-compressed JSON.

        Mirror of save_embeddings() for chunk-level data.
        Path: index_chunk_embeddings.json.gz in .ccr/.
        """
        from ccr.context.embeddings import save_embeddings

        save_embeddings(self._chunk_embeddings, path)

    def load_chunk_embeddings(self, path: str) -> bool:
        """Load pre-computed chunk embeddings. Returns True if loaded.

        Mirror of load_embeddings() for chunk-level data.
        Also restores chunk.embedding on each ChunkEntry by matching keys.
        """
        from ccr.context.embeddings import load_embeddings

        loaded = load_embeddings(path)
        if not loaded:
            return False

        self._chunk_embeddings = loaded

        # Restore embeddings back into ChunkEntry objects
        for rel, entry in self.files.items():
            for chunk in entry.chunks:
                key = f"{rel}::{chunk.chunk_idx}"
                if key in self._chunk_embeddings:
                    chunk.embedding = self._chunk_embeddings[key]

        return True

    @staticmethod
    def extract_snippet(text: str, query: str, max_sentences: int = 3) -> str:
        """Extract sentences from text that contain query keywords.

        A-RAG §3.2 Eq. 2: snippet = sentences containing query terms.

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

        A-RAG §3.2: sentence-level dense retrieval + snippet extraction.

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

    def to_json(self) -> str:
        """Serialize to JSON for REPL loading. Excludes file contents for efficiency."""
        data = {
            "root": self.root,
            "built_at": self._built_at,
            "mtime_sig": self._mtime_sig or "",
            "file_count": len(self.files),
            "file_hashes": self._file_hashes,
            "files": {},
        }
        for rel, entry in self.files.items():
            data["files"][rel] = {
                "size": entry.size_bytes,
                "language": entry.language,
                "symbols": entry.symbols,
                "imports": entry.imports,
                "lines": entry.line_count,
            }
        return json.dumps(data, indent=None, separators=(",", ":"))

    def to_full_json(self) -> str:
        """Serialize including file contents — used for REPL context loading."""
        data = {
            "root": self.root,
            "built_at": self._built_at,
            "file_count": len(self.files),
            "files": {},
        }
        for rel, entry in self.files.items():
            data["files"][rel] = {
                "content": entry._content,
                "size": entry.size_bytes,
                "language": entry.language,
                "symbols": entry.symbols,
                "imports": entry.imports,
                "lines": entry.line_count,
            }
        return json.dumps(data, indent=None, separators=(",", ":"))

    def to_context_dict(self) -> dict:
        """Convert to a dict suitable for loading as a REPL context variable.

        Returns metadata (path, symbols, imports, size) for each file — NOT content.
        Content is fetched on-demand via get_file() to avoid loading everything.
        """
        files_meta = {}
        for rel, entry in self.files.items():
            files_meta[rel] = {
                "size": entry.size_bytes,
                "language": entry.language,
                "symbols": entry.symbols,
                "imports": entry.imports,
                "lines": entry.line_count,
            }
        return {
            "root": self.root,
            "file_count": len(self.files),
            "files": files_meta,
        }

    @classmethod
    def from_cache(cls, root: str, cache_json: str) -> RepoIndex | None:
        """Load from cache if mtime signatures match."""
        try:
            data = json.loads(cache_json)
            index = cls(root)
            # Backward compat: old caches won't have file_hashes
            index._file_hashes = data.get("file_hashes", {})
            for rel, fdata in data.get("files", {}).items():
                abs_path = os.path.join(root, rel)
                if not os.path.isfile(abs_path):
                    return None  # File deleted, cache invalid
                index.files[rel] = FileEntry(
                    rel_path=rel,
                    size_bytes=fdata.get("size", 0),
                    language=fdata.get("language", ""),
                    symbols=fdata.get("symbols", []),
                    imports=fdata.get("imports", []),
                    last_modified=0,
                    line_count=fdata.get("lines", 0),
                )
            index._built_at = data.get("built_at")
            index._mtime_sig = data.get("mtime_sig", "")
            return index
        except (json.JSONDecodeError, KeyError):
            return None

    def get_summary(self) -> str:
        """Human-readable summary of the indexed repo."""
        langs: dict[str, int] = {}
        total_lines = 0
        for entry in self.files.values():
            langs[entry.language] = langs.get(entry.language, 0) + 1
            total_lines += entry.line_count

        lang_str = ", ".join(f"{lang}: {count}" for lang, count in sorted(langs.items(), key=lambda x: -x[1])[:5])
        return (
            f"Repo: {self.root}\n"
            f"Files: {len(self.files)} | Lines: {total_lines:,}\n"
            f"Languages: {lang_str}\n"
        )

    # --- Internal ---

    @staticmethod
    def _walk(root: str, ignores: list[str]):
        """Walk directory tree, respecting ignore patterns."""
        try:
            entries = list(os.scandir(root))
        except PermissionError:
            return

        for entry in entries:
            name = entry.name
            if any(fnmatch.fnmatch(name, pat) for pat in ignores):
                continue
            if entry.is_dir(follow_symlinks=False):
                yield from RepoIndex._walk(entry.path, ignores)
            else:
                yield entry

    @staticmethod
    def _load_gitignore(root: str) -> list[str]:
        gitignore = os.path.join(root, ".gitignore")
        if not os.path.isfile(gitignore):
            return []
        patterns = []
        with open(gitignore, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line.rstrip("/"))
        return patterns

    @staticmethod
    def _extract_symbols(content: str, language: str) -> list[str]:
        patterns = SYMBOL_PATTERNS.get(language, [])
        symbols = []
        for pat in patterns:
            symbols.extend(pat.findall(content))
        return list(dict.fromkeys(symbols))  # dedupe, preserve order

    @staticmethod
    def _extract_imports(content: str, language: str) -> list[str]:
        imports = []
        if language == "python":
            for m in re.finditer(r"^(?:from|import)\s+(\S+)", content, re.MULTILINE):
                imports.append(m.group(1))
        elif language in ("typescript", "javascript"):
            for m in re.finditer(r"""(?:from|import|require)\s*\(?['"]([^'"]+)['"]""", content):
                imports.append(m.group(1))
        elif language == "go":
            for m in re.finditer(r'"([^"]+)"', content[:2000]):  # imports are at top
                imports.append(m.group(1))
        return imports[:50]  # cap (raised from 20 — large files can have many imports)

    def _compute_mtime_sig(self) -> str:
        """Quick signature of file mtimes for cache validation."""
        parts = sorted(
            f"{rel}:{entry.last_modified:.0f}"
            for rel, entry in self.files.items()
        )
        return "|".join(parts[:100])  # sample first 100 for speed
