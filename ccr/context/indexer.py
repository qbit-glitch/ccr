"""Repo indexer — builds an in-memory index of the entire codebase.

Zero LLM tokens. Pure filesystem + regex. Loaded into REPL as a variable.
Semantic search via optional ONNX embeddings (A-RAG §3.2) or BM25 fallback (CCR's own zero-dep alternative).
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

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
        # Semantic search state
        self._embeddings: dict[str, list[float]] = {}
        self._bm25_cache: dict | None = None

    @classmethod
    def build(
        cls,
        root: str,
        ignore_patterns: list[str] | None = None,
        max_file_size_kb: int = 500,
        extensions: set[str] | None = None,
        max_files: int = 50000,
    ) -> RepoIndex:
        """Build index from filesystem. No LLM calls."""
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
                continue

            language = LANGUAGE_MAP.get(ext, ext.lstrip(".") or "unknown")

            try:
                with open(entry.path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                continue

            symbols = cls._extract_symbols(content, language)
            imports = cls._extract_imports(content, language)
            lines = content.count("\n") + 1

            index.files[rel] = FileEntry(
                rel_path=rel,
                size_bytes=stat.st_size,
                language=language,
                symbols=symbols,
                imports=imports,
                last_modified=stat.st_mtime,
                line_count=lines,
                _content=content,
            )
            file_count += 1

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
    ) -> list[dict]:
        """Combine keyword + semantic scores (CCR's own score-fusion design).

        Normalizes both score types to [0,1] and combines with weights.
        Uses ONNX embeddings if model + embeddings available, else BM25.

        Note: A-RAG (arXiv:2602.03442) uses agent-driven tool selection
        (the agent chooses keyword_search vs semantic_search per step).
        This mechanical score fusion is CCR's own approach.
        """
        # Keyword scores
        kw_results = self.search(query, file_glob=file_glob)
        max_kw = 9.0  # path(3) + symbol(5) + content(1)
        kw_scores: dict[str, float] = {
            r["path"]: r["score"] / max_kw for r in kw_results
        }

        # Semantic scores
        sem_scores: dict[str, float] = {}
        if model is not None and self._embeddings:
            sem_results = self.semantic_search(
                query, model, top_k=100, file_glob=file_glob
            )
            for r in sem_results:
                sem_scores[r["path"]] = r["score"]
        else:
            # BM25 fallback — normalize by max score
            bm25_results = self.bm25_search(query, top_k=100, file_glob=file_glob)
            if bm25_results:
                max_bm25 = bm25_results[0]["score"]
                if max_bm25 > 0:
                    for r in bm25_results:
                        sem_scores[r["path"]] = r["score"] / max_bm25

        # Combine
        all_paths = set(kw_scores.keys()) | set(sem_scores.keys())
        combined: list[dict] = []
        for path in all_paths:
            kw = kw_scores.get(path, 0.0)
            sem = sem_scores.get(path, 0.0)
            final = keyword_weight * kw + semantic_weight * sem
            if final <= 0:
                continue
            entry = self.files.get(path)
            if entry is None:
                continue
            combined.append({
                "path": path,
                "score": round(final, 4),
                "language": entry.language,
                "symbols": entry.symbols[:10],
                "size": entry.size_bytes,
                "lines": entry.line_count,
            })

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

    def to_json(self) -> str:
        """Serialize to JSON for REPL loading. Excludes file contents for efficiency."""
        data = {
            "root": self.root,
            "built_at": self._built_at,
            "file_count": len(self.files),
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
        return imports[:20]  # cap

    def _compute_mtime_sig(self) -> str:
        """Quick signature of file mtimes for cache validation."""
        parts = sorted(
            f"{rel}:{entry.last_modified:.0f}"
            for rel, entry in self.files.items()
        )
        return "|".join(parts[:100])  # sample first 100 for speed
