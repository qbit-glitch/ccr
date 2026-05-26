"""Repo indexer — builds an in-memory index of the entire codebase.

Zero LLM tokens. Pure filesystem + regex. Loaded into REPL as a variable.
Semantic search via optional ONNX embeddings (A-RAG §3.2) or BM25 fallback (CCR's own zero-dep alternative).
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import re
import time

from ccr.context.indexer_types import (
    SYMBOL_PATTERNS,
    LANGUAGE_MAP,
    DEFAULT_IGNORES,
    _detect_mode,
    ChunkEntry,
    FileEntry,
)
from ccr.context.indexer_search import SearchMixin
from ccr.context.indexer_io import IOMixin

# Re-export public names for backward compatibility
__all__ = [
    "SYMBOL_PATTERNS",
    "LANGUAGE_MAP",
    "DEFAULT_IGNORES",
    "_detect_mode",
    "ChunkEntry",
    "FileEntry",
    "RepoIndex",
]

logger = logging.getLogger(__name__)


class RepoIndex(SearchMixin, IOMixin):
    """In-memory index of a repository. Serializable to JSON for REPL loading.

    Supports three search modes:
      - keyword: Exact substring matching on paths, symbols, content
        (inspired by A-RAG §3.2, but uses substring matching not frequency*length scoring)
      - semantic: Dense embedding cosine via ONNX (A-RAG §3.2 Eq 3 inspired)
        or BM25 zero-dep fallback (CCR's own, not from A-RAG)
      - hybrid: Combines keyword + semantic scores
        (CCR's own design — A-RAG uses agent-driven tool selection, not score fusion)
    """

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

    @classmethod
    def incremental_build(
        cls,
        root: str,
        db,
        ignore_patterns: list[str] | None = None,
        max_file_size_kb: int = 500,
        extensions: set[str] | None = None,
        max_files: int = 50000,
    ) -> tuple[RepoIndex, int]:
        """Incremental build: only re-index files whose mtime changed.

        Args:
            root: Project root directory.
            db: IndexDB instance with prior index state.

        Returns:
            (RepoIndex, changed_count) — the updated index and number of files re-indexed.
        """
        old_mtimes = db.get_file_mtimes()
        index = cls.build(
            root, ignore_patterns=ignore_patterns,
            max_file_size_kb=max_file_size_kb,
            extensions=extensions, max_files=max_files,
        )

        changed = 0
        for rel, entry in index.files.items():
            old_mtime = old_mtimes.get(rel)
            if old_mtime is None or entry.last_modified != old_mtime:
                changed += 1

        deleted = db.delete_missing(set(index.files.keys()))
        changed += deleted

        index.save_to_db(db)
        return index, changed

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
