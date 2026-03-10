"""Repo indexer — builds an in-memory index of the entire codebase.

Zero LLM tokens. Pure filesystem + regex. Loaded into REPL as a variable.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

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
    """In-memory index of a repository. Serializable to JSON for REPL loading."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.files: dict[str, FileEntry] = {}
        self._built_at: float | None = None
        self._mtime_sig: str = ""

    @classmethod
    def build(
        cls,
        root: str,
        ignore_patterns: list[str] | None = None,
        max_file_size_kb: int = 500,
        extensions: set[str] | None = None,
    ) -> RepoIndex:
        """Build index from filesystem. No LLM calls."""
        index = cls(root)
        ignores = (ignore_patterns or []) + DEFAULT_IGNORES
        ignores += cls._load_gitignore(root)
        max_bytes = max_file_size_kb * 1024

        for entry in cls._walk(root, ignores):
            if not entry.is_file():
                continue

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
