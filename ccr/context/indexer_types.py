"""Indexer types, constants, and module-level functions.

Extracted from indexer.py to keep files under 400 lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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
    """A-RAG S3.2: infer best search mode from query shape.

    Rules (in priority order):
    - <=3 tokens OR looks like a symbol/path -> keyword
    - Ends with '?' OR starts with a question word -> semantic
    - Otherwise -> hybrid
    """
    tokens = query.strip().split()
    if len(tokens) <= 3 or bool(_AUTO_SYMBOL_RE.match(query.strip())):
        return "keyword"
    if query.rstrip().endswith("?") or (tokens and tokens[0].lower() in _AUTO_QUESTION_WORDS):
        return "semantic"
    return "hybrid"


@dataclass
class ChunkEntry:
    """A sentence/paragraph-level chunk of a file (A-RAG S3.1)."""

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
