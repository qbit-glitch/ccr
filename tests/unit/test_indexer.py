"""Tests for RepoIndex — filesystem indexer."""

import gzip
import json
import os
import tempfile

import pytest

from ccr.context.indexer import RepoIndex


@pytest.fixture
def sample_repo():
    """Create a small sample repo for testing."""
    with tempfile.TemporaryDirectory() as d:
        # Python files
        os.makedirs(os.path.join(d, "src"))
        with open(os.path.join(d, "src", "main.py"), "w") as f:
            f.write("""
import os
from .utils import helper

class Application:
    def __init__(self):
        self.name = "test"

    def run(self):
        pass

def main():
    app = Application()
    app.run()
""")
        with open(os.path.join(d, "src", "utils.py"), "w") as f:
            f.write("""
def helper(x):
    return x + 1

def format_output(data):
    return str(data)

class Config:
    DEBUG = True
""")

        # TypeScript file
        with open(os.path.join(d, "src", "index.ts"), "w") as f:
            f.write("""
import { Router } from 'express';

export class ApiController {
    private router: Router;

    constructor() {
        this.router = Router();
    }
}

export function createApp() {
    return new ApiController();
}
""")

        # Config file
        with open(os.path.join(d, "config.yaml"), "w") as f:
            f.write("debug: true\nport: 8080\n")

        # Create .gitignore
        with open(os.path.join(d, ".gitignore"), "w") as f:
            f.write("*.pyc\n__pycache__/\n")

        # Binary file (should be ignored)
        with open(os.path.join(d, "image.png"), "wb") as f:
            f.write(b"\x89PNG" + b"\x00" * 1000)

        yield d


class TestBuild:
    def test_basic_build(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        assert len(idx.files) > 0

    def test_finds_python_files(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        paths = set(idx.files.keys())
        assert "src/main.py" in paths
        assert "src/utils.py" in paths

    def test_finds_typescript_files(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        assert "src/index.ts" in idx.files

    def test_ignores_binary_files(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        paths = set(idx.files.keys())
        assert "image.png" not in paths

    def test_respects_gitignore(self, sample_repo):
        # Create a .pyc file
        os.makedirs(os.path.join(sample_repo, "__pycache__"), exist_ok=True)
        with open(os.path.join(sample_repo, "__pycache__", "test.pyc"), "w") as f:
            f.write("bytecode")
        idx = RepoIndex.build(sample_repo)
        paths = set(idx.files.keys())
        assert not any("__pycache__" in p for p in paths)

    def test_extension_filter(self, sample_repo):
        idx = RepoIndex.build(sample_repo, extensions={".py"})
        for path in idx.files:
            assert path.endswith(".py")


class TestSymbols:
    def test_python_symbols(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        entry = idx.files["src/main.py"]
        assert "Application" in entry.symbols
        assert "main" in entry.symbols
        assert "run" in entry.symbols

    def test_typescript_symbols(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        entry = idx.files["src/index.ts"]
        assert "ApiController" in entry.symbols
        assert "createApp" in entry.symbols


class TestImports:
    def test_python_imports(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        entry = idx.files["src/main.py"]
        assert "os" in entry.imports

    def test_typescript_imports(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        entry = idx.files["src/index.ts"]
        assert "express" in entry.imports


class TestSearch:
    def test_search_by_symbol(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        results = idx.search("Application")
        assert len(results) > 0
        assert results[0]["path"] == "src/main.py"

    def test_search_by_content(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        results = idx.search("debug")
        assert len(results) > 0

    def test_search_by_path(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        results = idx.search("utils")
        paths = [r["path"] for r in results]
        assert "src/utils.py" in paths

    def test_search_with_glob(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        results = idx.search("", file_glob="*.ts")
        assert all(r["path"].endswith(".ts") for r in results)

    def test_search_no_results(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        results = idx.search("zzzznonexistent")
        assert len(results) == 0


class TestGetFile:
    def test_get_existing_file(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        content = idx.get_file("src/main.py")
        assert content is not None
        assert "Application" in content

    def test_get_nonexistent_file(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        assert idx.get_file("nonexistent.py") is None


class TestSerialization:
    def test_to_json(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        json_str = idx.to_json()
        assert "file_count" in json_str
        assert "src/main.py" in json_str

    def test_summary(self, sample_repo):
        idx = RepoIndex.build(sample_repo)
        summary = idx.get_summary()
        assert "Files:" in summary
        assert "Lines:" in summary


# --- Semantic search tests (A-RAG §3.2) ---


@pytest.fixture
def semantic_repo():
    """Sample repo with semantically related files for BM25/semantic testing."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src", "auth"))
        os.makedirs(os.path.join(d, "src", "db"))

        with open(os.path.join(d, "src", "auth", "verify.py"), "w") as f:
            f.write("""\
\"\"\"User authentication and token verification.\"\"\"

def verify_user_token(token: str) -> bool:
    \"\"\"Verify that a user's authentication token is valid.\"\"\"
    return check_signature(token) and not is_expired(token)

def check_session(session_id: str) -> dict:
    \"\"\"Check if a session is still active and authorized.\"\"\"
    return {"valid": True, "user": "test"}

def revoke_access(user_id: str) -> None:
    \"\"\"Revoke all authentication credentials for a user.\"\"\"
    pass
""")

        with open(os.path.join(d, "src", "auth", "login.py"), "w") as f:
            f.write("""\
\"\"\"Login and credential management.\"\"\"

def authenticate(username: str, password: str) -> str:
    \"\"\"Authenticate a user and return a session token.\"\"\"
    return "token_123"

def logout(session_id: str) -> None:
    \"\"\"End a user session.\"\"\"
    pass
""")

        with open(os.path.join(d, "src", "db", "connection.py"), "w") as f:
            f.write("""\
\"\"\"Database connection pooling.\"\"\"
import sqlite3

def get_connection(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)

def close_pool() -> None:
    pass
""")

        with open(os.path.join(d, "src", "db", "queries.py"), "w") as f:
            f.write("""\
\"\"\"Database query helpers.\"\"\"

def find_user_by_email(email: str) -> dict:
    return {"email": email, "name": "Test User"}

def insert_record(table: str, data: dict) -> int:
    return 1
""")

        yield d


class TestTokenizeSimple:
    def test_basic_tokenization(self):
        tokens = RepoIndex._tokenize_simple("Hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_stop_word_removal(self):
        tokens = RepoIndex._tokenize_simple("the import from class def return")
        # All are stop words
        assert len(tokens) == 0

    def test_min_length(self):
        tokens = RepoIndex._tokenize_simple("a b cd ef")
        # "a" and "b" are < 2 chars, filtered out
        assert "cd" in tokens
        assert "ef" in tokens
        assert "a" not in tokens


class TestBM25Search:
    def test_bm25_finds_related_terms(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        results = idx.bm25_search("authentication token verify")
        assert len(results) > 0
        paths = [r["path"] for r in results]
        # Should find auth/verify.py with highest score
        assert "src/auth/verify.py" in paths

    def test_bm25_no_results_empty_query(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        results = idx.bm25_search("")
        assert len(results) == 0

    def test_bm25_respects_file_glob(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        results = idx.bm25_search("user", file_glob="src/db/*")
        for r in results:
            assert r["path"].startswith("src/db/")

    def test_bm25_ranking(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        results = idx.bm25_search("authenticate user session token")
        assert len(results) >= 2
        # Auth files should rank above DB files
        auth_paths = [r["path"] for r in results if "auth" in r["path"]]
        db_paths = [r["path"] for r in results if "db" in r["path"]]
        if auth_paths and db_paths:
            auth_best = min(
                i for i, r in enumerate(results) if "auth" in r["path"]
            )
            db_best = min(
                i for i, r in enumerate(results) if "db" in r["path"]
            )
            assert auth_best < db_best

    def test_bm25_stop_words_filtered(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        # Searching for only stop words should return nothing
        results = idx.bm25_search("the is a an")
        assert len(results) == 0

    def test_bm25_cache_reused(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        assert idx._bm25_cache is None
        idx.bm25_search("test")
        assert idx._bm25_cache is not None
        cache_id = id(idx._bm25_cache)
        idx.bm25_search("another query")
        # Cache should be reused
        assert id(idx._bm25_cache) == cache_id


class TestFileSummary:
    def test_summary_includes_path_and_symbols(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        entry = idx.files["src/auth/verify.py"]
        summary = RepoIndex._file_summary(entry)
        assert "src/auth/verify.py" in summary
        assert "verify_user_token" in summary

    def test_summary_includes_first_lines(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        entry = idx.files["src/auth/verify.py"]
        summary = RepoIndex._file_summary(entry)
        assert "authentication" in summary.lower()

    def test_summary_empty_content(self):
        from ccr.context.indexer import FileEntry

        entry = FileEntry(
            rel_path="empty.py",
            size_bytes=0,
            language="python",
            symbols=[],
            imports=[],
            last_modified=0,
            line_count=0,
            _content="",
        )
        summary = RepoIndex._file_summary(entry)
        assert "empty.py" in summary


class TestHybridSearch:
    def test_hybrid_combines_keyword_and_bm25(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        # No embeddings loaded → falls back to BM25 for semantic component
        results = idx.hybrid_search("authenticate")
        assert len(results) > 0
        # Should find auth files
        paths = [r["path"] for r in results]
        assert any("auth" in p for p in paths)

    def test_hybrid_keyword_weight(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        # With keyword_weight=1.0, should behave like keyword search
        kw_results = idx.hybrid_search(
            "authenticate", keyword_weight=1.0, semantic_weight=0.0
        )
        sem_results = idx.hybrid_search(
            "authenticate", keyword_weight=0.0, semantic_weight=1.0
        )
        # Both should return results but possibly different rankings
        assert len(kw_results) > 0 or len(sem_results) > 0

    def test_hybrid_no_results(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        results = idx.hybrid_search("zzzznonexistentterm")
        assert len(results) == 0


class TestEmbeddingPersistence:
    def test_save_load_roundtrip(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        # Manually set some fake embeddings
        idx._embeddings = {
            "src/auth/verify.py": [0.1, 0.2, 0.3],
            "src/db/connection.py": [0.4, 0.5, 0.6],
        }
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = f.name
        try:
            idx.save_embeddings(path)
            # Load into a fresh index
            idx2 = RepoIndex.build(semantic_repo)
            assert len(idx2._embeddings) == 0
            loaded = idx2.load_embeddings(path)
            assert loaded is True
            assert len(idx2._embeddings) == 2
            assert idx2._embeddings["src/auth/verify.py"] == [0.1, 0.2, 0.3]
        finally:
            os.unlink(path)

    def test_load_missing_embeddings(self, semantic_repo):
        idx = RepoIndex.build(semantic_repo)
        loaded = idx.load_embeddings("/tmp/nonexistent_embeddings.json.gz")
        assert loaded is False
        assert len(idx._embeddings) == 0


# ---------------------------------------------------------------------------
# Chunking tests (A-RAG §3.1 sentence-level chunking)
# ---------------------------------------------------------------------------


class TestChunking:
    def test_split_python_file_on_function_boundaries(self):
        """Two top-level defs separated by blank line → 2 chunks."""
        content = """\
def foo():
    x = 1
    return x

def bar():
    y = 2
    return y
"""
        chunks = RepoIndex._split_into_chunks(content, "module.py", max_tokens=800)
        assert len(chunks) == 2
        assert "def foo" in chunks[0].text
        assert "def bar" in chunks[1].text

    def test_split_markdown_on_paragraphs(self):
        """Double-newline separated paragraphs → multiple chunks when large."""
        # Two paragraphs separated by blank line; each small enough individually
        para1 = "word " * 50  # 50 words
        para2 = "term " * 50  # 50 words
        content = para1.strip() + "\n\n" + para2.strip()
        chunks = RepoIndex._split_into_chunks(content, "README.md", max_tokens=800)
        # Should produce at least 1 chunk (content is well under 800 words each)
        assert len(chunks) >= 1
        full_text = " ".join(c.text for c in chunks)
        assert "word" in full_text
        assert "term" in full_text

    def test_single_small_file_one_chunk(self):
        """Content < 100 words → 1 chunk."""
        content = "This is a small file. It has very few words.\n"
        chunks = RepoIndex._split_into_chunks(content, "small.py", max_tokens=800)
        assert len(chunks) == 1
        assert "small" in chunks[0].text or "few" in chunks[0].text

    def test_chunk_line_numbers_correct(self):
        """Verify start_line/end_line match actual content positions."""
        lines = ["line one", "line two", "line three", "line four", "line five"]
        content = "\n".join(lines)
        chunks = RepoIndex._split_into_chunks(content, "test.md", max_tokens=800)
        assert len(chunks) >= 1
        # All line numbers must be within [1, len(lines)]
        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line <= len(lines)
            assert chunk.start_line <= chunk.end_line

    def test_large_chunk_hard_split(self):
        """Content with no natural boundaries > 800 words → split at line boundary."""
        # 900 single-word lines (no blank lines, no def/class)
        lines = [f"uniqueword{i}" for i in range(900)]
        content = "\n".join(lines)
        chunks = RepoIndex._split_into_chunks(content, "big.py", max_tokens=800)
        assert len(chunks) >= 2
        # No chunk should exceed max_tokens significantly
        for chunk in chunks:
            word_count = len(chunk.text.split())
            assert word_count <= 900  # sanity; hard limit enforced

    def test_empty_content_returns_empty(self):
        """Empty string → []."""
        chunks = RepoIndex._split_into_chunks("", "empty.py", max_tokens=800)
        assert chunks == []


# ---------------------------------------------------------------------------
# Chunk semantic search tests (A-RAG §3.2)
# ---------------------------------------------------------------------------


class TestChunkSearch:
    def _make_index_with_chunks(self):
        """Create a RepoIndex with manually injected chunks and embeddings."""
        from ccr.context.indexer import ChunkEntry, FileEntry

        idx = RepoIndex.__new__(RepoIndex)
        idx.root = "/fake"
        idx._built_at = None
        idx._mtime_sig = ""
        idx._embeddings = {}
        idx._bm25_cache = None
        idx._chunk_embeddings = {}

        # Create a file entry with 3 chunks
        entry = FileEntry(
            rel_path="src/auth.py",
            size_bytes=200,
            language="python",
            symbols=["login", "logout"],
            imports=[],
            last_modified=0,
            line_count=30,
            _content="",
        )
        chunks = [
            ChunkEntry("src/auth.py", 0, 1, 10, "def login(): authenticate user credentials"),
            ChunkEntry("src/auth.py", 1, 11, 20, "def logout(): end user session and clear token"),
            ChunkEntry("src/auth.py", 2, 21, 30, "def register(): create new user account in database"),
        ]
        entry.chunks = chunks
        idx.files = {"src/auth.py": entry}

        # Set up fake chunk embeddings (3-dim for simplicity)
        # login chunk ~ [1, 0, 0], logout ~ [0, 1, 0], register ~ [0, 0, 1]
        idx._chunk_embeddings = {
            "src/auth.py::0": [1.0, 0.0, 0.0],
            "src/auth.py::1": [0.0, 1.0, 0.0],
            "src/auth.py::2": [0.0, 0.0, 1.0],
        }
        # Also store back on chunks
        for chunk in chunks:
            key = f"src/auth.py::{chunk.chunk_idx}"
            chunk.embedding = idx._chunk_embeddings[key]

        return idx

    def _make_mock_model(self, query_vec: list[float]):
        """Return a mock model that embeds queries as the given vector."""
        import types
        import numpy as np

        model = types.SimpleNamespace()
        model.embed_query = lambda q: np.array(query_vec, dtype=np.float32)
        return model

    def test_chunk_search_returns_snippet_field(self):
        """chunk_semantic_search results include snippet field."""
        import numpy as np

        idx = self._make_index_with_chunks()
        model = self._make_mock_model([1.0, 0.0, 0.0])  # closest to login chunk
        results = idx.chunk_semantic_search("login authenticate", model, top_k=3)
        assert len(results) > 0
        assert "snippet" in results[0]
        assert isinstance(results[0]["snippet"], str)

    def test_chunk_search_returns_line_numbers(self):
        """chunk_semantic_search results include start_line and end_line."""
        import numpy as np

        idx = self._make_index_with_chunks()
        model = self._make_mock_model([0.0, 1.0, 0.0])  # closest to logout chunk
        results = idx.chunk_semantic_search("logout session", model, top_k=3)
        assert len(results) > 0
        top = results[0]
        assert "start_line" in top
        assert "end_line" in top
        assert top["start_line"] >= 1
        assert top["end_line"] >= top["start_line"]

    def test_chunk_search_fallback_empty_when_no_embeddings(self):
        """When _chunk_embeddings is empty, returns []."""
        idx = self._make_index_with_chunks()
        idx._chunk_embeddings = {}
        model = self._make_mock_model([1.0, 0.0, 0.0])
        results = idx.chunk_semantic_search("login", model, top_k=5)
        assert results == []

    def test_chunk_search_top_k_respected(self):
        """With 3 chunks and top_k=2 → at most 2 results."""
        idx = self._make_index_with_chunks()
        model = self._make_mock_model([0.33, 0.33, 0.34])  # roughly equal to all
        results = idx.chunk_semantic_search("user", model, top_k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Snippet extraction tests (A-RAG §3.2 Eq. 2)
# ---------------------------------------------------------------------------


class TestSnippetExtraction:
    def test_extract_returns_sentence_with_keyword(self):
        """Text with 3 sentences, only 1 contains query word → returns that sentence."""
        text = "The sky is blue. Python is great for coding. The cat sat on mat."
        snippet = RepoIndex.extract_snippet(text, "Python coding")
        assert "Python" in snippet or "coding" in snippet

    def test_extract_returns_fallback_when_no_match(self):
        """No query words in text → returns first 200 chars."""
        text = "The sky is blue. The sun is warm. The grass is green."
        snippet = RepoIndex.extract_snippet(text, "xyzzy quux wumpus")
        # Fallback: first 200 chars (query words all len <= 3 after filter, or no match)
        assert len(snippet) <= 203  # 200 + "..."

    def test_extract_caps_at_max_sentences(self):
        """5 matching sentences, max_sentences=2 → at most 2 returned."""
        sentences = [f"The algorithm processes data {i} efficiently." for i in range(5)]
        text = " ".join(sentences)
        snippet = RepoIndex.extract_snippet(text, "algorithm data", max_sentences=2)
        # Count how many of the original sentences appear by looking at " ... " splits
        parts = snippet.split(" ... ")
        assert len(parts) <= 2

    def test_extract_case_insensitive(self):
        """Query 'Python', sentence has 'python' → matches."""
        text = "This is a test line. python is used here for scripting. Another line."
        snippet = RepoIndex.extract_snippet(text, "Python")
        assert "python" in snippet.lower()
