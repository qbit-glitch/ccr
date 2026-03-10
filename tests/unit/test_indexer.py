"""Tests for RepoIndex — filesystem indexer."""

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
