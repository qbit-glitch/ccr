"""Integration tests for the context packer with a mock sub-model.

Verifies: index → search → rank → slice → ContextPack pipeline.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from ccr.context.indexer import RepoIndex
from ccr.context.packer import ContextPacker
from ccr.core.types import TokenUsage


class MockSubClient:
    """Mock sub-model that returns reasonable responses for packing."""

    def __init__(self):
        self.calls = []

    def completion(self, messages, **kwargs):
        self.calls.append(messages)
        last_msg = messages[-1]["content"] if messages else ""

        # Symbol extraction
        if "extract" in messages[0].get("content", "").lower():
            return json.dumps({
                "symbols": ["Application", "run", "helper"],
                "keywords": ["app", "run", "main"],
                "file_patterns": ["src/*.py"],
            })

        # Ranking
        if "rank" in messages[0].get("content", "").lower():
            return json.dumps([
                {"path": "src/main.py", "relevance": 0.95, "reason": "main app"},
                {"path": "src/utils.py", "relevance": 0.7, "reason": "helper"},
            ])

        return "mock response"

    def get_last_usage(self):
        return TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)

    def get_usage_summary(self):
        return None

    async def acompletion(self, messages, **kwargs):
        return self.completion(messages, **kwargs)


@pytest.fixture
def sample_repo():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src"))
        with open(os.path.join(d, "src", "main.py"), "w") as f:
            f.write(
                "from .utils import helper\n\n"
                "class Application:\n"
                "    def __init__(self):\n"
                "        self.name = 'myapp'\n\n"
                "    def run(self):\n"
                "        result = helper(42)\n"
                "        print(f'Result: {result}')\n\n"
                "def main():\n"
                "    app = Application()\n"
                "    app.run()\n"
            )
        with open(os.path.join(d, "src", "utils.py"), "w") as f:
            f.write(
                "def helper(x):\n"
                "    return x + 1\n\n"
                "def format_output(data):\n"
                "    return str(data)\n\n"
                "class Config:\n"
                "    DEBUG = True\n"
            )
        with open(os.path.join(d, "src", "tests.py"), "w") as f:
            f.write(
                "import pytest\n"
                "from .utils import helper\n\n"
                "def test_helper():\n"
                "    assert helper(1) == 2\n"
            )
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write("# My Project\nA sample project.\n")
        yield d


@pytest.fixture
def repo_index(sample_repo):
    return RepoIndex.build(sample_repo)


@pytest.fixture
def packer(repo_index):
    client = MockSubClient()
    return ContextPacker(
        repo_index=repo_index,
        sub_client=client,
        token_budget=4000,
    )


class TestContextPacking:
    def test_pack_returns_context_pack(self, packer):
        pack = packer.pack("Fix the bug in the Application.run method")
        assert pack is not None
        assert pack.task_description
        assert pack.pack_id

    def test_pack_includes_relevant_files(self, packer):
        pack = packer.pack("Fix the Application class")
        paths = [f[0] for f in pack.files]
        # Should include main.py since it has Application class
        assert any("main.py" in p for p in paths)

    def test_pack_respects_token_budget(self, packer):
        pack = packer.pack("Analyze all the code")
        assert pack.total_tokens <= 4000

    def test_pack_includes_memory_context(self, packer):
        pack = packer.pack("Fix a bug", memory_context="## Project: Testing\nFocus: bug fix")
        assert "Project: Testing" in pack.memory_context

    def test_pack_to_prompt_text(self, packer):
        pack = packer.pack("Fix a bug", memory_context="memory context here")
        text = pack.to_prompt_text()
        assert "<project_context>" in text
        assert "memory context here" in text
        assert "<file" in text

    def test_empty_pack_for_no_matches(self, packer):
        pack = packer.pack("zzzznonexistent topic that matches nothing")
        # Should return a pack (possibly empty) without crashing
        assert pack is not None

    def test_sub_model_called_for_extraction(self, packer):
        client = packer.sub_client
        packer.pack("Fix the helper function")
        # Should have made at least one call for symbol extraction
        assert len(client.calls) >= 1

    def test_pack_generates_symbols(self, packer):
        pack = packer.pack("Fix the Application")
        # Symbols should be populated from the index
        assert isinstance(pack.symbols, list)


class TestIndexerIntegration:
    def test_index_search_finds_classes(self, repo_index):
        results = repo_index.search("Application")
        assert len(results) > 0
        assert results[0]["path"] == "src/main.py"

    def test_index_search_finds_functions(self, repo_index):
        results = repo_index.search("helper")
        paths = {r["path"] for r in results}
        assert "src/utils.py" in paths

    def test_index_get_file_returns_content(self, repo_index):
        content = repo_index.get_file("src/main.py")
        assert content is not None
        assert "class Application" in content

    def test_index_summary(self, repo_index):
        summary = repo_index.get_summary()
        assert "Files:" in summary
        assert "python" in summary.lower()
