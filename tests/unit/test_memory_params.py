"""Tests for ALMA-inspired memory retrieval parameter evolution (MCE §3.9).

Validates:
- Memory metrics tracking (search calls, zero results, link creations)
- PlaybookSchema and SchemaMetrics new fields
- ace_evolve_schema proposals for ADJUST_SEARCH_THRESHOLD and ADJUST_SCAN_WINDOW
"""

import json
import os
import tempfile

import pytest

from ccr.mcp_server import (
    _init,
    _save_playbook,
    ace_apply_delta,
    ace_evolve_schema,
    gcc_commit,
    gcc_context,
)
import ccr.mcp_server as mcp_mod
from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig, PlaybookSchema, SchemaMetrics


@pytest.fixture(autouse=True)
def setup_project(tmp_path, monkeypatch):
    """Initialize CCR in a temp directory for each test."""
    # Redirect ~/.ccr/ to temp dir so tests don't touch real global playbook
    fake_home_ccr = tmp_path / "global_ccr"
    fake_home_ccr.mkdir()
    original_expanduser = os.path.expanduser
    def mock_expanduser(path):
        if path.startswith("~/.ccr"):
            return str(fake_home_ccr) + path[5:]
        return original_expanduser(path)
    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

    # Create a sample file for indexing
    src = tmp_path / "hello.py"
    src.write_text("def greet(name):\n    return f'Hello, {name}!'\n")

    _init(str(tmp_path))
    yield tmp_path

    # Cleanup globals
    mcp_mod._memory = None
    mcp_mod._playbook = None
    mcp_mod._global_playbook = None
    mcp_mod._repo_index = None
    mcp_mod._repl = None


class TestMemoryMetricsPath:
    def test_memory_metrics_path(self, tmp_path):
        """Verify path returns correct .ccr/ location."""
        mem = MemoryManager(str(tmp_path))
        path = mem._get_memory_metrics_path()
        assert path == os.path.join(str(tmp_path), ".ccr", "memory_metrics.json")


class TestIncrementMemoryMetric:
    def test_increment_creates_file(self, tmp_path):
        """Increment creates metrics file when none exists."""
        mem = MemoryManager(str(tmp_path))
        mem.ensure_structure()
        mem._increment_memory_metric("search_calls")
        path = mem._get_memory_metrics_path()
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["search_calls"] == 1
        assert "last_updated" in data

    def test_increment_updates_existing(self, tmp_path):
        """Increment updates an existing metric."""
        mem = MemoryManager(str(tmp_path))
        mem.ensure_structure()
        mem._increment_memory_metric("search_calls")
        mem._increment_memory_metric("search_calls")
        mem._increment_memory_metric("search_calls", 3)
        metrics = mem.get_memory_metrics()
        assert metrics["search_calls"] == 5


class TestGetMemoryMetrics:
    def test_empty_returns_zeros(self, tmp_path):
        """Returns zeros when no file exists."""
        mem = MemoryManager(str(tmp_path))
        mem.ensure_structure()
        metrics = mem.get_memory_metrics()
        assert metrics["search_calls"] == 0
        assert metrics["search_zero_results"] == 0
        assert metrics["link_creations"] == 0
        assert metrics["total_commits"] == 0
        assert metrics["last_updated"] == ""


class TestSearchTracking:
    def test_search_tracks_calls(self, tmp_path):
        """_search_commits increments search_calls."""
        mem = mcp_mod._memory
        # Do a search that won't find anything
        mem._search_commits("main", "nonexistent_term_xyz")
        metrics = mem.get_memory_metrics()
        assert metrics["search_calls"] >= 1

    def test_search_tracks_zero_results(self, tmp_path):
        """Zero-result search increments both counters."""
        mem = mcp_mod._memory
        mem._search_commits("main", "absolutely_nothing_here_xyz")
        metrics = mem.get_memory_metrics()
        assert metrics["search_calls"] >= 1
        assert metrics["search_zero_results"] >= 1


class TestCommitTracking:
    def test_commit_increments_total(self, tmp_path):
        """gcc_commit increments total_commits metric."""
        gcc_commit("T1", "did A", "reason A", ["a.py"], "next A")
        mem = mcp_mod._memory
        metrics = mem.get_memory_metrics()
        assert metrics["total_commits"] >= 1


class TestSchemaHasMemoryParams:
    def test_schema_has_new_fields(self):
        """PlaybookSchema has ALMA-inspired retrieval parameter fields."""
        schema = PlaybookSchema.default()
        assert schema.link_scan_window == 20
        assert schema.link_semantic_threshold == 0.3
        assert schema.context_level_default == 2
        assert schema.search_result_limit == 5

    def test_schema_to_dict_from_dict_roundtrip(self):
        """New fields survive serialization roundtrip."""
        schema = PlaybookSchema.default()
        schema.link_scan_window = 25
        schema.link_semantic_threshold = 0.25
        schema.context_level_default = 3
        schema.search_result_limit = 10
        d = schema.to_dict()
        restored = PlaybookSchema.from_dict(d)
        assert restored.link_scan_window == 25
        assert restored.link_semantic_threshold == 0.25
        assert restored.context_level_default == 3
        assert restored.search_result_limit == 10

    def test_schema_from_dict_defaults(self):
        """from_dict uses correct defaults for missing ALMA fields (backward compat)."""
        restored = PlaybookSchema.from_dict({})
        assert restored.link_scan_window == 20
        assert restored.link_semantic_threshold == 0.3
        assert restored.context_level_default == 2
        assert restored.search_result_limit == 5


class TestSchemaMetricsHasSearchFields:
    def test_schema_metrics_has_new_fields(self):
        """SchemaMetrics has ALMA-inspired memory retrieval metrics."""
        m = SchemaMetrics()
        assert m.search_zero_rate == 0.0
        assert m.link_density == 0.0
        assert m.embedding_coverage == 0.0

    def test_schema_metrics_to_dict_from_dict_roundtrip(self):
        """New SchemaMetrics fields survive serialization roundtrip."""
        m = SchemaMetrics(
            search_zero_rate=0.75,
            link_density=1.5,
            embedding_coverage=0.8,
        )
        d = m.to_dict()
        restored = SchemaMetrics.from_dict(d)
        assert restored.search_zero_rate == 0.75
        assert restored.link_density == 1.5
        assert restored.embedding_coverage == 0.8


def _populate_healthy_playbook():
    """Create a playbook state where structural proposals won't fire.

    Fills all 6 default sections with bullets, marks them as utilized,
    so no empty/overflow/decay/pruning proposals trigger. This lets
    ALMA-inspired retrieval proposals surface.
    """
    from ccr.mcp_server import ace_update_counters
    sections = [
        "STRATEGIES & INSIGHTS",
        "CODE SNIPPETS & TEMPLATES",
        "COMMON MISTAKES TO AVOID",
        "PROBLEM-SOLVING HEURISTICS",
        "CONTEXT CLUES & INDICATORS",
        "OTHERS",
    ]
    ops = []
    for i, section in enumerate(sections):
        ops.append({
            "type": "ADD",
            "section": section,
            "content": f"Bullet {i+1} for {section.lower()} testing",
        })
        ops.append({
            "type": "ADD",
            "section": section,
            "content": f"Another bullet {i+1} for {section.lower()} verification",
        })
    ace_apply_delta(operations=ops)
    # Mark all bullets as helpful so utilization_rate > 0, unused_ratio < 0.5,
    # and harmful_ratio stays low — prevents ADJUST_DECAY and ADJUST_PRUNING proposals
    from ccr.mcp_server import ace_update_counters
    pb = mcp_mod._playbook
    if pb:
        tags = [{"id": b.id, "tag": "helpful"} for b in pb.bullets]
        ace_update_counters(bullet_tags=tags)


class TestHighZeroRateProposal:
    def test_high_zero_rate_generates_proposal(self, tmp_path):
        """When search_zero_rate > 0.5, ace_evolve_schema proposes ADJUST_SEARCH_THRESHOLD."""
        # Seed memory metrics with high zero-result rate
        mem = mcp_mod._memory
        for _ in range(10):
            mem._increment_memory_metric("search_calls")
            mem._increment_memory_metric("search_zero_results")
        mem._increment_memory_metric("total_commits")

        # Populate all sections to prevent structural proposals
        _populate_healthy_playbook()

        result = ace_evolve_schema()
        assert "ADJUST_SEARCH_THRESHOLD" in result["message"]
        assert "link_semantic_threshold" in result["message"]


class TestLowLinkDensityProposal:
    def test_low_link_density_generates_proposal(self, tmp_path):
        """When link_density < 0.5, proposes ADJUST_SCAN_WINDOW."""
        # Seed: many commits, few links, some search calls (zero rate <= 0.5)
        mem = mcp_mod._memory
        mem._increment_memory_metric("total_commits", 10)
        mem._increment_memory_metric("link_creations", 2)
        mem._increment_memory_metric("search_calls", 10)
        mem._increment_memory_metric("search_zero_results", 3)  # 30% < 50%

        # Populate all sections to prevent structural proposals
        _populate_healthy_playbook()

        result = ace_evolve_schema()
        assert "ADJUST_SCAN_WINDOW" in result["message"]
        assert "link_scan_window" in result["message"]


class TestApplyThresholdProposal:
    def test_applying_threshold_proposal(self, tmp_path):
        """Apply ADJUST_SEARCH_THRESHOLD changes schema params."""
        # Seed high zero-result rate
        mem = mcp_mod._memory
        for _ in range(10):
            mem._increment_memory_metric("search_calls")
            mem._increment_memory_metric("search_zero_results")
        mem._increment_memory_metric("total_commits")

        _populate_healthy_playbook()

        # Verify proposal exists
        result = ace_evolve_schema()
        assert "ADJUST_SEARCH_THRESHOLD" in result["message"]

        # Apply the proposal
        result2 = ace_evolve_schema(apply_proposal=1)
        assert "Applied" in result2["message"] or "ADJUST_SEARCH_THRESHOLD" in result2["message"]
        assert "schema v" in result2["message"].lower() or "v2" in result2["message"]


class TestSchemaOverrides:
    def test_effective_link_scan_window_default(self, tmp_path):
        """Without override, effective_link_scan_window returns config value."""
        mem = MemoryManager(str(tmp_path))
        assert mem.effective_link_scan_window == 20

    def test_effective_link_scan_window_override(self, tmp_path):
        """With schema override, effective_link_scan_window returns override."""
        mem = MemoryManager(str(tmp_path))
        mem.set_schema_overrides({"link_scan_window": 30})
        assert mem.effective_link_scan_window == 30

    def test_effective_link_semantic_threshold_override(self, tmp_path):
        """With schema override, effective_link_semantic_threshold returns override."""
        mem = MemoryManager(str(tmp_path))
        mem.set_schema_overrides({"link_semantic_threshold": 0.2})
        assert mem.effective_link_semantic_threshold == 0.2

    def test_effective_fallback_to_config(self, tmp_path):
        """Schema override for one param doesn't affect others."""
        config = CCRConfig(link_scan_window=15, link_semantic_threshold=0.4)
        mem = MemoryManager(str(tmp_path), config)
        mem.set_schema_overrides({"link_scan_window": 30})
        assert mem.effective_link_scan_window == 30
        assert mem.effective_link_semantic_threshold == 0.4


class TestEvolveSchemaShowsMemoryMetrics:
    def test_health_report_includes_memory_metrics(self, tmp_path):
        """ace_evolve_schema health report includes memory retrieval metrics."""
        result = ace_evolve_schema()
        assert "Search Zero Rate" in result["message"]
        assert "Link Density" in result["message"]
        assert "Embedding Coverage" in result["message"]

    def test_health_report_includes_retrieval_params(self, tmp_path):
        """ace_evolve_schema shows ALMA retrieval parameters."""
        result = ace_evolve_schema()
        assert "Link scan window" in result["message"]
        assert "Link semantic threshold" in result["message"]
        assert "Context level default" in result["message"]
        assert "Search result limit" in result["message"]
