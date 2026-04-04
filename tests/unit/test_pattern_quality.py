"""Tests for EvolveR-inspired pattern quality scoring (arXiv:2510.16079).

Validates:
- PatternEntry quality fields (success_count, failure_count, quality_score)
- Bayesian quality score computation: (success+1)/(success+failure+2)
- Quality propagation via update_pattern_quality()
- get_patterns() sorting by quality_score
- Backward compatibility with old format (missing quality fields)
- Persistence across MemoryManager instances
"""

import json
import os
import tempfile

import pytest

from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig, PatternEntry


@pytest.fixture
def ccr_dir(tmp_path):
    ccr = tmp_path / ".ccr"
    ccr.mkdir()
    branches = ccr / "branches" / "main"
    branches.mkdir(parents=True)
    (branches / "_registry.md").write_text("## Active Branch\nmain\n")
    (branches / "commits.md").write_text("")
    (ccr / "metadata.yaml").write_text("project: test\ncreated: 2026-01-01\n")
    return str(ccr)


@pytest.fixture
def mem(ccr_dir, tmp_path):
    config = CCRConfig()
    return MemoryManager(os.path.dirname(ccr_dir), config)


class TestPatternEntryQuality:
    """Tests for PatternEntry dataclass quality fields."""

    def test_default_quality_score(self):
        p = PatternEntry(text="test", first_seen="C001")
        assert p.quality_score == 0.5
        assert p.success_count == 0
        assert p.failure_count == 0
        assert p.last_quality_update == ""

    def test_to_dict_includes_quality(self):
        p = PatternEntry(
            text="t", first_seen="C001",
            success_count=3, failure_count=1, quality_score=0.8,
            last_quality_update="2026-03-28T00:00:00+00:00",
        )
        d = p.to_dict()
        assert d["success_count"] == 3
        assert d["failure_count"] == 1
        assert d["quality_score"] == 0.8
        assert d["last_quality_update"] == "2026-03-28T00:00:00+00:00"

    def test_from_dict_with_quality(self):
        d = {
            "text": "t", "first_seen": "C001",
            "success_count": 5, "failure_count": 2, "quality_score": 0.75,
            "last_quality_update": "2026-03-28T12:00:00+00:00",
        }
        p = PatternEntry.from_dict(d)
        assert p.success_count == 5
        assert p.failure_count == 2
        assert p.quality_score == 0.75
        assert p.last_quality_update == "2026-03-28T12:00:00+00:00"

    def test_from_dict_defaults_without_quality(self):
        """Old format without quality fields should default correctly."""
        d = {"text": "t", "first_seen": "C001"}
        p = PatternEntry.from_dict(d)
        assert p.success_count == 0
        assert p.failure_count == 0
        assert p.quality_score == 0.5
        assert p.last_quality_update == ""

    def test_round_trip(self):
        """to_dict -> from_dict preserves quality fields."""
        original = PatternEntry(
            text="When adding {tools}, update tests",
            first_seen="C005",
            commit_ids=["C005", "C008"],
            occurrence_count=2,
            created_at="2026-01-15",
            promoted=True,
            success_count=7,
            failure_count=1,
            quality_score=0.8,
            last_quality_update="2026-03-28T00:00:00+00:00",
        )
        restored = PatternEntry.from_dict(original.to_dict())
        assert restored.success_count == original.success_count
        assert restored.failure_count == original.failure_count
        assert restored.quality_score == original.quality_score
        assert restored.last_quality_update == original.last_quality_update


class TestUpdatePatternQuality:
    """Tests for MemoryManager.update_pattern_quality()."""

    def _seed_pattern(self, mem, text="When adding {tools}, update tests together"):
        """Seed a pattern in the buffer."""
        patterns_path = mem._get_patterns_path()
        data = {
            "version": 1,
            "patterns": {
                "P001": {
                    "text": text,
                    "first_seen": "C001",
                    "commit_ids": ["C001"],
                    "occurrence_count": 1,
                    "created_at": "2026-01-01 00:00",
                    "promoted": True,
                    "success_count": 0,
                    "failure_count": 0,
                    "quality_score": 0.5,
                    "last_quality_update": "",
                }
            },
            "next_id": 2,
        }
        os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
        with open(patterns_path, "w") as f:
            json.dump(data, f)
        return text

    def test_success_increments(self, mem):
        text = self._seed_pattern(mem)
        result = mem.update_pattern_quality(text, success=True)
        assert result is True
        data = mem._load_patterns()
        p = data["patterns"]["P001"]
        assert p["success_count"] == 1
        assert p["failure_count"] == 0
        # Bayesian: (1+1)/(1+0+2) = 2/3
        assert abs(p["quality_score"] - 2 / 3) < 0.01

    def test_failure_increments(self, mem):
        text = self._seed_pattern(mem)
        result = mem.update_pattern_quality(text, success=False)
        assert result is True
        data = mem._load_patterns()
        p = data["patterns"]["P001"]
        assert p["success_count"] == 0
        assert p["failure_count"] == 1
        # Bayesian: (0+1)/(0+1+2) = 1/3
        assert abs(p["quality_score"] - 1 / 3) < 0.01

    def test_no_match_returns_false(self, mem):
        self._seed_pattern(mem)
        result = mem.update_pattern_quality(
            "completely unrelated text xyz abc nothing", success=True
        )
        assert result is False

    def test_empty_buffer_returns_false(self, mem):
        """No patterns file at all."""
        result = mem.update_pattern_quality("anything here", success=True)
        assert result is False

    def test_multiple_updates(self, mem):
        text = self._seed_pattern(mem)
        mem.update_pattern_quality(text, success=True)
        mem.update_pattern_quality(text, success=True)
        mem.update_pattern_quality(text, success=False)
        data = mem._load_patterns()
        p = data["patterns"]["P001"]
        assert p["success_count"] == 2
        assert p["failure_count"] == 1
        # Bayesian: (2+1)/(2+1+2) = 3/5 = 0.6
        assert abs(p["quality_score"] - 0.6) < 0.01

    def test_updates_last_quality_update(self, mem):
        text = self._seed_pattern(mem)
        mem.update_pattern_quality(text, success=True)
        data = mem._load_patterns()
        p = data["patterns"]["P001"]
        assert p["last_quality_update"] != ""
        # Should be a valid ISO-8601 timestamp
        from datetime import datetime
        datetime.fromisoformat(p["last_quality_update"])

    def test_quality_persists(self, mem, ccr_dir):
        """Quality updates persist across MemoryManager instances."""
        text = self._seed_pattern(mem)
        mem.update_pattern_quality(text, success=True)
        mem.update_pattern_quality(text, success=True)

        # Create new manager instance
        config = CCRConfig()
        mem2 = MemoryManager(os.path.dirname(ccr_dir), config)
        data = mem2._load_patterns()
        p = data["patterns"]["P001"]
        assert p["success_count"] == 2
        assert p["quality_score"] == (2 + 1) / (2 + 0 + 2)

    def test_bayesian_prior_is_uniform(self, mem):
        """With zero observations, Bayesian score = (0+1)/(0+0+2) = 0.5."""
        text = self._seed_pattern(mem)
        data = mem._load_patterns()
        p = data["patterns"]["P001"]
        assert p["quality_score"] == 0.5

    def test_bayesian_converges_with_data(self, mem):
        """With many successes, score approaches 1.0."""
        text = self._seed_pattern(mem)
        for _ in range(20):
            mem.update_pattern_quality(text, success=True)
        data = mem._load_patterns()
        p = data["patterns"]["P001"]
        # (20+1)/(20+0+2) = 21/22 ≈ 0.9545
        assert p["quality_score"] > 0.9

    def test_backward_compat_missing_quality_fields(self, mem):
        """Old patterns without quality fields get defaults when updated."""
        patterns_path = mem._get_patterns_path()
        data = {
            "version": 1,
            "patterns": {
                "P001": {
                    "text": "When adding {tools}, update tests together",
                    "first_seen": "C001",
                    "commit_ids": ["C001"],
                    "occurrence_count": 1,
                    "created_at": "2026-01-01 00:00",
                    "promoted": True,
                    # No quality fields — old format
                }
            },
            "next_id": 2,
        }
        os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
        with open(patterns_path, "w") as f:
            json.dump(data, f)

        result = mem.update_pattern_quality(
            "When adding {tools}, update tests together", success=True
        )
        assert result is True
        data = mem._load_patterns()
        p = data["patterns"]["P001"]
        assert p["success_count"] == 1
        assert p["failure_count"] == 0
        # Bayesian: (1+1)/(1+0+2) = 2/3
        assert abs(p["quality_score"] - 2 / 3) < 0.01


class TestPatternsReturnSortedByQuality:
    """Tests that get_patterns() sorts by quality_score descending."""

    def _seed_patterns(self, mem):
        patterns_path = mem._get_patterns_path()
        data = {
            "version": 1,
            "patterns": {
                "P001": {
                    "text": "Low quality pattern about testing failures",
                    "first_seen": "C001",
                    "commit_ids": ["C001"],
                    "occurrence_count": 1,
                    "created_at": "2026-01-01 00:00",
                    "promoted": False,
                    "success_count": 0,
                    "failure_count": 5,
                    "quality_score": 0.14,
                    "last_quality_update": "2026-03-28T00:00:00+00:00",
                },
                "P002": {
                    "text": "High quality pattern about architecture design",
                    "first_seen": "C002",
                    "commit_ids": ["C002"],
                    "occurrence_count": 1,
                    "created_at": "2026-01-01 00:00",
                    "promoted": False,
                    "success_count": 8,
                    "failure_count": 0,
                    "quality_score": 0.9,
                    "last_quality_update": "2026-03-28T00:00:00+00:00",
                },
                "P003": {
                    "text": "Medium quality pattern about refactoring code",
                    "first_seen": "C003",
                    "commit_ids": ["C003"],
                    "occurrence_count": 1,
                    "created_at": "2026-01-01 00:00",
                    "promoted": False,
                    "success_count": 2,
                    "failure_count": 2,
                    "quality_score": 0.5,
                    "last_quality_update": "2026-03-28T00:00:00+00:00",
                },
            },
            "next_id": 4,
        }
        os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
        with open(patterns_path, "w") as f:
            json.dump(data, f)

    def test_patterns_sorted_by_quality(self, mem):
        self._seed_patterns(mem)
        result = mem.get_patterns()
        entries = result["patterns"]
        assert len(entries) == 3
        # Should be sorted: high (0.9), medium (0.5), low (0.14)
        assert entries[0]["quality_score"] > entries[1]["quality_score"]
        assert entries[1]["quality_score"] > entries[2]["quality_score"]
        assert entries[0]["id"] == "P002"  # highest quality
        assert entries[2]["id"] == "P001"  # lowest quality

    def test_quality_tiebreaks_by_occurrence(self, mem):
        """When quality is equal, higher occurrence_count comes first."""
        patterns_path = mem._get_patterns_path()
        data = {
            "version": 1,
            "patterns": {
                "P001": {
                    "text": "Pattern alpha about building features",
                    "first_seen": "C001",
                    "commit_ids": ["C001"],
                    "occurrence_count": 1,
                    "created_at": "2026-01-01 00:00",
                    "promoted": False,
                    "success_count": 0,
                    "failure_count": 0,
                    "quality_score": 0.5,
                    "last_quality_update": "",
                },
                "P002": {
                    "text": "Pattern beta about deploying services",
                    "first_seen": "C002",
                    "commit_ids": ["C002", "C003", "C004"],
                    "occurrence_count": 3,
                    "created_at": "2026-01-01 00:00",
                    "promoted": False,
                    "success_count": 0,
                    "failure_count": 0,
                    "quality_score": 0.5,
                    "last_quality_update": "",
                },
            },
            "next_id": 3,
        }
        os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
        with open(patterns_path, "w") as f:
            json.dump(data, f)

        result = mem.get_patterns()
        entries = result["patterns"]
        assert len(entries) == 2
        # Same quality (0.5), P002 has higher occurrence count
        assert entries[0]["id"] == "P002"
        assert entries[1]["id"] == "P001"


class TestNewPatternHasQualityFields:
    """Tests that _process_patterns() initializes quality fields on new patterns."""

    def test_new_pattern_has_quality_defaults(self, mem):
        """New patterns created via _process_patterns() have quality fields."""
        mem._process_patterns("C001", ["When adding {tools}, update tests"], "2026-01-01")
        data = mem._load_patterns()
        p = list(data["patterns"].values())[0]
        assert p["success_count"] == 0
        assert p["failure_count"] == 0
        assert p["quality_score"] == 0.5
        assert p["last_quality_update"] == ""


class TestEvictionUsesQuality:
    """Tests that buffer eviction considers quality_score."""

    def test_low_quality_evicted_first(self, mem):
        """When buffer is full, lowest quality_score pattern is evicted first."""
        # Set max buffer size to 2
        mem.config.pattern_max_buffer_size = 2

        patterns_path = mem._get_patterns_path()
        data = {
            "version": 1,
            "patterns": {
                "P001": {
                    "text": "High quality pattern about architecture",
                    "first_seen": "C001",
                    "commit_ids": ["C001"],
                    "occurrence_count": 1,
                    "created_at": "2026-01-01 00:00",
                    "promoted": False,
                    "success_count": 5,
                    "failure_count": 0,
                    "quality_score": 0.86,
                    "last_quality_update": "",
                },
                "P002": {
                    "text": "Low quality pattern about debugging",
                    "first_seen": "C002",
                    "commit_ids": ["C002"],
                    "occurrence_count": 1,
                    "created_at": "2026-01-02 00:00",
                    "promoted": False,
                    "success_count": 0,
                    "failure_count": 5,
                    "quality_score": 0.14,
                    "last_quality_update": "",
                },
                "P003": {
                    "text": "Medium quality pattern about refactoring",
                    "first_seen": "C003",
                    "commit_ids": ["C003"],
                    "occurrence_count": 1,
                    "created_at": "2026-01-03 00:00",
                    "promoted": False,
                    "success_count": 2,
                    "failure_count": 2,
                    "quality_score": 0.5,
                    "last_quality_update": "",
                },
            },
            "next_id": 4,
        }
        os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
        with open(patterns_path, "w") as f:
            json.dump(data, f)

        # Trigger eviction
        mem._enforce_pattern_buffer_size(data)

        remaining = data["patterns"]
        assert len(remaining) == 2
        # P002 (quality=0.14) should have been evicted first
        assert "P002" not in remaining
        assert "P001" in remaining
        assert "P003" in remaining
