"""Tests for contribution-weighted counters (AgentEvolver-inspired).

Validates that ace_update_counters supports an optional weight parameter
(0.0-1.0) for proportional credit/blame, while still incrementing integer
counters for backward compatibility.
"""

import json
import os
import tempfile

import pytest

from ccr.ace.playbook import Bullet, Playbook


SAMPLE_PLAYBOOK = """## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Always verify data types before processing
[str-00002] helpful=0 harmful=0 :: Consider edge cases in financial data

## OTHERS
"""


class TestDefaultWeight:
    def test_default_weight_is_1(self):
        """When no weight given, weighted_helpful increments by 1.0."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{"id": "str-00001", "tag": "helpful"}])

        b = pb.get_bullet("str-00001")
        assert b is not None
        assert b.helpful == 1
        assert b.weighted_helpful == 1.0

    def test_default_weight_harmful(self):
        """When no weight given on harmful tag, weighted_harmful increments by 1.0."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{"id": "str-00001", "tag": "harmful"}])

        b = pb.get_bullet("str-00001")
        assert b is not None
        assert b.harmful == 1
        assert b.weighted_harmful == 1.0


class TestExplicitWeight:
    def test_explicit_weight(self):
        """weight=0.5 adds 0.5 to weighted_helpful."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{"id": "str-00001", "tag": "helpful", "weight": 0.5}])

        b = pb.get_bullet("str-00001")
        assert b is not None
        assert b.helpful == 1  # Integer counter always increments by 1
        assert b.weighted_helpful == pytest.approx(0.5)

    def test_multiple_weighted_updates_accumulate(self):
        """Multiple weighted updates accumulate correctly."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{"id": "str-00001", "tag": "helpful", "weight": 0.3}])
        pb.update_bullet_counts([{"id": "str-00001", "tag": "helpful", "weight": 0.5}])

        b = pb.get_bullet("str-00001")
        assert b is not None
        assert b.helpful == 2
        assert b.weighted_helpful == pytest.approx(0.8)


class TestWeightClamping:
    def test_weight_clamped_high(self):
        """weight=5.0 clamped to 1.0."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{"id": "str-00001", "tag": "helpful", "weight": 5.0}])

        b = pb.get_bullet("str-00001")
        assert b is not None
        assert b.weighted_helpful == pytest.approx(1.0)

    def test_weight_clamped_low(self):
        """weight=-1.0 clamped to 0.0."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{"id": "str-00001", "tag": "helpful", "weight": -1.0}])

        b = pb.get_bullet("str-00001")
        assert b is not None
        assert b.helpful == 1  # Integer counter still increments
        assert b.weighted_helpful == pytest.approx(0.0)


class TestEffectiveScore:
    def test_effective_score_uses_weighted(self):
        """When weighted fields > 0, effective_score uses them instead of integer counts."""
        b = Bullet(
            id="str-00001",
            helpful=3,
            harmful=1,
            content="test",
            weighted_helpful=1.5,
            weighted_harmful=0.3,
        )
        # No last_updated = no decay, so effective_score = raw
        # Weighted raw = 1.5 - 0.3 = 1.2
        assert b.effective_score() == pytest.approx(1.2)
        # Verify integer score would be different
        assert b.score == 2  # 3 - 1

    def test_effective_score_falls_back_to_integer(self):
        """When both weighted fields are 0.0, uses integer counts."""
        b = Bullet(
            id="str-00001",
            helpful=5,
            harmful=2,
            content="test",
            weighted_helpful=0.0,
            weighted_harmful=0.0,
        )
        # No decay, falls back to integer: 5 - 2 = 3
        assert b.effective_score() == pytest.approx(3.0)

    def test_effective_score_weighted_with_only_harmful(self):
        """When only weighted_harmful > 0, weighted path is used."""
        b = Bullet(
            id="str-00001",
            helpful=2,
            harmful=3,
            content="test",
            weighted_helpful=0.0,
            weighted_harmful=0.7,
        )
        # weighted_harmful > 0 triggers weighted path: 0.0 - 0.7 = -0.7
        assert b.effective_score() == pytest.approx(-0.7)


class TestWeightedHarmful:
    def test_weighted_harmful(self):
        """weight on harmful tag accumulates in weighted_harmful."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{"id": "str-00002", "tag": "harmful", "weight": 0.4}])

        b = pb.get_bullet("str-00002")
        assert b is not None
        assert b.harmful == 1
        assert b.weighted_harmful == pytest.approx(0.4)
        assert b.weighted_helpful == pytest.approx(0.0)


class TestPersistenceRoundTrip:
    def test_persistence_round_trip(self):
        """save + load preserves weighted fields."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([
            {"id": "str-00001", "tag": "helpful", "weight": 0.7},
            {"id": "str-00002", "tag": "harmful", "weight": 0.3},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            pb.save_failure_lessons(path)

            # Verify the JSON contains weighted fields
            with open(path) as f:
                data = json.load(f)
            assert data["str-00001"]["weighted_helpful"] == pytest.approx(0.7)
            assert data["str-00002"]["weighted_harmful"] == pytest.approx(0.3)

            # Load into a fresh playbook
            pb2 = Playbook(SAMPLE_PLAYBOOK)
            pb2.load_failure_lessons(path)

            b1 = pb2.get_bullet("str-00001")
            assert b1 is not None
            assert b1.weighted_helpful == pytest.approx(0.7)
            assert b1.weighted_harmful == pytest.approx(0.0)

            b2 = pb2.get_bullet("str-00002")
            assert b2 is not None
            assert b2.weighted_harmful == pytest.approx(0.3)
            assert b2.weighted_helpful == pytest.approx(0.0)

    def test_backward_compat_load_missing_weighted(self):
        """Loading old-format JSON (no weighted fields) defaults to 0.0."""
        pb = Playbook(SAMPLE_PLAYBOOK)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            # Write old-format data without weighted fields
            old_data = {
                "str-00001": {
                    "last_updated": "2026-01-01T00:00:00+00:00",
                }
            }
            with open(path, "w") as f:
                json.dump(old_data, f)

            pb.load_failure_lessons(path)

            b = pb.get_bullet("str-00001")
            assert b is not None
            assert b.weighted_helpful == pytest.approx(0.0)
            assert b.weighted_harmful == pytest.approx(0.0)
            assert b.last_updated == "2026-01-01T00:00:00+00:00"
