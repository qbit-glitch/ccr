"""Tests for ACE paper gap implementations.

Covers: iterative reflection, offline adaptation, warmup, parallel delta merging,
curator prompt updates, enhanced trigram similarity, and AdaptationResult fields.
"""

import json
import os
import tempfile

import pytest

from ccr.ace.engine import ACEConfig, ACEEngine, AdaptationResult
from ccr.ace.playbook import (
    Bullet,
    DeltaOperation,
    Playbook,
)
from ccr.ace.prompts import CURATOR_SYSTEM, CURATOR_USER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reflector_response(key_insight="Keep validating inputs", bullet_tags=None):
    return json.dumps({
        "reasoning": "Analysis",
        "error_identification": "No errors",
        "root_cause_analysis": "N/A",
        "correct_approach": "Current approach works",
        "key_insight": key_insight,
        "bullet_tags": bullet_tags or [],
    })


def _make_curator_response(operations=None):
    return json.dumps({
        "reasoning": "Adding insight",
        "operations": operations or [],
    })


class _SequenceClient:
    """Returns pre-configured responses in order, cycling if needed."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0

    def completion(self, messages, **kwargs):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


# ---------------------------------------------------------------------------
# 1-3. Reflector iterative refinement
# ---------------------------------------------------------------------------

class TestReflectorIterativeRefinement:
    def test_runs_multiple_rounds(self):
        """Reflector should run multiple rounds when key_insight is non-trivial."""
        # Round 1: returns a real insight -> should continue
        # Round 2: returns a real insight -> should continue
        # Round 3: returns "n/a" -> should stop
        reflector_r1 = _make_reflector_response(key_insight="Check edge cases")
        reflector_r2 = _make_reflector_response(key_insight="Validate types")
        reflector_r3 = _make_reflector_response(key_insight="n/a")
        curator_resp = _make_curator_response()

        # Reflector is called up to max_reflection_rounds times, then curator
        client = _SequenceClient([reflector_r1, reflector_r2, reflector_r3, curator_resp])
        config = ACEConfig(max_reflection_rounds=3)
        engine = ACEEngine(client, config=config)

        result = engine.adapt_online("task", "trace", "result", True)
        assert result.reflection_ran
        assert result.reflection_rounds == 3

    def test_early_stop_when_no_insight(self):
        """Reflector stops early if key_insight is empty/n/a."""
        reflector_resp = _make_reflector_response(key_insight="")
        curator_resp = _make_curator_response()

        client = _SequenceClient([reflector_resp, curator_resp])
        config = ACEConfig(max_reflection_rounds=5)
        engine = ACEEngine(client, config=config)

        result = engine.adapt_online("task", "trace", "result", True)
        assert result.reflection_ran
        assert result.reflection_rounds == 1

    def test_first_round_failure_returns_early(self):
        """If the first reflection round raises past the reflector's internal
        error handling, adapt_online returns immediately.

        Note: ACEReflector catches its own LLM errors internally and returns
        (error_str, [], ""). To test the engine's own error handling, we need
        the reflector.reflect() call itself to raise (e.g., a non-LLM error).
        """
        from unittest.mock import patch

        reflector_resp = _make_reflector_response(key_insight="n/a")
        curator_resp = _make_curator_response()
        client = _SequenceClient([reflector_resp, curator_resp])
        engine = ACEEngine(client)

        # Patch the reflector to raise on the first call
        with patch.object(engine.reflector, "reflect", side_effect=ValueError("bad input")):
            result = engine.adapt_online("task", "trace", "result", True)

        assert not result.reflection_ran
        assert result.reflection_rounds == 0
        # Should not crash, curator should not have run
        assert not result.curator_ran


# ---------------------------------------------------------------------------
# 4-5. Offline adaptation
# ---------------------------------------------------------------------------

class TestAdaptOffline:
    def test_offline_multiple_samples(self):
        """adapt_offline processes all samples and returns results."""
        reflector_resp = _make_reflector_response(key_insight="n/a")
        curator_resp = _make_curator_response([
            {"type": "ADD", "section": "OTHERS", "content": "Tip"}
        ])
        client = _SequenceClient([reflector_resp, curator_resp])
        engine = ACEEngine(client)

        samples = [
            {"task": "task1", "execution_trace": "t1", "execution_result": "r1", "was_successful": True},
            {"task": "task2", "execution_trace": "t2", "execution_result": "r2", "was_successful": False},
            {"task": "task3", "execution_trace": "t3", "execution_result": "r3", "was_successful": True},
        ]
        results = engine.adapt_offline(samples, epochs=1)
        assert len(results) == 3
        assert all(isinstance(r, AdaptationResult) for r in results)

    def test_offline_multiple_epochs(self):
        """adapt_offline with epochs=2 processes samples twice."""
        reflector_resp = _make_reflector_response(key_insight="n/a")
        curator_resp = _make_curator_response()
        client = _SequenceClient([reflector_resp, curator_resp])
        engine = ACEEngine(client)

        samples = [
            {"task": "task1", "execution_trace": "t", "execution_result": "r", "was_successful": True},
            {"task": "task2", "execution_trace": "t", "execution_result": "r", "was_successful": True},
        ]
        results = engine.adapt_offline(samples, epochs=2)
        assert len(results) == 4  # 2 samples * 2 epochs


# ---------------------------------------------------------------------------
# 6-7. Warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_warmup_adds_seed_bullets(self):
        """warmup adds seed bullets to the playbook."""
        reflector_resp = _make_reflector_response(key_insight="n/a")
        client = _SequenceClient([reflector_resp])
        engine = ACEEngine(client)

        seeds = [
            {"section": "STRATEGIES & INSIGHTS", "content": "Seed tip 1"},
            {"section": "COMMON MISTAKES TO AVOID", "content": "Seed warning 1"},
            {"section": "OTHERS", "content": "Seed other"},
        ]
        added = engine.warmup(seeds)
        assert added == 3
        assert len(engine.playbook.bullets) == 3

    def test_warmup_saves_playbook(self):
        """warmup persists the playbook to disk."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name

        try:
            client = _SequenceClient([_make_reflector_response()])
            engine = ACEEngine(client, playbook_path=path)

            engine.warmup([{"section": "OTHERS", "content": "Persisted tip"}])

            with open(path) as f:
                content = f.read()
            assert "Persisted tip" in content
        finally:
            os.unlink(path)

    def test_warmup_skips_empty_content(self):
        """warmup ignores bullets with empty content."""
        client = _SequenceClient([_make_reflector_response()])
        engine = ACEEngine(client)

        seeds = [
            {"section": "OTHERS", "content": ""},
            {"section": "OTHERS", "content": "Real tip"},
        ]
        added = engine.warmup(seeds)
        assert added == 1


# ---------------------------------------------------------------------------
# 8. Parallel delta merging order (ADDs before REMOVEs)
# ---------------------------------------------------------------------------

class TestParallelDeltaMerging:
    def test_adds_before_removes(self):
        """ADDs are applied before REMOVEs so that a newly-added bullet
        is not accidentally removed by a REMOVE targeting the same ID."""
        text = "## OTHERS\n[oth-00001] helpful=0 harmful=0 :: Old bullet\n"
        pb = Playbook(text)

        # Mix: REMOVE first in list, ADD second — but ADDs should execute first
        ops = [
            DeltaOperation(op_type="REMOVE", section="", content="", bullet_id="oth-00001"),
            DeltaOperation(op_type="ADD", section="OTHERS", content="Brand new"),
        ]
        applied = pb.apply_delta(ops)
        assert applied == 2
        # The old bullet should be removed and the new one present
        assert pb.get_bullet("oth-00001") is None
        assert any("Brand new" in b.content for b in pb.bullets)

    def test_operation_order_add_update_merge_remove(self):
        """Operations are grouped: ADDs, UPDATEs, MERGEs, REMOVEs."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=2 harmful=0 :: First\n"
            "[str-00002] helpful=1 harmful=0 :: Second\n"
            "[str-00003] helpful=0 harmful=0 :: Third\n"
        )
        pb = Playbook(text)

        ops = [
            DeltaOperation(op_type="REMOVE", section="", content="", bullet_id="str-00003"),
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Fourth"),
            DeltaOperation(op_type="UPDATE", section="", content="Updated first", bullet_id="str-00001"),
        ]
        applied = pb.apply_delta(ops)
        assert applied == 3

        # ADD happened (new bullet exists)
        assert any("Fourth" in b.content for b in pb.bullets)
        # UPDATE happened
        assert pb.get_bullet("str-00001").content == "Updated first"
        # REMOVE happened
        assert pb.get_bullet("str-00003") is None


# ---------------------------------------------------------------------------
# 9. Curator prompt mentions UPDATE/MERGE/REMOVE
# ---------------------------------------------------------------------------

class TestCuratorPromptOperations:
    def test_curator_system_mentions_update(self):
        assert "UPDATE" in CURATOR_SYSTEM

    def test_curator_system_mentions_merge(self):
        assert "MERGE" in CURATOR_SYSTEM

    def test_curator_system_mentions_remove(self):
        assert "REMOVE" in CURATOR_SYSTEM

    def test_curator_user_shows_all_op_types(self):
        assert "UPDATE" in CURATOR_USER
        assert "MERGE" in CURATOR_USER
        assert "REMOVE" in CURATOR_USER


# ---------------------------------------------------------------------------
# 10-11. Enhanced trigram similarity
# ---------------------------------------------------------------------------

class TestTrigramSimilarity:
    def test_char_trigrams_basic(self):
        result = Playbook._char_trigrams("hello")
        assert "hel" in result
        assert "ell" in result
        assert "llo" in result
        assert len(result) == 3

    def test_char_trigrams_short_text(self):
        assert Playbook._char_trigrams("ab") == set()
        assert Playbook._char_trigrams("") == set()

    def test_char_trigrams_exact_three(self):
        result = Playbook._char_trigrams("abc")
        assert result == {"abc"}

    def test_enhanced_similarity_detects_paraphrases(self):
        """Trigram similarity helps catch paraphrases that word-only Jaccard misses."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=1 harmful=0 :: Always validate user input before processing data\n"
            "[str-00002] helpful=1 harmful=0 :: Always validate user input before processing results\n"
        )
        pb = Playbook(text)
        pairs = pb.find_similar_pairs(threshold=0.5)
        assert len(pairs) >= 1
        # The combined score should be reported
        assert pairs[0][2] > 0.5

    def test_dissimilar_bullets_not_matched(self):
        """Very different bullets should not be paired even with trigrams."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=1 harmful=0 :: Use binary search for sorted arrays\n"
            "[str-00002] helpful=1 harmful=0 :: Handle timezone conversions carefully in dates\n"
        )
        pb = Playbook(text)
        pairs = pb.find_similar_pairs(threshold=0.6)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# 12. AdaptationResult has reflection_rounds field
# ---------------------------------------------------------------------------

class TestAdaptationResultFields:
    def test_has_reflection_rounds(self):
        result = AdaptationResult()
        assert hasattr(result, "reflection_rounds")
        assert result.reflection_rounds == 0

    def test_reflection_rounds_default(self):
        result = AdaptationResult()
        assert result.reflection_rounds == 0
        assert result.reflection_ran is False

    def test_all_fields_present(self):
        result = AdaptationResult()
        expected_fields = [
            "task_summary", "was_successful", "reflection_ran", "reflection_rounds",
            "curator_ran", "refinement_ran", "bullets_added", "bullets_updated",
            "bullets_pruned", "bullets_merged", "playbook_size",
        ]
        for field in expected_fields:
            assert hasattr(result, field), f"Missing field: {field}"
