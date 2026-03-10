"""Tests for Phase 5: Grow-and-Refine & Deduplication."""

import json
import pytest

from ccr.ace.agents import ACEDeduplicator
from ccr.utils.parsing import extract_json_from_llm as _extract_json
from ccr.ace.engine import ACEConfig, ACEEngine, AdaptationResult
from ccr.ace.playbook import (
    Bullet,
    DeltaOperation,
    Playbook,
    parse_delta_operations,
)


PLAYBOOK_WITH_DUPES = """## STRATEGIES & INSIGHTS
[str-00001] helpful=5 harmful=0 :: Always validate input data types before processing
[str-00002] helpful=3 harmful=0 :: Verify data types of inputs before processing them
[str-00003] helpful=8 harmful=0 :: Use binary search for sorted collections
[str-00004] helpful=2 harmful=0 :: Check edge cases for empty lists and None values

## COMMON MISTAKES TO AVOID
[mis-00005] helpful=4 harmful=0 :: Do not forget to handle timezone conversions
[mis-00006] helpful=1 harmful=0 :: Always handle timezones when working with dates

## OTHERS
[oth-00007] helpful=0 harmful=0 :: Unique strategy with no duplicates
"""


# --- Playbook delta operations: UPDATE, MERGE, REMOVE ---


class TestPlaybookUpdate:
    def test_update_bullet_content(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="UPDATE", section="", content="New improved content", bullet_id="str-00001")]
        applied = pb.apply_delta(ops)
        assert applied == 1
        b = pb.get_bullet("str-00001")
        assert b.content == "New improved content"
        assert b.helpful == 5  # counts preserved

    def test_update_preserves_counts(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="UPDATE", section="", content="Updated", bullet_id="str-00003")]
        pb.apply_delta(ops)
        b = pb.get_bullet("str-00003")
        assert b.helpful == 8
        assert b.harmful == 0

    def test_update_nonexistent_bullet(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="UPDATE", section="", content="New", bullet_id="fake-99999")]
        applied = pb.apply_delta(ops)
        assert applied == 0

    def test_update_no_bullet_id(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="UPDATE", section="", content="New")]
        applied = pb.apply_delta(ops)
        assert applied == 0

    def test_update_moves_section(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="UPDATE", section="OTHERS", content="Moved", bullet_id="str-00001")]
        pb.apply_delta(ops)
        b = pb.get_bullet("str-00001")
        assert b.section == "OTHERS"


class TestPlaybookMerge:
    def test_merge_two_bullets(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(
            op_type="MERGE",
            section="",
            content="Always validate input data types before processing",
            bullet_id="str-00001",
            merge_target="str-00002",
        )]
        applied = pb.apply_delta(ops)
        assert applied == 1
        # Keeper exists with combined counts
        keeper = pb.get_bullet("str-00001")
        assert keeper is not None
        assert keeper.helpful == 8  # 5 + 3
        assert keeper.content == "Always validate input data types before processing"
        # Absorbed is gone
        assert pb.get_bullet("str-00002") is None

    def test_merge_combines_harmful_counts(self):
        text = "## STRATEGIES & INSIGHTS\n[str-00001] helpful=2 harmful=1 :: A\n[str-00002] helpful=3 harmful=2 :: B\n"
        pb = Playbook(text)
        ops = [DeltaOperation(
            op_type="MERGE", section="", content="A+B",
            bullet_id="str-00001", merge_target="str-00002",
        )]
        pb.apply_delta(ops)
        keeper = pb.get_bullet("str-00001")
        assert keeper.helpful == 5  # 2 + 3
        assert keeper.harmful == 3  # 1 + 2

    def test_merge_nonexistent_keeper(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(
            op_type="MERGE", section="", content="X",
            bullet_id="fake-99999", merge_target="str-00001",
        )]
        applied = pb.apply_delta(ops)
        assert applied == 0

    def test_merge_nonexistent_target(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(
            op_type="MERGE", section="", content="X",
            bullet_id="str-00001", merge_target="fake-99999",
        )]
        applied = pb.apply_delta(ops)
        assert applied == 0

    def test_merge_no_ids(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="MERGE", section="", content="X")]
        applied = pb.apply_delta(ops)
        assert applied == 0

    def test_merge_reduces_bullet_count(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        initial = len(pb.bullets)
        ops = [DeltaOperation(
            op_type="MERGE", section="", content="merged",
            bullet_id="str-00001", merge_target="str-00002",
        )]
        pb.apply_delta(ops)
        assert len(pb.bullets) == initial - 1


class TestPlaybookRemove:
    def test_remove_bullet(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="REMOVE", section="", content="", bullet_id="oth-00007")]
        applied = pb.apply_delta(ops)
        assert applied == 1
        assert pb.get_bullet("oth-00007") is None

    def test_remove_nonexistent(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="REMOVE", section="", content="", bullet_id="fake-99999")]
        applied = pb.apply_delta(ops)
        assert applied == 0

    def test_remove_no_id(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="REMOVE", section="", content="")]
        applied = pb.apply_delta(ops)
        assert applied == 0

    def test_remove_preserves_others(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        ops = [DeltaOperation(op_type="REMOVE", section="", content="", bullet_id="str-00003")]
        pb.apply_delta(ops)
        assert pb.get_bullet("str-00001") is not None
        assert pb.get_bullet("str-00004") is not None


# --- Find similar pairs (text similarity heuristic) ---


class TestFindSimilarPairs:
    def test_finds_duplicates(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        pairs = pb.find_similar_pairs(threshold=0.35)
        # str-00001 and str-00002 are very similar (Jaccard ~0.36)
        assert len(pairs) >= 1
        ids_in_pairs = {(a.id, b.id) for a, b, _ in pairs}
        assert ("str-00001", "str-00002") in ids_in_pairs or ("str-00002", "str-00001") in ids_in_pairs

    def test_high_overlap_pair_detected(self):
        """Bullets with very high word overlap are detected."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=1 harmful=0 :: Always validate user input before processing data\n"
            "[str-00002] helpful=1 harmful=0 :: Always validate user input before processing results\n"
        )
        pb = Playbook(text)
        pairs = pb.find_similar_pairs(threshold=0.5)
        assert len(pairs) >= 1

    def test_no_pairs_high_threshold(self):
        text = "## STRATEGIES & INSIGHTS\n[str-00001] helpful=1 harmful=0 :: Apples\n[str-00002] helpful=1 harmful=0 :: Bananas\n"
        pb = Playbook(text)
        pairs = pb.find_similar_pairs(threshold=0.9)
        assert len(pairs) == 0

    def test_empty_playbook_no_pairs(self):
        pb = Playbook()
        pairs = pb.find_similar_pairs()
        assert len(pairs) == 0

    def test_single_bullet_no_pairs(self):
        text = "## STRATEGIES & INSIGHTS\n[str-00001] helpful=1 harmful=0 :: Only one\n"
        pb = Playbook(text)
        pairs = pb.find_similar_pairs()
        assert len(pairs) == 0

    def test_pairs_sorted_by_similarity(self):
        pb = Playbook(PLAYBOOK_WITH_DUPES)
        pairs = pb.find_similar_pairs(threshold=0.3)
        if len(pairs) >= 2:
            assert pairs[0][2] >= pairs[1][2]


# --- parse_delta_operations for new op types ---


class TestParseDeltaNewOps:
    def test_parse_update(self):
        output = {"operations": [
            {"type": "UPDATE", "bullet_id": "str-00001", "content": "New content"},
        ]}
        ops = parse_delta_operations(output)
        assert len(ops) == 1
        assert ops[0].op_type == "UPDATE"
        assert ops[0].bullet_id == "str-00001"
        assert ops[0].content == "New content"

    def test_parse_merge(self):
        output = {"operations": [
            {"type": "MERGE", "bullet_id": "str-00001", "merge_target": "str-00002", "content": "Merged"},
        ]}
        ops = parse_delta_operations(output)
        assert len(ops) == 1
        assert ops[0].op_type == "MERGE"
        assert ops[0].merge_target == "str-00002"

    def test_parse_remove(self):
        output = {"operations": [
            {"type": "REMOVE", "bullet_id": "str-00003"},
        ]}
        ops = parse_delta_operations(output)
        assert len(ops) == 1
        assert ops[0].op_type == "REMOVE"
        assert ops[0].bullet_id == "str-00003"

    def test_parse_mixed_operations(self):
        output = {"operations": [
            {"type": "ADD", "section": "OTHERS", "content": "New tip"},
            {"type": "MERGE", "bullet_id": "str-00001", "merge_target": "str-00002", "content": "Merged"},
            {"type": "REMOVE", "bullet_id": "str-00003"},
            {"type": "UPDATE", "bullet_id": "str-00004", "content": "Updated"},
        ]}
        ops = parse_delta_operations(output)
        assert len(ops) == 4
        types = [op.op_type for op in ops]
        assert types == ["ADD", "MERGE", "REMOVE", "UPDATE"]

    def test_parse_update_missing_id_ignored(self):
        output = {"operations": [
            {"type": "UPDATE", "content": "No ID"},
        ]}
        ops = parse_delta_operations(output)
        assert len(ops) == 0

    def test_parse_merge_missing_target_ignored(self):
        output = {"operations": [
            {"type": "MERGE", "bullet_id": "str-00001", "content": "Missing target"},
        ]}
        ops = parse_delta_operations(output)
        assert len(ops) == 0

    def test_parse_remove_missing_id_ignored(self):
        output = {"operations": [
            {"type": "REMOVE"},
        ]}
        ops = parse_delta_operations(output)
        assert len(ops) == 0


# --- ACEDeduplicator agent ---


class MockSubClient:
    def __init__(self, response="{}"):
        self._response = response
        self._captured_messages = []

    def completion(self, messages, **kwargs):
        self._captured_messages.append(messages)
        return self._response


class TestACEDeduplicator:
    def test_deduplicate_empty_pairs(self):
        client = MockSubClient()
        dedup = ACEDeduplicator(client)
        raw, ops = dedup.deduplicate([], "stats")
        assert raw == ""
        assert ops == []

    def test_deduplicate_returns_merge_ops(self):
        response = json.dumps({
            "reasoning": "These two say the same thing",
            "operations": [
                {"type": "MERGE", "bullet_id": "str-00001", "merge_target": "str-00002", "content": "Validate inputs"},
            ],
        })
        client = MockSubClient(response)
        dedup = ACEDeduplicator(client)
        pairs = [("str-00001", "Validate input types", "str-00002", "Check input types", 0.8)]
        raw, ops = dedup.deduplicate(pairs, "Total: 10 bullets")
        assert len(ops) == 1
        assert ops[0]["type"] == "MERGE"

    def test_deduplicate_returns_remove_ops(self):
        response = json.dumps({
            "reasoning": "Second is redundant",
            "operations": [
                {"type": "REMOVE", "bullet_id": "str-00002"},
            ],
        })
        client = MockSubClient(response)
        dedup = ACEDeduplicator(client)
        pairs = [("str-00001", "Full advice", "str-00002", "Subset of advice", 0.7)]
        raw, ops = dedup.deduplicate(pairs, "stats")
        assert len(ops) == 1
        assert ops[0]["type"] == "REMOVE"

    def test_deduplicate_passes_pairs_to_prompt(self):
        client = MockSubClient('{"reasoning":"x","operations":[]}')
        dedup = ACEDeduplicator(client)
        pairs = [("str-00001", "Content A", "str-00002", "Content B", 0.75)]
        dedup.deduplicate(pairs, "Total: 5")
        messages = client._captured_messages[0]
        user_msg = messages[1]["content"]
        assert "str-00001" in user_msg
        assert "Content A" in user_msg
        assert "0.75" in user_msg

    def test_deduplicate_caps_pairs(self):
        client = MockSubClient('{"reasoning":"x","operations":[]}')
        dedup = ACEDeduplicator(client)
        pairs = [(f"str-{i:05d}", f"Content {i}", f"str-{i+100:05d}", f"Content {i+100}", 0.7)
                 for i in range(20)]
        dedup.deduplicate(pairs, "stats")
        messages = client._captured_messages[0]
        user_msg = messages[1]["content"]
        # Should only include first 10 pairs
        assert "Pair 10" in user_msg
        assert "Pair 11" not in user_msg

    def test_deduplicate_handles_error(self):
        class FailClient:
            def completion(self, messages, **kw):
                raise RuntimeError("LLM down")

        dedup = ACEDeduplicator(FailClient())
        raw, ops = dedup.deduplicate(
            [("a", "x", "b", "y", 0.8)], "stats"
        )
        assert "LLM down" in raw
        assert ops == []

    def test_deduplicate_handles_bad_json(self):
        client = MockSubClient("not json at all")
        dedup = ACEDeduplicator(client)
        raw, ops = dedup.deduplicate(
            [("a", "x", "b", "y", 0.8)], "stats"
        )
        assert ops == []


# --- ACE Engine refinement integration ---


class TestACEEngineRefinement:
    def _make_engine_with_dupes(self, refinement_frequency=1):
        """Create an engine with a playbook containing duplicates."""
        merge_response = json.dumps({
            "reasoning": "Merging duplicates",
            "operations": [
                {"type": "MERGE", "bullet_id": "str-00001", "merge_target": "str-00002",
                 "content": "Always validate input data types before processing"},
            ],
        })

        class DedupMockClient:
            _call_idx = 0
            def completion(self, messages, **kwargs):
                self._call_idx += 1
                # Reflector response
                if self._call_idx % 3 == 1:
                    return json.dumps({
                        "reasoning": "ok", "error_identification": "none",
                        "root_cause_analysis": "n/a", "correct_approach": "good",
                        "key_insight": "n/a", "bullet_tags": [],
                    })
                # Curator response
                if self._call_idx % 3 == 2:
                    return json.dumps({"reasoning": "nothing to add", "operations": []})
                # Deduplicator response
                return merge_response

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(PLAYBOOK_WITH_DUPES)
            path = f.name

        config = ACEConfig(
            refinement_frequency=refinement_frequency,
            dedup_similarity_threshold=0.35,
        )
        engine = ACEEngine(DedupMockClient(), config=config, playbook_path=path)
        return engine, path

    def test_refinement_runs_at_frequency(self):
        engine, path = self._make_engine_with_dupes(refinement_frequency=1)
        try:
            result = engine.adapt_online("task", "trace", "result", True)
            assert result.refinement_ran
        finally:
            import os
            os.unlink(path)

    def test_refinement_skips_when_not_due(self):
        engine, path = self._make_engine_with_dupes(refinement_frequency=5)
        try:
            # Step 1: not a multiple of 5
            result = engine.adapt_online("task", "trace", "result", True)
            assert not result.refinement_ran
        finally:
            import os
            os.unlink(path)

    def test_refinement_merges_bullets(self):
        engine, path = self._make_engine_with_dupes(refinement_frequency=1)
        try:
            initial_count = len(engine.playbook.bullets)
            result = engine.adapt_online("task", "trace", "result", True)
            assert result.bullets_merged >= 1
            assert len(engine.playbook.bullets) < initial_count
        finally:
            import os
            os.unlink(path)

    def test_adaptation_result_has_merge_field(self):
        result = AdaptationResult()
        assert result.bullets_merged == 0
        assert result.refinement_ran is False

    def test_ace_config_has_refinement_fields(self):
        config = ACEConfig()
        assert config.refinement_frequency == 10
        assert config.dedup_similarity_threshold == 0.6

    def test_refinement_disabled_when_frequency_zero(self):
        """refinement_frequency=0 means never refine."""
        class NoCallClient:
            _call_idx = 0
            def completion(self, messages, **kwargs):
                self._call_idx += 1
                if self._call_idx % 2 == 1:
                    return json.dumps({
                        "reasoning": "ok", "key_insight": "n/a", "bullet_tags": [],
                        "error_identification": "none", "root_cause_analysis": "n/a",
                        "correct_approach": "ok",
                    })
                return json.dumps({"reasoning": "ok", "operations": []})

        config = ACEConfig(refinement_frequency=0)
        engine = ACEEngine(NoCallClient(), config=config)
        # Run many steps — refinement should never fire
        for i in range(15):
            result = engine.adapt_online(f"task {i}", "trace", "result", True)
            assert not result.refinement_ran

    def test_refinement_needs_min_bullets(self):
        """Refinement skips when playbook has < 3 bullets."""
        class SimpleClient:
            _call_idx = 0
            def completion(self, messages, **kwargs):
                self._call_idx += 1
                if self._call_idx % 2 == 1:
                    return json.dumps({
                        "reasoning": "ok", "key_insight": "n/a", "bullet_tags": [],
                        "error_identification": "none", "root_cause_analysis": "n/a",
                        "correct_approach": "ok",
                    })
                return json.dumps({"reasoning": "nothing", "operations": []})

        config = ACEConfig(refinement_frequency=1)
        engine = ACEEngine(SimpleClient(), config=config)
        # Empty playbook: < 3 bullets
        result = engine.adapt_online("task", "trace", "result", True)
        assert not result.refinement_ran
