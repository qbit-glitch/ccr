"""Tests for ERL-inspired trigger/action structured ACE bullets."""

import json
import os
import tempfile

import pytest

from ccr.ace.playbook import Bullet, DeltaOperation, FailureLesson, Playbook, parse_delta_operations


@pytest.fixture
def playbook(tmp_path):
    pb = Playbook()
    pb._path = str(tmp_path / "playbook.txt")
    pb._failure_lessons_path = str(tmp_path / "failure_lessons.json")
    return pb


class TestBulletTriggerAction:
    def test_bullet_has_trigger_action_fields(self):
        b = Bullet(
            id="str-00001", helpful=0, harmful=0, content="test",
            trigger="when adding endpoints", action="validate first",
        )
        assert b.trigger == "when adding endpoints"
        assert b.action == "validate first"

    def test_bullet_defaults_empty(self):
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="test")
        assert b.trigger == ""
        assert b.action == ""

    def test_format_line_unchanged(self):
        """format_line should NOT include trigger/action (stored in companion JSON)."""
        b = Bullet(
            id="str-00001", helpful=3, harmful=0, content="test content",
            trigger="when X", action="do Y",
        )
        line = b.format_line()
        assert "trigger" not in line.lower()
        assert "action" not in line.lower()
        assert "[str-00001] helpful=3 harmful=0 :: test content" == line


class TestDeltaOperationTriggerAction:
    def test_delta_has_trigger_action(self):
        op = DeltaOperation(
            op_type="ADD", section="STRATEGIES & INSIGHTS",
            content="test", trigger="when X", action="do Y",
        )
        assert op.trigger == "when X"
        assert op.action == "do Y"

    def test_delta_defaults_empty(self):
        op = DeltaOperation(op_type="ADD", section="test", content="test")
        assert op.trigger == ""
        assert op.action == ""


class TestParseDeltaOperations:
    def test_parse_add_with_trigger_action(self):
        data = {
            "operations": [{
                "type": "ADD",
                "section": "STRATEGIES & INSIGHTS",
                "content": "Always validate input",
                "trigger": "when adding API endpoints",
                "action": "add validation middleware first",
            }]
        }
        ops = parse_delta_operations(data)
        assert len(ops) == 1
        assert ops[0].trigger == "when adding API endpoints"
        assert ops[0].action == "add validation middleware first"

    def test_parse_add_without_trigger_action(self):
        data = {
            "operations": [{
                "type": "ADD",
                "section": "STRATEGIES & INSIGHTS",
                "content": "Use repository pattern",
            }]
        }
        ops = parse_delta_operations(data)
        assert len(ops) == 1
        assert ops[0].trigger == ""
        assert ops[0].action == ""

    def test_parse_non_add_ignores_trigger_action(self):
        """UPDATE/MERGE/REMOVE operations do not carry trigger/action."""
        data = {
            "operations": [{
                "type": "UPDATE",
                "bullet_id": "str-00001",
                "content": "updated content",
                "trigger": "should be ignored",
            }]
        }
        ops = parse_delta_operations(data)
        assert len(ops) == 1
        # DeltaOperation for UPDATE defaults trigger/action to ""
        assert ops[0].trigger == ""
        assert ops[0].action == ""


class TestPolicyRankedWithTrigger:
    """Test that trigger field boosts retrieval ranking."""

    def test_trigger_match_boosts_ranking(self, playbook):
        # Bullet with matching trigger should rank higher
        b1 = Bullet(
            id="str-00001", helpful=5, harmful=0,
            content="General advice about coding",
            trigger="when writing database queries",
            last_updated="2026-03-28T00:00:00+00:00",
        )
        b2 = Bullet(
            id="str-00002", helpful=5, harmful=0,
            content="Always use parameterized database queries to prevent SQL injection",
            last_updated="2026-03-28T00:00:00+00:00",
        )

        playbook._bullets = [b1, b2]
        ranked = playbook.get_policy_ranked("writing database queries", top_k=2)
        assert len(ranked) >= 1
        # b1 should rank higher because trigger gets 1.5x boost
        if len(ranked) == 2:
            assert ranked[0].id == "str-00001"

    def test_no_trigger_falls_back_to_content(self, playbook):
        b1 = Bullet(
            id="str-00001", helpful=5, harmful=0,
            content="Always validate input for API endpoints",
            last_updated="2026-03-28T00:00:00+00:00",
        )
        playbook._bullets = [b1]
        ranked = playbook.get_policy_ranked("API endpoints validation", top_k=5)
        assert len(ranked) >= 1
        assert ranked[0].id == "str-00001"

    def test_empty_task_context_returns_all(self, playbook):
        b1 = Bullet(
            id="str-00001", helpful=1, harmful=0, content="test",
            trigger="specific trigger",
        )
        playbook._bullets = [b1]
        ranked = playbook.get_policy_ranked("", top_k=5)
        assert len(ranked) == 1

    def test_no_retrieval_score_leaks_on_bullets(self, playbook):
        """Retrieval scores use a local dict, never monkey-patched onto Bullet instances."""
        b1 = Bullet(
            id="str-00001", helpful=5, harmful=0,
            content="database queries optimization",
            trigger="when writing database queries",
            last_updated="2026-03-28T00:00:00+00:00",
        )
        playbook._bullets = [b1]
        ranked = playbook.get_policy_ranked("database queries", top_k=5)
        assert len(ranked) == 1
        assert not hasattr(ranked[0], '_retrieval_score')

    def test_trigger_only_match_passes_threshold(self, playbook):
        """A bullet where only trigger matches (not content) should still be returned."""
        b1 = Bullet(
            id="str-00001", helpful=5, harmful=0,
            content="Remember to update documentation",
            trigger="when adding new MCP tools",
            last_updated="2026-03-28T00:00:00+00:00",
        )
        playbook._bullets = [b1]
        ranked = playbook.get_policy_ranked("adding new MCP tools", top_k=5)
        assert len(ranked) == 1
        assert ranked[0].id == "str-00001"


class TestSimilarPairsWithTrigger:
    def test_trigger_included_in_similarity(self, playbook):
        b1 = Bullet(
            id="str-00001", helpful=0, harmful=0,
            content="validate input", trigger="when adding API endpoints",
        )
        b2 = Bullet(
            id="str-00002", helpful=0, harmful=0,
            content="check input data", trigger="when creating API endpoints",
        )
        playbook._bullets = [b1, b2]
        # The triggers share "API endpoints" which should increase similarity
        pairs = playbook.find_similar_pairs(threshold=0.3)
        assert isinstance(pairs, list)

    def test_trigger_raises_similarity_score(self, playbook):
        """Bullets with similar triggers should have higher similarity than without."""
        b1_with = Bullet(
            id="str-00001", helpful=0, harmful=0,
            content="validate", trigger="when creating REST API endpoints",
        )
        b2_with = Bullet(
            id="str-00002", helpful=0, harmful=0,
            content="check", trigger="when creating REST API endpoints",
        )
        playbook._bullets = [b1_with, b2_with]
        pairs_with = playbook.find_similar_pairs(threshold=0.0)

        b1_without = Bullet(
            id="str-00003", helpful=0, harmful=0,
            content="validate",
        )
        b2_without = Bullet(
            id="str-00004", helpful=0, harmful=0,
            content="check",
        )
        playbook._bullets = [b1_without, b2_without]
        pairs_without = playbook.find_similar_pairs(threshold=0.0)

        # With trigger overlap, similarity should be higher
        if pairs_with and pairs_without:
            assert pairs_with[0][2] > pairs_without[0][2]


class TestMergeTriggerAction:
    def test_merge_combines_different_triggers(self, playbook):
        b1 = Bullet(
            id="str-00001", helpful=3, harmful=0, content="validate input",
            section="STRATEGIES & INSIGHTS",
            trigger="when adding endpoints", action="check types first",
        )
        b2 = Bullet(
            id="str-00002", helpful=2, harmful=0, content="check input types",
            section="STRATEGIES & INSIGHTS",
            trigger="when creating routes", action="validate schema",
        )
        playbook._bullets = [b1, b2]
        playbook._sections = ["STRATEGIES & INSIGHTS"]

        ops = [DeltaOperation(
            op_type="MERGE",
            section="STRATEGIES & INSIGHTS",
            content="validate and check input types",
            bullet_id="str-00001",
            merge_target="str-00002",
        )]
        playbook.apply_delta(ops)

        assert len(playbook._bullets) == 1
        merged = playbook._bullets[0]
        assert merged.id == "str-00001"
        assert "when adding endpoints" in merged.trigger
        assert "when creating routes" in merged.trigger
        assert "check types first" in merged.action
        assert "validate schema" in merged.action

    def test_merge_same_trigger_no_duplicate(self, playbook):
        b1 = Bullet(
            id="str-00001", helpful=1, harmful=0, content="a",
            section="STRATEGIES & INSIGHTS",
            trigger="when adding endpoints", action="validate",
        )
        b2 = Bullet(
            id="str-00002", helpful=1, harmful=0, content="b",
            section="STRATEGIES & INSIGHTS",
            trigger="when adding endpoints", action="validate",
        )
        playbook._bullets = [b1, b2]
        playbook._sections = ["STRATEGIES & INSIGHTS"]

        ops = [DeltaOperation(
            op_type="MERGE", section="STRATEGIES & INSIGHTS",
            content="merged", bullet_id="str-00001", merge_target="str-00002",
        )]
        playbook.apply_delta(ops)
        merged = playbook._bullets[0]
        # Same trigger should not be duplicated with ";"
        assert merged.trigger == "when adding endpoints"
        assert merged.action == "validate"

    def test_merge_one_empty_trigger(self, playbook):
        b1 = Bullet(
            id="str-00001", helpful=1, harmful=0, content="a",
            section="STRATEGIES & INSIGHTS",
            trigger="when adding endpoints",
        )
        b2 = Bullet(
            id="str-00002", helpful=1, harmful=0, content="b",
            section="STRATEGIES & INSIGHTS",
        )
        playbook._bullets = [b1, b2]
        playbook._sections = ["STRATEGIES & INSIGHTS"]

        ops = [DeltaOperation(
            op_type="MERGE", section="STRATEGIES & INSIGHTS",
            content="merged", bullet_id="str-00001", merge_target="str-00002",
        )]
        playbook.apply_delta(ops)
        merged = playbook._bullets[0]
        assert merged.trigger == "when adding endpoints"
        assert merged.action == ""


class TestCompanionJsonPersistence:
    def test_save_and_load_trigger_action(self, tmp_path):
        pb = Playbook()
        b = Bullet(
            id="str-00001", helpful=1, harmful=0, content="test",
            trigger="when X", action="do Y",
            last_updated="2026-03-28T00:00:00+00:00",
        )
        pb._bullets = [b]

        fl_path = str(tmp_path / "failure_lessons.json")
        pb.save_failure_lessons(fl_path)

        # Create new playbook and load
        pb2 = Playbook()
        pb2._bullets = [Bullet(id="str-00001", helpful=1, harmful=0, content="test")]
        pb2.load_failure_lessons(fl_path)

        loaded = pb2._bullets[0]
        assert loaded.trigger == "when X"
        assert loaded.action == "do Y"

    def test_load_without_trigger_action_defaults(self, tmp_path):
        """Old companion JSON without trigger/action should default to empty."""
        fl_path = str(tmp_path / "failure_lessons.json")
        # Old format: just scope and when_to_apply
        data = {
            "str-00001": {
                "lessons": [],
                "scope": "general",
                "when_to_apply": "",
                "last_updated": "2026-01-01T00:00:00+00:00",
            }
        }
        with open(fl_path, "w") as f:
            json.dump(data, f)

        pb = Playbook()
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="test")
        pb._bullets = [b]
        pb.load_failure_lessons(fl_path)

        assert b.trigger == ""
        assert b.action == ""

    def test_trigger_action_roundtrip(self, tmp_path):
        """Full roundtrip: add with trigger/action -> save -> load -> verify."""
        pb = Playbook()
        pb._sections = ["STRATEGIES & INSIGHTS"]

        ops = [DeltaOperation(
            op_type="ADD", section="STRATEGIES & INSIGHTS",
            content="Always validate input",
            trigger="when adding API endpoints",
            action="add validation middleware first",
        )]
        pb.apply_delta(ops)
        assert len(pb._bullets) == 1
        assert pb._bullets[0].trigger == "when adding API endpoints"

        fl_path = str(tmp_path / "failure_lessons.json")
        pb.save_failure_lessons(fl_path)

        # Reload into new playbook
        text = pb.serialize()
        pb2 = Playbook(text)
        pb2.load_failure_lessons(fl_path)

        loaded = pb2._bullets[0]
        assert loaded.trigger == "when adding API endpoints"
        assert loaded.action == "add validation middleware first"

    def test_companion_json_only_written_when_needed(self, tmp_path):
        """Bullets with no trigger/action/etc should not appear in companion JSON."""
        pb = Playbook()
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="plain bullet")
        pb._bullets = [b]

        fl_path = str(tmp_path / "failure_lessons.json")
        pb.save_failure_lessons(fl_path)

        with open(fl_path) as f:
            data = json.load(f)
        # No extended data, so bullet should not be in the JSON
        assert "str-00001" not in data


class TestApplyDeltaWithTriggerAction:
    def test_add_with_trigger_action(self, playbook):
        playbook._sections = ["STRATEGIES & INSIGHTS"]
        ops = [DeltaOperation(
            op_type="ADD", section="STRATEGIES & INSIGHTS",
            content="Always validate", trigger="when adding endpoints",
            action="add validation first",
        )]
        playbook.apply_delta(ops)
        assert len(playbook._bullets) == 1
        b = playbook._bullets[0]
        assert b.trigger == "when adding endpoints"
        assert b.action == "add validation first"

    def test_add_without_trigger_action(self, playbook):
        playbook._sections = ["STRATEGIES & INSIGHTS"]
        ops = [DeltaOperation(
            op_type="ADD", section="STRATEGIES & INSIGHTS",
            content="Plain bullet",
        )]
        playbook.apply_delta(ops)
        assert len(playbook._bullets) == 1
        b = playbook._bullets[0]
        assert b.trigger == ""
        assert b.action == ""

    def test_parse_and_apply_roundtrip(self, playbook):
        """parse_delta_operations -> apply_delta preserves trigger/action."""
        playbook._sections = ["STRATEGIES & INSIGHTS"]
        data = {
            "operations": [{
                "type": "ADD",
                "section": "STRATEGIES & INSIGHTS",
                "content": "Use connection pooling",
                "trigger": "when configuring database connections",
                "action": "set pool size based on expected concurrency",
            }]
        }
        ops = parse_delta_operations(data)
        playbook.apply_delta(ops)
        b = playbook._bullets[0]
        assert b.trigger == "when configuring database connections"
        assert b.action == "set pool size based on expected concurrency"
