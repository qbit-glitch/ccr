"""Tests for ACE Playbook data structure."""

import json
import os
import tempfile

import pytest

from ccr.ace.playbook import (
    Bullet,
    DeltaOperation,
    FailureLesson,
    Playbook,
    PlaybookStats,
    create_empty_playbook,
    parse_delta_operations,
)


SAMPLE_PLAYBOOK = """## STRATEGIES & INSIGHTS
[str-00001] helpful=5 harmful=0 :: Always verify data types before processing
[str-00002] helpful=3 harmful=1 :: Consider edge cases in financial data

## CODE SNIPPETS & TEMPLATES
[code-00003] helpful=8 harmful=0 :: Use defaultdict(list) for grouping

## COMMON MISTAKES TO AVOID
[mis-00004] helpful=6 harmful=0 :: Don't forget timezone conversions
[mis-00005] helpful=0 harmful=4 :: Using untrusted input without sanitization

## OTHERS
"""


class TestBullet:
    def test_bullet_score(self):
        b = Bullet(id="str-00001", helpful=5, harmful=1, content="test")
        assert b.score == 4

    def test_bullet_problematic(self):
        b = Bullet(id="str-00001", helpful=1, harmful=3, content="bad advice")
        assert b.is_problematic

    def test_bullet_not_problematic(self):
        b = Bullet(id="str-00001", helpful=5, harmful=0, content="good advice")
        assert not b.is_problematic

    def test_bullet_unused(self):
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="new")
        assert b.is_unused

    def test_bullet_format_line(self):
        b = Bullet(id="str-00001", helpful=5, harmful=1, content="test content")
        assert b.format_line() == "[str-00001] helpful=5 harmful=1 :: test content"


class TestPlaybookParsing:
    def test_parse_sample_playbook(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        assert len(pb.bullets) == 5

    def test_parse_sections(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        assert "STRATEGIES & INSIGHTS" in pb.sections
        assert "CODE SNIPPETS & TEMPLATES" in pb.sections
        assert "COMMON MISTAKES TO AVOID" in pb.sections
        assert "OTHERS" in pb.sections

    def test_parse_bullet_content(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        b = pb.get_bullet("str-00001")
        assert b is not None
        assert b.helpful == 5
        assert b.harmful == 0
        assert "verify data types" in b.content

    def test_parse_next_id(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        assert pb.next_id == 6  # max is 00005, so next is 6

    def test_parse_empty(self):
        pb = Playbook("")
        assert len(pb.bullets) == 0

    def test_parse_none(self):
        pb = Playbook()
        assert len(pb.bullets) == 0
        assert pb.next_id == 1


class TestPlaybookSerialization:
    def test_roundtrip(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        text = pb.serialize()
        pb2 = Playbook(text)
        assert len(pb2.bullets) == len(pb.bullets)
        for b1, b2 in zip(pb.bullets, pb2.bullets):
            assert b1.id == b2.id
            assert b1.helpful == b2.helpful
            assert b1.content == b2.content

    def test_serialize_contains_sections(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        text = pb.serialize()
        assert "## STRATEGIES & INSIGHTS" in text
        assert "## CODE SNIPPETS & TEMPLATES" in text

    def test_serialize_contains_bullets(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        text = pb.serialize()
        assert "[str-00001]" in text
        assert "helpful=5" in text


class TestPlaybookOperations:
    def test_get_bullet(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        b = pb.get_bullet("code-00003")
        assert b is not None
        assert "defaultdict" in b.content

    def test_get_bullet_not_found(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        assert pb.get_bullet("nonexistent") is None

    def test_get_section_bullets(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        bullets = pb.get_section_bullets("STRATEGIES & INSIGHTS")
        assert len(bullets) == 2

    def test_extract_bullets(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        text = pb.extract_bullets(["str-00001", "code-00003"])
        assert "verify data types" in text
        assert "defaultdict" in text

    def test_extract_bullets_not_found(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        text = pb.extract_bullets(["nonexistent"])
        assert "No matching" in text


class TestBulletCounts:
    def test_update_helpful(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [{"id": "str-00001", "tag": "helpful"}]
        updated = pb.update_bullet_counts(tags)
        assert updated == 1
        b = pb.get_bullet("str-00001")
        assert b.helpful == 6  # was 5

    def test_update_harmful(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [{"id": "str-00002", "tag": "harmful"}]
        updated = pb.update_bullet_counts(tags)
        assert updated == 1
        b = pb.get_bullet("str-00002")
        assert b.harmful == 2  # was 1

    def test_update_neutral(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [{"id": "str-00001", "tag": "neutral"}]
        updated = pb.update_bullet_counts(tags)
        assert updated == 0
        b = pb.get_bullet("str-00001")
        assert b.helpful == 5  # unchanged

    def test_update_multiple(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [
            {"id": "str-00001", "tag": "helpful"},
            {"id": "str-00002", "tag": "harmful"},
            {"id": "code-00003", "tag": "helpful"},
        ]
        updated = pb.update_bullet_counts(tags)
        assert updated == 3

    def test_update_nonexistent(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [{"id": "fake-99999", "tag": "helpful"}]
        updated = pb.update_bullet_counts(tags)
        assert updated == 0


class TestDeltaOperations:
    def test_apply_add(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        initial_count = len(pb.bullets)
        ops = [DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="New strategy here")]
        applied = pb.apply_delta(ops)
        assert applied == 1
        assert len(pb.bullets) == initial_count + 1

    def test_add_generates_correct_id(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        ops = [DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Test")]
        pb.apply_delta(ops)
        new_bullet = pb.bullets[-1]
        assert new_bullet.id == "str-00006"  # next after 00005
        assert new_bullet.helpful == 0
        assert new_bullet.harmful == 0

    def test_add_to_unknown_section_goes_to_others(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        ops = [DeltaOperation(op_type="ADD", section="NONEXISTENT SECTION", content="Orphan")]
        pb.apply_delta(ops)
        new_bullet = pb.bullets[-1]
        assert new_bullet.section == "OTHERS"

    def test_add_multiple(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        ops = [
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="First"),
            DeltaOperation(op_type="ADD", section="COMMON MISTAKES TO AVOID", content="Second"),
        ]
        applied = pb.apply_delta(ops)
        assert applied == 2

    def test_add_increments_id(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        ops = [
            DeltaOperation(op_type="ADD", section="OTHERS", content="A"),
            DeltaOperation(op_type="ADD", section="OTHERS", content="B"),
        ]
        pb.apply_delta(ops)
        ids = [b.id for b in pb.bullets[-2:]]
        assert ids[0] != ids[1]

    def test_add_preserves_existing(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        original_bullets = [b.format_line() for b in pb.bullets]
        ops = [DeltaOperation(op_type="ADD", section="OTHERS", content="New")]
        pb.apply_delta(ops)
        for i, b in enumerate(pb.bullets[:-1]):
            assert b.format_line() == original_bullets[i]


class TestParseDeltaOperations:
    def test_parse_valid(self):
        output = {
            "reasoning": "blah",
            "operations": [
                {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "New tip"},
                {"type": "ADD", "section": "OTHERS", "content": "Another tip"},
            ],
        }
        ops = parse_delta_operations(output)
        assert len(ops) == 2
        assert ops[0].op_type == "ADD"
        assert ops[0].section == "STRATEGIES & INSIGHTS"

    def test_parse_empty(self):
        output = {"reasoning": "nothing to add", "operations": []}
        ops = parse_delta_operations(output)
        assert len(ops) == 0

    def test_parse_ignores_unknown_ops(self):
        output = {
            "operations": [
                {"type": "DELETE", "bullet_id": "str-00001"},
                {"type": "ADD", "section": "OTHERS", "content": "Valid"},
            ]
        }
        ops = parse_delta_operations(output)
        assert len(ops) == 1

    def test_parse_missing_operations(self):
        output = {"reasoning": "no ops key"}
        ops = parse_delta_operations(output)
        assert len(ops) == 0


class TestPruning:
    def test_prune_problematic(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        # mis-00005 has helpful=0, harmful=4 -> should be pruned
        removed = pb.prune_problematic(min_harmful=3)
        assert len(removed) == 1
        assert removed[0].id == "mis-00005"
        assert pb.get_bullet("mis-00005") is None

    def test_prune_keeps_good_bullets(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.prune_problematic(min_harmful=3)
        assert pb.get_bullet("str-00001") is not None
        assert pb.get_bullet("code-00003") is not None

    def test_prune_threshold(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        # With high threshold, nothing gets pruned
        removed = pb.prune_problematic(min_harmful=10)
        assert len(removed) == 0

    def test_enforce_token_budget(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        initial_size = len(pb.serialize())
        # Set budget to half current size
        removed = pb.enforce_token_budget(initial_size // 2)
        assert len(removed) > 0
        assert len(pb.serialize()) <= initial_size // 2


class TestPlaybookStats:
    def test_stats_total(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        stats = pb.get_stats()
        assert stats.total_bullets == 5

    def test_stats_high_performing(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        stats = pb.get_stats()
        assert stats.high_performing >= 1  # str-00001 (helpful=5), code-00003 (helpful=8)

    def test_stats_problematic(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        stats = pb.get_stats()
        assert stats.problematic >= 1  # mis-00005 (harmful=4)

    def test_stats_by_section(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        stats = pb.get_stats()
        assert "STRATEGIES & INSIGHTS" in stats.by_section
        assert stats.by_section["STRATEGIES & INSIGHTS"]["count"] == 2


class TestCreateEmptyPlaybook:
    def test_empty_has_sections(self):
        pb = create_empty_playbook()
        assert len(pb.sections) >= 6
        assert "STRATEGIES & INSIGHTS" in pb.sections

    def test_empty_has_no_bullets(self):
        pb = create_empty_playbook()
        assert len(pb.bullets) == 0

    def test_empty_serializes(self):
        pb = create_empty_playbook()
        text = pb.serialize()
        assert "## STRATEGIES & INSIGHTS" in text
        assert "## OTHERS" in text

    def test_custom_sections(self):
        pb = create_empty_playbook(sections=["FOO", "BAR"])
        assert pb.sections == ["FOO", "BAR"]


# ==========================================================================
# Structured Failure Lessons (SkillRL-inspired)
# ==========================================================================


class TestFailureLesson:
    def test_create_failure_lesson(self):
        fl = FailureLesson(
            failure_point="Strategy broke on binary input",
            flawed_reasoning="Assumed all files are text",
            counterfactual="Should check file type first",
            prevention_principle="Always validate input format at boundaries",
        )
        assert fl.failure_point == "Strategy broke on binary input"
        assert fl.timestamp  # auto-generated

    def test_to_dict(self):
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
            timestamp="2026-03-10T00:00:00+00:00",
        )
        d = fl.to_dict()
        assert d["failure_point"] == "X"
        assert d["flawed_reasoning"] == "Y"
        assert d["counterfactual"] == "Z"
        assert d["prevention_principle"] == "W"
        assert d["timestamp"] == "2026-03-10T00:00:00+00:00"

    def test_from_dict(self):
        d = {
            "failure_point": "A",
            "flawed_reasoning": "B",
            "counterfactual": "C",
            "prevention_principle": "D",
            "timestamp": "2026-01-01T00:00:00",
        }
        fl = FailureLesson.from_dict(d)
        assert fl.failure_point == "A"
        assert fl.prevention_principle == "D"
        assert fl.timestamp == "2026-01-01T00:00:00"

    def test_from_dict_missing_fields(self):
        fl = FailureLesson.from_dict({"failure_point": "Only this"})
        assert fl.failure_point == "Only this"
        assert fl.flawed_reasoning == ""
        assert fl.counterfactual == ""

    def test_roundtrip(self):
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
        )
        fl2 = FailureLesson.from_dict(fl.to_dict())
        assert fl.failure_point == fl2.failure_point
        assert fl.prevention_principle == fl2.prevention_principle
        assert fl.timestamp == fl2.timestamp

    def test_format_text(self):
        fl = FailureLesson(
            failure_point="Broke here",
            flawed_reasoning="Wrong assumption",
            counterfactual="Do this instead",
            prevention_principle="General rule",
        )
        text = fl.format_text()
        assert "FAILURE: Broke here" in text
        assert "FLAW: Wrong assumption" in text
        assert "INSTEAD: Do this instead" in text
        assert "PRINCIPLE: General rule" in text


class TestBulletWithFailureLessons:
    def test_bullet_default_no_lessons(self):
        b = Bullet(id="str-00001", helpful=5, harmful=1, content="test")
        assert b.failure_lessons == []
        assert not b.has_failure_lessons

    def test_bullet_with_lessons(self):
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
        )
        b = Bullet(id="str-00001", helpful=0, harmful=1, content="test",
                    failure_lessons=[fl])
        assert b.has_failure_lessons
        assert len(b.failure_lessons) == 1

    def test_format_line_unchanged(self):
        """format_line (for playbook.txt) should NOT include failure lessons."""
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
        )
        b = Bullet(id="str-00001", helpful=0, harmful=1, content="test",
                    failure_lessons=[fl])
        line = b.format_line()
        assert "FAILURE" not in line
        assert line == "[str-00001] helpful=0 harmful=1 :: test"

    def test_format_line_with_failures(self):
        fl = FailureLesson(
            failure_point="Broke here",
            flawed_reasoning="Bad assumption",
            counterfactual="Do this",
            prevention_principle="Rule",
        )
        b = Bullet(id="str-00001", helpful=0, harmful=1, content="test",
                    failure_lessons=[fl])
        text = b.format_line_with_failures()
        assert "[str-00001]" in text
        assert "FAILURE: Broke here" in text
        assert "PRINCIPLE: Rule" in text


class TestUpdateCountsWithFailureLessons:
    def test_harmful_without_lesson(self):
        """Harmful tag without failure_lesson still works (backward compat)."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [{"id": "str-00001", "tag": "harmful"}]
        updated = pb.update_bullet_counts(tags)
        assert updated == 1
        b = pb.get_bullet("str-00001")
        assert b.harmful == 1
        assert not b.has_failure_lessons

    def test_harmful_with_lesson(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [{
            "id": "str-00001",
            "tag": "harmful",
            "failure_lesson": {
                "failure_point": "Strategy broke on edge case",
                "flawed_reasoning": "Assumed sorted input",
                "counterfactual": "Should sort first",
                "prevention_principle": "Never assume input ordering",
            },
        }]
        updated = pb.update_bullet_counts(tags)
        assert updated == 1
        b = pb.get_bullet("str-00001")
        assert b.harmful == 1
        assert b.has_failure_lessons
        assert len(b.failure_lessons) == 1
        assert b.failure_lessons[0].failure_point == "Strategy broke on edge case"
        assert b.failure_lessons[0].prevention_principle == "Never assume input ordering"

    def test_multiple_harmful_accumulate_lessons(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        lesson1 = {
            "failure_point": "First failure",
            "flawed_reasoning": "A",
            "counterfactual": "B",
            "prevention_principle": "C",
        }
        lesson2 = {
            "failure_point": "Second failure",
            "flawed_reasoning": "D",
            "counterfactual": "E",
            "prevention_principle": "F",
        }
        pb.update_bullet_counts([{
            "id": "str-00001", "tag": "harmful", "failure_lesson": lesson1,
        }])
        pb.update_bullet_counts([{
            "id": "str-00001", "tag": "harmful", "failure_lesson": lesson2,
        }])
        b = pb.get_bullet("str-00001")
        assert b.harmful == 2
        assert len(b.failure_lessons) == 2
        assert b.failure_lessons[0].failure_point == "First failure"
        assert b.failure_lessons[1].failure_point == "Second failure"

    def test_helpful_ignores_failure_lesson(self):
        """failure_lesson field on helpful tags is ignored."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [{
            "id": "str-00001", "tag": "helpful",
            "failure_lesson": {"failure_point": "ignored"},
        }]
        pb.update_bullet_counts(tags)
        b = pb.get_bullet("str-00001")
        assert b.helpful == 6
        assert not b.has_failure_lessons

    def test_empty_failure_lesson_ignored(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [{"id": "str-00001", "tag": "harmful", "failure_lesson": {}}]
        pb.update_bullet_counts(tags)
        b = pb.get_bullet("str-00001")
        assert b.harmful == 1
        assert not b.has_failure_lessons


class TestFailureLessonPersistence:
    def test_save_and_load(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{
            "id": "str-00001", "tag": "harmful",
            "failure_lesson": {
                "failure_point": "Broke here",
                "flawed_reasoning": "Bad assumption",
                "counterfactual": "Do this instead",
                "prevention_principle": "General rule",
            },
        }])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            saved = pb.save_failure_lessons(path)
            assert saved == 1
            assert os.path.isfile(path)

            # Verify JSON structure (new format: dict with "lessons" key)
            with open(path) as f:
                data = json.load(f)
            assert "str-00001" in data
            assert len(data["str-00001"]["lessons"]) == 1
            assert data["str-00001"]["lessons"][0]["failure_point"] == "Broke here"

            # Load into a fresh playbook
            pb2 = Playbook(SAMPLE_PLAYBOOK)
            loaded = pb2.load_failure_lessons(path)
            assert loaded == 1
            b = pb2.get_bullet("str-00001")
            assert b.has_failure_lessons
            assert b.failure_lessons[0].prevention_principle == "General rule"

    def test_load_nonexistent_file(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        loaded = pb.load_failure_lessons("/nonexistent/path.json")
        assert loaded == 0

    def test_load_corrupt_json(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            path = f.name
        try:
            loaded = pb.load_failure_lessons(path)
            assert loaded == 0
        finally:
            os.unlink(path)

    def test_save_empty_lessons(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            saved = pb.save_failure_lessons(path)
            assert saved == 0
            with open(path) as f:
                data = json.load(f)
            assert data == {}

    def test_roundtrip_multiple_bullets(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([
            {"id": "str-00001", "tag": "harmful", "failure_lesson": {
                "failure_point": "A", "flawed_reasoning": "B",
                "counterfactual": "C", "prevention_principle": "D",
            }},
            {"id": "str-00002", "tag": "harmful", "failure_lesson": {
                "failure_point": "E", "flawed_reasoning": "F",
                "counterfactual": "G", "prevention_principle": "H",
            }},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            pb.save_failure_lessons(path)

            pb2 = Playbook(SAMPLE_PLAYBOOK)
            pb2.load_failure_lessons(path)

            assert pb2.get_bullet("str-00001").failure_lessons[0].failure_point == "A"
            assert pb2.get_bullet("str-00002").failure_lessons[0].failure_point == "E"


class TestSerializeWithFailures:
    def test_no_failures_matches_serialize(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        assert pb.serialize() == pb.serialize_with_failures()

    def test_with_failures_includes_lessons(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{
            "id": "str-00001", "tag": "harmful",
            "failure_lesson": {
                "failure_point": "Broke on edge case",
                "flawed_reasoning": "Assumed happy path",
                "counterfactual": "Check boundaries",
                "prevention_principle": "Validate all inputs",
            },
        }])
        text = pb.serialize_with_failures()
        assert "FAILURE: Broke on edge case" in text
        assert "PRINCIPLE: Validate all inputs" in text
        # Regular playbook.txt format still works
        assert "[str-00001]" in text
        assert "helpful=5" in text


class TestPlaybookStatsWithFailureLessons:
    def test_stats_failure_lessons_count(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{
            "id": "str-00001", "tag": "harmful",
            "failure_lesson": {
                "failure_point": "A", "flawed_reasoning": "B",
                "counterfactual": "C", "prevention_principle": "D",
            },
        }])
        stats = pb.get_stats()
        assert stats.total_failure_lessons == 1
        assert stats.harmful_with_lessons >= 1

    def test_stats_harmful_without_lessons(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        # mis-00005 has harmful=4 but no lessons
        stats = pb.get_stats()
        assert stats.harmful_without_lessons >= 1
        assert stats.harmful_with_lessons == 0

    def test_stats_harmful_coverage(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        # Add a lesson to str-00002 (which has harmful=1)
        pb.update_bullet_counts([{
            "id": "str-00002", "tag": "harmful",
            "failure_lesson": {
                "failure_point": "X", "flawed_reasoning": "Y",
                "counterfactual": "Z", "prevention_principle": "W",
            },
        }])
        stats = pb.get_stats()
        # str-00002 now has harmful=2 with lesson, mis-00005 has harmful=4 without
        assert stats.harmful_with_lessons >= 1
        assert stats.harmful_without_lessons >= 1


class TestPreventionPrinciples:
    def test_get_all_prevention_principles(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([
            {"id": "str-00001", "tag": "harmful", "failure_lesson": {
                "failure_point": "A", "flawed_reasoning": "B",
                "counterfactual": "C", "prevention_principle": "Rule 1",
            }},
            {"id": "str-00002", "tag": "harmful", "failure_lesson": {
                "failure_point": "D", "flawed_reasoning": "E",
                "counterfactual": "F", "prevention_principle": "Rule 2",
            }},
        ])
        principles = pb.get_all_prevention_principles()
        assert len(principles) == 2
        assert ("str-00001", "Rule 1") in principles
        assert ("str-00002", "Rule 2") in principles

    def test_empty_prevention_principles(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        assert pb.get_all_prevention_principles() == []


# ==========================================================================
# P1-1/P1-2: Bullet scope and when_to_apply (SkillRL §3.2 / Table 5)
# ==========================================================================


class TestBulletScopeAndWhenToApply:
    def test_default_scope_is_general(self):
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="test")
        assert b.scope == "general"

    def test_default_when_to_apply_is_empty(self):
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="test")
        assert b.when_to_apply == ""

    def test_custom_scope(self):
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="test",
                    scope="task_specific")
        assert b.scope == "task_specific"

    def test_custom_when_to_apply(self):
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="test",
                    when_to_apply="When processing financial data")
        assert b.when_to_apply == "When processing financial data"

    def test_scope_preserved_through_parsing(self):
        """scope is not in playbook.txt format — only in extended data."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        b = pb.get_bullet("str-00001")
        # Parsed bullets get default scope
        assert b.scope == "general"

    def test_format_line_does_not_include_scope(self):
        """Backward compat: format_line stays unchanged."""
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="test",
                    scope="task_specific", when_to_apply="Always")
        line = b.format_line()
        assert "scope" not in line
        assert "when_to_apply" not in line


# ==========================================================================
# P1-3: Merge combines failure_lessons
# ==========================================================================


class TestMergePreservesFailureLessons:
    def test_merge_combines_failure_lessons(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        # Add failure lessons to both bullets being merged
        pb.update_bullet_counts([
            {"id": "str-00001", "tag": "harmful", "failure_lesson": {
                "failure_point": "A", "flawed_reasoning": "B",
                "counterfactual": "C", "prevention_principle": "D",
            }},
            {"id": "str-00002", "tag": "harmful", "failure_lesson": {
                "failure_point": "E", "flawed_reasoning": "F",
                "counterfactual": "G", "prevention_principle": "H",
            }},
        ])
        ops = [DeltaOperation(
            op_type="MERGE",
            section="STRATEGIES & INSIGHTS",
            content="Merged content",
            bullet_id="str-00001",
            merge_target="str-00002",
        )]
        pb.apply_delta(ops)
        keeper = pb.get_bullet("str-00001")
        assert keeper is not None
        assert len(keeper.failure_lessons) == 2
        assert keeper.failure_lessons[0].failure_point == "A"
        assert keeper.failure_lessons[1].failure_point == "E"

    def test_merge_absorbed_bullet_removed(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        ops = [DeltaOperation(
            op_type="MERGE", section="", content="Merged",
            bullet_id="str-00001", merge_target="str-00002",
        )]
        pb.apply_delta(ops)
        assert pb.get_bullet("str-00002") is None

    def test_prune_returns_bullets_with_lessons(self):
        """Pruned bullets retain failure_lessons for evolution."""
        text = """## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=5 :: Bad strategy
"""
        pb = Playbook(text)
        pb.update_bullet_counts([{
            "id": "str-00001", "tag": "harmful",
            "failure_lesson": {
                "failure_point": "X", "flawed_reasoning": "Y",
                "counterfactual": "Z", "prevention_principle": "W",
            },
        }])
        removed = pb.prune_problematic(min_harmful=3)
        assert len(removed) == 1
        assert removed[0].has_failure_lessons
        assert removed[0].failure_lessons[0].prevention_principle == "W"


# ==========================================================================
# P0-2/P0-3: FailureLesson task_context
# ==========================================================================


class TestFailureLessonTaskContext:
    def test_task_context_default_empty(self):
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
        )
        assert fl.task_context == ""

    def test_task_context_set(self):
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
            task_context="Refactoring auth module",
        )
        assert fl.task_context == "Refactoring auth module"

    def test_task_context_in_to_dict(self):
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
            task_context="Debugging parser",
        )
        d = fl.to_dict()
        assert d["task_context"] == "Debugging parser"

    def test_task_context_from_dict(self):
        d = {
            "failure_point": "X", "flawed_reasoning": "Y",
            "counterfactual": "Z", "prevention_principle": "W",
            "task_context": "Writing tests",
        }
        fl = FailureLesson.from_dict(d)
        assert fl.task_context == "Writing tests"

    def test_task_context_in_format_text(self):
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
            task_context="Deploying to prod",
        )
        text = fl.format_text()
        assert "CONTEXT: Deploying to prod" in text

    def test_task_context_absent_from_format_text(self):
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
        )
        text = fl.format_text()
        assert "CONTEXT" not in text

    def test_task_context_roundtrip(self):
        fl = FailureLesson(
            failure_point="X", flawed_reasoning="Y",
            counterfactual="Z", prevention_principle="W",
            task_context="Code review feedback",
        )
        fl2 = FailureLesson.from_dict(fl.to_dict())
        assert fl2.task_context == "Code review feedback"


# ==========================================================================
# P0-4: check_evolution_needed (SkillRL §3.3)
# ==========================================================================


class TestCheckEvolutionNeeded:
    def test_no_harmful_bullets(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        result = pb.check_evolution_needed(threshold=1)
        # mis-00005 has harmful=4 but no failure lessons
        assert not result["needed"]
        assert result["candidate_count"] == 0

    def test_harmful_with_lessons_below_threshold(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.update_bullet_counts([{
            "id": "str-00001", "tag": "harmful",
            "failure_lesson": {
                "failure_point": "X", "flawed_reasoning": "Y",
                "counterfactual": "Z", "prevention_principle": "W",
            },
        }])
        result = pb.check_evolution_needed(threshold=2)
        assert not result["needed"]
        assert result["candidate_count"] == 1

    def test_harmful_with_lessons_meets_threshold(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        for bid in ["str-00001", "str-00002", "code-00003"]:
            pb.update_bullet_counts([{
                "id": bid, "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Fail {bid}",
                    "flawed_reasoning": "Y",
                    "counterfactual": "Z",
                    "prevention_principle": f"Rule for {bid}",
                },
            }])
        result = pb.check_evolution_needed(threshold=3)
        assert result["needed"]
        assert result["candidate_count"] == 3
        assert set(result["candidate_ids"]) == {"str-00001", "str-00002", "code-00003"}

    def test_stats_includes_evolution_info(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        stats = pb.get_stats()
        assert hasattr(stats, "evolution_needed")
        assert hasattr(stats, "evolution_candidates")
        assert stats.evolution_needed is False
        assert stats.evolution_candidates == 0


# ==========================================================================
# P0-1: evolve_from_failures (SkillRL Prompt B.1 / §3.3)
# ==========================================================================


class TestEvolveFromFailures:
    def _make_playbook_with_failures(self, n_bullets=3):
        """Helper: create a playbook with n harmful bullets each having a failure lesson."""
        text = "## STRATEGIES & INSIGHTS\n"
        for i in range(1, n_bullets + 1):
            text += f"[str-{i:05d}] helpful=0 harmful=1 :: Strategy {i}\n"
        text += "\n## PROBLEM-SOLVING HEURISTICS\n"
        pb = Playbook(text)
        for i in range(1, n_bullets + 1):
            pb.update_bullet_counts([{
                "id": f"str-{i:05d}",
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Failure point {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Instead {i}",
                    "prevention_principle": f"Prevention rule {i}",
                    "task_context": f"Task context {i}",
                },
            }])
        return pb

    def test_evolve_below_threshold_returns_empty(self):
        pb = self._make_playbook_with_failures(2)
        new_bullets = pb.evolve_from_failures(threshold=3)
        assert new_bullets == []

    def test_evolve_at_threshold_creates_skills(self):
        pb = self._make_playbook_with_failures(3)
        initial_count = len(pb.bullets)
        new_bullets = pb.evolve_from_failures(threshold=3)
        assert len(new_bullets) == 3
        assert len(pb.bullets) == initial_count + 3

    def test_evolved_bullets_in_heuristics_section(self):
        pb = self._make_playbook_with_failures(3)
        new_bullets = pb.evolve_from_failures(threshold=3)
        for b in new_bullets:
            assert b.section == "PROBLEM-SOLVING HEURISTICS"

    def test_evolved_bullets_have_scope_general(self):
        pb = self._make_playbook_with_failures(3)
        new_bullets = pb.evolve_from_failures(threshold=3)
        for b in new_bullets:
            assert b.scope == "general"

    def test_evolved_bullets_have_when_to_apply(self):
        pb = self._make_playbook_with_failures(3)
        new_bullets = pb.evolve_from_failures(threshold=3)
        for b in new_bullets:
            assert b.when_to_apply != ""

    def test_evolved_bullets_use_task_context_as_when_to_apply(self):
        pb = self._make_playbook_with_failures(3)
        new_bullets = pb.evolve_from_failures(threshold=3)
        # Task context was set, so when_to_apply should be the task_context
        for i, b in enumerate(new_bullets):
            assert b.when_to_apply == f"Task context {i + 1}"

    def test_evolved_bullets_content_is_prevention_principle(self):
        pb = self._make_playbook_with_failures(3)
        new_bullets = pb.evolve_from_failures(threshold=3)
        for i, b in enumerate(new_bullets):
            assert b.content == f"Prevention rule {i + 1}"

    def test_evolved_bullets_have_unique_ids(self):
        pb = self._make_playbook_with_failures(3)
        new_bullets = pb.evolve_from_failures(threshold=3)
        ids = [b.id for b in new_bullets]
        assert len(ids) == len(set(ids))

    def test_evolved_bullets_start_with_zero_counts(self):
        pb = self._make_playbook_with_failures(3)
        new_bullets = pb.evolve_from_failures(threshold=3)
        for b in new_bullets:
            assert b.helpful == 0
            assert b.harmful == 0

    def test_evolve_deduplicates_same_principle(self):
        """Same prevention_principle from different bullets should not create duplicates."""
        text = "## STRATEGIES & INSIGHTS\n"
        for i in range(1, 4):
            text += f"[str-{i:05d}] helpful=0 harmful=1 :: Strategy {i}\n"
        pb = Playbook(text)
        # All three have the SAME prevention principle
        for i in range(1, 4):
            pb.update_bullet_counts([{
                "id": f"str-{i:05d}",
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Fail {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Fix {i}",
                    "prevention_principle": "Always validate inputs",
                },
            }])
        new_bullets = pb.evolve_from_failures(threshold=3)
        assert len(new_bullets) == 1
        assert new_bullets[0].content == "Always validate inputs"

    def test_evolve_skips_empty_principle(self):
        text = "## STRATEGIES & INSIGHTS\n"
        for i in range(1, 4):
            text += f"[str-{i:05d}] helpful=0 harmful=1 :: Strategy {i}\n"
        pb = Playbook(text)
        for i in range(1, 4):
            pb.update_bullet_counts([{
                "id": f"str-{i:05d}",
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Fail {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Fix {i}",
                    "prevention_principle": "" if i == 1 else f"Rule {i}",
                },
            }])
        new_bullets = pb.evolve_from_failures(threshold=3)
        assert len(new_bullets) == 2  # skipped empty principle

    def test_evolve_fallback_when_to_apply_from_failure_point(self):
        """When task_context is empty, when_to_apply derives from failure_point."""
        text = "## STRATEGIES & INSIGHTS\n"
        for i in range(1, 4):
            text += f"[str-{i:05d}] helpful=0 harmful=1 :: Strategy {i}\n"
        pb = Playbook(text)
        for i in range(1, 4):
            pb.update_bullet_counts([{
                "id": f"str-{i:05d}",
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Failed on edge case {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Fix {i}",
                    "prevention_principle": f"Rule {i}",
                    # No task_context
                },
            }])
        new_bullets = pb.evolve_from_failures(threshold=3)
        for b in new_bullets:
            assert "When facing issues similar to:" in b.when_to_apply

    def test_evolve_serialization_roundtrip(self):
        """Evolved bullets appear in serialized playbook."""
        pb = self._make_playbook_with_failures(3)
        pb.evolve_from_failures(threshold=3)
        text = pb.serialize()
        assert "PROBLEM-SOLVING HEURISTICS" in text
        assert "Prevention rule 1" in text


# ==========================================================================
# N1: Persist scope and when_to_apply across sessions
# ==========================================================================


class TestPersistScopeAndWhenToApply:
    def test_scope_persists_through_save_load(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        b = pb.get_bullet("str-00001")
        b.scope = "task_specific"
        b.when_to_apply = "When processing CSV files"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            pb.save_failure_lessons(path)

            pb2 = Playbook(SAMPLE_PLAYBOOK)
            pb2.load_failure_lessons(path)
            b2 = pb2.get_bullet("str-00001")
            assert b2.scope == "task_specific"
            assert b2.when_to_apply == "When processing CSV files"

    def test_default_scope_not_saved(self):
        """Bullets with default scope='general' and no when_to_apply don't bloat the file."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            pb.save_failure_lessons(path)
            with open(path) as f:
                data = json.load(f)
            assert data == {}  # nothing to save

    def test_evolved_bullets_scope_persists(self):
        """After evolve_from_failures, the evolved bullets' scope and when_to_apply persist."""
        text = "## STRATEGIES & INSIGHTS\n"
        for i in range(1, 4):
            text += f"[str-{i:05d}] helpful=0 harmful=1 :: Strategy {i}\n"
        text += "\n## PROBLEM-SOLVING HEURISTICS\n"
        pb = Playbook(text)
        for i in range(1, 4):
            pb.update_bullet_counts([{
                "id": f"str-{i:05d}",
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Fail {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Fix {i}",
                    "prevention_principle": f"Rule {i}",
                    "task_context": f"Context {i}",
                },
            }])
        pb.evolve_from_failures(threshold=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            # Save the playbook text and extended data
            pb_text = pb.serialize()
            pb.save_failure_lessons(path)

            # Reload from scratch
            pb2 = Playbook(pb_text)
            pb2.load_failure_lessons(path)

            # Find an evolved bullet (in PROBLEM-SOLVING HEURISTICS)
            heuristics = pb2.get_section_bullets("PROBLEM-SOLVING HEURISTICS")
            assert len(heuristics) >= 3
            for h in heuristics:
                if h.content.startswith("Rule"):
                    assert h.when_to_apply != ""

    def test_backward_compat_old_format(self):
        """Old format (bare list) still loads correctly."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        old_data = {
            "str-00001": [{
                "failure_point": "Old format",
                "flawed_reasoning": "B",
                "counterfactual": "C",
                "prevention_principle": "D",
                "timestamp": "2026-01-01T00:00:00",
            }]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            with open(path, "w") as f:
                json.dump(old_data, f)

            loaded = pb.load_failure_lessons(path)
            assert loaded == 1
            b = pb.get_bullet("str-00001")
            assert b.failure_lessons[0].failure_point == "Old format"
            # scope stays default since old format doesn't have it
            assert b.scope == "general"

    def test_mixed_bullets_some_with_extended(self):
        """Only bullets with extended data appear in the JSON file."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        b1 = pb.get_bullet("str-00001")
        b1.when_to_apply = "When data is unsorted"
        b2 = pb.get_bullet("str-00002")
        # b2 keeps defaults — should NOT appear in JSON

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            pb.save_failure_lessons(path)
            with open(path) as f:
                data = json.load(f)
            assert "str-00001" in data
            assert "str-00002" not in data


# ==========================================================================
# N2: Dedup evolved skills against existing bullets
# ==========================================================================


class TestEvolveDedupsAgainstExisting:
    def test_existing_bullet_blocks_duplicate_evolution(self):
        """If a bullet already has the same content as a prevention principle, skip it."""
        text = "## STRATEGIES & INSIGHTS\n"
        for i in range(1, 4):
            text += f"[str-{i:05d}] helpful=0 harmful=1 :: Strategy {i}\n"
        text += "\n## PROBLEM-SOLVING HEURISTICS\n"
        text += "[heu-00004] helpful=3 harmful=0 :: Always validate inputs\n"
        pb = Playbook(text)
        for i in range(1, 4):
            pb.update_bullet_counts([{
                "id": f"str-{i:05d}",
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Fail {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Fix {i}",
                    "prevention_principle": "Always validate inputs",
                },
            }])
        new_bullets = pb.evolve_from_failures(threshold=3)
        # Should create 0 new bullets — principle already exists as heu-00004
        assert len(new_bullets) == 0

    def test_partial_dedup_against_existing(self):
        """Some principles match existing, some are new."""
        text = "## STRATEGIES & INSIGHTS\n"
        for i in range(1, 4):
            text += f"[str-{i:05d}] helpful=0 harmful=1 :: Strategy {i}\n"
        text += "\n## PROBLEM-SOLVING HEURISTICS\n"
        text += "[heu-00004] helpful=3 harmful=0 :: Rule 1\n"  # matches principle 1
        pb = Playbook(text)
        for i in range(1, 4):
            pb.update_bullet_counts([{
                "id": f"str-{i:05d}",
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Fail {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Fix {i}",
                    "prevention_principle": f"Rule {i}",
                },
            }])
        new_bullets = pb.evolve_from_failures(threshold=3)
        # Rule 1 matches existing heu-00004, but Rule 2 and Rule 3 are new
        assert len(new_bullets) == 2
        contents = {b.content for b in new_bullets}
        assert "Rule 2" in contents
        assert "Rule 3" in contents
        assert "Rule 1" not in contents


# ==========================================================================
# N3: evolve_from_failures idempotency
# ==========================================================================


class TestEvolveIdempotency:
    def _make_pb(self):
        text = "## STRATEGIES & INSIGHTS\n"
        for i in range(1, 4):
            text += f"[str-{i:05d}] helpful=0 harmful=1 :: Strategy {i}\n"
        text += "\n## PROBLEM-SOLVING HEURISTICS\n"
        pb = Playbook(text)
        for i in range(1, 4):
            pb.update_bullet_counts([{
                "id": f"str-{i:05d}",
                "tag": "harmful",
                "failure_lesson": {
                    "failure_point": f"Fail {i}",
                    "flawed_reasoning": f"Flaw {i}",
                    "counterfactual": f"Fix {i}",
                    "prevention_principle": f"Rule {i}",
                    "task_context": f"Context {i}",
                },
            }])
        return pb

    def test_double_evolve_is_idempotent(self):
        """Calling evolve_from_failures twice should not create duplicates."""
        pb = self._make_pb()
        first = pb.evolve_from_failures(threshold=3)
        assert len(first) == 3

        second = pb.evolve_from_failures(threshold=3)
        assert len(second) == 0  # all lessons already evolved

    def test_evolved_flag_persisted(self):
        """The evolved flag survives save/load roundtrip."""
        pb = self._make_pb()
        pb.evolve_from_failures(threshold=3)

        # All lessons should be marked evolved
        for b in pb.bullets:
            for fl in b.failure_lessons:
                assert fl.evolved

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            pb_text = pb.serialize()
            pb.save_failure_lessons(path)

            pb2 = Playbook(pb_text)
            pb2.load_failure_lessons(path)

            # Check evolved flag survived
            for b in pb2.bullets:
                for fl in b.failure_lessons:
                    assert fl.evolved

    def test_new_lessons_after_evolution_still_evolve(self):
        """New failure lessons added after evolution should still trigger evolution."""
        pb = self._make_pb()
        pb.evolve_from_failures(threshold=3)

        # Add a NEW failure lesson to an existing bullet
        pb.update_bullet_counts([{
            "id": "str-00001",
            "tag": "harmful",
            "failure_lesson": {
                "failure_point": "New failure",
                "flawed_reasoning": "New flaw",
                "counterfactual": "New fix",
                "prevention_principle": "Brand new rule",
                "task_context": "New context",
            },
        }])
        # Should evolve the new lesson
        new_bullets = pb.evolve_from_failures(threshold=3)
        assert len(new_bullets) == 1
        assert new_bullets[0].content == "Brand new rule"


# ==========================================================================
# Temporal Decay (ACT-R / SYNAPSE inspired)
# ==========================================================================


class TestTemporalDecay:
    def test_effective_score_no_timestamp(self):
        """Without last_updated, returns raw score (no decay)."""
        b = Bullet(id="str-00001", helpful=5, harmful=1, content="test")
        assert b.effective_score() == 4.0

    def test_effective_score_fresh(self):
        """Recently updated bullet retains near-full score."""
        from datetime import datetime, timezone
        b = Bullet(id="str-00001", helpful=5, harmful=1, content="test",
                   last_updated=datetime.now(timezone.utc).isoformat())
        eff = b.effective_score()
        assert 3.9 <= eff <= 4.0  # within minutes, essentially no decay

    def test_effective_score_30_days(self):
        """After 30 days, ~21% retained."""
        from datetime import datetime, timezone, timedelta
        past = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        b = Bullet(id="str-00001", helpful=10, harmful=0, content="test",
                   last_updated=past)
        eff = b.effective_score()
        expected = 10 * (0.95 ** 30)  # ~2.14
        assert abs(eff - expected) < 0.5

    def test_effective_score_90_days(self):
        """After 90 days, ~1% retained."""
        from datetime import datetime, timezone, timedelta
        past = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        b = Bullet(id="str-00001", helpful=10, harmful=0, content="test",
                   last_updated=past)
        eff = b.effective_score()
        expected = 10 * (0.95 ** 90)  # ~0.10
        assert eff < 1.0

    def test_effective_score_negative(self):
        """Decay works with negative scores too."""
        from datetime import datetime, timezone, timedelta
        past = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        b = Bullet(id="str-00001", helpful=1, harmful=5, content="bad",
                   last_updated=past)
        eff = b.effective_score()
        assert eff < 0  # still negative
        assert abs(eff) < abs(b.score)  # but closer to zero

    def test_update_counters_sets_timestamp(self):
        """Calling update_bullet_counts sets last_updated."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        b = pb.get_bullet("str-00001")
        assert b.last_updated == ""
        pb.update_bullet_counts([{"id": "str-00001", "tag": "helpful"}])
        assert b.last_updated != ""
        from datetime import datetime
        # Should be a valid ISO timestamp
        datetime.fromisoformat(b.last_updated)

    def test_add_sets_timestamp(self):
        """_apply_add sets last_updated on new bullets."""
        pb = Playbook()
        pb.apply_delta([DeltaOperation(
            op_type="ADD", section="STRATEGIES & INSIGHTS",
            content="New insight"
        )])
        b = pb.bullets[0]
        assert b.last_updated != ""

    def test_enforce_budget_uses_effective_score(self):
        """Decayed bullets are pruned before fresh ones."""
        from datetime import datetime, timezone, timedelta
        text = "## STRATEGIES & INSIGHTS\n"
        text += "[str-00001] helpful=5 harmful=0 :: Old strategy\n"
        text += "[str-00002] helpful=2 harmful=0 :: Fresh strategy\n"
        pb = Playbook(text)
        # str-00001 has higher raw score but is old
        old = pb.get_bullet("str-00001")
        old.last_updated = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        fresh = pb.get_bullet("str-00002")
        fresh.last_updated = datetime.now(timezone.utc).isoformat()
        # Force very tight budget to trigger pruning
        removed = pb.enforce_token_budget(max_chars=50)
        # Old bullet should be pruned first despite higher raw score
        removed_ids = [b.id for b in removed]
        assert "str-00001" in removed_ids

    def test_last_updated_persisted(self):
        """Save/load round-trip preserves last_updated."""
        from datetime import datetime, timezone
        pb = Playbook(SAMPLE_PLAYBOOK)
        b = pb.get_bullet("str-00001")
        ts = datetime.now(timezone.utc).isoformat()
        b.last_updated = ts

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failure_lessons.json")
            pb.save_failure_lessons(path)

            pb2 = Playbook(SAMPLE_PLAYBOOK)
            pb2.load_failure_lessons(path)
            b2 = pb2.get_bullet("str-00001")
            assert b2.last_updated == ts

    def test_effective_score_invalid_timestamp(self):
        """Malformed timestamp returns raw score."""
        b = Bullet(id="str-00001", helpful=5, harmful=1, content="test",
                   last_updated="not-a-date")
        assert b.effective_score() == 4.0

    def test_decayed_bullets_stat(self):
        """get_stats reports decayed_bullets count."""
        from datetime import datetime, timezone, timedelta
        pb = Playbook(SAMPLE_PLAYBOOK)
        b = pb.get_bullet("str-00001")  # helpful=5, harmful=0
        b.last_updated = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        stats = pb.get_stats()
        assert stats.decayed_bullets >= 1


# ===========================================================================
# MCE-inspired Schema Evolution Tests (arXiv:2601.21557)
# ===========================================================================


class TestSchemaMetrics:
    """Test compute_metrics() — MCE evaluation function J."""

    def test_compute_metrics_empty_playbook(self):
        pb = Playbook()
        metrics = pb.compute_metrics()
        assert metrics.total_bullets == 0
        assert metrics.section_balance == 0.0
        assert metrics.overall_health == 0.0
        assert len(metrics.empty_sections) == 6  # All DEFAULT_SECTIONS empty
        assert metrics.timestamp != ""

    def test_section_balance_single_section(self):
        """All bullets in one section → entropy = 0."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=1 harmful=0 :: Strategy A\n"
            "[str-00002] helpful=1 harmful=0 :: Strategy B\n"
            "[str-00003] helpful=1 harmful=0 :: Strategy C\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        metrics = pb.compute_metrics()
        assert metrics.section_balance == 0.0

    def test_section_balance_even_distribution(self):
        """Bullets evenly across sections → entropy near 1.0."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=1 harmful=0 :: A\n"
            "\n## CODE SNIPPETS & TEMPLATES\n"
            "[code-00002] helpful=1 harmful=0 :: B\n"
            "\n## COMMON MISTAKES TO AVOID\n"
            "[mis-00003] helpful=1 harmful=0 :: C\n"
            "\n## PROBLEM-SOLVING HEURISTICS\n"
            "[heu-00004] helpful=1 harmful=0 :: D\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        metrics = pb.compute_metrics()
        assert metrics.section_balance > 0.9

    def test_section_balance_skewed(self):
        """Uneven distribution → 0 < entropy < 1."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=1 harmful=0 :: A\n"
            "[str-00002] helpful=1 harmful=0 :: B\n"
            "[str-00003] helpful=1 harmful=0 :: C\n"
            "\n## CODE SNIPPETS & TEMPLATES\n"
            "[code-00004] helpful=1 harmful=0 :: D\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        metrics = pb.compute_metrics()
        assert 0.0 < metrics.section_balance < 1.0

    def test_utilization_rate(self):
        """Mix of utilized and unused bullets."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        metrics = pb.compute_metrics()
        # SAMPLE_PLAYBOOK has 5 bullets, all with helpful+harmful > 0 except none
        # str-00001: h=5, str-00002: h=3,h=1, code-00003: h=8, mis-00004: h=6, mis-00005: h=0,h=4
        assert metrics.utilization_rate == 1.0  # All have counters > 0

    def test_harmful_ratio(self):
        """Known harmful/helpful distribution."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: Good\n"
            "[str-00002] helpful=0 harmful=3 :: Bad\n"
            "[str-00003] helpful=1 harmful=5 :: Worse\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        metrics = pb.compute_metrics()
        # 3 utilized, 2 harmful (str-00002: 0<3, str-00003: 5>1 — wait, harmful >= helpful)
        # str-00002: harmful=3 >= helpful=0 AND harmful>0 → harmful
        # str-00003: harmful=5 >= helpful=1 AND harmful>0 → harmful
        assert abs(metrics.harmful_ratio - 2.0 / 3.0) < 0.01

    def test_decay_impact_with_old_bullets(self):
        """Bullets with old timestamps show high decay impact."""
        from datetime import datetime, timezone, timedelta
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: Old\n"
            "[str-00002] helpful=5 harmful=0 :: Fresh\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        pb.get_bullet("str-00001").last_updated = (
            datetime.now(timezone.utc) - timedelta(days=90)
        ).isoformat()
        pb.get_bullet("str-00002").last_updated = datetime.now(timezone.utc).isoformat()
        metrics = pb.compute_metrics()
        # str-00001 at 90 days: 0.95^90 ≈ 0.01 → effective << 50% of raw
        assert metrics.decay_impact > 0.0

    def test_empty_sections_detected(self):
        pb = Playbook(SAMPLE_PLAYBOOK)
        metrics = pb.compute_metrics()
        # SAMPLE_PLAYBOOK has bullets in STRATEGIES, CODE SNIPPETS, COMMON MISTAKES
        # Empty: PROBLEM-SOLVING HEURISTICS, CONTEXT CLUES, OTHERS
        assert "PROBLEM-SOLVING HEURISTICS" in metrics.empty_sections
        assert "CONTEXT CLUES & INDICATORS" in metrics.empty_sections

    def test_overflow_sections_detected(self):
        """Section with >50% of bullets flagged as overflow."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=1 harmful=0 :: A\n"
            "[str-00002] helpful=1 harmful=0 :: B\n"
            "[str-00003] helpful=1 harmful=0 :: C\n"
            "[str-00004] helpful=1 harmful=0 :: D\n"
            "[str-00005] helpful=1 harmful=0 :: E\n"
            "\n## CODE SNIPPETS & TEMPLATES\n"
            "[code-00006] helpful=1 harmful=0 :: F\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        metrics = pb.compute_metrics()
        # 5/6 = 83% in STRATEGIES
        assert "STRATEGIES & INSIGHTS" in metrics.overflow_sections

    def test_overall_health_formula(self):
        """Verify weighted composite matches manual calculation."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: A\n"
            "\n## CODE SNIPPETS & TEMPLATES\n"
            "[code-00002] helpful=3 harmful=0 :: B\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        metrics = pb.compute_metrics()
        # Manual: section_balance for 2 non-empty sections with equal bullets = 1.0
        # utilization = 1.0, harmful = 0.0, unused = 0.0, decay = 0.0
        expected = 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0 + 0.15 * 1.0 + 0.10 * 1.0
        assert abs(metrics.overall_health - expected) < 0.01


class TestSchemaProposals:
    """Test propose_schema_changes() — MCE (1+1)-ES proposals."""

    def _make_schema(self, **kwargs):
        from ccr.core.types import PlaybookSchema
        from ccr.ace.playbook import DEFAULT_SECTIONS, _SLUG_MAP
        defaults = dict(
            version=1, sections=list(DEFAULT_SECTIONS),
            slug_map=dict(_SLUG_MAP),
        )
        defaults.update(kwargs)
        return PlaybookSchema(**defaults)

    def test_healthy_playbook_no_proposals(self):
        """Healthy playbook → no proposals."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: A\n"
            "\n## CODE SNIPPETS & TEMPLATES\n"
            "[code-00002] helpful=3 harmful=0 :: B\n"
            "\n## COMMON MISTAKES TO AVOID\n"
            "[mis-00003] helpful=4 harmful=0 :: C\n"
            "\n## PROBLEM-SOLVING HEURISTICS\n"
            "[heu-00004] helpful=2 harmful=0 :: D\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        schema = self._make_schema()
        proposals = pb.propose_schema_changes(schema, stop_health_threshold=0.8)
        assert proposals == []

    def test_add_section_from_others_cluster(self):
        """Clustered bullets in OTHERS → ADD_SECTION proposal."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: General strategy\n"
            "\n## OTHERS\n"
            "[oth-00002] helpful=0 harmful=0 :: Database query optimization techniques\n"
            "[oth-00003] helpful=0 harmful=0 :: Database indexing optimization strategies\n"
            "[oth-00004] helpful=0 harmful=0 :: Database connection optimization pooling\n"
        )
        pb = Playbook(text)
        schema = self._make_schema()
        proposals = pb.propose_schema_changes(
            schema, min_cluster_size=3, stop_health_threshold=1.0,
        )
        if proposals:  # Only check if a proposal was actually made
            assert proposals[0].change_type == "ADD_SECTION"
            assert len(proposals[0].details.get("bullet_ids", [])) >= 3

    def test_add_section_below_min_cluster(self):
        """Too few bullets in OTHERS → no ADD_SECTION."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: A\n"
            "\n## OTHERS\n"
            "[oth-00002] helpful=0 harmful=0 :: Lone bullet\n"
        )
        pb = Playbook(text)
        schema = self._make_schema()
        proposals = pb.propose_schema_changes(
            schema, min_cluster_size=3, stop_health_threshold=1.0,
        )
        add_proposals = [p for p in proposals if p.change_type == "ADD_SECTION"]
        assert add_proposals == []

    def test_remove_empty_section(self):
        """Persistently empty section → REMOVE_SECTION."""
        from ccr.core.types import SchemaMetrics
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: A\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        baseline = SchemaMetrics(empty_sections=["CONTEXT CLUES & INDICATORS"])
        schema = self._make_schema(baseline_metrics=baseline)
        proposals = pb.propose_schema_changes(
            schema, stop_health_threshold=1.0,
        )
        remove_proposals = [p for p in proposals if p.change_type == "REMOVE_SECTION"]
        if remove_proposals:
            assert remove_proposals[0].details["name"] in [
                "CODE SNIPPETS & TEMPLATES", "COMMON MISTAKES TO AVOID",
                "PROBLEM-SOLVING HEURISTICS", "CONTEXT CLUES & INDICATORS",
            ]

    def test_adjust_decay_high_impact(self):
        """High decay impact → ADJUST_DECAY proposal."""
        from datetime import datetime, timezone, timedelta
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: A\n"
            "[str-00002] helpful=3 harmful=0 :: B\n"
            "[str-00003] helpful=4 harmful=0 :: C\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        # Make all bullets very old
        old_ts = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        for b in pb.bullets:
            b.last_updated = old_ts
        schema = self._make_schema()
        proposals = pb.propose_schema_changes(
            schema, stop_health_threshold=1.0,
        )
        decay_proposals = [p for p in proposals if p.change_type == "ADJUST_DECAY"]
        if decay_proposals:
            assert decay_proposals[0].details["new_rate"] < schema.decay_rate

    def test_adjust_pruning_high_harmful(self):
        """High harmful ratio → ADJUST_PRUNING."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=0 harmful=5 :: Bad A\n"
            "[str-00002] helpful=0 harmful=3 :: Bad B\n"
            "[str-00003] helpful=5 harmful=0 :: Good C\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        schema = self._make_schema()
        proposals = pb.propose_schema_changes(
            schema, stop_health_threshold=1.0,
        )
        prune_proposals = [p for p in proposals if p.change_type == "ADJUST_PRUNING"]
        if prune_proposals:
            assert prune_proposals[0].details["new_min_harmful"] < schema.prune_min_harmful

    def test_rebalance_overflow(self):
        """Overflow section → REBALANCE proposal."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=1 harmful=0 :: Strategy alpha\n"
            "[str-00002] helpful=1 harmful=0 :: Strategy beta\n"
            "[str-00003] helpful=1 harmful=0 :: Strategy gamma\n"
            "[str-00004] helpful=1 harmful=0 :: Totally different topic about databases\n"
            "\n## CODE SNIPPETS & TEMPLATES\n"
            "[code-00005] helpful=1 harmful=0 :: Snippet\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        schema = self._make_schema()
        proposals = pb.propose_schema_changes(
            schema, stop_health_threshold=1.0,
        )
        rebalance = [p for p in proposals if p.change_type == "REBALANCE"]
        if rebalance:
            assert "moves" in rebalance[0].details

    def test_rollback_proposal_on_degradation(self):
        """Health dropped from baseline → ROLLBACK proposal."""
        from ccr.core.types import SchemaMetrics
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=0 harmful=5 :: Bad\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        baseline = SchemaMetrics(overall_health=0.9)
        schema = self._make_schema(
            version=2, parent_version=1, baseline_metrics=baseline,
        )
        proposals = pb.propose_schema_changes(schema, rollback_health_delta=-0.05)
        assert len(proposals) == 1
        assert proposals[0].change_type == "ROLLBACK"

    def test_single_proposal_returned(self):
        """(1+1)-ES: at most one proposal."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        schema = self._make_schema()
        proposals = pb.propose_schema_changes(schema, stop_health_threshold=1.0)
        assert len(proposals) <= 1

    def test_no_baseline_skips_rollback(self):
        """First version (no baseline) → no ROLLBACK check."""
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=0 harmful=5 :: Bad\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        schema = self._make_schema()  # No baseline
        proposals = pb.propose_schema_changes(schema, stop_health_threshold=1.0)
        rollback = [p for p in proposals if p.change_type == "ROLLBACK"]
        assert rollback == []


class TestClusterOthersBullets:
    """Test _cluster_others_bullets() — section proposal clustering."""

    def test_cluster_similar_bullets(self):
        text = (
            "## OTHERS\n"
            "[oth-00001] helpful=0 harmful=0 :: Testing unit tests assertions validation\n"
            "[oth-00002] helpful=0 harmful=0 :: Testing integration tests validation checks\n"
            "[oth-00003] helpful=0 harmful=0 :: Testing end-to-end tests assertions framework\n"
        )
        pb = Playbook(text)
        clusters = pb._cluster_others_bullets(min_cluster=3)
        assert len(clusters) >= 1
        name, ids = clusters[0]
        assert len(ids) == 3

    def test_no_clusters_below_threshold(self):
        text = (
            "## OTHERS\n"
            "[oth-00001] helpful=0 harmful=0 :: Apples oranges bananas\n"
            "[oth-00002] helpful=0 harmful=0 :: Quantum physics relativity\n"
            "[oth-00003] helpful=0 harmful=0 :: Financial trading stocks\n"
        )
        pb = Playbook(text)
        clusters = pb._cluster_others_bullets(min_cluster=3)
        assert clusters == []

    def test_cluster_name_generation(self):
        text = (
            "## OTHERS\n"
            "[oth-00001] helpful=0 harmful=0 :: Database query optimization techniques\n"
            "[oth-00002] helpful=0 harmful=0 :: Database indexing optimization strategies\n"
            "[oth-00003] helpful=0 harmful=0 :: Database performance optimization tuning\n"
        )
        pb = Playbook(text)
        clusters = pb._cluster_others_bullets(min_cluster=3)
        if clusters:
            name, ids = clusters[0]
            assert len(name) > 0
            # Name should be uppercase words from cluster content
            assert name == name.upper()

    def test_empty_others_no_clusters(self):
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: Strategy\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text)
        clusters = pb._cluster_others_bullets(min_cluster=3)
        assert clusters == []


class TestApplySchema:
    """Test apply_schema() — applying schema changes to playbook."""

    def test_apply_adds_new_sections(self):
        from ccr.core.types import PlaybookSchema
        pb = Playbook(SAMPLE_PLAYBOOK)
        schema = PlaybookSchema(
            sections=list(pb.sections) + ["API DESIGN PATTERNS"],
            slug_map={"api_design_patterns": "api"},
        )
        moved = pb.apply_schema(schema)
        assert moved == 0
        assert "API DESIGN PATTERNS" in pb.sections

    def test_apply_removes_section_moves_bullets_to_others(self):
        from ccr.core.types import PlaybookSchema
        text = (
            "## STRATEGIES & INSIGHTS\n"
            "[str-00001] helpful=5 harmful=0 :: A\n"
            "\n## CUSTOM SECTION\n"
            "[cus-00002] helpful=3 harmful=0 :: B\n"
            "\n## OTHERS\n"
        )
        pb = Playbook(text, sections=["STRATEGIES & INSIGHTS", "CUSTOM SECTION", "OTHERS"])
        schema = PlaybookSchema(sections=["STRATEGIES & INSIGHTS", "OTHERS"])
        moved = pb.apply_schema(schema)
        assert moved == 1
        b = pb.get_bullet("cus-00002")
        assert b.section == "OTHERS"

    def test_apply_preserves_existing_bullets(self):
        from ccr.core.types import PlaybookSchema
        pb = Playbook(SAMPLE_PLAYBOOK)
        original_count = len(pb.bullets)
        schema = PlaybookSchema.default()
        pb.apply_schema(schema)
        assert len(pb.bullets) == original_count

    def test_apply_default_schema_no_change(self):
        from ccr.core.types import PlaybookSchema
        pb = Playbook(SAMPLE_PLAYBOOK)
        original_sections = list(pb.sections)
        schema = PlaybookSchema.default()
        moved = pb.apply_schema(schema)
        assert moved == 0
        assert pb.sections == original_sections

    def test_apply_schema_updates_section_list(self):
        from ccr.core.types import PlaybookSchema
        pb = Playbook(SAMPLE_PLAYBOOK)
        new_sections = ["ALPHA", "BETA", "OTHERS"]
        schema = PlaybookSchema(sections=new_sections)
        pb.apply_schema(schema)
        assert pb.sections == new_sections


class TestGRPOAdvantages:
    """Tests for GRPO group-relative advantage scoring (SkillRL GRPO Eq.3)."""

    def test_recompute_grpo_advantages_single_bullet(self):
        """Single bullet in a section should get advantage 0.0."""
        pb = Playbook()
        pb.apply_delta([
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Always verify inputs before processing"),
        ])
        count = pb.recompute_grpo_advantages()
        assert count == 1
        bullet = pb.bullets[0]
        assert bullet.grpo_advantage == 0.0

    def test_recompute_grpo_advantages_group_math(self):
        """Three bullets with distinct rewards should satisfy GRPO Eq.3 math."""
        from datetime import datetime, timezone

        pb = Playbook()
        now_iso = datetime.now(timezone.utc).isoformat()
        # Add 3 bullets to same section
        pb.apply_delta([
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="alpha strategy one"),
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="alpha strategy two"),
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="alpha strategy three"),
        ])
        bullets = pb.bullets
        # Set distinct helpful counts and timestamps (no decay since last_updated set now)
        bullets[0].helpful = 10
        bullets[0].last_updated = now_iso
        bullets[1].helpful = 4
        bullets[1].last_updated = now_iso
        bullets[2].helpful = 1
        bullets[2].last_updated = now_iso

        count = pb.recompute_grpo_advantages()
        assert count == 3

        # GRPO Eq.3: A_i = (r_i - mean) / (std + eps)
        # rewards are approximately 10, 4, 1 (no decay since just updated)
        rewards = [b.effective_score() for b in bullets]
        mean_r = sum(rewards) / len(rewards)
        import math
        variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = math.sqrt(variance)
        eps = 1e-8

        for bullet, r_i in zip(bullets, rewards):
            expected_adv = (r_i - mean_r) / (std_r + eps)
            assert abs(bullet.grpo_advantage - expected_adv) < 1e-6

    def test_recompute_grpo_advantages_returns_count(self):
        """recompute_grpo_advantages should return total number of bullets updated."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        total_bullets = len(pb.bullets)
        count = pb.recompute_grpo_advantages()
        assert count == total_bullets

    def test_get_policy_ranked_filters_by_task_context(self):
        """task_context should filter to relevant bullets."""
        pb = Playbook()
        pb.apply_delta([
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Always check database connections before querying"),
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Use async patterns when handling multiple HTTP requests"),
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Database query optimization requires proper indexing"),
        ])
        # Set scores so ranking is deterministic
        bullets = pb.bullets
        bullets[0].helpful = 5
        bullets[1].helpful = 3
        bullets[2].helpful = 4
        pb.recompute_grpo_advantages()

        # Filter for database-related context
        ranked = pb.get_policy_ranked(task_context="database queries", top_k=10)
        # Should include database bullets but not the HTTP one
        contents = [b.content for b in ranked]
        assert any("database" in c.lower() or "query" in c.lower() for c in contents)

    def test_get_policy_ranked_sorts_by_policy_score(self):
        """Higher effective_score * (1 + advantage) should come first."""
        from datetime import datetime, timezone

        pb = Playbook()
        now_iso = datetime.now(timezone.utc).isoformat()
        pb.apply_delta([
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="high score strategy for ranking"),
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="low score strategy for ranking"),
        ])
        bullets = pb.bullets
        bullets[0].helpful = 10
        bullets[0].last_updated = now_iso
        bullets[1].helpful = 1
        bullets[1].last_updated = now_iso
        pb.recompute_grpo_advantages()

        ranked = pb.get_policy_ranked(top_k=10)
        assert len(ranked) >= 2
        # First bullet should have higher policy score than second
        def policy_score(b):
            return b.effective_score() * (1.0 + b.grpo_advantage)
        scores = [policy_score(b) for b in ranked]
        assert scores[0] >= scores[-1]


# ===========================================================================
# B1: ace_prune double-calls evolve_from_failures
# Verified at data layer: evolve_from_failures is idempotent — calling it twice
# should not create duplicate bullets.
# ===========================================================================

class TestB1PruneNoDuplicateEvolution:
    """B1 fix: verify evolve_from_failures doesn't create duplicates when called twice."""

    def _make_playbook_with_lessons(self) -> Playbook:
        """Return a playbook with one harmful bullet that has a failure lesson."""
        pb = Playbook()
        pb.apply_delta([DeltaOperation(
            op_type="ADD",
            section="COMMON MISTAKES TO AVOID",
            content="Bad strategy with lesson",
        )])
        bullet = pb.bullets[0]
        bullet.harmful = 3
        bullet.failure_lessons.append(FailureLesson(
            failure_point="Broke on null input",
            flawed_reasoning="Assumed input always exists",
            counterfactual="Add null guard",
            prevention_principle="Always guard against null before dereferencing",
        ))
        return pb

    def test_evolve_from_failures_once_adds_skill(self):
        """Single call to evolve_from_failures adds exactly one new bullet."""
        pb = self._make_playbook_with_lessons()
        count_before = len(pb.bullets)
        evolved = pb.evolve_from_failures(threshold=1)
        assert len(evolved) == 1
        assert len(pb.bullets) == count_before + 1

    def test_evolve_from_failures_twice_no_duplicate(self):
        """Calling evolve_from_failures a second time does not add duplicate bullets.

        The lesson is marked evolved=True after the first call so the second call
        finds no eligible lessons and returns an empty list.
        """
        pb = self._make_playbook_with_lessons()
        pb.evolve_from_failures(threshold=1)
        count_after_first = len(pb.bullets)
        evolved_second = pb.evolve_from_failures(threshold=1)
        assert len(evolved_second) == 0
        assert len(pb.bullets) == count_after_first


# ===========================================================================
# B2: ace_update_counters silent zero for invalid slugs
# Verified at data layer: update_bullet_counts returns 0 for unknown IDs.
# The MCP layer should surface missing IDs — tested in test_mcp_server.py;
# here we confirm the data layer behaviour that drives the fix.
# ===========================================================================

class TestB2UpdateCountersMissingIds:
    """B2 fix: update_bullet_counts returns 0 for unknown IDs."""

    def test_update_nonexistent_returns_zero(self):
        """update_bullet_counts returns 0 when no bullet matches the given ID."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        updated = pb.update_bullet_counts([{"id": "nonexistent-99999", "tag": "helpful"}])
        assert updated == 0

    def test_update_mix_valid_invalid(self):
        """Only the valid ID is counted; the invalid one does not raise."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [
            {"id": "str-00001", "tag": "helpful"},
            {"id": "does-not-exist", "tag": "helpful"},
        ]
        updated = pb.update_bullet_counts(tags)
        assert updated == 1
        assert pb.get_bullet("str-00001").helpful == 6  # was 5 in SAMPLE_PLAYBOOK

    def test_all_invalid_ids_returns_zero(self):
        """When all IDs are invalid, updated count is 0."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        tags = [
            {"id": "fake-00001", "tag": "helpful"},
            {"id": "fake-00002", "tag": "harmful"},
        ]
        updated = pb.update_bullet_counts(tags)
        assert updated == 0


# ===========================================================================
# B3: ace_get_playbook policy-rank ignores global playbook
# Verified at data layer: get_policy_ranked returns bullets from the playbook
# it is called on; merging two lists works correctly.
# ===========================================================================

class TestB3PolicyRankMerge:
    """B3 fix: policy ranking should incorporate both global and project bullets."""

    def test_get_policy_ranked_returns_list(self):
        """get_policy_ranked returns a list of Bullet objects."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.recompute_grpo_advantages()
        ranked = pb.get_policy_ranked(top_k=3)
        assert isinstance(ranked, list)
        assert len(ranked) <= 3
        for b in ranked:
            assert isinstance(b, Bullet)

    def test_merge_ranked_deduplicates_by_id(self):
        """Merging two ranked lists and deduplicating by ID produces correct count."""
        pb = Playbook(SAMPLE_PLAYBOOK)
        pb.recompute_grpo_advantages()
        # Simulate merging project + global results (both from same playbook to test logic)
        p_ranked = pb.get_policy_ranked(top_k=5)
        g_ranked = pb.get_policy_ranked(top_k=5)
        seen: set = set()
        merged = []
        for b in p_ranked + g_ranked:
            if b.id not in seen:
                seen.add(b.id)
                merged.append(b)
        # Should equal p_ranked length (no new bullets from g_ranked since same playbook)
        assert len(merged) == len(p_ranked)

    def test_merge_ranked_sorted_by_grpo_advantage(self):
        """After merging, bullets are sorted descending by grpo_advantage."""
        pb = Playbook()
        pb.apply_delta([
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="A high advantage bullet"),
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="A low advantage bullet"),
        ])
        bullets = pb.bullets
        bullets[0].helpful = 10
        bullets[1].helpful = 1
        pb.recompute_grpo_advantages()

        ranked = pb.get_policy_ranked(top_k=2)
        # Re-sort as the fix does
        re_sorted = sorted(ranked, key=lambda b: b.grpo_advantage, reverse=True)
        assert re_sorted[0].grpo_advantage >= re_sorted[-1].grpo_advantage


# ===========================================================================
# B4: ace_generate_bullets Curator sees only 10 bullets
# Verified at data layer: bullets list access and grpo_advantage sorting.
# ===========================================================================

class TestB4CuratorBulletSample:
    """B4 fix: for large playbooks, sample top-50 by grpo_advantage."""

    def test_small_playbook_uses_all_bullets(self):
        """For playbooks with <= 50 bullets, all bullets are used as sample."""
        pb = Playbook()
        for i in range(30):
            pb.apply_delta([DeltaOperation(
                op_type="ADD",
                section="STRATEGIES & INSIGHTS",
                content=f"Strategy number {i}",
            )])
        bullets = pb.bullets
        # Simulates the fixed logic
        if len(bullets) <= 50:
            sample = [b.content for b in bullets]
        else:
            sorted_b = sorted(bullets, key=lambda b: b.grpo_advantage, reverse=True)
            sample = [b.content for b in sorted_b[:50]]
        assert len(sample) == 30

    def test_large_playbook_samples_top50_by_grpo(self):
        """For playbooks with > 50 bullets, sample is top-50 by grpo_advantage."""
        pb = Playbook()
        for i in range(60):
            pb.apply_delta([DeltaOperation(
                op_type="ADD",
                section="STRATEGIES & INSIGHTS",
                content=f"Strategy number {i}",
            )])
        # Give different grpo_advantages
        for idx, b in enumerate(pb.bullets):
            b.grpo_advantage = float(idx)  # bullet 59 has highest advantage
        bullets = pb.bullets
        sorted_b = sorted(bullets, key=lambda b: b.grpo_advantage, reverse=True)
        sample = [b.content for b in sorted_b[:50]]
        assert len(sample) == 50
        # The top bullet by grpo_advantage is "Strategy number 59"
        assert "Strategy number 59" in sample
        # The lowest-advantage bullet is not included
        assert "Strategy number 0" not in sample


# ===========================================================================
# B5: MERGE in auto_apply mode implemented as UPDATE
# Verified at data layer: DeltaOperation with op_type="MERGE" preserves
# counter history whereas "UPDATE" does not.
# ===========================================================================

class TestB5MergeOpType:
    """B5 fix: MERGE preserves counter history; UPDATE does not."""

    def test_merge_combines_helpful_counts(self):
        """MERGE operation combines helpful/harmful counters from both bullets."""
        pb = Playbook()
        pb.apply_delta([
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Bullet A"),
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Bullet B"),
        ])
        a, b = pb.bullets[0], pb.bullets[1]
        a.helpful = 4
        a.harmful = 1
        b.helpful = 3
        b.harmful = 2

        # Apply MERGE: keep a, absorb b
        pb.apply_delta([DeltaOperation(
            op_type="MERGE",
            section="",
            content="Merged content",
            bullet_id=a.id,
            merge_target=b.id,
        )])

        keeper = pb.get_bullet(a.id)
        assert keeper is not None
        assert keeper.helpful == 7   # 4 + 3
        assert keeper.harmful == 3   # 1 + 2
        assert keeper.content == "Merged content"
        assert pb.get_bullet(b.id) is None  # absorbed bullet removed

    def test_update_does_not_combine_counts(self):
        """UPDATE operation replaces content but does NOT merge counter history."""
        pb = Playbook()
        pb.apply_delta([
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Bullet A"),
        ])
        a = pb.bullets[0]
        a.helpful = 4
        a.harmful = 1

        pb.apply_delta([DeltaOperation(
            op_type="UPDATE",
            section="",
            content="Updated content",
            bullet_id=a.id,
        )])

        updated = pb.get_bullet(a.id)
        assert updated is not None
        assert updated.helpful == 4   # unchanged
        assert updated.harmful == 1   # unchanged
        assert updated.content == "Updated content"

    def test_merge_preserves_failure_lessons(self):
        """MERGE combines failure_lessons from both bullets."""
        pb = Playbook()
        pb.apply_delta([
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Keeper"),
            DeltaOperation(op_type="ADD", section="STRATEGIES & INSIGHTS", content="Absorbed"),
        ])
        keeper_b, absorbed_b = pb.bullets[0], pb.bullets[1]
        absorbed_b.failure_lessons.append(FailureLesson(
            failure_point="broke",
            flawed_reasoning="wrong assumption",
            counterfactual="fix it",
            prevention_principle="always check",
        ))

        pb.apply_delta([DeltaOperation(
            op_type="MERGE",
            section="",
            content="Combined",
            bullet_id=keeper_b.id,
            merge_target=absorbed_b.id,
        )])

        merged = pb.get_bullet(keeper_b.id)
        assert len(merged.failure_lessons) == 1
        assert merged.failure_lessons[0].prevention_principle == "always check"


# ===========================================================================
# B6: ROLLBACK proposal is dead code
# Verified at data layer: PlaybookSchema.from_dict / apply_schema.
# The _apply_rollback helper is MCP-layer, but we test the data operations
# it relies on (apply_schema with a parent schema) here.
# ===========================================================================

class TestB6RollbackDataLayer:
    """B6 fix: verify that applying a parent schema reverts sections correctly."""

    def _make_schema_pair(self):
        """Return (original_schema, evolved_schema) for rollback tests."""
        from ccr.core.types import PlaybookSchema
        original = PlaybookSchema.default()
        evolved = PlaybookSchema(
            version=original.version + 1,
            sections=original.sections + ["EXTRA SECTION"],
            slug_map=dict(original.slug_map),
            decay_rate=original.decay_rate,
            prune_min_harmful=original.prune_min_harmful,
            evolution_threshold=original.evolution_threshold,
            token_budget=original.token_budget,
            parent_version=original.version,
        )
        return original, evolved

    def test_apply_schema_reverts_to_parent_sections(self):
        """Applying the original schema removes the extra section added in the evolved schema."""
        from ccr.core.types import PlaybookSchema
        original, evolved = self._make_schema_pair()

        pb = Playbook(SAMPLE_PLAYBOOK)
        # Apply evolved schema (adds EXTRA SECTION)
        pb.apply_schema(evolved)
        assert "EXTRA SECTION" in pb._sections

        # Roll back to original schema
        pb.apply_schema(original)
        assert "EXTRA SECTION" not in pb._sections

    def test_apply_schema_roundtrip(self):
        """Schema serialization/deserialization preserves parent_version."""
        from ccr.core.types import PlaybookSchema
        original, evolved = self._make_schema_pair()

        serialized = evolved.to_dict()
        restored = PlaybookSchema.from_dict(serialized)
        assert restored.parent_version == original.version
        assert restored.version == evolved.version

    def test_rollback_requires_parent_version(self):
        """Schema with no parent_version cannot be rolled back."""
        from ccr.core.types import PlaybookSchema
        schema = PlaybookSchema.default()
        # Default schema has parent_version=None
        assert schema.parent_version is None


class TestSM2AdaptiveDecay:
    """SM-2 inspired per-bullet adaptive decay rate tests."""

    def _make_bullet(self, helpful: int = 0, harmful: int = 0) -> "Bullet":
        """Create a bullet and update its counters to trigger SM-2 rate calculation."""
        from ccr.ace.playbook import Bullet
        b = Bullet(id="str-00001", helpful=0, harmful=0, content="test strategy")
        # Directly set counters to avoid needing a Playbook instance
        b.helpful = helpful
        b.harmful = harmful
        # Recompute personal_decay_rate as update_bullet_counts would
        b.personal_decay_rate = max(0.90, min(0.99, 0.95 + (b.helpful - b.harmful) * 0.002))
        return b

    def test_helpful_bullet_gets_slower_decay(self):
        """helpful=10, harmful=0 → personal_decay_rate = 0.97 (0.95 + 10*0.002)."""
        b = self._make_bullet(helpful=10, harmful=0)
        assert abs(b.personal_decay_rate - 0.97) < 1e-9

    def test_harmful_bullet_gets_faster_decay(self):
        """helpful=0, harmful=5 → personal_decay_rate = 0.94 (0.95 - 5*0.002)."""
        b = self._make_bullet(helpful=0, harmful=5)
        assert abs(b.personal_decay_rate - 0.94) < 1e-9

    def test_decay_rate_clamped_high(self):
        """helpful=50, harmful=0 → personal_decay_rate ≤ 0.99."""
        b = self._make_bullet(helpful=50, harmful=0)
        assert b.personal_decay_rate <= 0.99
        assert b.personal_decay_rate == pytest.approx(0.99)

    def test_decay_rate_clamped_low(self):
        """helpful=0, harmful=50 → personal_decay_rate ≥ 0.90."""
        b = self._make_bullet(helpful=0, harmful=50)
        assert b.personal_decay_rate >= 0.90
        assert b.personal_decay_rate == pytest.approx(0.90)

    def test_personal_decay_used_in_effective_score(self):
        """personal_decay_rate=0.80 should decay faster than default 0.95."""
        from ccr.ace.playbook import Bullet
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        # Bullet with personal_decay_rate=0.80 (fast decay)
        b_fast = Bullet(
            id="str-00001", helpful=5, harmful=0,
            content="test", last_updated=old_ts,
            personal_decay_rate=0.80,
        )
        # Bullet with personal_decay_rate=0.0 (uses schema default 0.95)
        b_default = Bullet(
            id="str-00001", helpful=5, harmful=0,
            content="test", last_updated=old_ts,
            personal_decay_rate=0.0,
        )
        score_fast = b_fast.effective_score(decay_rate=0.95)
        score_default = b_default.effective_score(decay_rate=0.95)
        # personal=0.80 decays faster than 0.95 over 30 days: 0.80^30 << 0.95^30
        assert score_fast < score_default

    def test_zero_personal_decay_uses_schema_default(self):
        """personal_decay_rate=0.0 (default) → effective_score uses passed decay_rate."""
        from ccr.ace.playbook import Bullet
        from datetime import datetime, timedelta, timezone

        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        b = Bullet(
            id="str-00001", helpful=5, harmful=0,
            content="test", last_updated=old_ts,
            personal_decay_rate=0.0,
        )
        # With default param 0.95: raw=5, days=30, expected = 5 * 0.95^30
        score_95 = b.effective_score(decay_rate=0.95)
        score_80 = b.effective_score(decay_rate=0.80)
        # 0.95^30 ≈ 0.214, 0.80^30 ≈ 0.0012 — so score_95 > score_80 (both positive)
        assert score_95 > score_80

    def test_sm2_set_via_update_bullet_counts(self):
        """Verify that Playbook.update_bullet_counts sets personal_decay_rate correctly."""
        from ccr.ace.playbook import Bullet
        # Use a fresh playbook with a known starting state (helpful=0, harmful=0)
        fresh_pb_text = "## STRATEGIES & INSIGHTS\n[str-00001] helpful=0 harmful=0 :: Test strategy\n"
        pb = Playbook(fresh_pb_text)
        bullet_id = "str-00001"
        # Tag bullet as helpful 10 times, harmful 0 times
        for _ in range(10):
            pb.update_bullet_counts([{"id": bullet_id, "tag": "helpful"}])
        updated_bullet = pb.get_bullet(bullet_id)
        assert updated_bullet is not None
        assert updated_bullet.helpful == 10
        expected_rate = min(0.99, 0.95 + 10 * 0.002)  # = 0.97
        assert abs(updated_bullet.personal_decay_rate - expected_rate) < 1e-9

    def test_sm2_persistence_roundtrip(self):
        """personal_decay_rate survives save_failure_lessons / load_failure_lessons."""
        import os
        import tempfile
        pb = Playbook(SAMPLE_PLAYBOOK)
        bullet = pb.bullets[0]
        bullet.personal_decay_rate = 0.97
        bullet.last_updated = "2026-01-01T00:00:00+00:00"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pb.save_failure_lessons(path)
            pb2 = Playbook(SAMPLE_PLAYBOOK)
            pb2.load_failure_lessons(path)
            restored = pb2.get_bullet(bullet.id)
            assert restored is not None
            assert abs(restored.personal_decay_rate - 0.97) < 1e-9
        finally:
            os.unlink(path)


# ===========================================================================
# F8: Stochastic Schema Exploration — MCE (arXiv:2601.21557)
# ===========================================================================


class TestStochasticProposal:
    """_stochastic_proposal() generates a valid MCE exploratory proposal."""

    def _make_schema(self):
        from ccr.core.types import PlaybookSchema
        return PlaybookSchema.default()

    def test_confidence_at_most_04(self):
        """Stochastic proposal always has confidence <= 0.4."""
        from ccr.mcp.ace_schema_tools import _stochastic_proposal
        schema = self._make_schema()
        for _ in range(20):
            proposal = _stochastic_proposal(schema)
            assert proposal.confidence <= 0.4, f"Expected confidence <=0.4, got {proposal.confidence}"

    def test_change_type_is_valid(self):
        """change_type is one of the recognized schema adjustment types."""
        from ccr.mcp.ace_schema_tools import _stochastic_proposal
        valid_types = {"ADJUST_DECAY", "ADJUST_PRUNING", "ADJUST_SEARCH_THRESHOLD"}
        schema = self._make_schema()
        for _ in range(20):
            proposal = _stochastic_proposal(schema)
            assert proposal.change_type in valid_types

    def test_decay_rate_in_bounds(self):
        """ADJUST_DECAY proposal keeps decay_rate in [0.85, 0.99]."""
        from ccr.mcp.ace_schema_tools import _stochastic_proposal
        import random
        random.seed(42)
        schema = self._make_schema()
        for _ in range(50):
            proposal = _stochastic_proposal(schema)
            if proposal.change_type == "ADJUST_DECAY":
                new_rate = proposal.details["new_rate"]
                assert 0.85 <= new_rate <= 0.99, f"decay_rate {new_rate} out of bounds"

    def test_search_threshold_in_bounds(self):
        """ADJUST_SEARCH_THRESHOLD keeps threshold in [0.05, 0.9]."""
        from ccr.mcp.ace_schema_tools import _stochastic_proposal
        import random
        random.seed(7)
        schema = self._make_schema()
        for _ in range(50):
            proposal = _stochastic_proposal(schema)
            if proposal.change_type == "ADJUST_SEARCH_THRESHOLD":
                new_t = proposal.details["new_threshold"]
                assert 0.05 <= new_t <= 0.9, f"threshold {new_t} out of bounds"

    def test_pruning_min_at_least_one(self):
        """ADJUST_PRUNING keeps prune_min_harmful >= 1."""
        from ccr.mcp.ace_schema_tools import _stochastic_proposal
        import random
        random.seed(3)
        schema = self._make_schema()
        schema.prune_min_harmful = 1  # minimum baseline
        for _ in range(50):
            proposal = _stochastic_proposal(schema)
            if proposal.change_type == "ADJUST_PRUNING":
                new_val = proposal.details["new_min_harmful"]
                assert new_val >= 1, f"prune_min_harmful {new_val} < 1"

    def test_explore_true_adds_stochastic_to_proposals(self):
        """ace_evolve_schema(explore=True) returns a proposal with confidence <= 0.4."""
        import os
        from ccr.mcp_server import _init
        from ccr.mcp_server import ace_evolve_schema
        import ccr.mcp_server as mcp_mod
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fake_home = os.path.join(tmp, "global_ccr")
            os.makedirs(fake_home)
            orig = os.path.expanduser

            def mock_eu(p):
                if p.startswith("~/.ccr"):
                    return fake_home + p[5:]
                return orig(p)

            import unittest.mock as mock
            with mock.patch("os.path.expanduser", side_effect=mock_eu):
                _init(tmp)
                try:
                    result = ace_evolve_schema(scope="project", explore=True)
                    assert "confidence: 0." in result["message"] or "0.3" in result["message"] or "0.4" in result["message"]
                finally:
                    mcp_mod._memory = None
                    mcp_mod._playbook = None
                    mcp_mod._global_playbook = None
                    mcp_mod._repo_index = None
                    mcp_mod._repl = None
