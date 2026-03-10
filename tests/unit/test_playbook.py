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
