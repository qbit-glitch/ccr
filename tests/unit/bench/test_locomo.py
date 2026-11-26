"""Tests for the LoCoMo adapter using a small inline fixture."""

from __future__ import annotations

from ccr.bench import datasets
from ccr.bench.datasets import locomo
from ccr.bench.datasets.locomo import LOAD_STATS, parse_locomo

FIXTURE = [{
    "sample_id": "conv-1",
    "conversation": {
        "speaker_a": "Caroline", "speaker_b": "Mel",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
            {"speaker": "Mel", "dia_id": "D1:2", "text": "Hi there"},
            {"speaker": "Caroline", "dia_id": "D1:3", "text": "Look at this",
             "blip_caption": "a photo of a sunset"},
        ],
        "session_2_date_time": "9:00 am on 9 May, 2023",
        "session_2": [
            {"speaker": "Mel", "dia_id": "D2:1", "text": "Morning"},
            {"speaker": "Caroline", "dia_id": "D2:2", "text": "Hello again"},
        ],
    },
    "qa": [
        {"question": "When?", "answer": "7 May 2023", "evidence": ["D1:3"], "category": 2},
        {"question": "Semicolon packed", "answer": "x", "evidence": ["D1:1; D2:1"], "category": 1},
        {"question": "Space packed", "answer": "y", "evidence": ["D1:2 D2:2"], "category": 1},
        {"question": "Has a ghost id", "answer": "z", "evidence": ["D1:1", "D9:9"], "category": 1},
        {"question": "Adversarial", "adversarial_answer": "nope", "evidence": ["D1:1"],
         "category": 5},
        {"question": "Int answer no evidence", "answer": 2023, "evidence": [], "category": 3},
    ],
}]


def _queries():
    return {q.text: q for q in parse_locomo(FIXTURE)[0].queries}


def test_one_world_per_conversation_with_turn_docs() -> None:
    worlds = parse_locomo(FIXTURE)
    assert [w.id for w in worlds] == ["conv-1"]
    docs = {d.id: d.text for d in worlds[0].docs}
    assert set(docs) == {"D1:1", "D1:2", "D1:3", "D2:1", "D2:2"}
    assert docs["D1:1"] == "[1:56 pm on 8 May, 2023] Caroline: Hey Mel!"
    assert "a photo of a sunset" in docs["D1:3"]


def test_packed_evidence_strings_are_split() -> None:
    qs = _queries()
    assert qs["Semicolon packed"].gold_ids == frozenset({"D1:1", "D2:1"})
    assert qs["Space packed"].gold_ids == frozenset({"D1:2", "D2:2"})


def test_evidence_ids_missing_from_conversation_are_dropped_and_counted() -> None:
    parse_locomo(FIXTURE)
    assert _queries()["Has a ghost id"].gold_ids == frozenset({"D1:1"})
    assert LOAD_STATS["dropped_evidence_ids"] == 1


def test_adversarial_category_is_excluded_from_scoring() -> None:
    q = _queries()["Adversarial"]
    assert q.gold_ids == frozenset() and q.answer is None
    assert q.category == "cat5"
    assert LOAD_STATS["adversarial_excluded"] == 1


def test_non_string_answers_become_strings_and_keep_answer_only_queries() -> None:
    q = _queries()["Int answer no evidence"]
    assert q.answer == "2023" and q.gold_ids == frozenset()
    assert _queries()["When?"].category == "cat2"


def test_registered_as_locomo_adapter() -> None:
    assert datasets.ADAPTERS["locomo"] is locomo.load_locomo
