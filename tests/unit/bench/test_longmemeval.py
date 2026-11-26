"""Tests for the LongMemEval retrieval adapter (inline fixture)."""

from __future__ import annotations

from ccr.bench import datasets
from ccr.bench.datasets import longmemeval
from ccr.bench.datasets.longmemeval import LOAD_STATS, parse_longmemeval
from ccr.bench.types import BenchConfig

FIXTURE = [
    {
        "question_id": "q-aaa", "question_type": "single-session-user",
        "question": "What degree did I graduate with?", "answer": "Business Administration",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_session_ids": ["s1", "answer_s2", "s3"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 10:00", "2023/05/22 (Mon) 11:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            [{"role": "user", "content": "I graduated in Business Administration",
              "has_answer": True}],
            [{"role": "user", "content": "weather talk"}],
        ],
        "answer_session_ids": ["answer_s2"],
    },
    {
        "question_id": "q-bbb_abs", "question_type": "temporal-reasoning",
        "question": "Abstention?", "answer": "unanswerable",
        "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_session_ids": ["s1"], "haystack_dates": ["2023/05/20 (Sat) 02:21"],
        "haystack_sessions": [[{"role": "user", "content": "x"}]],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "q-ccc", "question_type": "multi-session", "question": "Ghost gold?",
        "answer": 3, "question_date": "2023/05/30 (Tue) 23:40",
        "haystack_session_ids": ["s1"], "haystack_dates": ["2023/05/20 (Sat) 02:21"],
        "haystack_sessions": [[{"role": "user", "content": "y"}]],
        "answer_session_ids": ["s1", "ghost"],
    },
]


def _by_id():
    return {w.id: w for w in parse_longmemeval(FIXTURE)}


def test_one_world_per_question_with_one_doc_per_session() -> None:
    w = _by_id()["q-aaa"]
    assert [d.id for d in w.docs] == ["s1", "answer_s2", "s3"]
    text = {d.id: d.text for d in w.docs}
    assert text["s1"] == "[2023/05/20 (Sat) 02:21]\nuser: hello\nassistant: hi"
    assert w.group == "single-session-user"


def test_gold_is_answer_session_ids_and_answer_kept() -> None:
    q = _by_id()["q-aaa"].queries[0]
    assert q.gold_ids == frozenset({"answer_s2"})
    assert q.answer == "Business Administration"
    assert q.category == "single-session-user"
    assert q.text == "What degree did I graduate with?"


def test_abstention_questions_are_excluded_and_counted() -> None:
    q = _by_id()["q-bbb_abs"].queries[0]
    assert q.gold_ids == frozenset() and q.answer is None
    assert LOAD_STATS["abstention_excluded"] == 1


def test_missing_gold_sessions_dropped_and_counted_and_answer_stringified() -> None:
    q = _by_id()["q-ccc"].queries[0]
    assert q.gold_ids == frozenset({"s1"})
    assert q.answer == "3"
    assert LOAD_STATS["dropped_gold_ids"] == 1


def test_registered_with_type_stratified_sampler() -> None:
    assert datasets.ADAPTERS["longmemeval"] is longmemeval.load_longmemeval
    worlds = parse_longmemeval(FIXTURE)
    cfg = BenchConfig(ks=(5,), seed=1, sample_fraction=0.34, chunk_tokens=1, cache_dir="/x")
    picked = datasets.SAMPLERS["longmemeval"](worlds, cfg)
    assert {w.group for w in picked} == {w.group for w in worlds}
