"""Tests for the MemoryAgentBench judge-free adapter."""

from __future__ import annotations

import pytest

from ccr.bench import datasets
from ccr.bench.datasets import mab
from ccr.bench.datasets.mab import LOAD_STATS, is_judge_free, parse_mab


def _row(source: str, context: str = "line one\nline two\nline three", questions=("q1", "q2"),
         answers=(("a1",), ("a2", "a2b"))) -> dict:
    return {"context": context, "questions": list(questions),
            "answers": [list(a) for a in answers], "metadata": {"source": source}}


@pytest.mark.parametrize("source", [
    "ruler_qa1_197K", "eventqa_65536", "factconsolidation_sh_6k",
    "icl_banking77_5900shot_balance", "detective_qa_x",
])
def test_judge_free_sources_are_kept(source) -> None:
    assert is_judge_free(source)


@pytest.mark.parametrize("source", ["longmemeval_s*", "recsys_redial_full", "infbench_sum_eng"])
def test_judge_based_or_unsupported_sources_are_dropped(source) -> None:
    assert not is_judge_free(source)


def test_dropped_sources_are_counted_and_not_loaded() -> None:
    worlds = parse_mab([_row("ruler_qa1_197K"), _row("longmemeval_s*"), _row("recsys_redial_full")], 512)
    assert len(worlds) == 1
    assert LOAD_STATS["dropped_examples"] == 2


def test_one_world_per_example_with_one_query_per_question() -> None:
    world = parse_mab([_row("ruler_qa1_197K")], 512)[0]
    assert world.group == "ruler_qa1_197K"
    assert [q.text for q in world.queries] == ["q1", "q2"]
    assert all(q.gold_ids == frozenset() for q in world.queries)
    assert world.queries[1].answer == "a2" and world.queries[1].alt_answers == ("a2b",)
    assert world.queries[0].category == "ruler_qa1_197K"


def test_chunks_split_on_line_boundaries_without_losing_text() -> None:
    context = "\n".join(f"line number {i}" for i in range(40))
    world = parse_mab([_row("ruler_qa1_197K", context=context)], chunk_tokens=10)[0]
    assert len(world.docs) > 3
    assert all(d.text.strip() for d in world.docs)
    assert "\n".join(d.text for d in world.docs) == context
    assert [d.id for d in world.docs][:2] == [f"{world.id}#0", f"{world.id}#1"]


def test_overlong_single_line_is_split_not_dropped() -> None:
    world = parse_mab([_row("ruler_qa1_197K", context="word " * 500)], chunk_tokens=10)[0]
    assert len(world.docs) > 1
    assert sum(len(d.text.split()) for d in world.docs) == 500


def test_world_ids_are_unique_across_examples_of_same_source() -> None:
    worlds = parse_mab([_row("eventqa_65536"), _row("eventqa_65536")], 512)
    assert len({w.id for w in worlds}) == 2


def test_registered_and_marked_as_needing_reader() -> None:
    assert datasets.ADAPTERS["memoryagentbench"] is mab.load_mab
    assert "memoryagentbench" in datasets.NEEDS_READER
    assert "memoryagentbench" in datasets.SAMPLERS


def test_parquet_round_trip_through_loader(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from ccr.bench.types import BenchConfig

    table = pa.table({
        "context": ["alpha beta\ngamma"],
        "questions": [["what?"]],
        "answers": [[["x"]]],
        "metadata": [{"source": "ruler_qa1_197K"}],
    })
    for split in mab.SPLITS:
        pq.write_table(table, tmp_path / f"{split}-00000-of-00001.parquet")
    cfg = BenchConfig(ks=(5,), seed=1, sample_fraction=1.0, chunk_tokens=512,
                      cache_dir=str(tmp_path / "cache"))
    worlds = mab.load_mab(cfg, base_url=(tmp_path).as_uri())
    assert len(worlds) == len(mab.SPLITS)
    assert worlds[0].queries[0].answer == "x"
