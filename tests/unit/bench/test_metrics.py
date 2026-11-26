"""Tests for judge-free benchmark metrics."""

from __future__ import annotations

import math

import pytest

from ccr.bench.metrics import (
    bleu1,
    count_tokens,
    exact_match,
    mrr,
    ndcg_at_k,
    recall_at_k,
    sub_exact_match,
    token_f1,
)
from ccr.bench.types import BenchConfig, Doc, Query, World


def test_recall_at_k_counts_gold_found_in_top_k() -> None:
    assert recall_at_k(["a", "b", "c"], frozenset({"b", "z"}), 2) == 0.5


def test_recall_at_k_ignores_hits_below_cutoff() -> None:
    assert recall_at_k(["a", "b", "c"], frozenset({"c"}), 2) == 0.0


def test_recall_at_k_empty_gold_is_zero_not_error() -> None:
    assert recall_at_k(["a"], frozenset(), 5) == 0.0


def test_mrr_uses_first_gold_rank() -> None:
    assert mrr(["x", "b", "b"], frozenset({"b"})) == 0.5
    assert mrr(["x"], frozenset({"b"})) == 0.0


def test_ndcg_linear_gain_matches_hand_value() -> None:
    assert ndcg_at_k(["a", "b"], {"b": 1}, 2) == pytest.approx(1 / math.log2(3))


def test_ndcg_perfect_ranking_is_one_and_graded_gains_count() -> None:
    assert ndcg_at_k(["b", "a"], {"b": 2, "a": 1}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b"], {"b": 2, "a": 1}, 2) < 1.0


def test_ndcg_without_gains_is_zero() -> None:
    assert ndcg_at_k(["a"], {}, 10) == 0.0


def test_token_f1_normalizes_articles_and_penalizes_extra_tokens() -> None:
    assert token_f1("a big cat sat", "cat sat") == pytest.approx(0.8)
    assert token_f1("The cat", "cat") == pytest.approx(1.0)


def test_token_f1_empty_inputs_are_zero() -> None:
    assert token_f1("", "cat") == 0.0
    assert token_f1("cat", "") == 0.0


def test_exact_match_ignores_case_punctuation_articles() -> None:
    assert exact_match("The Paris!", "paris") == 1.0
    assert exact_match("Paris, France", "paris") == 0.0


def test_sub_exact_match_finds_gold_inside_prediction() -> None:
    assert sub_exact_match("It was Paris.", "paris") == 1.0
    assert sub_exact_match("It was Rome.", "paris") == 0.0
    assert sub_exact_match("anything", "") == 0.0


def test_bleu1_is_unigram_precision_with_brevity_penalty() -> None:
    assert bleu1("cat sat", "cat sat") == pytest.approx(1.0)
    assert bleu1("cat", "cat sat") == pytest.approx(math.exp(1 - 2 / 1))
    assert bleu1("", "cat") == 0.0


def test_count_tokens_is_positive_for_text_and_zero_for_empty() -> None:
    assert count_tokens("hello world, this is a sentence") > 0
    assert count_tokens("") == 0


def test_types_are_frozen_and_config_has_expected_fields() -> None:
    doc = Doc(id="d1", text="hi")
    query = Query(id="q1", text="?", gold_ids=frozenset({"d1"}), answer=None, category="c")
    world = World(id="w", docs=(doc,), queries=(query,), group="g")
    cfg = BenchConfig(
        ks=(1, 5), seed=7, sample_fraction=0.1, chunk_tokens=512, cache_dir="/tmp/x"
    )
    with pytest.raises(Exception):
        doc.text = "changed"  # type: ignore[misc]
    assert world.queries[0].gold_ids == frozenset({"d1"})
    assert cfg.ks == (1, 5)
