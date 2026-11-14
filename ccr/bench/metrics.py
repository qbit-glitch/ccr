"""Deterministic, judge-free metrics for retrieval and short-answer scoring."""

from __future__ import annotations

import math
import re
import string
from collections import Counter

from ccr.utils.tokens import count_tokens as _count_tokens

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def count_tokens(text: str) -> int:
    """Token count used for every 'tokens returned' figure in reports."""
    if not text:
        return 0
    return _count_tokens(text)


def _normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation and articles."""
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCT)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return " ".join(no_articles.split())


def recall_at_k(ranked: list[str], gold: frozenset[str], k: int) -> float:
    """Fraction of gold ids found in the top ``k`` ranked ids."""
    if not gold:
        return 0.0
    return len(gold.intersection(ranked[:k])) / len(gold)


def mrr(ranked: list[str], gold: frozenset[str]) -> float:
    """Reciprocal rank of the first gold id, 0.0 when none is retrieved."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def _dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], gains: dict[str, int], k: int) -> float:
    """nDCG@k with linear gain (trec_eval style), graded relevance supported."""
    if not gains:
        return 0.0
    actual = _dcg([gains.get(doc_id, 0) for doc_id in ranked[:k]])
    ideal = _dcg(sorted(gains.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 after SQuAD normalization."""
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    """1.0 when normalized prediction equals normalized gold."""
    norm_gold = _normalize(gold)
    if not norm_gold:
        return 0.0
    return float(_normalize(pred) == norm_gold)


def sub_exact_match(pred: str, gold: str) -> float:
    """1.0 when normalized gold appears inside the normalized prediction."""
    norm_gold = _normalize(gold)
    if not norm_gold:
        return 0.0
    return float(norm_gold in _normalize(pred))


def bleu1(pred: str, gold: str) -> float:
    """Unigram BLEU: clipped precision times brevity penalty."""
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    clipped = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    precision = clipped / len(pred_tokens)
    if len(pred_tokens) >= len(gold_tokens):
        penalty = 1.0
    else:
        penalty = math.exp(1 - len(gold_tokens) / len(pred_tokens))
    return precision * penalty
