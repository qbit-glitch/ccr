"""Token counting and context limit utilities."""

from __future__ import annotations

from functools import lru_cache

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-3-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5": 272_000,
    "gpt-5-mini": 272_000,
    "gpt-oss-20b": 128_000,
    "gpt-oss-120b": 128_000,
    "qwen3-8b": 128_000,
    "qwen3-30b": 128_000,
    "qwen3-coder": 262_000,
    "deepseek-v3": 128_000,
}
DEFAULT_CONTEXT_LIMIT = 128_000

_tiktoken_cache: dict[str, object] = {}


def _get_tiktoken_encoding(model: str = "cl100k_base"):
    """Get tiktoken encoding, with caching."""
    if model not in _tiktoken_cache:
        try:
            import tiktoken
            _tiktoken_cache[model] = tiktoken.get_encoding(model)
        except (ImportError, Exception):
            _tiktoken_cache[model] = None
    return _tiktoken_cache[model]


def count_tokens(text: str | list[dict], model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken if available, else heuristic."""
    if isinstance(text, list):
        combined = ""
        for msg in text:
            content = msg.get("content", "")
            if isinstance(content, str):
                combined += content + "\n"
            elif isinstance(content, list):
                for block in content:
                    combined += block.get("text", "") + "\n"
            combined += "role: " + msg.get("role", "") + "\n"
        text = combined

    return _count_tokens_cached(text, model)


@lru_cache(maxsize=512)
def _count_tokens_cached(text: str, model: str = "cl100k_base") -> int:
    """LRU-cached token counting for repeated strings."""
    enc = _get_tiktoken_encoding(model)
    if enc is not None:
        return len(enc.encode(text))
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Fast heuristic: ~4 chars per token for English/code."""
    return max(1, len(text) // 4)


def get_context_limit(model: str) -> int:
    """Return known context window size for a model."""
    model_lower = model.lower()
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in model_lower:
            return limit
    return DEFAULT_CONTEXT_LIMIT


def fits_in_budget(text: str, budget: int, model: str = "cl100k_base") -> bool:
    return count_tokens(text, model) <= budget
