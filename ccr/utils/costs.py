"""Cost tracking and savings calculation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

# (input_cost_per_1M, output_cost_per_1M) in USD
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-5": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-3-5": (0.80, 4.0),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5": (10.0, 40.0),
    "gpt-5-mini": (1.0, 4.0),
    # Local / cheap API
    "gpt-oss-20b": (0.06, 0.06),  # local vLLM — effectively free
    "gpt-oss-120b": (0.15, 0.15),
    "qwen3-8b": (0.06, 0.06),
    "qwen3-30b-a3b": (0.10, 0.10),
    "qwen3-coder": (0.20, 0.60),
    "deepseek-v3": (0.27, 1.10),
    "minimax-m2.5": (0.50, 2.0),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Calculate cost in USD for a given model and token counts."""
    model_lower = model.lower()
    for key, (inp_cost, out_cost) in MODEL_COSTS.items():
        if key in model_lower:
            return (input_tokens * inp_cost + output_tokens * out_cost) / 1_000_000
    return None


@dataclass
class _ModelRecord:
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class CostTracker:
    """Thread-safe cost accumulator for a CCR session."""

    def __init__(self, baseline_model: str = "claude-sonnet-4-5"):
        self._lock = threading.Lock()
        self._records: dict[str, _ModelRecord] = {}
        self._baseline_model = baseline_model

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        cost = calculate_cost(model, input_tokens, output_tokens)
        with self._lock:
            if model not in self._records:
                self._records[model] = _ModelRecord(model=model)
            rec = self._records[model]
            rec.calls += 1
            rec.input_tokens += input_tokens
            rec.output_tokens += output_tokens
            if cost is not None:
                rec.cost_usd += cost
        return cost

    def get_total_cost(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records.values())

    def get_savings_vs_baseline(self) -> float:
        """Compare actual spend to hypothetical all-baseline cost."""
        with self._lock:
            total_input = sum(r.input_tokens for r in self._records.values())
            total_output = sum(r.output_tokens for r in self._records.values())
        baseline_cost = calculate_cost(self._baseline_model, total_input, total_output)
        if baseline_cost is None:
            return 0.0
        actual = self.get_total_cost()
        return max(0.0, baseline_cost - actual)

    def get_report(self) -> dict:
        with self._lock:
            models = {}
            for name, rec in self._records.items():
                models[name] = {
                    "calls": rec.calls,
                    "input_tokens": rec.input_tokens,
                    "output_tokens": rec.output_tokens,
                    "cost_usd": round(rec.cost_usd, 6),
                }
            total_input = sum(r.input_tokens for r in self._records.values())
            total_output = sum(r.output_tokens for r in self._records.values())
        baseline = calculate_cost(self._baseline_model, total_input, total_output) or 0.0
        actual = self.get_total_cost()
        return {
            "models": models,
            "total_cost_usd": round(actual, 6),
            "baseline_cost_usd": round(baseline, 6),
            "savings_usd": round(max(0.0, baseline - actual), 6),
            "savings_pct": round((1 - actual / baseline) * 100, 1) if baseline > 0 else 0.0,
        }
