"""Tests for cost tracking."""

from ccr.utils.costs import CostTracker, calculate_cost


class TestCalculateCost:
    def test_claude_sonnet_cost(self):
        cost = calculate_cost("claude-sonnet-4-5", 1_000_000, 1_000_000)
        assert cost is not None
        assert cost == pytest.approx(18.0)  # $3 input + $15 output

    def test_qwen_cost(self):
        cost = calculate_cost("qwen3-8b", 1_000_000, 1_000_000)
        assert cost is not None
        assert cost == pytest.approx(0.12)  # $0.06 + $0.06

    def test_unknown_model_returns_none(self):
        cost = calculate_cost("unknown-model-xyz", 1000, 1000)
        assert cost is None


class TestCostTracker:
    def test_record_and_total(self):
        tracker = CostTracker()
        tracker.record("qwen3-8b", 10000, 5000)
        tracker.record("qwen3-8b", 10000, 5000)
        total = tracker.get_total_cost()
        assert total > 0

    def test_savings_calculation(self):
        tracker = CostTracker(baseline_model="claude-sonnet-4-5")
        # Simulate routing to Qwen instead of Claude
        tracker.record("qwen3-8b", 100_000, 50_000)
        savings = tracker.get_savings_vs_baseline()
        assert savings > 0  # Qwen is much cheaper than Claude

    def test_report(self):
        tracker = CostTracker()
        tracker.record("qwen3-8b", 10000, 5000)
        tracker.record("claude-sonnet-4-5", 5000, 2000)
        report = tracker.get_report()
        assert "models" in report
        assert "total_cost_usd" in report
        assert "savings_usd" in report
        assert "savings_pct" in report


import pytest
