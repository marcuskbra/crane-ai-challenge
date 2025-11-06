"""
Tests for MetricsTracker.

Tests metrics collection for planner performance tracking.
"""

import pytest

from challenge.services.planning.metrics_tracker import MetricsTracker, PlannerStats


class TestMetricsTracker:
    """Test cases for MetricsTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh MetricsTracker instance."""
        return MetricsTracker()

    def test_initial_state(self, tracker):
        """Test tracker starts with zero counters."""
        stats = tracker.get_stats()

        assert stats.total_plans == 0
        assert stats.llm_plans == 0
        assert stats.pattern_plans == 0
        assert stats.total_tokens == 0
        assert stats.total_latency_ms == 0.0

    def test_record_llm_plan(self, tracker):
        """Test recording an LLM plan with tokens."""
        tracker.record_plan(latency_ms=150.0, token_count=500)

        stats = tracker.get_stats()
        assert stats.total_plans == 1
        assert stats.llm_plans == 1
        assert stats.pattern_plans == 0
        assert stats.total_tokens == 500
        assert stats.total_latency_ms == 150.0

    def test_record_pattern_plan_with_none(self, tracker):
        """Test recording a pattern plan with None token count."""
        tracker.record_plan(latency_ms=10.0, token_count=None)

        stats = tracker.get_stats()
        assert stats.total_plans == 1
        assert stats.llm_plans == 0
        assert stats.pattern_plans == 1
        assert stats.total_tokens == 0
        assert stats.total_latency_ms == 10.0

    def test_record_pattern_plan_with_zero(self, tracker):
        """Test recording a pattern plan with zero token count."""
        tracker.record_plan(latency_ms=8.5, token_count=0)

        stats = tracker.get_stats()
        assert stats.total_plans == 1
        assert stats.llm_plans == 0
        assert stats.pattern_plans == 1
        assert stats.total_tokens == 0
        assert stats.total_latency_ms == 8.5

    def test_multiple_llm_plans(self, tracker):
        """Test recording multiple LLM plans accumulates correctly."""
        tracker.record_plan(latency_ms=100.0, token_count=400)
        tracker.record_plan(latency_ms=150.0, token_count=600)
        tracker.record_plan(latency_ms=200.0, token_count=800)

        stats = tracker.get_stats()
        assert stats.total_plans == 3
        assert stats.llm_plans == 3
        assert stats.pattern_plans == 0
        assert stats.total_tokens == 1800  # 400 + 600 + 800
        assert stats.total_latency_ms == 450.0  # 100 + 150 + 200

    def test_multiple_pattern_plans(self, tracker):
        """Test recording multiple pattern plans accumulates correctly."""
        tracker.record_plan(latency_ms=5.0, token_count=None)
        tracker.record_plan(latency_ms=10.0, token_count=0)
        tracker.record_plan(latency_ms=7.5, token_count=None)

        stats = tracker.get_stats()
        assert stats.total_plans == 3
        assert stats.llm_plans == 0
        assert stats.pattern_plans == 3
        assert stats.total_tokens == 0
        assert stats.total_latency_ms == 22.5  # 5.0 + 10.0 + 7.5

    def test_mixed_llm_and_pattern_plans(self, tracker):
        """Test recording mix of LLM and pattern plans."""
        # LLM plans
        tracker.record_plan(latency_ms=120.0, token_count=450)
        tracker.record_plan(latency_ms=180.0, token_count=550)

        # Pattern plans
        tracker.record_plan(latency_ms=8.0, token_count=None)
        tracker.record_plan(latency_ms=12.0, token_count=0)

        stats = tracker.get_stats()
        assert stats.total_plans == 4
        assert stats.llm_plans == 2
        assert stats.pattern_plans == 2
        assert stats.total_tokens == 1000  # 450 + 550
        assert stats.total_latency_ms == 320.0  # 120 + 180 + 8 + 12

    def test_reset_clears_all_metrics(self, tracker):
        """Test reset() clears all metrics to zero."""
        # Record some plans
        tracker.record_plan(latency_ms=150.0, token_count=500)
        tracker.record_plan(latency_ms=10.0, token_count=None)

        # Verify data exists
        stats_before = tracker.get_stats()
        assert stats_before.total_plans == 2

        # Reset
        tracker.reset()

        # Verify everything is zero
        stats_after = tracker.get_stats()
        assert stats_after.total_plans == 0
        assert stats_after.llm_plans == 0
        assert stats_after.pattern_plans == 0
        assert stats_after.total_tokens == 0
        assert stats_after.total_latency_ms == 0.0

    def test_reset_allows_new_recordings(self, tracker):
        """Test tracker can record plans after reset."""
        # Record, reset, record again
        tracker.record_plan(latency_ms=100.0, token_count=200)
        tracker.reset()
        tracker.record_plan(latency_ms=50.0, token_count=150)

        stats = tracker.get_stats()
        assert stats.total_plans == 1
        assert stats.llm_plans == 1
        assert stats.total_tokens == 150
        assert stats.total_latency_ms == 50.0

    @pytest.mark.parametrize(
        ("latency", "token_count", "expected_llm", "expected_pattern"),
        [
            (100.0, 500, 1, 0),  # LLM plan
            (10.0, None, 0, 1),  # Pattern plan (None)
            (15.0, 0, 0, 1),  # Pattern plan (zero)
            (200.0, 1000, 1, 0),  # LLM plan with many tokens
            (5.0, 1, 1, 0),  # LLM plan with minimal tokens
        ],
    )
    def test_record_plan_variations(self, tracker, latency, token_count, expected_llm, expected_pattern):
        """Test various plan recording scenarios."""
        tracker.record_plan(latency_ms=latency, token_count=token_count)

        stats = tracker.get_stats()
        assert stats.llm_plans == expected_llm
        assert stats.pattern_plans == expected_pattern
        assert stats.total_plans == 1

    def test_get_stats_returns_model(self, tracker):
        """Test get_stats() returns PlannerStats model with expected fields."""
        stats = tracker.get_stats()

        assert isinstance(stats, PlannerStats)
        # Verify all expected fields exist
        assert hasattr(stats, "total_plans")
        assert hasattr(stats, "llm_plans")
        assert hasattr(stats, "pattern_plans")
        assert hasattr(stats, "total_tokens")
        assert hasattr(stats, "total_latency_ms")

    def test_get_stats_returns_correct_types(self, tracker):
        """Test get_stats() returns correct data types."""
        tracker.record_plan(latency_ms=150.0, token_count=500)
        stats = tracker.get_stats()

        assert isinstance(stats.total_plans, int)
        assert isinstance(stats.llm_plans, int)
        assert isinstance(stats.pattern_plans, int)
        assert isinstance(stats.total_tokens, int)
        assert isinstance(stats.total_latency_ms, float)

    def test_negative_latency(self, tracker):
        """Test recording with negative latency (edge case)."""
        # While unusual, tracker should accept it
        tracker.record_plan(latency_ms=-10.0, token_count=None)

        stats = tracker.get_stats()
        assert stats.total_latency_ms == -10.0
        assert stats.total_plans == 1

    def test_large_token_counts(self, tracker):
        """Test handling of large token counts."""
        # Very large token count (edge case)
        tracker.record_plan(latency_ms=500.0, token_count=100000)

        stats = tracker.get_stats()
        assert stats.total_tokens == 100000
        assert stats.llm_plans == 1

    def test_accumulation_preserves_precision(self, tracker):
        """Test latency accumulation preserves floating point precision."""
        tracker.record_plan(latency_ms=10.1, token_count=None)
        tracker.record_plan(latency_ms=20.2, token_count=None)
        tracker.record_plan(latency_ms=30.3, token_count=None)

        stats = tracker.get_stats()
        # Account for floating point arithmetic
        assert abs(stats.total_latency_ms - 60.6) < 0.01
