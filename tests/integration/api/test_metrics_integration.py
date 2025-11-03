"""
Integration tests for metrics endpoint with planner metrics tracking.

Tests verify that planner metrics (token counting, LLM vs pattern detection,
latency tracking) are correctly tracked and exposed via the metrics endpoint.
"""

import asyncio

import pytest


class TestMetricsIntegration:
    """Integration tests for metrics endpoint with real planner metrics."""

    @pytest.mark.asyncio
    async def test_planner_metrics_increment_after_run(self, test_client):
        """Test that planner metrics increment after creating runs."""
        # Get initial metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        initial_metrics = response.json()
        initial_total = initial_metrics["planner"]["total_plans_generated"]

        # Create a run
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": "calculate 10 + 5"},
        )
        assert response.status_code == 201

        # Wait for execution
        await asyncio.sleep(0.2)

        # Get updated metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        updated_metrics = response.json()

        # Verify metrics incremented
        assert updated_metrics["planner"]["total_plans_generated"] == initial_total + 1
        assert updated_metrics["planner"]["pattern_plans"] >= 1
        assert updated_metrics["planner"]["avg_latency_ms"] >= 0.0  # Can be very small for fast planning

    @pytest.mark.asyncio
    async def test_pattern_planner_metrics_tracking(self, test_client):
        """Test metrics tracking for pattern-based planner (no LLM)."""
        # Get initial metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        initial_metrics = response.json()
        initial_pattern = initial_metrics["planner"]["pattern_plans"]

        # Create runs that use pattern-based planner
        prompts = [
            "calculate 2 + 2",
            "add a todo Buy milk",
            "list todos",
        ]

        for prompt in prompts:
            response = test_client.post("/api/v1/runs", json={"prompt": prompt})
            assert response.status_code == 201

        # Wait for execution
        await asyncio.sleep(0.3)

        # Get updated metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        updated_metrics = response.json()

        # Verify pattern plans incremented
        assert updated_metrics["planner"]["pattern_plans"] == initial_pattern + len(prompts)
        assert updated_metrics["planner"]["total_plans_generated"] >= initial_pattern + len(prompts)

        # No LLM plans since using pattern-based planner
        assert updated_metrics["planner"]["llm_plans"] == 0
        assert updated_metrics["planner"]["avg_tokens_per_plan"] == 0.0

        # Fallback rate should be 100% for pattern-only
        assert updated_metrics["planner"]["fallback_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_metrics_latency_tracking(self, test_client):
        """Test that planning latency is tracked correctly."""
        # Get initial metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        initial_metrics = response.json()
        initial_metrics["planner"]["avg_latency_ms"]

        # Create multiple runs
        for _ in range(3):
            response = test_client.post(
                "/api/v1/runs",
                json={"prompt": "calculate 5 * 5"},
            )
            assert response.status_code == 201

        # Wait for execution
        await asyncio.sleep(0.3)

        # Get updated metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        updated_metrics = response.json()

        # Verify latency tracking
        new_latency = updated_metrics["planner"]["avg_latency_ms"]
        total_plans = updated_metrics["planner"]["total_plans_generated"]

        # Should have recorded latency (with reasonable bounds)
        assert new_latency >= 0.0  # Can be very small for pattern-based
        assert total_plans >= 3  # Should have at least the 3 runs we created
        assert new_latency < 1000.0  # Less than 1 second (pattern-based is fast)

    @pytest.mark.asyncio
    async def test_metrics_fallback_rate_calculation(self, test_client):
        """Test fallback rate calculation (pattern_plans / total_plans)."""
        # Get initial metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        response.json()

        # Create runs (all pattern-based since no LLM configured)
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": "calculate 100 / 4"},
        )
        assert response.status_code == 201

        # Wait for execution
        await asyncio.sleep(0.2)

        # Get updated metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        updated_metrics = response.json()

        planner = updated_metrics["planner"]

        # Calculate expected fallback rate
        total_plans = planner["total_plans_generated"]
        pattern_plans = planner["pattern_plans"]

        if total_plans > 0:
            expected_fallback_rate = pattern_plans / total_plans
            assert planner["fallback_rate"] == pytest.approx(expected_fallback_rate, rel=0.01)

    @pytest.mark.asyncio
    async def test_metrics_structure_and_types(self, test_client):
        """Test that metrics response has correct structure and types."""
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200

        data = response.json()

        # Verify top-level structure
        assert "timestamp" in data
        assert "runs" in data
        assert "execution" in data
        assert "tools" in data
        assert "planner" in data

        # Verify planner metrics structure
        planner = data["planner"]
        assert isinstance(planner["total_plans_generated"], int)
        assert isinstance(planner["llm_plans"], int)
        assert isinstance(planner["pattern_plans"], int)
        assert isinstance(planner["cached_plans"], int)
        assert isinstance(planner["fallback_rate"], float)
        assert isinstance(planner["avg_tokens_per_plan"], float)
        assert isinstance(planner["avg_latency_ms"], float)
        assert isinstance(planner["cache_hit_rate"], float)

        # Verify value constraints
        assert planner["total_plans_generated"] >= 0
        assert planner["llm_plans"] >= 0
        assert planner["pattern_plans"] >= 0
        assert 0.0 <= planner["fallback_rate"] <= 1.0
        assert planner["avg_tokens_per_plan"] >= 0.0
        assert planner["avg_latency_ms"] >= 0.0
        assert 0.0 <= planner["cache_hit_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_metrics_consistency_across_multiple_runs(self, test_client):
        """Test that metrics remain consistent across multiple API calls."""
        # Create several runs
        run_count = 5
        for i in range(run_count):
            response = test_client.post(
                "/api/v1/runs",
                json={"prompt": f"calculate {i} + {i}"},
            )
            assert response.status_code == 201

        # Wait for execution
        await asyncio.sleep(0.5)

        # Get metrics multiple times - should be consistent
        responses = []
        for _ in range(3):
            response = test_client.get("/api/v1/metrics")
            assert response.status_code == 200
            responses.append(response.json())
            await asyncio.sleep(0.1)

        # All responses should have same planner metrics (no new runs created)
        for i in range(1, len(responses)):
            assert responses[i]["planner"]["total_plans_generated"] == responses[0]["planner"]["total_plans_generated"]
            assert responses[i]["planner"]["pattern_plans"] == responses[0]["planner"]["pattern_plans"]
            assert responses[i]["planner"]["avg_latency_ms"] == responses[0]["planner"]["avg_latency_ms"]

    @pytest.mark.asyncio
    async def test_zero_division_handling_in_metrics(self, test_client):
        """Test that metrics handle zero division gracefully."""
        # Get metrics when there might be zero plans
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200

        data = response.json()
        planner = data["planner"]

        # Should not raise errors even with zero values
        assert isinstance(planner["fallback_rate"], float)
        assert isinstance(planner["avg_tokens_per_plan"], float)
        assert isinstance(planner["avg_latency_ms"], float)

        # All should be valid numbers (not NaN or Inf)
        assert 0.0 <= planner["fallback_rate"] <= 1.0
        assert planner["avg_tokens_per_plan"] >= 0.0
        assert planner["avg_latency_ms"] >= 0.0
