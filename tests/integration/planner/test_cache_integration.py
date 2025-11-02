"""
Integration tests for semantic caching with planner workflows.

Tests cover realistic usage scenarios with the orchestrator and different
planner implementations to verify caching behavior in production-like conditions.
"""

import pytest

from challenge.models.plan import Plan, PlanStep
from challenge.planner.cache import CachingPlanner
from challenge.planner.llm_planner import LLMPlanner
from challenge.planner.planner import PatternBasedPlanner


@pytest.mark.integration
@pytest.mark.asyncio
class TestCacheIntegration:
    """Integration tests for cache with real planner workflows."""

    async def test_pattern_planner_caching_workflow(self, semantic_cache_factory):
        """Test caching with PatternBasedPlanner in realistic workflow."""
        base_planner = PatternBasedPlanner()
        cache = semantic_cache_factory(similarity_threshold=0.80)
        caching_planner = CachingPlanner(base_planner, cache=cache)

        # Scenario: User repeats similar calculation requests
        plan1 = await caching_planner.create_plan("calculate 10 + 5")
        plan2 = await caching_planner.create_plan("calculate 10 + 5")  # Exact match
        plan3 = await caching_planner.create_plan("compute 10 plus 5")  # Similar

        # First is cache miss, second and third should be hits
        assert plan1 is not None
        assert plan2 == plan1  # Exact match cache hit
        assert plan3 == plan1  # Semantic similarity cache hit

        metrics = caching_planner.get_metrics()
        assert metrics.total_requests == 3
        assert metrics.hits == 2
        assert metrics.misses == 1
        assert metrics.hit_rate == pytest.approx(0.666, rel=0.01)
        assert metrics.total_entries == 1

    async def test_multi_user_simulation(self):
        """Test cache behavior with multiple users making similar requests."""
        base_planner = PatternBasedPlanner()
        caching_planner = CachingPlanner(base_planner, similarity_threshold=0.80)

        # Simulate multiple users with overlapping requests
        users_requests = [
            "calculate 2 + 2",
            "add a todo buy groceries",
            "calculate 2 + 2",  # Same as user 1
            "add a todo buy milk",
            "compute 2 plus 2",  # Similar to user 1
            "show my todos",
            "calculate 2 + 2",  # Same as user 1
        ]

        plans = []
        for request in users_requests:
            plan = await caching_planner.create_plan(request)
            plans.append(plan)

        # Verify cache effectiveness
        metrics = caching_planner.get_metrics()
        assert metrics.total_requests == 7
        assert metrics.hits >= 2  # At least exact matches should hit
        assert metrics.total_entries <= 4  # Unique request types

        # Verify that identical requests return same plan
        assert plans[0] == plans[2]  # Same "calculate 2 + 2"
        assert plans[0] == plans[6]  # Same "calculate 2 + 2"

    async def test_cache_performance_with_todo_workflow(self):
        """Test cache with realistic todo management workflow."""
        base_planner = PatternBasedPlanner()
        caching_planner = CachingPlanner(base_planner, similarity_threshold=0.80)

        # Simulate user adding multiple todos and checking list
        workflow_requests = [
            "add a todo buy milk",
            "add a todo call dentist",
            "show my todos",
            "add a todo finish report",
            "list all my todos",  # Similar to "show my todos"
            "show my todos",  # Exact match
        ]

        for request in workflow_requests:
            await caching_planner.create_plan(request)

        metrics = caching_planner.get_metrics()
        assert metrics.total_requests == 6
        # "list all my todos" and second "show my todos" should be cache hits
        assert metrics.hits >= 2
        assert metrics.hit_rate > 0.3

    async def test_cache_threshold_impact(self):
        """Test how similarity threshold affects cache hit rate."""
        base_planner = PatternBasedPlanner()

        # Test with strict threshold
        strict_cache = CachingPlanner(base_planner, similarity_threshold=0.95)
        await strict_cache.create_plan("calculate 2 + 2")
        await strict_cache.create_plan("compute 2 plus 2")  # Likely miss with strict threshold

        strict_metrics = strict_cache.get_metrics()

        # Test with lenient threshold
        lenient_cache = CachingPlanner(base_planner, similarity_threshold=0.70)
        await lenient_cache.create_plan("calculate 2 + 2")
        await lenient_cache.create_plan("compute 2 plus 2")  # Likely hit with lenient threshold

        lenient_metrics = lenient_cache.get_metrics()

        # Lenient cache should have higher hit rate
        assert lenient_metrics.hit_rate >= strict_metrics.hit_rate

    async def test_cache_with_llm_planner_mock(self, semantic_cache_factory):
        """Test caching with LLMPlanner (mock to avoid API calls)."""
        # Create LLM planner with dummy API key (won't be called due to cache)
        llm_planner = LLMPlanner(api_key="dummy-key-for-testing")
        cache = semantic_cache_factory(similarity_threshold=0.80)
        caching_planner = CachingPlanner(llm_planner, cache=cache)

        # Pre-populate cache with a plan to avoid actual LLM call
        cached_plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "2 + 2"},
                    reasoning="Add numbers",
                )
            ],
            final_goal="calculate 2 + 2",
        )
        cache.add("calculate 2 + 2", cached_plan)

        # Request should hit cache without calling LLM
        result = await caching_planner.create_plan("calculate 2 + 2")

        assert result == cached_plan
        metrics = caching_planner.get_metrics()
        assert metrics.hits == 1
        assert metrics.misses == 0

    async def test_cache_isolation_between_instances(self):
        """Test that different cache instances are isolated."""
        base_planner = PatternBasedPlanner()

        # Create two separate caching planners
        cache1 = CachingPlanner(base_planner)
        cache2 = CachingPlanner(base_planner)

        # Add to first cache
        plan1 = await cache1.create_plan("calculate 5 + 5")

        # Second cache should not have this entry
        plan2 = await cache2.create_plan("calculate 5 + 5")

        # Plans should be equal but from different cache instances
        assert plan1 is not None
        assert plan2 is not None
        assert plan1 == plan2

        # Both should work but have independent metrics
        metrics1 = cache1.get_metrics()
        metrics2 = cache2.get_metrics()

        assert metrics1.total_entries == 1
        assert metrics2.total_entries == 1
        assert metrics1.hits == 0  # First request was a miss
        assert metrics2.hits == 0  # Independent cache, also a miss

    async def test_cache_clear_in_workflow(self):
        """Test cache clearing during workflow execution."""
        base_planner = PatternBasedPlanner()
        caching_planner = CachingPlanner(base_planner)

        # Build up cache
        await caching_planner.create_plan("calculate 1 + 1")
        await caching_planner.create_plan("calculate 2 + 2")
        await caching_planner.create_plan("add a todo test")

        assert caching_planner.cache.size() == 3

        # Clear cache mid-workflow
        caching_planner.clear_cache()

        assert caching_planner.cache.size() == 0
        metrics = caching_planner.get_metrics()
        assert metrics.total_entries == 0
        assert metrics.total_requests == 0

        # Continue workflow - should rebuild cache
        await caching_planner.create_plan("calculate 1 + 1")

        assert caching_planner.cache.size() == 1

    async def test_high_volume_caching(self):
        """Test cache behavior with high volume of requests."""
        base_planner = PatternBasedPlanner()
        caching_planner = CachingPlanner(base_planner, similarity_threshold=0.85)

        # Generate 50 requests with some repetition
        requests = [f"calculate {i} + {i}" for i in range(10)] * 5  # 50 total requests, 10 unique

        for request in requests:
            await caching_planner.create_plan(request)

        metrics = caching_planner.get_metrics()
        assert metrics.total_requests == 50
        assert metrics.total_entries == 10  # 10 unique patterns
        # First 10 are misses, remaining 40 should be hits
        assert metrics.hits == 40
        assert metrics.misses == 10
        assert metrics.hit_rate == 0.8

    async def test_cache_hit_counter_in_workflow(self):
        """Test that cache hit counters increment correctly in realistic workflow."""
        base_planner = PatternBasedPlanner()
        caching_planner = CachingPlanner(base_planner)

        # Create a popular query that gets requested multiple times
        popular_query = "calculate 42 + 8"

        # Request it 5 times
        for _ in range(5):
            await caching_planner.create_plan(popular_query)

        # Check entry hit counter
        cache_entry = caching_planner.cache.entries[0]
        assert cache_entry.prompt == popular_query
        assert cache_entry.hits == 4  # First request doesn't count as a hit

        metrics = caching_planner.get_metrics()
        assert metrics.hits == 4
        assert metrics.misses == 1
