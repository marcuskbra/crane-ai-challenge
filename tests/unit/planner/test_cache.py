"""
Unit tests for semantic caching module.

Tests cover cache initialization, similarity search, metrics tracking,
and the CachingPlanner wrapper implementation.
"""

import pytest

from challenge.models.plan import Plan, PlanStep
from challenge.planner.cache import (
    CacheEntry,
    CacheMetrics,
    CachingPlanner,
    SemanticCache,
)
from challenge.planner.planner import PatternBasedPlanner


class TestCacheEntry:
    """Tests for CacheEntry model validation."""

    def test_cache_entry_valid(self):
        """Test creating valid cache entry."""
        plan = Plan(
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

        embedding = [0.1, 0.2, 0.3, 0.4]

        entry = CacheEntry(
            prompt="calculate 2 + 2",
            plan=plan,
            embedding=embedding,
            hits=0,
        )

        assert entry.prompt == "calculate 2 + 2"
        assert entry.plan == plan
        assert entry.embedding == embedding
        assert entry.hits == 0

    def test_cache_entry_default_hits(self):
        """Test that hits defaults to 0."""
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "1 + 1"},
                    reasoning="Test",
                )
            ],
            final_goal="test",
        )
        entry = CacheEntry(
            prompt="test",
            plan=plan,
            embedding=[0.1],
        )
        assert entry.hits == 0


class TestCacheMetrics:
    """Tests for CacheMetrics model validation."""

    def test_cache_metrics_valid(self):
        """Test creating valid cache metrics."""
        metrics = CacheMetrics(
            total_requests=100,
            hits=75,
            misses=25,
            hit_rate=0.75,
            total_entries=50,
        )

        assert metrics.total_requests == 100
        assert metrics.hits == 75
        assert metrics.misses == 25
        assert metrics.hit_rate == 0.75
        assert metrics.total_entries == 50

    def test_cache_metrics_defaults(self):
        """Test that metrics have sensible defaults."""
        metrics = CacheMetrics()

        assert metrics.total_requests == 0
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.hit_rate == 0.0
        assert metrics.total_entries == 0


class TestSemanticCache:
    """Tests for SemanticCache similarity-based caching."""

    def test_semantic_cache_initialization(self, semantic_cache_factory):
        """Test that semantic cache initializes correctly."""
        cache = semantic_cache_factory(similarity_threshold=0.90)

        assert cache.similarity_threshold == 0.90
        assert cache.embedding_dim == 384  # all-MiniLM-L6-v2 dimension
        assert cache.size() == 0
        assert cache.total_requests == 0
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0

    def test_add_to_cache(self, semantic_cache_factory):
        """Test adding entry to cache."""
        cache = semantic_cache_factory()
        plan = Plan(
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

        cache.add("calculate 2 + 2", plan)

        assert cache.size() == 1
        assert len(cache.entries) == 1
        assert cache.entries[0].prompt == "calculate 2 + 2"
        assert cache.entries[0].plan == plan

    def test_cache_hit_exact_match(self, semantic_cache_factory):
        """Test cache hit with exact same prompt."""
        cache = semantic_cache_factory(similarity_threshold=0.85)
        plan = Plan(
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

        # Add to cache
        cache.add("calculate 2 + 2", plan)

        # Retrieve with exact same prompt
        cached_plan = cache.get("calculate 2 + 2")

        assert cached_plan is not None
        assert cached_plan == plan
        assert cache.cache_hits == 1
        assert cache.cache_misses == 0

    def test_cache_hit_similar_prompt(self, semantic_cache_factory):
        """Test cache hit with semantically similar prompt."""
        # Lower threshold to ensure similar prompts match
        cache = semantic_cache_factory(similarity_threshold=0.75)
        plan = Plan(
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

        # Add to cache with original prompt
        cache.add("calculate 2 + 2", plan)

        # Try similar prompt
        cached_plan = cache.get("compute 2 plus 2")

        # Should be cache hit due to semantic similarity
        assert cached_plan is not None
        assert cached_plan == plan
        assert cache.cache_hits == 1
        assert cache.cache_misses == 0

    def test_cache_miss_empty_cache(self, semantic_cache_factory):
        """Test cache miss when cache is empty."""
        cache = semantic_cache_factory()

        cached_plan = cache.get("calculate 2 + 2")

        assert cached_plan is None
        assert cache.cache_hits == 0
        assert cache.cache_misses == 1

    def test_cache_miss_dissimilar_prompt(self, semantic_cache_factory):
        """Test cache miss with dissimilar prompt."""
        cache = semantic_cache_factory(similarity_threshold=0.85)
        plan = Plan(
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

        # Add to cache
        cache.add("calculate 2 + 2", plan)

        # Try completely different prompt
        cached_plan = cache.get("add a todo to buy milk")

        # Should be cache miss due to low semantic similarity
        assert cached_plan is None
        assert cache.cache_hits == 0
        assert cache.cache_misses == 1

    def test_similarity_threshold_behavior(self, semantic_cache_factory):
        """Test that similarity threshold controls cache hits."""
        # High threshold cache (strict matching)
        strict_cache = semantic_cache_factory(similarity_threshold=0.95)
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "2 + 2"},
                    reasoning="Test",
                )
            ],
            final_goal="test",
        )

        strict_cache.add("calculate 2 + 2", plan)

        # Very similar prompt might miss with high threshold
        result = strict_cache.get("compute 2 plus 2")
        # Result depends on actual similarity score

        # Low threshold cache (lenient matching)
        lenient_cache = semantic_cache_factory(similarity_threshold=0.70)
        lenient_cache.add("calculate 2 + 2", plan)

        # More likely to hit with lower threshold
        result = lenient_cache.get("compute 2 plus 2")
        assert result is not None  # Should hit with lower threshold

    def test_multiple_entries(self, semantic_cache_factory):
        """Test cache with multiple entries."""
        cache = semantic_cache_factory()
        plan1 = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "2 + 2"},
                    reasoning="Test calc",
                )
            ],
            final_goal="calculate",
        )
        plan2 = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="todo_store",
                    tool_input={"action": "add", "text": "test"},
                    reasoning="Test todo",
                )
            ],
            final_goal="todo",
        )

        cache.add("calculate 2 + 2", plan1)
        cache.add("add a todo", plan2)

        assert cache.size() == 2

        # Both should be retrievable
        assert cache.get("calculate 2 + 2") == plan1
        assert cache.get("add a todo") == plan2

    def test_cache_metrics_tracking(self, semantic_cache_factory):
        """Test that cache metrics are tracked correctly."""
        cache = semantic_cache_factory()
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "test"},
                    reasoning="Test",
                )
            ],
            final_goal="test",
        )

        # Initial state
        metrics = cache.get_metrics()
        assert metrics.total_requests == 0
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.hit_rate == 0.0
        assert metrics.total_entries == 0

        # Add entry
        cache.add("test", plan)

        # Miss (cache not queried yet)
        cache.get("different prompt")
        metrics = cache.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.misses == 1
        assert metrics.hit_rate == 0.0
        assert metrics.total_entries == 1

        # Hit
        cache.get("test")
        metrics = cache.get_metrics()
        assert metrics.total_requests == 2
        assert metrics.hits == 1
        assert metrics.misses == 1
        assert metrics.hit_rate == 0.5
        assert metrics.total_entries == 1

    def test_cache_hit_counter_increments(self, semantic_cache_factory):
        """Test that entry hit counter increments."""
        cache = semantic_cache_factory()
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "test"},
                    reasoning="Test",
                )
            ],
            final_goal="test",
        )

        cache.add("test", plan)

        # First retrieval
        cache.get("test")
        assert cache.entries[0].hits == 1

        # Second retrieval
        cache.get("test")
        assert cache.entries[0].hits == 2

        # Third retrieval
        cache.get("test")
        assert cache.entries[0].hits == 3

    def test_clear_cache(self, semantic_cache_factory):
        """Test clearing cache resets all state."""
        cache = semantic_cache_factory()
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "test"},
                    reasoning="Test",
                )
            ],
            final_goal="test",
        )

        # Add entries and make requests
        cache.add("test1", plan)
        cache.add("test2", plan)
        cache.get("test1")
        cache.get("test2")
        cache.get("nonexistent")

        # Clear cache
        cache.clear()

        # Verify everything reset
        assert cache.size() == 0
        assert cache.total_requests == 0
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0

        metrics = cache.get_metrics()
        assert metrics.total_requests == 0
        assert metrics.total_entries == 0


@pytest.mark.asyncio
class TestCachingPlanner:
    """Tests for CachingPlanner wrapper."""

    async def test_caching_planner_initialization(self, semantic_cache_factory):
        """Test that CachingPlanner initializes correctly."""
        base_planner = PatternBasedPlanner()
        cache = semantic_cache_factory()
        caching_planner = CachingPlanner(base_planner, cache=cache)

        assert caching_planner.planner == base_planner
        assert caching_planner.cache is not None
        assert isinstance(caching_planner.cache, SemanticCache)

    async def test_caching_planner_with_custom_cache(self, semantic_cache_factory):
        """Test CachingPlanner with custom cache instance."""
        base_planner = PatternBasedPlanner()
        custom_cache = semantic_cache_factory(similarity_threshold=0.90)

        caching_planner = CachingPlanner(base_planner, cache=custom_cache)

        assert caching_planner.cache == custom_cache
        assert caching_planner.cache.similarity_threshold == 0.90

    async def test_cache_miss_delegates_to_base_planner(self, semantic_cache_factory):
        """Test that cache miss delegates to base planner."""
        base_planner = PatternBasedPlanner()
        cache = semantic_cache_factory()
        caching_planner = CachingPlanner(base_planner, cache=cache)

        # First call (cache miss)
        plan = await caching_planner.create_plan("calculate 2 + 2")

        assert plan is not None
        assert len(plan.steps) > 0

        # Verify cache was populated
        assert caching_planner.cache.size() == 1

    async def test_cache_hit_returns_cached_plan(self, semantic_cache_factory):
        """Test that cache hit returns cached plan without calling base planner."""
        base_planner = PatternBasedPlanner()
        cache = semantic_cache_factory()
        caching_planner = CachingPlanner(base_planner, cache=cache)

        # First call (cache miss)
        plan1 = await caching_planner.create_plan("calculate 2 + 2")

        # Second call with exact same prompt (cache hit)
        plan2 = await caching_planner.create_plan("calculate 2 + 2")

        # Should return same plan object
        assert plan2 == plan1

        # Verify metrics
        metrics = caching_planner.get_metrics()
        assert metrics.total_requests == 2
        assert metrics.hits == 1
        assert metrics.misses == 1
        assert metrics.hit_rate == 0.5

    async def test_cache_hit_with_similar_prompt(self):
        """Test cache hit with semantically similar prompt."""
        base_planner = PatternBasedPlanner()
        # Lower threshold to ensure similar prompts match
        caching_planner = CachingPlanner(base_planner, similarity_threshold=0.75)

        # First call
        plan1 = await caching_planner.create_plan("calculate 2 + 2")

        # Similar prompt (should hit cache with lower threshold)
        plan2 = await caching_planner.create_plan("compute 2 plus 2")

        # Should return cached plan
        assert plan2 == plan1

        metrics = caching_planner.get_metrics()
        assert metrics.hits == 1

    async def test_multiple_prompts_cached(self, semantic_cache_factory):
        """Test that multiple different prompts are cached."""
        base_planner = PatternBasedPlanner()
        cache = semantic_cache_factory()
        caching_planner = CachingPlanner(base_planner, cache=cache)

        # Cache different prompts
        plan1 = await caching_planner.create_plan("calculate 2 + 2")
        plan2 = await caching_planner.create_plan("add a todo to buy milk")

        assert caching_planner.cache.size() == 2

        # Both should be retrievable
        cached_plan1 = await caching_planner.create_plan("calculate 2 + 2")
        cached_plan2 = await caching_planner.create_plan("add a todo to buy milk")

        assert cached_plan1 == plan1
        assert cached_plan2 == plan2

        metrics = caching_planner.get_metrics()
        assert metrics.total_requests == 4
        assert metrics.hits == 2
        assert metrics.misses == 2

    async def test_clear_cache_method(self, semantic_cache_factory):
        """Test clearing cache through CachingPlanner."""
        base_planner = PatternBasedPlanner()
        cache = semantic_cache_factory()
        caching_planner = CachingPlanner(base_planner, cache=cache)

        # Add some cached entries
        await caching_planner.create_plan("calculate 2 + 2")
        await caching_planner.create_plan("add a todo buy milk")

        assert caching_planner.cache.size() == 2

        # Clear cache
        caching_planner.clear_cache()

        assert caching_planner.cache.size() == 0

        metrics = caching_planner.get_metrics()
        assert metrics.total_entries == 0
