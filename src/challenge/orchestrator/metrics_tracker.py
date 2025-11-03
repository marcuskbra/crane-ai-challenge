"""
Metrics tracking for planner performance.

This module provides centralized metrics collection for planner operations,
tracking LLM vs pattern-based planning, token usage, and latency.
"""

import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks planner performance metrics.

    Collects and aggregates metrics about plan generation, including:
    - Total plans created
    - LLM-based vs pattern-based plans
    - Token usage for LLM plans
    - Planning latency

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.record_plan(latency_ms=150.0, token_count=500)
        >>> stats = tracker.get_stats()
        >>> print(f"Total plans: {stats['total_plans']}")

    """

    def __init__(self):
        """Initialize metrics tracker with zero counters."""
        self.total_plans = 0
        self.llm_plans = 0
        self.pattern_plans = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

    def record_plan(self, latency_ms: float, token_count: int | None = None) -> None:
        """
        Record a plan generation event.

        Args:
            latency_ms: Planning latency in milliseconds
            token_count: Number of tokens used (LLM plans only).
                        None or 0 indicates pattern-based planning.

        Example:
            >>> # Record LLM plan
            >>> tracker.record_plan(latency_ms=250.0, token_count=450)

            >>> # Record pattern-based plan
            >>> tracker.record_plan(latency_ms=10.0, token_count=None)

        """
        self.total_plans += 1
        self.total_latency_ms += latency_ms

        if token_count and token_count > 0:
            # LLM plan (successful API call)
            self.llm_plans += 1
            self.total_tokens += token_count
            logger.debug(f"LLM plan generated: {token_count} tokens, {latency_ms:.1f}ms")
        else:
            # Pattern-based plan (no LLM or LLM fallback)
            self.pattern_plans += 1
            logger.debug(f"Pattern plan generated: {latency_ms:.1f}ms")

    def get_stats(self) -> dict[str, int | float]:
        """
        Get current metrics snapshot.

        Returns:
            Dictionary with metrics:
                - total_plans: Total number of plans created
                - llm_plans: Plans generated using LLM
                - pattern_plans: Plans generated using patterns
                - total_tokens: Total tokens consumed by LLM
                - total_latency_ms: Cumulative planning latency

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.record_plan(150.0, 500)
            >>> stats = tracker.get_stats()
            >>> assert stats["total_plans"] == 1
            >>> assert stats["llm_plans"] == 1
            >>> assert stats["total_tokens"] == 500

        """
        return {
            "total_plans": self.total_plans,
            "llm_plans": self.llm_plans,
            "pattern_plans": self.pattern_plans,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
        }

    def reset(self) -> None:
        """
        Reset all metrics to zero.

        Useful for testing or periodic metric resets.

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.record_plan(100.0, 200)
            >>> tracker.reset()
            >>> assert tracker.total_plans == 0

        """
        self.total_plans = 0
        self.llm_plans = 0
        self.pattern_plans = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
