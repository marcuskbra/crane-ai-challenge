"""
Metrics tracking for planner performance.

This module provides centralized metrics collection for planner operations,
tracking LLM vs pattern-based planning, token usage, and latency.
"""

import logging

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class PlannerStats(BaseModel):
    """
    Planner performance statistics model.

    Provides type-safe access to planner metrics including plan counts,
    token usage, and latency measurements.
    """

    total_plans: int = Field(..., description="Total number of plans created")
    llm_plans: int = Field(..., description="Plans generated using LLM")
    pattern_plans: int = Field(..., description="Plans generated using patterns")
    total_tokens: int = Field(..., description="Total tokens consumed by LLM")
    total_latency_ms: float = Field(..., description="Cumulative planning latency in milliseconds")

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid",
    )


class MetricsTracker:
    """
    Tracks planner performance metrics.

    Collects and aggregates metrics about plan generation, including:
    - Total plans created
    - LLM-based vs pattern-based plans
    - Token usage for LLM plans
    - Planning latency

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

    def get_stats(self) -> PlannerStats:
        """
        Get current metrics snapshot.

        Returns:
            PlannerStats model with metrics:
                - total_plans: Total number of plans created
                - llm_plans: Plans generated using LLM
                - pattern_plans: Plans generated using patterns
                - total_tokens: Total tokens consumed by LLM
                - total_latency_ms: Cumulative planning latency

        """
        return PlannerStats(
            total_plans=self.total_plans,
            llm_plans=self.llm_plans,
            pattern_plans=self.pattern_plans,
            total_tokens=self.total_tokens,
            total_latency_ms=self.total_latency_ms,
        )

    def reset(self) -> None:
        """
        Reset all metrics to zero.

        Useful for testing or periodic metric resets.

        """
        self.total_plans = 0
        self.llm_plans = 0
        self.pattern_plans = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
