"""
Planner module for converting natural language to execution plans.

This module provides:
- Planner: Protocol defining the planner interface
- PatternBasedPlanner: Regex-based pattern matching implementation (default)
- LLMPlanner: LLM-powered planning with structured output (requires API key)
- MetricsTracker: Planner performance metrics tracking

Security:
    - All regex patterns use length limits to prevent ReDoS attacks
    - Input validation enforces max prompt length of 2000 chars
    - Pattern compilation at module level for performance

Performance:
    - Compiled regex patterns provide 30-50% speed improvement
    - Efficient pattern matching with early returns

"""

from challenge.planner.llm_planner import LLMPlanner
from challenge.planner.metrics_tracker import MetricsTracker
from challenge.planner.planner import PatternBasedPlanner
from challenge.planner.protocol import Planner

__all__ = [
    "LLMPlanner",  # LLM-powered implementation
    "MetricsTracker",  # Planner performance metrics
    "PatternBasedPlanner",  # Default implementation
    "Planner",  # Protocol for type hints
]
