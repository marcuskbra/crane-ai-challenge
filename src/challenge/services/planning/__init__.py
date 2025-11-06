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

from challenge.services.planning.llm_planner import LLMPlanner
from challenge.services.planning.metrics_tracker import MetricsTracker
from challenge.services.planning.planner import PatternBasedPlanner
from challenge.services.planning.protocol import Planner

__all__ = [
    "LLMPlanner",  # LLM-powered implementation
    "MetricsTracker",  # Planner performance metrics
    "PatternBasedPlanner",  # Default implementation
    "Planner",  # Protocol for type hints
]
