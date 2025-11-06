"""
Planner protocol defining the interface for all planner implementations.

This module uses Protocol for structural subtyping, allowing any class
with a compatible create_plan method to be used as a planner without
requiring explicit inheritance.
"""

from typing import Protocol

from challenge.domain.models.plan import Plan


class Planner(Protocol):
    """
    Protocol defining the planner interface.

    Any class implementing a create_plan method with this signature
    can be used as a planner, regardless of inheritance hierarchy.

    This follows the Dependency Inversion Principle (SOLID):
    - Orchestrator depends on the Planner abstraction
    - Concrete planners implement the interface independently
    - New planners can be added without modifying orchestrator

    Note:
        Both sync and async implementations are supported.
        The orchestrator handles async planners automatically.

    """

    async def create_plan(self, prompt: str) -> Plan:
        """
        Create execution plan from natural language prompt.

        Args:
            prompt: Natural language task description

        Returns:
            Plan with ordered execution steps

        Raises:
            ValueError: If prompt is invalid or cannot be parsed

        """
        ...

    @property
    def last_token_count(self) -> int | None:
        """
        Token count from last planning operation.

        Returns:
            Number of tokens used in last create_plan call, or None if
            token counting is not supported by this planner implementation.

        Note:
            This property enables type-safe access to token metrics without
            runtime attribute lookup (getattr). Planners that don't track
            tokens should return None.

        """
        ...
