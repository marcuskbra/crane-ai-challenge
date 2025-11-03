"""
Protocol definitions for orchestrator module.

This module defines structural protocols for dependency injection,
enabling flexible implementation while maintaining type safety.
"""

from typing import Protocol, runtime_checkable

from challenge.tools.base import BaseTool


@runtime_checkable
class ToolProvider(Protocol):
    """
    Protocol for tool registry implementations.

    This protocol defines the interface for accessing tools by name.
    Both dict[str, BaseTool] and ToolRegistry implement this protocol
    through structural subtyping (duck typing with type safety).

    The @runtime_checkable decorator enables isinstance() checks,
    though this is rarely needed due to Protocol's structural typing.

    Example:
        >>> class MyRegistry:
        ...     def get(self, tool_name: str) -> BaseTool | None:
        ...         return self._tools.get(tool_name)
        >>> # MyRegistry is a ToolProvider (no explicit inheritance needed)

    """

    def get(self, tool_name: str) -> BaseTool | None:
        """
        Retrieve a tool by name.

        Args:
            tool_name: Unique identifier for the tool

        Returns:
            BaseTool instance if found, None otherwise

        Example:
            >>> tool = provider.get("calculator")
            >>> if tool:
            ...     result = await tool.execute(operation="add", a=2, b=3)

        """
        ...
