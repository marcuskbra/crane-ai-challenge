"""
Tool registry for managing available tools.

This module provides a centralized registry for tool discovery and access.
"""

from functools import lru_cache

from challenge.tools.base import BaseTool
from challenge.tools.calculator import CalculatorTool
from challenge.tools.todo_store import TodoStoreTool


class ToolRegistry:
    """
    Registry for managing available tools.

    The registry provides a centralized way to discover and access tools
    by name. Tools are initialized once and reused for efficiency.

    This class implements the ToolProvider protocol through structural
    subtyping (duck typing). No explicit inheritance required.

    Example:
        >>> registry = ToolRegistry()
        >>> calculator = registry.get("calculator")
        >>> result = await calculator.execute(expression="2 + 2")

    """

    def __init__(self):
        """Initialize tool registry with available tools."""
        self._tools: dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register all default tools."""
        # Initialize tools
        calculator = CalculatorTool()
        todo_store = TodoStoreTool()

        # Register by name
        self._tools[calculator.metadata.name] = calculator
        self._tools[todo_store.metadata.name] = todo_store

    def get(self, tool_name: str) -> BaseTool | None:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool instance or None if not found

        """
        return self._tools.get(tool_name)

    def list_tools(self) -> list[str]:
        """
        List all available tool names.

        Returns:
            List of registered tool names

        """
        return list(self._tools.keys())

    def get_all_metadata(self) -> dict[str, dict]:
        """
        Get metadata for all registered tools.

        Returns:
            Dictionary mapping tool names to their metadata

        """
        return {name: tool.metadata.model_dump() for name, tool in self._tools.items()}


@lru_cache
def get_tool_registry() -> ToolRegistry:
    """
    Get cached tool registry instance.

    This function uses LRU cache to ensure the registry is created only once
    and reused throughout the application lifecycle.

    Returns:
        Singleton ToolRegistry instance

    """
    return ToolRegistry()
