"""Tool system infrastructure for AI Agent Runtime."""

from challenge.infrastructure.tools.base import BaseTool, ToolMetadata, ToolResult
from challenge.infrastructure.tools.implementations.calculator import CalculatorTool
from challenge.infrastructure.tools.implementations.todo_store import TodoStoreTool
from challenge.infrastructure.tools.registry import ToolRegistry, get_tool_registry
from challenge.infrastructure.tools.type_guards import (
    is_calculator_output,
    is_pydantic_model,
    is_todo_list_output,
    is_todo_single_output,
    is_tool_input_model,
)

__all__ = [
    # Base classes
    "BaseTool",
    # Implementations
    "CalculatorTool",
    "TodoStoreTool",
    "ToolMetadata",
    # Registry
    "ToolRegistry",
    "ToolResult",
    "get_tool_registry",
    # Type guards
    "is_calculator_output",
    "is_pydantic_model",
    "is_todo_list_output",
    "is_todo_single_output",
    "is_tool_input_model",
]
