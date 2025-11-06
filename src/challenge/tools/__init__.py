"""
Tool system for AI Agent Runtime.

This module provides the base tool interface and tool implementations.
"""

from challenge.tools.base import BaseTool, ToolMetadata, ToolResult
from challenge.tools.calculator import CalculatorTool
from challenge.tools.registry import ToolRegistry, get_tool_registry
from challenge.tools.todo_store import TodoStoreTool
from challenge.tools.type_guards import (
    is_calculator_output,
    is_pydantic_model,
    is_todo_list_output,
    is_todo_single_output,
    is_tool_input_model,
)

__all__ = [
    "BaseTool",
    "CalculatorTool",
    "TodoStoreTool",
    "ToolMetadata",
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
