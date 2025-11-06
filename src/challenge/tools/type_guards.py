"""
Type guards for orchestrator type checking.

This module provides TypeGuard functions for runtime type checking
that are compliant with TYPING_GUIDE.md rules (lines 127-134).

Per TYPING_GUIDE.md:
- ❌ BANNED: hasattr(obj, 'model_dump') for type checking
- ✅ USE: Proper TypeGuard functions with isinstance checks
"""

from typing import Any, TypeGuard

from pydantic import BaseModel

from challenge.tools.types import (
    CalculatorOutput,
    TodoAddOutput,
    TodoCompleteOutput,
    TodoDeleteOutput,
    TodoGetOutput,
    TodoListOutput,
    ToolInput,
    ToolOutput,
)


def is_tool_input_model(obj: Any) -> TypeGuard[ToolInput]:
    """
    Type guard to check if object is a ToolInput Pydantic model.

    This replaces the banned pattern: hasattr(obj, 'model_dump')

    Args:
        obj: Object to check

    Returns:
        True if obj is a ToolInput model, False otherwise

    """
    # Check if object is a Pydantic BaseModel instance
    # This is type-safe because ToolInput is a union of Pydantic models
    return isinstance(obj, BaseModel)


def is_pydantic_model(obj: Any) -> TypeGuard[BaseModel]:
    """
    Type guard to check if object is any Pydantic model.

    This is a general-purpose type guard for Pydantic model detection.

    Args:
        obj: Object to check

    Returns:
        True if obj is a Pydantic BaseModel, False otherwise

    """
    return isinstance(obj, BaseModel)


def is_todo_list_output(output: ToolOutput) -> TypeGuard[TodoListOutput]:
    """
    Type guard for TodoListOutput.

    Args:
        output: Tool output to check

    Returns:
        True if output is TodoListOutput, False otherwise

    """
    return isinstance(output, TodoListOutput)


def is_todo_single_output(
    output: ToolOutput,
) -> TypeGuard[TodoAddOutput | TodoGetOutput | TodoCompleteOutput | TodoDeleteOutput]:
    """
    Type guard for single todo outputs.

    Matches TodoAddOutput, TodoGetOutput, TodoCompleteOutput, or TodoDeleteOutput.

    Args:
        output: Tool output to check

    Returns:
        True if output is any single todo output type, False otherwise

    """
    return isinstance(output, (TodoAddOutput, TodoGetOutput, TodoCompleteOutput, TodoDeleteOutput))


def is_calculator_output(output: ToolOutput) -> TypeGuard[CalculatorOutput]:
    """
    Type guard for CalculatorOutput.

    Args:
        output: Tool output to check

    Returns:
        True if output is CalculatorOutput, False otherwise

    """
    return isinstance(output, CalculatorOutput)
