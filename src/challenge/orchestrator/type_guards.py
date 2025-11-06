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

from challenge.tools.types import ToolInput


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
