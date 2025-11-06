"""
Base tool interface for AI Agent Runtime.

This module defines the abstract base class for all tools and standard
result types for tool execution.

Following TYPING_GUIDE.md:
- Uses Generic[TOutput] for type-safe tool outputs
- Eliminates Any in favor of generic type parameters
- Enables compile-time type checking for tool results
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

# Generic type parameter for tool output types
# Allows tools to specify their exact output type (e.g., float, dict, list)
TOutput = TypeVar("TOutput")


class ToolResult(BaseModel, Generic[TOutput]):
    """
    Generic result format for tool execution.

    Type parameter:
        TOutput: The type of the output value (e.g., float for calculator, list for todo_store)

    Attributes:
        success: Whether the tool execution succeeded
        output: The strongly-typed result value if successful
        error: Error message if execution failed
        metadata: Additional metadata about the execution

    """

    success: bool = Field(..., description="Whether execution succeeded")
    output: TOutput | None = Field(None, description="Strongly-typed result value if successful")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class ToolMetadata(BaseModel):
    """
    Tool capability description.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of tool capabilities
        input_schema: JSON schema describing expected inputs

    """

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Human-readable description")
    input_schema: dict[str, Any] = Field(..., description="JSON schema for inputs")


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    All tool implementations must inherit from this class and implement
    the metadata property and execute method.
    """

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """
        Get tool metadata describing capabilities.

        Returns:
            ToolMetadata describing the tool

        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with validated inputs.

        Args:
            **kwargs: Tool-specific input parameters

        Returns:
            ToolResult with execution outcome

        """
        pass
