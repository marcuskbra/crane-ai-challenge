"""
Strongly-typed tool input and output definitions.

This module provides discriminated unions for tool inputs and outputs,
ensuring type safety throughout the execution pipeline per TYPING_GUIDE.md.

All tool inputs and outputs use Pydantic models instead of Dict[str, Any],
enabling compile-time type checking and IDE autocomplete.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Calculator Tool Types
# ============================================================================


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    expression: str = Field(..., min_length=1, description="Math expression to evaluate")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class CalculatorOutput(BaseModel):
    """Output from calculator tool."""

    result: float = Field(..., description="Calculation result")
    expression: str = Field(..., description="Original expression")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


# ============================================================================
# TodoStore Tool Types
# ============================================================================


class TodoItem(BaseModel):
    """A todo item with metadata."""

    id: str = Field(..., description="Unique todo ID")
    text: str = Field(..., min_length=1, description="Todo text content")
    completed: bool = Field(..., description="Completion status")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    completed_at: str | None = Field(None, description="Completion timestamp (ISO format)")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


# TodoStore Inputs (discriminated by action field)


class TodoAddInput(BaseModel):
    """Input for todo_store add action."""

    action: Literal["add"] = "add"
    text: str = Field(..., min_length=1, description="Todo text content")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class TodoListInput(BaseModel):
    """Input for todo_store list action."""

    action: Literal["list"] = "list"

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class TodoGetInput(BaseModel):
    """Input for todo_store get action."""

    action: Literal["get"] = "get"
    todo_id: str = Field(..., min_length=1, description="Todo ID to retrieve")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class TodoCompleteInput(BaseModel):
    """Input for todo_store complete action."""

    action: Literal["complete"] = "complete"
    todo_id: str = Field(..., min_length=1, description="Todo ID to complete")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class TodoDeleteInput(BaseModel):
    """Input for todo_store delete action."""

    action: Literal["delete"] = "delete"
    todo_id: str = Field(..., min_length=1, description="Todo ID to delete")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


# TodoStore Outputs (discriminated by action field)


class TodoAddOutput(BaseModel):
    """Output from todo_store add action."""

    todo: TodoItem = Field(..., description="Created todo item")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class TodoListOutput(BaseModel):
    """Output from todo_store list action."""

    todos: list[TodoItem] = Field(..., description="List of all todos")
    total_count: int = Field(..., ge=0, description="Total number of todos")
    completed_count: int = Field(..., ge=0, description="Number of completed todos")
    pending_count: int = Field(..., ge=0, description="Number of pending todos")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class TodoGetOutput(BaseModel):
    """Output from todo_store get action."""

    todo: TodoItem = Field(..., description="Retrieved todo item")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class TodoCompleteOutput(BaseModel):
    """Output from todo_store complete action."""

    todo: TodoItem = Field(..., description="Completed todo item")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class TodoDeleteOutput(BaseModel):
    """Output from todo_store delete action."""

    todo: TodoItem = Field(..., description="Deleted todo item")
    remaining_count: int = Field(..., ge=0, description="Number of remaining todos")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


# ============================================================================
# Discriminated Unions
# ============================================================================


# All possible tool inputs (discriminated by structure)
ToolInput = CalculatorInput | TodoAddInput | TodoListInput | TodoGetInput | TodoCompleteInput | TodoDeleteInput

# All possible tool outputs (discriminated by structure)
ToolOutput = CalculatorOutput | TodoAddOutput | TodoListOutput | TodoGetOutput | TodoCompleteOutput | TodoDeleteOutput
