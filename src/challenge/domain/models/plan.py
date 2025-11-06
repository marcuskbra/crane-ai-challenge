"""
Plan models for structured task execution.

This module defines the data structures for representing execution plans
generated from natural language prompts.
"""

from pydantic import BaseModel, ConfigDict, Field

from challenge.domain.types import ToolInput


class PlanStep(BaseModel):
    """
    A single step in an execution plan.

    Attributes:
        step_number: Sequential step number (1-indexed)
        tool_name: Name of the tool to execute
        tool_input: Type-safe discriminated union of tool inputs
        reasoning: Human-readable explanation of why this step is needed

    Note:
        Uses discriminated union from tools.types for full type safety.
        ToolInput provides compile-time guarantees that inputs match
        the expected tool schema.

    """

    step_number: int = Field(..., ge=1, description="Sequential step number")
    tool_name: str = Field(..., min_length=1, description="Tool to execute")
    tool_input: ToolInput = Field(..., description="Tool input parameters")
    reasoning: str = Field(..., min_length=1, description="Step reasoning")

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on attribute assignment
        use_enum_values=False,  # Keep enums as enums, not strings
        strict=True,  # Strict type checking
        extra="forbid",  # Reject unexpected fields
    )


class Plan(BaseModel):
    """
    Complete execution plan with multiple steps.

    Attributes:
        steps: Ordered list of plan steps
        final_goal: Original natural language prompt describing the goal

    """

    steps: list[PlanStep] = Field(..., min_length=1, description="Execution steps")
    final_goal: str = Field(..., min_length=1, description="Original goal prompt")

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on attribute assignment
        use_enum_values=False,  # Keep enums as enums, not strings
        strict=True,  # Strict type checking
        extra="forbid",  # Reject unexpected fields
    )
