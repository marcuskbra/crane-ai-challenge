"""
Plan models for structured task execution.

This module defines the data structures for representing execution plans
generated from natural language prompts.
"""

from typing import Any

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """
    A single step in an execution plan.

    Attributes:
        step_number: Sequential step number (1-indexed)
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool
        reasoning: Human-readable explanation of why this step is needed

    """

    step_number: int = Field(..., ge=1, description="Sequential step number")
    tool_name: str = Field(..., min_length=1, description="Tool to execute")
    tool_input: dict[str, Any] = Field(..., description="Tool input parameters")
    reasoning: str = Field(..., min_length=1, description="Step reasoning")


class Plan(BaseModel):
    """
    Complete execution plan with multiple steps.

    Attributes:
        steps: Ordered list of plan steps
        final_goal: Original natural language prompt describing the goal

    """

    steps: list[PlanStep] = Field(..., min_length=1, description="Execution steps")
    final_goal: str = Field(..., min_length=1, description="Original goal prompt")
