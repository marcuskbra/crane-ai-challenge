"""
Run models for tracking execution state.

This module defines the data structures for representing execution runs
and their progress through the system.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from challenge.models.plan import Plan
from challenge.tools.types import ToolInput


class RunStatus(str, Enum):
    """
    Status of an execution run.

    Attributes:
        PENDING: Run created but not yet started
        RUNNING: Run is currently executing
        COMPLETED: Run finished successfully
        FAILED: Run encountered an error

    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionStep(BaseModel):
    """
    Result of executing a single plan step.

    Attributes:
        step_number: Step number from the plan
        tool_name: Tool that was executed
        tool_input: Type-safe discriminated union of tool inputs
        success: Whether execution succeeded
        output: Tool execution result (Any type since tools return raw Python types)
        error: Error message if failed
        attempts: Number of execution attempts (for retry tracking)
        duration_ms: Execution duration in milliseconds

    Note:
        tool_input uses strict ToolInput typing from tools.types.
        output remains Any since tools return raw Python types (list, dict, float, etc.)
        rather than Pydantic model instances.

    """

    step_number: int = Field(..., ge=1, description="Plan step number")
    tool_name: str = Field(..., description="Tool executed")
    tool_input: ToolInput = Field(..., description="Tool input used")
    success: bool = Field(..., description="Execution success")
    output: Any | None = Field(None, description="Result if successful")
    error: str | None = Field(None, description="Error if failed")
    attempts: int = Field(default=1, ge=1, description="Execution attempts")
    duration_ms: float = Field(default=0.0, ge=0, description="Execution duration in milliseconds")

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on attribute assignment
        use_enum_values=False,  # Keep enums as enums, not strings
        strict=True,  # Strict type checking
        extra="forbid",  # Reject unexpected fields
    )


class Run(BaseModel):
    """
    Complete execution run tracking.

    Attributes:
        run_id: Unique run identifier
        prompt: Original natural language prompt
        status: Current execution status
        plan: Generated execution plan (None until planning complete)
        execution_log: History of executed steps
        result: Final result from the last successful step (Any type)
        error: Error message if run failed
        created_at: Run creation timestamp
        started_at: Execution start timestamp
        completed_at: Execution completion timestamp

    Note:
        result uses Any since it's the output of the last step,
        which returns raw Python types (list, dict, float, etc.).

    """

    run_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique run ID")
    prompt: str = Field(..., min_length=1, description="Original prompt")
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current status")
    plan: Plan | None = Field(None, description="Execution plan")
    execution_log: list[ExecutionStep] = Field(default_factory=list, description="Step execution history")
    result: Any | None = Field(None, description="Final result")
    error: str | None = Field(None, description="Error message if failed")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    started_at: datetime | None = Field(None, description="Start timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on attribute assignment
        use_enum_values=False,  # Keep enums as enums, not strings
        strict=True,  # Strict type checking
        extra="forbid",  # Reject unexpected fields
    )
