"""Metrics API schemas with strong typing."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RunsByStatusMetrics(BaseModel):
    """Run count by status."""

    pending: int = Field(ge=0, description="Number of pending runs")
    running: int = Field(ge=0, description="Number of running runs")
    completed: int = Field(ge=0, description="Number of completed runs")
    failed: int = Field(ge=0, description="Number of failed runs")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        json_schema_extra={"example": {"pending": 2, "running": 1, "completed": 140, "failed": 7}},
    )


class RunMetrics(BaseModel):
    """Run statistics."""

    total: int = Field(ge=0, description="Total number of runs")
    by_status: RunsByStatusMetrics = Field(description="Runs grouped by status")
    success_rate: float = Field(ge=0.0, le=1.0, description="Success rate (0.0 to 1.0)")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "total": 150,
                "by_status": {"pending": 2, "running": 1, "completed": 140, "failed": 7},
                "success_rate": 0.933,
            }
        },
    )


class ExecutionMetrics(BaseModel):
    """Execution time statistics."""

    avg_duration_seconds: float = Field(ge=0.0, description="Average execution duration in seconds")
    total_steps_executed: int = Field(ge=0, description="Total number of steps executed")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        json_schema_extra={"example": {"avg_duration_seconds": 1.2, "total_steps_executed": 450}},
    )


class ToolMetrics(BaseModel):
    """Tool usage statistics."""

    total_executions: int = Field(ge=0, description="Total number of tool executions")
    by_tool: dict[str, int] = Field(description="Tool execution counts by tool name")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        json_schema_extra={"example": {"total_executions": 450, "by_tool": {"calculator": 250, "todo_store": 200}}},
    )


class MetricsResponse(BaseModel):
    """Complete metrics response."""

    timestamp: datetime = Field(description="Metrics collection timestamp (UTC)")
    runs: RunMetrics = Field(description="Run statistics")
    execution: ExecutionMetrics = Field(description="Execution statistics")
    tools: ToolMetrics = Field(description="Tool usage statistics")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "timestamp": "2025-01-29T10:00:00.000Z",
                "runs": {
                    "total": 150,
                    "by_status": {"pending": 2, "running": 1, "completed": 140, "failed": 7},
                    "success_rate": 0.933,
                },
                "execution": {"avg_duration_seconds": 1.2, "total_steps_executed": 450},
                "tools": {"total_executions": 450, "by_tool": {"calculator": 250, "todo_store": 200}},
            }
        },
    )
