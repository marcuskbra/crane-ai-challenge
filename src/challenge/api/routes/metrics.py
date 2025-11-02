"""
Metrics endpoint for observability.

Provides basic operational metrics including run statistics,
execution times, and success rates.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from challenge.api.schemas.metrics import (
    ExecutionMetrics,
    MetricsResponse,
    RunMetrics,
    RunsByStatusMetrics,
    ToolMetrics,
)
from challenge.models.run import RunStatus
from challenge.orchestrator.orchestrator import Orchestrator

from ..dependencies import get_orchestrator

router = APIRouter()


@router.get("/metrics")
async def get_metrics(
    orchestrator: Orchestrator = Depends(get_orchestrator),  # noqa: B008
) -> MetricsResponse:
    """
    Get system metrics for observability.

    Returns operational metrics including:
    - Run statistics (total, by status)
    - Success rate
    - Average execution time
    - Tool usage statistics

    Args:
        orchestrator: Orchestrator instance (injected)

    Returns:
        MetricsResponse: Typed metrics response model

    Example:
        >>> GET /api/v1/metrics
        {
            "timestamp": "2025-01-29T10:00:00.000Z",
            "runs": {
                "total": 150,
                "by_status": {
                    "pending": 2,
                    "running": 1,
                    "completed": 140,
                    "failed": 7
                },
                "success_rate": 0.933
            },
            "execution": {
                "avg_duration_seconds": 1.2,
                "total_steps_executed": 450
            },
            "tools": {
                "total_executions": 450,
                "by_tool": {
                    "calculator": 250,
                    "todo_store": 200
                }
            }
        }

    """
    runs = orchestrator.runs

    # Run statistics
    total_runs = len(runs)
    status_counts = {
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
    }

    # Execution statistics
    total_duration = 0.0
    completed_runs = 0
    total_steps = 0
    tool_executions: dict[str, int] = {}

    for run in runs.values():
        # Count by status
        status_counts[run.status.value] += 1

        # Calculate execution time for completed/failed runs
        if run.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            if run.started_at and run.completed_at:
                duration = (run.completed_at - run.started_at).total_seconds()
                total_duration += duration
                completed_runs += 1

        # Count tool executions
        for step in run.execution_log:
            total_steps += 1
            tool_name = step.tool_name.lower()
            tool_executions[tool_name] = tool_executions.get(tool_name, 0) + 1

    # Calculate averages
    avg_duration = total_duration / completed_runs if completed_runs > 0 else 0.0
    success_rate = (
        status_counts["completed"] / (status_counts["completed"] + status_counts["failed"])
        if (status_counts["completed"] + status_counts["failed"]) > 0
        else 0.0
    )

    # Build typed response
    return MetricsResponse(
        timestamp=datetime.now(timezone.utc),
        runs=RunMetrics(
            total=total_runs,
            by_status=RunsByStatusMetrics(
                pending=status_counts["pending"],
                running=status_counts["running"],
                completed=status_counts["completed"],
                failed=status_counts["failed"],
            ),
            success_rate=round(success_rate, 3),
        ),
        execution=ExecutionMetrics(
            avg_duration_seconds=round(avg_duration, 2),
            total_steps_executed=total_steps,
        ),
        tools=ToolMetrics(
            total_executions=total_steps,
            by_tool=tool_executions,
        ),
    )
