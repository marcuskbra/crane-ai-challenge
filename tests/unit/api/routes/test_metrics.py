"""
Unit tests for metrics endpoint.
"""

import asyncio
from datetime import datetime, timezone

import pytest
from fastapi import status

from challenge.api.routes.metrics import get_metrics
from challenge.api.schemas.metrics import MetricsResponse
from challenge.domain.models.plan import Plan, PlanStep
from challenge.domain.models.run import ExecutionStep, Run, RunStatus
from challenge.infrastructure.tools.registry import get_tool_registry
from challenge.services.orchestration.orchestrator import Orchestrator
from challenge.services.planning.planner import PatternBasedPlanner


@pytest.fixture
def orchestrator():
    """Create an orchestrator instance for testing."""
    return Orchestrator(
        planner=PatternBasedPlanner(),
        tools=get_tool_registry(),
    )


@pytest.mark.asyncio
async def test_get_metrics_empty(test_client):
    """Test metrics endpoint with no runs."""
    response = test_client.get("/api/v1/metrics")

    assert response.status_code == status.HTTP_200_OK

    # Parse response with typed model
    metrics = MetricsResponse.model_validate(response.json())

    # Verify structure using typed attributes
    assert isinstance(metrics.timestamp, datetime)
    assert metrics.timestamp.tzinfo is not None

    # Verify empty state using typed access
    assert metrics.runs.total == 0
    assert metrics.runs.by_status.pending == 0
    assert metrics.runs.by_status.running == 0
    assert metrics.runs.by_status.completed == 0
    assert metrics.runs.by_status.failed == 0
    assert metrics.runs.success_rate == 0.0
    assert metrics.execution.total_steps_executed == 0
    assert metrics.tools.total_executions == 0


@pytest.mark.asyncio
async def test_get_metrics_with_runs(orchestrator):
    """Test metrics endpoint with multiple runs."""
    # Create completed run
    completed_run = Run(prompt="calculate 2 + 2")
    completed_run.status = RunStatus.COMPLETED
    completed_run.started_at = datetime(2025, 1, 29, 10, 0, 0, tzinfo=timezone.utc)
    completed_run.completed_at = datetime(2025, 1, 29, 10, 0, 2, tzinfo=timezone.utc)
    completed_run.plan = Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={"expression": "2 + 2"},
                reasoning="Add numbers",
            )
        ],
        final_goal="calculate 2 + 2",
    )
    completed_run.execution_log = [
        ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "2 + 2"},
            success=True,
            output=4.0,
            attempts=1,
        )
    ]

    # Create failed run
    failed_run = Run(prompt="invalid operation")
    failed_run.status = RunStatus.FAILED
    failed_run.started_at = datetime(2025, 1, 29, 10, 1, 0, tzinfo=timezone.utc)
    failed_run.completed_at = datetime(2025, 1, 29, 10, 1, 1, tzinfo=timezone.utc)
    failed_run.plan = Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={"expression": "invalid"},
                reasoning="Test failure",
            )
        ],
        final_goal="invalid operation",
    )
    failed_run.execution_log = [
        ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "invalid"},
            success=False,
            error="Invalid expression",
            attempts=3,
        )
    ]

    # Create pending run
    pending_run = Run(prompt="pending task")
    pending_run.status = RunStatus.PENDING
    pending_run.plan = Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List todos",
            )
        ],
        final_goal="pending task",
    )

    # Add runs to orchestrator via run_manager
    orchestrator.run_manager.create_run(completed_run)
    orchestrator.run_manager.create_run(failed_run)
    orchestrator.run_manager.create_run(pending_run)

    # Get metrics - returns typed model
    metrics = await get_metrics(orchestrator=orchestrator)

    # Verify it's the correct type
    assert isinstance(metrics, MetricsResponse)

    # Verify run statistics using typed access
    assert metrics.runs.total == 3
    assert metrics.runs.by_status.pending == 1
    assert metrics.runs.by_status.running == 0
    assert metrics.runs.by_status.completed == 1
    assert metrics.runs.by_status.failed == 1
    assert metrics.runs.success_rate == 0.5  # 1 completed / (1 completed + 1 failed)

    # Verify execution statistics using typed access
    assert metrics.execution.total_steps_executed == 2  # Only completed and failed runs have execution log
    assert metrics.execution.avg_duration_seconds == 1.5  # (2s + 1s) / 2

    # Verify tool statistics using typed access
    assert metrics.tools.total_executions == 2
    assert metrics.tools.by_tool["calculator"] == 2


@pytest.mark.asyncio
async def test_get_metrics_success_rate_calculation(orchestrator):
    """Test success rate calculation with various scenarios."""
    # Scenario 1: All completed
    run1 = Run(prompt="task 1")
    run1.status = RunStatus.COMPLETED
    run1.started_at = datetime.now(timezone.utc)
    run1.completed_at = datetime.now(timezone.utc)

    run2 = Run(prompt="task 2")
    run2.status = RunStatus.COMPLETED
    run2.started_at = datetime.now(timezone.utc)
    run2.completed_at = datetime.now(timezone.utc)

    orchestrator.run_manager.create_run(run1)
    orchestrator.run_manager.create_run(run2)

    metrics = await get_metrics(orchestrator=orchestrator)
    assert isinstance(metrics, MetricsResponse)
    assert metrics.runs.success_rate == 1.0  # 100% success

    # Scenario 2: All failed
    run1.status = RunStatus.FAILED
    run2.status = RunStatus.FAILED

    metrics = await get_metrics(orchestrator=orchestrator)
    assert isinstance(metrics, MetricsResponse)
    assert metrics.runs.success_rate == 0.0  # 0% success

    # Scenario 3: Mixed with pending (pending doesn't affect success rate)
    run3 = Run(prompt="task 3")
    run3.status = RunStatus.PENDING

    run1.status = RunStatus.COMPLETED
    run2.status = RunStatus.FAILED

    orchestrator.run_manager.create_run(run1)
    orchestrator.run_manager.create_run(run2)
    orchestrator.run_manager.create_run(run3)

    metrics = await get_metrics(orchestrator=orchestrator)
    assert isinstance(metrics, MetricsResponse)
    assert metrics.runs.success_rate == 0.5  # 1 completed / (1 completed + 1 failed)


@pytest.mark.asyncio
async def test_get_metrics_tool_aggregation(orchestrator):
    """Test tool execution aggregation."""
    run = Run(prompt="multi-tool task")
    run.status = RunStatus.COMPLETED
    run.started_at = datetime.now(timezone.utc)
    run.completed_at = datetime.now(timezone.utc)
    run.plan = Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={"expression": "1 + 1"},
                reasoning="Calculate",
            ),
            PlanStep(
                step_number=2,
                tool_name="calculator",
                tool_input={"expression": "2 + 2"},
                reasoning="Calculate again",
            ),
            PlanStep(
                step_number=3,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List todos",
            ),
        ],
        final_goal="multi-tool task",
    )
    run.execution_log = [
        ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "1 + 1"},
            success=True,
            output=2.0,
            attempts=1,
        ),
        ExecutionStep(
            step_number=2,
            tool_name="calculator",
            tool_input={"expression": "2 + 2"},
            success=True,
            output=4.0,
            attempts=1,
        ),
        ExecutionStep(
            step_number=3,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output={"todos": []},
            attempts=1,
        ),
    ]

    orchestrator.run_manager.create_run(run)

    metrics = await get_metrics(orchestrator=orchestrator)

    assert isinstance(metrics, MetricsResponse)
    assert metrics.tools.total_executions == 3
    assert metrics.tools.by_tool["calculator"] == 2
    assert metrics.tools.by_tool["todo_store"] == 1


@pytest.mark.asyncio
async def test_metrics_endpoint_integration(test_client):
    """Test metrics endpoint through HTTP client."""
    # Create a run
    response = test_client.post(
        "/api/v1/runs",
        json={"prompt": "calculate 10 + 20"},
    )
    assert response.status_code == status.HTTP_201_CREATED

    # Wait a bit for run to complete (or poll until complete)
    await asyncio.sleep(0.5)

    # Get metrics
    response = test_client.get("/api/v1/metrics")
    assert response.status_code == status.HTTP_200_OK

    # Parse response with typed model
    metrics = MetricsResponse.model_validate(response.json())

    # Should have at least 1 run using typed access
    assert metrics.runs.total >= 1

    # Verify timestamp format using typed access
    assert isinstance(metrics.timestamp, datetime)
    assert metrics.timestamp.tzinfo is not None  # Should be timezone-aware


@pytest.mark.asyncio
async def test_planner_metrics_included(test_client):
    """Test that planner metrics are included in metrics response."""
    response = test_client.get("/api/v1/metrics")
    assert response.status_code == status.HTTP_200_OK

    # Parse response with typed model
    metrics = MetricsResponse.model_validate(response.json())

    # Verify planner metrics are present
    assert metrics.planner is not None
    assert hasattr(metrics.planner, "total_plans_generated")
    assert hasattr(metrics.planner, "llm_plans")
    assert hasattr(metrics.planner, "pattern_plans")
    assert hasattr(metrics.planner, "cached_plans")
    assert hasattr(metrics.planner, "fallback_rate")
    assert hasattr(metrics.planner, "avg_tokens_per_plan")
    assert hasattr(metrics.planner, "avg_latency_ms")
    assert hasattr(metrics.planner, "cache_hit_rate")

    # Verify types and default values (before tracking is implemented)
    assert isinstance(metrics.planner.total_plans_generated, int)
    assert isinstance(metrics.planner.llm_plans, int)
    assert isinstance(metrics.planner.pattern_plans, int)
    assert isinstance(metrics.planner.cached_plans, int)
    assert isinstance(metrics.planner.fallback_rate, float)
    assert isinstance(metrics.planner.avg_tokens_per_plan, float)
    assert isinstance(metrics.planner.avg_latency_ms, float)
    assert isinstance(metrics.planner.cache_hit_rate, float)

    # Verify constraints
    assert metrics.planner.total_plans_generated >= 0
    assert metrics.planner.llm_plans >= 0
    assert metrics.planner.pattern_plans >= 0
    assert metrics.planner.cached_plans >= 0
    assert 0.0 <= metrics.planner.fallback_rate <= 1.0
    assert metrics.planner.avg_tokens_per_plan >= 0.0
    assert metrics.planner.avg_latency_ms >= 0.0
    assert 0.0 <= metrics.planner.cache_hit_rate <= 1.0


@pytest.mark.asyncio
async def test_empty_planner_metrics_default_values(orchestrator):
    """Test that empty planner metrics have sensible default values."""
    # Get metrics with no activity
    metrics = await get_metrics(orchestrator)

    # Verify planner metrics default to zero/zero state
    assert metrics.planner.total_plans_generated == 0
    assert metrics.planner.llm_plans == 0
    assert metrics.planner.pattern_plans == 0
    assert metrics.planner.cached_plans == 0
    assert metrics.planner.fallback_rate == 0.0
    assert metrics.planner.avg_tokens_per_plan == 0.0
    assert metrics.planner.avg_latency_ms == 0.0
    assert metrics.planner.cache_hit_rate == 0.0
