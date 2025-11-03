"""
Orchestrator for executing plans with retry logic.

This module implements the execution engine that runs plans step-by-step
with exponential backoff retry for failed steps.
"""

import asyncio
import inspect
import logging
import time
from datetime import datetime, timezone

from challenge.models.run import Run, RunStatus
from challenge.orchestrator.execution_engine import ExecutionEngine
from challenge.orchestrator.metrics_tracker import MetricsTracker
from challenge.orchestrator.run_manager import RunManager
from challenge.planner.planner import PatternBasedPlanner
from challenge.planner.protocol import Planner
from challenge.tools.registry import ToolRegistry, get_tool_registry

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator for executing multi-step plans with retry logic.

    The orchestrator coordinates:
    - Planning: Converting prompts to structured plans
    - Execution: Running plan steps sequentially with timeout protection
    - Retry: Automatic retry with exponential backoff
    - State: Tracking run status and execution history

    Retry strategy: Exponential backoff (1s, 2s, 4s)
    Timeout: Configurable per-step timeout (default: 30s)

    Example:
        >>> orchestrator = Orchestrator(step_timeout=60.0)
        >>> run = await orchestrator.create_run("calculate 2 + 3")
        >>> # Wait for async execution
        >>> assert run.status == RunStatus.COMPLETED
        >>> assert run.result == 5.0

    """

    def __init__(
        self,
        planner: Planner | None = None,
        tools: ToolRegistry | None = None,
        max_retries: int = 3,
        step_timeout: float = 30.0,
    ):
        """
        Initialize orchestrator.

        Args:
            planner: Planner implementation (any object with create_plan method).
                     Creates PatternBasedPlanner if None.
            tools: Tool registry dict (uses default if None)
            max_retries: Maximum retry attempts per step
            step_timeout: Timeout in seconds for each step execution (default: 30.0)

        """
        self.planner = planner or PatternBasedPlanner()
        self.tools = tools if tools is not None else get_tool_registry()
        self.step_timeout = step_timeout

        # Component composition
        self.metrics = MetricsTracker()
        self.engine = ExecutionEngine(tools=self.tools, max_retries=max_retries)
        self.run_manager = RunManager()

    async def create_run(self, prompt: str) -> Run:
        """
        Create and start executing a run.

        This method:
        1. Creates a run in PENDING status
        2. Generates an execution plan
        3. Starts async execution
        4. Returns immediately without waiting

        Args:
            prompt: Natural language task to execute

        Returns:
            Run instance (execution continues asynchronously)

        """
        run = Run(prompt=prompt)

        try:
            # Track planning start time for metrics (use perf_counter for higher resolution)
            planning_start = time.perf_counter()

            # Generate plan (handle both sync and async planners)
            if inspect.iscoroutinefunction(self.planner.create_plan):
                plan = await self.planner.create_plan(prompt)
            else:
                plan = self.planner.create_plan(prompt)

            # Calculate planning latency
            planning_latency_ms = (time.perf_counter() - planning_start) * 1000

            # Update planner metrics
            token_count = getattr(self.planner, "last_token_count", None)
            self.metrics.record_plan(planning_latency_ms, token_count)

            run.plan = plan

            # Start async execution (don't await - returns immediately)
            task = asyncio.create_task(self._execute_run(run.run_id))

            # Store run and task
            self.run_manager.create_run(run, task)

            return run

        except Exception as e:
            # Planning failed - count as pattern attempt (fallback logic)
            self.metrics.record_plan(0.0, token_count=None)

            run.status = RunStatus.FAILED
            run.error = f"Planning failed: {e!s}"
            self.run_manager.create_run(run)
            logger.error(f"Planning failed for run {run.run_id}: {e}")
            return run

    def get_run(self, run_id: str) -> Run | None:
        """
        Get run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run instance or None if not found

        """
        return self.run_manager.get_run(run_id)

    def list_runs(self, limit: int = 10, offset: int = 0) -> list[Run]:
        """
        List runs with pagination.

        Runs are returned in reverse chronological order (most recent first).

        Args:
            limit: Maximum number of runs to return (default: 10)
            offset: Number of runs to skip (default: 0)

        Returns:
            List of Run instances

        """
        all_runs = self.run_manager.list_runs()

        # Sort by creation time (most recent first)
        sorted_runs = sorted(all_runs, key=lambda r: r.created_at, reverse=True)

        # Apply pagination
        start = offset
        end = offset + limit
        return sorted_runs[start:end]

    async def _execute_run(self, run_id: str) -> None:
        """
        Execute a run asynchronously.

        This method:
        1. Updates status to RUNNING
        2. Executes each step sequentially
        3. Updates status to COMPLETED or FAILED
        4. Records timestamps and results

        Args:
            run_id: Run identifier

        """
        run = self.run_manager.get_run(run_id)

        try:
            # Start execution
            run.status = RunStatus.RUNNING
            run.started_at = datetime.now(timezone.utc)
            logger.info(f"Starting execution for run {run_id}")

            # Execute plan with execution engine
            run.execution_log = await self.engine.execute_plan(run.plan.steps, step_timeout=self.step_timeout)

            # Check if all steps succeeded
            if run.execution_log and all(step.success for step in run.execution_log):
                # All steps succeeded
                run.status = RunStatus.COMPLETED
                # Set result to output of last step
                run.result = run.execution_log[-1].output
                logger.info(f"Run {run_id} completed successfully")
            else:
                # At least one step failed
                failed_step = next((step for step in run.execution_log if not step.success), None)
                if failed_step:
                    run.status = RunStatus.FAILED
                    run.error = f"Step {failed_step.step_number} failed: {failed_step.error}"
                    logger.error(f"Run {run_id} failed at step {failed_step.step_number}: {failed_step.error}")
                else:
                    # No steps executed (empty plan)
                    run.status = RunStatus.FAILED
                    run.error = "No steps executed"
                    logger.error(f"Run {run_id} failed: No steps in plan")

        except Exception as e:
            # Unexpected error
            run.status = RunStatus.FAILED
            run.error = f"Execution error: {e!s}"
            logger.error(f"Unexpected error in run {run_id}: {e}", exc_info=True)

        finally:
            run.completed_at = datetime.now(timezone.utc)
