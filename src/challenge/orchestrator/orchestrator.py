"""
Orchestrator for executing plans with retry logic.

This module implements the execution engine that runs plans step-by-step
with exponential backoff retry for failed steps.
"""

import asyncio
import logging
from datetime import datetime, timezone

from challenge.models.run import ExecutionStep, Run, RunStatus
from challenge.planner.planner import PatternBasedPlanner
from challenge.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator for executing multi-step plans with retry logic.

    The orchestrator coordinates:
    - Planning: Converting prompts to structured plans
    - Execution: Running plan steps sequentially
    - Retry: Automatic retry with exponential backoff
    - State: Tracking run status and execution history

    Retry strategy: Exponential backoff (1s, 2s, 4s)

    Example:
        >>> orchestrator = Orchestrator()
        >>> run = await orchestrator.create_run("calculate 2 + 3")
        >>> # Wait for async execution
        >>> assert run.status == RunStatus.COMPLETED
        >>> assert run.result == 5.0

    """

    def __init__(
        self,
        planner: PatternBasedPlanner | None = None,
        tools: dict | None = None,
        max_retries: int = 3,
    ):
        """
        Initialize orchestrator.

        Args:
            planner: Planner instance (creates default if None)
            tools: Tool registry dict (uses default if None)
            max_retries: Maximum retry attempts per step

        """
        self.planner = planner or PatternBasedPlanner()
        self.tools = tools if tools is not None else get_tool_registry()
        self.max_retries = max_retries
        self.runs: dict[str, Run] = {}
        self.tasks: dict[str, asyncio.Task] = {}  # Track background tasks

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
            # Generate plan
            plan = self.planner.create_plan(prompt)
            run.plan = plan
            self.runs[run.run_id] = run

            # Start async execution (don't await - returns immediately)
            task = asyncio.create_task(self._execute_run(run.run_id))
            self.tasks[run.run_id] = task

            return run

        except Exception as e:
            # Planning failed
            run.status = RunStatus.FAILED
            run.error = f"Planning failed: {e!s}"
            self.runs[run.run_id] = run
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
        return self.runs.get(run_id)

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
        run = self.runs[run_id]

        try:
            # Start execution
            run.status = RunStatus.RUNNING
            run.started_at = datetime.now(timezone.utc)
            logger.info(f"Starting execution for run {run_id}")

            # Execute each step
            for step in run.plan.steps:
                step_result = await self._execute_step_with_retry(step)
                run.execution_log.append(step_result)

                if not step_result.success:
                    # Step failed after retries
                    run.status = RunStatus.FAILED
                    run.error = f"Step {step.step_number} failed: {step_result.error}"
                    logger.error(f"Run {run_id} failed at step {step.step_number}: {step_result.error}")
                    return

            # All steps succeeded
            run.status = RunStatus.COMPLETED
            # Set result to output of last step
            if run.execution_log:
                run.result = run.execution_log[-1].output
            logger.info(f"Run {run_id} completed successfully")

        except Exception as e:
            # Unexpected error
            run.status = RunStatus.FAILED
            run.error = f"Execution error: {e!s}"
            logger.error(f"Unexpected error in run {run_id}: {e}", exc_info=True)

        finally:
            run.completed_at = datetime.now(timezone.utc)

    async def _execute_step_with_retry(self, step) -> ExecutionStep:
        """
        Execute a single step with exponential backoff retry.

        Retry delays: 1s, 2s, 4s (exponential backoff)

        Args:
            step: PlanStep to execute

        Returns:
            ExecutionStep with result

        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Get tool
                if isinstance(self.tools, dict):
                    tool = self.tools.get(step.tool_name)
                else:
                    # ToolRegistry instance
                    tool = self.tools.get(step.tool_name)

                if not tool:
                    return ExecutionStep(
                        step_number=step.step_number,
                        tool_name=step.tool_name,
                        tool_input=step.tool_input,
                        success=False,
                        error=f"Tool not found: {step.tool_name}",
                        attempts=attempt,
                    )

                # Execute tool
                result = await tool.execute(**step.tool_input)

                # Return result
                return ExecutionStep(
                    step_number=step.step_number,
                    tool_name=step.tool_name,
                    tool_input=step.tool_input,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    attempts=attempt,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Step {step.step_number} attempt {attempt}/{self.max_retries} failed: {e}")

                # Retry with exponential backoff (unless last attempt)
                if attempt < self.max_retries:
                    delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    logger.info(f"Retrying step {step.step_number} in {delay}s...")
                    await asyncio.sleep(delay)

        # All retries exhausted
        return ExecutionStep(
            step_number=step.step_number,
            tool_name=step.tool_name,
            tool_input=step.tool_input,
            success=False,
            error=f"Failed after {self.max_retries} attempts: {last_error}",
            attempts=self.max_retries,
        )
