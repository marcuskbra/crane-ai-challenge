"""
Execution engine for running plan steps with retry logic.

This module provides the execution engine that runs individual plan steps
with exponential backoff retry and timeout handling.
"""

import asyncio
import logging

from challenge.models.plan import PlanStep
from challenge.models.run import ExecutionStep
from challenge.orchestrator.protocols import ToolProvider

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Executes plan steps with automatic retry and error handling.

    The execution engine handles:
    - Step execution with tool lookup
    - Exponential backoff retry (1s, 2s, 4s)
    - Tool not found errors
    - Execution result wrapping

    Example:
        >>> from challenge.tools.registry import get_tool_registry
        >>> engine = ExecutionEngine(tools=get_tool_registry(), max_retries=3)
        >>> result = await engine.execute_step_with_retry(step)
        >>> assert result.success or result.attempts <= 3

    """

    def __init__(
        self,
        tools: ToolProvider,
        max_retries: int = 3,
    ):
        """
        Initialize execution engine.

        Args:
            tools: Tool provider for tool lookup
            max_retries: Maximum retry attempts per step (default: 3)

        """
        self.tools = tools
        self.max_retries = max_retries

    async def execute_step_with_retry(self, step: PlanStep) -> ExecutionStep:
        """
        Execute a single step with exponential backoff retry.

        Retry delays: 1s, 2s, 4s (exponential backoff)

        Args:
            step: PlanStep to execute

        Returns:
            ExecutionStep with execution result

        Example:
            >>> step = PlanStep(
            ...     step_number=1,
            ...     tool_name="calculator",
            ...     tool_input={"expression": "2 + 2"}
            ... )
            >>> result = await engine.execute_step_with_retry(step)
            >>> assert result.success
            >>> assert result.output == 4.0

        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Get tool
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

    async def execute_plan(self, plan_steps: list[PlanStep], step_timeout: float = 30.0) -> list[ExecutionStep]:
        """
        Execute all steps in a plan sequentially.

        Each step is executed with timeout protection. If a step fails,
        execution stops and returns the results up to that point.

        Args:
            plan_steps: List of plan steps to execute
            step_timeout: Timeout in seconds for each step (default: 30.0)

        Returns:
            List of ExecutionStep results (may be incomplete if step fails)

        Raises:
            None - failures are captured in ExecutionStep.error

        Example:
            >>> plan = [
            ...     PlanStep(1, "calculator", {"expression": "2 + 2"}),
            ...     PlanStep(2, "calculator", {"expression": "5 * 3"}),
            ... ]
            >>> results = await engine.execute_plan(plan)
            >>> assert len(results) == 2
            >>> assert all(r.success for r in results)

        """
        execution_log: list[ExecutionStep] = []

        for step in plan_steps:
            try:
                # Execute step with timeout
                step_result = await asyncio.wait_for(self.execute_step_with_retry(step), timeout=step_timeout)
            except asyncio.TimeoutError:
                # Step timed out
                step_result = ExecutionStep(
                    step_number=step.step_number,
                    tool_name=step.tool_name,
                    tool_input=step.tool_input,
                    success=False,
                    error=f"Step timed out after {step_timeout}s",
                    attempts=1,
                )
                logger.error(f"Step {step.step_number} timed out after {step_timeout}s")

            execution_log.append(step_result)

            # Stop on first failure
            if not step_result.success:
                logger.error(f"Step {step.step_number} failed: {step_result.error}. Stopping execution.")
                break

        return execution_log
