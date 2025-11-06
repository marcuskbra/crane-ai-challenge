"""
Execution engine for running plan steps with retry logic.

This module provides the execution engine that runs individual plan steps
with exponential backoff retry and timeout handling, including support for
variable resolution between steps via ExecutionContext.
"""

import asyncio
import logging
import time

from challenge.models.plan import PlanStep
from challenge.models.run import ExecutionStep
from challenge.orchestrator.execution_context import ExecutionContext
from challenge.tools import BaseTool
from challenge.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Executes plan steps with automatic retry and error handling.

    The execution engine handles:
    - Step execution with tool lookup
    - Exponential backoff retry (1s, 2s, 4s)
    - Tool not found errors
    - Execution result wrapping

    """

    def __init__(
        self,
        tools: ToolRegistry,
        max_retries: int = 3,
    ):
        """
        Initialize execution engine.

        Args:
            tools: Tool registry for tool lookup
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

        """
        start_time = time.perf_counter()
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Get tool
                tool: BaseTool | None = self.tools.get(step.tool_name)

                if not tool:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    return ExecutionStep(
                        step_number=step.step_number,
                        tool_name=step.tool_name,
                        tool_input=step.tool_input,
                        success=False,
                        error=f"Tool not found: {step.tool_name}",
                        attempts=attempt,
                        duration_ms=duration_ms,
                    )

                # Execute tool (convert Pydantic model to dict for unpacking)
                # PlanStep.tool_input is strictly typed as ToolInput (Pydantic models)
                # so we always call model_dump() to serialize for tool execution
                tool_input_dict = step.tool_input.model_dump()
                result = await tool.execute(**tool_input_dict)

                # Calculate duration
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Return result
                return ExecutionStep(
                    step_number=step.step_number,
                    tool_name=step.tool_name,
                    tool_input=step.tool_input,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    attempts=attempt,
                    duration_ms=duration_ms,
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
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ExecutionStep(
            step_number=step.step_number,
            tool_name=step.tool_name,
            tool_input=step.tool_input,
            success=False,
            error=f"Failed after {self.max_retries} attempts: {last_error}",
            attempts=self.max_retries,
            duration_ms=duration_ms,
        )

    async def execute_plan(
        self,
        plan_steps: list[PlanStep],
        step_timeout: float = 30.0,
        context: ExecutionContext | None = None,
    ) -> list[ExecutionStep]:
        """
        Execute all steps in a plan sequentially with variable resolution.

        Each step is executed with timeout protection. If a step fails,
        execution stops and returns the results up to that point.

        When ExecutionContext is provided:
        - Variables in tool_input are resolved from context
        - Step outputs are recorded in context for later steps
        - Enables multi-step workflows with data dependencies

        Args:
            plan_steps: List of plan steps to execute
            step_timeout: Timeout in seconds for each step (default: 30.0)
            context: Optional execution context for variable resolution

        Returns:
            List of ExecutionStep results (may be incomplete if step fails)

        Raises:
            None - failures are captured in ExecutionStep.error

        """
        execution_log: list[ExecutionStep] = []

        for step in plan_steps:
            # Resolve variables in tool_input if context provided
            tool_input = step.tool_input
            if context:
                try:
                    # Resolve variables in ToolInput model
                    # resolve_variables() accepts ToolInput and returns dict with resolved values
                    resolved_dict = context.resolve_variables(step.tool_input)
                    # Use resolved dict as tool_input for execution
                    tool_input = resolved_dict
                    if resolved_dict != step.tool_input.model_dump():
                        logger.info(
                            f"Step {step.step_number}: Resolved variables in tool_input: "
                            f"{step.tool_input} → {tool_input}"
                        )
                except ValueError as e:
                    # Variable resolution failed - record as error
                    step_result = ExecutionStep(
                        step_number=step.step_number,
                        tool_name=step.tool_name,
                        tool_input=step.tool_input,
                        success=False,
                        error=f"Variable resolution failed: {e}",
                        attempts=1,
                        duration_ms=0.0,
                    )
                    logger.error(f"Step {step.step_number} variable resolution failed: {e}")
                    execution_log.append(step_result)
                    break

            # Create step with resolved input
            resolved_step = PlanStep(
                step_number=step.step_number,
                tool_name=step.tool_name,
                tool_input=tool_input,
                reasoning=step.reasoning,
            )

            try:
                # Execute step with timeout
                step_result = await asyncio.wait_for(self.execute_step_with_retry(resolved_step), timeout=step_timeout)
            except asyncio.TimeoutError:
                # Step timed out
                step_result = ExecutionStep(
                    step_number=step.step_number,
                    tool_name=step.tool_name,
                    tool_input=tool_input,
                    success=False,
                    error=f"Step timed out after {step_timeout}s",
                    attempts=1,
                    duration_ms=step_timeout * 1000,  # Timeout duration
                )
                logger.error(f"Step {step.step_number} timed out after {step_timeout}s")

            execution_log.append(step_result)

            # Record step output in context (even if failed)
            if context:
                context.record_step(step_result)

            # Stop on first failure
            if not step_result.success:
                logger.error(f"Step {step.step_number} failed: {step_result.error}. Stopping execution.")
                break

        return execution_log
