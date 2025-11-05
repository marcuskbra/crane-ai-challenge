"""
Tests for ExecutionEngine.

Tests step execution, retry logic, and error handling.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from challenge.models.plan import PlanStep
from challenge.orchestrator.execution_engine import ExecutionEngine
from challenge.tools.base import BaseTool, ToolMetadata, ToolResult


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str, should_fail: bool = False, output=None):
        self.name = name
        self.should_fail = should_fail
        self.output_value = output
        self.call_count = 0

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Mock tool for testing",
            input_schema={},
        )

    async def execute(self, **kwargs) -> ToolResult:
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(f"Mock failure from {self.name}")
        return ToolResult(success=True, output=self.output_value)


class MockToolProvider:
    """Mock tool provider for testing."""

    def __init__(self, tools: dict[str, BaseTool] | None = None):
        self.tools = tools or {}

    def get(self, tool_name: str) -> BaseTool | None:
        return self.tools.get(tool_name)


class TestExecutionEngine:
    """Test cases for ExecutionEngine."""

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool provider with test tools."""
        calculator = MockTool("calculator", output=4.0)
        todo = MockTool("todo_store", output="added")
        return MockToolProvider({"calculator": calculator, "todo_store": todo})

    @pytest.fixture
    def engine(self, mock_tools):
        """Create ExecutionEngine with mock tools."""
        return ExecutionEngine(tools=mock_tools, max_retries=3)

    @pytest.mark.asyncio
    async def test_execute_step_success(self, engine, mock_tools):
        """Test successful step execution."""
        step = PlanStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "2 + 2"},
            reasoning="Calculate 2 + 2",
        )

        result = await engine.execute_step_with_retry(step)

        assert result.success is True
        assert result.output == 4.0
        assert result.attempts == 1
        assert result.error is None
        assert mock_tools.tools["calculator"].call_count == 1

    @pytest.mark.asyncio
    async def test_execute_step_tool_not_found(self, engine):
        """Test handling of missing tool."""
        step = PlanStep(
            step_number=1,
            tool_name="nonexistent",
            tool_input={},
            reasoning="Try nonexistent tool",
        )

        result = await engine.execute_step_with_retry(step)

        assert result.success is False
        assert "Tool not found" in result.error
        assert "nonexistent" in result.error
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_execute_step_with_retry_success_on_second_attempt(self, mock_tools):
        """Test retry logic succeeds on second attempt."""
        failing_tool = MockTool("flaky", should_fail=True)
        mock_tools.tools["flaky"] = failing_tool
        engine = ExecutionEngine(tools=mock_tools, max_retries=3)

        step = PlanStep(
            step_number=1,
            tool_name="flaky",
            tool_input={},
            reasoning="Test retry logic",
        )

        # Make tool succeed on second attempt
        async def execute_with_delay(**kwargs):
            await asyncio.sleep(0.01)
            failing_tool.call_count += 1
            if failing_tool.call_count < 2:
                raise RuntimeError("First attempt fails")
            return ToolResult(success=True, output="success")

        failing_tool.execute = execute_with_delay  # type: ignore[invalid-assignment]

        result = await engine.execute_step_with_retry(step)

        assert result.success is True
        assert result.output == "success"
        assert result.attempts == 2

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_execute_step_exhaust_retries(self, mock_sleep, mock_tools):
        """Test all retries exhausted without waiting."""
        failing_tool = MockTool("always_fails", should_fail=True)
        mock_tools.tools["always_fails"] = failing_tool
        engine = ExecutionEngine(tools=mock_tools, max_retries=3)

        step = PlanStep(
            step_number=1,
            tool_name="always_fails",
            tool_input={},
            reasoning="Test exhausting retries",
        )

        result = await engine.execute_step_with_retry(step)

        assert result.success is False
        assert result.error is not None
        assert "Failed after 3 attempts" in result.error
        assert result.attempts == 3
        assert failing_tool.call_count == 3
        assert mock_sleep.call_count == 2  # 2 delays between 3 attempts

    @pytest.mark.asyncio
    async def test_execute_plan_all_steps_succeed(self, engine, mock_tools):
        """Test executing a plan where all steps succeed."""
        steps = [
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={"expression": "2 + 2"},
                reasoning="Calculate 2 + 2",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={"task": "test"},
                reasoning="Store todo task",
            ),
            PlanStep(
                step_number=3,
                tool_name="calculator",
                tool_input={"expression": "5 * 3"},
                reasoning="Calculate 5 * 3",
            ),
        ]

        results = await engine.execute_plan(steps, step_timeout=5.0)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].output == 4.0
        assert results[1].output == "added"
        assert results[2].output == 4.0

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_execute_plan_stops_on_first_failure(self, mock_sleep, engine, mock_tools):
        """Test plan execution stops at first failed step without retry delays."""
        failing_tool = MockTool("fail", should_fail=True)
        mock_tools.tools["fail"] = failing_tool

        steps = [
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={"expression": "2 + 2"},
                reasoning="Calculate 2 + 2",
            ),
            PlanStep(
                step_number=2,
                tool_name="fail",
                tool_input={},
                reasoning="This will fail",
            ),
            PlanStep(
                step_number=3,
                tool_name="calculator",
                tool_input={"expression": "5 * 3"},
                reasoning="Should not execute",
            ),
        ]

        results = await engine.execute_plan(steps, step_timeout=5.0)

        assert len(results) == 2  # Only first 2 steps
        assert results[0].success is True
        assert results[1].success is False
        # Third step never executed
        assert mock_tools.tools["calculator"].call_count == 1  # Only first step
        assert mock_sleep.call_count == 2  # 2 delays from failing step retries

    @pytest.mark.asyncio
    async def test_execute_plan_with_timeout(self, engine, mock_tools):
        """Test step timeout handling."""
        slow_tool = MockTool("slow")

        async def slow_execute(**kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout
            return ToolResult(success=True, output="done")

        slow_tool.execute = slow_execute  # type: ignore[method-assign]
        mock_tools.tools["slow"] = slow_tool

        steps = [
            PlanStep(
                step_number=1,
                tool_name="slow",
                tool_input={},
                reasoning="Test timeout",
            )
        ]

        results = await engine.execute_plan(steps, step_timeout=0.1)

        assert len(results) == 1
        assert results[0].success is False
        assert "timed out" in results[0].error.lower()
        assert results[0].attempts == 1

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_exponential_backoff_delays(self, mock_sleep, mock_tools):
        """Test exponential backoff retry delays without waiting."""
        failing_tool = MockTool("flaky", should_fail=True)
        mock_tools.tools["flaky"] = failing_tool
        engine = ExecutionEngine(tools=mock_tools, max_retries=3)

        step = PlanStep(
            step_number=1,
            tool_name="flaky",
            tool_input={},
            reasoning="Test exponential backoff",
        )

        await engine.execute_step_with_retry(step)

        # Verify exponential backoff pattern: 2^0=1s, 2^1=2s
        assert mock_sleep.call_count == 2  # 2 delays between 3 attempts
        assert mock_sleep.call_args_list[0][0][0] == 1.0  # First delay: 1s
        assert mock_sleep.call_args_list[1][0][0] == 2.0  # Second delay: 2s
        assert failing_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_plan_empty_steps(self, engine):
        """Test executing plan with no steps."""
        results = await engine.execute_plan([], step_timeout=5.0)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_execute_step_preserves_step_info(self, engine):
        """Test that execution result preserves step information."""
        step = PlanStep(
            step_number=5,
            tool_name="calculator",
            tool_input={"expression": "10 + 20"},
            reasoning="Calculate 10 + 20",
        )

        result = await engine.execute_step_with_retry(step)

        assert result.step_number == 5
        assert result.tool_name == "calculator"
        assert result.tool_input == {"expression": "10 + 20"}

    @pytest.mark.asyncio
    async def test_tool_returns_failure_result(self, mock_tools):
        """Test when tool returns success=False in result."""
        failing_result_tool = MockTool("fail_result")

        async def execute_with_failure(**kwargs):
            return ToolResult(success=False, error="Tool logic failed")

        failing_result_tool.execute = execute_with_failure  # type: ignore[method-assign]
        mock_tools.tools["fail_result"] = failing_result_tool
        engine = ExecutionEngine(tools=mock_tools, max_retries=3)

        step = PlanStep(
            step_number=1,
            tool_name="fail_result",
            tool_input={},
            reasoning="Test failure result",
        )

        result = await engine.execute_step_with_retry(step)

        # Tool didn't raise exception, so no retries
        assert result.success is False
        assert result.error == "Tool logic failed"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_multiple_tools_in_plan(self, engine, mock_tools):
        """Test plan with different tools."""
        steps = [
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={},
                reasoning="First calculation",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={},
                reasoning="Store todo",
            ),
            PlanStep(
                step_number=3,
                tool_name="calculator",
                tool_input={},
                reasoning="Second calculation",
            ),
        ]

        results = await engine.execute_plan(steps, step_timeout=5.0)

        assert len(results) == 3
        assert all(r.success for r in results)
        # Verify correct tools were called
        assert mock_tools.tools["calculator"].call_count == 2
        assert mock_tools.tools["todo_store"].call_count == 1

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_engine_with_different_max_retries(self, mock_sleep, mock_tools):
        """Test engine with custom max_retries without waiting."""
        failing_tool = MockTool("fail", should_fail=True)
        mock_tools.tools["fail"] = failing_tool
        engine = ExecutionEngine(tools=mock_tools, max_retries=5)

        step = PlanStep(
            step_number=1,
            tool_name="fail",
            tool_input={},
            reasoning="Test custom max_retries",
        )

        result = await engine.execute_step_with_retry(step)

        assert result.attempts == 5
        assert failing_tool.call_count == 5
        assert result.error is not None
        assert "Failed after 5 attempts" in result.error
        assert mock_sleep.call_count == 4  # 4 delays between 5 attempts

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_execute_plan_partial_completion(self, mock_sleep, engine, mock_tools):
        """Test that partial results are returned on failure without retry delays."""
        failing_tool = MockTool("fail", should_fail=True)
        mock_tools.tools["fail"] = failing_tool

        steps = [
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={},
                reasoning="First step",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={},
                reasoning="Second step",
            ),
            PlanStep(
                step_number=3,
                tool_name="fail",
                tool_input={},
                reasoning="Fails here",
            ),
            PlanStep(
                step_number=4,
                tool_name="calculator",
                tool_input={},
                reasoning="Never executed",
            ),
        ]

        results = await engine.execute_plan(steps, step_timeout=5.0)

        # Returns results up to and including the failed step
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is True
        assert results[2].success is False
        assert mock_sleep.call_count == 2  # 2 delays from failing step retries
