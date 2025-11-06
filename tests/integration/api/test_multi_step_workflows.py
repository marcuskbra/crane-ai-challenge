"""
Integration tests for multi-step workflows with variable resolution.

This module tests the complete workflow of:
1. LLM planning with variable references
2. ExecutionContext variable resolution
3. Multi-step execution with dependencies
"""

import pytest

from challenge.models import ExecutionStep
from challenge.models.plan import PlanStep
from challenge.orchestrator.execution_context import ExecutionContext
from challenge.orchestrator.execution_engine import ExecutionEngine
from challenge.tools.registry import get_tool_registry
from challenge.tools.types import TodoCompleteInput, TodoItem, TodoListOutput


@pytest.fixture(autouse=True)
def clear_todo_store():
    """Clear todo store before each test to ensure test isolation."""
    # Get singleton tool registry
    registry = get_tool_registry()
    todo_tool = registry.get("todo_store")

    # Clear todos before test
    if todo_tool:
        todo_tool.todos.clear()

    yield

    # Clear todos after test (cleanup)
    if todo_tool:
        todo_tool.todos.clear()


class TestExecutionContextVariableResolution:
    """Test ExecutionContext variable extraction and resolution."""

    def test_first_todo_id_extraction(self):
        """Test that first_todo_id is extracted from TodoListOutput."""
        context = ExecutionContext()

        # Simulate step 1: list todos
        todos_output = TodoListOutput(
            todos=[
                TodoItem(id="todo-abc-123", text="Buy milk", completed=False, created_at="2024-01-01T00:00:00Z"),
                TodoItem(id="todo-def-456", text="Pay bills", completed=False, created_at="2024-01-01T00:00:00Z"),
            ],
            total_count=2,
            completed_count=0,
            pending_count=2,
        )
        step1 = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output=todos_output,
            attempts=1,
        )
        context.record_step(step1)

        # Check variable extraction
        assert "first_todo_id" in context.variables
        assert context.variables["first_todo_id"] == "todo-abc-123"
        assert "last_todo_id" in context.variables
        assert context.variables["last_todo_id"] == "todo-def-456"

    def test_variable_resolution_in_tool_input(self):
        """Test that variables are resolved in tool_input."""
        context = ExecutionContext()
        context.variables["first_todo_id"] = "todo-abc-123"

        # Tool input with variable reference
        tool_input = TodoCompleteInput(action="complete", todo_id="{first_todo_id}")

        # Resolve variables
        resolved = context.resolve_variables(tool_input)

        assert resolved["action"] == "complete"
        assert resolved["todo_id"] == "todo-abc-123"

    def test_variable_resolution_error_handling(self):
        """Test error handling for undefined variables."""
        context = ExecutionContext()

        tool_input = TodoCompleteInput(action="complete", todo_id="{undefined_variable}")

        with pytest.raises(ValueError, match="Variable 'undefined_variable' not found"):
            context.resolve_variables(tool_input)


class TestMultiStepExecution:
    """Test multi-step execution with ExecutionEngine."""

    @pytest.mark.asyncio
    async def test_list_and_complete_first_todo(self):
        """Test complete workflow: list todos → complete first."""
        # Setup
        tools = get_tool_registry()
        engine = ExecutionEngine(tools=tools)
        context = ExecutionContext()

        # Add some todos first
        todo_tool = tools.get("todo_store")
        await todo_tool.execute(action="add", text="Buy milk")
        await todo_tool.execute(action="add", text="Pay bills")

        # Create plan with variable reference
        plan = [
            PlanStep(
                step_number=1,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List all todos to get IDs",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={"action": "complete", "todo_id": "{first_todo_id}"},
                reasoning="Complete first todo using ID from step 1",
            ),
        ]

        # Execute plan with context
        results = await engine.execute_plan(plan, context=context)

        # Verify execution
        assert len(results) == 2
        assert results[0].success, f"Step 1 failed: {results[0].error}"
        assert results[1].success, f"Step 2 failed: {results[1].error}"

        # Verify first todo was completed
        list_result = await todo_tool.execute(action="list")
        todos = list_result.output.todos  # TodoListOutput has .todos attribute
        assert todos[0].completed is True
        assert todos[1].completed is False

    @pytest.mark.asyncio
    async def test_list_and_delete_last_todo(self):
        """Test complete workflow: list todos → delete last."""
        # Setup
        tools = get_tool_registry()
        engine = ExecutionEngine(tools=tools)
        context = ExecutionContext()

        # Add some todos first
        todo_tool = tools.get("todo_store")
        await todo_tool.execute(action="add", text="Buy milk")
        await todo_tool.execute(action="add", text="Pay bills")
        await todo_tool.execute(action="add", text="Call dentist")

        # Create plan with variable reference
        plan = [
            PlanStep(
                step_number=1,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List all todos to get IDs",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={"action": "delete", "todo_id": "{last_todo_id}"},
                reasoning="Delete last todo using ID from step 1",
            ),
        ]

        # Execute plan with context
        results = await engine.execute_plan(plan, context=context)

        # Verify execution
        assert len(results) == 2
        assert results[0].success, f"Step 1 failed: {results[0].error}"
        assert results[1].success, f"Step 2 failed: {results[1].error}"

        # Verify last todo was deleted (should have 2 left)
        list_result = await todo_tool.execute(action="list")
        todos = list_result.output.todos  # TodoListOutput has .todos attribute
        assert len(todos) == 2
        assert todos[0].text == "Buy milk"
        assert todos[1].text == "Pay bills"

    @pytest.mark.asyncio
    async def test_variable_resolution_failure_stops_execution(self):
        """Test that variable resolution errors stop execution."""
        # Setup
        tools = get_tool_registry()
        engine = ExecutionEngine(tools=tools)
        context = ExecutionContext()

        # Create plan with undefined variable
        plan = [
            PlanStep(
                step_number=1,
                tool_name="todo_store",
                tool_input={"action": "complete", "todo_id": "{nonexistent_variable}"},
                reasoning="Try to use undefined variable",
            ),
        ]

        # Execute plan with context
        results = await engine.execute_plan(plan, context=context)

        # Verify execution stopped with error
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
        assert "Variable resolution failed" in results[0].error
        assert "nonexistent_variable" in results[0].error
