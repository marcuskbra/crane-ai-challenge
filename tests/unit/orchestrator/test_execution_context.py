"""Tests for ExecutionContext variable resolution and state management.

Tests verify inter-step state management, variable resolution, and
the fix for the "Todo not found: <first_todo_id>" error.
"""

import pytest

from challenge.models.run import ExecutionStep
from challenge.orchestrator.execution_context import ExecutionContext
from challenge.tools.types import (
    CalculatorInput,
    CalculatorOutput,
    TodoAddInput,
    TodoAddOutput,
    TodoCompleteInput,
    TodoCompleteOutput,
    TodoGetInput,
    TodoGetOutput,
    TodoItem,
    TodoListInput,
    TodoListOutput,
)


class TestExecutionContext:
    """Test ExecutionContext basic functionality."""

    def test_initial_state(self):
        """Test ExecutionContext initializes with empty state."""
        context = ExecutionContext()

        assert len(context.step_outputs) == 0
        assert len(context.variables) == 0
        assert len(context.get_execution_log()) == 0

    def test_record_step_basic(self):
        """Test basic step recording."""
        context = ExecutionContext()
        step = ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "2+2"},
            success=True,
            output=4.0,
            attempts=1,
        )

        context.record_step(step)

        assert context.step_outputs[1] == 4.0
        assert context.variables["step_1_output"] == 4.0
        assert context.variables["step_1_value"] == 4.0
        assert len(context.get_execution_log()) == 1

    def test_record_step_failed(self):
        """Test recording failed step."""
        context = ExecutionContext()
        step = ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "invalid"},
            success=False,
            error="Invalid expression",
            attempts=3,
        )

        context.record_step(step)

        assert context.step_outputs[1] is None
        # Failed steps shouldn't extract variables
        assert "step_1_value" not in context.variables


class TestVariableExtractionFromList:
    """Test automatic variable extraction from list outputs."""

    def test_extract_first_todo_id(self):
        """Test extraction of first_todo_id from TodoListOutput (fixes original error)."""
        context = ExecutionContext()
        todos = TodoListOutput(
            todos=[
                TodoItem(id="abc-123", text="Buy milk", completed=False, created_at="2024-01-01T00:00:00Z"),
                TodoItem(id="def-456", text="Walk dog", completed=False, created_at="2024-01-01T00:00:00Z"),
            ],
            total_count=2,
            completed_count=0,
            pending_count=2,
        )
        step = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output=todos,
            attempts=1,
        )

        context.record_step(step)

        # This is the critical fix for "Todo not found: <first_todo_id>" error
        assert context.variables["first_todo_id"] == "abc-123"
        assert context.variables["step_1_first_id"] == "abc-123"

    def test_extract_last_todo_id(self):
        """Test extraction of last_todo_id from TodoListOutput."""
        context = ExecutionContext()
        todos = TodoListOutput(
            todos=[
                TodoItem(id="abc-123", text="Buy milk", completed=False, created_at="2024-01-01T00:00:00Z"),
                TodoItem(id="def-456", text="Walk dog", completed=False, created_at="2024-01-01T00:00:00Z"),
                TodoItem(id="ghi-789", text="Clean room", completed=False, created_at="2024-01-01T00:00:00Z"),
            ],
            total_count=3,
            completed_count=0,
            pending_count=3,
        )
        step = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output=todos,
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["last_todo_id"] == "ghi-789"
        assert context.variables["step_1_last_id"] == "ghi-789"

    def test_extract_single_todo_id(self):
        """Test extraction from single-item TodoListOutput."""
        context = ExecutionContext()
        todos = TodoListOutput(
            todos=[TodoItem(id="abc-123", text="Only task", completed=False, created_at="2024-01-01T00:00:00Z")],
            total_count=1,
            completed_count=0,
            pending_count=1,
        )
        step = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output=todos,
            attempts=1,
        )

        context.record_step(step)

        # Single item should set first but not last
        assert context.variables["first_todo_id"] == "abc-123"
        assert "last_todo_id" not in context.variables

    def test_extract_list_count(self):
        """Test extraction of item count from TodoListOutput."""
        context = ExecutionContext()
        todos = TodoListOutput(
            todos=[
                TodoItem(id="1", text="Task 1", completed=False, created_at="2024-01-01T00:00:00Z"),
                TodoItem(id="2", text="Task 2", completed=False, created_at="2024-01-01T00:00:00Z"),
                TodoItem(id="3", text="Task 3", completed=False, created_at="2024-01-01T00:00:00Z"),
            ],
            total_count=3,
            completed_count=0,
            pending_count=3,
        )
        step = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output=todos,
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["step_1_count"] == 3

    def test_extract_from_empty_list(self):
        """Test handling of empty TodoListOutput."""
        context = ExecutionContext()
        todos = TodoListOutput(todos=[], total_count=0, completed_count=0, pending_count=0)
        step = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output=todos,
            attempts=1,
        )

        context.record_step(step)

        # Empty list shouldn't create id variables
        assert "first_todo_id" not in context.variables
        assert "last_todo_id" not in context.variables
        # But should still have step output (the TodoListOutput model)
        assert context.variables["step_1_output"] == todos


class TestVariableExtractionFromDict:
    """Test automatic variable extraction from dict outputs."""

    def test_extract_id_from_dict(self):
        """Test extraction of ID from TodoAddOutput."""
        context = ExecutionContext()
        output = TodoAddOutput(
            todo=TodoItem(id="abc-123", text="New task", completed=False, created_at="2024-01-01T00:00:00Z")
        )
        step = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "add", "text": "New task"},
            success=True,
            output=output,
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["last_todo_id"] == "abc-123"
        assert context.variables["step_1_id"] == "abc-123"

    def test_extract_result_from_dict(self):
        """Test extraction of result field from CalculatorOutput."""
        context = ExecutionContext()
        output = CalculatorOutput(result=42.0, expression="40 + 2")
        step = ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={},
            success=True,
            output=output,
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["step_1_result"] == 42.0


class TestVariableExtractionFromScalar:
    """Test automatic variable extraction from scalar outputs."""

    def test_extract_from_int(self):
        """Test extraction from integer output."""
        context = ExecutionContext()
        step = ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "2+2"},
            success=True,
            output=4,
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["step_1_value"] == 4

    def test_extract_from_float(self):
        """Test extraction from float output."""
        context = ExecutionContext()
        step = ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "10/3"},
            success=True,
            output=3.333333,
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["step_1_value"] == 3.333333

    def test_extract_from_string(self):
        """Test extraction from string output."""
        context = ExecutionContext()
        step = ExecutionStep(
            step_number=1,
            tool_name="text_tool",
            tool_input={},
            success=True,
            output="result text",
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["step_1_value"] == "result text"


class TestVariableResolution:
    """Test variable resolution in tool inputs."""

    def test_resolve_simple_variable(self):
        """Test simple variable resolution with {var} syntax."""
        context = ExecutionContext()
        context.variables["user_id"] = "abc-123"

        tool_input = TodoGetInput(action="get", todo_id="{user_id}")
        resolved = context.resolve_variables(tool_input)

        assert resolved["action"] == "get"
        assert resolved["todo_id"] == "abc-123"

    def test_resolve_angle_bracket_syntax(self):
        """Test variable resolution with <var> syntax."""
        context = ExecutionContext()
        context.variables["user_id"] = "abc-123"

        tool_input = TodoGetInput(action="get", todo_id="<user_id>")
        resolved = context.resolve_variables(tool_input)

        assert resolved["action"] == "get"
        assert resolved["todo_id"] == "abc-123"

    def test_resolve_nested_dict(self):
        """Test variable resolution with calculator expression containing variables."""
        context = ExecutionContext()
        context.variables["value"] = "10"

        tool_input = CalculatorInput(expression="{value} + 5")
        resolved = context.resolve_variables(tool_input)

        assert resolved["expression"] == "10 + 5"

    def test_resolve_partial_string_replacement(self):
        """Test partial string variable replacement in todo text."""
        context = ExecutionContext()
        context.variables["user"] = "alice"

        tool_input = TodoAddInput(action="add", text="Hello {user}!")
        resolved = context.resolve_variables(tool_input)

        assert resolved["action"] == "add"
        assert resolved["text"] == "Hello alice!"

    def test_resolve_multiple_variables_in_string(self):
        """Test multiple variable replacement in single string."""
        context = ExecutionContext()
        context.variables["first"] = "Alice"
        context.variables["last"] = "Smith"

        tool_input = TodoAddInput(action="add", text="{first} {last}")
        resolved = context.resolve_variables(tool_input)

        assert resolved["action"] == "add"
        assert resolved["text"] == "Alice Smith"

    def test_resolve_type_preservation(self):
        """Test that string variables are substituted into string fields."""
        context = ExecutionContext()
        context.variables["todo_id"] = "abc-123"

        tool_input = TodoCompleteInput(action="complete", todo_id="{todo_id}")
        resolved = context.resolve_variables(tool_input)

        assert resolved["action"] == "complete"
        assert resolved["todo_id"] == "abc-123"  # string variable resolved

    def test_resolve_variable_not_found_error(self):
        """Test error on undefined variable."""
        context = ExecutionContext()
        context.variables["foo"] = "bar"

        tool_input = TodoGetInput(action="get", todo_id="{undefined}")

        with pytest.raises(ValueError, match="Variable 'undefined' not found"):
            context.resolve_variables(tool_input)

    def test_resolve_helpful_error_message(self):
        """Test error message includes available variables."""
        context = ExecutionContext()
        context.variables["var1"] = "value1"
        context.variables["var2"] = "value2"

        tool_input = TodoGetInput(action="get", todo_id="{missing}")

        with pytest.raises(ValueError, match="Variable 'missing' not found") as exc_info:
            context.resolve_variables(tool_input)

        error_msg = str(exc_info.value)
        assert "var1" in error_msg
        assert "var2" in error_msg


class TestStepOutputRetrieval:
    """Test direct step output retrieval."""

    def test_get_step_output(self):
        """Test direct step output retrieval."""
        context = ExecutionContext()
        step = ExecutionStep(
            step_number=1,
            tool_name="test",
            tool_input={},
            success=True,
            output="result",
            attempts=1,
        )
        context.record_step(step)

        assert context.get_step_output(1) == "result"

    def test_get_step_output_not_found(self):
        """Test error on missing step output."""
        context = ExecutionContext()

        with pytest.raises(KeyError, match="No output recorded for step 99"):
            context.get_step_output(99)


class TestManualVariableManagement:
    """Test manual variable management."""

    def test_set_variable_manually(self):
        """Test manual variable setting."""
        context = ExecutionContext()

        context.set_variable("custom_var", "custom_value")

        assert context.variables["custom_var"] == "custom_value"

    def test_manual_variable_resolution(self):
        """Test resolution of manually set variables."""
        context = ExecutionContext()
        context.set_variable("api_key", "secret-123")

        tool_input = CalculatorInput(expression="{api_key}")
        resolved = context.resolve_variables(tool_input)

        assert resolved["expression"] == "secret-123"

    def test_overwrite_variable(self):
        """Test that manual setting can overwrite auto-extracted variables."""
        context = ExecutionContext()

        # Auto-extract
        step = ExecutionStep(
            step_number=1,
            tool_name="tool",
            tool_input={},
            success=True,
            output=42,
            attempts=1,
        )
        context.record_step(step)
        assert context.variables["step_1_value"] == 42

        # Overwrite
        context.set_variable("step_1_value", 100)
        assert context.variables["step_1_value"] == 100


class TestExecutionLogManagement:
    """Test execution log functionality."""

    def test_get_execution_log(self):
        """Test retrieving execution log."""
        context = ExecutionContext()

        step1 = ExecutionStep(
            step_number=1,
            tool_name="tool1",
            tool_input={},
            success=True,
            output="result1",
            attempts=1,
        )
        step2 = ExecutionStep(
            step_number=2,
            tool_name="tool2",
            tool_input={},
            success=True,
            output="result2",
            attempts=1,
        )

        context.record_step(step1)
        context.record_step(step2)

        log = context.get_execution_log()
        assert len(log) == 2
        assert log[0].step_number == 1
        assert log[1].step_number == 2

    def test_execution_log_is_copy(self):
        """Test that get_execution_log returns a copy."""
        context = ExecutionContext()

        step = ExecutionStep(
            step_number=1,
            tool_name="tool",
            tool_input={},
            success=True,
            output="result",
            attempts=1,
        )
        context.record_step(step)

        log1 = context.get_execution_log()
        log2 = context.get_execution_log()

        # Should be different list instances
        assert log1 is not log2
        # But contain same data
        assert len(log1) == len(log2)


class TestContextClear:
    """Test context clearing functionality."""

    def test_clear(self):
        """Test clearing context state."""
        context = ExecutionContext()
        context.variables["foo"] = "bar"
        context.step_outputs[1] = "test"
        step = ExecutionStep(
            step_number=1,
            tool_name="tool",
            tool_input={},
            success=True,
            output="result",
            attempts=1,
        )
        context.record_step(step)

        context.clear()

        assert len(context.variables) == 0
        assert len(context.step_outputs) == 0
        assert len(context.get_execution_log()) == 0


class TestMultiStepWorkflow:
    """Test complete multi-step workflows (integration scenarios)."""

    def test_todo_workflow_with_variable_resolution(self):
        """Test complete todo workflow: list → complete first (fixes original error)."""
        context = ExecutionContext()

        # Step 1: List todos
        todos_output = TodoListOutput(
            todos=[
                TodoItem(id="abc-123", text="Buy milk", completed=False, created_at="2024-01-01T00:00:00Z"),
                TodoItem(id="def-456", text="Walk dog", completed=False, created_at="2024-01-01T00:00:00Z"),
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

        # Verify first_todo_id extracted
        assert context.variables["first_todo_id"] == "abc-123"

        # Step 2: Complete first todo using variable
        tool_input_step2 = TodoCompleteInput(action="complete", todo_id="{first_todo_id}")
        resolved_input = context.resolve_variables(tool_input_step2)

        # This resolves the original "Todo not found: <first_todo_id>" error
        assert resolved_input["todo_id"] == "abc-123"

        completed_output = TodoCompleteOutput(
            todo=TodoItem(
                id="abc-123",
                text="Buy milk",
                completed=True,
                created_at="2024-01-01T00:00:00Z",
                completed_at="2024-01-01T01:00:00Z",
            )
        )
        step2 = ExecutionStep(
            step_number=2,
            tool_name="todo_store",
            tool_input=resolved_input,
            success=True,
            output=completed_output,
            attempts=1,
        )
        context.record_step(step2)

        # Verify workflow completed successfully
        assert len(context.get_execution_log()) == 2
        assert context.step_outputs[2].todo.completed is True

    def test_calculator_chain_workflow(self):
        """Test calculator chain: calc → reference → calc."""
        context = ExecutionContext()

        # Step 1: Calculate initial value
        step1 = ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "10 * 5"},
            success=True,
            output=50.0,
            attempts=1,
        )
        context.record_step(step1)

        # Step 2: Use previous result
        tool_input_step2 = CalculatorInput(expression="{step_1_value} / 2")
        resolved_input = context.resolve_variables(tool_input_step2)
        assert resolved_input["expression"] == "50.0 / 2"

        step2 = ExecutionStep(
            step_number=2,
            tool_name="calculator",
            tool_input=resolved_input,
            success=True,
            output=25.0,
            attempts=1,
        )
        context.record_step(step2)

        assert context.variables["step_2_value"] == 25.0

    def test_complex_multi_step_workflow(self):
        """Test complex workflow with multiple variable types using real tool inputs."""

        context = ExecutionContext()

        # Step 1: List todos operation
        todos_output = TodoListOutput(
            todos=[
                TodoItem(id="item-1", text="First", completed=False, created_at="2024-01-01T00:00:00Z"),
                TodoItem(id="item-2", text="Second", completed=False, created_at="2024-01-01T00:00:00Z"),
            ],
            total_count=2,
            completed_count=0,
            pending_count=2,
        )
        step1 = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input=TodoListInput(action="list"),
            success=True,
            output=todos_output,
            attempts=1,
        )
        context.record_step(step1)

        # Step 2: Get first todo using variable
        tool_input_step2 = TodoGetInput(action="get", todo_id="{first_todo_id}")
        resolved_input = context.resolve_variables(tool_input_step2)
        assert resolved_input["todo_id"] == "item-1"

        get_output = TodoGetOutput(
            todo=TodoItem(id="item-1", text="First todo", completed=False, created_at="2024-01-01T00:00:00Z")
        )
        step2 = ExecutionStep(
            step_number=2,
            tool_name="todo_store",
            tool_input=TodoGetInput(todo_id=resolved_input["todo_id"]),
            success=True,
            output=get_output,
            attempts=1,
        )
        context.record_step(step2)

        # Step 3: Complete todo using variable (testing step_N_id extraction)
        tool_input_step3 = TodoCompleteInput(action="complete", todo_id="{step_2_id}")
        resolved_input = context.resolve_variables(tool_input_step3)
        assert resolved_input["todo_id"] == "item-1"
