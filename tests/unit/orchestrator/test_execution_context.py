"""Tests for ExecutionContext variable resolution and state management.

Tests verify inter-step state management, variable resolution, and
the fix for the "Todo not found: <first_todo_id>" error.
"""

import pytest

from challenge.models.run import ExecutionStep
from challenge.orchestrator.execution_context import ExecutionContext


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
        """Test extraction of first_todo_id from list output (fixes original error)."""
        context = ExecutionContext()
        todos = [
            {"id": "abc-123", "text": "Buy milk"},
            {"id": "def-456", "text": "Walk dog"},
        ]
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
        """Test extraction of last_todo_id from list output."""
        context = ExecutionContext()
        todos = [
            {"id": "abc-123", "text": "Buy milk"},
            {"id": "def-456", "text": "Walk dog"},
            {"id": "ghi-789", "text": "Clean room"},
        ]
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
        """Test extraction from single-item list."""
        context = ExecutionContext()
        todos = [{"id": "abc-123", "text": "Only task"}]
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
        """Test extraction of item count from list."""
        context = ExecutionContext()
        todos = [
            {"id": "1", "text": "Task 1"},
            {"id": "2", "text": "Task 2"},
            {"id": "3", "text": "Task 3"},
        ]
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
        """Test handling of empty list output."""
        context = ExecutionContext()
        step = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output=[],
            attempts=1,
        )

        context.record_step(step)

        # Empty list shouldn't create id variables
        assert "first_todo_id" not in context.variables
        assert "last_todo_id" not in context.variables
        # But should still have step output
        assert context.variables["step_1_output"] == []


class TestVariableExtractionFromDict:
    """Test automatic variable extraction from dict outputs."""

    def test_extract_id_from_dict(self):
        """Test extraction of ID from dict output."""
        context = ExecutionContext()
        todo = {"id": "abc-123", "text": "New task", "completed": False}
        step = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "add", "text": "New task"},
            success=True,
            output=todo,
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["last_todo_id"] == "abc-123"
        assert context.variables["step_1_id"] == "abc-123"

    def test_extract_result_from_dict(self):
        """Test extraction of result field from dict."""
        context = ExecutionContext()
        output = {"result": 42, "status": "success"}
        step = ExecutionStep(
            step_number=1,
            tool_name="custom_tool",
            tool_input={},
            success=True,
            output=output,
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["step_1_result"] == 42


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

        tool_input = {"action": "get", "id": "{user_id}"}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"action": "get", "id": "abc-123"}

    def test_resolve_angle_bracket_syntax(self):
        """Test variable resolution with <var> syntax."""
        context = ExecutionContext()
        context.variables["user_id"] = "abc-123"

        tool_input = {"action": "get", "id": "<user_id>"}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"action": "get", "id": "abc-123"}

    def test_resolve_nested_dict(self):
        """Test nested dict variable resolution."""
        context = ExecutionContext()
        context.variables["todo_id"] = "xyz-789"

        tool_input = {"action": "update", "params": {"id": "{todo_id}", "status": "complete"}}
        resolved = context.resolve_variables(tool_input)

        assert resolved["params"]["id"] == "xyz-789"
        assert resolved["params"]["status"] == "complete"

    def test_resolve_list_variables(self):
        """Test variable resolution in lists."""
        context = ExecutionContext()
        context.variables["id1"] = "aaa"
        context.variables["id2"] = "bbb"

        tool_input = {"ids": ["{id1}", "{id2}", "ccc"]}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"ids": ["aaa", "bbb", "ccc"]}

    def test_resolve_partial_string_replacement(self):
        """Test partial string variable replacement."""
        context = ExecutionContext()
        context.variables["user"] = "alice"

        tool_input = {"message": "Hello {user}!"}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"message": "Hello alice!"}

    def test_resolve_multiple_variables_in_string(self):
        """Test multiple variable replacement in single string."""
        context = ExecutionContext()
        context.variables["first"] = "Alice"
        context.variables["last"] = "Smith"

        tool_input = {"name": "{first} {last}"}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"name": "Alice Smith"}

    def test_resolve_type_preservation(self):
        """Test type preservation for full variable replacement."""
        context = ExecutionContext()
        context.variables["count"] = 42
        context.variables["items"] = ["a", "b", "c"]
        context.variables["enabled"] = True

        tool_input = {
            "count": "{count}",
            "items": "{items}",
            "enabled": "{enabled}",
        }
        resolved = context.resolve_variables(tool_input)

        assert resolved["count"] == 42  # int preserved
        assert resolved["items"] == ["a", "b", "c"]  # list preserved
        assert resolved["enabled"] is True  # bool preserved

    def test_resolve_variable_not_found_error(self):
        """Test error on undefined variable."""
        context = ExecutionContext()
        context.variables["foo"] = "bar"

        tool_input = {"key": "{undefined}"}

        with pytest.raises(ValueError, match="Variable 'undefined' not found"):
            context.resolve_variables(tool_input)

    def test_resolve_helpful_error_message(self):
        """Test error message includes available variables."""
        context = ExecutionContext()
        context.variables["var1"] = "value1"
        context.variables["var2"] = "value2"

        tool_input = {"key": "{missing}"}

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

        tool_input = {"auth": "{api_key}"}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"auth": "secret-123"}

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
        todos = [
            {"id": "abc-123", "text": "Buy milk"},
            {"id": "def-456", "text": "Walk dog"},
        ]
        step1 = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output=todos,
            attempts=1,
        )
        context.record_step(step1)

        # Verify first_todo_id extracted
        assert context.variables["first_todo_id"] == "abc-123"

        # Step 2: Complete first todo using variable
        tool_input_step2 = {"action": "complete", "todo_id": "{first_todo_id}"}
        resolved_input = context.resolve_variables(tool_input_step2)

        # This resolves the original "Todo not found: <first_todo_id>" error
        assert resolved_input["todo_id"] == "abc-123"

        step2 = ExecutionStep(
            step_number=2,
            tool_name="todo_store",
            tool_input=resolved_input,
            success=True,
            output={"id": "abc-123", "completed": True},
            attempts=1,
        )
        context.record_step(step2)

        # Verify workflow completed successfully
        assert len(context.get_execution_log()) == 2
        assert context.step_outputs[2]["completed"] is True

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
        tool_input_step2 = {"expression": "{step_1_value} / 2"}
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
        """Test complex workflow with multiple variable types."""
        context = ExecutionContext()

        # Step 1: List operation
        items = [
            {"id": "item-1", "name": "First"},
            {"id": "item-2", "name": "Second"},
        ]
        step1 = ExecutionStep(
            step_number=1,
            tool_name="list_tool",
            tool_input={},
            success=True,
            output=items,
            attempts=1,
        )
        context.record_step(step1)

        # Step 2: Process first item
        tool_input_step2 = {
            "id": "{first_todo_id}",
            "count": "{step_1_count}",
        }
        resolved_input = context.resolve_variables(tool_input_step2)
        assert resolved_input["id"] == "item-1"
        assert resolved_input["count"] == 2

        step2 = ExecutionStep(
            step_number=2,
            tool_name="process_tool",
            tool_input=resolved_input,
            success=True,
            output={"processed": True, "result": 42},
            attempts=1,
        )
        context.record_step(step2)

        # Step 3: Use result from step 2
        tool_input_step3 = {"value": "{step_2_result}"}
        resolved_input = context.resolve_variables(tool_input_step3)
        assert resolved_input["value"] == 42
