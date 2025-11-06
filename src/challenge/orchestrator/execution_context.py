"""
Execution context for maintaining state across plan steps.

This module provides ExecutionContext for tracking step outputs and
enabling variable resolution between steps in multi-step plans.
"""

import logging
import re
from typing import Any, TypeAlias

from challenge.models.run import ExecutionStep
from challenge.orchestrator.type_guards import is_pydantic_model
from challenge.tools.types import ToolInput, ToolOutput

logger = logging.getLogger(__name__)

# Type-safe variable value constraints
# Variables store extracted values from tool outputs:
# - Scalar types: str, int, float, bool, None (extracted IDs, counts, results)
# - Pydantic models: ToolOutput (for step_N_output references storing complete tool outputs)
# All tools return strongly-typed Pydantic models (CalculatorOutput, TodoListOutput, etc.)
VariableValue: TypeAlias = str | int | float | bool | ToolOutput | None


class ExecutionContext:
    """
    Maintains execution state across plan steps.

    The ExecutionContext enables:
    - Variable resolution: Replace {placeholders} with actual values
    - Step output tracking: Store and retrieve outputs from previous steps
    - Reference syntax: Support {step_N_output}, {first_todo_id}, etc.

    Note:
        Type safety approach:
        - step_outputs uses dict[int, ToolOutput] to store strongly-typed tool outputs
          (CalculatorOutput, TodoListOutput, etc.)
        - variables uses dict[str, VariableValue] for either full ToolOutput or
          extracted scalar values (str, int, float, bool, None)

    """

    def __init__(self):
        """Initialize empty execution context."""
        # Note: step_outputs uses Any because tools return raw Python types (lists, dicts, floats)
        # not ToolOutput Pydantic models. ToolResult[TOutput] is generic, but output field
        # contains raw types (e.g., float for calculator, list for todo_store).
        # See Hybrid Typing Strategy in TYPING_GUIDE.md: strict inputs, flexible outputs.
        self.step_outputs: dict[int, Any] = {}
        self.variables: dict[str, VariableValue] = {}
        self._execution_log: list[ExecutionStep] = []

    def record_step(self, step: ExecutionStep) -> None:
        """
        Record a completed step and extract variables.

        Args:
            step: Completed execution step with output

        """
        self._execution_log.append(step)
        self.step_outputs[step.step_number] = step.output

        # Auto-extract common variables from output
        if step.success and step.output is not None:
            self._extract_variables(step.step_number, step.output)

        logger.debug(
            f"Recorded step {step.step_number}: {step.tool_name} "
            f"(success={step.success}, variables={list(self.variables.keys())})"
        )

    def _extract_variables(self, step_number: int, output: ToolOutput) -> None:
        """
        Auto-extract common variables from step output.

        Extracts:
        - first_X_id: First item's ID from list output
        - last_X_id: Last item's ID from list output
        - X_count: Count of items in list
        - step_N_output: Direct reference to step output

        Args:
            step_number: Step number that produced the output
            output: Strongly-typed tool output (CalculatorOutput, TodoListOutput, etc.)

        """
        # Always provide step_N_output reference
        self.variables[f"step_{step_number}_output"] = output

        # Handle Pydantic model outputs (TodoListOutput, TodoAddOutput, etc.)
        if is_pydantic_model(output):
            # TodoListOutput case: extract from .todos list
            if hasattr(output, "todos") and isinstance(output.todos, list) and output.todos:
                todos_list = output.todos
                self.variables[f"step_{step_number}_count"] = len(todos_list)

                # Extract first/last todo IDs
                if hasattr(todos_list[0], "id"):
                    self.variables["first_todo_id"] = todos_list[0].id
                    self.variables[f"step_{step_number}_first_id"] = todos_list[0].id

                    if len(todos_list) > 1:
                        self.variables["last_todo_id"] = todos_list[-1].id
                        self.variables[f"step_{step_number}_last_id"] = todos_list[-1].id

            # TodoAddOutput, TodoGetOutput, TodoCompleteOutput, TodoDeleteOutput: extract from .todo
            elif hasattr(output, "todo") and hasattr(output.todo, "id"):
                self.variables["last_todo_id"] = output.todo.id
                self.variables[f"step_{step_number}_id"] = output.todo.id

            # CalculatorOutput case: extract from .result
            elif hasattr(output, "result"):
                self.variables[f"step_{step_number}_result"] = output.result
        # Extract from scalar outputs (float, int, str returned directly by tools)
        elif isinstance(output, (int, float, str)):
            self.variables[f"step_{step_number}_value"] = output

    def resolve_variables(self, tool_input: ToolInput) -> dict[str, Any]:
        """
        Resolve variable placeholders in tool input.

        Supports syntax:
        - {variable_name}: Direct variable reference
        - {step_N_output}: Output from step N
        - {first_todo_id}, {last_todo_id}: Extracted IDs
        - <variable_name>: Alternative bracket style (for compatibility)

        Args:
            tool_input: ToolInput Pydantic model (discriminated union) potentially containing variables

        Returns:
            New dict with variables resolved to actual values

        Raises:
            ValueError: If variable reference cannot be resolved

        """
        # Convert ToolInput model to dict for variable resolution
        tool_input_dict = tool_input.model_dump()

        resolved = {}

        for key, value in tool_input_dict.items():
            # Resolve string values containing variables
            if isinstance(value, str):
                resolved[key] = self._resolve_string(value)
            # Resolve lists containing strings with variables
            elif isinstance(value, list):
                resolved[key] = [self._resolve_string(item) if isinstance(item, str) else item for item in value]
            else:
                # Pass through other types (int, float, bool, None, dict) unchanged
                # ToolInput models are flat - no nested models needing variable resolution
                resolved[key] = value

        return resolved

    def _resolve_string(self, value: str) -> str | VariableValue:
        """
        Resolve variables in a string value.

        Supports:
        - {var_name} or <var_name> syntax
        - Full replacement: "{var}" becomes the variable value (preserves type)
        - Partial replacement: "prefix {var} suffix" becomes string with var substituted

        Args:
            value: String potentially containing variable references

        Returns:
            - If single variable reference: Returns the actual VariableValue (preserves type)
            - If partial/multiple variables: Returns str with substitutions
            - If no variables: Returns original str

        Raises:
            ValueError: If variable reference cannot be resolved

        """
        # Pattern matches {var_name} or <var_name>
        pattern = r"\{([^}]+)\}|<([^>]+)>"

        matches = list(re.finditer(pattern, value))

        # No variables - return as-is
        if not matches:
            return value

        # Single variable taking entire string - return actual value type
        if len(matches) == 1 and matches[0].group(0) == value:
            var_name = matches[0].group(1) or matches[0].group(2)
            return self._get_variable(var_name)

        # Multiple variables or mixed text - return string with substitutions
        result = value
        for match in matches:
            var_name = match.group(1) or match.group(2)
            var_value = self._get_variable(var_name)

            # Convert to string for substitution
            var_str = str(var_value) if var_value is not None else ""
            result = result.replace(match.group(0), var_str)

        return result

    def _get_variable(self, var_name: str) -> VariableValue:
        """
        Get variable value by name.

        Args:
            var_name: Variable name (without brackets)

        Returns:
            Strongly-typed variable value (ToolOutput, str, int, float, bool, or None)

        Raises:
            ValueError: If variable not found in context

        """
        if var_name in self.variables:
            return self.variables[var_name]

        # Provide helpful error message with available variables
        available = ", ".join(sorted(self.variables.keys()))
        raise ValueError(
            f"Variable '{var_name}' not found in execution context. Available variables: {available or 'none'}"
        )

    def get_step_output(self, step_number: int) -> ToolOutput:
        """
        Get output from a specific step.

        Args:
            step_number: Step number (1-indexed)

        Returns:
            Strongly-typed tool output from the specified step
            (CalculatorOutput, TodoListOutput, etc.)

        Raises:
            KeyError: If step not found

        """
        if step_number not in self.step_outputs:
            raise KeyError(f"No output recorded for step {step_number}")
        return self.step_outputs[step_number]

    def set_variable(self, name: str, value: VariableValue) -> None:
        """
        Manually set a variable in the context.

        Args:
            name: Variable name (without brackets)
            value: Variable value (must be JSON-serializable type)

        """
        self.variables[name] = value
        logger.debug(f"Set variable '{name}' = {value!r}")

    def get_execution_log(self) -> list[ExecutionStep]:
        """
        Get complete execution log.

        Returns:
            List of all recorded execution steps

        """
        return self._execution_log.copy()

    def clear(self) -> None:
        """Clear all context state (useful for testing)."""
        self.step_outputs.clear()
        self.variables.clear()
        self._execution_log.clear()
        logger.debug("Cleared execution context")
