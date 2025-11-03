"""
Execution context for maintaining state across plan steps.

This module provides ExecutionContext for tracking step outputs and
enabling variable resolution between steps in multi-step plans.
"""

import logging
import re
from typing import Any

from challenge.models.run import ExecutionStep

logger = logging.getLogger(__name__)


class ExecutionContext:
    """
    Maintains execution state across plan steps.

    The ExecutionContext enables:
    - Variable resolution: Replace {placeholders} with actual values
    - Step output tracking: Store and retrieve outputs from previous steps
    - Reference syntax: Support {step_N_output}, {first_todo_id}, etc.

    Example:
        >>> context = ExecutionContext()
        >>> # Step 1 returns list of todos
        >>> context.record_step(ExecutionStep(
        ...     step_number=1,
        ...     tool_name="todo_store",
        ...     tool_input={"action": "list"},
        ...     success=True,
        ...     output=[{"id": "abc-123", "text": "Buy milk"}],
        ...     attempts=1
        ... ))
        >>> # Step 2 uses variable from step 1
        >>> tool_input = {"action": "complete", "todo_id": "{first_todo_id}"}
        >>> resolved = context.resolve_variables(tool_input)
        >>> resolved["todo_id"]
        'abc-123'

    """

    def __init__(self):
        """Initialize empty execution context."""
        self.step_outputs: dict[int, Any] = {}
        self.variables: dict[str, Any] = {}
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

    def _extract_variables(self, step_number: int, output: Any) -> None:
        """
        Auto-extract common variables from step output.

        Extracts:
        - first_X_id: First item's ID from list output
        - last_X_id: Last item's ID from list output
        - X_count: Count of items in list
        - step_N_output: Direct reference to step output

        Args:
            step_number: Step number that produced the output
            output: Step output value (any type)

        """
        # Always provide step_N_output reference
        self.variables[f"step_{step_number}_output"] = output

        # Extract from list outputs
        if isinstance(output, list) and output:
            # Count
            self.variables[f"step_{step_number}_count"] = len(output)

            # First/last item IDs (if items are dicts with 'id' field)
            if isinstance(output[0], dict) and "id" in output[0]:
                self.variables["first_todo_id"] = output[0]["id"]
                self.variables[f"step_{step_number}_first_id"] = output[0]["id"]

                if len(output) > 1:
                    self.variables["last_todo_id"] = output[-1]["id"]
                    self.variables[f"step_{step_number}_last_id"] = output[-1]["id"]

        # Extract from dict outputs
        elif isinstance(output, dict):
            # If dict has 'id', provide as last_id
            if "id" in output:
                self.variables["last_todo_id"] = output["id"]
                self.variables[f"step_{step_number}_id"] = output["id"]

            # Extract result field if present
            if "result" in output:
                self.variables[f"step_{step_number}_result"] = output["result"]

        # Extract from scalar outputs
        elif isinstance(output, (int, float, str)):
            self.variables[f"step_{step_number}_value"] = output

    def resolve_variables(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve variable placeholders in tool input.

        Supports syntax:
        - {variable_name}: Direct variable reference
        - {step_N_output}: Output from step N
        - {first_todo_id}, {last_todo_id}: Extracted IDs
        - <variable_name>: Alternative bracket style (for compatibility)

        Args:
            tool_input: Tool input dict potentially containing variables

        Returns:
            New dict with variables resolved to actual values

        Raises:
            ValueError: If variable reference cannot be resolved

        Example:
            >>> context = ExecutionContext()
            >>> context.variables["user_id"] = "abc-123"
            >>> tool_input = {"action": "get", "id": "{user_id}"}
            >>> resolved = context.resolve_variables(tool_input)
            >>> resolved["id"]
            'abc-123'

        """
        resolved = {}

        for key, value in tool_input.items():
            # Resolve string values containing variables
            if isinstance(value, str):
                resolved[key] = self._resolve_string(value)
            # Recursively resolve nested dicts
            elif isinstance(value, dict):
                resolved[key] = self.resolve_variables(value)
            # Recursively resolve lists
            elif isinstance(value, list):
                resolved[key] = [self._resolve_string(item) if isinstance(item, str) else item for item in value]
            else:
                resolved[key] = value

        return resolved

    def _resolve_string(self, value: str) -> Any:
        """
        Resolve variables in a string value.

        Supports:
        - {var_name} or <var_name> syntax
        - Full replacement: "{var}" becomes the variable value
        - Partial replacement: "prefix {var} suffix" becomes string with var substituted

        Args:
            value: String potentially containing variable references

        Returns:
            Resolved value (might be non-string if fully replaced)

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

    def _get_variable(self, var_name: str) -> Any:
        """
        Get variable value by name.

        Args:
            var_name: Variable name (without brackets)

        Returns:
            Variable value

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

    def get_step_output(self, step_number: int) -> Any:
        """
        Get output from a specific step.

        Args:
            step_number: Step number (1-indexed)

        Returns:
            Step output value

        Raises:
            KeyError: If step not found

        """
        if step_number not in self.step_outputs:
            raise KeyError(f"No output recorded for step {step_number}")
        return self.step_outputs[step_number]

    def set_variable(self, name: str, value: Any) -> None:
        """
        Manually set a variable in the context.

        Args:
            name: Variable name (without brackets)
            value: Variable value

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
