"""
Pattern-based planner for natural language to structured plans.

This module implements a regex-based planner that converts natural language
prompts into structured execution plans with tool calls.
"""

import re

from challenge.models.plan import Plan, PlanStep


class PatternBasedPlanner:
    """
    Pattern-based planner using regex for natural language understanding.

    This planner converts natural language prompts into structured plans by
    matching against predefined regex patterns for each supported tool.

    Supported patterns:
        - Calculator: "calculate/compute/evaluate X"
        - Todo add: "add todo/task X"
        - Todo list: "list/show todos/tasks"
        - Todo get: "get todo/task X"
        - Todo complete: "complete/finish todo/task X"
        - Todo delete: "delete/remove todo/task X"
        - Multi-step: "X and Y", "X then Y"

    Example:
        >>> planner = PatternBasedPlanner()
        >>> plan = planner.create_plan("calculate 2 + 3 and add todo Buy milk")
        >>> len(plan.steps)
        2

    """

    def create_plan(self, prompt: str) -> Plan:
        """
        Create execution plan from natural language prompt.

        Args:
            prompt: Natural language task description

        Returns:
            Plan with ordered execution steps

        Raises:
            ValueError: If prompt is empty or cannot be parsed

        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        prompt_lower = prompt.lower().strip()
        steps: list[PlanStep] = []

        # Split on "and" or "then" for multi-step plans
        sub_prompts = re.split(r"\s+(?:and|then)\s+", prompt_lower)

        for sub_prompt in sub_prompts:
            step = self._parse_single_step(sub_prompt.strip(), len(steps) + 1)
            if step:
                steps.append(step)

        if not steps:
            raise ValueError(f"Could not parse prompt: {prompt}")

        return Plan(steps=steps, final_goal=prompt)

    def _parse_single_step(self, prompt: str, step_number: int) -> PlanStep | None:
        """
        Parse a single prompt into a plan step.

        Args:
            prompt: Lowercase prompt text
            step_number: Step number to assign

        Returns:
            PlanStep if pattern matched, None otherwise

        """
        # Pattern 1: Calculator operations
        calc_pattern = r"(?:calculate|compute|evaluate|math|solve)\s+(.+)"
        if match := re.search(calc_pattern, prompt):
            expression = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="calculator",
                tool_input={"expression": expression},
                reasoning=f"Calculate: {expression}",
            )

        # Pattern 2: Add todo
        add_pattern = r"(?:add|create)\s+(?:a\s+)?(?:todo|task)(?:\s+(?:to|for|saying|that says))?\s+(.+)"
        if match := re.search(add_pattern, prompt):
            text = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="todo_store",
                tool_input={"action": "add", "text": text},
                reasoning=f"Add todo: {text}",
            )

        # Pattern 3: List todos
        list_pattern = r"(?:list|show|get|display|see)(?:\s+all)?(?:\s+(?:my|the))?\s+(?:todos|tasks)"
        if re.search(list_pattern, prompt):
            return PlanStep(
                step_number=step_number,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List all todos",
            )

        # Pattern 4: Get specific todo (with ID or description)
        get_pattern = r"(?:get|show|find)\s+(?:todo|task)\s+(.+)"
        if match := re.search(get_pattern, prompt):
            # Check if it looks like an ID (uuid-like) or description
            identifier = match.group(1).strip()
            # For now, assume it's an ID if it contains hyphens (uuid pattern)
            if "-" in identifier:
                return PlanStep(
                    step_number=step_number,
                    tool_name="todo_store",
                    tool_input={"action": "get", "todo_id": identifier},
                    reasoning=f"Get todo: {identifier}",
                )

        # Pattern 5: Complete todo
        complete_pattern = r"(?:complete|finish|mark\s+(?:as\s+)?(?:done|completed?))\s+(?:todo|task)\s+(.+)"
        if match := re.search(complete_pattern, prompt):
            todo_id = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="todo_store",
                tool_input={"action": "complete", "todo_id": todo_id},
                reasoning=f"Complete todo: {todo_id}",
            )

        # Pattern 6: Delete todo
        delete_pattern = r"(?:delete|remove)\s+(?:todo|task)\s+(.+)"
        if match := re.search(delete_pattern, prompt):
            todo_id = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="todo_store",
                tool_input={"action": "delete", "todo_id": todo_id},
                reasoning=f"Delete todo: {todo_id}",
            )

        return None
