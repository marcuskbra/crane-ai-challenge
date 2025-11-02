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

        # Special case: Handle "add todos for X, Y, and Z" pattern
        # This creates multiple todo additions from a comma-separated list
        multi_todo_pattern = r"add\s+(?:todos|tasks)\s+for\s+(.+)"
        if match := re.search(multi_todo_pattern, prompt_lower):
            items_text = match.group(1)
            # Split on commas and "and"
            items = re.split(r",\s*(?:and\s+)?|,?\s+and\s+", items_text)
            for item in items:
                item_clean = item.strip()
                if item_clean:
                    steps.append(
                        PlanStep(
                            step_number=len(steps) + 1,
                            tool_name="todo_store",
                            tool_input={"action": "add", "text": item_clean},
                            reasoning=f"Add todo: {item_clean}",
                        )
                    )
            if steps:
                return Plan(steps=steps, final_goal=prompt)

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
        # Matches: "calculate X", "what is X", "X divided by Y", "X plus Y", etc.
        calc_pattern = r"(?:calculate|compute|evaluate|math|solve|what\s+is)\s+(.+)"
        if match := re.search(calc_pattern, prompt):
            expression = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="calculator",
                tool_input={"expression": expression},
                reasoning=f"Calculate: {expression}",
            )

        # Pattern 1b: Natural language math expressions without keywords
        # Matches: "100 divided by 4", "5 plus 3", "10 minus 2", "multiply X by Y"
        natural_math_pattern = r"(?:(?:\d+(?:\.\d+)?)|(?:that|it|result))\s+(?:divided\s+by|plus|minus|times|multiplied\s+by)\s+(?:\d+(?:\.\d+)?)"
        if re.search(natural_math_pattern, prompt):
            return PlanStep(
                step_number=step_number,
                tool_name="calculator",
                tool_input={"expression": prompt},
                reasoning=f"Calculate: {prompt}",
            )

        # Pattern 1c: Operation verbs (multiply, divide, add, subtract)
        # Matches: "multiply that by 2", "divide this by 5", "add 10"
        operation_pattern = r"(?:multiply|divide|add|subtract)\s+(.+)"
        if re.search(operation_pattern, prompt):
            return PlanStep(
                step_number=step_number,
                tool_name="calculator",
                tool_input={"expression": prompt},
                reasoning=f"Calculate: {prompt}",
            )

        # Pattern 2a: Add X as a todo/task
        # Matches: "add the result as a todo", "add this as a task"
        add_as_pattern = r"(?:add|create)\s+(.+?)\s+as\s+(?:a\s+)?(?:todo|task)"
        if match := re.search(add_as_pattern, prompt):
            text = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="todo_store",
                tool_input={"action": "add", "text": text},
                reasoning=f"Add todo: {text}",
            )

        # Pattern 2b: Add todo/task X
        # Matches: "add todo X", "add task to do X", "add todo: X", "create a task for X"
        add_todo_pattern = r"(?:add|create)\s+(?:a\s+)?(?:todo|task)(?:\s*:\s*|\s+(?:to|for|saying|that says)\s+)(.+)"
        if match := re.search(add_todo_pattern, prompt):
            text = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="todo_store",
                tool_input={"action": "add", "text": text},
                reasoning=f"Add todo: {text}",
            )

        # Pattern 3: List todos
        # Matches: "list todos", "show todos", "show me my todos", "see all tasks", etc.
        list_pattern = r"(?:list|show|get|display|see)(?:\s+me)?(?:\s+all)?(?:\s+(?:my|the))?\s+(?:todos|tasks)"
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
