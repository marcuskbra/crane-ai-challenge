"""
Few-shot examples for prompt engineering with LLM planner.

This module provides high-quality examples demonstrating different task patterns
to improve LLM planning consistency and error handling through in-context learning.

Pattern Categories:
- Simple: Single-tool, single-step operations
- Moderate: Multi-tool or multi-step workflows
- Complex: Coordinated multi-tool operations with dependencies
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from challenge.domain.models.plan import Plan, PlanStep


class ExampleComplexity(str, Enum):
    """
    Task complexity levels for few-shot examples.

    Used to categorize examples by difficulty for adaptive prompt engineering.
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class FewShotExample(BaseModel):
    """
    Single few-shot example demonstrating prompt → plan pattern.

    Attributes:
        prompt: User's natural language request
        reasoning: Chain-of-thought reasoning explaining the approach
        plan: Structured execution plan with tools and steps
        complexity: Categorization for example selection strategy

    """

    prompt: str = Field(..., min_length=1, description="Natural language task")
    reasoning: str = Field(..., min_length=1, description="Chain-of-thought reasoning")
    plan: Plan = Field(..., description="Structured execution plan")
    complexity: ExampleComplexity = Field(..., description="Task complexity level")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


# Example 1: Simple calculation
EXAMPLE_SIMPLE_CALCULATION = FewShotExample(
    prompt="calculate (10 + 5) * 2",
    reasoning="""
    This is a straightforward arithmetic calculation request.
    Steps:
    1. Use calculator tool with the expression (10 + 5) * 2
    2. The calculator will evaluate and return 30
    """,
    plan=Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={"expression": "(10 + 5) * 2"},
                reasoning="Evaluate arithmetic expression to get result",
            )
        ],
        final_goal="Calculate the result of (10 + 5) * 2",
    ),
    complexity=ExampleComplexity.SIMPLE,
)

# Example 2: Todo CRUD operations
EXAMPLE_TODO_WORKFLOW = FewShotExample(
    prompt="add a todo to buy milk and then show me all my tasks",
    reasoning="""
    This requires two sequential todo_store operations:
    1. First, add the new todo item "buy milk"
    2. Then, list all existing todos to show complete task list
    The user wants to see all tasks after adding, not just the new one.
    """,
    plan=Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="todo_store",
                tool_input={"action": "add", "text": "buy milk"},
                reasoning="Add new todo item for buying milk",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List all todos to show complete task list including the newly added item",
            ),
        ],
        final_goal="Add todo to buy milk and display all tasks",
    ),
    complexity=ExampleComplexity.MODERATE,
)

# Example 3: Multi-step with calculation + todo using variable resolution
EXAMPLE_CALCULATION_THEN_TODO = FewShotExample(
    prompt="calculate 3 * 4 and add the result as a todo",
    reasoning="""
    This requires coordinating two different tools with variable resolution:
    1. First, calculate 3 * 4 using calculator
    2. Then, add a todo with the calculation result using {step_1_output}
    CRITICAL: Use {step_1_output} to reference the calculator result, not hardcoded values.
    """,
    plan=Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={"expression": "3 * 4"},
                reasoning="Calculate 3 * 4 to get numeric result",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={"action": "add", "text": "Result: {step_1_output}"},
                reasoning="Store calculation result as todo using {step_1_output} variable",
            ),
        ],
        final_goal="Calculate 3 * 4 and store result as todo",
    ),
    complexity=ExampleComplexity.COMPLEX,
)

# Example 4: Complex multi-todo workflow
EXAMPLE_MULTI_TODO_OPERATIONS = FewShotExample(
    prompt="add tasks to buy groceries and pay bills, then list all my todos",
    reasoning="""
    This requires adding multiple todos followed by a list operation:
    1. Add first todo: "buy groceries"
    2. Add second todo: "pay bills"
    3. List all todos to show complete task list
    Each todo is added separately, then we show the final state.
    """,
    plan=Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="todo_store",
                tool_input={"action": "add", "text": "buy groceries"},
                reasoning="Add first todo item for grocery shopping",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={"action": "add", "text": "pay bills"},
                reasoning="Add second todo item for bill payment",
            ),
            PlanStep(
                step_number=3,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List all todos to show complete updated task list",
            ),
        ],
        final_goal="Add multiple todos and display complete task list",
    ),
    complexity=ExampleComplexity.COMPLEX,
)

# Example 5: Edge case - simple list operation
EXAMPLE_SIMPLE_LIST = FewShotExample(
    prompt="show my todos",
    reasoning="""
    This is a simple query for existing todos.
    Steps:
    1. Use todo_store with list action to retrieve all todos
    No additions or modifications needed, just retrieval.
    """,
    plan=Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="Retrieve and display all existing todo items",
            )
        ],
        final_goal="Display all todo items",
    ),
    complexity=ExampleComplexity.SIMPLE,
)

# Example 6: Variable resolution - complete first todo
EXAMPLE_COMPLETE_FIRST_TODO = FewShotExample(
    prompt="List all my todos and mark the first one as complete",
    reasoning="""
    This requires variable resolution between steps:
    1. First, list all todos to get the todo IDs
    2. Then, complete the first todo using {first_todo_id} variable
    The {first_todo_id} variable will be automatically extracted from step 1's output.
    """,
    plan=Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List all todos to get their IDs",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={"action": "complete", "todo_id": "{first_todo_id}"},
                reasoning="Complete the first todo using ID from step 1",
            ),
        ],
        final_goal="List todos and complete the first one",
    ),
    complexity=ExampleComplexity.COMPLEX,
)

# Example 7: Variable resolution - delete last todo
EXAMPLE_DELETE_LAST_TODO = FewShotExample(
    prompt="Show me my tasks and delete the last one",
    reasoning="""
    This requires variable resolution for the last item:
    1. First, list all todos to get the todo IDs
    2. Then, delete the last todo using {last_todo_id} variable
    The {last_todo_id} variable will be automatically extracted from step 1's output.
    """,
    plan=Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List all todos to get their IDs",
            ),
            PlanStep(
                step_number=2,
                tool_name="todo_store",
                tool_input={"action": "delete", "todo_id": "{last_todo_id}"},
                reasoning="Delete the last todo using ID from step 1",
            ),
        ],
        final_goal="Display todos and delete the last one",
    ),
    complexity=ExampleComplexity.COMPLEX,
)

# Example 8: Chained calculator operations with variable resolution
EXAMPLE_CHAINED_CALCULATIONS = FewShotExample(
    prompt="Calculate 10 + 5, then multiply the result by 2, and add it as a todo",
    reasoning="""
    This requires chaining calculator operations with variable resolution:
    1. First, calculate 10 + 5 using calculator
    2. Then, multiply the result by 2 using {step_1_output}
    3. Finally, add the final result as a todo using {step_2_output}
    CRITICAL: Each step references the previous step's output using {step_N_output} syntax.
    """,
    plan=Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={"expression": "10 + 5"},
                reasoning="Calculate initial value 10 + 5",
            ),
            PlanStep(
                step_number=2,
                tool_name="calculator",
                tool_input={"expression": "{step_1_output} * 2"},
                reasoning="Multiply result from step 1 by 2 using {step_1_output}",
            ),
            PlanStep(
                step_number=3,
                tool_name="todo_store",
                tool_input={"action": "add", "text": "Calculation result: {step_2_output}"},
                reasoning="Store final result as todo using {step_2_output} variable",
            ),
        ],
        final_goal="Perform chained calculations and store final result as todo",
    ),
    complexity=ExampleComplexity.COMPLEX,
)


# Collection of all examples for easy access
ALL_EXAMPLES: list[FewShotExample] = [
    EXAMPLE_SIMPLE_CALCULATION,
    EXAMPLE_TODO_WORKFLOW,
    EXAMPLE_CALCULATION_THEN_TODO,
    EXAMPLE_MULTI_TODO_OPERATIONS,
    EXAMPLE_SIMPLE_LIST,
    EXAMPLE_COMPLETE_FIRST_TODO,
    EXAMPLE_DELETE_LAST_TODO,
    EXAMPLE_CHAINED_CALCULATIONS,
]


def get_examples_by_complexity(complexity: ExampleComplexity) -> list[FewShotExample]:
    """
    Filter examples by complexity level.

    Args:
        complexity: Desired complexity level

    Returns:
        List of examples matching complexity

    """
    return [ex for ex in ALL_EXAMPLES if ex.complexity == complexity]


def format_example_for_prompt(example: FewShotExample) -> str:
    """
    Format a few-shot example for inclusion in system prompt.

    Args:
        example: Example to format

    Returns:
        Formatted string showing prompt → reasoning → plan pattern

    """
    # Format steps as concise JSON
    steps_str = "[\n"
    for step in example.plan.steps:
        steps_str += f'  {{"step_number": {step.step_number}, "tool_name": "{step.tool_name}", '
        steps_str += f'"tool_input": {step.tool_input}, "reasoning": "{step.reasoning}"}},\n'
    steps_str += "]"

    return f"""Example:
User: "{example.prompt}"

Reasoning: {example.reasoning}

Plan:
{{
  "steps": {steps_str},
  "final_goal": "{example.plan.final_goal}"
}}
"""
