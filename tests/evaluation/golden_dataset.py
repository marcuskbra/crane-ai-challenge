"""
Golden dataset for systematic agent quality evaluation.

This module contains 20+ test cases covering diverse scenarios to systematically
measure planner quality, consistency, and error handling across different complexity levels.

Test Categories:
- Simple Operations: Basic single-tool operations
- Multi-Step Workflows: Complex coordinated operations
- Edge Cases: Unusual inputs and boundary conditions
- Error Handling: Invalid inputs and impossible requests
- Boundary Conditions: Limits and special characters
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ExpectedBehavior(str, Enum):
    """Expected system behavior for test case."""

    SUCCESS = "success"
    ERROR_HANDLING = "error_handling"
    GRACEFUL_FAILURE = "graceful_failure"
    CLARIFICATION = "clarification"


class GoldenTestCase(BaseModel):
    """
    Single golden dataset test case.

    Attributes:
        id: Unique test case identifier
        category: Test category for organization
        prompt: User's natural language request
        expected_tools: List of tools that should be used
        expected_steps_min: Minimum number of steps expected
        expected_steps_max: Maximum number of steps expected
        expected_behavior: Expected system behavior
        notes: Additional context or explanation

    """

    id: str = Field(..., pattern=r"^[a-z0-9_]+$", description="Unique test case ID")
    category: str = Field(..., min_length=1, description="Test category")
    prompt: str = Field(..., description="User prompt")  # No min_length to allow testing empty prompts
    expected_tools: list[str] = Field(default_factory=list, description="Expected tools to be used")
    expected_steps_min: int | None = Field(default=None, ge=1, description="Minimum expected steps")
    expected_steps_max: int | None = Field(default=None, ge=1, description="Maximum expected steps")
    expected_behavior: ExpectedBehavior = Field(description="Expected system behavior")
    notes: str = Field(default="", description="Additional context")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


# ============================================================================
# Simple Operations (5 cases)
# ============================================================================

GOLDEN_SIMPLE_001 = GoldenTestCase(
    id="simple_001_basic_calc",
    category="simple_operations",
    prompt="calculate 2 + 2",
    expected_tools=["calculator"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Basic arithmetic operation with single tool",
)

GOLDEN_SIMPLE_002 = GoldenTestCase(
    id="simple_002_complex_calc",
    category="simple_operations",
    prompt="calculate (10 + 5) * 2",
    expected_tools=["calculator"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Arithmetic with parentheses and operator precedence",
)

GOLDEN_SIMPLE_003 = GoldenTestCase(
    id="simple_003_add_todo",
    category="simple_operations",
    prompt="add a task to buy milk",
    expected_tools=["todo_store"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Basic todo addition with single tool",
)

GOLDEN_SIMPLE_004 = GoldenTestCase(
    id="simple_004_list_todos",
    category="simple_operations",
    prompt="show me my todos",
    expected_tools=["todo_store"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Simple todo retrieval operation",
)

GOLDEN_SIMPLE_005 = GoldenTestCase(
    id="simple_005_division",
    category="simple_operations",
    prompt="what is 100 divided by 4",
    expected_tools=["calculator"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Division operation with natural language phrasing",
)

# ============================================================================
# Multi-Step Workflows (5 cases)
# ============================================================================

GOLDEN_MULTI_001 = GoldenTestCase(
    id="multi_001_add_and_list",
    category="multi_step",
    prompt="add a todo to buy groceries and then show me all my tasks",
    expected_tools=["todo_store", "todo_store"],
    expected_steps_min=2,
    expected_steps_max=2,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Sequential todo operations - add followed by list",
)

GOLDEN_MULTI_002 = GoldenTestCase(
    id="multi_002_calc_then_todo",
    category="multi_step",
    prompt="calculate 3 * 4 and add the result as a todo",
    expected_tools=["calculator", "todo_store"],
    expected_steps_min=2,
    expected_steps_max=2,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Cross-tool workflow - calculator result used in todo",
)

GOLDEN_MULTI_003 = GoldenTestCase(
    id="multi_003_multiple_additions",
    category="multi_step",
    prompt="add todos for buying milk, paying bills, and calling dentist",
    expected_tools=["todo_store", "todo_store", "todo_store"],
    expected_steps_min=3,
    expected_steps_max=3,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Multiple sequential todo additions from one prompt",
)

GOLDEN_MULTI_004 = GoldenTestCase(
    id="multi_004_calc_chain",
    category="multi_step",
    prompt="calculate 5 + 3, then multiply that by 2",
    expected_tools=["calculator", "calculator"],
    expected_steps_min=2,
    expected_steps_max=2,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Chained calculations with dependency between steps",
)

GOLDEN_MULTI_005 = GoldenTestCase(
    id="multi_005_mixed_operations",
    category="multi_step",
    prompt="add task to review budget, calculate 1500 * 0.2, and list all tasks",
    expected_tools=["todo_store", "calculator", "todo_store"],
    expected_steps_min=3,
    expected_steps_max=3,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Complex workflow mixing different tool types",
)

# ============================================================================
# Edge Cases (5 cases)
# ============================================================================

GOLDEN_EDGE_001 = GoldenTestCase(
    id="edge_001_empty_prompt",
    category="edge_cases",
    prompt="",
    expected_tools=[],
    expected_steps_min=None,
    expected_steps_max=None,
    expected_behavior=ExpectedBehavior.ERROR_HANDLING,
    notes="Empty prompt should be rejected with clear error",
)

GOLDEN_EDGE_002 = GoldenTestCase(
    id="edge_002_whitespace_only",
    category="edge_cases",
    prompt="   ",
    expected_tools=[],
    expected_steps_min=None,
    expected_steps_max=None,
    expected_behavior=ExpectedBehavior.ERROR_HANDLING,
    notes="Whitespace-only prompt should be handled gracefully",
)

GOLDEN_EDGE_003 = GoldenTestCase(
    id="edge_003_very_long_expression",
    category="edge_cases",
    prompt="calculate " + " + ".join(str(i) for i in range(1, 101)),
    expected_tools=["calculator"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Very long arithmetic expression (1+2+3+...+100)",
)

GOLDEN_EDGE_004 = GoldenTestCase(
    id="edge_004_unicode_todo",
    category="edge_cases",
    prompt="add todo: 买牛奶 🥛 and review",
    expected_tools=["todo_store"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Unicode characters and emoji in todo text",
)

GOLDEN_EDGE_005 = GoldenTestCase(
    id="edge_005_ambiguous_request",
    category="edge_cases",
    prompt="do the thing",
    expected_tools=[],
    expected_steps_min=None,
    expected_steps_max=None,
    expected_behavior=ExpectedBehavior.GRACEFUL_FAILURE,
    notes="Vague request with no clear actionable intent",
)

# ============================================================================
# Error Handling (3 cases)
# ============================================================================

GOLDEN_ERROR_001 = GoldenTestCase(
    id="error_001_invalid_math",
    category="error_handling",
    prompt="calculate abc + def",
    expected_tools=["calculator"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.ERROR_HANDLING,
    notes="Invalid mathematical expression should fail gracefully",
)

GOLDEN_ERROR_002 = GoldenTestCase(
    id="error_002_impossible_task",
    category="error_handling",
    prompt="delete the internet",
    expected_tools=[],
    expected_steps_min=None,
    expected_steps_max=None,
    expected_behavior=ExpectedBehavior.GRACEFUL_FAILURE,
    notes="Impossible/nonsensical request should be rejected",
)

GOLDEN_ERROR_003 = GoldenTestCase(
    id="error_003_unknown_tool",
    category="error_handling",
    prompt="send email to boss",
    expected_tools=[],
    expected_steps_min=None,
    expected_steps_max=None,
    expected_behavior=ExpectedBehavior.GRACEFUL_FAILURE,
    notes="Request for unavailable tool should fail gracefully",
)

# ============================================================================
# Boundary Conditions (2 cases)
# ============================================================================

GOLDEN_BOUNDARY_001 = GoldenTestCase(
    id="boundary_001_division_by_zero",
    category="boundary_conditions",
    prompt="calculate 10 / 0",
    expected_tools=["calculator"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.ERROR_HANDLING,
    notes="Division by zero should be handled with error",
)

GOLDEN_BOUNDARY_002 = GoldenTestCase(
    id="boundary_002_very_large_number",
    category="boundary_conditions",
    prompt="calculate 999999999999 * 999999999999",
    expected_tools=["calculator"],
    expected_steps_min=1,
    expected_steps_max=1,
    expected_behavior=ExpectedBehavior.SUCCESS,
    notes="Very large number calculations should work correctly",
)

# ============================================================================
# Collection of all test cases
# ============================================================================

ALL_GOLDEN_CASES: list[GoldenTestCase] = [
    # Simple Operations (5)
    GOLDEN_SIMPLE_001,
    GOLDEN_SIMPLE_002,
    GOLDEN_SIMPLE_003,
    GOLDEN_SIMPLE_004,
    GOLDEN_SIMPLE_005,
    # Multi-Step Workflows (5)
    GOLDEN_MULTI_001,
    GOLDEN_MULTI_002,
    GOLDEN_MULTI_003,
    GOLDEN_MULTI_004,
    GOLDEN_MULTI_005,
    # Edge Cases (5)
    GOLDEN_EDGE_001,
    GOLDEN_EDGE_002,
    GOLDEN_EDGE_003,
    GOLDEN_EDGE_004,
    GOLDEN_EDGE_005,
    # Error Handling (3)
    GOLDEN_ERROR_001,
    GOLDEN_ERROR_002,
    GOLDEN_ERROR_003,
    # Boundary Conditions (2)
    GOLDEN_BOUNDARY_001,
    GOLDEN_BOUNDARY_002,
]


def get_cases_by_category(category: str) -> list[GoldenTestCase]:
    """
    Filter test cases by category.

    Args:
        category: Category to filter by

    Returns:
        List of test cases in the specified category

    """
    return [case for case in ALL_GOLDEN_CASES if case.category == category]


def get_cases_by_behavior(behavior: ExpectedBehavior) -> list[GoldenTestCase]:
    """
    Filter test cases by expected behavior.

    Args:
        behavior: Expected behavior to filter by

    Returns:
        List of test cases with the specified expected behavior

    """
    return [case for case in ALL_GOLDEN_CASES if case.expected_behavior == behavior]
