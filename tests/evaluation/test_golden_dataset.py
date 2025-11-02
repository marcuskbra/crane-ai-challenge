"""
Automated evaluation tests for golden dataset.

Systematically evaluates planner quality across 20+ diverse test cases,
measuring accuracy, consistency, and error handling capabilities.
"""

import asyncio
import time
from typing import Any

import pytest
from fastapi import status
from pydantic import BaseModel, ConfigDict, Field

from tests.evaluation.golden_dataset import (
    ALL_GOLDEN_CASES,
    ExpectedBehavior,
    GoldenTestCase,
    get_cases_by_behavior,
    get_cases_by_category,
)


class EvaluationResult(BaseModel):
    """
    Captures outcome of single golden test case evaluation.

    Attributes:
        test_id: Unique identifier of test case
        passed: Whether evaluation passed all criteria
        actual_status_code: HTTP status code returned
        actual_tools: List of tools used in execution
        actual_steps: Number of steps in execution
        latency_ms: Execution time in milliseconds
        errors: List of error messages (empty if successful)
        notes: Additional context about execution
    """

    test_id: str = Field(..., min_length=1, description="Test case identifier")
    passed: bool = Field(..., description="Overall pass/fail status")
    actual_status_code: int = Field(..., ge=100, le=599, description="HTTP status code")
    actual_tools: list[str] = Field(default_factory=list, description="Tools used")
    actual_steps: int = Field(default=0, ge=0, description="Number of steps executed")
    latency_ms: float = Field(..., ge=0.0, description="Execution time in milliseconds")
    errors: list[str] = Field(default_factory=list, description="Error messages")
    notes: str = Field(default="", description="Additional context")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class EvaluationSummary(BaseModel):
    """
    Aggregate evaluation metrics across all test cases.

    Attributes:
        total_cases: Total number of test cases evaluated
        passed: Number of test cases that passed
        failed: Number of test cases that failed
        success_rate: Percentage of tests passed (0.0 to 1.0)
        avg_latency_ms: Average execution time across all tests
        by_category: Pass rate by test category
        by_behavior: Pass rate by expected behavior
    """

    total_cases: int = Field(..., ge=0, description="Total test cases")
    passed: int = Field(..., ge=0, description="Tests passed")
    failed: int = Field(..., ge=0, description="Tests failed")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Pass rate")
    avg_latency_ms: float = Field(..., ge=0.0, description="Average latency")
    by_category: dict[str, float] = Field(description="Pass rate by category")
    by_behavior: dict[str, float] = Field(description="Pass rate by behavior")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


async def evaluate_test_case(
    test_case: GoldenTestCase,
    test_client: Any,
) -> EvaluationResult:
    """
    Execute single golden test case and evaluate results.

    Args:
        test_case: Golden test case to evaluate
        test_client: FastAPI test client

    Returns:
        EvaluationResult with pass/fail status and metrics

    """
    start_time = time.time()
    errors: list[str] = []
    passed = True

    try:
        # Execute the prompt through API
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": test_case.prompt},
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract actual execution data
        actual_status_code = response.status_code
        actual_tools: list[str] = []
        actual_steps = 0

        if response.status_code == status.HTTP_201_CREATED:
            response_data = response.json()

            # Extract tools from plan (defensive checks for None/missing keys)
            if (
                response_data is not None
                and "plan" in response_data
                and response_data["plan"] is not None
                and "steps" in response_data["plan"]
            ):
                actual_tools = [step["tool_name"].lower() for step in response_data["plan"]["steps"]]
                actual_steps = len(response_data["plan"]["steps"])

        # Validate expected behavior
        if test_case.expected_behavior == ExpectedBehavior.SUCCESS:
            if actual_status_code != status.HTTP_201_CREATED:
                errors.append(f"Expected success (201), got {actual_status_code}")
                passed = False

            # Validate expected tools
            if test_case.expected_tools:
                expected_tools_lower = [t.lower() for t in test_case.expected_tools]
                if actual_tools != expected_tools_lower:
                    errors.append(f"Tool mismatch: expected {expected_tools_lower}, got {actual_tools}")
                    passed = False

            # Validate step count range
            if test_case.expected_steps_min is not None:
                if actual_steps < test_case.expected_steps_min:
                    errors.append(f"Too few steps: expected >={test_case.expected_steps_min}, got {actual_steps}")
                    passed = False

            if test_case.expected_steps_max is not None:
                if actual_steps > test_case.expected_steps_max:
                    errors.append(f"Too many steps: expected <={test_case.expected_steps_max}, got {actual_steps}")
                    passed = False

        elif test_case.expected_behavior == ExpectedBehavior.ERROR_HANDLING:
            # Should return error status code (4xx or 5xx) with proper error message
            if actual_status_code < 400:
                errors.append(f"Expected error status (4xx/5xx), got {actual_status_code}")
                passed = False

        elif test_case.expected_behavior == ExpectedBehavior.GRACEFUL_FAILURE:
            # Should fail gracefully with clear error message
            if actual_status_code != status.HTTP_400_BAD_REQUEST:
                errors.append(f"Expected graceful failure (400), got {actual_status_code}")
                passed = False

        elif test_case.expected_behavior == ExpectedBehavior.CLARIFICATION:
            # Should request clarification (implementation-dependent)
            # For now, accept 400 status with clarification message
            if actual_status_code != status.HTTP_400_BAD_REQUEST:
                errors.append(f"Expected clarification request (400), got {actual_status_code}")
                passed = False

        return EvaluationResult(
            test_id=test_case.id,
            passed=passed,
            actual_status_code=actual_status_code,
            actual_tools=actual_tools,
            actual_steps=actual_steps,
            latency_ms=latency_ms,
            errors=errors,
            notes=test_case.notes,
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return EvaluationResult(
            test_id=test_case.id,
            passed=False,
            actual_status_code=500,
            actual_tools=[],
            actual_steps=0,
            latency_ms=latency_ms,
            errors=[f"Exception during evaluation: {e!s}"],
            notes=test_case.notes,
        )


def calculate_summary(results: list[EvaluationResult]) -> EvaluationSummary:
    """
    Calculate aggregate evaluation metrics from individual results.

    Args:
        results: List of evaluation results

    Returns:
        EvaluationSummary with aggregate metrics

    """
    total_cases = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total_cases - passed
    success_rate = passed / total_cases if total_cases > 0 else 0.0
    avg_latency_ms = sum(r.latency_ms for r in results) / total_cases if total_cases > 0 else 0.0

    # Calculate pass rate by category
    by_category: dict[str, float] = {}
    for category in ["simple_operations", "multi_step", "edge_cases", "error_handling", "boundary_conditions"]:
        category_cases = get_cases_by_category(category)
        if category_cases:
            category_results = [r for r in results if any(c.id == r.test_id for c in category_cases)]
            category_passed = sum(1 for r in category_results if r.passed)
            by_category[category] = category_passed / len(category_results) if category_results else 0.0

    # Calculate pass rate by behavior
    by_behavior: dict[str, float] = {}
    for behavior in ExpectedBehavior:
        behavior_cases = get_cases_by_behavior(behavior)
        if behavior_cases:
            behavior_results = [r for r in results if any(c.id == r.test_id for c in behavior_cases)]
            behavior_passed = sum(1 for r in behavior_results if r.passed)
            by_behavior[behavior.value] = behavior_passed / len(behavior_results) if behavior_results else 0.0

    return EvaluationSummary(
        total_cases=total_cases,
        passed=passed,
        failed=failed,
        success_rate=success_rate,
        avg_latency_ms=avg_latency_ms,
        by_category=by_category,
        by_behavior=by_behavior,
    )


# ============================================================================
# Pytest Tests
# ============================================================================


@pytest.mark.golden
@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", ALL_GOLDEN_CASES, ids=lambda tc: tc.id)
async def test_golden_case(test_case: GoldenTestCase, test_client):
    """
    Execute and evaluate single golden test case.

    This test is parameterized to run all 20+ golden test cases individually,
    allowing for detailed failure analysis and targeted debugging.
    """
    # Wait briefly for any previous runs to complete
    await asyncio.sleep(0.1)

    result = await evaluate_test_case(test_case, test_client)

    # Assert with detailed failure message
    failure_message = "\n".join(
        [
            f"\n{'=' * 70}",
            f"Golden Test Case Failed: {test_case.id}",
            f"{'=' * 70}",
            f"Prompt: {test_case.prompt}",
            f"Category: {test_case.category}",
            f"Expected Behavior: {test_case.expected_behavior.value}",
            f"Expected Tools: {test_case.expected_tools}",
            f"Expected Steps: {test_case.expected_steps_min}-{test_case.expected_steps_max}",
            "",
            f"Actual Status Code: {result.actual_status_code}",
            f"Actual Tools: {result.actual_tools}",
            f"Actual Steps: {result.actual_steps}",
            f"Latency: {result.latency_ms:.2f}ms",
            "",
            "Errors:",
            *[f"  - {error}" for error in result.errors],
            "",
            f"Notes: {test_case.notes}",
            f"{'=' * 70}",
        ]
    )

    assert result.passed, failure_message


@pytest.mark.golden
@pytest.mark.asyncio
async def test_golden_dataset_summary(test_client):
    """
    Execute all golden test cases and generate aggregate evaluation report.

    This test provides overall quality metrics and identifies patterns in
    failures across categories and expected behaviors.
    """
    # Execute all test cases
    results: list[EvaluationResult] = []
    for test_case in ALL_GOLDEN_CASES:
        # Wait briefly between tests to avoid overwhelming the system
        await asyncio.sleep(0.1)
        result = await evaluate_test_case(test_case, test_client)
        results.append(result)

    # Calculate summary metrics
    summary = calculate_summary(results)

    # Print detailed report
    print("\n" + "=" * 70)
    print("GOLDEN DATASET EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Cases: {summary.total_cases}")
    print(f"Passed: {summary.passed}")
    print(f"Failed: {summary.failed}")
    print(f"Success Rate: {summary.success_rate:.1%}")
    print(f"Average Latency: {summary.avg_latency_ms:.2f}ms")
    print()
    print("Pass Rate by Category:")
    for category, rate in summary.by_category.items():
        print(f"  {category}: {rate:.1%}")
    print()
    print("Pass Rate by Expected Behavior:")
    for behavior, rate in summary.by_behavior.items():
        print(f"  {behavior}: {rate:.1%}")
    print()

    # Print failures
    failures = [r for r in results if not r.passed]
    if failures:
        print("Failed Test Cases:")
        for result in failures:
            print(f"  - {result.test_id}:")
            for error in result.errors:
                print(f"      {error}")
        print()

    print("=" * 70)

    # Assert minimum quality thresholds
    # Note: These thresholds are initial targets and may need adjustment
    # as the system improves or golden dataset difficulty increases
    assert summary.success_rate >= 0.7, f"Success rate {summary.success_rate:.1%} below minimum threshold of 70%"


@pytest.mark.golden
@pytest.mark.asyncio
async def test_golden_simple_operations(test_client):
    """Test all simple operation cases (should have high success rate)."""
    simple_cases = get_cases_by_category("simple_operations")
    results = []

    for test_case in simple_cases:
        await asyncio.sleep(0.1)
        result = await evaluate_test_case(test_case, test_client)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    success_rate = passed / len(results) if results else 0.0

    assert success_rate >= 0.8, f"Simple operations success rate {success_rate:.1%} below 80% threshold"


@pytest.mark.golden
@pytest.mark.asyncio
async def test_golden_multi_step_workflows(test_client):
    """Test all multi-step workflow cases (complex coordination)."""
    multi_step_cases = get_cases_by_category("multi_step")
    results = []

    for test_case in multi_step_cases:
        await asyncio.sleep(0.1)
        result = await evaluate_test_case(test_case, test_client)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    success_rate = passed / len(results) if results else 0.0

    assert success_rate >= 0.6, f"Multi-step workflows success rate {success_rate:.1%} below 60% threshold"


@pytest.mark.golden
@pytest.mark.asyncio
async def test_golden_edge_cases(test_client):
    """Test all edge cases (unusual inputs and boundary conditions)."""
    edge_cases = get_cases_by_category("edge_cases")
    results = []

    for test_case in edge_cases:
        await asyncio.sleep(0.1)
        result = await evaluate_test_case(test_case, test_client)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    success_rate = passed / len(results) if results else 0.0

    # Edge cases expected to have lower success rate
    assert success_rate >= 0.5, f"Edge cases success rate {success_rate:.1%} below 50% threshold"


@pytest.mark.golden
@pytest.mark.asyncio
async def test_golden_error_handling(test_client):
    """Test all error handling cases (graceful failure validation)."""
    error_cases = get_cases_by_category("error_handling")
    results = []

    for test_case in error_cases:
        await asyncio.sleep(0.1)
        result = await evaluate_test_case(test_case, test_client)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    success_rate = passed / len(results) if results else 0.0

    assert success_rate >= 0.7, f"Error handling success rate {success_rate:.1%} below 70% threshold"


@pytest.mark.golden
@pytest.mark.asyncio
async def test_evaluation_result_model():
    """Test EvaluationResult model validation."""
    result = EvaluationResult(
        test_id="test_001",
        passed=True,
        actual_status_code=201,
        actual_tools=["calculator"],
        actual_steps=1,
        latency_ms=125.5,
        errors=[],
        notes="Test notes",
    )

    assert result.test_id == "test_001"
    assert result.passed is True
    assert result.actual_status_code == 201
    assert result.actual_tools == ["calculator"]
    assert result.actual_steps == 1
    assert result.latency_ms == 125.5
    assert result.errors == []
    assert result.notes == "Test notes"


@pytest.mark.golden
@pytest.mark.asyncio
async def test_evaluation_summary_calculation():
    """Test EvaluationSummary calculation from results."""
    results = [
        EvaluationResult(
            test_id="test_001",
            passed=True,
            actual_status_code=201,
            actual_tools=["calculator"],
            actual_steps=1,
            latency_ms=100.0,
            errors=[],
        ),
        EvaluationResult(
            test_id="test_002",
            passed=False,
            actual_status_code=400,
            actual_tools=[],
            actual_steps=0,
            latency_ms=50.0,
            errors=["Test error"],
        ),
    ]

    summary = calculate_summary(results)

    assert summary.total_cases == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.success_rate == 0.5
    assert summary.avg_latency_ms == 75.0
