"""
Tests for few-shot prompt engineering examples.

Verifies:
- Example model validation and structure
- Example collection integrity
- Complexity filtering
- Prompt formatting for LLM injection
- Integration with LLM planner
"""

import pytest
from pydantic import ValidationError

from challenge.models.plan import Plan, PlanStep
from challenge.planner.examples import (
    ALL_EXAMPLES,
    EXAMPLE_CALCULATION_THEN_TODO,
    EXAMPLE_MULTI_TODO_OPERATIONS,
    EXAMPLE_SIMPLE_CALCULATION,
    EXAMPLE_SIMPLE_LIST,
    EXAMPLE_TODO_WORKFLOW,
    FewShotExample,
    format_example_for_prompt,
    get_examples_by_complexity,
)
from challenge.planner.llm_planner import LLMPlanner


class TestFewShotExample:
    """Tests for FewShotExample model validation and structure."""

    def test_few_shot_example_valid(self):
        """Test creating valid few-shot example."""
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "2 + 2"},
                    reasoning="Calculate sum",
                )
            ],
            final_goal="Add 2 and 2",
        )

        example = FewShotExample(
            prompt="calculate 2 + 2",
            reasoning="Simple arithmetic",
            plan=plan,
            complexity="simple",
        )

        assert example.prompt == "calculate 2 + 2"
        assert example.reasoning == "Simple arithmetic"
        assert example.plan == plan
        assert example.complexity == "simple"

    def test_few_shot_example_invalid_complexity(self):
        """Test that invalid complexity level raises ValidationError."""
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "2 + 2"},
                    reasoning="Calculate sum",
                )
            ],
            final_goal="Add 2 and 2",
        )

        with pytest.raises(ValidationError) as exc_info:
            # Use model_validate to bypass type checking for invalid data
            FewShotExample.model_validate(
                {
                    "prompt": "calculate 2 + 2",
                    "reasoning": "Simple arithmetic",
                    "plan": plan,
                    "complexity": "invalid_complexity",  # Invalid value
                }
            )

        assert "complexity" in str(exc_info.value)

    def test_few_shot_example_empty_prompt(self):
        """Test that empty prompt raises ValidationError."""
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "2 + 2"},
                    reasoning="Calculate sum",
                )
            ],
            final_goal="Add 2 and 2",
        )

        with pytest.raises(ValidationError) as exc_info:
            FewShotExample(
                prompt="",  # Empty string
                reasoning="Simple arithmetic",
                plan=plan,
                complexity="simple",
            )

        assert "prompt" in str(exc_info.value)

    def test_few_shot_example_extra_fields_forbidden(self):
        """Test that extra fields raise ValidationError due to strict mode."""
        plan = Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "2 + 2"},
                    reasoning="Calculate sum",
                )
            ],
            final_goal="Add 2 and 2",
        )

        with pytest.raises(ValidationError) as exc_info:
            # Use model_validate to bypass type checking for invalid data
            FewShotExample.model_validate(
                {
                    "prompt": "calculate 2 + 2",
                    "reasoning": "Simple arithmetic",
                    "plan": plan,
                    "complexity": "simple",
                    "extra_field": "should fail",  # Extra field not allowed
                }
            )

        assert "extra_field" in str(exc_info.value).lower() or "extra fields" in str(exc_info.value).lower()


class TestExampleCollection:
    """Tests for example collection integrity and structure."""

    def test_all_examples_count(self):
        """Test that we have exactly 8 examples."""
        assert len(ALL_EXAMPLES) == 8

    def test_all_examples_are_valid(self):
        """Test that all examples are valid FewShotExample instances."""
        for example in ALL_EXAMPLES:
            assert isinstance(example, FewShotExample)
            assert len(example.prompt) > 0
            assert len(example.reasoning) > 0
            assert isinstance(example.plan, Plan)
            assert len(example.plan.steps) > 0
            assert example.complexity in ["simple", "moderate", "complex"]

    def test_all_examples_have_unique_prompts(self):
        """Test that all examples have unique prompts."""
        prompts = [ex.prompt for ex in ALL_EXAMPLES]
        assert len(prompts) == len(set(prompts)), "Duplicate prompts found"

    def test_example_complexity_distribution(self):
        """Test that we have examples across all complexity levels."""
        complexities = {ex.complexity for ex in ALL_EXAMPLES}
        assert "simple" in complexities
        assert "moderate" in complexities
        assert "complex" in complexities

    def test_predefined_examples_exist(self):
        """Test that all predefined example constants are accessible."""
        assert EXAMPLE_SIMPLE_CALCULATION is not None
        assert EXAMPLE_TODO_WORKFLOW is not None
        assert EXAMPLE_CALCULATION_THEN_TODO is not None
        assert EXAMPLE_MULTI_TODO_OPERATIONS is not None
        assert EXAMPLE_SIMPLE_LIST is not None

    def test_example_simple_calculation_structure(self):
        """Test specific structure of simple calculation example."""
        assert EXAMPLE_SIMPLE_CALCULATION.complexity == "simple"
        assert len(EXAMPLE_SIMPLE_CALCULATION.plan.steps) == 1
        assert EXAMPLE_SIMPLE_CALCULATION.plan.steps[0].tool_name == "calculator"

    def test_example_todo_workflow_structure(self):
        """Test specific structure of todo workflow example."""
        assert EXAMPLE_TODO_WORKFLOW.complexity == "moderate"
        assert len(EXAMPLE_TODO_WORKFLOW.plan.steps) == 2
        assert all(step.tool_name == "todo_store" for step in EXAMPLE_TODO_WORKFLOW.plan.steps)

    def test_example_complex_operations_structure(self):
        """Test specific structure of complex operations example."""
        assert EXAMPLE_MULTI_TODO_OPERATIONS.complexity == "complex"
        assert len(EXAMPLE_MULTI_TODO_OPERATIONS.plan.steps) == 3
        assert all(step.tool_name == "todo_store" for step in EXAMPLE_MULTI_TODO_OPERATIONS.plan.steps)


class TestComplexityFiltering:
    """Tests for filtering examples by complexity level."""

    def test_get_simple_examples(self):
        """Test filtering simple examples."""
        simple_examples = get_examples_by_complexity("simple")
        assert len(simple_examples) >= 1
        assert all(ex.complexity == "simple" for ex in simple_examples)

    def test_get_moderate_examples(self):
        """Test filtering moderate examples."""
        moderate_examples = get_examples_by_complexity("moderate")
        assert len(moderate_examples) >= 1
        assert all(ex.complexity == "moderate" for ex in moderate_examples)

    def test_get_complex_examples(self):
        """Test filtering complex examples."""
        complex_examples = get_examples_by_complexity("complex")
        assert len(complex_examples) >= 1
        assert all(ex.complexity == "complex" for ex in complex_examples)

    def test_filtering_returns_subset(self):
        """Test that filtered results are subsets of ALL_EXAMPLES."""
        simple = get_examples_by_complexity("simple")
        moderate = get_examples_by_complexity("moderate")
        complex_ex = get_examples_by_complexity("complex")

        total_filtered = len(simple) + len(moderate) + len(complex_ex)
        assert total_filtered == len(ALL_EXAMPLES)


class TestPromptFormatting:
    """Tests for formatting examples for LLM prompt injection."""

    def test_format_example_contains_key_sections(self):
        """Test that formatted example contains all required sections."""
        formatted = format_example_for_prompt(EXAMPLE_SIMPLE_CALCULATION)

        assert "Example:" in formatted
        assert "User:" in formatted
        assert "Reasoning:" in formatted
        assert "Plan:" in formatted
        assert EXAMPLE_SIMPLE_CALCULATION.prompt in formatted
        assert "steps" in formatted
        assert "final_goal" in formatted

    def test_format_example_includes_step_details(self):
        """Test that formatted example includes step information."""
        formatted = format_example_for_prompt(EXAMPLE_TODO_WORKFLOW)

        # Should include step details
        assert "step_number" in formatted
        assert "tool_name" in formatted
        assert "tool_input" in formatted
        assert "reasoning" in formatted
        assert "todo_store" in formatted

    def test_format_example_preserves_structure(self):
        """Test that formatted example preserves JSON-like structure."""
        formatted = format_example_for_prompt(EXAMPLE_CALCULATION_THEN_TODO)

        # Should look like valid JSON structure
        assert "{" in formatted
        assert "}" in formatted
        assert "[" in formatted
        assert "]" in formatted

    def test_format_all_examples_without_errors(self):
        """Test that all examples can be formatted without errors."""
        for example in ALL_EXAMPLES:
            formatted = format_example_for_prompt(example)
            assert len(formatted) > 0
            assert example.prompt in formatted


class TestLLMPlannerIntegration:
    """Tests for integration of examples with LLM planner."""

    def test_llm_planner_with_examples_enabled(self):
        """Test that LLM planner includes examples when enabled."""
        planner = LLMPlanner(use_examples=True, api_key="dummy-key-for-testing")
        system_prompt = planner._system_prompt()

        # Should include example section
        assert "examples of good planning patterns" in system_prompt.lower() or "example:" in system_prompt.lower()

        # Should include at least one example prompt
        example_prompts = [ex.prompt for ex in ALL_EXAMPLES]
        assert any(prompt in system_prompt for prompt in example_prompts)

    def test_llm_planner_with_examples_disabled(self):
        """Test that LLM planner excludes few-shot examples when disabled."""
        planner = LLMPlanner(use_examples=False, api_key="dummy-key-for-testing")
        system_prompt = planner._system_prompt()

        # Should NOT include few-shot examples section header
        assert "Here are examples of good planning patterns" not in system_prompt

        # Should NOT include the reasoning section from formatted examples
        # (format_example_for_prompt includes "Reasoning: {example.reasoning}")
        # Documentation examples don't have this "Reasoning:" label
        example_reasonings = [ex.reasoning.strip()[:50] for ex in ALL_EXAMPLES]  # First 50 chars
        assert not any(reasoning in system_prompt for reasoning in example_reasonings)

    def test_llm_planner_default_uses_examples(self):
        """Test that LLM planner uses examples by default."""
        planner = LLMPlanner(api_key="dummy-key-for-testing")  # No explicit use_examples parameter
        assert planner.use_examples is True

        system_prompt = planner._system_prompt()
        assert len(system_prompt) > 1000  # Should be longer with examples

    def test_system_prompt_length_difference(self):
        """Test that examples significantly increase system prompt length."""
        planner_without = LLMPlanner(use_examples=False, api_key="dummy-key-for-testing")
        planner_with = LLMPlanner(use_examples=True, api_key="dummy-key-for-testing")

        prompt_without = planner_without._system_prompt()
        prompt_with = planner_with._system_prompt()

        # Prompt with examples should be significantly longer
        assert len(prompt_with) > len(prompt_without) * 1.5
