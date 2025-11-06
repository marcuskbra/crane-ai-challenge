"""Tests for LLM-based planner with mocked OpenAI responses."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from challenge.models.plan import Plan
from challenge.models.run import ExecutionStep
from challenge.orchestrator.execution_context import ExecutionContext
from challenge.planner.llm_planner import LLMPlanner


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("challenge.planner.llm_planner.AsyncOpenAI") as mock:
        yield mock


@pytest.mark.asyncio
async def test_llm_planner_success(mock_openai):
    """Test successful LLM planning with structured output."""
    # Mock successful OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"steps": [{"step_number": 1, "tool_name": "calculator", "tool_input": {"expression": "2+3"}, "reasoning": "Calculate sum"}], "final_goal": "Calculate 2+3"}'
            )
        )
    ]
    mock_response.usage = MagicMock(total_tokens=125)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test planner
    planner = LLMPlanner()
    plan = await planner.create_plan("calculate 2+3")

    # Assertions
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].tool_input.expression == "2+3"
    assert planner.last_token_count == 125


@pytest.mark.asyncio
async def test_llm_planner_multi_step(mock_openai):
    """Test LLM planning with multiple steps."""
    # Mock response with multiple steps
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"steps": [{"step_number": 1, "tool_name": "calculator", "tool_input": {"expression": "2+3"}, "reasoning": "Calculate sum"}, {"step_number": 2, "tool_name": "todo_store", "tool_input": {"action": "add", "text": "Buy milk"}, "reasoning": "Add todo item"}], "final_goal": "Calculate and add todo"}'
            )
        )
    ]
    mock_response.usage = MagicMock(total_tokens=200)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test planner
    planner = LLMPlanner()
    plan = await planner.create_plan("calculate 2+3 and add todo Buy milk")

    # Assertions
    assert len(plan.steps) == 2
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[1].tool_name == "todo_store"
    assert plan.steps[1].tool_input.action == "add"
    assert plan.steps[1].tool_input.text == "Buy milk"


@pytest.mark.asyncio
async def test_llm_planner_fallback_on_api_error(mock_openai):
    """Test graceful fallback to pattern-based on API failure."""
    # Mock API failure
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API rate limit exceeded"))

    # Test planner with fallback
    planner = LLMPlanner()
    plan = await planner.create_plan("calculate 2+3")

    # Should fall back to pattern-based planner
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "calculator"
    # Token count should be 0 since LLM wasn't used
    assert planner.last_token_count == 0


@pytest.mark.asyncio
async def test_llm_planner_fallback_on_invalid_json(mock_openai):
    """Test fallback when LLM returns invalid JSON."""
    # Mock response with invalid JSON
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="invalid json {"))]
    mock_response.usage = MagicMock(total_tokens=50)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test planner
    planner = LLMPlanner()
    plan = await planner.create_plan("calculate 2+3")

    # Should fall back
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_llm_planner_fallback_on_invalid_schema(mock_openai):
    """Test fallback when LLM returns valid JSON but invalid schema."""
    # Mock response with valid JSON but missing required fields
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"invalid": "schema"}'))]
    mock_response.usage = MagicMock(total_tokens=60)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test planner
    planner = LLMPlanner()
    plan = await planner.create_plan("calculate 2+3")

    # Should fall back
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_cost_tracking(mock_openai):
    """Test token usage and cost tracking."""
    # Mock response with valid plan
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"steps": [{"step_number": 1, "tool_name": "calculator", "tool_input": {"expression": "1+1"}, "reasoning": "test"}], "final_goal": "test"}'
            )
        )
    ]
    mock_response.usage = MagicMock(total_tokens=200)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test planner
    planner = LLMPlanner()
    await planner.create_plan("calculate 1+1")

    # Check cost estimate (now returns CostEstimate model)
    cost_info = planner.get_cost_estimate()
    assert cost_info.tokens == 200
    assert cost_info.model == "gpt-4o-mini"
    assert cost_info.estimated_cost_usd > 0
    assert cost_info.cost_per_1k_tokens == 0.00015


@pytest.mark.asyncio
async def test_temperature_configuration(mock_openai):
    """Test that temperature can be configured."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"steps": [{"step_number": 1, "tool_name": "calculator", "tool_input": {"expression": "1+1"}, "reasoning": "Add"}], "final_goal": "test"}'
            )
        )
    ]
    mock_response.usage = MagicMock(total_tokens=100)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test with custom temperature
    planner = LLMPlanner(temperature=0.5)
    await planner.create_plan("calculate 1+1")

    # Verify temperature was passed to API call
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 0.5


@pytest.mark.asyncio
async def test_model_configuration(mock_openai):
    """Test that model can be configured."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"steps": [{"step_number": 1, "tool_name": "calculator", "tool_input": {"expression": "1+1"}, "reasoning": "Add"}], "final_goal": "test"}'
            )
        )
    ]
    mock_response.usage = MagicMock(total_tokens=150)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test with custom model
    planner = LLMPlanner(model="gpt-4o")
    await planner.create_plan("calculate 1+1")

    # Verify model was passed to API call
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_system_prompt_includes_tool_descriptions(mock_openai):
    """Test that system prompt includes tool documentation."""
    planner = LLMPlanner()
    system_prompt = planner._system_prompt()

    # Verify tool documentation is present
    assert "calculator" in system_prompt
    assert "todo_store" in system_prompt
    assert "expression" in system_prompt
    assert "action" in system_prompt
    assert "add" in system_prompt
    assert "list" in system_prompt


@pytest.mark.asyncio
async def test_empty_prompt_fallback(mock_openai):
    """Test that empty prompts use fallback and raise ValueError."""
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Should not be called"))

    planner = LLMPlanner()

    # Empty prompt should fallback and then raise from pattern planner
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await planner.create_plan("")


@pytest.mark.asyncio
async def test_todo_operations(mock_openai):
    """Test LLM planning for to-do operations."""
    # Mock response for adding a to-do
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"steps": [{"step_number": 1, "tool_name": "todo_store", "tool_input": {"action": "add", "text": "Buy groceries"}, "reasoning": "Add new todo item"}], "final_goal": "Add todo"}'
            )
        )
    ]
    mock_response.usage = MagicMock(total_tokens=180)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test planner
    planner = LLMPlanner()
    plan = await planner.create_plan("add todo Buy groceries")

    # Assertions
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "todo_store"
    assert plan.steps[0].tool_input.action == "add"
    assert plan.steps[0].tool_input.text == "Buy groceries"


@pytest.mark.asyncio
async def test_complex_multi_step_with_variable_resolution(mock_openai):
    """Test complex multi-step plan with variable resolution across steps.

    Tests the prompt: "Calculate (42 * 8) + 15, then use the result and multiply by 2,
    and add the result as a todo"

    This validates:
    1. LLM can create a multi-step plan with variable references
    2. Variables are properly referenced using {step_N_output} syntax
    3. Plan structure correctly chains calculations with todo creation
    """
    # Mock LLM response with multi-step plan using variable references
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""{
                    "steps": [
                        {
                            "step_number": 1,
                            "tool_name": "calculator",
                            "tool_input": {"expression": "(42 * 8) + 15"},
                            "reasoning": "Calculate the initial expression (42 * 8) + 15 which equals 351"
                        },
                        {
                            "step_number": 2,
                            "tool_name": "calculator",
                            "tool_input": {"expression": "{step_1_output} * 2"},
                            "reasoning": "Multiply the result from step 1 by 2"
                        },
                        {
                            "step_number": 3,
                            "tool_name": "todo_store",
                            "tool_input": {
                                "action": "add",
                                "text": "Result: {step_2_output}"
                            },
                            "reasoning": "Add the final calculation result as a todo"
                        }
                    ],
                    "final_goal": "Calculate (42 * 8) + 15, multiply by 2, and add as todo"
                }"""
            )
        )
    ]
    mock_response.usage = MagicMock(total_tokens=350)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Test planner
    planner = LLMPlanner()
    plan = await planner.create_plan(
        "Calculate (42 * 8) + 15, then use the result and multiply by 2, and add the result as a todo"
    )

    # Assertions - Plan structure
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 3, "Should have 3 steps: calc, calc, todo"

    # Step 1: Initial calculation
    assert plan.steps[0].step_number == 1
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[0].tool_input.expression == "(42 * 8) + 15"
    assert "initial expression" in plan.steps[0].reasoning.lower()

    # Step 2: Multiply by 2 using variable reference
    assert plan.steps[1].step_number == 2
    assert plan.steps[1].tool_name == "calculator"
    assert "{step_1_output}" in plan.steps[1].tool_input.expression
    assert plan.steps[1].tool_input.expression == "{step_1_output} * 2"
    assert "multiply" in plan.steps[1].reasoning.lower()

    # Step 3: Add result as todo using variable reference
    assert plan.steps[2].step_number == 3
    assert plan.steps[2].tool_name == "todo_store"
    assert plan.steps[2].tool_input.action == "add"
    assert "{step_2_output}" in plan.steps[2].tool_input.text
    assert "Result: {step_2_output}" == plan.steps[2].tool_input.text
    assert "todo" in plan.steps[2].reasoning.lower()

    # Verify token tracking
    assert planner.last_token_count == 350


@pytest.mark.asyncio
async def test_complex_multi_step_execution_with_context(mock_openai):
    """Test that complex multi-step plan executes correctly with ExecutionContext.

    This is an integration test that validates variable resolution actually works
    when the plan is executed by the orchestrator with ExecutionContext.
    """
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="""{
                    "steps": [
                        {
                            "step_number": 1,
                            "tool_name": "calculator",
                            "tool_input": {"expression": "(42 * 8) + 15"},
                            "reasoning": "Calculate initial value"
                        },
                        {
                            "step_number": 2,
                            "tool_name": "calculator",
                            "tool_input": {"expression": "{step_1_output} * 2"},
                            "reasoning": "Multiply by 2"
                        },
                        {
                            "step_number": 3,
                            "tool_name": "todo_store",
                            "tool_input": {
                                "action": "add",
                                "text": "Result: {step_2_output}"
                            },
                            "reasoning": "Add as todo"
                        }
                    ],
                    "final_goal": "Multi-step calculation with todo"
                }"""
            )
        )
    ]
    mock_response.usage = MagicMock(total_tokens=300)

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Create plan
    planner = LLMPlanner()
    plan = await planner.create_plan("Calculate (42 * 8) + 15, multiply by 2, add as todo")

    # Simulate execution with ExecutionContext
    context = ExecutionContext()

    # Step 1: Calculate (42 * 8) + 15 = 351
    step1_result = 351
    step1_execution = ExecutionStep(
        step_number=1,
        tool_name="calculator",
        tool_input={"expression": "(42 * 8) + 15"},
        success=True,
        output=step1_result,
        attempts=1,
    )
    context.record_step(step1_execution)

    # Verify step 1 result is stored
    assert context.variables["step_1_output"] == 351

    # Step 2: Resolve variables and execute
    step2_input = context.resolve_variables(plan.steps[1].tool_input)
    assert step2_input["expression"] == "351 * 2", "Variable should be resolved to 351"

    # Calculate step 2: 351 * 2 = 702
    step2_result = 702
    step2_execution = ExecutionStep(
        step_number=2,
        tool_name="calculator",
        tool_input=step2_input,
        success=True,
        output=step2_result,
        attempts=1,
    )
    context.record_step(step2_execution)

    # Verify step 2 result is stored
    assert context.variables["step_2_output"] == 702

    # Step 3: Resolve variables for todo
    step3_input = context.resolve_variables(plan.steps[2].tool_input)
    assert step3_input["action"] == "add"
    assert step3_input["text"] == "Result: 702", "Variable should be resolved to 702"

    # Verify the complete workflow
    assert len(context.get_execution_log()) == 2
    assert context.variables["step_1_output"] == 351
    assert context.variables["step_2_output"] == 702
