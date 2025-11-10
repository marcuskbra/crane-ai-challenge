"""Tests for LLM-based planner with mocked LiteLLM responses."""

from unittest.mock import MagicMock, patch

import litellm
import pytest

from challenge.core.exceptions import LLMConfigurationError
from challenge.domain.models.plan import Plan
from challenge.domain.models.run import ExecutionStep
from challenge.services.orchestration.execution_context import ExecutionContext
from challenge.services.planning.llm_planner import LLMPlanner


@pytest.fixture
def mock_litellm():
    """Mock LiteLLM acompletion function."""
    with patch("challenge.services.planning.llm_planner.litellm.acompletion") as mock:
        yield mock


@pytest.mark.asyncio
async def test_llm_planner_success(mock_litellm):
    """Test successful LLM planning with structured output."""
    # Mock successful LiteLLM response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"steps": [{"step_number": 1, "tool_name": "calculator", "tool_input": {"expression": "2+3"}, "reasoning": "Calculate sum"}], "final_goal": "Calculate 2+3"}'
            )
        )
    ]
    mock_response.usage = MagicMock(total_tokens=125)

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

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
async def test_llm_planner_multi_step(mock_litellm):
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

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

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
async def test_llm_planner_fallback_on_rate_limit(mock_litellm):
    """Test graceful fallback to pattern-based on rate limit (transient error)."""
    # Mock rate limit error (transient - should fall back)
    mock_litellm.side_effect = litellm.RateLimitError(
        message="Rate limit exceeded", llm_provider="openai", model="gpt-4o-mini"
    )

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
async def test_llm_planner_fails_on_auth_error(mock_litellm):
    """Test that authentication errors fail loudly (configuration error)."""
    # Mock authentication error (configuration - should NOT fall back)
    mock_litellm.side_effect = litellm.AuthenticationError(
        message="Invalid API key", llm_provider="openai", model="gpt-4o-mini"
    )

    # Test planner - should raise LLMConfigurationError
    planner = LLMPlanner()
    with pytest.raises(LLMConfigurationError) as exc_info:
        await planner.create_plan("calculate 2+3")

    # Verify error details
    assert "Invalid API key" in str(exc_info.value)
    error = exc_info.value
    assert isinstance(error, LLMConfigurationError)
    assert error.provider in ["LiteLLM", "gpt-4o-mini"]
    assert error.fix_hint is not None
    assert "LLM_API_KEY" in error.fix_hint


@pytest.mark.asyncio
async def test_llm_planner_fails_on_invalid_model(mock_litellm):
    """Test that invalid model errors fail loudly (configuration error)."""
    # Mock model not found error (configuration - should NOT fall back)
    mock_litellm.side_effect = litellm.NotFoundError(
        message="Model not found", llm_provider="openai", model="invalid-model-name"
    )

    # Test planner - should raise LLMConfigurationError
    planner = LLMPlanner(model="invalid-model-name")
    with pytest.raises(LLMConfigurationError) as exc_info:
        await planner.create_plan("calculate 2+3")

    # Verify error details
    assert "Model not found" in str(exc_info.value)
    error = exc_info.value
    assert isinstance(error, LLMConfigurationError)
    assert error.fix_hint is not None
    assert "invalid-model-name" in error.fix_hint


@pytest.mark.asyncio
async def test_llm_planner_fallback_on_service_unavailable(mock_litellm):
    """Test graceful fallback on service unavailable (transient error)."""
    # Mock service unavailable (transient - should fall back)
    mock_litellm.side_effect = litellm.ServiceUnavailableError(
        message="Service temporarily down", llm_provider="openai", model="gpt-4o-mini"
    )

    # Test planner with fallback
    planner = LLMPlanner()
    plan = await planner.create_plan("calculate 2+3")

    # Should fall back to pattern-based planner
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_llm_planner_fallback_on_invalid_json(mock_litellm):
    """Test fallback when LLM returns invalid JSON."""
    # Mock response with invalid JSON
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="invalid json {"))]
    mock_response.usage = MagicMock(total_tokens=50)

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

    # Test planner
    planner = LLMPlanner()
    plan = await planner.create_plan("calculate 2+3")

    # Should fall back
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_llm_planner_fallback_on_invalid_schema(mock_litellm):
    """Test fallback when LLM returns valid JSON but invalid schema."""
    # Mock response with valid JSON but missing required fields
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"invalid": "schema"}'))]
    mock_response.usage = MagicMock(total_tokens=60)

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

    # Test planner
    planner = LLMPlanner()
    plan = await planner.create_plan("calculate 2+3")

    # Should fall back
    assert isinstance(plan, Plan)
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_cost_tracking(mock_litellm):
    """Test token usage and cost tracking with LiteLLM."""
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

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

    # Test planner
    planner = LLMPlanner()
    await planner.create_plan("calculate 1+1")

    # Check cost estimate (uses LiteLLM's built-in pricing)
    cost_info = planner.get_cost_estimate()
    assert cost_info.tokens == 200
    assert cost_info.model == "gpt-4o-mini"
    # LiteLLM calculates cost automatically, should be >= 0
    assert cost_info.estimated_cost_usd >= 0
    assert cost_info.cost_per_1k_tokens >= 0


@pytest.mark.asyncio
async def test_temperature_configuration(mock_litellm):
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

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

    # Test with custom temperature
    planner = LLMPlanner(temperature=0.5)
    await planner.create_plan("calculate 1+1")

    # Verify temperature was passed to API call
    call_kwargs = mock_litellm.call_args[1]
    assert call_kwargs["temperature"] == 0.5


@pytest.mark.asyncio
async def test_model_configuration(mock_litellm):
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

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

    # Test with custom model
    planner = LLMPlanner(model="gpt-4o")
    await planner.create_plan("calculate 1+1")

    # Verify model was passed to API call
    call_kwargs = mock_litellm.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_system_prompt_includes_tool_descriptions(mock_litellm):
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
async def test_empty_prompt_raises_error(mock_litellm):
    # Use a transient error so fallback happens (not a configuration error)
    mock_litellm.side_effect = litellm.ServiceUnavailableError(
        message="Service down", llm_provider="openai", model="gpt-4o-mini"
    )

    planner = LLMPlanner()

    # Empty prompt should fallback and then raise from pattern planner
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await planner.create_plan("")


@pytest.mark.asyncio
async def test_todo_operations(mock_litellm):
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

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

    # Test planner
    planner = LLMPlanner()
    plan = await planner.create_plan("add todo Buy groceries")

    # Assertions
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "todo_store"
    assert plan.steps[0].tool_input.action == "add"
    assert plan.steps[0].tool_input.text == "Buy groceries"


@pytest.mark.asyncio
async def test_complex_multi_step_with_variable_resolution(mock_litellm):
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

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

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
async def test_complex_multi_step_execution_with_context(mock_litellm):
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

    # Mock litellm.acompletion directly
    mock_litellm.return_value = mock_response

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
