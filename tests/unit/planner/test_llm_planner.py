"""Tests for LLM-based planner with mocked OpenAI responses."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from challenge.models.plan import Plan
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
    assert plan.steps[0].tool_input["expression"] == "2+3"
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
    assert plan.steps[1].tool_input["action"] == "add"
    assert plan.steps[1].tool_input["text"] == "Buy milk"


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

    # Check cost estimate
    cost_info = planner.get_cost_estimate()
    assert cost_info["tokens"] == 200
    assert cost_info["model"] == "gpt-4o-mini"
    assert cost_info["estimated_cost_usd"] > 0
    assert cost_info["cost_per_1k_tokens"] == 0.00015


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
    mock_client = mock_openai.return_value
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
    """Test LLM planning for todo operations."""
    # Mock response for adding a todo
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
    assert plan.steps[0].tool_input["action"] == "add"
    assert plan.steps[0].tool_input["text"] == "Buy groceries"
