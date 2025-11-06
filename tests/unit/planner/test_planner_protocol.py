"""
Tests for planner protocol demonstrating structural subtyping.

This test suite shows how the Protocol enables:
- Any class with create_plan method to work as a planner
- No inheritance required
- Both sync and async implementations supported
- Type safety maintained
"""

import pytest

from challenge.models.plan import Plan, PlanStep
from challenge.orchestrator.orchestrator import Orchestrator
from challenge.planner.llm_planner import LLMPlanner
from challenge.planner.planner import PatternBasedPlanner
from challenge.planner.protocol import Planner


# Custom planner without inheriting from any base class
class CustomSyncPlanner:
    """Custom planner that doesn't inherit from anything."""

    def create_plan(self, prompt: str) -> Plan:
        """Create a simple single-step plan."""
        return Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "1+1"},
                    reasoning="Custom planner always returns 1+1",
                )
            ],
            final_goal=prompt,
        )

    @property
    def last_token_count(self) -> int | None:
        """Custom planner doesn't track tokens."""
        return None


class CustomAsyncPlanner:
    """Custom async planner without inheritance."""

    async def create_plan(self, prompt: str) -> Plan:
        """Create a simple async plan."""
        return Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "2+2"},
                    reasoning="Async custom planner always returns 2+2",
                )
            ],
            final_goal=prompt,
        )

    @property
    def last_token_count(self) -> int | None:
        """Custom async planner doesn't track tokens."""
        return None


def test_pattern_based_planner_conforms_to_protocol():
    """Verify PatternBasedPlanner works with Protocol."""
    planner = PatternBasedPlanner()
    orchestrator = Orchestrator(planner=planner)

    # Type checker should accept this
    assert orchestrator.planner is planner


def test_llm_planner_conforms_to_protocol():
    """Verify LLMPlanner works with Protocol."""
    planner = LLMPlanner(api_key="test-key")
    orchestrator = Orchestrator(planner=planner)

    # Type checker should accept this
    assert orchestrator.planner is planner


def test_custom_sync_planner_conforms_to_protocol():
    """Verify custom sync planner works without inheritance."""
    planner = CustomSyncPlanner()
    orchestrator = Orchestrator(planner=planner)

    # Type checker should accept this - no inheritance needed!
    assert orchestrator.planner is planner


def test_custom_async_planner_conforms_to_protocol():
    """Verify custom async planner works without inheritance."""
    planner = CustomAsyncPlanner()
    orchestrator = Orchestrator(planner=planner)

    # Type checker should accept this - no inheritance needed!
    assert orchestrator.planner is planner


@pytest.mark.asyncio
async def test_orchestrator_handles_sync_planner():
    """Test orchestrator correctly executes with sync planner."""
    planner = CustomSyncPlanner()
    orchestrator = Orchestrator(planner=planner)

    run = await orchestrator.create_run("test prompt")

    assert run.plan is not None
    assert len(run.plan.steps) == 1
    assert run.plan.steps[0].tool_input.expression == "1+1"


@pytest.mark.asyncio
async def test_orchestrator_handles_async_planner():
    """Test orchestrator correctly executes with async planner."""
    planner = CustomAsyncPlanner()
    orchestrator = Orchestrator(planner=planner)

    run = await orchestrator.create_run("test prompt")

    assert run.plan is not None
    assert len(run.plan.steps) == 1
    assert run.plan.steps[0].tool_input.expression == "2+2"


@pytest.mark.asyncio
async def test_orchestrator_with_pattern_based_planner():
    """Test orchestrator with original PatternBasedPlanner."""
    planner = PatternBasedPlanner()
    orchestrator = Orchestrator(planner=planner)

    run = await orchestrator.create_run("calculate 5 + 3")

    assert run.plan is not None
    assert len(run.plan.steps) == 1
    assert run.plan.steps[0].tool_name == "calculator"
    assert "5 + 3" in run.plan.steps[0].tool_input.expression


def test_protocol_documentation():
    """Test that protocol is well-documented."""

    # Protocol should have docstring
    assert Planner.__doc__ is not None
    assert "Protocol" in Planner.__doc__

    # Method should have docstring
    assert Planner.create_plan.__doc__ is not None


def test_protocol_enables_duck_typing():
    """
    Demonstrate the power of Protocol: duck typing with type safety.

    Any object with a create_plan method that matches the signature
    can be used as a planner, regardless of its class hierarchy.
    """

    class MinimalPlanner:
        """Absolutely minimal planner - just the method."""

        def create_plan(self, prompt: str) -> Plan:
            return Plan(steps=[], final_goal=prompt)

        @property
        def last_token_count(self) -> int | None:
            """Minimal planner doesn't track tokens."""
            return None

    # This works! No inheritance, no registration, just matching signature
    orchestrator = Orchestrator(planner=MinimalPlanner())
    assert orchestrator.planner is not None
