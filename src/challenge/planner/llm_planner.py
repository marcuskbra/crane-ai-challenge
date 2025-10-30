"""
LLM-based planner using structured outputs for reliable agent planning.

This planner uses OpenAI's structured output feature to generate valid plans
with automatic fallback to pattern-based planning on failure.
"""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from challenge.models.plan import Plan
from challenge.planner.planner import PatternBasedPlanner

logger = logging.getLogger(__name__)

# JSON Schema for structured output enforcement
PLAN_SCHEMA = {
    "name": "execution_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_number": {"type": "integer"},
                        "tool_name": {"type": "string"},
                        "tool_input": {"type": "object"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["step_number", "tool_name", "tool_input", "reasoning"],
                    "additionalProperties": False,
                },
            },
            "final_goal": {"type": "string"},
        },
        "required": ["steps", "final_goal"],
        "additionalProperties": False,
    },
}


class LLMPlanner:
    """
    LLM-based planner with structured outputs and pattern-based fallback.

    This planner demonstrates production LLM engineering patterns:
    - Structured outputs for reliability (JSON schema enforcement)
    - Fallback chain for resilience (LLM → Pattern → Error)
    - Cost tracking for observability (token counting)
    - Low temperature for consistency (0.1 for planning tasks)

    Example:
        >>> planner = LLMPlanner(model="gpt-4o-mini")
        >>> plan = await planner.create_plan("calculate 2+3 and add to-do Buy milk")
        >>> print(f"Tokens used: {planner.last_token_count}")

    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        fallback: PatternBasedPlanner | None = None,
        temperature: float = 0.1,
    ):
        """
        Initialize LLM planner.

        Args:
            model: OpenAI model name (default: gpt-4o-mini for cost efficiency)
            api_key: OpenAI API key (uses env var if None)
            fallback: Fallback planner for LLM failures (creates default if None)
            temperature: Sampling temperature (low for consistency)

        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.fallback = fallback or PatternBasedPlanner()
        self.temperature = temperature
        self.last_token_count: int = 0

    async def create_plan(self, prompt: str) -> Plan:
        """
        Create execution plan using LLM with fallback.

        Args:
            prompt: Natural language task description

        Returns:
            Plan with ordered execution steps

        Raises:
            ValueError: If both LLM and fallback fail

        """
        try:
            # Call LLM with structured output
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_schema", "json_schema": PLAN_SCHEMA},
                temperature=self.temperature,
            )

            # Track token usage for cost monitoring
            self.last_token_count = response.usage.total_tokens
            logger.info(f"LLM planning succeeded - tokens: {self.last_token_count}, model: {self.model}")

            # Parse and validate structured output
            plan_dict = json.loads(response.choices[0].message.content)
            return Plan.model_validate(plan_dict)

        except Exception as e:
            # Fallback to pattern-based on any failure
            logger.warning(f"LLM planning failed ({e.__class__.__name__}: {e}), using pattern-based fallback")
            return self.fallback.create_plan(prompt)

    def _system_prompt(self) -> str:
        """
        System prompt defining available tools and output format.

        Returns:
            System prompt string

        """
        return """You are a task planning agent. Convert user requests into structured execution plans.

Available Tools:
1. calculator
   - Purpose: Evaluate arithmetic expressions
   - Input: expression (string) - math expression like "2 + 3 * 4"
   - Example: {"expression": "(10 + 5) / 3"}

2. todo_store
   - Purpose: Manage todo items
   - Input:
     - action (string): "add", "list", "get", "complete", or "delete"
     - text (string, optional): todo text (for "add")
     - todo_id (string, optional): todo ID (for "get", "complete", "delete")
   - Examples:
     - {"action": "add", "text": "Buy milk"}
     - {"action": "list"}
     - {"action": "complete", "todo_id": "uuid-here"}

Output Format:
Return a JSON plan with:
- steps: array of step objects (step_number, tool_name, tool_input, reasoning)
- final_goal: description of overall goal

Rules:
- Break complex tasks into sequential steps
- Use specific tool inputs (don't invent new tools)
- Provide clear reasoning for each step
- Number steps sequentially starting from 1"""

    def get_cost_estimate(self) -> dict[str, Any]:
        """
        Get cost estimate for last LLM call.

        Returns:
            Dict with token count and estimated cost

        """
        # GPT-4o-mini pricing (as of 2024)
        cost_per_1k_tokens = 0.00015  # $0.15 per 1M tokens
        estimated_cost = (self.last_token_count / 1000) * cost_per_1k_tokens

        return {
            "tokens": self.last_token_count,
            "model": self.model,
            "estimated_cost_usd": estimated_cost,
            "cost_per_1k_tokens": cost_per_1k_tokens,
        }
