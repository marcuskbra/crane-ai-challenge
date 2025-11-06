"""
LLM-based planner using structured outputs for reliable agent planning.

This planner uses OpenAI's structured output feature to generate valid plans
with automatic fallback to pattern-based planning on failure.

Includes few-shot prompt engineering for improved consistency and error handling.
"""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from challenge.models.plan import Plan
from challenge.planner.examples import ALL_EXAMPLES, format_example_for_prompt
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

    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        fallback: PatternBasedPlanner | None = None,
        temperature: float = 0.1,
        use_examples: bool = True,
    ):
        """
        Initialize LLM planner.

        Args:
            model: OpenAI model name (default: gpt-4o-mini for cost efficiency)
            api_key: OpenAI API key (uses env var if None)
            base_url: Custom API base URL (default: None for OpenAI, set for local LLMs via LiteLLM)
            fallback: Fallback planner for LLM failures (creates default if None)
            temperature: Sampling temperature (low for consistency)
            use_examples: Whether to include few-shot examples in system prompt (default: True)

        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_url = base_url
        self.fallback = fallback or PatternBasedPlanner()
        self.temperature = temperature
        self.use_examples = use_examples
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

        Includes few-shot examples when use_examples is True to demonstrate
        desired planning patterns and improve consistency.

        Returns:
            System prompt string

        """
        base_prompt = """You are a task planning agent. Convert user requests into structured execution plans.

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
- Number steps sequentially starting from 1

Variable Resolution:
CRITICAL: When a step needs data from a previous step's output, you MUST use ONLY these exact variable patterns:
- {step_N_output} - Full output from step N (replace N with step number, e.g., {step_1_output}, {step_2_output})
- {first_todo_id} - ID of first item in a list output
- {last_todo_id} - ID of last item in a list output
- {step_N_first_id} - First ID from step N's list output
- {step_N_count} - Count of items from step N's list output

DO NOT use custom variable names like {calculation_result}, {result}, {value}, etc.
ALWAYS use {step_N_output} where N is the step number.

Examples:
1. Calculator workflow:
   User: "Calculate (42 * 8) + 15, then multiply by 2"
   Step 1: {"expression": "(42 * 8) + 15"}
   Step 2: {"expression": "{step_1_output} * 2"} ✓ CORRECT - uses {step_1_output}
   Step 2: {"expression": "{calculation_result} * 2"} ✗ WRONG - custom variable name

2. Todo workflow:
   User: "List all my todos and mark the first one as complete"
   Step 1: {"action": "list"}
   Step 2: {"action": "complete", "todo_id": "{first_todo_id}"} ✓ CORRECT

IMPORTANT: Always use curly braces {} for variable references, never angle brackets <>"""

        # Add few-shot examples if enabled
        if self.use_examples:
            examples_section = "\n\n---\n\nHere are examples of good planning patterns:\n\n"
            for example in ALL_EXAMPLES:
                examples_section += format_example_for_prompt(example) + "\n"
            return base_prompt + examples_section

        return base_prompt

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
