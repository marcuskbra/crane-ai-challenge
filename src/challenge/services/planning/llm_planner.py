"""
LLM-based planner using structured outputs for reliable agent planning.

This planner uses LiteLLM for multi-provider support (OpenAI, Anthropic, local models)
with automatic fallback to pattern-based planning on failure.

Includes few-shot prompt engineering for improved consistency and error handling.
Features comprehensive observability via LiteLLM callbacks for monitoring and debugging.
"""

import json
import logging
import re
from typing import Any

import litellm
from litellm import cost_per_token
from pydantic import BaseModel, ConfigDict, Field

from challenge.domain.models.plan import Plan
from challenge.services.planning.examples import ALL_EXAMPLES, format_example_for_prompt
from challenge.services.planning.planner import PatternBasedPlanner

logger = logging.getLogger(__name__)


# ============================================================================
# LiteLLM Callbacks for Observability
# ============================================================================


def log_llm_success(kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
    """
    Log successful LLM completion details.

    Logs detailed information about successful LLM calls including:
    - Model and provider used
    - Token usage and estimated cost
    - Response time
    - Request parameters

    Args:
        kwargs: Request parameters passed to LiteLLM
        response_obj: Response object from LLM provider
        start_time: Request start timestamp
        end_time: Request end timestamp

    """
    try:
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        prompt_tokens = getattr(response_obj.usage, "prompt_tokens", 0) if hasattr(response_obj, "usage") else 0
        completion_tokens = getattr(response_obj.usage, "completion_tokens", 0) if hasattr(response_obj, "usage") else 0
        total_tokens = getattr(response_obj.usage, "total_tokens", 0) if hasattr(response_obj, "usage") else 0

        # Calculate cost using LiteLLM's built-in pricing
        try:
            cost = litellm.completion_cost(completion_response=response_obj)
        except Exception:
            cost = 0.0

        # Calculate duration
        duration_ms = int((end_time - start_time).total_seconds() * 1000) if start_time and end_time else 0

        logger.info(
            f"LLM SUCCESS: model={model}, tokens={total_tokens} "
            f"(prompt={prompt_tokens}, completion={completion_tokens}), "
            f"cost=${cost:.6f}, duration={duration_ms}ms, "
            f"messages={len(messages)}"
        )

    except Exception as e:
        logger.warning(f"Error in LLM success callback: {e}")


def log_llm_failure(kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
    """
    Log failed LLM completion errors.

    Logs detailed error information for debugging and monitoring:
    - Model and provider attempted
    - Error type and message
    - Request parameters
    - Failure timestamp

    Args:
        kwargs: Request parameters passed to LiteLLM
        response_obj: Response/error object from LLM provider
        start_time: Request start timestamp
        end_time: Request end timestamp

    """
    try:
        model = kwargs.get("model", "unknown")
        error_message = str(response_obj) if response_obj else "Unknown error"

        logger.error(f"LLM FAILURE: model={model}, error={error_message}, will_fallback_to_pattern_planner=True")

    except Exception as e:
        logger.warning(f"Error in LLM failure callback: {e}")


# Configure LiteLLM callbacks globally
litellm.success_callback = [log_llm_success]
litellm.failure_callback = [log_llm_failure]

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


class CostEstimate(BaseModel):
    """
    Cost estimate for LLM operation.

    Provides token usage and cost information for monitoring and optimization.
    """

    tokens: int = Field(..., ge=0, description="Total tokens used in operation")
    model: str = Field(..., min_length=1, description="Model name used for operation")
    estimated_cost_usd: float = Field(..., ge=0.0, description="Estimated cost in USD")
    cost_per_1k_tokens: float = Field(..., ge=0.0, description="Cost per 1K tokens in USD")

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid",
    )


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
        Initialize LLM planner with LiteLLM proxy support.

        This implementation assumes you're using a LiteLLM proxy for unified access
        to multiple LLM providers (OpenAI, Anthropic, Ollama, etc.).

        Args:
            model: Model name (default: gpt-4o-mini for cost efficiency)
                   Examples: gpt-4o-mini, claude-3-5-sonnet, qwen2.5:3b
            api_key: API key for cloud providers (not needed for local proxy)
            base_url: LiteLLM proxy URL (default: http://localhost:4000)
                     For cloud providers, use their API endpoints
            fallback: Fallback planner for LLM failures (creates default if None)
            temperature: Sampling temperature (low for consistency, 0.0-2.0)
            use_examples: Whether to include few-shot examples in system prompt

        """
        self.model = model
        self.api_key = api_key or "dummy-key"  # Dummy key for local proxy
        self.base_url = base_url
        self.fallback = fallback or PatternBasedPlanner()
        self.temperature = temperature
        self.use_examples = use_examples
        self.last_token_count: int = 0

        logger.info(
            f"Initialized LLMPlanner: model={model}, base_url={base_url or 'default'}, use_examples={use_examples}"
        )

    def _format_model_name(self) -> str:
        """
        Format model name for LiteLLM proxy compatibility.

        The LiteLLM SDK requires an "openai/" prefix when using a custom base_url
        to indicate it should use the OpenAI-compatible API format.

        Examples:
        - Local models via proxy: "openai/qwen2.5:3b", "openai/llama3.2:3b"
        - Cloud OpenAI: "gpt-4o-mini" (no prefix needed)
        - Cloud Anthropic: "claude-3-5-sonnet-20241022" (no prefix needed)

        Returns:
            Formatted model name

        """
        # If model already has a provider prefix, keep it as-is
        if "/" in self.model:
            return self.model

        # If using a custom base_url (proxy), add openai/ prefix
        # This tells LiteLLM SDK to use OpenAI-compatible API format
        if self.base_url:
            return f"openai/{self.model}"

        # Cloud providers (OpenAI, Anthropic) don't need prefix
        return self.model

    def _clean_json_response(self, content: str) -> str:
        """
        Clean and fix common JSON formatting issues from Ollama models.

        Ollama models sometimes generate JavaScript-style "JSON" when not using json_schema:
        - Single quotes instead of double quotes
        - Backticks for strings (template literals)
        - Unquoted object keys (e.g., {action: "add"} instead of {"action": "add"})
        - Trailing commas
        - Comments
        - Markdown code blocks

        Args:
            content: Raw content from LLM

        Returns:
            Cleaned JSON string ready for parsing

        """
        # Step 1: Remove markdown code blocks
        if "```" in content:
            # Try to extract JSON from code block
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            else:
                # Extract content between ``` markers
                parts = content.split("```")
                if len(parts) >= 3:
                    content = parts[1].strip()
                elif len(parts) == 2:
                    content = parts[1].strip()

        # Step 2: Find JSON object boundaries
        start_idx = content.find("{")
        end_idx = content.rfind("}")

        if start_idx == -1 or end_idx == -1:
            return content

        content = content[start_idx : end_idx + 1]

        # Step 3: Fix JavaScript-style JSON syntax

        # Fix unquoted object keys: {action: "add"} -> {"action": "add"}
        # Match word characters followed by colon (but not inside strings)
        content = re.sub(r"(\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', content)

        # Fix backtick strings: `text` -> "text"
        # Replace backticks with double quotes
        content = content.replace("`", '"')

        # Fix single-quoted strings: 'text' -> "text"
        # This is tricky because single quotes might be inside double-quoted strings
        # Simple approach: replace 'key': with "key": and : 'value' with : "value"
        content = re.sub(r"'([^']*)':", r'"\1":', content)  # Keys
        content = re.sub(r":\s*'([^']*)'", r': "\1"', content)  # Values

        # Remove trailing commas before } or ]
        content = re.sub(r",(\s*[}\]])", r"\1", content)

        # Remove comments (// or /* */)
        content = re.sub(r"//.*?\n", "\n", content)  # Single-line comments
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)  # Multi-line comments

        return content.strip()

    def _fix_variable_names(self, plan_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Fix common variable naming mistakes in LLM-generated plans.

        Small models often use custom variable names like {calculation_result}, {result}, {value}
        instead of the required {step_N_output} format. This method detects and corrects these mistakes.

        Args:
            plan_dict: Parsed plan dictionary from LLM

        Returns:
            Plan dictionary with corrected variable names

        """
        # Common incorrect variable names to fix
        incorrect_patterns = [
            "calculation_result",
            "calc_result",
            "result",
            "value",
            "answer",
            "output",
            "number",
            "sum",
            "product",
            "difference",
            "quotient",
        ]

        # Track which step outputs which variables (for context-aware fixing)
        steps = plan_dict.get("steps", [])

        for step_idx, step in enumerate(steps):
            step_number = step.get("step_number", step_idx + 1)
            tool_input = step.get("tool_input", {})

            # Convert tool_input to JSON string for easier pattern matching
            tool_input_str = json.dumps(tool_input)

            # Fix each incorrect pattern by replacing with correct step reference
            for pattern in incorrect_patterns:
                # Match {pattern} or {pattern_N} where N is a digit
                regex_pattern = rf"\{{{pattern}(_\d+)?\}}"

                # Find all matches
                matches = re.findall(regex_pattern, tool_input_str)

                if matches:
                    # For step N, we need to reference the previous step (N-1)
                    # The previous step's output is step_(N-1)_output
                    if step_number > 1:
                        correct_var = f"{{step_{step_number - 1}_output}}"
                        tool_input_str = re.sub(regex_pattern, correct_var, tool_input_str)
                        logger.warning(f"Step {step_number}: Fixed variable name '{{{pattern}}}' → '{correct_var}'")

            # Parse back to dict
            try:
                fixed_tool_input = json.loads(tool_input_str)
                step["tool_input"] = fixed_tool_input
            except json.JSONDecodeError:
                # If parsing fails, keep original
                logger.warning(f"Step {step_number}: Could not parse fixed tool_input, keeping original")

        return plan_dict

    async def create_plan(self, prompt: str) -> Plan:
        """
        Create execution plan using LLM via LiteLLM proxy with fallback.

        Uses LiteLLM SDK to communicate with a LiteLLM proxy that handles
        routing to different LLM providers (OpenAI, Anthropic, Ollama, etc.).

        Args:
            prompt: Natural language task description

        Returns:
            Plan with ordered execution steps

        Raises:
            ValueError: If both LLM and fallback fail

        """
        try:
            # Format model name with provider prefix if needed
            formatted_model = self._format_model_name()

            # Detect if using local Ollama models (they don't support json_schema)
            uses_ollama_model = any(m in self.model.lower() for m in ["qwen", "llama", "phi", "mistral", "tinyllama"])

            # Prepare LLM request
            completion_args = {
                "model": formatted_model,
                "messages": [
                    {"role": "system", "content": self._system_prompt(is_ollama=uses_ollama_model)},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
            }

            # Add base_url if using proxy
            if self.base_url:
                completion_args["base_url"] = self.base_url

            # Add API key (dummy key for local proxy, real key for cloud)
            completion_args["api_key"] = self.api_key

            # Only use json_schema for models that support it (OpenAI, Anthropic)
            # Ollama models don't support json_schema, rely on prompt instructions
            if not uses_ollama_model:
                completion_args["response_format"] = {
                    "type": "json_schema",
                    "json_schema": PLAN_SCHEMA,
                }

            # Call LLM via LiteLLM proxy
            response = await litellm.acompletion(**completion_args)

            # Track token usage
            self.last_token_count = response.usage.total_tokens
            logger.info(
                f"LLM planning succeeded - tokens: {self.last_token_count}, "
                f"model: {self.model} (formatted: {formatted_model})"
            )

            # Parse response
            content = response.choices[0].message.content

            # Debug: Log raw content for Ollama models
            if uses_ollama_model:
                logger.debug(f"Raw Ollama response: {content}")
                content = self._clean_json_response(content)
                logger.debug(f"Cleaned Ollama response: {content}")

            # Parse and validate
            plan_dict = json.loads(content)

            # Fix common variable naming mistakes for Ollama models
            if uses_ollama_model:
                plan_dict = self._fix_variable_names(plan_dict)

            return Plan.model_validate(plan_dict)

        except json.JSONDecodeError as e:
            # Log the problematic JSON for debugging
            raw_content = response.choices[0].message.content if "response" in locals() else "N/A"
            cleaned_content = content if "content" in locals() else "N/A"
            logger.error(
                f"JSON parsing failed: {e!s}\n"
                f"Raw content: {raw_content[:1000]}\n"
                f"Cleaned content: {cleaned_content[:1000]}"
            )
            logger.warning("LLM planning failed (invalid JSON), using pattern-based fallback")
            return self.fallback.create_plan(prompt)

        except Exception as e:
            # Log error and fall back to pattern-based planner
            logger.error(f"LLM FAILURE: model={self.model}, error={e.__class__.__name__}: {str(e)[:150]}")
            logger.warning(f"LLM planning failed ({e.__class__.__name__}), using pattern-based fallback")
            return self.fallback.create_plan(prompt)

    def _system_prompt(self, is_ollama: bool = False) -> str:
        """
        System prompt defining available tools and output format.

        Includes few-shot examples when use_examples is True to demonstrate
        desired planning patterns and improve consistency.

        Args:
            is_ollama: If True, add explicit JSON format instructions for Ollama

        Returns:
            System prompt string

        """
        base_prompt = """
You are an expert task decomposition specialist!

Before generating the plan:
1. Identify the user's primary goal
2. Break down into atomic, sequential steps
3. Verify each step uses only available tools
4. Ensure steps are properly ordered

If ambiguous:
- Prefer simpler interpretations
- Use calculator for any numeric operations
- Break complex todos into multiple steps

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

        # Add explicit JSON format instructions for Ollama models
        if is_ollama:
            json_format = """

CRITICAL JSON FORMAT REQUIREMENTS (YOU MUST FOLLOW THIS EXACTLY):
You MUST respond with ONLY valid JSON. No markdown, no code blocks, no explanations.
Your response must start with { and end with }

Required JSON structure:
{
  "steps": [
    {
      "step_number": 1,
      "tool_name": "calculator" or "todo_store",
      "tool_input": { ... },
      "reasoning": "explanation of this step"
    }
  ],
  "final_goal": "overall task description"
}

CRITICAL VARIABLE NAMING RULE (MOST IMPORTANT):
When referencing previous step outputs, you MUST use ONLY this exact format:
- Use {step_1_output} for step 1's output
- Use {step_2_output} for step 2's output
- Use {step_N_output} for step N's output
NEVER use custom names like {calculation_result}, {result}, {value}, {answer}, etc.

Example with variable resolution (FOLLOW THIS EXACTLY):
{
  "steps": [
    {
      "step_number": 1,
      "tool_name": "calculator",
      "tool_input": {"expression": "10 + 5"},
      "reasoning": "Calculate the sum"
    },
    {
      "step_number": 2,
      "tool_name": "todo_store",
      "tool_input": {"action": "add", "text": "Result: {step_1_output}"},
      "reasoning": "Add result as todo using {step_1_output}"
    }
  ],
  "final_goal": "Calculate 10 + 5 and add as todo"
}

DO NOT wrap your response in markdown code blocks like ```json
DO NOT add any text before or after the JSON
DO NOT use custom variable names - ONLY {step_N_output} format
ONLY return the raw JSON object"""
            base_prompt += json_format

        # Add few-shot examples if enabled
        if self.use_examples:
            examples_section = "\n\n---\n\nHere are examples of good planning patterns:\n\n"
            for example in ALL_EXAMPLES:
                examples_section += format_example_for_prompt(example) + "\n"
            return base_prompt + examples_section

        return base_prompt

    def get_cost_estimate(self) -> CostEstimate:
        """
        Get cost estimate for last LLM call using LiteLLM's built-in pricing.

        LiteLLM automatically calculates costs for all supported providers:
        - OpenAI (gpt-4o, gpt-4o-mini, etc.)
        - Anthropic (claude-3-5-sonnet, claude-3-5-haiku, etc.)
        - Local models (free, returns $0.00)

        Returns:
            Typed cost estimate with token count and pricing information

        """
        try:
            # Use LiteLLM's cost_per_token for accurate multi-provider pricing
            # Approximation: treat all tokens as completion tokens
            formatted_model = self._format_model_name()
            _, completion_cost_per_token = cost_per_token(
                model=formatted_model,
                prompt_tokens=0,
                completion_tokens=self.last_token_count,
            )
            estimated_cost = completion_cost_per_token * self.last_token_count
            cost_per_1k_tokens = completion_cost_per_token * 1000

        except Exception:
            # Fallback to zero cost if model pricing not found
            estimated_cost = 0.0
            cost_per_1k_tokens = 0.0

        return CostEstimate(
            tokens=self.last_token_count,
            model=self.model,
            estimated_cost_usd=estimated_cost,
            cost_per_1k_tokens=cost_per_1k_tokens,
        )
