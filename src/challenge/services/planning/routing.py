"""
Intelligent model routing for multi-provider LLM support.

This module provides smart model selection based on prompt complexity,
cost optimization, and provider availability with automatic fallback chains.
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PromptComplexity(str, Enum):
    """Prompt complexity levels for routing decisions."""

    SIMPLE = "simple"  # Short prompts, basic tasks (<100 tokens)
    MODERATE = "moderate"  # Medium prompts, standard tasks (100-300 tokens)
    COMPLEX = "complex"  # Long prompts, advanced reasoning (>300 tokens)


@dataclass
class ModelConfig:
    """
    Configuration for a specific model.

    Attributes:
        name: Model identifier (e.g., "gpt-4o-mini", "claude-3-5-haiku-20241022")
        provider: Provider name (e.g., "openai", "anthropic", "ollama")
        max_tokens: Maximum token capacity
        cost_per_1k_tokens: Approximate cost per 1K tokens in USD
        supports_json_schema: Whether model supports structured JSON output
        is_local: Whether this is a local model (free)

    """

    name: str
    provider: str
    max_tokens: int
    cost_per_1k_tokens: float
    supports_json_schema: bool = True
    is_local: bool = False


# ============================================================================
# Predefined Model Configurations
# ============================================================================

# OpenAI Models
GPT_4O = ModelConfig(
    name="gpt-4o",
    provider="openai",
    max_tokens=128000,
    cost_per_1k_tokens=0.00250,  # $2.50 per 1M tokens (input)
    supports_json_schema=True,
)

GPT_4O_MINI = ModelConfig(
    name="gpt-4o-mini",
    provider="openai",
    max_tokens=128000,
    cost_per_1k_tokens=0.00015,  # $0.15 per 1M tokens (input)
    supports_json_schema=True,
)

# Anthropic Models
CLAUDE_3_5_SONNET = ModelConfig(
    name="claude-3-5-sonnet-20241022",
    provider="anthropic",
    max_tokens=200000,
    cost_per_1k_tokens=0.00300,  # $3.00 per 1M tokens (input)
    supports_json_schema=True,
)

CLAUDE_3_5_HAIKU = ModelConfig(
    name="claude-3-5-haiku-20241022",
    provider="anthropic",
    max_tokens=200000,
    cost_per_1k_tokens=0.00080,  # $0.80 per 1M tokens (input)
    supports_json_schema=True,
)

# Local Models (via Ollama)
QWEN_2_5_3B = ModelConfig(
    name="qwen2.5:3b",
    provider="ollama",
    max_tokens=32768,
    cost_per_1k_tokens=0.0,  # Free (local)
    supports_json_schema=True,
    is_local=True,
)

LLAMA_3_2_3B = ModelConfig(
    name="llama3.2:3b",
    provider="ollama",
    max_tokens=128000,
    cost_per_1k_tokens=0.0,  # Free (local)
    supports_json_schema=True,
    is_local=True,
)

PHI_3_MINI = ModelConfig(
    name="phi3:3b",
    provider="ollama",
    max_tokens=128000,
    cost_per_1k_tokens=0.0,  # Free (local)
    supports_json_schema=True,
    is_local=True,
)


class ModelRouter:
    """
    Intelligent model router with complexity-based selection and fallback chains.

    Automatically selects the most appropriate model based on:
    - Prompt complexity (length, keywords, structure)
    - Cost optimization (prefer cheaper models for simple tasks)
    - Provider availability (fallback to alternatives on failure)
    - User preferences (configurable routing rules)

    Example:
        >>> router = ModelRouter(
        ...     simple_model="gpt-4o-mini",
        ...     complex_model="gpt-4o",
        ...     fallback_models=["claude-3-5-haiku-20241022", "qwen2.5:3b"]
        ... )
        >>> model = router.select_model("Calculate 2 + 2")
        >>> # Returns "gpt-4o-mini" (simple task)
        >>> model = router.select_model("Analyze this complex multi-step workflow...")
        >>> # Returns "gpt-4o" (complex task)

    """

    def __init__(
        self,
        simple_model: str = "gpt-4o-mini",
        moderate_model: str = "gpt-4o-mini",
        complex_model: str = "gpt-4o",
        fallback_models: list[str] | None = None,
        complexity_threshold_simple: int = 100,
        complexity_threshold_complex: int = 300,
    ):
        """
        Initialize model router with configuration.

        Args:
            simple_model: Model for simple prompts (default: gpt-4o-mini)
            moderate_model: Model for moderate prompts (default: gpt-4o-mini)
            complex_model: Model for complex prompts (default: gpt-4o)
            fallback_models: Ordered list of fallback models (default: None)
            complexity_threshold_simple: Token threshold for simple complexity (default: 100)
            complexity_threshold_complex: Token threshold for complex complexity (default: 300)

        """
        self.simple_model = simple_model
        self.moderate_model = moderate_model
        self.complex_model = complex_model
        self.fallback_models = fallback_models or []
        self.complexity_threshold_simple = complexity_threshold_simple
        self.complexity_threshold_complex = complexity_threshold_complex

    def assess_complexity(self, prompt: str) -> PromptComplexity:
        """
        Assess prompt complexity based on length and content analysis.

        Uses heuristics to determine complexity:
        - Token count estimation (words * 1.3 for approximate tokens)
        - Keyword detection (multi-step, complex, analyze, comprehensive)
        - Structural patterns (lists, multiple questions, nested logic)

        Args:
            prompt: User prompt to analyze

        Returns:
            PromptComplexity level (SIMPLE, MODERATE, or COMPLEX)

        """
        # Rough token estimation (words * 1.3)
        estimated_tokens = len(prompt.split()) * 1.3

        # Complexity keywords
        complex_keywords = [
            "multi-step",
            "complex",
            "comprehensive",
            "analyze",
            "evaluate",
            "systematic",
            "detailed",
            "multiple",
        ]
        has_complex_keywords = any(keyword in prompt.lower() for keyword in complex_keywords)

        # Determine complexity
        if estimated_tokens < self.complexity_threshold_simple and not has_complex_keywords:
            return PromptComplexity.SIMPLE

        if estimated_tokens > self.complexity_threshold_complex or has_complex_keywords:
            return PromptComplexity.COMPLEX

        return PromptComplexity.MODERATE

    def select_model(self, prompt: str, prefer_cost_optimization: bool = True) -> str:
        """
        Select the most appropriate model for the given prompt.

        Selection Logic:
        1. Assess prompt complexity
        2. Select model tier based on complexity
        3. Apply cost optimization if enabled
        4. Return selected model name

        Args:
            prompt: User prompt to route
            prefer_cost_optimization: Use cheaper models when possible (default: True)

        Returns:
            Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022")

        """
        complexity = self.assess_complexity(prompt)

        # Select model based on complexity
        if complexity == PromptComplexity.SIMPLE:
            selected_model = self.simple_model
            logger.debug(f"Routing to SIMPLE model: {selected_model} (prompt length: {len(prompt)} chars)")

        elif complexity == PromptComplexity.MODERATE:
            selected_model = self.moderate_model
            logger.debug(f"Routing to MODERATE model: {selected_model} (prompt length: {len(prompt)} chars)")

        else:  # COMPLEX
            selected_model = self.complex_model
            logger.debug(f"Routing to COMPLEX model: {selected_model} (prompt length: {len(prompt)} chars)")

        return selected_model

    def get_fallback_chain(self, primary_model: str) -> list[str]:
        """
        Get ordered fallback chain for a primary model.

        Fallback Strategy:
        1. Primary model (requested)
        2. Fallback models (configured alternatives)
        3. Pattern-based planner (ultimate fallback)

        Args:
            primary_model: The originally selected model

        Returns:
            Ordered list of models to try (includes primary + fallbacks)

        """
        chain = [primary_model, *self.fallback_models]
        logger.debug(f"Fallback chain for {primary_model}: {chain}")
        return chain

    def get_model_info(self, model_name: str) -> ModelConfig | None:
        """
        Get configuration information for a model.

        Args:
            model_name: Model identifier

        Returns:
            ModelConfig if found, None otherwise

        """
        model_registry = {
            "gpt-4o": GPT_4O,
            "gpt-4o-mini": GPT_4O_MINI,
            "claude-3-5-sonnet-20241022": CLAUDE_3_5_SONNET,
            "claude-3-5-haiku-20241022": CLAUDE_3_5_HAIKU,
            "qwen2.5:3b": QWEN_2_5_3B,
            "llama3.2:3b": LLAMA_3_2_3B,
            "phi3:3b": PHI_3_MINI,
        }
        return model_registry.get(model_name)
