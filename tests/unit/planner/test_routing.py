"""
Tests for intelligent model routing with multi-provider support.

This test suite validates:
- Prompt complexity assessment (simple/moderate/complex)
- Model selection based on complexity and cost optimization
- Fallback chain generation for resilience
- Model configuration and metadata retrieval
"""

import pytest

from challenge.services.planning.routing import (
    CLAUDE_3_5_HAIKU,
    CLAUDE_3_5_SONNET,
    GPT_4O,
    GPT_4O_MINI,
    LLAMA_3_2_3B,
    PHI_3_MINI,
    QWEN_2_5_3B,
    ModelConfig,
    ModelRouter,
    PromptComplexity,
)


class TestPromptComplexityAssessment:
    """Test prompt complexity assessment logic."""

    @pytest.fixture
    def router(self) -> ModelRouter:
        """Create a ModelRouter with default settings."""
        return ModelRouter()

    @pytest.mark.parametrize(
        ("prompt", "expected_complexity"),
        [
            # Simple prompts (< 100 tokens, no complex keywords)
            ("Calculate 2+2", PromptComplexity.SIMPLE),
            ("Add todo Buy milk", PromptComplexity.SIMPLE),
            ("List todos", PromptComplexity.SIMPLE),
            ("Complete todo 123", PromptComplexity.SIMPLE),
            # Moderate prompts (100-300 tokens ≈ 77-231 words, no complex keywords)
            (
                "Calculate the sum of 42 and 15, then multiply that result by 3. "
                "After getting that value, add 10 to it and divide the total by 2. "
                "Round the result to the nearest integer. Then take that final number "
                "and use it to create a calculation plan for the next steps in this process. "
                "Make sure to document each intermediate result clearly so that the calculation "
                "can be verified and audited later. Include proper formatting and labels for "
                "each step to make the plan easy to follow and understand.",
                PromptComplexity.MODERATE,
            ),
            (
                "Add a todo item to buy groceries including the following items: milk, eggs, bread, "
                "vegetables, fruits, and cheese. Then create another separate todo to call the dentist "
                "to schedule an appointment for next week, preferably in the afternoon. Make sure both "
                "todo items are properly categorized under the correct categories and prioritized "
                "according to their urgency. Also ensure that each todo has an appropriate due date "
                "assigned and any necessary notes about special requirements or additional context "
                "that might be needed when completing these tasks.",
                PromptComplexity.MODERATE,
            ),
            (
                "List all my current todo items including their assigned priorities and category "
                "classifications. After displaying the list, mark the first todo item as complete "
                "and then provide a summary of what tasks are still remaining to be done. Include in "
                "the summary the estimated time required for each task, the total estimated time for "
                "all remaining tasks, and organize the output in a clear format that shows the "
                "categories, priorities, and time estimates. Make sure the summary is easy to read "
                "and understand at a glance with proper organization and structure.",
                PromptComplexity.MODERATE,
            ),
            # Complex prompts (> 300 tokens or complex keywords)
            (
                "Calculate (42 * 8) + 15, then use the result and multiply by 2, "
                "and add the result as a todo with a detailed description",
                PromptComplexity.COMPLEX,
            ),
            (
                "Perform a multi-step comprehensive analysis of the calculator results",
                PromptComplexity.COMPLEX,
            ),
            (
                "Systematically evaluate and analyze multiple todo items across "
                "different categories with detailed priority assessment",
                PromptComplexity.COMPLEX,
            ),
        ],
    )
    def test_complexity_assessment(
        self, router: ModelRouter, prompt: str, expected_complexity: PromptComplexity
    ) -> None:
        """Test prompt complexity assessment with various inputs."""
        actual_complexity = router.assess_complexity(prompt)
        assert actual_complexity == expected_complexity

    def test_complexity_keywords_trigger_complex(self, router: ModelRouter) -> None:
        """Test that complexity keywords trigger COMPLEX classification."""
        # Short prompt but with complex keyword
        prompt = "multi-step task"
        complexity = router.assess_complexity(prompt)
        assert complexity == PromptComplexity.COMPLEX

    def test_very_long_prompt_is_complex(self, router: ModelRouter) -> None:
        """Test that very long prompts are classified as COMPLEX."""
        # Generate a prompt with > 300 estimated tokens
        long_prompt = " ".join(["word"] * 250)  # 250 words * 1.3 = 325 tokens
        complexity = router.assess_complexity(long_prompt)
        assert complexity == PromptComplexity.COMPLEX

    def test_custom_thresholds(self) -> None:
        """Test router with custom complexity thresholds."""
        router = ModelRouter(complexity_threshold_simple=50, complexity_threshold_complex=150)

        # 30 words * 1.3 = 39 tokens → SIMPLE
        short_prompt = " ".join(["word"] * 30)
        assert router.assess_complexity(short_prompt) == PromptComplexity.SIMPLE

        # 100 words * 1.3 = 130 tokens → MODERATE
        medium_prompt = " ".join(["word"] * 100)
        assert router.assess_complexity(medium_prompt) == PromptComplexity.MODERATE

        # 120 words * 1.3 = 156 tokens → COMPLEX
        long_prompt = " ".join(["word"] * 120)
        assert router.assess_complexity(long_prompt) == PromptComplexity.COMPLEX


class TestModelSelection:
    """Test model selection based on complexity."""

    @pytest.fixture
    def router(self) -> ModelRouter:
        """Create a ModelRouter with default settings."""
        return ModelRouter(
            simple_model="gpt-4o-mini",
            moderate_model="gpt-4o-mini",
            complex_model="gpt-4o",
        )

    def test_simple_prompt_selects_simple_model(self, router: ModelRouter) -> None:
        """Test that simple prompts select the simple model."""
        model = router.select_model("Calculate 2+2")
        assert model == "gpt-4o-mini"

    def test_moderate_prompt_selects_moderate_model(self, router: ModelRouter) -> None:
        """Test that moderate prompts select the moderate model."""
        model = router.select_model("Calculate 42 + 15 and multiply by 3")
        assert model == "gpt-4o-mini"

    def test_complex_prompt_selects_complex_model(self, router: ModelRouter) -> None:
        """Test that complex prompts select the complex model."""
        model = router.select_model("Perform a comprehensive multi-step analysis with detailed evaluation")
        assert model == "gpt-4o"

    def test_complex_keyword_triggers_complex_model(self, router: ModelRouter) -> None:
        """Test that complexity keywords trigger complex model selection."""
        model = router.select_model("multi-step workflow")
        assert model == "gpt-4o"

    def test_cost_optimization_mode(self) -> None:
        """Test cost optimization mode (default behavior)."""
        router = ModelRouter(simple_model="gpt-4o-mini", moderate_model="gpt-4o-mini", complex_model="gpt-4o")

        # Cost optimization enabled by default
        simple_model = router.select_model("Calculate 2+2", prefer_cost_optimization=True)
        assert simple_model == "gpt-4o-mini"  # Cheapest for simple tasks

    def test_custom_model_configuration(self) -> None:
        """Test router with custom model configurations."""
        router = ModelRouter(
            simple_model="claude-3-5-haiku-20241022",
            moderate_model="gpt-4o-mini",
            complex_model="claude-3-5-sonnet-20241022",
        )

        assert router.select_model("Simple task") == "claude-3-5-haiku-20241022"
        # Use a moderate-length prompt (100-300 tokens ≈ 77-231 words) without complex keywords
        moderate_prompt = (
            "Calculate the sum of several numbers including 42, 15, and 30, "
            "then multiply the result by 2 and store the final value. "
            "After that, take the stored value and perform additional arithmetic operations "
            "such as dividing by 5, adding 10, and subtracting 3. Make sure to keep track "
            "of each intermediate result for verification purposes. Document all steps clearly "
            "so that the calculations can be reviewed later if needed. Format the output "
            "in a way that makes it easy to understand the sequence of operations performed."
        )
        assert router.select_model(moderate_prompt) == "gpt-4o-mini"
        assert router.select_model("Complex comprehensive multi-step analysis") == "claude-3-5-sonnet-20241022"


class TestFallbackChainGeneration:
    """Test fallback chain generation for resilience."""

    @pytest.fixture
    def router(self) -> ModelRouter:
        """Create a ModelRouter with fallback models."""
        return ModelRouter(fallback_models=["claude-3-5-haiku-20241022", "qwen2.5:3b"])

    def test_fallback_chain_includes_primary_and_fallbacks(self, router: ModelRouter) -> None:
        """Test that fallback chain includes primary model and fallbacks."""
        chain = router.get_fallback_chain("gpt-4o-mini")
        assert len(chain) == 3
        assert chain[0] == "gpt-4o-mini"
        assert chain[1] == "claude-3-5-haiku-20241022"
        assert chain[2] == "qwen2.5:3b"

    def test_fallback_chain_order(self, router: ModelRouter) -> None:
        """Test that fallback chain maintains correct order."""
        chain = router.get_fallback_chain("primary-model")
        assert chain[0] == "primary-model"  # Primary first
        assert chain[1:] == router.fallback_models  # Then fallbacks in order

    def test_empty_fallback_chain(self) -> None:
        """Test router without fallback models."""
        router = ModelRouter(fallback_models=[])
        chain = router.get_fallback_chain("gpt-4o-mini")
        assert chain == ["gpt-4o-mini"]  # Only primary model


class TestModelConfigRetrieval:
    """Test model configuration and metadata retrieval."""

    @pytest.fixture
    def router(self) -> ModelRouter:
        """Create a ModelRouter for config testing."""
        return ModelRouter()

    @pytest.mark.parametrize(
        ("model_name", "expected_config"),
        [
            ("gpt-4o", GPT_4O),
            ("gpt-4o-mini", GPT_4O_MINI),
            ("claude-3-5-sonnet-20241022", CLAUDE_3_5_SONNET),
            ("claude-3-5-haiku-20241022", CLAUDE_3_5_HAIKU),
            ("qwen2.5:3b", QWEN_2_5_3B),
            ("llama3.2:3b", LLAMA_3_2_3B),
            ("phi3:3b", PHI_3_MINI),
        ],
    )
    def test_get_model_info(self, router: ModelRouter, model_name: str, expected_config: ModelConfig) -> None:
        """Test retrieving model configuration information."""
        config = router.get_model_info(model_name)
        assert config is not None
        assert config.name == expected_config.name
        assert config.provider == expected_config.provider
        assert config.max_tokens == expected_config.max_tokens
        assert config.cost_per_1k_tokens == expected_config.cost_per_1k_tokens
        assert config.supports_json_schema == expected_config.supports_json_schema
        assert config.is_local == expected_config.is_local

    def test_get_model_info_unknown_model(self, router: ModelRouter) -> None:
        """Test that unknown models return None."""
        config = router.get_model_info("unknown-model-xyz")
        assert config is None


class TestModelConfigDataclass:
    """Test ModelConfig dataclass attributes and validation."""

    def test_openai_model_config(self) -> None:
        """Test OpenAI model configuration."""
        assert GPT_4O_MINI.name == "gpt-4o-mini"
        assert GPT_4O_MINI.provider == "openai"
        assert GPT_4O_MINI.max_tokens == 128000
        assert GPT_4O_MINI.cost_per_1k_tokens == 0.00015
        assert GPT_4O_MINI.supports_json_schema is True
        assert GPT_4O_MINI.is_local is False

        assert GPT_4O.name == "gpt-4o"
        assert GPT_4O.provider == "openai"
        assert GPT_4O.max_tokens == 128000
        assert GPT_4O.cost_per_1k_tokens == 0.00250
        assert GPT_4O.supports_json_schema is True
        assert GPT_4O.is_local is False

    def test_anthropic_model_config(self) -> None:
        """Test Anthropic model configuration."""
        assert CLAUDE_3_5_SONNET.name == "claude-3-5-sonnet-20241022"
        assert CLAUDE_3_5_SONNET.provider == "anthropic"
        assert CLAUDE_3_5_SONNET.max_tokens == 200000
        assert CLAUDE_3_5_SONNET.cost_per_1k_tokens == 0.00300
        assert CLAUDE_3_5_SONNET.supports_json_schema is True
        assert CLAUDE_3_5_SONNET.is_local is False

        assert CLAUDE_3_5_HAIKU.name == "claude-3-5-haiku-20241022"
        assert CLAUDE_3_5_HAIKU.provider == "anthropic"
        assert CLAUDE_3_5_HAIKU.max_tokens == 200000
        assert CLAUDE_3_5_HAIKU.cost_per_1k_tokens == 0.00080
        assert CLAUDE_3_5_HAIKU.supports_json_schema is True
        assert CLAUDE_3_5_HAIKU.is_local is False

    def test_local_model_config(self) -> None:
        """Test local model (Ollama) configuration."""
        assert QWEN_2_5_3B.name == "qwen2.5:3b"
        assert QWEN_2_5_3B.provider == "ollama"
        assert QWEN_2_5_3B.max_tokens == 32768
        assert QWEN_2_5_3B.cost_per_1k_tokens == 0.0  # Free local model
        assert QWEN_2_5_3B.supports_json_schema is True
        assert QWEN_2_5_3B.is_local is True

        assert LLAMA_3_2_3B.name == "llama3.2:3b"
        assert LLAMA_3_2_3B.provider == "ollama"
        assert LLAMA_3_2_3B.cost_per_1k_tokens == 0.0
        assert LLAMA_3_2_3B.is_local is True

        assert PHI_3_MINI.name == "phi3:3b"
        assert PHI_3_MINI.provider == "ollama"
        assert PHI_3_MINI.cost_per_1k_tokens == 0.0
        assert PHI_3_MINI.is_local is True


class TestRouterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_prompt(self) -> None:
        """Test handling of empty prompt."""
        router = ModelRouter()
        # Empty prompt should be classified as SIMPLE
        complexity = router.assess_complexity("")
        assert complexity == PromptComplexity.SIMPLE

    def test_whitespace_only_prompt(self) -> None:
        """Test handling of whitespace-only prompt."""
        router = ModelRouter()
        complexity = router.assess_complexity("   \n\t  ")
        assert complexity == PromptComplexity.SIMPLE

    def test_unicode_prompt(self) -> None:
        """Test handling of unicode characters in prompt."""
        router = ModelRouter()
        prompt = "Calculate 2+2 with émoji 🧮 support"
        model = router.select_model(prompt)
        assert model in ["gpt-4o-mini", "gpt-4o"]  # Should select valid model

    def test_very_long_fallback_chain(self) -> None:
        """Test router with many fallback models."""
        router = ModelRouter(
            fallback_models=[
                "claude-3-5-haiku-20241022",
                "qwen2.5:3b",
                "llama3.2:3b",
                "phi3:3b",
            ]
        )
        chain = router.get_fallback_chain("gpt-4o-mini")
        assert len(chain) == 5  # Primary + 4 fallbacks
        assert chain[0] == "gpt-4o-mini"
