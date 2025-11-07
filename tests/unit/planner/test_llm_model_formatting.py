"""Tests for LLM model name formatting with LiteLLM proxy support."""

from challenge.services.planning.llm_planner import LLMPlanner


class TestModelNameFormatting:
    """Test model name formatting for LiteLLM proxy compatibility."""

    def test_ollama_model_without_base_url_unchanged(self) -> None:
        """Test that Ollama models without base_url remain unchanged (cloud mode)."""
        planner = LLMPlanner(model="qwen2.5:3b")
        formatted = planner._format_model_name()
        # Without base_url, assume cloud provider (no prefix)
        assert formatted == "qwen2.5:3b"

    def test_ollama_model_with_proxy_gets_openai_prefix(self) -> None:
        """Test that Ollama models with proxy base_url get openai/ prefix."""
        planner = LLMPlanner(
            model="qwen2.5:3b",
            base_url="http://localhost:4000",
        )
        formatted = planner._format_model_name()
        # LiteLLM proxy: use openai/ prefix for OpenAI-compatible API
        assert formatted == "openai/qwen2.5:3b"

    def test_various_ollama_models_with_proxy(self) -> None:
        """Test various Ollama model naming formats with proxy."""
        test_cases = [
            ("qwen2.5:3b", "openai/qwen2.5:3b"),
            ("llama3.2:3b", "openai/llama3.2:3b"),
            ("phi3:mini", "openai/phi3:mini"),
            ("mistral:7b", "openai/mistral:7b"),
        ]

        for model, expected in test_cases:
            planner = LLMPlanner(model=model, base_url="http://localhost:4000")
            assert planner._format_model_name() == expected

    def test_openai_models_without_base_url_unchanged(self) -> None:
        """Test that OpenAI models without base_url remain unchanged."""
        openai_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]

        for model in openai_models:
            planner = LLMPlanner(model=model)
            assert planner._format_model_name() == model

    def test_openai_models_with_proxy_get_prefix(self) -> None:
        """Test that OpenAI models via proxy get openai/ prefix."""
        planner = LLMPlanner(
            model="gpt-4o-mini",
            base_url="http://localhost:4000",
        )
        formatted = planner._format_model_name()
        assert formatted == "openai/gpt-4o-mini"

    def test_anthropic_models_without_base_url_unchanged(self) -> None:
        """Test that Anthropic models without base_url remain unchanged."""
        anthropic_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]

        for model in anthropic_models:
            planner = LLMPlanner(model=model)
            assert planner._format_model_name() == model

    def test_already_prefixed_model_unchanged(self) -> None:
        """Test that already-prefixed models don't get double-prefixed."""
        test_cases = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-5-sonnet",
            "ollama/qwen2.5:3b",
            "ollama_chat/llama3.2:3b",
        ]

        for model in test_cases:
            planner = LLMPlanner(model=model, base_url="http://localhost:4000")
            formatted = planner._format_model_name()
            assert formatted == model

    def test_custom_model_without_base_url_unchanged(self) -> None:
        """Test that custom models without base_url remain unchanged."""
        planner = LLMPlanner(model="custom-model-name")
        formatted = planner._format_model_name()
        assert formatted == "custom-model-name"

    def test_custom_model_with_proxy_gets_prefix(self) -> None:
        """Test that custom models with proxy base_url get openai/ prefix."""
        planner = LLMPlanner(
            model="custom-model-name",
            base_url="http://localhost:4000",
        )
        formatted = planner._format_model_name()
        assert formatted == "openai/custom-model-name"
