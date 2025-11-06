"""
FastAPI dependencies for dependency injection.

This module provides centralized dependency injection for routes and services.
Using Annotated types (Python 3.12+) for cleaner dependency declarations.

Example usage in routes:
    from challenge.api.dependencies import SettingsDep

    @router.get("/endpoint")
    async def my_endpoint(settings: SettingsDep):
        return {"app_name": settings.app_name}
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from challenge.core.config import Settings, get_settings
from challenge.infrastructure.tools import ToolRegistry
from challenge.infrastructure.tools.registry import get_tool_registry
from challenge.services.orchestration.orchestrator import Orchestrator
from challenge.services.planning.llm_planner import LLMPlanner
from challenge.services.planning.planner import PatternBasedPlanner

# ============================================================================
# Core Dependencies
# ============================================================================

# Settings dependency - inject application configuration
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ============================================================================
# AI Agent Runtime Dependencies
# ============================================================================


@lru_cache
def get_orchestrator() -> Orchestrator:
    """
    Get cached orchestrator instance with LLM planner.

    Returns singleton orchestrator with LLM planner (with pattern-based fallback) and tools.
    This hybrid approach provides:
    - Intelligent planning for complex requests via LLM
    - Automatic fallback to pattern-based on LLM failures
    - Cost optimization with GPT-4o-mini (or local LLM via configuration)
    - Reliability through graceful degradation

    The planner configuration is read from environment variables:
    - OPENAI_API_KEY: API key (required for OpenAI, optional for local LLMs)
    - OPENAI_BASE_URL: Custom base URL (e.g., http://localhost:4000 for LiteLLM)
    - OPENAI_MODEL: Model name (default: gpt-4o-mini, can be local model like qwen2.5:3b)
    - OPENAI_TEMPERATURE: Sampling temperature (default: 0.1)

    Returns:
        Orchestrator instance

    """
    settings = get_settings()

    # Use LLM planner with pattern-based fallback for production resilience
    planner = LLMPlanner(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,  # None for OpenAI, set for local LLMs
        temperature=settings.openai_temperature,
        fallback=PatternBasedPlanner(),
    )

    tool_registry: ToolRegistry = get_tool_registry()
    return Orchestrator(
        planner=planner,
        tools=tool_registry,
    )


# Type alias for cleaner route signatures
OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
