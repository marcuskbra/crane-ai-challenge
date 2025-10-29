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
from challenge.orchestrator.orchestrator import Orchestrator
from challenge.planner.llm_planner import LLMPlanner
from challenge.planner.planner import PatternBasedPlanner
from challenge.tools.registry import get_tool_registry

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
    - Cost optimization with GPT-4o-mini
    - Reliability through graceful degradation

    Returns:
        Orchestrator instance

    """
    # Use LLM planner with pattern-based fallback for production resilience
    planner = LLMPlanner(model="gpt-4o-mini", fallback=PatternBasedPlanner())

    return Orchestrator(
        planner=planner,
        tools=get_tool_registry(),
    )


# Type alias for cleaner route signatures
OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
