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
    Get cached orchestrator instance.

    Returns singleton orchestrator with default planner and tools.

    Returns:
        Orchestrator instance

    """
    return Orchestrator(
        planner=PatternBasedPlanner(),
        tools=get_tool_registry(),
    )


# Type alias for cleaner route signatures
OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
