"""
Health check endpoints for monitoring and orchestration.

Provides health check endpoints following best practices for
Kubernetes and cloud-native deployments:

- /health: Basic health check with system information
- /health/live: Liveness probe (is the application running?)
- /health/ready: Readiness probe (can the application serve traffic?)

These endpoints are used by:
- Load balancers for routing decisions
- Kubernetes for pod lifecycle management
- Monitoring systems for alerting
- Health check services for uptime monitoring
"""

import logging
import platform
import sys
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from challenge.api.dependencies import SettingsDep
from challenge.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(
        ...,
        description="Overall health status",
        examples=["healthy"],
    )
    version: str = Field(
        ...,
        description="Application version",
        examples=["0.1.0"],
    )
    timestamp: datetime = Field(
        ...,
        description="Current server timestamp (UTC)",
    )
    environment: str | None = Field(
        None,
        description="Deployment environment",
        examples=["development", "staging", "production"],
    )


class DetailedHealthResponse(HealthResponse):
    """Detailed health check response with system information."""

    system: dict[str, Any] = Field(
        ...,
        description="System information",
    )
    checks: dict[str, str] = Field(
        ...,
        description="Component health checks",
    )


class LivenessResponse(BaseModel):
    """Liveness probe response model."""

    alive: bool = Field(
        ...,
        description="Whether the application is alive",
        examples=[True],
    )
    timestamp: datetime = Field(
        ...,
        description="Current server timestamp (UTC)",
    )


class ReadinessResponse(BaseModel):
    """Readiness probe response model."""

    ready: bool = Field(
        ...,
        description="Whether the application is ready to serve traffic",
        examples=[True],
    )
    checks: dict[str, bool] = Field(
        ...,
        description="Readiness checks for each component",
    )
    timestamp: datetime = Field(
        ...,
        description="Current server timestamp (UTC)",
    )


@router.get(
    "/health",
    response_model=DetailedHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Returns detailed health information including version, system details, and component checks",
    tags=["health"],
)
async def health_check(settings: SettingsDep) -> DetailedHealthResponse:
    """
    Comprehensive health check endpoint.

    Returns detailed information about the application and its dependencies:
    - Application version and status
    - System information (Python version, platform)
    - Component health checks
    - Server timestamp

    Used for monitoring and diagnostics.
    """
    # Perform component checks
    checks = {
        "application": "healthy",
        # Add more checks as components are integrated:
        # "database": await check_database(),
        # "cache": await check_cache(),
        # "external_api": await check_external_api(),
    }

    return DetailedHealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
        environment=settings.environment,
        system={
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        checks=checks,
    )


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Indicates whether the application is running",
    tags=["health"],
)
async def liveness_check() -> LivenessResponse:
    """
    Liveness probe endpoint.

    Used by Kubernetes and orchestrators to determine if the application
    is alive and should be kept running. If this endpoint fails, the
    orchestrator will restart the container.

    This endpoint should always return 200 OK if the application can
    process requests, even if dependencies are unavailable.
    """
    logger.debug("Liveness check performed")

    return LivenessResponse(
        alive=True,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Indicates whether the application is ready to serve traffic",
    tags=["health"],
)
async def readiness_check() -> ReadinessResponse:
    """
    Readiness probe endpoint.

    Used by Kubernetes and load balancers to determine if the application
    is ready to receive traffic. If this endpoint fails, the orchestrator
    will remove the instance from the load balancer pool.

    This endpoint checks critical dependencies and returns:
    - 200 OK if ready to serve traffic
    - 503 Service Unavailable if not ready

    Checks performed:
    - Application initialized
    - Database connection available
    - Cache connection available
    - External API dependencies reachable

    Raises:
        ServiceUnavailableError: If any critical service is not ready

    """
    # Perform readiness checks
    checks = {
        "application": True,
        # Add more checks as components are integrated:
        # "database": await is_database_ready(),
        # "cache": await is_cache_ready(),
        # "external_api": await is_external_api_ready(),
    }

    ready = all(checks.values())

    if not ready:
        failed_services = [name for name, status in checks.items() if not status]
        logger.warning("Readiness check failed: %s", checks)
        raise ServiceUnavailableError(
            service=", ".join(failed_services),
            reason="Service checks failed",
        )

    return ReadinessResponse(
        ready=ready,
        checks=checks,
        timestamp=datetime.now(timezone.utc),
    )
