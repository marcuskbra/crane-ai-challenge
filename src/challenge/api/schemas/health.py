"""Health check request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SystemInfo(BaseModel):
    """System information model for health checks."""

    python_version: str = Field(..., description="Python runtime version")
    platform: str = Field(..., description="Operating system platform")
    architecture: str = Field(..., description="CPU architecture")

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid",
    )


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

    system: SystemInfo = Field(
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
