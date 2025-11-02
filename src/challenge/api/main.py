"""
FastAPI application factory and configuration.

This module provides the FastAPI application factory pattern, enabling:
- Testable application instances
- Environment-specific configurations
- Middleware registration
- Error handler setup
- API versioning and routing
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from challenge import __version__
from challenge.api.routes import health, metrics, runs
from challenge.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown events.

    Handles:
    - Database connection initialization
    - Cache warmup
    - Background task startup
    - Resource cleanup on shutdown
    """
    # Startup
    logger.info("Starting crane-challenge API v%s", __version__)

    yield

    # Shutdown
    logger.info("Shutting down crane-challenge API")


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    FastAPI application factory.

    Creates and configures a FastAPI application instance with:
    - CORS middleware
    - Error handlers
    - API routing
    - OpenAPI documentation

    Args:
        settings: Application settings (defaults to get_settings() if not provided)

    Returns:
        Configured FastAPI application instance

    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        description="A modern Python API with simplified 3-layer architecture",
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url=settings.get_docs_url(),
        redoc_url=settings.get_redoc_url(),
        openapi_url="/api/openapi.json" if not settings.is_production() else None,
    )

    # Configure CORS
    _configure_cors(app, settings)

    # Add middleware
    _configure_middleware(app, settings)

    # Register error handlers
    _register_error_handlers(app, settings)

    # Register routes
    _register_routes(app)

    return app


def _configure_cors(app: FastAPI, settings: Settings) -> None:
    """
    Configure CORS middleware based on settings.

    SECURITY NOTE: Never use allow_origins=["*"] with allow_credentials=True.
    This violates the CORS specification and creates a security vulnerability.

    Args:
        app: FastAPI application instance
        settings: Application settings with CORS configuration

    """
    if not settings.cors_enabled:
        logger.info("CORS middleware is disabled")
        return

    logger.info("Configuring CORS with origins: %s", settings.cors_origins)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
        max_age=settings.cors_max_age,
    )


def _configure_middleware(app: FastAPI, settings: Settings) -> None:
    """
    Configure additional middleware.

    Args:
        app: FastAPI application instance
        settings: Application settings

    """
    # GZip compression for responses
    app.add_middleware(GZipMiddleware, minimum_size=settings.gzip_minimum_size)

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests and responses."""
        logger.info(
            "Incoming request: %s %s",
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        logger.info(
            "Response: %s %s - Status: %d",
            request.method,
            request.url.path,
            response.status_code,
        )

        return response


def _register_error_handlers(app: FastAPI, settings: Settings) -> None:
    """
    Register custom error handlers for API exceptions.

    Handles FastAPI and Pydantic validation errors with consistent responses.
    Custom exceptions should be handled at the endpoint level by raising
    appropriate HTTPException instances.

    Args:
        app: FastAPI application instance
        settings: Application settings

    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle FastAPI request validation errors."""
        logger.warning("Validation error: %s", exc.errors())

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(_: Request, exc: ValidationError) -> JSONResponse:
        """Handle Pydantic validation errors."""
        logger.warning("Pydantic validation error: %s", exc.errors())

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "validation_error",
                "message": "Data validation failed",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception("Unhandled exception: %s", str(exc))

        content = {
            "error": "internal_error",
            "message": "An unexpected error occurred",
        }

        # Include error details in development/staging for debugging
        if not settings.is_production():
            content["detail"] = str(exc)
            content["type"] = type(exc).__name__

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=content,
        )


def _register_routes(app: FastAPI) -> None:
    """
    Register API routes and routers.

    Args:
        app: FastAPI application instance

    """
    # Include API routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(runs.router, prefix="/api/v1", tags=["runs"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])


# Create default application instance for uvicorn
app = create_app()
