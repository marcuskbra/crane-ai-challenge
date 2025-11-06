"""
Centralized exception handlers for FastAPI application.

This module registers exception handlers that map custom exceptions to
appropriate HTTP responses. Follows best practices:
- Consistent error response format
- Proper HTTP status codes
- Comprehensive logging
- No sensitive data in responses

Usage:
    Register handlers in main.py:
    ```python
    from challenge.api.exception_handlers import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)
    ```
"""

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from challenge.core.exceptions import (
    ApplicationError,
    ExecutionError,
    InvalidPromptError,
    PlanGenerationError,
    RunNotFoundError,
    ServiceUnavailableError,
    ValidationError,
)

logger = logging.getLogger(__name__)


# Standard error response schema
class ErrorResponse:
    """Standard error response format."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> None:
        """
        Initialize error response.

        Args:
            message: Human-readable error description
            details: Additional context (optional)
            error_type: Error classification (optional)

        """
        self.message = message
        self.details = details or {}
        self.error_type = error_type

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response."""
        response: dict[str, Any] = {"detail": self.message}
        if self.details:
            response["details"] = self.details
        if self.error_type:
            response["error_type"] = self.error_type
        return response


# Custom exception handlers
async def run_not_found_handler(request: Request, exc: RunNotFoundError) -> JSONResponse:
    """
    Handle RunNotFoundError exceptions.

    Args:
        request: HTTP request
        exc: RunNotFoundError exception

    Returns:
        JSONResponse with 404 status

    """
    logger.warning(f"Run not found: {exc.run_id}")

    error = ErrorResponse(
        message=exc.message,
        details=exc.details,
        error_type="run_not_found",
    )

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=error.to_dict(),
    )


async def invalid_prompt_handler(request: Request, exc: InvalidPromptError) -> JSONResponse:
    """
    Handle InvalidPromptError exceptions.

    Args:
        request: HTTP request
        exc: InvalidPromptError exception

    Returns:
        JSONResponse with 400 status

    """
    logger.warning(f"Invalid prompt: {exc.reason}")

    error = ErrorResponse(
        message=exc.message,
        details=exc.details,
        error_type="invalid_prompt",
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error.to_dict(),
    )


async def plan_generation_handler(request: Request, exc: PlanGenerationError) -> JSONResponse:
    """
    Handle PlanGenerationError exceptions.

    Args:
        request: HTTP request
        exc: PlanGenerationError exception

    Returns:
        JSONResponse with 400 status

    """
    logger.warning(f"Plan generation failed: {exc.reason}")

    error = ErrorResponse(
        message=exc.message,
        details=exc.details,
        error_type="plan_generation_failed",
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error.to_dict(),
    )


async def execution_error_handler(request: Request, exc: ExecutionError) -> JSONResponse:
    """
    Handle ExecutionError exceptions.

    Args:
        request: HTTP request
        exc: ExecutionError exception

    Returns:
        JSONResponse with 500 status

    """
    logger.error(f"Execution failed: {exc.reason}", exc_info=True)

    error = ErrorResponse(
        message=exc.message,
        details=exc.details,
        error_type="execution_failed",
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error.to_dict(),
    )


async def service_unavailable_handler(request: Request, exc: ServiceUnavailableError) -> JSONResponse:
    """
    Handle ServiceUnavailableError exceptions.

    Args:
        request: HTTP request
        exc: ServiceUnavailableError exception

    Returns:
        JSONResponse with 503 status

    """
    logger.error(f"Service unavailable: {exc.service} - {exc.reason}")

    error = ErrorResponse(
        message=exc.message,
        details=exc.details,
        error_type="service_unavailable",
    )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=error.to_dict(),
    )


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handle custom ValidationError exceptions.

    Args:
        request: HTTP request
        exc: ValidationError exception

    Returns:
        JSONResponse with 422 status

    """
    logger.warning(f"Validation failed: {exc.field} - {exc.constraint}")

    error = ErrorResponse(
        message=exc.message,
        details=exc.details,
        error_type="validation_error",
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error.to_dict(),
    )


async def application_error_handler(request: Request, exc: ApplicationError) -> JSONResponse:
    """
    Handle generic ApplicationError exceptions.

    Fallback handler for application errors without specific handler.

    Args:
        request: HTTP request
        exc: ApplicationError exception

    Returns:
        JSONResponse with 500 status

    """
    logger.error(f"Application error: {exc.message}", exc_info=True)

    error = ErrorResponse(
        message=exc.message,
        details=exc.details,
        error_type="application_error",
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error.to_dict(),
    )


# Override default FastAPI handlers for enhanced logging
async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle FastAPI request validation errors with enhanced logging.

    Args:
        request: HTTP request
        exc: RequestValidationError from FastAPI

    Returns:
        JSONResponse with 422 status

    """
    logger.warning(f"Request validation failed: {exc.errors()}")

    # Process validation errors to ensure JSON serializability
    validation_errors = []
    for error in exc.errors():
        # Create JSON-safe error dict
        safe_error = {
            "type": error.get("type"),
            "loc": error.get("loc"),
            "msg": error.get("msg"),
            "input": str(error.get("input"))[:100] if error.get("input") is not None else None,
        }
        # Only include context if it doesn't contain non-serializable objects
        if "ctx" in error and isinstance(error["ctx"], dict):
            safe_ctx = {}
            for key, value in error["ctx"].items():
                # Convert non-serializable values to strings
                if isinstance(value, (str, int, float, bool, type(None))):
                    safe_ctx[key] = value
                else:
                    safe_ctx[key] = str(value)
            if safe_ctx:
                safe_error["ctx"] = safe_ctx
        validation_errors.append(safe_error)

    error = ErrorResponse(
        message="Request validation failed",
        details={"validation_errors": validation_errors},
        error_type="request_validation_error",
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error.to_dict(),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTPException with enhanced logging.

    Args:
        request: HTTP request
        exc: StarletteHTTPException

    Returns:
        JSONResponse with exception status code

    """
    # Log based on severity
    if exc.status_code >= 500:
        logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    elif exc.status_code >= 400:
        logger.warning(f"HTTP {exc.status_code}: {exc.detail}")

    error = ErrorResponse(
        message=str(exc.detail),
        error_type="http_exception",
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error.to_dict(),
        headers=getattr(exc, "headers", None),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle any unhandled exceptions.

    Last resort handler for unexpected errors.

    Args:
        request: HTTP request
        exc: Any unhandled exception

    Returns:
        JSONResponse with 500 status

    """
    logger.error(f"Unhandled exception: {exc!s}", exc_info=True)

    # Don't expose internal error details in production
    error = ErrorResponse(
        message="An unexpected error occurred. Please try again later.",
        error_type="internal_server_error",
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error.to_dict(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers with FastAPI application.

    This function should be called once during application startup.

    Args:
        app: FastAPI application instance

    """
    # Custom domain exceptions
    app.add_exception_handler(RunNotFoundError, run_not_found_handler)
    app.add_exception_handler(InvalidPromptError, invalid_prompt_handler)
    app.add_exception_handler(PlanGenerationError, plan_generation_handler)
    app.add_exception_handler(ExecutionError, execution_error_handler)
    app.add_exception_handler(ServiceUnavailableError, service_unavailable_handler)
    app.add_exception_handler(ValidationError, validation_error_handler)

    # Generic application error (fallback)
    app.add_exception_handler(ApplicationError, application_error_handler)

    # Override FastAPI defaults for enhanced logging
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Catch-all for unhandled exceptions
    app.add_exception_handler(Exception, unhandled_exception_handler)

    logger.info("Exception handlers registered successfully")
