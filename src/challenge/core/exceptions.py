"""
Custom exceptions for the application.

This module defines domain-specific exceptions following FastAPI best practices:
- Custom exceptions inherit from standard Python exceptions
- Exception handlers map exceptions to appropriate HTTP responses
- Each exception includes context for clear error messages

Pattern:
    1. Define custom exception class
    2. Register handler in exception_handlers module
    3. Raise in business logic (services/routes)
    4. Handler converts to HTTPException with proper status code
"""


class ApplicationError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: dict[str, str] | None = None) -> None:
        """
        Initialize application error.

        Args:
            message: Error description
            details: Additional context (optional)

        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Run-related exceptions
class RunNotFoundError(ApplicationError):
    """Raised when run doesn't exist."""

    def __init__(self, run_id: str) -> None:
        """
        Initialize run not found error.

        Args:
            run_id: The ID of the run that wasn't found

        """
        super().__init__(
            message=f"Run not found: {run_id}",
            details={"run_id": run_id},
        )
        self.run_id = run_id


class InvalidPromptError(ApplicationError):
    """Raised when prompt is invalid or cannot be processed."""

    def __init__(self, prompt: str, reason: str) -> None:
        """
        Initialize invalid prompt error.

        Args:
            prompt: The invalid prompt text
            reason: Why the prompt is invalid

        """
        super().__init__(
            message=f"Invalid prompt: {reason}",
            details={"prompt": prompt[:100], "reason": reason},
        )
        self.prompt = prompt
        self.reason = reason


class PlanGenerationError(ApplicationError):
    """Raised when execution plan generation fails."""

    def __init__(self, prompt: str, reason: str) -> None:
        """
        Initialize plan generation error.

        Args:
            prompt: The prompt that failed planning
            reason: Why planning failed

        """
        super().__init__(
            message=f"Failed to generate execution plan: {reason}",
            details={"prompt": prompt[:100], "reason": reason},
        )
        self.prompt = prompt
        self.reason = reason


class ExecutionError(ApplicationError):
    """Raised when run execution fails."""

    def __init__(self, run_id: str, step: str, reason: str) -> None:
        """
        Initialize execution error.

        Args:
            run_id: The run that failed
            step: The execution step that failed
            reason: Why execution failed

        """
        super().__init__(
            message=f"Execution failed at step '{step}': {reason}",
            details={"run_id": run_id, "step": step, "reason": reason},
        )
        self.run_id = run_id
        self.step = step
        self.reason = reason


# Service availability exceptions
class ServiceUnavailableError(ApplicationError):
    """Raised when service or dependency is unavailable."""

    def __init__(self, service: str, reason: str) -> None:
        """
        Initialize service unavailable error.

        Args:
            service: Name of the unavailable service
            reason: Why the service is unavailable

        """
        super().__init__(
            message=f"Service unavailable: {service}",
            details={"service": service, "reason": reason},
        )
        self.service = service
        self.reason = reason


# Validation exceptions
class ValidationError(ApplicationError):
    """Raised when validation fails."""

    def __init__(self, field: str, value: str, constraint: str) -> None:
        """
        Initialize validation error.

        Args:
            field: Field name that failed validation
            value: Invalid value
            constraint: Validation constraint that was violated

        """
        super().__init__(
            message=f"Validation failed for {field}: {constraint}",
            details={"field": field, "value": str(value)[:100], "constraint": constraint},
        )
        self.field = field
        self.value = value
        self.constraint = constraint
