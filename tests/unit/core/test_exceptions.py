"""
Unit tests for custom exceptions.

Tests the custom exception classes defined in core.exceptions module.
"""

from challenge.core.exceptions import (
    ApplicationError,
    ExecutionError,
    InvalidPromptError,
    PlanGenerationError,
    RunNotFoundError,
    ServiceUnavailableError,
    ValidationError,
)


class TestApplicationError:
    """Test ApplicationError base exception."""

    def test_application_error_with_message(self):
        """Test creating ApplicationError with message only."""
        error = ApplicationError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}

    def test_application_error_with_details(self):
        """Test creating ApplicationError with message and details."""
        details = {"key": "value", "status": "failed"}
        error = ApplicationError("Test error", details)

        assert error.message == "Test error"
        assert error.details == details


class TestRunNotFoundError:
    """Test RunNotFoundError exception."""

    def test_run_not_found_error_creation(self):
        """Test creating RunNotFoundError."""
        run_id = "test-run-123"
        error = RunNotFoundError(run_id)

        assert error.run_id == run_id
        assert error.message == f"Run not found: {run_id}"
        assert error.details == {"run_id": run_id}

    def test_run_not_found_error_inheritance(self):
        """Test RunNotFoundError inherits from ApplicationError."""
        error = RunNotFoundError("test-id")
        assert isinstance(error, ApplicationError)
        assert isinstance(error, Exception)


class TestInvalidPromptError:
    """Test InvalidPromptError exception."""

    def test_invalid_prompt_error_creation(self):
        """Test creating InvalidPromptError."""
        prompt = "invalid prompt text"
        reason = "Prompt is too short"
        error = InvalidPromptError(prompt, reason)

        assert error.prompt == prompt
        assert error.reason == reason
        assert error.message == f"Invalid prompt: {reason}"
        assert error.details["prompt"] == prompt
        assert error.details["reason"] == reason

    def test_invalid_prompt_error_truncates_long_prompt(self):
        """Test that long prompts are truncated in details."""
        long_prompt = "a" * 200
        error = InvalidPromptError(long_prompt, "too long")

        # Prompt should be truncated to 100 characters in details
        assert len(error.details["prompt"]) == 100
        assert error.prompt == long_prompt  # Original is preserved


class TestPlanGenerationError:
    """Test PlanGenerationError exception."""

    def test_plan_generation_error_creation(self):
        """Test creating PlanGenerationError."""
        prompt = "calculate something"
        reason = "LLM service unavailable"
        error = PlanGenerationError(prompt, reason)

        assert error.prompt == prompt
        assert error.reason == reason
        assert "Failed to generate execution plan" in error.message
        assert error.details["reason"] == reason


class TestExecutionError:
    """Test ExecutionError exception."""

    def test_execution_error_creation(self):
        """Test creating ExecutionError."""
        run_id = "run-123"
        step = "calculate_step"
        reason = "Division by zero"
        error = ExecutionError(run_id, step, reason)

        assert error.run_id == run_id
        assert error.step == step
        assert error.reason == reason
        assert "Execution failed at step" in error.message
        assert error.details["run_id"] == run_id
        assert error.details["step"] == step


class TestServiceUnavailableError:
    """Test ServiceUnavailableError exception."""

    def test_service_unavailable_error_creation(self):
        """Test creating ServiceUnavailableError."""
        service = "database"
        reason = "Connection timeout"
        error = ServiceUnavailableError(service, reason)

        assert error.service == service
        assert error.reason == reason
        assert f"Service unavailable: {service}" == error.message
        assert error.details["service"] == service
        assert error.details["reason"] == reason


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_creation(self):
        """Test creating ValidationError."""
        field = "email"
        value = "invalid-email"
        constraint = "must be valid email format"
        error = ValidationError(field, value, constraint)

        assert error.field == field
        assert error.value == value
        assert error.constraint == constraint
        assert "Validation failed for" in error.message
        assert error.details["field"] == field

    def test_validation_error_truncates_long_value(self):
        """Test that long values are truncated in details."""
        long_value = "x" * 200
        error = ValidationError("field", long_value, "constraint")

        # Value should be truncated to 100 characters in details
        assert len(error.details["value"]) == 100
        assert error.value == long_value  # Original is preserved
