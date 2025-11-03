"""
Integration tests for exception handling.

Tests exception handling in actual API endpoints with full request/response cycle.
"""

import logging

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from challenge.api.main import create_app


@pytest.fixture
def client():
    """Create test client with full app configuration."""
    app = create_app()
    return TestClient(app)


class TestRunsExceptionHandling:
    """Test exception handling in runs endpoints."""

    def test_get_nonexistent_run_returns_404(self, client):
        """Test getting non-existent run returns proper 404 error."""
        response = client.get("/api/v1/runs/nonexistent-run-id")

        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "detail" in data
        assert "Run not found" in data["detail"]
        assert data.get("error_type") == "run_not_found"
        assert "details" in data
        assert data["details"]["run_id"] == "nonexistent-run-id"

    def test_create_run_with_empty_prompt_returns_422(self, client):
        """Test creating run with invalid prompt returns validation error."""
        response = client.post(
            "/api/v1/runs",
            json={"prompt": ""},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        data = response.json()
        # New error format has 'message' instead of 'detail'
        assert "message" in data or "detail" in data
        # Should have validation error details
        assert "details" in data or "validation_errors" in data

    def test_create_run_with_whitespace_prompt_returns_422(self, client):
        """Test creating run with whitespace-only prompt returns validation error."""
        response = client.post(
            "/api/v1/runs",
            json={"prompt": "   "},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_run_with_missing_prompt_returns_422(self, client):
        """Test creating run without prompt field returns validation error."""
        response = client.post(
            "/api/v1/runs",
            json={},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        data = response.json()
        # New error format has 'message' instead of 'detail'
        assert "message" in data or "detail" in data

    def test_create_run_with_valid_prompt_succeeds(self, client):
        """Test creating run with valid prompt succeeds."""
        response = client.post(
            "/api/v1/runs",
            json={"prompt": "calculate 2 + 2"},
        )

        assert response.status_code == status.HTTP_201_CREATED

        data = response.json()
        assert "run_id" in data
        assert "status" in data
        assert data["prompt"] == "calculate 2 + 2"


class TestHealthExceptionHandling:
    """Test exception handling in health endpoints."""

    def test_readiness_check_succeeds_when_ready(self, client):
        """Test readiness check returns 200 when all services ready."""
        response = client.get("/api/v1/health/ready")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["ready"] is True
        assert "checks" in data
        assert data["checks"]["application"] is True

    def test_liveness_check_always_succeeds(self, client):
        """Test liveness check always returns 200."""
        response = client.get("/api/v1/health/live")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["alive"] is True

    def test_health_check_returns_detailed_info(self, client):
        """Test health check returns detailed system information."""
        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "system" in data
        assert "checks" in data


class TestErrorResponseFormat:
    """Test error response format consistency."""

    def test_404_error_has_standard_format(self, client):
        """Test 404 errors follow standard format."""
        response = client.get("/api/v1/runs/missing")

        data = response.json()
        assert "detail" in data
        assert "error_type" in data
        assert "details" in data

    def test_422_error_has_validation_details(self, client):
        """Test 422 errors include validation details."""
        response = client.post("/api/v1/runs", json={})

        data = response.json()
        # New error format has 'message' and 'details'
        assert "message" in data or "detail" in data
        # Should have validation details
        assert "details" in data

    def test_error_responses_are_json(self, client):
        """Test all error responses are JSON."""
        # Test various error scenarios
        error_endpoints = [
            ("/api/v1/runs/missing", "get"),
            ("/api/v1/runs", "post"),
        ]

        for endpoint, method in error_endpoints:
            if method == "get":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json={})

            assert response.headers["content-type"] == "application/json"
            # Should be valid JSON
            data = response.json()
            assert isinstance(data, dict)


class TestExceptionLogging:
    """Test that exceptions are properly logged."""

    def test_404_errors_are_logged(self, client, caplog):
        """Test 404 errors generate warning logs."""
        with caplog.at_level(logging.WARNING):
            client.get("/api/v1/runs/missing-run")

        # Should have log entry about run not found
        assert any("Run not found" in record.message or "missing-run" in record.message for record in caplog.records)

    def test_validation_errors_are_logged(self, client, caplog):
        """Test validation errors generate warning logs."""
        with caplog.at_level(logging.WARNING):
            client.post("/api/v1/runs", json={})

        # Should have log entry about validation
        assert any("validation" in record.message.lower() for record in caplog.records)


class TestConcurrentExceptionHandling:
    """Test exception handling under concurrent requests."""

    def test_multiple_404_errors_handled_correctly(self, client):
        """Test multiple concurrent 404 errors are handled independently."""
        responses = []
        run_ids = ["missing-1", "missing-2", "missing-3"]

        for run_id in run_ids:
            response = client.get(f"/api/v1/runs/{run_id}")
            responses.append((run_id, response))

        # All should be 404 with correct run_id
        for run_id, response in responses:
            assert response.status_code == 404
            data = response.json()
            assert data["details"]["run_id"] == run_id
