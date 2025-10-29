"""
Tests for health check endpoints.

Tests verify:
- Basic health check returns correct structure
- Liveness probe always returns 200 OK
- Readiness probe checks component status
- Response models match expected schema
- Proper status codes and headers
"""

from datetime import datetime

import pytest
from fastapi import status

from challenge import __version__


@pytest.mark.unit
def test_health_endpoint_returns_200(test_client):
    """Test that /health endpoint returns 200 OK."""
    response = test_client.get("/api/v1/health")

    assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
def test_health_endpoint_response_structure(test_client):
    """Test that /health endpoint returns expected response structure."""
    response = test_client.get("/api/v1/health")
    data = response.json()

    # Verify required fields are present
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data
    assert "system" in data
    assert "checks" in data


@pytest.mark.unit
def test_health_endpoint_contains_version(test_client):
    """Test that /health endpoint contains correct version."""
    response = test_client.get("/api/v1/health")
    data = response.json()

    assert data["version"] == __version__


@pytest.mark.unit
def test_health_endpoint_status_is_healthy(test_client):
    """Test that /health endpoint status is 'healthy'."""
    response = test_client.get("/api/v1/health")
    data = response.json()

    assert data["status"] == "healthy"


@pytest.mark.unit
def test_health_endpoint_timestamp_is_valid(test_client):
    """Test that /health endpoint timestamp is valid ISO format."""
    response = test_client.get("/api/v1/health")
    data = response.json()

    # Verify timestamp can be parsed as datetime
    timestamp = data["timestamp"]
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


@pytest.mark.unit
def test_health_endpoint_includes_system_info(test_client):
    """Test that /health endpoint includes system information."""
    response = test_client.get("/api/v1/health")
    data = response.json()

    system = data["system"]
    assert "python_version" in system
    assert "platform" in system
    assert "architecture" in system


@pytest.mark.unit
def test_health_endpoint_includes_component_checks(test_client):
    """Test that /health endpoint includes component health checks."""
    response = test_client.get("/api/v1/health")
    data = response.json()

    checks = data["checks"]
    assert isinstance(checks, dict)
    assert "application" in checks
    assert checks["application"] == "healthy"


@pytest.mark.unit
def test_liveness_endpoint_returns_200(test_client):
    """Test that /health/live endpoint returns 200 OK."""
    response = test_client.get("/api/v1/health/live")

    assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
def test_liveness_endpoint_response_structure(test_client):
    """Test that /health/live endpoint returns expected response structure."""
    response = test_client.get("/api/v1/health/live")
    data = response.json()

    assert "alive" in data
    assert "timestamp" in data


@pytest.mark.unit
def test_liveness_endpoint_always_alive(test_client):
    """Test that /health/live endpoint always returns alive=True."""
    response = test_client.get("/api/v1/health/live")
    data = response.json()

    assert data["alive"] is True


@pytest.mark.unit
def test_liveness_endpoint_timestamp_is_valid(test_client):
    """Test that /health/live endpoint timestamp is valid ISO format."""
    response = test_client.get("/api/v1/health/live")
    data = response.json()

    # Verify timestamp can be parsed as datetime
    timestamp = data["timestamp"]
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


@pytest.mark.unit
def test_readiness_endpoint_returns_200(test_client):
    """Test that /health/ready endpoint returns 200 OK."""
    response = test_client.get("/api/v1/health/ready")

    assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
def test_readiness_endpoint_response_structure(test_client):
    """Test that /health/ready endpoint returns expected response structure."""
    response = test_client.get("/api/v1/health/ready")
    data = response.json()

    assert "ready" in data
    assert "checks" in data
    assert "timestamp" in data


@pytest.mark.unit
def test_readiness_endpoint_includes_checks(test_client):
    """Test that /health/ready endpoint includes readiness checks."""
    response = test_client.get("/api/v1/health/ready")
    data = response.json()

    checks = data["checks"]
    assert isinstance(checks, dict)
    assert "application" in checks
    assert isinstance(checks["application"], bool)


@pytest.mark.unit
def test_readiness_endpoint_ready_status(test_client):
    """Test that /health/ready endpoint returns ready status."""
    response = test_client.get("/api/v1/health/ready")
    data = response.json()

    assert data["ready"] is True


@pytest.mark.unit
def test_readiness_endpoint_timestamp_is_valid(test_client):
    """Test that /health/ready endpoint timestamp is valid ISO format."""
    response = test_client.get("/api/v1/health/ready")
    data = response.json()

    # Verify timestamp can be parsed as datetime
    timestamp = data["timestamp"]
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


@pytest.mark.unit
def test_all_health_endpoints_return_json(test_client):
    """Test that all health endpoints return JSON content type."""
    endpoints = [
        "/api/v1/health",
        "/api/v1/health/live",
        "/api/v1/health/ready",
    ]

    for endpoint in endpoints:
        response = test_client.get(endpoint)
        assert "application/json" in response.headers["content-type"]


@pytest.mark.unit
def test_health_endpoints_are_fast(test_client, benchmark_timer):
    """Test that health endpoints respond quickly (<100ms)."""
    endpoints = [
        "/api/v1/health",
        "/api/v1/health/live",
        "/api/v1/health/ready",
    ]

    for endpoint in endpoints:
        with benchmark_timer() as timer:
            test_client.get(endpoint)

        # Health checks should be very fast (<100ms)
        assert timer.elapsed < 0.1, f"{endpoint} took {timer.elapsed:.3f}s (>100ms)"


@pytest.mark.smoke
def test_health_check_smoke_test(test_client):
    """Smoke test: basic health check works."""
    response = test_client.get("/api/v1/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
