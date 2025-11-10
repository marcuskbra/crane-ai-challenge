# Crane AI Agent Runtime - Multi-Stage Docker Build
# Optimized for production deployment with minimal image size

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies using uv (much faster than pip)
# Create virtual environment and install production dependencies only
# Note: uv pip install . installs base dependencies (production)
#       Optional dependencies (dev, test) are excluded by default
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install .

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM python:3.12-slim

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# Copy configuration files
COPY --chown=appuser:appuser pyproject.toml README.md ./

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    MAX_RETRIES=3 \
    STEP_TIMEOUT=30.0

# Expose application port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Run the application
CMD ["python", "-m", "challenge"]
