# Deployment Guide

## Configuration

The application can be configured using environment variables or by creating a `.env` file:

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your settings
```

**Key Configuration Options:**

```bash
# LLM Configuration (optional - uses pattern-based fallback if not set)
# Primary configuration (LiteLLM multi-provider support)
LLM_API_KEY=sk-your-api-key-here
LLM_MODEL=gpt-4o-mini                    # or claude-3-5-sonnet-20241022, qwen2.5:3b
LLM_TEMPERATURE=0.1
LLM_BASE_URL=                            # Optional: http://localhost:11434/v1 for Ollama
LLM_PROVIDER=openai                      # openai, anthropic, or ollama

# Backward compatible (deprecated - use LLM_* above)
# OPENAI_API_KEY=sk-your-api-key-here
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_TEMPERATURE=0.0

# Application Settings
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
DEBUG=true
LOG_LEVEL=INFO

# Orchestrator Configuration
MAX_RETRIES=3           # Retry attempts for failed steps
STEP_TIMEOUT=30.0       # Timeout in seconds for each step execution
```

**Timeout Configuration:**

The `STEP_TIMEOUT` setting controls how long the orchestrator waits for each step to complete before timing out. This
prevents steps from hanging indefinitely:

- **Default**: 30 seconds (suitable for most operations)
- **Recommendation**: Increase for long-running operations (e.g., `60.0`)
- **Behavior**: On timeout, the step is marked as failed with a clear error message

Example for custom timeout:

```python
from challenge.orchestrator import Orchestrator

# Custom timeout configuration
orchestrator = Orchestrator(
    step_timeout=60.0  # 60 second timeout for long-running operations
)
```

## Docker Deployment

The application includes production-ready Docker support with multi-stage builds for optimal image size.

**Quick Start with Docker:**

```bash
# Build and run with docker-compose (production mode)
docker-compose up -d

# Or build manually
docker build -t crane-ai-agent:latest .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key-here \
  -e STEP_TIMEOUT=30.0 \
  crane-ai-agent:latest

# Check health
curl http://localhost:8000/api/v1/health
```

**Development Mode with Hot Reload:**

```bash
# Run development service with volume mounting
docker-compose --profile dev up

# This enables:
# - Code hot-reload on file changes
# - Debug logging
# - Source code mounted from host
```

**Docker Configuration:**

The Dockerfile uses multi-stage builds for:

- **Smaller Images**: Builder stage separate from runtime (~200MB final image)
- **Security**: Non-root user execution
- **Performance**: UV package manager for fast builds
- **Health Checks**: Built-in liveness probes

**Environment Variables:**

All configuration options from `.env.example` are supported as environment variables in Docker.

**Resource Limits:**

Default limits in docker-compose.yml:

- CPU: 1.0 core (0.5 reserved)
- Memory: 512MB (256MB reserved)

Adjust in `docker-compose.yml` based on workload.

## Related Documentation

- **Architecture**: See [System Architecture](./architecture.md) for system overview
- **Multi-Provider LLM**: See [Multi-Provider LLM Setup](./multi_provider_llm.md) for LLM configuration details
- **API Examples**: See [API Examples](./api_examples.md) for testing deployed application
- **Known Limitations**: See [Known Limitations](./limitations.md) for production considerations
