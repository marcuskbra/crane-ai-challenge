# Makefile LLM Commands Reference

Quick reference for local LLM testing commands added to the Makefile.

## Docker-Based Workflow (Recommended for CI/CD)

### Complete Docker Setup and Test Flow
```bash
# 1. Start all services (Ollama + LiteLLM)
make llm-docker-up

# 2. Run tests with local LLM
make llm-docker-test

# 3. View service logs (in another terminal)
make llm-docker-logs

# 4. Stop services when done
make llm-docker-down

# 5. Clean up volumes (optional)
make llm-docker-clean
```

### Individual Docker Commands

| Command | Description | Use Case |
|---------|-------------|----------|
| `make llm-docker-up` | Start Ollama + LiteLLM services | Initial setup, daily development |
| `make llm-docker-down` | Stop all LLM services | End of day, resource cleanup |
| `make llm-docker-logs` | View live service logs | Debugging, monitoring |
| `make llm-docker-test` | Run tests with Docker LLM | CI/CD, integration testing |
| `make llm-docker-clean` | Remove volumes and images | Full reset, disk space cleanup |

## Local Development Workflow (For Active Development)

### Complete Local Setup and Test Flow
```bash
# 1. One-time setup (installs Ollama + LiteLLM)
make llm-local-setup

# 2. Pull recommended models (one-time, ~2-3GB download)
make llm-local-pull

# 3. Start LiteLLM proxy (keep running in terminal)
make llm-local-start

# 4. In another terminal, run tests
make llm-local-test

# 5. Stop services when done
make llm-local-stop
```

### Individual Local Commands

| Command | Description | Use Case |
|---------|-------------|----------|
| `make llm-local-setup` | Install Ollama + LiteLLM | First-time setup |
| `make llm-local-pull` | Download models (Qwen2.5-3B, Phi-3-mini) | After setup, model updates |
| `make llm-local-start` | Start LiteLLM proxy (foreground) | Daily development sessions |
| `make llm-local-stop` | Stop proxy and Ollama | End of session, resource cleanup |
| `make llm-local-test` | Run tests with local LLM | Development testing |

## Utility Commands

| Command | Description | Output |
|---------|-------------|--------|
| `make llm-check` | Check installation status | Shows Ollama/LiteLLM versions, service status |
| `make llm-status` | Complete status overview | Installation + configuration + quick commands |
| `make llm-models` | List downloaded models | Shows all local models + recommendations |

## Quick Start Examples

### New Developer Setup
```bash
# Option 1: Docker (simplest, no local installation)
make llm-docker-up && make llm-docker-test

# Option 2: Local development (better for iterative testing)
make llm-local-setup
make llm-local-pull
make llm-local-start  # In terminal 1
make llm-local-test   # In terminal 2
```

### Daily Development
```bash
# Docker approach
make llm-docker-up     # Start services
# ... work on code ...
make llm-docker-test   # Test changes
make llm-docker-down   # End of day

# Local approach
make llm-local-start   # Terminal 1: start proxy
make llm-local-test    # Terminal 2: run tests
make llm-local-stop    # End of day
```

### Troubleshooting
```bash
# Check what's installed and running
make llm-status

# List downloaded models
make llm-models

# View Docker logs for debugging
make llm-docker-logs

# Reset Docker environment
make llm-docker-clean
make llm-docker-up
```

## Integration with Existing Commands

### Running Tests
```bash
# Standard tests (uses OpenAI or pattern-based fallback)
make test

# Tests with local LLM (Docker)
make llm-docker-test

# Tests with local LLM (local development)
make llm-local-test

# All tests with coverage (local LLM)
OPENAI_BASE_URL=http://localhost:4000 OPENAI_MODEL=qwen2.5:3b make coverage
```

### Development Workflow
```bash
# Traditional workflow
make dev-install
make test
make lint
make format

# With local LLM testing
make dev-install
make llm-local-setup       # One-time
make llm-local-pull        # One-time
make llm-local-start &     # Background
make llm-local-test        # Instead of make test
make lint
make format
```

## Environment Variables

The LLM commands use these environment variables (set automatically by make targets):

```bash
OPENAI_BASE_URL=http://localhost:4000  # LiteLLM proxy endpoint
OPENAI_MODEL=qwen2.5:3b                # Model to use
```

You can also set these manually in `.env` for persistent configuration:
```bash
# In .env file
OPENAI_BASE_URL=http://localhost:4000
OPENAI_MODEL=qwen2.5:3b
```

## Benefits Summary

### Docker Approach
✅ **Pros**: Isolated environment, no system changes, perfect for CI/CD
⚠️ **Cons**: Slower startup (30s wait), more resource intensive

### Local Development Approach
✅ **Pros**: Faster iteration, lower overhead, better for development
⚠️ **Cons**: Requires system installation, manages system services

## Related Documentation

- **Setup Guide**: `claudedocs/local_llm_testing_guide.md` - Comprehensive setup instructions
- **Configuration**: `config/litellm_config.yaml` - LiteLLM proxy configuration
- **Docker Compose**: `docker-compose.litellm.yml` - Container orchestration
- **README**: Main project README with quick start section
- **Environment**: `.env.example` - Configuration template

## Help

View all available commands:
```bash
make help
```

View LLM-specific commands:
```bash
make help | grep -A 20 "Local LLM"
```
