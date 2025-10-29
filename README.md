# Skeleton Challenge

A modern Python project with Clean Architecture, type-safe error handling, and following production-standards for development workflow.

## 🏗️ System Architecture

This project follows a **Simplified 3-Layer Clean Architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│  ┌────────────────────────────────────────────────────┐     │
│  │  API Endpoints - CLI Commands - Web Interfaces     │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │ Uses
┌────────────────────────▼────────────────────────────────────┐
│                      DOMAIN LAYER                           │
│  ┌────────────────────────────────────────────────────┐     │
│  │  • Domain Services (Business Logic)                │     │
│  │  • Entities & Value Objects                        │     │
│  │  • Discriminated Unions for Error Handling         │     │
│  │  • Business Rules & Validations                    │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │ Uses
┌────────────────────────▼────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                       │
│  ┌────────────────────────────────────────────────────┐     │
│  │  External APIs - Database - File System - Cache    │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Features

- **Simplified 3-Layer Architecture**: Clean separation between Presentation, Domain, and Infrastructure
- **Type Safety**: Full Pydantic validation and discriminated unions for error handling
- **Performance**: Async-first design for high concurrency
- **No Hidden Exceptions**: All errors handled through type-safe discriminated unions
- **Dependency Inversion**: Domain layer has no dependencies on external layers

### Core Benefits

- ✅ **Type-Safe Error Handling**: Discriminated unions instead of exceptions
- ✅ **Modern Python Tooling**: Ruff (Rust-based), uv (10-100x faster than pip), ty for type checking
- ✅ **Clean Architecture**: Clear separation of concerns with dependency inversion
- ✅ **Test-Driven Development**: Comprehensive test suite with builders pattern
- ✅ **Production-Ready**: Pre-commit hooks, CI/CD ready, environment-based config

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager (recommended for speed)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd skeleton-challenge

# Install dependencies with uv (recommended)
make install        # Production dependencies only
make dev-install    # All dependencies including dev tools

# Or with pip (slower)
pip install -e ".[dev,test-integration]"
```

### Running the Application

```bash
# Run the application
make run

# Or directly
uv run python -m challenge
```

### Running the API

The project includes a FastAPI-based REST API with health check endpoints:

```bash
# Start API in development mode (with auto-reload)
make api-dev

# Start API in production mode (4 workers)
make api-prod

# Run API tests
make api-test

# Check API health
make api-health

# Open API documentation (Swagger UI)
make api-docs
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/v1/health
- **Liveness**: http://localhost:8000/api/v1/health/live
- **Readiness**: http://localhost:8000/api/v1/health/ready
- **API Docs**: http://localhost:8000/api/docs (development only)
- **ReDoc**: http://localhost:8000/api/redoc (development only)

#### API Features

- ✅ **FastAPI Framework**: Modern, fast (high-performance), async-first web framework
- ✅ **Health Checks**: Liveness and readiness probes for Kubernetes/production
- ✅ **Error Handling**: Integration with domain error discriminated unions
- ✅ **CORS Middleware**: Configurable cross-origin resource sharing
- ✅ **Request Logging**: Automatic logging of all requests and responses
- ✅ **OpenAPI Documentation**: Auto-generated Swagger UI and ReDoc
- ✅ **Type Safety**: Full Pydantic validation for request/response models
- ✅ **Production Ready**: Application factory pattern for testability

### Development Workflow

This project uses a **dual approach** for optimal developer experience:

#### Local Development with `uv` (Recommended)

We use `uv` for local development due to its **10-100x speed improvement** over pip:

```bash
# All Makefile commands use uv automatically
make test           # Runs: uv run pytest tests/unit/
make lint           # Runs: uv run ruff check src/ tests/
make format         # Runs: uv run ruff format src/ tests/
make run            # Runs: uv run python -m challenge

# Or use uv directly for any Python command
uv run python script.py
uv run pytest -xvs
```

Benefits of `uv`:
- ⚡ **Lightning fast** - Written in Rust, 10-100x faster than pip
- 🔄 **Auto venv management** - No manual activation needed
- 🔒 **Lockfile support** - `uv.lock` ensures reproducible builds
- 📦 **Smart caching** - Dependencies cached across projects

#### CI/CD with `tox` (For Matrix Testing)

For CI/CD pipelines and testing across multiple Python versions:

```bash
# Run comprehensive tests with coverage
make tox-coverage   # Runs: tox -e coverage

# Test against specific Python version
tox -e py312

# Run all validation steps
tox -e validate
```

The `tox` configuration is **enhanced with `uv`** for 1.5x faster CI builds:
- Uses `uv-venv-runner` instead of virtualenv
- Leverages `uv sync --frozen` for lockfile-based installs
- All commands use `uv run` for consistent execution

#### Quick Command Reference

```bash
# Development
make test           # Unit tests only (fast)
make test-all       # All tests including integration
make coverage       # Generate coverage report

# Code quality
make lint           # Check code style
make format         # Format code
make type-check     # Run type checker
make validate       # Run all checks

# Quick commands
make fix           # Auto-fix lint and format issues
make quick         # Fast tests + quality checks
```

## 📁 Project Structure

```
skeleton-challenge/
├── src/challenge/      # Source code (Clean Architecture)
│   ├── domain/                 # Core business logic
│   │   ├── base_models.py     # Base classes for entities & value objects
│   │   ├── errors.py          # Discriminated unions for error handling
│   │   ├── entities/          # Domain entities
│   │   ├── value_objects/     # Immutable value objects
│   │   ├── services/          # Domain services
│   │   └── interfaces/        # Port definitions (abstractions)
│   ├── infrastructure/         # External world integration
│   │   ├── clients/           # External API clients
│   │   └── utils/             # Infrastructure utilities
│   └── presentation/          # User interface layer
│       └── [api|cli|web]/     # Specific interface implementation
├── tests/                      # Test suite
│   ├── unit/                  # Fast unit tests
│   ├── integration/           # Integration tests
│   ├── conftest.py           # Shared fixtures
│   └── builders.py           # Test data builders
├── Makefile                    # Development automation
├── pyproject.toml             # Project configuration
├── pytest.ini                 # Test configuration
├── .env.example               # Environment variables template
├── .env.test                  # Test environment settings
├── .gitignore                 # Git ignore rules
└── .pre-commit-config.yaml   # Pre-commit hooks

```

## 🧪 Testing

The project uses pytest with comprehensive fixtures and builders pattern:

### Test Categories

- **Unit Tests** (`tests/unit/`): Fast, no external dependencies
- **Integration Tests** (`tests/integration/`): Test external integrations
- **Contract Tests**: Define behavioral requirements

### Running Tests

```bash
# Run specific test categories
make test-unit        # Unit tests only
make test-integration # Integration tests only
make test-fast       # Quick test run

# Run with coverage
make coverage        # HTML report in htmlcov/

# Using pytest directly
uv run pytest -xvs tests/unit/
uv run pytest -m unit  # Run by marker
```

### Test Builders

Use the builder pattern for creating test data:

```python
from tests.builders import entity, request, response

# Create test entity
test_entity = (
    entity()
    .with_name("Test Item")
    .with_metadata({"key": "value"})
    .build()
)

# Create test request
test_request = (
    request()
    .post("/api/v1/items")
    .with_body({"name": "item"})
    .with_auth("token")
    .build()
)
```

## 🔧 Configuration

### Environment Variables

Configuration is managed through environment variables. Copy `.env.example` to `.env` and adjust values:

```bash
cp .env.example .env
```

Key configuration areas:
- Application settings (environment, debug mode)
- Database configuration
- Cache settings
- API configuration
- Security settings
- Feature flags

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pre-commit install
```

This will run:
- Code formatting (ruff)
- Linting (ruff)
- Type checking (ty)
- Security scanning (bandit)
- Unit tests (on push)

## 🏭 Production Deployment

### Using Tox for CI/CD

The project includes tox configuration for multi-environment testing:

```bash
# Run validation suite
tox -e validate

# Test specific Python version
tox -e py312

# Run all environments
tox
```

### Docker Support (Optional)

Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN uv pip install -e .

# Run application
CMD ["python", "-m", "challenge"]
```

## 📚 Documentation

### Domain-Driven Design

The project follows DDD principles:

- **Entities**: Objects with identity that can change over time
- **Value Objects**: Immutable objects defined by their attributes
- **Aggregates**: Consistency boundaries with aggregate roots
- **Domain Services**: Business logic that doesn't belong to entities
- **Domain Events**: Things that happened in the domain

### Type-Safe Error Handling

This project uses **discriminated unions** for error handling instead of exceptions, providing compile-time safety and explicit error handling:

```python
from challenge.domain.errors import (
    CreateEntityResult,
    CreateEntitySuccess,
    ValidationError,
    ConflictError
)

# Every domain operation returns Success/Error union
result = await service.create_entity(data)

# Type-safe handling - compiler ensures all cases are handled
if isinstance(result, CreateEntitySuccess):
    # Type checker knows result is CreateEntitySuccess here
    return {"id": result.entity_id, "data": result.entity}

elif isinstance(result, ValidationError):
    # Handle validation errors
    logger.warning(f"Validation failed: {result.field_errors}")
    return {"error": "validation", "details": result.field_errors}, 400

elif isinstance(result, ConflictError):
    # Handle conflicts
    logger.error(f"Conflict on {result.conflicting_field}")
    return {"error": "conflict", "field": result.conflicting_field}, 409

# Type checker ensures exhaustive handling
```

#### Benefits of Discriminated Unions

- **No Unexpected Runtime Exceptions**: All failure modes are explicit in the return type
- **Type-Safe Error Handling**: Can't forget to handle errors - compiler ensures it
- **Clear Operation Contracts**: Return types clearly show all possible outcomes
- **Better IDE Support**: Auto-completion knows exact error types and their fields
- **Self-Documenting**: Function signatures document all success and failure cases

#### Error Handling Pattern Example

```python
# Domain service with discriminated union return
async def update_entity(self, entity_id: str, data: dict) -> UpdateEntityResult:
    # Returns one of: UpdateEntitySuccess | NotFoundError | ValidationError | BusinessRuleViolationError

    entity = await self.repository.get(entity_id)
    if not entity:
        return NotFoundError(
            message="Entity not found",
            resource_type="Entity",
            resource_id=entity_id
        )

    # Validate business rules
    if not self.can_update(entity):
        return BusinessRuleViolationError(
            message="Cannot update entity in current state",
            rule_name="entity_state_transition"
        )

    # Update and return success
    updated = await self.repository.update(entity_id, data)
    return UpdateEntitySuccess(
        entity=updated.model_dump(),
        updated_fields=list(data.keys())
    )
```

## 🤝 Contributing

1. Follow Clean Architecture principles
2. Write tests for new features
3. Use type hints everywhere
4. Run `make validate` before committing
5. Follow conventional commit messages

### Development Guidelines

- **DRY**: Don't repeat yourself - use base classes and builders
- **KISS**: Keep it simple - prefer clarity over cleverness
- **YAGNI**: You ain't gonna need it - implement only what's needed
- **SOLID**: Follow SOLID principles for maintainable code

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

This project structure is inspired by:
- Clean Architecture by Robert C. Martin
- Domain-Driven Design by Eric Evans
- Modern Python best practices from the Python community

---

Built with ❤️ and unemployment benefits using modern Python tooling
