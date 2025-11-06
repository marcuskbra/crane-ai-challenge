# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**crane-challenge** - A modern Python project using Clean Architecture, Python 3.12+, managed with `uv` for fast dependency management.

## Architecture Overview

This project uses **Clean Architecture** with clear separation of concerns and dependency inversion:

### Layer Structure

```
src/challenge/
├── domain/              # Domain Layer (Core Business Logic)
│   ├── models/         # Domain entities (Plan, Run, etc.)
│   ├── services/       # Domain services (complex business logic)
│   └── types.py        # Value objects and type definitions
│
├── services/           # Application Service Layer
│   ├── planning/       # Planning service
│   └── orchestration/  # Orchestration service
│
├── infrastructure/     # Infrastructure Layer
│   └── tools/         # Tool implementations
│       ├── base.py         # Tool protocol
│       ├── registry.py     # Tool registry
│       ├── type_guards.py  # Type utilities
│       └── implementations/
│           ├── calculator.py
│           └── todo_store.py
│
├── api/                # API Layer (HTTP Interface)
│   ├── routes/        # Route handlers
│   ├── schemas/       # API request/response schemas
│   └── dependencies.py # Dependency injection
│
└── core/              # Cross-cutting Concerns
    ├── config.py      # Configuration
    └── exceptions.py  # Custom exceptions
```

### Design Principles
- **Clean Architecture**: Dependencies flow inward (API → Services → Domain)
- **Domain-Centric**: Core business logic independent of external concerns
- **Dependency Inversion**: Infrastructure depends on domain abstractions
- **Clear separation of concerns**: Each layer has specific, well-defined responsibilities
- **Type safety**: Strong typing with Pydantic models throughout

### Layer Responsibilities

**Domain Layer** (`domain/`):
- Core business entities (Plan, Run, ExecutionStep)
- Domain types and value objects (ToolInput, ToolOutput)
- Domain services (when business logic spans multiple entities)
- **Independent** of infrastructure and external concerns

**Application Services** (`services/`):
- Planning service: Converts prompts to execution plans
- Orchestration service: Executes plans and manages runs
- Coordinates domain logic and infrastructure

**Infrastructure** (`infrastructure/`):
- Tool implementations (calculator, todo_store)
- External integrations
- Depends on domain abstractions (protocols, types)

**API Layer** (`api/`):
- HTTP interface with FastAPI
- Request/response schemas (separate from domain models)
- Route handlers delegate to services

**Core** (`core/`):
- Shared utilities, configuration, exceptions
- Used across all layers

### Dependency Flow
- API → Services → Domain ← Infrastructure
- Domain layer has NO dependencies on other layers
- Infrastructure implements domain protocols

### Architecture Benefits
- **Maintainability**: Clear boundaries and responsibilities
- **Testability**: Each layer testable in isolation
- **Scalability**: Easy to add complexity in appropriate layers
- **Type safety**: Strong typing throughout with Pydantic
- **Flexibility**: Can swap infrastructure implementations

## Build & Test Commands

- Install dependencies: `uv sync --no-dev` (or use `make install`)
- Install dev dependencies: `uv sync --all-extras` (or use `make dev-install`)
- Run application: `uv run python -m challenge` (or use `make run`)
- Run unit tests only (default): `pytest tests/unit/` (or use `make test`)
- Run all tests: `pytest tests/` (or use `make test-all`)
- Run integration tests: `pytest tests/integration/` (or use `make test-integration`)
- Run single test: `pytest tests/path/to/test_file.py::test_function_name -v`
- Run tests with coverage: `pytest --cov=src --cov-report=html` (or use `make coverage`)
- Run linter: `ruff check src/ tests/` (or use `make lint`)
- Format code: `ruff format src/ tests/` (or use `make format`)
- Type check: `ty check src/ tests/` (not pyright)
- Run all validation: `make validate`

## Technical Stack

- **Python version**: Python 3.12+ (for performance and modern features)
- **Async support**: Full async/await support for high concurrency
- **Data validation**: Pydantic for data validation and serialization
- **Testing**: `pytest` for unit and integration tests with fixtures
- **Package management**: `uv` for fast, reliable package management (10-100x faster than pip)
- **Project config**: `pyproject.toml` for configuration and dependency management
- **Environment**: Use virtual environment in `.venv` for dependency isolation
- **Dependencies**: Separate production and dev dependencies in `pyproject.toml`
- **Linting**: `ruff` for style and error checking (Rust-based, very fast)
- **Type checking**: Strong typing throughout. Use `ty` from Astral team
- **Project layout**: Organize code with `src/` layout for better packaging

## Code Style Guidelines

- **Formatting**: Black-compatible formatting via `ruff format` with 120 char line length
- **Imports**: Sort imports with `ruff` (stdlib, third-party, local)
- **f-strings**: Prefer f-strings for string interpolation over `.format()` or `%`
- **Type hints**: Use native Python type hints (e.g., `list[str]` not `List[str]`)
- **Documentation**: Google-style docstrings for all modules, classes, functions
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Function length**: Keep functions short (< 30 lines) and single-purpose
- **PEP 8**: Follow PEP 8 style guide (enforced via `ruff`)

## Python Best Practices

- **File handling**: Prefer `pathlib.Path` over `os.path`
- **Debugging**: Use `logging` module instead of `print`
- **Error handling**: Use standard Python exceptions with FastAPI HTTPException for HTTP errors
- **Function arguments**: Avoid mutable default arguments (use `None` and check)
- **Data containers**: Leverage `Pydantic` models for data validation and serialization
- **Configuration**: Use environment variables for configuration (python-dotenv)
- **Security**: Never store/log credentials, validate inputs, set appropriate timeouts
- **Context managers**: Use `with` statements for resource management

## Development Patterns & Best Practices

- **Favor simplicity**: Choose the simplest solution that meets requirements
- **KISS principle**: Keep It Simple, Stupid - avoid unnecessary complexity
- **YAGNI principle**: You Aren't Gonna Need It - avoid over-engineering
- **SOLID principles**:
  - Single Responsibility
  - Open/Closed
  - Liskov Substitution
  - Interface Segregation
  - Dependency Inversion
- **Layered architecture**: Separate code into api, services (when needed), models (when needed), core
- **Business logic organization**: Keep business logic in service layer when it grows beyond simple CRUD
- **DRY principle**: Avoid code duplication; reuse existing functionality
- **Configuration management**: Use environment variables for different environments
- **Focused changes**: Only implement explicitly requested changes
- **Preserve patterns**: Follow existing code patterns when making changes
- **File size**: Keep files under 300 lines; refactor when exceeding this limit
- **Test coverage**: Write comprehensive unit and integration tests (≥80% coverage)
- **Modular design**: Create reusable, modular components
- **Logging**: Implement appropriate logging levels (debug, info, warning, error)
- **Error handling**: Use standard Python exceptions with proper HTTP status codes
- **Security best practices**: Input validation, output encoding, secure defaults
- **Performance**: Profile before optimizing, optimize critical paths only
- **Dependency management**: Add libraries only when essential

## Testing Guidelines

### Testing Strategy

- **Test Types**:
  - Unit tests for API routes and business logic (fast, isolated)
  - Integration tests for external integrations (when added)
  - Performance tests for critical operations
- **Simple Fixtures**: Use pytest fixtures in `conftest.py` for test data
- **Parameterized Tests**: Use `@pytest.mark.parametrize` for test variations
- **Layer Testing**: Test each layer independently with appropriate mocks
- **Error Scenarios**: Test both success and error paths with proper exception handling

### Test Organization

```
tests/
├── unit/               # Fast, isolated unit tests
│   ├── api/           # API layer tests
│   │   └── routes/    # Route handler tests
│   ├── services/      # Service layer tests (when added)
│   └── models/        # Model tests (when added)
├── integration/       # Tests with real external dependencies
└── conftest.py        # Shared fixtures and test configuration
```

### Testing Patterns

1. **Simple Fixtures for Test Data**:
   ```python
   # In conftest.py
   @pytest.fixture
   def sample_user_data():
       return {
           "id": "user-123",
           "name": "Test User",
           "email": "test@example.com"
       }

   # In test file
   def test_create_user(test_client, sample_user_data):
       response = test_client.post("/api/v1/users", json=sample_user_data)
       assert response.status_code == 201
   ```

2. **Parameterized Testing**:
   ```python
   @pytest.mark.parametrize("input_value,expected", [
       ("valid@email.com", True),
       ("invalid-email", False),
   ])
   def test_email_validation(input_value, expected):
       assert is_valid_email(input_value) == expected
   ```

3. **Testing Async Code**:
   ```python
   @pytest.mark.asyncio
   async def test_async_operation():
       result = await async_function()
       assert result is not None
   ```

4. **Testing Error Cases**:
   ```python
   def test_not_found_error(test_client):
       response = test_client.get("/api/v1/users/nonexistent")
       assert response.status_code == 404
       assert "not found" in response.json()["detail"].lower()
   ```

### Test Coverage Requirements

- **Unit Tests**: ≥ 80% coverage for all business logic
- **Integration Tests**: Cover all external integration points (when added)
- **Error Paths**: Test all error scenarios with proper status codes
- **Edge Cases**: Include boundary conditions and edge cases

## Development Workflow

- **Version control**: Commit frequently with clear, conventional commit messages
- **Test-Driven Development**: Write tests before implementation
- **TDD Cycle**:
  1. **Red**: Create failing tests that define requirements
  2. **Green**: Implement minimal code to pass tests
  3. **Refactor**: Optimize while maintaining test compliance
- **Continuous Testing**: Run tests before every commit
- **Code Review**: Self-review changes before committing
- **Impact assessment**: Evaluate how changes affect other parts of the codebase
- **Documentation**: Document complex logic and public APIs
- **Branch strategy**: Use feature branches for development

## Error Handling Pattern

**IMPORTANT**: This project uses standard Python exceptions with FastAPI's HTTPException for HTTP errors.

### Pattern Overview
Use standard Python exceptions for business logic errors, and FastAPI's HTTPException for HTTP responses:

```python
from fastapi import HTTPException, status
from pydantic import ValidationError

# Service method with standard exceptions
async def get_user(user_id: str) -> dict:
    # Input validation
    if not user_id:
        raise ValueError("User ID is required")

    # Business logic
    user = await database.get_user(user_id)
    if not user:
        raise ValueError(f"User not found: {user_id}")

    return user

# API endpoint handles exceptions
@router.get("/users/{user_id}")
async def get_user_endpoint(user_id: str):
    try:
        user = await get_user(user_id)
        return user
    except ValueError as e:
        # Convert to appropriate HTTP error
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

### Custom Exceptions (When Needed)
Create custom exceptions for specific business errors:

```python
# In core/exceptions.py
class UserNotFoundError(Exception):
    """Raised when user doesn't exist."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f"User not found: {user_id}")

class InsufficientBalanceError(Exception):
    """Raised when account balance is insufficient."""
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(f"Insufficient balance: need {required}, have {available}")

# Usage in service
async def transfer_money(from_id: str, to_id: str, amount: float):
    from_account = await get_account(from_id)
    if from_account.balance < amount:
        raise InsufficientBalanceError(amount, from_account.balance)
    # ... rest of transfer logic

# API endpoint maps to HTTP status
@router.post("/transfer")
async def transfer_endpoint(request: TransferRequest):
    try:
        result = await transfer_money(request.from_id, request.to_id, request.amount)
        return result
    except InsufficientBalanceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

### Benefits of Standard Exceptions
- **Pythonic**: Follows standard Python patterns and idioms
- **Simple**: No additional complexity or type machinery
- **FastAPI Integration**: Natural integration with FastAPI exception handling
- **Familiar**: Standard pattern understood by all Python developers

### Testing with Standard Exceptions
Test both success and error paths:

```python
import pytest

# Test success case
async def test_get_user_success():
    user = await get_user("valid-id")
    assert user["id"] == "valid-id"

# Test error case
async def test_get_user_not_found():
    with pytest.raises(ValueError, match="User not found"):
        await get_user("invalid-id")

# Test HTTP endpoint error handling
def test_get_user_endpoint_not_found(test_client):
    response = test_client.get("/api/v1/users/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
```

## Common Patterns

### Adding a New API Endpoint
1. Create route handler in `src/challenge/api/routes/`
2. Define request/response schemas in `api/schemas/` (ALWAYS separate from routes)
3. Import domain models from `domain/models/`
4. Delegate to services for business logic
5. Handle exceptions and convert to HTTP responses
6. Write tests in `tests/unit/api/routes/`

Example:
```python
# api/schemas/users.py
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    name: str

# api/routes/users.py
from fastapi import APIRouter, HTTPException, status
from challenge.api.schemas.users import UserCreate
from challenge.domain.models.user import User

router = APIRouter()

@router.post("/users", response_model=User)
async def create_user(data: UserCreate):
    try:
        user = await create_user_service(data)
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Adding Domain Models
Core business entities go in the domain layer:

1. Create Pydantic models in `src/challenge/domain/models/`
2. Models should be independent of infrastructure
3. Import domain types from `domain/types.py`
4. Write validation tests in `tests/unit/domain/`

Example:
```python
# domain/models/user.py
from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    """User domain entity."""
    id: str = Field(..., description="User ID")
    email: EmailStr = Field(..., description="Email address")
    name: str = Field(..., min_length=1, max_length=100)
```

### Adding Application Services
Business logic and orchestration:

1. Create service in `src/challenge/services/` (planning, orchestration, etc.)
2. Import domain models from `domain/models/`
3. Use dependency injection via `api/dependencies.py`
4. Raise standard exceptions for errors
5. Write tests in `tests/unit/services/`

Example:
```python
# services/user_service.py
from challenge.domain.models.user import User

class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    async def create_user(self, email: str, name: str) -> User:
        if await self.repository.exists(email):
            raise ValueError("User already exists")
        return await self.repository.create(User(email=email, name=name))

# api/dependencies.py
from challenge.services.user_service import UserService

def get_user_service() -> UserService:
    return UserService(repository=get_user_repository())
```

### Adding Infrastructure (Tools, External Services)
Implementation of external concerns:

1. Create implementation in `src/challenge/infrastructure/`
2. Implement domain protocols/interfaces
3. Import domain types, never the reverse
4. Write tests in `tests/unit/infrastructure/`

Example:
```python
# infrastructure/tools/implementations/new_tool.py
from challenge.infrastructure.tools.base import BaseTool, ToolResult
from challenge.domain.types import ToolInput

class NewTool(BaseTool):
    """New tool implementation."""

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        # Implementation
        pass
```

## Important Reminders

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files unless explicitly requested
- Follow existing patterns in the codebase
- Use type hints for all function signatures
- Write tests for all new functionality
- Keep functions focused and single-purpose
- Use meaningful variable and function names
