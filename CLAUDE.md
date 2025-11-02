# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**crane-challenge** - A modern Python project using Clean Architecture, Python 3.12+, managed with `uv` for fast dependency management.

## Architecture Overview

This project uses a **simplified 3-layer architecture** following YAGNI principles:

### Layer Structure
- **API Layer** (`api/`): HTTP interface and request/response handling
- **Service Layer** (`services/`): Business logic and orchestration (add when needed)
- **Data Layer** (`models/`): Data models and repository abstractions (add when needed)
- **Core** (`core/`): Shared utilities, config, and exceptions

### Design Principles
- **YAGNI (You Aren't Gonna Need It)**: Start simple, add complexity only when needed
- **Clear separation of concerns**: Each layer has specific, well-defined responsibilities
- **Pythonic patterns**: Standard exceptions, simple fixtures, FastAPI dependency injection
- **Gradual adoption**: Begin with API layer, add Services/Models as requirements grow
- **Testability**: Easy to test each layer in isolation with simple fixtures

### When to Add Each Layer

**Start with API Layer Only** (Current state):
- Simple CRUD operations
- Stateless transformations
- Direct database queries via ORM

**Add Service Layer When**:
- Business logic spans multiple resources
- Complex orchestration needed
- Shared logic across endpoints
- Transaction management required

**Add Repository/Models Layer When**:
- Multiple data sources (database + cache + external API)
- Complex query logic needs abstraction
- Need to swap implementations (testing, migration)

### Architecture Benefits
- **Simplicity**: No over-engineering, appropriate for project size
- **Maintainability**: Clear boundaries between concerns
- **Extensibility**: Can add layers incrementally as complexity grows
- **Type safety**: Strong typing with Pydantic models throughout
- **Fast feedback**: Simple structure means faster development cycles

### Layer Details

For a complete example with CRUD operations, dependency injection, and testing patterns,
see `claudedocs/simplified_architecture_example.md`

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
2. Define request/response schemas in `api/schemas/` (if needed)
3. Implement business logic inline or call service layer
4. Handle exceptions and convert to HTTP responses
5. Write tests in `tests/unit/api/routes/`

Example:
```python
# api/routes/users.py
from fastapi import APIRouter, HTTPException, status

router = APIRouter()

@router.get("/users/{user_id}")
async def get_user(user_id: str):
    try:
        user = await get_user_from_db(user_id)
        return user
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

### Adding Business Logic (Service Layer)
When logic grows beyond simple CRUD:

1. Create service class in `src/challenge/services/`
2. Use dependency injection via `api/dependencies.py`
3. Raise standard exceptions for errors
4. Write tests in `tests/unit/services/`

Example:
```python
# services/user_service.py
class UserService:
    def __init__(self, db: Database):
        self.db = db

    async def create_user(self, data: dict) -> dict:
        if await self.db.user_exists(data["email"]):
            raise ValueError("User already exists")
        return await self.db.create_user(data)

# api/dependencies.py
def get_user_service() -> UserService:
    return UserService(database=get_database())

# api/routes/users.py
@router.post("/users")
async def create_user(
    data: UserCreate,
    service: UserService = Depends(get_user_service)
):
    try:
        user = await service.create_user(data.model_dump())
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Adding Data Models
When you need structured data types:

1. Create Pydantic models in `src/challenge/models/`
2. Use for request/response schemas
3. Include validation rules
4. Write validation tests

Example:
```python
# models/user.py
from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    id: str = Field(..., description="User ID")
    email: EmailStr = Field(..., description="Email address")
    name: str = Field(..., min_length=1, max_length=100)

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
