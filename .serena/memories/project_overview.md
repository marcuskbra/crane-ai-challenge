# Crane Challenge - Project Overview

## Project Identity
**Name:** crane-challenge (previously skeleton-challenge)
**Type:** AI Agent Runtime System - Technical Assessment for AI Engineer Position
**Time Allocation:** 2-4 hours (half-day POC)
**Architecture:** Simplified 3-Layer Clean Architecture with Python 3.12+

## Core Objectives
Build a minimal AI agent runtime that demonstrates:
1. Natural language task processing
2. Structured plan generation with available tools
3. Plan execution with state management, error handling, and retry logic
4. REST API for interaction and monitoring

## Technical Stack
- **Python:** 3.12+ for performance and modern features
- **Package Manager:** `uv` (10-100x faster than pip, Rust-based)
- **Framework:** FastAPI for REST API (async-first, high performance)
- **Data Validation:** Pydantic V2 with strict typing
- **Testing:** pytest with unit and integration tests
- **Linting:** ruff (Rust-based, very fast)
- **Type Checking:** ty (from Astral team, NOT pyright)
- **Project Layout:** src/ layout for better packaging

## Project Structure
```
crane-challenge/
├── src/challenge/           # Source code
│   ├── api/                # API Layer - HTTP interface
│   │   ├── routes/        # Route handlers
│   │   ├── schemas/       # Request/response models
│   │   ├── dependencies.py # Dependency injection
│   │   └── main.py        # FastAPI application
│   ├── services/           # Service Layer - Business logic (when needed)
│   ├── models/             # Data Layer - Models and repositories (when needed)
│   ├── core/               # Shared utilities, config, exceptions
│   │   ├── config.py      # Environment-based configuration
│   │   └── exceptions.py  # Standard Python exceptions
│   └── challenges/         # Challenge-specific implementations
├── tests/
│   ├── unit/              # Fast, isolated unit tests
│   ├── integration/       # Integration tests
│   ├── conftest.py        # Shared fixtures
│   └── builders.py        # Test data builders
├── pyproject.toml          # Project configuration
├── pytest.ini              # Test configuration
├── Makefile                # Development automation
└── .pre-commit-config.yaml # Pre-commit hooks
```

## Architecture Philosophy

### YAGNI Principles (You Aren't Gonna Need It)
- Start simple, add complexity only when needed
- Begin with API layer only for simple CRUD
- Add Service layer when business logic spans multiple resources
- Add Repository/Models layer when multiple data sources exist

### Current State: API Layer Only
- Simple CRUD operations
- Stateless transformations
- Direct database queries via ORM

### When to Add Layers
**Add Service Layer When:**
- Business logic spans multiple resources
- Complex orchestration needed
- Shared logic across endpoints
- Transaction management required

**Add Repository/Models When:**
- Multiple data sources (DB + cache + external API)
- Complex query logic needs abstraction
- Need to swap implementations (testing, migration)

## Key Design Principles
1. **KISS:** Keep It Simple, Stupid
2. **YAGNI:** You Aren't Gonna Need It
3. **SOLID:** All 5 principles enforced
4. **DRY:** Don't Repeat Yourself
5. **Type Safety:** Strong typing with Pydantic throughout
6. **Pythonic Patterns:** Standard exceptions, simple fixtures
7. **Testability:** Easy to test each layer in isolation

## Error Handling Pattern
- Use standard Python exceptions for business logic
- Use FastAPI HTTPException for HTTP responses
- Custom exceptions for specific business errors when needed
- NO hidden exceptions - all errors explicit in type system

Example:
```python
async def get_user(user_id: str) -> dict:
    if not user_id:
        raise ValueError("User ID is required")
    user = await database.get_user(user_id)
    if not user:
        raise ValueError(f"User not found: {user_id}")
    return user

@router.get("/users/{user_id}")
async def get_user_endpoint(user_id: str):
    try:
        user = await get_user(user_id)
        return user
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

## Development Commands (Quick Reference)
```bash
# Installation
make install           # Production dependencies
make dev-install      # All dependencies including dev tools

# Application
make run              # Run application
make api-dev          # Start API in development mode
make api-prod         # Start API in production mode (4 workers)

# Testing
make test             # Unit tests only (fast)
make test-all         # All tests
make test-integration # Integration tests only
make coverage         # Generate coverage report

# Code Quality
make lint             # Check code style
make format           # Format code
make type-check       # Run type checker (ty, not pyright)
make validate         # Run all checks

# Quick Commands
make fix              # Auto-fix lint and format issues
make quick            # Fast tests + quality checks

# API
make api-health       # Check API health
make api-docs         # Open API documentation (Swagger UI)
```

## Type Safety Requirements (CRITICAL)
1. **NO Dict[str, Any]** - Use Pydantic models for all structured data
2. **NO Mixed Return Types** - Use discriminated unions
3. **NO String-based Type Checking** - Use enums and proper type guards
4. **NO Runtime Type Discovery** - Types known at compile time
5. **Serialize Only at Boundaries** - Keep Pydantic models throughout

## Testing Strategy
- **Unit tests:** Fast, isolated (API routes and business logic)
- **Integration tests:** External integrations (when added)
- **Test Coverage:** ≥80% for business logic
- **Simple Fixtures:** Use pytest fixtures in conftest.py
- **Parameterized Tests:** Use @pytest.mark.parametrize
- **Error Scenarios:** Test both success and error paths

## Important Reminders
- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files over creating new ones
- NEVER proactively create documentation files
- Follow existing patterns in the codebase
- Use type hints for all function signatures
- Write tests for all new functionality
- Keep functions focused and single-purpose
