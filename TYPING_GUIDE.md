# Type Safety and Strong Typing Guide

This guide establishes the typing standards and best practices for this project. All code contributions must follow these guidelines to ensure type safety, maintainability, and developer experience.

> **Note**: This project uses a simplified 3-layer Clean Architecture where the Domain layer includes services and interfaces. This reduces boilerplate while maintaining strong typing throughout.

## 🎯 Core Principles

1. **No Dict[str, Any]** - Use Pydantic models for all structured data
2. **No Mixed Return Types** - Use discriminated unions for multiple return scenarios
3. **No String-based Type Checking** - Use enums and proper type guards
4. **No Runtime Type Discovery** - Types should be known at compile time
5. **Serialize Only at Boundaries** - Keep Pydantic models throughout internal code

## ✅ Required Practices

### 1. Pydantic Models for All Data Structures

```python
# ❌ NEVER DO THIS
def process_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    status = entity.get("status")
    if status == "active":
        entity["priority"] = 1
    return entity

# ✅ ALWAYS DO THIS
from pydantic import BaseModel
from challenge.domain.entities import Entity, EntityStatus, EntityPriority

def process_entity(entity: Entity) -> Entity:
    if entity.status == EntityStatus.ACTIVE:
        entity.priority = EntityPriority.HIGH
    return entity
```

### 2. Discriminated Unions for Multiple Return Types

```python
# ❌ NEVER DO THIS
async def search(query: str) -> Union[Dict[str, Any], SearchResponse]:
    if error:
        return {"error": "message"}
    return SearchResponse(...)

# ✅ ALWAYS DO THIS
from typing import Literal, Union
from pydantic import BaseModel

class SearchSuccess(BaseModel):
    success: Literal[True] = True
    data: SearchResponse

class SearchError(BaseModel):
    success: Literal[False] = False
    error_code: str
    message: str

SearchResult = Union[SearchSuccess, SearchError]

async def search(query: str) -> SearchResult:
    if error:
        return SearchError(error_code="INVALID_QUERY", message="...")
    return SearchSuccess(data=SearchResponse(...))
```

**IMPORTANT: Pydantic Model Building Requirements**

When using discriminated unions with Pydantic models that reference other models:
- Use actual imports, NOT conditional `TYPE_CHECKING` imports
- Pydantic needs the actual classes at runtime to build models
- This is especially critical in the domain layer

```python
# ❌ WRONG - Will cause runtime errors
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .entities import Entity

class EntitySuccess(BaseModel):
    entities: list["Entity"]  # Will fail at runtime!

# ✅ CORRECT - Actual imports for Pydantic
from .entities import Entity  # Direct import

class EntitySuccess(BaseModel):
    entities: list[Entity]  # Works at runtime
```

### 3. Enums for Constrained Values

```python
# ❌ NEVER DO THIS
if status == "active":
    process()
if priority in ["high", "critical"]:
    escalate()

# ✅ ALWAYS DO THIS
from enum import Enum

class EntityStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

if status == EntityStatus.ACTIVE:
    process()
if priority in [Priority.HIGH, Priority.CRITICAL]:
    escalate()
```

### 4. Type-Safe Error Handling

```python
# ❌ NEVER DO THIS
if isinstance(result, dict) and "error" in result:
    handle_error(result["error"])

# ✅ ALWAYS DO THIS
from typing import TypeGuard

def is_error_response(response: Response) -> TypeGuard[ErrorResponse]:
    return not response.success

if is_error_response(result):
    handle_error(result.error_code, result.message)  # Type-safe access
```

### 5. Proper Model Configuration

```python
# ✅ REQUIRED MODEL CONFIGURATION
from pydantic import BaseModel, ConfigDict, Field

class StronglyTypedModel(BaseModel):
    # Use descriptive field definitions
    entity_id: str = Field(..., description="Unique entity identifier")
    status: EntityStatus = Field(..., description="Current entity status")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # CRITICAL: Never use use_enum_values=True
    model_config = ConfigDict(
        validate_assignment=True,      # Validate on attribute assignment
        use_enum_values=False,         # Keep enums as enums, not strings
        strict=True,                   # Strict type checking
        str_strip_whitespace=True,     # Auto-strip string fields
        extra="forbid",               # Prevent unexpected fields
        json_schema_extra={
            "example": {
                "entity_id": "ENT-123",
                "status": "active"
            }
        }
    )
```

### 6. Serialization Only at API Boundaries

```python
# ✅ CORRECT SERIALIZATION PATTERN
async def api_endpoint(request: Request) -> Response:
    # Step 1: Parse input into Pydantic model
    data = EntityRequest.model_validate(request.json())

    # Step 2: Process with typed models (no dicts!)
    result = await entity_service.process(data)  # Returns Entity model

    # Step 3: Serialize ONLY at the response boundary
    return JSONResponse(content=result.model_dump())

# Internal services work with models, not dicts
class EntityService:
    async def process(self, entity: Entity) -> Entity:
        # Work with typed models throughout
        return entity
```

## 📋 Type Patterns by Layer (3-Layer Architecture)

### Presentation Layer (API/CLI)

```python
# FastAPI endpoint with typed models
from fastapi import APIRouter, HTTPException
from challenge.domain.services import EntityService
from challenge.domain.errors import GetEntityResult, NotFoundError

router = APIRouter()

@router.get("/entities/{entity_id}")
async def get_entity(
    entity_id: str,
    service: EntityService = Depends(get_entity_service)
) -> EntityResponse:  # Typed response model
    # Call domain service
    result = await service.get_entity(entity_id)

    # Handle discriminated union result
    if isinstance(result, NotFoundError):
        raise HTTPException(status_code=404, detail=result.message)

    # Type checker knows result is GetEntitySuccess here
    return EntityResponse.from_domain(result.entity)
```

### Domain Layer

#### Domain Services
```python
# domain/services/entity_service.py
from challenge.domain.errors import (
    GetEntityResult,
    GetEntitySuccess,
    NotFoundError,
    ValidationError
)

class EntityService:
    """Domain service with business logic orchestration."""

    def __init__(self, repository: EntityRepository):
        self.repository = repository

    async def get_entity(self, entity_id: str) -> GetEntityResult:
        """Get entity by ID - returns discriminated union."""
        # Validate input
        if not entity_id:
            return ValidationError(
                message="Entity ID is required",
                field_errors={"entity_id": "Cannot be empty"}
            )

        # Business logic
        entity = await self.repository.get(entity_id)
        if not entity:
            return NotFoundError(
                message="Entity not found",
                resource_type="Entity",
                resource_id=entity_id
            )

        return GetEntitySuccess(entity=entity)
```

#### Domain Interfaces
```python
# domain/interfaces/repository.py
from abc import ABC, abstractmethod
from typing import Optional

class EntityRepository(ABC):
    """Domain interface - no infrastructure dependencies."""

    @abstractmethod
    async def get(self, entity_id: str) -> Optional[Entity]:
        """Retrieve entity by ID."""
        pass

    @abstractmethod
    async def save(self, entity: Entity) -> Entity:
        """Persist entity."""
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity."""
        pass
```

### Infrastructure Layer

```python
# infrastructure/repositories/postgres_entity_repository.py
from sqlalchemy.ext.asyncio import AsyncSession
from challenge.domain.interfaces import EntityRepository
from challenge.domain.entities import Entity

class PostgresEntityRepository(EntityRepository):
    """PostgreSQL implementation of EntityRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, entity_id: str) -> Optional[Entity]:
        """Retrieve entity from PostgreSQL."""
        result = await self.session.execute(
            select(EntityModel).where(EntityModel.id == entity_id)
        )
        db_entity = result.scalar_one_or_none()

        if db_entity:
            # Convert DB model to domain model
            return Entity.model_validate(db_entity.to_dict())
        return None

    async def save(self, entity: Entity) -> Entity:
        """Save entity to PostgreSQL."""
        db_entity = EntityModel.from_domain(entity)
        self.session.add(db_entity)
        await self.session.commit()
        return entity
```

## 🧪 Testing with Types

### Type-Safe Test Fixtures

```python
import pytest
from challenge.domain.entities import Entity, EntityStatus

@pytest.fixture
def test_entity() -> Entity:
    """Returns typed model, not dict."""
    return Entity(
        id="test-123",
        name="Test Entity",
        status=EntityStatus.ACTIVE,
        created_at=datetime.utcnow()
    )

@pytest.fixture
def inactive_entity(test_entity: Entity) -> Entity:
    """Modify existing fixture."""
    test_entity.status = EntityStatus.INACTIVE
    return test_entity
```

### Type-Safe Assertions

```python
# ❌ NEVER DO THIS
assert "error" in result
assert result["error"] == "not_found"

# ✅ ALWAYS DO THIS
from challenge.domain.errors import NotFoundError

assert isinstance(result, NotFoundError)
assert result.resource_type == "Entity"
assert result.resource_id == entity_id
```

### Using Test Builders

```python
from tests.builders import entity, request

def test_entity_processing():
    # Use builder pattern for test data
    test_entity = (
        entity()
        .with_name("Test")
        .with_status(EntityStatus.PENDING)
        .build()
    )

    result = process_entity(test_entity)

    assert isinstance(result, Entity)
    assert result.status == EntityStatus.ACTIVE
```

## 🚫 Anti-Patterns to Avoid

### 1. Never Use Dict[str, Any]

```python
# ❌ BANNED PATTERNS
def process(data: Dict[str, Any]) -> Dict[str, Any]
entities: List[Dict[str, Any]]
Optional[Dict[str, Any]]

# ✅ USE INSTEAD
def process(data: ProcessRequest) -> ProcessResponse
entities: list[Entity]
Optional[EntityData]
```

### 2. Never Use hasattr/getattr for Type Checking

```python
# ❌ BANNED
if hasattr(obj, 'error'):
    error = getattr(obj, 'error')

# ✅ USE INSTEAD
if isinstance(obj, ErrorResponse):
    error = obj.error_code
```

### 3. Never Use isinstance with Primitives

```python
# ❌ BANNED
if isinstance(value, dict):
    if "error" in value:
        ...

# ✅ USE INSTEAD
if isinstance(value, ErrorResponse):
    handle_error(value)
```

### 4. Never Compare Enums to Strings

```python
# ❌ BANNED
if status == "active":
    ...

# ✅ USE INSTEAD
if status == EntityStatus.ACTIVE:
    ...
```

### 5. Never Mix Return Types

```python
# ❌ BANNED
def search() -> Union[list[Item], dict[str, str]]:
    if error:
        return {"error": "message"}
    return items

# ✅ USE INSTEAD
def search() -> SearchResult:
    if error:
        return SearchError(message="...")
    return SearchSuccess(items=items)
```

### 6. Settings Access Best Practices

```python
# ❌ NEVER USE - No type safety
api_key = settings.get("api_key", "")  # BaseSettings doesn't have .get()
api_key = getattr(settings, "api_key", "")  # Runtime string lookups

# ✅ ALWAYS USE - Type-safe direct access
from challenge.infrastructure.config import Settings

class ExternalAPIClient:
    def __init__(self, settings: Settings):
        # Direct attribute access with type safety
        self.api_key = settings.external_api_key or ""
        self.base_url = settings.external_api_url

        # Type checker knows these are Optional[str]
        if settings.external_api_enabled:
            self.configured = True
```

**Key Points:**
- Always add settings as typed fields in the Pydantic settings class
- Use direct attribute access for type safety
- Use `or ""` pattern for Optional[str] fields that need defaults
- Never use string-based attribute access (getattr, dict-style access)

## 🔧 Tooling Configuration

### Type Checking with `ty`

This project uses `ty` (from Astral team) for type checking. Run type checks using:

```bash
# Using make
make type-check

# Using tox
tox -e type

# Direct command
uv run ty check src/ tests/
```

### Pre-commit Hooks

Type checking is already configured in `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: type-check
      name: Type checking with ty
      entry: uv run ty check
      language: system
      types: [python]
      pass_filenames: false
      require_serial: true
      always_run: true
```

### VS Code / PyCharm Configuration

For VS Code, add to `.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.ruffEnabled": true,
    "python.analysis.typeCheckingMode": "strict"
}
```

For PyCharm:
- Enable type checking in Settings → Editor → Inspections → Python
- Set inspection profile to "Type checker"

## 📖 References

- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [PEP 604 - Union Types with |](https://www.python.org/dev/peps/pep-0604/)
- [ty Documentation](https://github.com/rustedpy/ty) - The type checker used in this project

---

**Remember: Strong typing is not optional - it's a requirement for code quality, maintainability, and developer experience.**
