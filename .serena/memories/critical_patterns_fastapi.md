# Critical FastAPI Patterns & Best Practices

## Configuration Management Pattern

### Always Use Pydantic Settings
```python
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    app_name: str = Field(
        default="My API",
        description="Application name",
    )
    
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Deployment environment",
    )

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
```

## Dependency Injection Pattern

### Modern Annotated Types (Python 3.12+)
```python
# In api/dependencies.py
from typing import Annotated
from fastapi import Depends

SettingsDep = Annotated[Settings, Depends(get_settings)]

# In routes
@router.get("/endpoint")
async def my_endpoint(settings: SettingsDep):
    return {"name": settings.app_name}
```

## CORS Security Pattern

### CRITICAL: Never Use Wildcard with Credentials
```python
# ❌ SECURITY VULNERABILITY (violates CORS spec)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # INVALID with wildcard
)

# ✅ CORRECT (specific origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,  # Valid with specific origins
)
```

## Application Factory Pattern

### Testable App Creation
```python
def create_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI application factory."""
    if settings is None:
        settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.app_version,
        docs_url=settings.get_docs_url(),  # Environment-aware
    )
    
    return app

# For tests
test_settings = Settings(environment="test", debug=True)
app = create_app(settings=test_settings)
```

## Health Check Pattern

### Kubernetes-Compliant Endpoints
```python
@router.get("/health/ready")
async def readiness_check() -> ReadinessResponse:
    """Readiness probe for load balancers."""
    checks = {
        "application": True,
        "database": await check_database(),
    }
    
    ready = all(checks.values())
    
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "ready": False,
                "checks": checks,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    
    return ReadinessResponse(ready=ready, checks=checks)
```

## Environment-Aware Error Handling

### Production vs Development Responses
```python
@app.exception_handler(Exception)
async def handle_error(request: Request, exc: Exception) -> JSONResponse:
    content = {
        "error": "internal_error",
        "message": "An unexpected error occurred",
    }
    
    # Include details only in non-production
    if not settings.is_production():
        content["detail"] = str(exc)
        content["type"] = type(exc).__name__
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content,
    )
```

## Field Validation Pattern

### Parse Environment Variables
```python
@field_validator("cors_origins", mode="before")
@classmethod
def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
    """Parse comma-separated string or list."""
    if isinstance(v, str):
        origins = [origin.strip() for origin in v.split(",") if origin.strip()]
        return origins if origins else ["http://localhost:3000"]
    return v
```

## Test Fixture Pattern

### Settings-Based Test Client
```python
@pytest.fixture
def test_client():
    """FastAPI test client with test settings."""
    test_settings = Settings(
        environment="test",
        debug=True,
        cors_origins=["http://testserver"],
    )
    
    app = create_app(settings=test_settings)
    with TestClient(app) as client:
        yield client
```

## Common Pitfalls to Avoid

### 1. Hardcoded Configuration
❌ `app = FastAPI(title="My API", debug=True)`
✅ `app = FastAPI(title=settings.api_title, debug=settings.debug)`

### 2. Missing Environment Support
❌ Only supporting "development" and "production"
✅ Support "development", "staging", "production", "test"

### 3. Inconsistent Dependency Injection
❌ Creating instances in route handlers
✅ Using FastAPI Depends() with Annotated types

### 4. CORS Misconfiguration
❌ `allow_origins=["*"]` with `allow_credentials=True`
✅ Specific origins from configuration

### 5. Poor Error Context
❌ Same error details in all environments
✅ Environment-aware error responses

## Performance Optimizations

### 1. Cache Settings
```python
@lru_cache  # Load once per application
def get_settings() -> Settings:
    return Settings()
```

### 2. Lazy Loading
Only load heavy dependencies when needed via dependency injection.

### 3. Connection Pooling
Configure database/cache connection pools in Settings.

## Security Best Practices

### 1. Environment-Based Secrets
Use environment variables, never hardcode credentials.

### 2. CORS Configuration
Always use specific origins in production.

### 3. Rate Limiting
Consider implementing rate limiting for public endpoints.

### 4. Input Validation
Use Pydantic models for all request/response schemas.

## Documentation Standards

### Comprehensive Docstrings
```python
def create_app(settings: Settings | None = None) -> FastAPI:
    """
    FastAPI application factory.

    Creates and configures a FastAPI application instance with:
    - CORS middleware
    - Error handlers
    - API routing
    - OpenAPI documentation

    Args:
        settings: Application settings (defaults to get_settings())

    Returns:
        Configured FastAPI application instance

    Example:
        >>> settings = Settings(environment="production")
        >>> app = create_app(settings=settings)

    """
```

## Key Takeaways

1. **Always use Pydantic Settings** for configuration management
2. **Never mix wildcard CORS origins with credentials** - security violation
3. **Use Annotated types** for modern dependency injection
4. **Implement application factory pattern** for testability
5. **Return 503 for failed readiness checks** - Kubernetes requirement
6. **Cache settings with @lru_cache** - performance optimization
7. **Environment-aware error handling** - security best practice
8. **Support test environment** in Settings validation
