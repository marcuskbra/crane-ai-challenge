"""
Application configuration management.

This module provides centralized configuration management using Pydantic V2.
All settings are loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation.

    All settings can be overridden via environment variables.
    For production, use a .env file or system environment variables.

    Example:
        >>> settings = get_settings()
        >>> print(settings.app_name)
        'Crane Challenge API'

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env variables
    )

    # ============================================================================
    # Application Settings
    # ============================================================================

    app_name: str = Field(
        default="Crane Challenge API",
        description="Application name displayed in API documentation",
    )

    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )

    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Deployment environment",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode (development only)",
    )

    # ============================================================================
    # Server Configuration
    # ============================================================================

    host: str = Field(
        default="0.0.0.0",  # noqa: S104
        description="Server host address",
    )

    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port number",
    )

    reload: bool = Field(
        default=False,
        description="Enable auto-reload on code changes (development only)",
    )

    # ============================================================================
    # API Configuration
    # ============================================================================

    api_prefix: str = Field(
        default="/api/v1",
        description="API route prefix",
    )

    api_title: str = Field(
        default="Crane Challenge API",
        description="API title for OpenAPI documentation",
    )

    api_docs_enabled: bool = Field(
        default=True,
        description="Enable API documentation (Swagger UI and ReDoc)",
    )

    api_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="API request timeout in seconds",
    )

    # ============================================================================
    # Security Configuration
    # ============================================================================

    cors_enabled: bool = Field(
        default=True,
        description="Enable CORS middleware",
    )

    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ],
        description="Allowed CORS origins (comma-separated string or list)",
    )

    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests",
    )

    cors_allow_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        description="Allowed HTTP methods for CORS",
    )

    cors_allow_headers: list[str] = Field(
        default=["Content-Type", "Authorization", "X-Request-ID"],
        description="Allowed headers for CORS",
    )

    cors_max_age: int = Field(
        default=600,
        ge=0,
        description="Max age for CORS preflight cache in seconds",
    )

    # ============================================================================
    # Logging Configuration
    # ============================================================================

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # ============================================================================
    # Performance Configuration
    # ============================================================================

    gzip_minimum_size: int = Field(
        default=1000,
        ge=0,
        description="Minimum response size for GZip compression in bytes",
    )

    # ============================================================================
    # Validators
    # ============================================================================

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """
        Parse CORS origins from comma-separated string or list.

        Args:
            v: CORS origins as string or list

        Returns:
            List of CORS origins

        Example:
            >>> Settings.parse_cors_origins("http://a.com,http://b.com")
            ['http://a.com', 'http://b.com']

        """
        if isinstance(v, str):
            # Handle comma-separated string
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            return origins if origins else ["http://localhost:3000"]
        return v

    @field_validator("cors_allow_methods", mode="before")
    @classmethod
    def parse_cors_methods(cls, v: str | list[str]) -> list[str]:
        """
        Parse CORS methods from comma-separated string or list.

        Args:
            v: HTTP methods as string or list

        Returns:
            List of HTTP methods

        """
        if isinstance(v, str):
            methods = [method.strip().upper() for method in v.split(",") if method.strip()]
            return methods if methods else ["GET", "POST"]
        return v

    @field_validator("cors_allow_headers", mode="before")
    @classmethod
    def parse_cors_headers(cls, v: str | list[str]) -> list[str]:
        """
        Parse CORS headers from comma-separated string or list.

        Args:
            v: Headers as string or list

        Returns:
            List of allowed headers

        """
        if isinstance(v, str):
            headers = [header.strip() for header in v.split(",") if header.strip()]
            return headers if headers else ["Content-Type"]
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """
        Validate environment value.

        Args:
            v: Environment name

        Returns:
            Validated environment name

        Raises:
            ValueError: If environment is invalid

        """
        valid_environments = {"development", "staging", "production", "test"}
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def get_docs_url(self) -> str | None:
        """Get Swagger UI URL based on environment."""
        if self.api_docs_enabled and not self.is_production():
            return "/api/docs"
        return None

    def get_redoc_url(self) -> str | None:
        """Get ReDoc URL based on environment."""
        if self.api_docs_enabled and not self.is_production():
            return "/api/redoc"
        return None


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function uses LRU cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.

    Returns:
        Settings instance with all configuration values

    Example:
        >>> settings = get_settings()
        >>> print(settings.app_name)
        'Crane Challenge API'

    """
    return Settings()
