"""
Application configuration management.

This module provides centralized configuration management using Pydantic V2.
All settings are loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation.

    All settings can be overridden via environment variables.
    For production, use a .env file or system environment variables.

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
            "http://localhost:3001",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
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
    # LLM Configuration
    # ============================================================================

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (required for OpenAI, dummy value for local LLMs)",
    )

    openai_base_url: str | None = Field(
        default=None,
        description="Custom OpenAI API base URL for local LLMs (e.g., http://localhost:4000 for LiteLLM, http://localhost:11434/v1 for Ollama)",
    )

    openai_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name (e.g., 'gpt-4o-mini' for OpenAI, 'qwen2.5:3b' for local models)",
    )

    openai_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (lower = more deterministic)",
    )

    @model_validator(mode="after")
    def validate_llm_config(self) -> "Settings":
        """
        Validate LLM configuration and provide dummy API key when needed.

        The OpenAI client requires an API key even when using custom base_url.
        This validator runs after all fields are loaded and provides:
        - Dummy key "sk-local-llm-dummy-key" when base_url is set (local LLM)
        - Dummy key "sk-no-key-pattern-fallback" when neither is set (pattern-based fallback)

        Returns:
            Settings: Self with validated/updated API key

        Note:
            This ensures the application can start even without API key,
            relying on the pattern-based planner fallback.

        """
        # If API key is already provided, use it
        if self.openai_api_key:
            return self

        # If using local LLM (base_url is set), provide dummy key
        if self.openai_base_url:
            self.openai_api_key = "sk-local-llm-dummy-key"
            return self

        # No API key and no base_url - will use pattern-based planner fallback
        # Return dummy key to allow initialization
        self.openai_api_key = "sk-no-key-pattern-fallback"
        return self

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

    @field_validator("openai_base_url")
    @classmethod
    def validate_base_url(cls, v: str | None) -> str | None:
        """
        Validate OpenAI base URL format.

        Args:
            v: Base URL string or None

        Returns:
            Validated base URL or None

        Raises:
            ValueError: If URL format is invalid

        """
        if v is not None:
            # Basic URL format validation
            if not v.startswith(("http://", "https://")):
                raise ValueError(
                    f"OpenAI base URL must start with http:// or https://, got: {v}\n"
                    "Examples:\n"
                    "  - http://localhost:4000 (LiteLLM proxy)\n"
                    "  - http://localhost:11434/v1 (Ollama)\n"
                    "  - http://localhost:1234/v1 (LM Studio)"
                )
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

    def is_using_local_llm(self) -> bool:
        """
        Check if configured to use local LLM.

        Returns:
            True if base_url is configured (indicating local LLM), False otherwise

        """
        return self.openai_base_url is not None

    def get_llm_config_status(self) -> dict[str, str]:
        """
        Get human-readable LLM configuration status.

        Returns:
            Dictionary with LLM configuration details for logging/debugging

        """
        if self.is_using_local_llm():
            return {
                "provider": "Local LLM",
                "base_url": self.openai_base_url or "Not configured",
                "model": self.openai_model,
                "api_key_set": "Yes (dummy)" if self.openai_api_key else "No",
            }
        return {
            "provider": "OpenAI",
            "base_url": "Default (https://api.openai.com/v1)",
            "model": self.openai_model,
            "api_key_set": "Yes" if self.openai_api_key else "No (will fail)",
        }


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function uses LRU cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.

    Returns:
        Settings instance with all configuration values

    """
    return Settings()
