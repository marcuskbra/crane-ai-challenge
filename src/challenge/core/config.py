"""
Application configuration management.

This module provides centralized configuration management using Pydantic V2.
All settings are loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configuration constants
DEFAULT_LLM_TEMPERATURE = 0.1


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
    # LLM Configuration (LiteLLM Multi-Provider Support)
    # ============================================================================

    llm_provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, ollama, or custom via base_url)",
    )

    llm_api_key: str | None = Field(
        default=None,
        description="Primary LLM API key (provider-specific, dummy value for local LLMs)",
    )

    llm_base_url: str | None = Field(
        default=None,
        description="Custom API base URL for local LLMs (e.g., http://localhost:4000 for LiteLLM proxy, http://localhost:11434/v1 for Ollama)",
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet-20241022', 'qwen2.5:3b' for Ollama)",
    )

    llm_temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (lower = more deterministic, 0.0-2.0)",
    )

    # Backward compatibility: keep old openai_* names as aliases
    openai_api_key: str | None = Field(
        default=None,
        description="[DEPRECATED] Use llm_api_key instead. Kept for backward compatibility.",
    )

    openai_base_url: str | None = Field(
        default=None,
        description="[DEPRECATED] Use llm_base_url instead. Kept for backward compatibility.",
    )

    openai_model: str = Field(
        default="gpt-4o-mini",
        description="[DEPRECATED] Use llm_model instead. Kept for backward compatibility.",
    )

    openai_temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="[DEPRECATED] Use llm_temperature instead. Kept for backward compatibility.",
    )

    @model_validator(mode="after")
    def validate_llm_config(self) -> "Settings":
        """
        Validate LLM configuration with backward compatibility and multi-provider support.

        Handles migration from openai_* fields to generic llm_* fields while maintaining
        backward compatibility. Provides dummy API keys for local LLMs and fallback scenarios.

        Migration Strategy:
        - If llm_* fields are not set, copy from openai_* fields (backward compatibility)
        - If both are set, llm_* fields take precedence
        - Provide dummy keys for local LLMs and pattern-based fallback

        Returns:
            Settings: Self with validated/updated configuration

        Note:
            This ensures the application can start without API keys,
            relying on the pattern-based planner fallback when needed.

        """
        # Backward compatibility: migrate openai_* to llm_* if llm_* not explicitly set
        if self.llm_api_key is None and self.openai_api_key:
            self.llm_api_key = self.openai_api_key

        if self.llm_base_url is None and self.openai_base_url:
            self.llm_base_url = self.openai_base_url

        if self.llm_model == "gpt-4o-mini" and self.openai_model != "gpt-4o-mini":
            self.llm_model = self.openai_model

        if self.llm_temperature == DEFAULT_LLM_TEMPERATURE and self.openai_temperature != DEFAULT_LLM_TEMPERATURE:
            self.llm_temperature = self.openai_temperature

        # Provide dummy key for local LLMs (base_url is set)
        if self.llm_base_url and not self.llm_api_key:
            self.llm_api_key = "sk-local-llm-dummy-key"
            self.llm_provider = "ollama"  # Assume Ollama for local setups

        # Provide dummy key for pattern-based fallback (no API key or base_url)
        if not self.llm_api_key and not self.llm_base_url:
            self.llm_api_key = "sk-no-key-pattern-fallback"

        # Update openai_* fields for backward compatibility with existing code
        self.openai_api_key = self.llm_api_key
        self.openai_base_url = self.llm_base_url
        self.openai_model = self.llm_model
        self.openai_temperature = self.llm_temperature

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

    @field_validator("llm_base_url", "openai_base_url")
    @classmethod
    def validate_base_url(cls, v: str | None) -> str | None:
        """
        Validate LLM base URL format (applies to both llm_base_url and openai_base_url).

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
                    f"LLM base URL must start with http:// or https://, got: {v}\n"
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
        return self.llm_base_url is not None

    def get_llm_config_status(self) -> dict[str, str]:
        """
        Get human-readable LLM configuration status with multi-provider support.

        Returns:
            Dictionary with LLM configuration details for logging/debugging

        """
        if self.is_using_local_llm():
            return {
                "provider": f"Local LLM ({self.llm_provider})",
                "base_url": self.llm_base_url or "Not configured",
                "model": self.llm_model,
                "api_key_set": "Yes (dummy)" if self.llm_api_key else "No",
            }

        # Determine provider from model name
        provider_name = self.llm_provider.title()
        if "claude" in self.llm_model.lower():
            provider_name = "Anthropic"
        elif "gpt" in self.llm_model.lower():
            provider_name = "OpenAI"

        return {
            "provider": provider_name,
            "base_url": "Default (provider-specific)",
            "model": self.llm_model,
            "api_key_set": "Yes"
            if self.llm_api_key and not self.llm_api_key.startswith("sk-no-key")
            else "No (pattern fallback)",
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
