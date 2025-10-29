"""
Skeleton Challenge - A modern Python project with Clean Architecture.

This is the main package for the Skeleton Challenge project, implementing a
simplified 3-layer Clean Architecture with:

- Presentation Layer: User interfaces (API, CLI, Web)
- Domain Layer: Core business logic and rules
- Infrastructure Layer: External dependencies and integrations

The project emphasizes type safety, using discriminated unions for error
handling and Pydantic models for data validation throughout.

Key Features:
- Type-safe error handling with discriminated unions
- Clean Architecture with dependency inversion
- Async-first design for high concurrency
- Comprehensive testing with builders pattern
- Modern Python tooling (uv, ruff, ty)
"""

__version__ = "0.1.0"
__author__ = "Your Name"
