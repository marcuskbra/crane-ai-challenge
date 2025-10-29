"""
Unit Tests - Fast, isolated tests for individual components.

This package contains unit tests that test individual units of code in
isolation from external dependencies. Unit tests should be fast, focused,
and independent.

Characteristics of unit tests:
- No external dependencies (database, network, file system)
- Fast execution (milliseconds)
- Test single units (functions, methods, classes)
- Use mocks/stubs for dependencies
- Deterministic (same result every run)
- Can run in any order
- No shared state between tests

Test organization mirrors source structure:
- domain/: Tests for domain entities, value objects, services
- infrastructure/: Tests for infrastructure components (with mocked I/O)
- presentation/: Tests for presentation layer (with mocked services)

Best practices:
- One assertion per test (when practical)
- Descriptive test names that explain the scenario
- Use AAA pattern (Arrange, Act, Assert)
- Mock at the boundary (interfaces, not internals)
- Test behavior, not implementation
- Focus on edge cases and error conditions
- Use parameterized tests for similar scenarios

These tests should run on every commit and provide
fast feedback during development.
"""
