"""
Test Suite - Comprehensive testing for all layers.

This package contains all tests for the application, organized by test type
and layer. Tests follow the Arrange-Act-Assert pattern and use pytest as
the testing framework.

Test organization:
- unit/: Fast, isolated unit tests
- integration/: Tests with external dependencies
- e2e/: End-to-end tests (if applicable)
- fixtures/: Shared test fixtures and utilities
- builders.py: Test data builders using Builder pattern

Testing principles:
- Tests are documentation - they show how code should be used
- Fast tests run frequently, slow tests run in CI
- Each test tests one thing
- Test names describe what is being tested
- Use builders for complex test data
- Mock external dependencies in unit tests
- Use real implementations in integration tests

Test categories:
- Unit tests: Test single units in isolation
- Integration tests: Test integration between components
- Contract tests: Verify interface implementations
- Performance tests: Measure performance characteristics
- Security tests: Verify security measures

All tests should be:
- Independent (can run in any order)
- Repeatable (same result every time)
- Self-validating (clear pass/fail)
- Timely (written with or before code)
"""
