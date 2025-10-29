"""
Integration Tests - Tests with external dependencies and integrations.

This package contains integration tests that verify the interaction between
different components of the system and with external dependencies. These tests
run against real or simulated external systems.

Characteristics of integration tests:
- Test integration between components
- May use real external dependencies (database, APIs)
- Slower than unit tests (seconds to minutes)
- Test data flow across boundaries
- Verify configuration and wiring
- Test error handling across layers
- Validate contracts between systems

Test organization by integration type:
- database/: Tests with real database connections
- api/: Tests with external API integrations
- messaging/: Tests with message queue systems
- cache/: Tests with caching layer
- filesystem/: Tests with file system operations

Integration test patterns:
- Use test containers for databases
- Mock external APIs when necessary
- Use test fixtures for data setup
- Clean up after tests (transactions, rollback)
- Test both success and failure scenarios
- Verify data consistency across systems
- Test timeouts and retry logic

These tests should run in CI/CD pipelines
and before deployments to ensure system integrity.
"""
