# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Timeout protection for step execution with configurable timeout (default: 30s)
- `/api/v1/metrics` endpoint for system observability
  - Run statistics (total, by status, success rate)
  - Execution metrics (average duration, total steps)
  - Tool usage statistics (executions by tool)
- Environment configuration file (`.env.example`) with comprehensive documentation
- Timeout configuration support via `STEP_TIMEOUT` environment variable
- Comprehensive metrics tests with 100% coverage

### Changed
- Updated orchestrator to wrap step execution in `asyncio.wait_for()` for timeout protection
- Enhanced README with configuration section documenting timeout and environment variables
- Improved architecture documentation to highlight timeout protection
- Updated "Retry Strategy" section to include timeout protection details

### Fixed
- Prevented indefinite hangs in step execution with timeout mechanism
- Added clear timeout error messages for debugging

## [1.0.0] - 2025-01-29

### Added
- Initial release of Crane AI Agent Runtime
- REST API with FastAPI framework
- Hybrid planning system (LLM + pattern-based fallback)
  - LLMPlanner using GPT-4o-mini with structured outputs
  - PatternBasedPlanner for deterministic planning
  - Automatic fallback on LLM failures
- Orchestration layer with retry logic
  - Sequential step execution
  - Exponential backoff (1s → 2s → 4s)
  - Complete execution history
- Tool system with extensible architecture
  - Calculator tool with AST-based parsing (security-first)
  - TodoStore tool for task management
  - Tool registry for dynamic tool management
- Comprehensive test suite
  - 103 tests with 84% coverage
  - Unit tests for all components
  - Integration tests for E2E workflows
  - Security injection tests
- Production-quality error handling
  - Standard Python exceptions with FastAPI HTTPException
  - Clear error messages and HTTP status codes
  - Graceful degradation
- API endpoints
  - `POST /api/v1/runs` - Create and execute runs
  - `GET /api/v1/runs/{run_id}` - Get run status and results
  - `GET /api/v1/health` - Health check with system info
  - `GET /api/v1/health/live` - Liveness probe
  - `GET /api/v1/health/ready` - Readiness probe
- Documentation
  - Comprehensive README with architecture diagrams
  - API usage examples with request/response samples
  - Design decision documentation with trade-offs
  - Known limitations and improvement roadmap

### Security
- AST-based expression parser preventing code injection
- 5 security injection tests validating attack vector protection
- No use of `eval()` or `exec()` for user input
- Whitelisted operators only for calculator

---

## Version History

### [Unreleased]
**Focus**: Production readiness and observability
**Key Features**: Timeout protection, metrics endpoint, configuration management

### [1.0.0] - 2025-01-29
**Focus**: Core agent runtime implementation
**Key Features**: Hybrid planning, tool system, comprehensive testing

---

## Migration Guide

### Upgrading to Unreleased from 1.0.0

**Breaking Changes**: None

**New Features**:
1. **Timeout Configuration**: Add `STEP_TIMEOUT=30.0` to your `.env` file
2. **Metrics Endpoint**: Access system metrics at `GET /api/v1/metrics`
3. **Environment Variables**: Copy `.env.example` to `.env` and configure

**Configuration Updates**:
```bash
# Copy example configuration
cp .env.example .env

# Add timeout setting (optional, defaults to 30.0)
STEP_TIMEOUT=30.0
```

**Code Changes**:
```python
# Before (still works)
orchestrator = Orchestrator()

# After (with custom timeout)
orchestrator = Orchestrator(step_timeout=60.0)
```

**Monitoring**:
```bash
# Check system metrics
curl http://localhost:8000/api/v1/metrics

# Response includes:
# - Run statistics and success rate
# - Average execution duration
# - Tool usage breakdown
```

---

## Deprecation Notices

None at this time.

---

## Contributors

- Marcus Carvalho - Initial implementation and interview preparation enhancements

---

## License

[Add License Information]
