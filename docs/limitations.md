# Known Limitations

## 1. Scalability Constraints

### **In-Memory State (No Persistence)**

**Current Implementation**:

```python
class RunManager:
    def __init__(self):
        self._runs: dict[str, Run] = {}  # Lost on restart
        self._tasks: dict[str, asyncio.Task] = {}
```

**Limitations**:

- ❌ State lost on server restart
- ❌ Cannot distribute across multiple instances (no horizontal scaling)
- ❌ Memory grows unbounded (no TTL or archival strategy)
- ❌ Cannot resume failed runs after restart

**Impact**: Not production-ready for multi-instance deployments or long-running services

**Documented**: README line 988 explicitly acknowledges this limitation

---

### **Sequential Execution Only**

**Current Behavior**:

```python
for step in plan_steps:
    result = await execute_step_with_retry(step)  # Serial execution
```

**Limitations**:

- ❌ Independent steps executed serially
- ❌ No parallelization for I/O-bound operations
- ❌ Wastes async/await benefits

**Example Impact**:

- 3 concurrent API calls: 3x time instead of 1x with parallel execution
- Network-bound operations don't benefit from async design

---

## 2. Observability Gaps

### **No Production Monitoring**

**Missing Infrastructure**:

- ❌ No structured logging (console logs only)
- ❌ No metrics collection (Prometheus/StatsD)
- ❌ No distributed tracing (OpenTelemetry)
- ❌ No request correlation IDs
- ❌ No alerting or dashboards

**Impact**: Difficult to debug production issues, no SLA tracking, blind to performance degradation

---

### **Limited Metrics Tracking**

**Current Tracking** (`metrics_tracker.py`):

```python
total_plans: int
llm_plans: int
pattern_plans: int
total_tokens: int
total_latency_ms: float
```

**Missing Metrics**:

- ❌ Step-level latency distribution (p50, p95, p99)
- ❌ Error rate by tool/step type
- ❌ Retry success rate
- ❌ Queue depth and backpressure signals
- ❌ Model performance metrics (accuracy, drift detection)

**Impact**: Cannot optimize performance or detect degradation

---

### **No Structured Logging**

**Current**:

```python
logger.error(f"Run {run_id} failed at step {step_number}: {error}")
```

**Missing**:

```python
logger.error("run_execution_failed", extra={
    "run_id": run_id,
    "step_number": step_number,
    "error_type": type(error).__name__,
    "duration_ms": duration_ms
})
```

**Impact**: Difficult to query logs in production systems (Splunk, DataDog)

---

## 3. API Limitations

### **No Real-Time Updates**

**Current Approach**: Polling required

```python
# Client must repeatedly poll
GET / api / v1 / runs / {run_id}  # Check status every N seconds
```

**Missing**:

- ❌ No WebSocket support for real-time updates
- ❌ No Server-Sent Events (SSE)
- ❌ No push notifications

**Impact**: Inefficient for long-running operations, poor user experience

---

### **No Run Cancellation**

**Missing Endpoint**:

```python
# NOT IMPLEMENTED
DELETE / api / v1 / runs / {run_id}  # Cancel running execution
```

**Impact**: Cannot stop runaway or incorrect executions

---

### **Missing Pagination/Filtering**

**Current**:

```python
def list_runs(self, limit=10, offset=0):
# Missing: total_count, next_page, has_more
```

**Missing Features**:

- ❌ Cannot filter by status (running, completed, failed)
- ❌ Cannot search by prompt text
- ❌ Cannot filter by date range
- ❌ Cannot get runs by tool usage
- ❌ No pagination metadata (total_count, has_more)

---

## 4. Reliability Gaps

### **No Circuit Breaker**

**Current Risk**:

```python
# Retry storms possible during LLM provider outages
for attempt in range(3):
    try:
        plan = await litellm.acompletion(...)  # No circuit breaker
    except Exception:
        await asyncio.sleep(2 ** attempt)  # Exponential backoff only
```

**Missing**:

- ❌ No circuit breaker pattern
- ❌ No fail-fast during cascading failures
- ❌ No jitter in retry logic

**Impact**: Thundering herd problem during API downtime, retry storms

---

### **No Graceful Shutdown**

**Missing**:

```python
# __main__.py - No shutdown handler
import signal


def handle_sigterm(signum, frame):
# Drain in-flight requests
# Close connections gracefully
```

**Impact**: In-flight requests interrupted during deployment, potential data loss during rolling updates

---

### **No Rate Limiting**

**Missing Protection**:

```python
@router.post("/runs")  # No rate limiting
async def create_run(request: RunRequest):
# Vulnerable to API abuse
```

**Risks**:

- ❌ API abuse and resource exhaustion
- ❌ Cost spikes from malicious usage
- ❌ Service degradation under load
- ❌ No backpressure handling

---

### **No Idempotency for Run Restart**

**Current Limitation**: Step-level retry only, no run-level restart

**Missing**:

```python
# NOT IMPLEMENTED
POST / api / v1 / runs / {run_id} / restart  # Resume from checkpoint
```

**Impact**:

- ❌ Cannot resume from step 3/5 after failure
- ❌ Must re-execute entire run on failure
- ❌ No checkpointing mechanism

**Note**: "Idempotent failed runs" requirement interpretation unclear (step-level vs run-level)

---

## 5. Tool System Limitations

### **Calculator Tool**

**Supported Operations**: Only `+`, `-`, `*`, `/`, `**`

**Missing**:

- ❌ No scientific functions (sqrt, sin, cos, tan, log)
- ❌ No constants (π, e)
- ❌ No variable support or memory
- ❌ No unit conversions

**Example Failures**:

```python
"calculate square root of 16" → Error(no
sqrt)
"calculate sin(45 degrees)" → Error(no
sin)
"calculate pi * radius^2" → Error(no
π
constant)
```

---

### **TodoStore Tool**

**Storage**: In-memory only (no persistence)

**Missing Features**:

- ❌ No search or filtering
- ❌ No pagination for large lists
- ❌ No priorities or due dates
- ❌ No tags or categories
- ❌ No sorting options
- ❌ No bulk operations

**Impact**: Resets on every restart, limited functionality

---

### **Tool Registry**

**Limitations**:

- ❌ Static registration (no hot-reload)
- ❌ No versioning support
- ❌ No usage metrics per tool
- ❌ No tool discovery API
- ❌ No runtime tool addition

---

## 6. LLM Planning Limitations

### **Pattern Planner Brittleness**

**Current Approach**: ~10-15 hardcoded regex patterns

**Limitations**:

- ❌ Cannot handle novel task structures
- ❌ No learning or adaptation
- ❌ Brittle regex maintenance
- ❌ Cannot handle ambiguous language

**Example Failures**:

```python
"Calculate pi times radius squared" → Pattern
doesn
't know π = 3.14159
"Add a todo for tomorrow's meeting" → No
date / time
handling
"Complete the urgent task" → No
semantic
understanding
of
"urgent"
```

---

### **LLM Planner Observability**

**Missing Features**:

- ❌ No token budget enforcement
- ❌ No cost alerting
- ❌ No semantic caching (only exact-match via LiteLLM)
- ❌ No prompt optimization framework
- ❌ No A/B testing support
- ❌ No fallback quality metrics

**Impact**:

- Paraphrased prompts re-execute LLM (cost penalty)
- No way to compare LLM vs pattern-based quality
- Cannot optimize prompt strategies systematically

---

### **Variable Resolution Edge Cases**

**Known Issue** (from git commit `3cae5c5`):

- LLM sometimes invents custom variable names
- Fix: Strict prompt enforcement of allowed patterns
- Limitation: Requires ongoing prompt engineering

**Allowed Patterns**: `{step_N_output}`, `{first_todo_id}`, `{last_todo_id}`

**Missing**:

- ❌ No complex expressions
- ❌ No type checking on variables
- ❌ No runtime validation of variable references

---

### **Model-Specific Quirks**

**Local Models** (from README.md):

- Qwen2.5-3B: 97% accuracy but 2.8s latency (too slow for interactive use)
- Qwen2.5-1.5B: 91% accuracy, 1.2s latency (recommended for development)
- Phi-3: Inconsistent JSON generation

**Cloud Models**:

- GPT-4o-mini: Reliable but less reasoning than GPT-4
- Claude-3-5-Haiku: Fast but requires Anthropic-specific configuration

**Impact**: Model selection requires understanding provider-specific trade-offs

---

### **Prompt Token Overhead**

**Current Size**: ~2K tokens per request

**Breakdown**:

- System prompt: ~800 tokens
- Few-shot examples (8): ~1200 tokens
- User prompt: ~50-300 tokens

**Impact**:

- Cost: ~$0.0003 per request (input tokens)
- Latency: Increased processing time
- Context window: Limits space for complex prompts

---

## 7. Testing Gaps

### **Unit Tests**

**Current**: 87% coverage, 83 passing tests

**Missing**:

- ❌ No property-based testing (Hypothesis)
- ❌ No mutation testing (test quality validation)
- ❌ No contract testing for tool interfaces
- ❌ No snapshot testing for API responses

---

### **Integration Tests**

**Missing**:

- ❌ No load testing (Locust, k6)
- ❌ No chaos engineering (fault injection)
- ❌ No soak testing (long-running stability)
- ❌ No performance regression tests

**Impact**: Unknown behavior under load, no performance baselines

---

### **Security Testing**

**Current**: 5 injection attack tests (good!)

**Missing**:

- ❌ No OWASP Top 10 coverage
- ❌ No dependency scanning (Snyk, Safety)
- ❌ No secrets scanning
- ❌ No fuzzing tests

---

## 8. Documentation Gaps

### **API Documentation**

**Current**: OpenAPI/Swagger available

**Missing**:

- ❌ No API changelog
- ❌ No migration guides for breaking changes
- ❌ No deprecation policy
- ❌ No client SDK examples (Python, JavaScript, etc.)
- ❌ No rate limit documentation

---

### **Developer Documentation**

**Current**: Comprehensive README (1138 lines), inline docstrings

**Missing**:

- ❌ No architecture decision records (ADRs)
- ❌ No contributing guidelines
- ❌ No debugging guides
- ❌ No performance tuning guide
- ❌ No deployment documentation

---

## 9. Configuration & Security

### **No Authentication/Authorization**

**Missing**:

```python
# No API key validation
# No user management
# No role-based access control (RBAC)
```

**Impact**: Open to abuse, no multi-tenancy support

---

### **No Security Headers**

**Missing**:

```python
X - Content - Type - Options: nosniff
X - Frame - Options: DENY
Strict - Transport - Security: max - age = 31536000
Content - Security - Policy: ...
```

**Impact**: Vulnerable to XSS, clickjacking, and other attacks

---

### **No Secrets Management**

**Current**: Environment variables only

**Missing**:

- ❌ No integration with vault systems (AWS Secrets Manager, HashiCorp Vault)
- ❌ No key rotation support
- ❌ No secrets encryption at rest

---

## 10. Performance Limitations

### **No Caching Layer**

**Missing**:

- ❌ No Redis for frequent queries
- ❌ No CDN for static assets
- ❌ No response caching (HTTP cache headers)

---

### **No Request Batching**

**Current**: Each request processed individually

**Missing**:

- ❌ No batch API for multiple plans
- ❌ No request coalescing for duplicate prompts

---

### **No Connection Pooling**

**Missing**:

- ❌ No database connection pool (when added)
- ❌ No HTTP client connection pooling for LLM APIs

---

## Summary

**Critical Production Blockers**:

1. ❌ No persistent state (data loss on restart)
2. ❌ No observability (blind in production)
3. ❌ No authentication/authorization
4. ❌ Sequential execution only (performance limitation)
5. ❌ No circuit breaker or graceful shutdown

**Acknowledged Limitations** (documented in README):

- In-memory state management
- Limited tool capabilities
- Pattern planner brittleness

**Testing Gaps**: Load testing, chaos engineering, property-based testing

**Production Readiness Score**: 4/10

- Solid foundation but missing critical infrastructure
- Clear path forward with realistic timeline estimates (20-30 hours)

---

## Related Documentation

- **[Potential Improvements](./improvements.md)**: Detailed roadmap to address these limitations
- **[Design Decisions](./design_decisions.md)**: Understanding the rationale behind current choices
- **[System Architecture](./architecture.md)**: Current system design and structure
- **[Deployment Guide](./deployment.md)**: Current deployment capabilities and configuration
- **[API Examples](./api_examples.md)**: Current API capabilities and constraints
