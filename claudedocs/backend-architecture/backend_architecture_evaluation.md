# Backend Architecture Evaluation: Crane AI Agent Runtime

**Evaluator Persona**: Senior Backend Architect
**Date**: 2025-01-29
**Current State**: 4-Layer Architecture (API → Planner → Orchestrator → Tools)
**Coverage**: 83% | **Tests**: 83/83 passing | **Time Investment**: ~6 hours

---

## Executive Summary

This is a **well-structured POC** that demonstrates solid software engineering fundamentals. The candidate has made appropriate trade-offs for a time-boxed assignment while showing awareness of production requirements. However, **significant architectural gaps** prevent this from being production-ready or horizontally scalable.

**Senior Engineer Evidence**:
- ✅ Clean separation of concerns with 4-layer architecture
- ✅ Hybrid planner design (LLM + fallback) shows operational maturity
- ✅ Async/await properly leveraged with background task execution
- ✅ Security-first approach (AST-based calculator, no eval/exec)
- ✅ Honest documentation of limitations and trade-offs

**Critical Gaps**:
- ❌ No persistent state management (in-memory only)
- ❌ No horizontal scalability (single-process, shared state)
- ❌ No distributed coordination for long-running workflows
- ❌ Sequential-only execution (no DAG or parallel orchestration)
- ❌ Missing production observability patterns

---

## I. Current Architecture Analysis

### 1.1 Architectural Strengths ✅

#### **Clear Layer Separation**
```
API Layer (FastAPI)          → HTTP interface, validation
  ↓ Dependency Injection
Planner Layer (Hybrid)        → LLM + Pattern-based fallback
  ↓ Plan Generation
Orchestrator Layer (Sequential) → Retry logic, state tracking
  ↓ Tool Execution
Tool Layer (Plugin)           → Calculator, TodoStore
```

**Why This Works**:
- Each layer has single responsibility
- Dependency injection enables testing and mocking
- Async/await throughout prevents thread blocking
- Pydantic models provide runtime validation

**Current DI Pattern**:
```python
# dependencies.py
@lru_cache
def get_orchestrator() -> Orchestrator:
    planner = LLMPlanner(model="gpt-4o-mini", fallback=PatternBasedPlanner())
    return Orchestrator(planner=planner, tools=get_tool_registry())

# Usage in routes
async def create_run(orchestrator: OrchestratorDep) -> Run:
    return await orchestrator.create_run(prompt)
```

**Assessment**: ✅ Clean, testable, but singleton pattern limits scalability.

---

#### **Hybrid Planner Strategy**
```python
planner = LLMPlanner(
    model="gpt-4o-mini",
    fallback=PatternBasedPlanner()  # Graceful degradation
)
```

**Why This is Senior-Level Thinking**:
- Acknowledges LLM reliability issues (rate limits, API failures)
- Provides deterministic fallback for core operations
- Cost-optimized (GPT-4o-mini: $0.15 per 1M tokens)
- Demonstrates operational maturity beyond "just use LLM"

---

#### **Async Background Execution**
```python
async def create_run(self, prompt: str) -> Run:
    run = Run(prompt=prompt)
    plan = await self.planner.create_plan(prompt)

    # Non-blocking execution
    task = asyncio.create_task(self._execute_run(run.run_id))
    self.tasks[run.run_id] = task

    return run  # Returns immediately (PENDING status)
```

**Why This is Critical**:
- API endpoints don't block on long-running operations
- FastAPI serves other requests while runs execute
- Properly leverages Python async/await for I/O concurrency

---

#### **Retry with Exponential Backoff + Timeout**
```python
# Exponential backoff: 1s → 2s → 4s
for attempt in range(1, self.max_retries + 1):
    try:
        result = await asyncio.wait_for(
            tool.execute(**step.tool_input),
            timeout=self.step_timeout  # Default: 30s
        )
        return result
    except asyncio.TimeoutError:
        # Timeout protection
    except Exception:
        if attempt < self.max_retries:
            await asyncio.sleep(2 ** (attempt - 1))
```

**Why This is Industry Standard**:
- Handles transient failures (network glitches, temporary unavailability)
- Prevents indefinite hangs with timeout protection
- Exponential spacing reduces load during outages

---

### 1.2 Critical Architectural Weaknesses ❌

#### **1. State Management: In-Memory Only**

**Current Implementation**:
```python
class Orchestrator:
    def __init__(self):
        self.runs: dict[str, Run] = {}  # ❌ In-memory dict
        self.tasks: dict[str, asyncio.Task] = {}
```

**Problems**:
| Issue | Impact | Production Risk |
|-------|--------|----------------|
| State lost on restart | All runs disappear | 🔴 **CRITICAL** |
| Not horizontally scalable | Can't add servers | 🔴 **CRITICAL** |
| Memory exhaustion | Large runs OOM server | 🟡 High |
| No persistence | Can't resume failed runs | 🟡 High |
| No run history | Zero observability | 🟡 High |

**Real-World Failure Scenario**:
```
User: "Calculate complex math, add 5 todos, list results"
System: Run executes for 2 minutes
Deploy: New version deployed (server restart)
Result: ❌ All progress lost, user sees 404 on GET /runs/{id}
```

---

#### **2. No Horizontal Scalability**

**Current Architecture**:
```
┌────────────────────────────────────┐
│   FastAPI Process (Single)         │
│                                    │
│  ┌──────────────────────────┐    │
│  │  In-Memory State         │    │
│  │  runs: dict[str, Run]    │    │
│  │  tasks: dict[str, Task]  │    │
│  └──────────────────────────┘    │
│                                    │
│  Background: asyncio.create_task() │
└────────────────────────────────────┘
         ↓ User requests
```

**Scaling Attempt (fails)**:
```
Load Balancer
    ↓
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Server1 │  │ Server2 │  │ Server3 │
│ runs:{} │  │ runs:{} │  │ runs:{} │  ❌ Isolated state
└─────────┘  └─────────┘  └─────────┘
```

**Why This Breaks**:
- Server 1 creates run → state in Server 1 memory
- Server 2 receives status check → 404 (doesn't have state)
- Background tasks tied to specific process (can't migrate)
- No cross-process coordination

---

#### **3. Sequential-Only Execution**

**Current Implementation**:
```python
# orchestrator.py - Sequential loop
for step in run.plan.steps:
    step_result = await self._execute_step_with_retry(step)
    run.execution_log.append(step_result)

    if not step_result.success:
        return  # Stop on failure
```

**Missed Parallelization Opportunities**:
```python
# Example plan that could parallelize:
steps = [
    {"tool": "calculator", "input": {"expression": "2+3"}},     # Independent
    {"tool": "calculator", "input": {"expression": "10*5"}},    # Independent
    {"tool": "todo_store", "input": {"action": "list"}},       # Independent
]
# Current: 3 * avg_latency = Total time
# Optimal: max(latencies) = Total time (3x speedup)
```

**When Sequential is Wrong**:
- Independent calculations (no data dependency)
- Parallel API calls to different services
- Bulk operations (process 100 todos → could batch)

---

#### **4. No Distributed Coordination**

**Missing Patterns**:
- **Saga Pattern**: Long-running workflows with compensating transactions
- **Event Sourcing**: Reconstruct state from event history
- **CQRS**: Separate read/write models for scale
- **Circuit Breaker**: Prevent cascading failures

**Real-World Agent Scenario** (unsupported):
```
User: "Analyze this dataset, train model, deploy to prod"
Expected:
  Step 1: Data prep (30 min)   → persist checkpoint
  Step 2: Model training (2 hr) → persist checkpoint
  Step 3: Deployment (10 min)   → rollback on failure

Current System:
  ❌ No checkpoints (server restart = start over)
  ❌ No rollback (step 3 fails = stuck state)
  ❌ No resume (can't continue from step 2)
```

---

#### **5. Missing API Capabilities**

**Current API**:
```
POST   /runs         → Create run
GET    /runs/{id}    → Get status
GET    /health       → Health check
GET    /metrics      → Basic metrics
```

**Missing Production Endpoints**:
```
DELETE /runs/{id}           → Cancel running operation
POST   /runs/{id}/retry     → Retry from failed step
GET    /runs                → List runs (pagination)
POST   /runs/{id}/resume    → Resume from checkpoint
GET    /runs/{id}/stream    → SSE/WebSocket for real-time updates
POST   /tools/{name}/test   → Dry-run tool execution
```

---

#### **6. No Observability Infrastructure**

**Current Logging**:
```python
logger.info(f"Starting execution for run {run_id}")  # Basic
logger.error(f"Run {run_id} failed: {error}")        # Basic
```

**Missing Production Observability**:
| Component | Current | Production Needs |
|-----------|---------|------------------|
| Logging | Basic print-style | Structured JSON, correlation IDs |
| Metrics | Manual counter | Prometheus/StatsD |
| Tracing | None | OpenTelemetry spans |
| Alerting | None | Dead letter queue, PagerDuty |
| Dashboard | None | Grafana, run success rates |

---

## II. Senior Backend Architect Recommendations

### 2.1 Immediate Critical Fixes (Next 4-8 Hours)

#### **Priority 1: Persistent State Management** 🔴

**Problem**: In-memory state lost on restart.

**Solution**: Dual-store pattern (Redis + PostgreSQL)

```python
# Architecture
┌─────────────────────────────────────────────────────┐
│ API Layer (FastAPI)                                 │
└────────────┬────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────┐
│ Orchestrator (State Coordinator)                   │
│                                                     │
│  ┌─────────────────┐      ┌────────────────────┐  │
│  │ Redis           │      │ PostgreSQL         │  │
│  │ (Hot State)     │      │ (Historical)       │  │
│  │                 │      │                    │  │
│  │ Active runs     │  →   │ Completed runs     │  │
│  │ TTL: 1 hour     │      │ Archived forever   │  │
│  │ Fast: <1ms      │      │ Analytics ready    │  │
│  └─────────────────┘      └────────────────────┘  │
└────────────────────────────────────────────────────┘
```

**Implementation**:
```python
# core/state/repository.py
from abc import ABC, abstractmethod
from typing import Protocol

class StateRepository(Protocol):
    """State persistence interface."""

    async def save_run(self, run: Run) -> None:
        """Save run state."""
        ...

    async def get_run(self, run_id: str) -> Run | None:
        """Retrieve run state."""
        ...

    async def update_status(self, run_id: str, status: RunStatus) -> None:
        """Update run status atomically."""
        ...


# core/state/redis_repository.py
import redis.asyncio as redis
from datetime import timedelta

class RedisStateRepository:
    """Fast state storage for active runs."""

    def __init__(self, redis_url: str, ttl: timedelta = timedelta(hours=1)):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl

    async def save_run(self, run: Run) -> None:
        """Save with automatic expiration."""
        key = f"run:{run.run_id}"
        await self.redis.setex(
            key,
            self.ttl,
            run.model_dump_json()  # Pydantic serialization
        )

    async def get_run(self, run_id: str) -> Run | None:
        """Fast retrieval from Redis."""
        data = await self.redis.get(f"run:{run_id}")
        return Run.model_validate_json(data) if data else None

    async def update_status(self, run_id: str, status: RunStatus) -> None:
        """Atomic status update."""
        run = await self.get_run(run_id)
        if run:
            run.status = status
            await self.save_run(run)


# core/state/postgres_repository.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

class PostgresStateRepository:
    """Long-term storage for completed runs."""

    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url)

    async def archive_run(self, run: Run) -> None:
        """Move completed run to permanent storage."""
        async with AsyncSession(self.engine) as session:
            db_run = RunModel(
                run_id=run.run_id,
                prompt=run.prompt,
                status=run.status,
                plan=run.plan.model_dump() if run.plan else None,
                execution_log=[step.model_dump() for step in run.execution_log],
                result=run.result,
                created_at=run.created_at,
                completed_at=run.completed_at,
            )
            session.add(db_run)
            await session.commit()

    async def query_runs(
        self,
        status: RunStatus | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Run]:
        """Query historical runs for analytics."""
        # Implementation with SQLAlchemy query
        ...


# orchestrator/orchestrator.py (updated)
class Orchestrator:
    def __init__(
        self,
        hot_state: RedisStateRepository,
        cold_state: PostgresStateRepository,
        planner: Planner,
        tools: dict,
    ):
        self.hot_state = hot_state  # Active runs
        self.cold_state = cold_state  # Historical data
        self.planner = planner
        self.tools = tools

    async def create_run(self, prompt: str) -> Run:
        """Create run with persistent state."""
        run = Run(prompt=prompt)
        plan = await self.planner.create_plan(prompt)
        run.plan = plan

        # Persist to Redis
        await self.hot_state.save_run(run)

        # Start execution (background task will update state)
        asyncio.create_task(self._execute_run(run.run_id))

        return run

    async def get_run(self, run_id: str) -> Run | None:
        """Get run from hot storage, fallback to cold."""
        # Try Redis first (fast path)
        run = await self.hot_state.get_run(run_id)
        if run:
            return run

        # Fallback to PostgreSQL (historical)
        return await self.cold_state.get_run_by_id(run_id)

    async def _execute_run(self, run_id: str) -> None:
        """Execute with state updates."""
        run = await self.hot_state.get_run(run_id)

        run.status = RunStatus.RUNNING
        await self.hot_state.save_run(run)

        for step in run.plan.steps:
            result = await self._execute_step_with_retry(step)
            run.execution_log.append(result)

            # Update state after each step
            await self.hot_state.save_run(run)

            if not result.success:
                run.status = RunStatus.FAILED
                await self.hot_state.save_run(run)
                await self.cold_state.archive_run(run)  # Archive failure
                return

        run.status = RunStatus.COMPLETED
        await self.hot_state.save_run(run)

        # Archive to PostgreSQL for long-term storage
        await self.cold_state.archive_run(run)
```

**Benefits**:
- ✅ State survives server restarts
- ✅ Horizontally scalable (all servers share Redis)
- ✅ Fast active run queries (<1ms from Redis)
- ✅ Historical analytics (PostgreSQL queries)
- ✅ Automatic cleanup (Redis TTL expires old runs)

**Trade-offs**:
- Adds infrastructure complexity (Redis + PostgreSQL)
- Requires serialization overhead
- Network latency on every state update (~1-2ms)

---

#### **Priority 2: Horizontal Scalability via Queue** 🔴

**Problem**: Background tasks tied to single process.

**Solution**: Queue-based execution with worker pool

```python
# Architecture
┌──────────────────────────────────────────────────────┐
│ Load Balancer                                        │
└─────┬────────────┬────────────┬──────────────────────┘
      ↓            ↓            ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│ API Pod1 │ │ API Pod2 │ │ API Pod3 │  ← Stateless
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     └────────────┼────────────┘
                  ↓
         ┌────────────────┐
         │ Redis Queue    │  ← Durable queue
         │ (Bull/Celery)  │
         └────────┬───────┘
                  ↓
     ┌────────────┼────────────┐
     ↓            ↓            ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Worker 1 │ │ Worker 2 │ │ Worker N │  ← Scalable workers
└──────────┘ └──────────┘ └──────────┘
```

**Implementation**:
```python
# orchestrator/queue.py
from arq import create_pool
from arq.connections import RedisSettings

async def execute_run_task(ctx: dict, run_id: str) -> None:
    """
    Background worker function.

    Executes independently of API process.
    Can scale horizontally by adding workers.
    """
    orchestrator = ctx["orchestrator"]

    # Retrieve run from Redis
    run = await orchestrator.hot_state.get_run(run_id)

    run.status = RunStatus.RUNNING
    await orchestrator.hot_state.save_run(run)

    # Execute steps
    for step in run.plan.steps:
        result = await orchestrator._execute_step_with_retry(step)
        run.execution_log.append(result)
        await orchestrator.hot_state.save_run(run)

        if not result.success:
            run.status = RunStatus.FAILED
            await orchestrator.hot_state.save_run(run)
            await orchestrator.cold_state.archive_run(run)
            return

    run.status = RunStatus.COMPLETED
    await orchestrator.hot_state.save_run(run)
    await orchestrator.cold_state.archive_run(run)


# orchestrator/orchestrator.py (updated)
class Orchestrator:
    def __init__(
        self,
        redis_pool,  # arq Redis pool
        hot_state: RedisStateRepository,
        cold_state: PostgresStateRepository,
        planner: Planner,
        tools: dict,
    ):
        self.redis = redis_pool
        self.hot_state = hot_state
        self.cold_state = cold_state
        self.planner = planner
        self.tools = tools

    async def create_run(self, prompt: str) -> Run:
        """Create run and enqueue for execution."""
        run = Run(prompt=prompt)
        plan = await self.planner.create_plan(prompt)
        run.plan = plan

        # Save to Redis
        await self.hot_state.save_run(run)

        # Enqueue for background execution
        await self.redis.enqueue_job(
            "execute_run_task",
            run_id=run.run_id
        )

        return run


# workers/run_worker.py
from arq import create_pool
from arq.connections import RedisSettings

async def startup(ctx: dict) -> None:
    """Initialize worker dependencies."""
    ctx["orchestrator"] = Orchestrator(...)

async def shutdown(ctx: dict) -> None:
    """Cleanup on worker shutdown."""
    await ctx["orchestrator"].close()

class WorkerSettings:
    functions = [execute_run_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn("redis://localhost")


# Start workers
# $ arq workers.run_worker.WorkerSettings --watch
```

**Scaling**:
```bash
# Production deployment
kubectl scale deployment api-pods --replicas=10      # 10 API servers
kubectl scale deployment worker-pods --replicas=50   # 50 workers

# Each API pod: Stateless, handles HTTP
# Each worker pod: Executes runs from queue
# Redis: Shared state + queue coordination
```

**Benefits**:
- ✅ Horizontal scalability (add API pods + workers independently)
- ✅ Resilient (worker crash → run requeued automatically)
- ✅ Load distribution (queue ensures even work distribution)
- ✅ Backpressure (queue size = system health metric)

**Trade-offs**:
- Adds queue infrastructure (Redis/RabbitMQ)
- Slightly higher latency (queue hop adds ~5-10ms)
- Requires worker deployment and monitoring

---

#### **Priority 3: DAG-Based Parallel Execution** 🟡

**Problem**: Sequential execution wastes time on independent steps.

**Solution**: Directed Acyclic Graph (DAG) execution engine

```python
# models/plan.py (enhanced)
from typing import Set

class PlanStep(BaseModel):
    step_number: int
    tool_name: str
    tool_input: dict[str, Any]
    reasoning: str
    depends_on: Set[int] = Field(default_factory=set)  # NEW: Dependencies

class Plan(BaseModel):
    steps: list[PlanStep]
    final_goal: str

    def get_execution_dag(self) -> dict[int, Set[int]]:
        """Build dependency graph for parallel execution."""
        return {
            step.step_number: step.depends_on
            for step in self.steps
        }


# orchestrator/dag_executor.py
import asyncio
from collections import defaultdict

class DAGExecutor:
    """Execute plan steps in parallel respecting dependencies."""

    async def execute_plan(
        self,
        plan: Plan,
        tool_executor: callable
    ) -> list[ExecutionStep]:
        """
        Execute plan with maximum parallelism.

        Algorithm: Topological sort + level-based execution
        """
        # Build dependency graph
        dag = plan.get_execution_dag()
        in_degree = {step.step_number: len(step.depends_on) for step in plan.steps}

        # Track results
        results = {}
        execution_log = []

        # Find steps with no dependencies (can start immediately)
        ready_queue = [
            step for step in plan.steps
            if in_degree[step.step_number] == 0
        ]

        while ready_queue:
            # Execute all ready steps in parallel
            tasks = [
                tool_executor(step) for step in ready_queue
            ]
            step_results = await asyncio.gather(*tasks)

            # Update results and log
            for step, result in zip(ready_queue, step_results):
                results[step.step_number] = result
                execution_log.append(result)

                # Check if this unlocks dependent steps
                for next_step in plan.steps:
                    if step.step_number in next_step.depends_on:
                        in_degree[next_step.step_number] -= 1

            # Find newly ready steps
            ready_queue = [
                step for step in plan.steps
                if in_degree[step.step_number] == 0
                and step.step_number not in results
            ]

        return execution_log


# Example: Parallel plan
plan = Plan(
    steps=[
        PlanStep(1, "calculator", {"expression": "2+3"}, depends_on=set()),
        PlanStep(2, "calculator", {"expression": "10*5"}, depends_on=set()),
        PlanStep(3, "todo_store", {"action": "list"}, depends_on=set()),
        PlanStep(4, "calculator", {"expression": "step1 + step2"}, depends_on={1, 2}),
    ]
)

# Execution timeline:
# T=0:   Steps 1, 2, 3 start in parallel (no dependencies)
# T=100: Steps 1, 2, 3 complete
# T=100: Step 4 starts (dependencies satisfied)
# T=150: Step 4 completes
#
# Sequential: 150ms total
# Parallel:   100ms total (33% faster)
```

**When to Use DAG**:
- Independent calculations (no shared state)
- Bulk operations (process 100 todos → 10 parallel batches)
- External API calls (fetch from multiple sources)

**When to Stay Sequential**:
- Steps modify shared state (TodoStore operations)
- User expects specific order
- Simplicity > performance (early optimization is evil)

---

### 2.2 Medium Priority Improvements (8-16 Hours)

#### **Enhancement 1: Circuit Breaker Pattern**

**Problem**: Tool failures cascade through system.

**Solution**: Circuit breaker with health tracking

```python
# orchestrator/circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit breaker for tool reliability.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests blocked immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: timedelta = timedelta(seconds=60),
        half_open_max_calls: int = 3
    ):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.half_open_successes = 0
        self.half_open_max_calls = half_open_max_calls

    async def call(self, func: callable, *args, **kwargs):
        """Execute function through circuit breaker."""

        # Check if circuit should transition
        if self.state == CircuitState.OPEN:
            if datetime.now() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_successes = 0
            else:
                raise CircuitBreakerOpen("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_max_calls:
                # Recovered!
                self.state = CircuitState.CLOSED
                self.failure_count = 0

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


# Integration with Orchestrator
class Orchestrator:
    def __init__(self, ...):
        # Circuit breaker per tool
        self.circuit_breakers = {
            tool_name: CircuitBreaker()
            for tool_name in tools.keys()
        }

    async def _execute_step_with_retry(self, step: PlanStep) -> ExecutionStep:
        """Execute with circuit breaker protection."""
        breaker = self.circuit_breakers[step.tool_name]

        try:
            result = await breaker.call(
                self._execute_tool,
                step.tool_name,
                step.tool_input
            )
            return ExecutionStep(success=True, output=result)

        except CircuitBreakerOpen:
            return ExecutionStep(
                success=False,
                error=f"Tool {step.tool_name} temporarily unavailable (circuit breaker open)"
            )
```

**Benefits**:
- Prevents cascading failures
- Automatic recovery detection
- Fail-fast for degraded services

---

#### **Enhancement 2: Event Sourcing for Auditability**

**Problem**: Can't reconstruct how run reached current state.

**Solution**: Event-sourced state management

```python
# models/events.py
from typing import Literal

class RunEvent(BaseModel):
    """Immutable event representing state change."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    event_type: Literal[
        "run_created",
        "plan_generated",
        "execution_started",
        "step_started",
        "step_completed",
        "step_failed",
        "run_completed",
        "run_failed"
    ]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


# orchestrator/event_store.py
class EventStore:
    """Append-only event storage."""

    async def append_event(self, event: RunEvent) -> None:
        """Write event to immutable log."""
        await self.postgres.execute(
            "INSERT INTO run_events (event_id, run_id, event_type, timestamp, data) "
            "VALUES ($1, $2, $3, $4, $5)",
            event.event_id, event.run_id, event.event_type, event.timestamp, event.data
        )

    async def get_events(self, run_id: str) -> list[RunEvent]:
        """Retrieve all events for run."""
        rows = await self.postgres.fetch(
            "SELECT * FROM run_events WHERE run_id = $1 ORDER BY timestamp",
            run_id
        )
        return [RunEvent(**row) for row in rows]

    def reconstruct_run(self, events: list[RunEvent]) -> Run:
        """Rebuild run state from event history."""
        run = Run(run_id=events[0].run_id, prompt=events[0].data["prompt"])

        for event in events:
            if event.event_type == "plan_generated":
                run.plan = Plan(**event.data["plan"])
            elif event.event_type == "execution_started":
                run.status = RunStatus.RUNNING
                run.started_at = event.timestamp
            elif event.event_type == "step_completed":
                run.execution_log.append(ExecutionStep(**event.data))
            # ... handle other events

        return run
```

**Benefits**:
- Complete audit trail
- Time-travel debugging
- Replay events for testing

---

#### **Enhancement 3: API Rate Limiting & Authentication**

```python
# middleware/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/runs")
@limiter.limit("10/minute")  # 10 requests per minute
async def create_run(request: Request, ...):
    ...


# middleware/auth.py
from fastapi import Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Authorization header."""
    api_key = credentials.credentials

    # Check against database or environment
    if not await is_valid_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


@router.post("/runs")
async def create_run(api_key: str = Depends(verify_api_key)):
    ...
```

---

### 2.3 Advanced Patterns for Production (16+ Hours)

#### **Pattern 1: CQRS (Command Query Responsibility Segregation)**

**Scenario**: High read load on `/runs/{id}` endpoint.

```python
# Architecture
┌─────────────┐
│ Write Model │  ← POST /runs (writes to PostgreSQL + Redis)
└─────────────┘
       ↓
  Event Stream
       ↓
┌─────────────┐
│ Read Model  │  ← GET /runs/{id} (reads from optimized view)
│ (Materialized│
│  View)      │
└─────────────┘
```

**Implementation**:
```python
# Command side: Write operations
class CommandService:
    async def create_run(self, prompt: str) -> Run:
        run = Run(prompt=prompt)

        # Write to event store
        await self.event_store.append(RunCreatedEvent(run_id=run.run_id, ...))

        # Write to command database
        await self.write_db.insert_run(run)

        return run


# Query side: Read operations
class QueryService:
    async def get_run(self, run_id: str) -> RunView:
        """Read from optimized materialized view."""
        return await self.read_db.get_run_view(run_id)


# Event handler: Update read model
async def on_run_event(event: RunEvent):
    """Update materialized view when events occur."""
    if event.event_type == "step_completed":
        await read_db.update_run_view(
            event.run_id,
            latest_step=event.data
        )
```

---

#### **Pattern 2: Saga Pattern for Long-Running Workflows**

**Scenario**: Multi-step workflow with rollback requirements.

```python
# Example: Deploy ML model
class DeployModelSaga:
    """
    Saga: Train model → Deploy to staging → Run tests → Deploy to prod

    Compensating transactions:
    - If prod deploy fails → rollback staging
    - If tests fail → delete staging deployment
    - If training fails → cleanup resources
    """

    async def execute(self, dataset_id: str):
        saga_state = SagaState(saga_id=str(uuid4()))

        try:
            # Step 1: Train model
            model = await self.train_model(dataset_id)
            saga_state.add_step("train", model.id, compensate=self.delete_model)

            # Step 2: Deploy to staging
            staging = await self.deploy_staging(model.id)
            saga_state.add_step("deploy_staging", staging.url, compensate=self.undeploy_staging)

            # Step 3: Run tests
            results = await self.run_tests(staging.url)
            if not results.passed:
                raise TestsFailed("Staging tests failed")

            # Step 4: Deploy to production
            prod = await self.deploy_production(model.id)
            saga_state.add_step("deploy_prod", prod.url, compensate=self.undeploy_prod)

            saga_state.mark_completed()
            return prod

        except Exception as e:
            # Rollback all completed steps
            await saga_state.compensate()
            raise
```

---

## III. Architecture Comparison Matrix

| Dimension | Current POC | Redis + Queue | + DAG Execution | + CQRS + Saga |
|-----------|-------------|---------------|-----------------|---------------|
| **Scalability** | 1 process | 10s of workers | 100s of workers | 1000s of workers |
| **State Durability** | ❌ Lost on restart | ✅ Survives restarts | ✅ Survives restarts | ✅ Event-sourced |
| **Execution Model** | Sequential | Sequential | Parallel (DAG) | Parallel + Compensating |
| **Latency (avg)** | 100ms | 110ms (+10ms queue) | 50ms (parallel) | 60ms (optimized reads) |
| **Fault Tolerance** | ❌ Crash = lost | ✅ Requeue on crash | ✅ Checkpoint + resume | ✅ Rollback on failure |
| **Observability** | Basic logging | Metrics + logs | Tracing + metrics | Full audit trail |
| **Operational Cost** | $50/mo (1 server) | $200/mo (Redis + workers) | $300/mo (+ compute) | $500/mo (+ storage) |
| **Development Effort** | ✅ 6 hours | 🟡 +8 hours | 🟡 +12 hours | 🔴 +24 hours |

---

## IV. Recommended Architecture Roadmap

### Phase 1: Production Baseline (Sprint 1-2, 2 weeks)
**Goal**: Make system production-ready and scalable

**Changes**:
1. ✅ Add Redis for active state (hot storage)
2. ✅ Add PostgreSQL for historical runs (cold storage)
3. ✅ Implement queue-based execution (arq/Celery)
4. ✅ Add structured logging with correlation IDs
5. ✅ Add basic metrics (Prometheus)
6. ✅ Add health checks with dependency validation

**Result**: Horizontally scalable, durable state, 99.9% uptime possible

---

### Phase 2: Performance & Reliability (Sprint 3-4, 2 weeks)
**Goal**: Optimize performance and fault tolerance

**Changes**:
1. ✅ Implement DAG-based parallel execution
2. ✅ Add circuit breakers per tool
3. ✅ Implement graceful degradation
4. ✅ Add retry with jitter
5. ✅ Add distributed tracing (OpenTelemetry)
6. ✅ Add API rate limiting & authentication

**Result**: 2-3x throughput increase, better reliability

---

### Phase 3: Enterprise Features (Sprint 5-6, 2 weeks)
**Goal**: Long-running workflows and audit compliance

**Changes**:
1. ✅ Event sourcing for audit trail
2. ✅ CQRS for read scalability
3. ✅ Saga pattern for compensating transactions
4. ✅ Checkpoint/resume for long runs
5. ✅ WebSocket/SSE for real-time updates

**Result**: Enterprise-ready, audit-compliant, handles multi-hour workflows

---

## V. Senior Backend Architect Differentiators

### What Demonstrates Deep Architectural Thinking?

#### ✅ **1. Trade-off Awareness**
The candidate explicitly documented trade-offs:
- In-memory vs persistent state
- Sequential vs parallel execution
- LLM vs pattern-based planning

**Senior-Level**: Understanding there are no perfect solutions, only trade-offs.

---

#### ✅ **2. Operational Mindset**
Hybrid planner with fallback shows production thinking:
```python
planner = LLMPlanner(fallback=PatternBasedPlanner())
```
Most candidates would just use LLM and ignore API failure scenarios.

**Senior-Level**: Designing for failure, not just success.

---

#### ✅ **3. Performance Through Async**
Proper use of `asyncio.create_task()` for non-blocking execution:
```python
task = asyncio.create_task(self._execute_run(run_id))
return run  # Returns immediately
```

**Senior-Level**: Understanding async/await concurrency model.

---

#### ❌ **4. Missing Distributed Systems Patterns**
No consideration for:
- Horizontal scalability (load balancing across pods)
- State synchronization (Redis/PostgreSQL)
- Queue-based decoupling
- Circuit breakers / bulkheads

**Gap**: POC doesn't demonstrate distributed systems experience.

---

#### ❌ **5. Missing Observability**
Basic logging, no structured metrics/tracing:
```python
logger.info(f"Starting execution")  # Basic
```

vs. Production:
```python
span = tracer.start_span("execute_run", attributes={"run_id": run.run_id})
metrics.incr("runs.started", tags={"tool": tool.name})
logger.info("execution.started", extra={"run_id": run.run_id, "correlation_id": ctx.correlation_id})
```

**Gap**: Doesn't show production debugging/monitoring experience.

---

## VI. Final Recommendation

### Summary Assessment

**Score**: 7.5/10 for Senior Backend Engineer

**Strengths**:
- ✅ Clean architecture with clear layer separation
- ✅ Async/await properly leveraged
- ✅ Security-aware (AST calculator)
- ✅ Honest about limitations
- ✅ Hybrid planner shows operational maturity

**Critical Gaps**:
- ❌ No persistent state management (production blocker)
- ❌ No horizontal scalability (can't add servers)
- ❌ No distributed system patterns (queues, circuit breakers)
- ❌ Missing observability infrastructure

---

### Hiring Decision Matrix

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| **Early-stage startup** | ✅ **HIRE** | Can build MVPs quickly, will learn distributed systems on job |
| **Mid-size company** | 🟡 **MAYBE** | Strong fundamentals, needs mentoring on scale patterns |
| **Large tech company** | ❌ **PASS** | Insufficient distributed systems experience for L5+ |

---

### Next Interview Questions

**To Probe Distributed Systems Understanding**:
1. "How would you scale this to 1000 concurrent executions?"
2. "What happens if a worker crashes mid-execution? How do you ensure exactly-once semantics?"
3. "How would you implement a 2-hour ML training workflow that survives server restarts?"
4. "Walk me through debugging a production issue where 5% of runs are mysteriously failing."

**Expected Senior-Level Answers**:
- Queue-based execution (Celery/arq)
- Redis/PostgreSQL for state
- Idempotency keys + event sourcing
- Distributed tracing + structured logs

---

## VII. Conclusion

This is a **well-executed POC** that demonstrates solid software engineering fundamentals but **lacks production-scale architecture patterns**. The candidate shows:

✅ **Strong Foundation**: Clean code, async patterns, security awareness
🟡 **Growing Experience**: Understands trade-offs but hasn't built distributed systems
❌ **Production Gaps**: No persistent state, scalability, or observability patterns

**Recommendation**:
- For senior IC role requiring distributed systems experience: **Additional interview round**
- For mid-level role with mentorship available: **Strong hire**
- For junior/early-career: **Exceptional candidate**

**Next Steps**:
1. Implement Redis + Queue architecture (Phase 1)
2. Add distributed tracing and metrics
3. Document scaling strategy for 10,000 concurrent runs
4. Demonstrate debugging production failure scenarios

This evaluation identifies specific, actionable improvements that would elevate the system from POC to production-grade.
