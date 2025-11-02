# Architecture Diagrams & Patterns

Visual representations of current architecture and recommended improvements for Crane AI Agent Runtime.

---

## Current Architecture (POC State)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT (curl/browser)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /runs
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI APPLICATION (Single Process)          │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ API LAYER (routes/runs.py)                             │   │
│  │ • Request validation (Pydantic)                        │   │
│  │ • HTTP status codes (201/404/500)                      │   │
│  │ • Dependency injection                                 │   │
│  └───────────────────────┬────────────────────────────────┘   │
│                          │ create_run(prompt)                  │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ ORCHESTRATOR LAYER (orchestrator.py)                   │   │
│  │ • In-memory state: self.runs = {}  ❌                  │   │
│  │ • Background tasks: asyncio.create_task()              │   │
│  │ • Sequential execution loop                            │   │
│  │ • Retry with exponential backoff (1s→2s→4s)            │   │
│  └───────────────────────┬────────────────────────────────┘   │
│                          │ create_plan(prompt)                 │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ PLANNER LAYER (hybrid)                                 │   │
│  │ • LLMPlanner (GPT-4o-mini) → structured JSON           │   │
│  │ • PatternBasedPlanner (fallback) → regex patterns      │   │
│  │ • Plan validation and tool verification                │   │
│  └───────────────────────┬────────────────────────────────┘   │
│                          │ execute(tool, input)                │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ TOOL LAYER (registry.py)                               │   │
│  │ ┌──────────────┐         ┌──────────────┐             │   │
│  │ │ Calculator   │         │ TodoStore    │             │   │
│  │ │ • AST-based  │         │ • In-memory  │             │   │
│  │ │ • No eval()  │         │ • CRUD ops   │             │   │
│  │ └──────────────┘         └──────────────┘             │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
│  LIMITATIONS:                                                   │
│  ❌ State lost on restart (no persistence)                     │
│  ❌ Can't scale horizontally (in-memory state)                 │
│  ❌ Background tasks tied to process (can't migrate)           │
│  ❌ Sequential execution (no parallelism)                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key Issues**:
1. **Single Point of Failure**: All state in one process
2. **No State Persistence**: Server restart = all runs lost
3. **Can't Horizontal Scale**: Load balancer would break state access
4. **No Work Distribution**: All execution in one thread pool

---

## Recommended Architecture: Phase 1 (Production Baseline)

### Redis + Queue + Worker Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                      LOAD BALANCER (nginx/ALB)                  │
└────┬────────────────────┬────────────────────┬──────────────────┘
     │                    │                    │
     ▼                    ▼                    ▼
┌──────────┐         ┌──────────┐         ┌──────────┐
│ API POD 1│         │ API POD 2│         │ API POD N│  ← STATELESS
│ FastAPI  │         │ FastAPI  │         │ FastAPI  │
└────┬─────┘         └────┬─────┘         └────┬─────┘
     │                    │                    │
     └────────────────────┼────────────────────┘
                          │
                          │ Write state + Enqueue job
                          ▼
     ┌──────────────────────────────────────────────────┐
     │           REDIS (Shared State)                   │
     │                                                   │
     │  ┌─────────────────┐    ┌────────────────────┐  │
     │  │ Hot State Store │    │ Queue (arq/Bull)   │  │
     │  │                 │    │                    │  │
     │  │ run:{id} → Run  │    │ execute_run_task   │  │
     │  │ TTL: 1 hour     │    │ {run_id, priority} │  │
     │  │ <1ms latency    │    │ Durable, ordered   │  │
     │  └─────────────────┘    └────────────────────┘  │
     └───────────────┬──────────────┬───────────────────┘
                     │              │
                     │              │ Dequeue jobs
                     │              ▼
                     │    ┌──────────────────────────┐
                     │    │   WORKER POOL            │
                     │    │                          │
                     │    │ ┌────────┐ ┌────────┐  │
                     │    │ │Worker 1│ │Worker 2│  │  ← SCALABLE
                     │    │ └────────┘ └────────┘  │
                     │    │ ┌────────┐ ┌────────┐  │
                     │    │ │Worker 3│ │Worker N│  │
                     │    │ └────────┘ └────────┘  │
                     │    └─────────┬────────────────┘
                     │              │
                     │              │ Update state
                     └──────────────┘
                                    │
                                    │ Archive completed runs
                                    ▼
     ┌──────────────────────────────────────────────────┐
     │           POSTGRESQL (Historical Storage)        │
     │                                                   │
     │  • Completed runs (permanent)                    │
     │  • Analytics queries                             │
     │  • Execution logs with indexing                  │
     │  • Run metrics for observability                 │
     └──────────────────────────────────────────────────┘
```

### Data Flow

```
1. CREATE RUN REQUEST
   Client → API Pod 1 → POST /runs {"prompt": "add todo"}

2. PLANNING
   API Pod 1 → LLM Planner → Plan(steps=[...])

3. STATE PERSISTENCE
   API Pod 1 → Redis.setex("run:abc-123", ttl=3600, run_json)

4. JOB ENQUEUE
   API Pod 1 → Redis Queue → {task: "execute_run", run_id: "abc-123"}
   Return 201 Created → Client

5. BACKGROUND EXECUTION
   Worker 2 → Dequeue job → Redis.get("run:abc-123")
   Worker 2 → Execute steps → Redis.update("run:abc-123")
   Worker 2 → PostgreSQL.archive_run(run)

6. STATUS CHECK (different pod!)
   Client → API Pod 3 → GET /runs/abc-123
   API Pod 3 → Redis.get("run:abc-123")  ✅ Found (shared state)
   Return 200 OK → Client
```

### Benefits
- ✅ **Horizontal Scalability**: Add API pods independently
- ✅ **Worker Scalability**: Scale workers based on queue depth
- ✅ **State Durability**: Redis persists state across restarts
- ✅ **Historical Archive**: PostgreSQL stores completed runs forever
- ✅ **Load Distribution**: Queue ensures even work distribution

---

## DAG-Based Parallel Execution

### Current: Sequential Execution

```
Plan: [Step1, Step2, Step3, Step4]

Timeline:
T=0ms   ─────────> Step 1: Calculate 2+3
T=100ms ─────────> Step 2: Calculate 10*5
T=200ms ─────────> Step 3: List todos
T=300ms ─────────> Step 4: Add result to todo

Total Time: 300ms ❌ Inefficient
```

### Recommended: DAG Execution

```
Plan with Dependencies:
  Step 1: Calculate 2+3       [depends_on: []]
  Step 2: Calculate 10*5      [depends_on: []]
  Step 3: List todos          [depends_on: []]
  Step 4: Add sum to todo     [depends_on: [1, 2]]

Timeline:
T=0ms   ─┬─────────> Step 1: Calculate 2+3
        ─┼─────────> Step 2: Calculate 10*5    ← PARALLEL
        ─┴─────────> Step 3: List todos        ← PARALLEL

T=100ms ─────────> All complete

T=100ms ─────────> Step 4: Add sum to todo (depends on 1+2)

T=150ms ─────────> All complete

Total Time: 150ms ✅ 2x speedup
```

### DAG Execution Algorithm

```python
# Topological sort with level-based execution

Level 0 (no dependencies):
  [Step 1, Step 2, Step 3]  ← Execute in parallel

Level 1 (depends on Level 0):
  [Step 4]  ← Execute after Level 0 completes

Pseudocode:
┌──────────────────────────────────────────────────┐
│ while unexecuted_steps:                          │
│   ready = steps with all dependencies satisfied  │
│   results = await asyncio.gather(*ready)         │
│   mark ready as executed                         │
│   unlock dependent steps                         │
└──────────────────────────────────────────────────┘
```

---

## Circuit Breaker Pattern

### Problem: Cascading Failures

```
Tool fails → Retry → Tool fails → Retry → ...
    ↓           ↓           ↓
System load increases
Response times degrade
Cascading failures across all operations
```

### Solution: Circuit Breaker

```
┌────────────────────────────────────────────────────┐
│            CIRCUIT BREAKER STATES                  │
│                                                    │
│  ┌─────────┐                     ┌──────────┐    │
│  │ CLOSED  │  ──5 failures──>    │  OPEN    │    │
│  │ Normal  │                     │ Blocking │    │
│  └─────────┘                     └──────────┘    │
│      ↑                                  │         │
│      │                                  │         │
│      │ 3 successes                      │ timeout │
│      │                                  │         │
│  ┌───────────┐                          │         │
│  │HALF-OPEN  │  <──────────────────────┘         │
│  │ Testing   │                                    │
│  └───────────┘                                    │
└────────────────────────────────────────────────────┘

Timeline with Circuit Breaker:

T=0s    Step 1 → Calculator.execute() ❌ Fail
T=1s    Step 1 → Retry → Calculator.execute() ❌ Fail
T=3s    Step 1 → Retry → Calculator.execute() ❌ Fail
T=7s    Step 1 → Retry → Calculator.execute() ❌ Fail
T=15s   Step 1 → Retry → Calculator.execute() ❌ Fail

Circuit Breaker: 5 failures detected → OPEN state

T=16s   Step 2 → Calculator.execute()
        ↓ Circuit OPEN → Fail-fast (no attempt)
        Return: "Calculator temporarily unavailable"

T=76s   Circuit timeout → HALF-OPEN state
T=76s   Step 3 → Calculator.execute() ✅ Success
T=77s   Circuit: 1/3 successes in HALF-OPEN
T=78s   Step 4 → Calculator.execute() ✅ Success
T=79s   Circuit: 2/3 successes in HALF-OPEN
T=80s   Step 5 → Calculator.execute() ✅ Success
T=81s   Circuit: 3/3 successes → CLOSED state (recovered)
```

**Benefits**:
- Prevents wasted retry attempts when service is down
- Allows service time to recover
- Detects recovery automatically
- Protects upstream services from overload

---

## Event Sourcing Architecture

### Traditional State Storage (Current)

```
State Table:
┌─────────┬─────────┬───────────┬─────────┐
│ run_id  │ status  │ result    │ updated │
├─────────┼─────────┼───────────┼─────────┤
│ abc-123 │completed│ 5.0       │ 10:05   │  ← Current state only
└─────────┴─────────┴───────────┴─────────┘

Problem: No audit trail, can't time-travel debug
```

### Event Sourcing (Recommended)

```
Event Log (Immutable, Append-Only):
┌──────────┬─────────┬─────────────────┬───────────┬──────────┐
│ event_id │ run_id  │ event_type      │ timestamp │ data     │
├──────────┼─────────┼─────────────────┼───────────┼──────────┤
│ ev-001   │abc-123  │ run_created     │ 10:00:00  │ {...}    │
│ ev-002   │abc-123  │ plan_generated  │ 10:00:10  │ {...}    │
│ ev-003   │abc-123  │ execution_start │ 10:00:15  │ {...}    │
│ ev-004   │abc-123  │ step_completed  │ 10:00:50  │ {...}    │
│ ev-005   │abc-123  │ step_completed  │ 10:01:20  │ {...}    │
│ ev-006   │abc-123  │ run_completed   │ 10:01:25  │ {...}    │
└──────────┴─────────┴─────────────────┴───────────┴──────────┘

Current State = Fold(Events):
  run_created → Run(status=PENDING)
  plan_generated → Run(status=PENDING, plan=...)
  execution_start → Run(status=RUNNING, started_at=...)
  step_completed → Run(execution_log=[step1])
  step_completed → Run(execution_log=[step1, step2])
  run_completed → Run(status=COMPLETED, completed_at=...)
```

### Reconstruction

```python
def reconstruct_run(event_log: list[Event]) -> Run:
    run = Run(run_id=event_log[0].run_id)

    for event in event_log:
        run = apply_event(run, event)

    return run

# Time-travel debugging
events_at_10_01_00 = filter(events, lambda e: e.timestamp < "10:01:00")
run_state_at_10_01 = reconstruct_run(events_at_10_01_00)
# → See exactly what state was before failure
```

**Benefits**:
- Complete audit trail for compliance
- Time-travel debugging (reconstruct past states)
- Event replay for testing
- Analytics on event patterns

---

## CQRS (Command Query Responsibility Segregation)

### Problem: Read/Write Contention

```
Current (Single Model):
┌─────────────────┐
│ PostgreSQL      │
│                 │
│ Writes: 100/s   │  ← POST /runs (inserts)
│ Reads: 1000/s   │  ← GET /runs/{id} (queries)
│                 │
│ Contention:     │
│ Writes slow ───>│<─── Reads block
└─────────────────┘
```

### Solution: Separate Read/Write Models

```
┌──────────────────────────────────────────────────────┐
│                  COMMAND SIDE (Writes)               │
│                                                      │
│  POST /runs → CommandHandler → Write DB             │
│                     ↓                                │
│               Event Stream                           │
└────────────────────┬─────────────────────────────────┘
                     │
                     │ Projections
                     ▼
┌──────────────────────────────────────────────────────┐
│                   QUERY SIDE (Reads)                 │
│                                                      │
│  GET /runs/{id} → QueryHandler → Read DB            │
│                                   (Materialized View)│
│                                                      │
│  Optimized for reads:                                │
│  • Denormalized data                                │
│  • Pre-computed aggregations                        │
│  • Caching layer (Redis)                            │
└──────────────────────────────────────────────────────┘

Write DB (PostgreSQL):
  • Normalized schema
  • ACID transactions
  • Event sourcing

Read DB (PostgreSQL Materialized View):
  • Denormalized for fast queries
  • Pre-joined data
  • Indexed for common queries

Flow:
1. POST /runs → Write to event store
2. Event handler → Update materialized view
3. GET /runs/{id} → Read from optimized view (10x faster)
```

---

## Saga Pattern for Long-Running Workflows

### Problem: No Rollback Mechanism

```
Workflow: Train Model → Deploy Staging → Test → Deploy Prod

Current:
  Step 1: ✅ Train (2 hours)
  Step 2: ✅ Deploy staging
  Step 3: ❌ Tests fail

Result: Stuck state, no cleanup, resources leaked
```

### Solution: Saga with Compensating Transactions

```
┌────────────────────────────────────────────────────┐
│                  SAGA PATTERN                      │
│                                                    │
│  Forward Flow:                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  Train   │ →  │  Deploy  │ →  │   Test    │   │
│  │  Model   │    │  Staging │    │           │   │
│  └──────────┘    └──────────┘    └──────────┘   │
│       ↓               ↓                ↓          │
│  Compensate:     Compensate:      Compensate:    │
│  Delete Model    Undeploy         N/A            │
│                  Staging                          │
│                                                    │
│  Rollback Flow (on failure):                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ Delete   │ ←  │ Undeploy  │ ← │ Test Fail │   │
│  │ Model    │    │ Staging   │   │           │   │
│  └──────────┘    └──────────┘    └──────────┘   │
└────────────────────────────────────────────────────┘

Timeline:
T=0      Start Saga(train_and_deploy)
T=7200   Train complete → Saga: add compensation(delete_model)
T=7300   Deploy staging → Saga: add compensation(undeploy_staging)
T=7400   Run tests → FAIL
T=7400   Saga: trigger rollback
T=7401   Execute: undeploy_staging() ✅
T=7402   Execute: delete_model() ✅
T=7403   Saga: rollback complete
```

**Implementation**:
```python
class Saga:
    def __init__(self):
        self.steps = []
        self.compensations = []

    async def execute(self):
        try:
            # Forward flow
            for step in self.steps:
                result = await step.execute()
                if step.compensation:
                    self.compensations.append(step.compensation)
        except Exception:
            # Rollback
            for compensation in reversed(self.compensations):
                await compensation()
            raise
```

---

## Observability Architecture

### Current: Basic Logging

```python
logger.info(f"Starting execution")
logger.error(f"Step failed: {error}")
```

### Recommended: Full Observability Stack

```
┌─────────────────────────────────────────────────────┐
│              APPLICATION CODE                       │
│                                                     │
│  • Structured logging (JSON)                        │
│  • Metrics emission (Prometheus)                    │
│  • Distributed tracing (OpenTelemetry)              │
└────────┬───────────────┬───────────────┬───────────┘
         │               │               │
         ▼               ▼               ▼
    ┌────────┐     ┌─────────┐    ┌──────────┐
    │ Logs   │     │ Metrics │    │ Traces   │
    │ (JSON) │     │ (Prom)  │    │ (OTLP)   │
    └───┬────┘     └────┬────┘    └────┬─────┘
        │               │              │
        ▼               ▼              ▼
    ┌────────┐     ┌─────────┐    ┌──────────┐
    │ Loki   │     │Prometheus│   │ Jaeger   │
    └───┬────┘     └────┬────┘    └────┬─────┘
        │               │              │
        └───────────────┴──────────────┘
                        │
                        ▼
                  ┌──────────┐
                  │ Grafana  │  ← Unified Dashboard
                  └──────────┘

Example Log (Structured JSON):
{
  "timestamp": "2025-01-29T10:00:00Z",
  "level": "INFO",
  "message": "execution.started",
  "run_id": "abc-123",
  "correlation_id": "req-456",
  "step_number": 1,
  "tool": "calculator",
  "duration_ms": 50
}

Example Metric:
runs_total{status="completed",tool="calculator"} 1234
runs_duration_seconds{quantile="0.99"} 2.5

Example Trace:
Span: create_run (200ms)
  └─ Span: plan_generation (100ms)
      └─ Span: llm_api_call (80ms)
  └─ Span: execute_run (100ms)
      └─ Span: execute_step (50ms)
          └─ Span: calculator.execute (30ms)
```

---

## Final Comparison: POC vs Production

```
┌────────────────────────────────────────────────────────────┐
│                     POC (Current)                          │
│                                                            │
│  1 FastAPI Process                                         │
│    ↓                                                       │
│  In-Memory State                                           │
│    ↓                                                       │
│  Sequential Execution                                      │
│    ↓                                                       │
│  Basic Logging                                             │
│                                                            │
│  Handles: 1 request/sec                                    │
│  Latency: 100ms                                            │
│  Uptime: 95% (restarts lose state)                         │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│              PRODUCTION (Recommended)                      │
│                                                            │
│  Load Balancer → 10 API Pods                               │
│       ↓                                                    │
│  Redis (Hot State) + PostgreSQL (Cold)                     │
│       ↓                                                    │
│  Queue → 50 Workers                                        │
│       ↓                                                    │
│  DAG Parallel Execution                                    │
│       ↓                                                    │
│  Circuit Breakers + Event Sourcing                         │
│       ↓                                                    │
│  Distributed Tracing + Metrics                             │
│                                                            │
│  Handles: 1000 requests/sec                                │
│  Latency: 50ms (parallel execution)                        │
│  Uptime: 99.9% (redundancy + durability)                   │
└────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Phase 1: Critical (2 weeks)
```
1. Redis for state       ← Enables horizontal scaling
2. Queue for execution   ← Decouples API from work
3. PostgreSQL archive    ← Enables analytics
4. Structured logging    ← Enables debugging
```

### Phase 2: Performance (2 weeks)
```
5. DAG execution         ← 2-3x throughput
6. Circuit breakers      ← Fault isolation
7. Distributed tracing   ← Production debugging
```

### Phase 3: Enterprise (2 weeks)
```
8. Event sourcing        ← Audit compliance
9. CQRS                  ← Read scalability
10. Saga pattern         ← Long workflows
```

---

## Conclusion

The current POC demonstrates solid fundamentals but lacks production-scale architecture patterns. The recommended improvements follow industry-standard distributed systems patterns:

- **Durability**: Redis + PostgreSQL
- **Scalability**: Queue + Worker pool
- **Performance**: DAG execution
- **Reliability**: Circuit breakers
- **Observability**: Logs + Metrics + Traces

These patterns are battle-tested at scale (Netflix, Amazon, Google) and are appropriate for production AI agent systems.
