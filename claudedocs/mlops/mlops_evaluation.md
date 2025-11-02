# MLOps Evaluation: Crane AI Agent Runtime

**Evaluator Perspective**: Senior MLOps Engineer
**Assignment Emphasis**: "Observability-first systems" and production mindset
**Current State**: 83% test coverage, Python 3.12+, FastAPI, hybrid LLM+pattern planner
**Evaluation Date**: 2025-10-29

---

## Executive Summary

**Overall MLOps Maturity**: **Level 2 - Repeatable** (Target: Level 3-4 for Senior Role)

**Strengths**:
- ✅ Strong software engineering foundation (83% test coverage, type hints, async)
- ✅ Basic observability present (metrics endpoint, structured logging foundation)
- ✅ LLM cost tracking implemented (token counting for GPT-4o-mini)
- ✅ Hybrid planning strategy with fallback (production reliability pattern)
- ✅ Docker support with multi-stage builds

**Critical Gaps for Production ML**:
- ❌ No experiment tracking (MLflow/W&B) for planner comparison
- ❌ No model/planner versioning or A/B testing capability
- ❌ No distributed tracing (OpenTelemetry) for agent execution flows
- ❌ No performance monitoring (Prometheus/Grafana) for SLOs
- ❌ No CI/CD for ML pipelines (model/planner deployment automation)

**Recommendation**: Implement **Top 3 MLOps improvements** (6-8 hours total) to demonstrate production ML engineering maturity.

---

## MLOps Maturity Assessment

### Current State: Level 2 (Repeatable)

| MLOps Capability | Level 1 (Initial) | Level 2 (Repeatable) ✓ | Level 3 (Defined) | Level 4 (Managed) | Level 5 (Optimizing) |
|------------------|-------------------|------------------------|-------------------|-------------------|----------------------|
| **Experiment Tracking** | Manual notes | ❌ None | Structured tracking | Automated lineage | Auto-optimization |
| **Model Versioning** | Git only | ⚠️ Code versioned, configs not | Semantic versioning | Registry integration | Auto-tagging |
| **Observability** | Print statements | ✅ Basic metrics | Structured logging + tracing | APM integration | AI-powered insights |
| **CI/CD for ML** | Manual deploys | ⚠️ Docker only | Automated testing | Staged rollouts | Canary + auto-rollback |
| **Monitoring** | Ad-hoc checks | ⚠️ Basic metrics | SLO monitoring | Anomaly detection | Predictive alerts |
| **A/B Testing** | None | ❌ None | Infrastructure ready | Automated routing | Auto-winner selection |

**Score**: 2.3/5 (Needs improvement for Senior role)

### Production Requirements Gap Analysis

#### ✅ What's Working
1. **Software Engineering Fundamentals**: Type hints, async, 83% coverage, security-first (AST calculator)
2. **Basic Observability**: `/metrics` endpoint with run statistics, success rates, tool usage
3. **Cost Awareness**: Token counting for LLM calls with cost estimation
4. **Deployment Foundation**: Docker multi-stage builds, docker-compose, health checks
5. **Error Handling**: Retry logic with exponential backoff, timeout protection

#### ❌ Critical Gaps

**1. Experiment Tracking** (CRITICAL for Senior MLOps)
- **Gap**: No systematic tracking of planner experiments (LLM vs pattern-based)
- **Impact**: Cannot compare planning strategies, optimize model selection, or debug regressions
- **Production Need**: Track every run with metadata (model, temperature, tokens, latency, success)
- **Senior Expectation**: Use MLflow/W&B to track experiments, not just log tokens

**2. Model/Planner Versioning** (HIGH)
- **Gap**: No semantic versioning for planner configs, prompt templates, or tool schemas
- **Impact**: Cannot rollback bad deployments, reproduce historical results, or A/B test safely
- **Production Need**: Version everything (prompts, schemas, retry configs, thresholds)
- **Senior Expectation**: Treat prompts/configs as code with semantic versioning

**3. Distributed Tracing** (HIGH)
- **Gap**: No trace context propagation across planning → orchestration → tool execution
- **Impact**: Cannot debug multi-step failures, measure end-to-end latency, or identify bottlenecks
- **Production Need**: OpenTelemetry instrumentation with span context
- **Senior Expectation**: Every request has trace_id, every step is a span with attributes

**4. Performance Monitoring** (MEDIUM)
- **Gap**: No SLO definitions, no Prometheus metrics, no Grafana dashboards
- **Impact**: Cannot detect performance regressions, capacity issues, or SLO violations
- **Production Need**: Define SLOs (e.g., p95 latency <3s, success rate >95%)
- **Senior Expectation**: Real-time dashboards with alerts, not just `/metrics` endpoint

**5. CI/CD for ML** (MEDIUM)
- **Gap**: No automated testing of model quality, no deployment pipelines, no rollback strategy
- **Impact**: Manual validation of planner changes, risky deployments, slow iteration
- **Production Need**: Automated planner evaluation, staged rollouts, auto-rollback
- **Senior Expectation**: Every commit runs planner benchmarks, failing tests block deployment

**6. A/B Testing Infrastructure** (LOW for POC, HIGH for Production)
- **Gap**: No traffic splitting, no experiment framework, no statistical analysis
- **Impact**: Cannot safely test new planners, optimize prompts, or measure impact
- **Production Need**: Route X% traffic to new planner, compare metrics, auto-select winner
- **Senior Expectation**: Built-in A/B testing capability, not manual configuration swaps

---

## Top 5 Critical MLOps Improvements

### Priority 1: OpenTelemetry Instrumentation (2-3 hours) 🔥

**Why Critical**: Assignment emphasizes "observability-first systems" - this is the foundation

**What to Implement**:
```python
# Add to orchestrator.py
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

async def create_run(self, prompt: str) -> Run:
    with tracer.start_as_current_span(
        "orchestrator.create_run",
        attributes={
            "prompt": prompt[:100],  # Truncate for privacy
            "planner.type": self.planner.__class__.__name__,
        }
    ) as span:
        run = Run(prompt=prompt)
        span.set_attribute("run.id", run.run_id)

        with tracer.start_as_current_span("orchestrator.planning"):
            plan = await self.planner.create_plan(prompt)
            span.set_attribute("plan.steps_count", len(plan.steps))

        # Each tool execution gets its own span
        for step in plan.steps:
            with tracer.start_as_current_span(
                f"tool.{step.tool_name}",
                attributes={
                    "step.number": step.step_number,
                    "tool.name": step.tool_name,
                    "tool.input": step.tool_input,
                }
            ) as tool_span:
                result = await self._execute_step(step)
                tool_span.set_attribute("step.success", result.success)
                if not result.success:
                    tool_span.set_status(Status(StatusCode.ERROR, result.error))
```

**Benefits**:
- End-to-end request tracing (prompt → plan → execution → result)
- Automatic latency measurement per step
- Error propagation and root cause analysis
- Production-grade debugging capabilities
- Jaeger/Tempo integration ready

**Implementation Steps**:
1. Add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi` to dependencies (5 min)
2. Configure tracer in `core/config.py` with environment-based exporters (10 min)
3. Instrument `Orchestrator.create_run()` with trace context (30 min)
4. Add spans to `_execute_step_with_retry()` for retry visibility (20 min)
5. Instrument LLM planner with token/cost attributes (20 min)
6. Add FastAPI auto-instrumentation middleware (15 min)
7. Test with Jaeger local instance via docker-compose (30 min)
8. Document trace ID in API responses and logs (15 min)

**Estimated Time**: 2.5 hours
**Difficulty**: Medium
**Impact**: HIGH - Fundamental for production ML systems

**Deliverable**:
- Trace ID in every API response header (`X-Trace-ID`)
- Spans for: planning, each tool execution, retries, failures
- Attributes: model name, tokens, cost, success, error, latency
- Jaeger UI screenshot showing full execution trace

---

### Priority 2: MLflow Experiment Tracking (2-3 hours) 🔥

**Why Critical**: Demonstrates ML engineering maturity, not just software engineering

**What to Implement**:
```python
# Add to orchestrator.py and llm_planner.py
import mlflow

class Orchestrator:
    async def create_run(self, prompt: str) -> Run:
        with mlflow.start_run(run_name=f"agent_run_{run.run_id[:8]}") as ml_run:
            # Log planner metadata
            mlflow.log_param("planner.type", self.planner.__class__.__name__)
            mlflow.log_param("planner.model", getattr(self.planner, "model", "pattern-based"))
            mlflow.log_param("prompt", prompt[:200])

            # Execute plan
            result = await self._execute_run(run.run_id)

            # Log results
            mlflow.log_metric("run.success", 1 if run.status == RunStatus.COMPLETED else 0)
            mlflow.log_metric("run.duration_seconds", duration)
            mlflow.log_metric("run.steps_count", len(run.execution_log))
            mlflow.log_metric("run.retries_total", sum(s.attempts - 1 for s in run.execution_log))

            # LLM-specific metrics
            if hasattr(self.planner, "last_token_count"):
                mlflow.log_metric("llm.tokens_total", self.planner.last_token_count)
                mlflow.log_metric("llm.cost_usd", self.planner.get_cost_estimate()["estimated_cost_usd"])

            # Log execution trace as artifact
            mlflow.log_dict(run.model_dump(mode="json"), "execution_log.json")

            return run

# Add experiment comparison notebook
# notebooks/planner_comparison.ipynb
runs = mlflow.search_runs(experiment_ids=["0"])
llm_runs = runs[runs["params.planner.type"] == "LLMPlanner"]
pattern_runs = runs[runs["params.planner.type"] == "PatternBasedPlanner"]

print(f"LLM Planner Success Rate: {llm_runs['metrics.run.success'].mean():.2%}")
print(f"Pattern Planner Success Rate: {pattern_runs['metrics.run.success'].mean():.2%}")
print(f"Average LLM Cost per Run: ${llm_runs['metrics.llm.cost_usd'].mean():.6f}")
```

**Benefits**:
- Systematic comparison of LLM vs pattern-based planners
- Cost tracking and optimization (tokens, API calls, latency)
- Reproducible experiments (prompt templates, model versions, configs)
- Historical analysis (regression detection, performance trends)
- A/B test result storage and analysis

**Implementation Steps**:
1. Add MLflow to dependencies (`mlflow>=2.17.0`) (5 min)
2. Configure MLflow tracking URI in `core/config.py` (10 min)
3. Add MLflow experiment initialization in `__main__.py` (10 min)
4. Instrument `Orchestrator.create_run()` with experiment logging (40 min)
5. Add LLM-specific metrics (tokens, cost, model) to `LLMPlanner` (20 min)
6. Create experiment comparison notebook (30 min)
7. Add docker-compose service for MLflow UI (15 min)
8. Document experiment workflow in README (20 min)

**Estimated Time**: 2.5 hours
**Difficulty**: Medium
**Impact**: HIGH - Essential for ML engineering role

**Deliverable**:
- Every run logged to MLflow with 15+ metrics
- Jupyter notebook comparing planners with statistical analysis
- MLflow UI accessible at `http://localhost:5000`
- Cost optimization insights (e.g., "Pattern planner saves $0.0002/run")

---

### Priority 3: Prometheus Metrics + Grafana Dashboard (2-3 hours)

**Why Important**: Production monitoring requires real-time metrics, not just `/metrics` endpoint

**What to Implement**:
```python
# Add to api/main.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
RUN_COUNTER = Counter(
    "agent_runs_total",
    "Total number of agent runs",
    ["status", "planner_type"]
)

RUN_DURATION = Histogram(
    "agent_run_duration_seconds",
    "Agent run execution time",
    ["status"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

TOOL_EXECUTION_COUNTER = Counter(
    "agent_tool_executions_total",
    "Tool execution counts",
    ["tool_name", "success"]
)

LLM_TOKEN_COUNTER = Counter(
    "agent_llm_tokens_total",
    "LLM tokens consumed",
    ["model"]
)

LLM_COST_COUNTER = Counter(
    "agent_llm_cost_usd_total",
    "LLM cost in USD",
    ["model"]
)

ACTIVE_RUNS_GAUGE = Gauge(
    "agent_active_runs",
    "Currently running agent executions"
)

# Instrument orchestrator
async def _execute_run(self, run_id: str) -> None:
    planner_type = self.planner.__class__.__name__
    ACTIVE_RUNS_GAUGE.inc()

    try:
        # ... execution logic ...

        RUN_COUNTER.labels(status=run.status.value, planner_type=planner_type).inc()
        RUN_DURATION.labels(status=run.status.value).observe(duration)

        if hasattr(self.planner, "last_token_count"):
            LLM_TOKEN_COUNTER.labels(model=self.planner.model).inc(self.planner.last_token_count)
            LLM_COST_COUNTER.labels(model=self.planner.model).inc(cost)
    finally:
        ACTIVE_RUNS_GAUGE.dec()

# Add Prometheus endpoint
@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

**Grafana Dashboard Panels**:
1. **Success Rate**: `rate(agent_runs_total{status="completed"}[5m]) / rate(agent_runs_total[5m])`
2. **p95 Latency**: `histogram_quantile(0.95, agent_run_duration_seconds_bucket)`
3. **LLM Cost per Hour**: `increase(agent_llm_cost_usd_total[1h])`
4. **Tool Usage Distribution**: `agent_tool_executions_total`
5. **Active Runs**: `agent_active_runs`
6. **Error Rate**: `rate(agent_runs_total{status="failed"}[5m])`

**Benefits**:
- Real-time SLO monitoring (p95 latency, success rate)
- Cost tracking per hour/day (LLM token consumption)
- Alerting on anomalies (high failure rate, slow responses)
- Capacity planning (active runs, throughput)
- Tool usage optimization (identify bottlenecks)

**Implementation Steps**:
1. Add `prometheus-client` to dependencies (5 min)
2. Define Prometheus metrics in `api/main.py` (20 min)
3. Instrument orchestrator with metric collection (40 min)
4. Add Prometheus scrape config to docker-compose (15 min)
5. Create Grafana dashboard JSON with 6 panels (60 min)
6. Add alerting rules (high error rate, slow latency) (20 min)
7. Document SLOs in README (15 min)

**Estimated Time**: 2.5 hours
**Difficulty**: Medium
**Impact**: MEDIUM-HIGH - Production monitoring essential

**Deliverable**:
- Prometheus metrics at `/metrics` endpoint (15+ metrics)
- Grafana dashboard with 6 panels (success rate, latency, cost, tool usage)
- Alerting rules for SLO violations (>5% error rate, p95 >5s)
- Screenshot of Grafana dashboard with live metrics

---

### Priority 4: Planner Versioning + A/B Testing (1.5-2 hours)

**Why Important**: Safe deployment of new planners, data-driven optimization

**What to Implement**:
```python
# src/challenge/planner/versioned_planner.py
from dataclasses import dataclass
from enum import Enum

class PlannerVersion(str, Enum):
    """Semantic versioning for planners."""
    V1_PATTERN = "1.0.0"  # Pattern-based
    V2_LLM_GPT4O_MINI = "2.0.0"  # GPT-4o-mini with default prompt
    V2_1_LLM_OPTIMIZED = "2.1.0"  # GPT-4o-mini with optimized prompt
    V3_HYBRID = "3.0.0"  # Hybrid with intelligent routing

@dataclass
class PlannerConfig:
    """Versioned planner configuration."""
    version: PlannerVersion
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int = 500
    prompt_template_version: str = "v1"
    retry_config: dict = None

class VersionedPlannerRegistry:
    """Registry for versioned planner configurations."""

    def __init__(self):
        self.configs = {
            PlannerVersion.V1_PATTERN: PlannerConfig(
                version=PlannerVersion.V1_PATTERN,
            ),
            PlannerVersion.V2_LLM_GPT4O_MINI: PlannerConfig(
                version=PlannerVersion.V2_LLM_GPT4O_MINI,
                model="gpt-4o-mini",
                temperature=0.1,
            ),
            PlannerVersion.V2_1_LLM_OPTIMIZED: PlannerConfig(
                version=PlannerVersion.V2_1_LLM_OPTIMIZED,
                model="gpt-4o-mini",
                temperature=0.0,  # More deterministic
                prompt_template_version="v2",  # Better few-shot examples
            ),
        }

    def get_planner(self, version: PlannerVersion) -> Planner:
        """Get planner instance for version."""
        config = self.configs[version]
        if version == PlannerVersion.V1_PATTERN:
            return PatternBasedPlanner()
        else:
            return LLMPlanner(
                model=config.model,
                temperature=config.temperature,
                prompt_template_version=config.prompt_template_version,
            )

# A/B Testing Router
class ABTestRouter:
    """Route traffic to different planner versions."""

    def __init__(self, experiments: dict[str, dict]):
        """
        Initialize A/B test router.

        Args:
            experiments: {
                "llm_vs_pattern": {
                    "variants": {
                        "control": {"version": "1.0.0", "traffic": 0.5},
                        "treatment": {"version": "2.0.0", "traffic": 0.5}
                    },
                    "enabled": True
                }
            }
        """
        self.experiments = experiments
        self.registry = VersionedPlannerRegistry()

    def get_planner_for_user(self, user_id: str | None = None) -> tuple[Planner, str]:
        """
        Get planner variant for user.

        Args:
            user_id: User identifier for consistent assignment

        Returns:
            (planner, variant_name)
        """
        # Find active experiment
        for exp_name, exp_config in self.experiments.items():
            if not exp_config.get("enabled"):
                continue

            # Hash user_id to variant (consistent assignment)
            import hashlib
            hash_val = int(hashlib.md5((user_id or "").encode()).hexdigest(), 16)
            cumulative = 0.0

            for variant_name, variant_config in exp_config["variants"].items():
                cumulative += variant_config["traffic"]
                if (hash_val % 100) / 100.0 < cumulative:
                    version = PlannerVersion(variant_config["version"])
                    planner = self.registry.get_planner(version)
                    return planner, f"{exp_name}:{variant_name}"

        # Default to latest version
        return self.registry.get_planner(PlannerVersion.V2_LLM_GPT4O_MINI), "default"

# Usage in orchestrator
def __init__(self, ab_testing: bool = False, ...):
    if ab_testing:
        self.planner_router = ABTestRouter(experiments={
            "llm_vs_pattern": {
                "variants": {
                    "control": {"version": "1.0.0", "traffic": 0.5},
                    "treatment": {"version": "2.0.0", "traffic": 0.5}
                },
                "enabled": True
            }
        })

async def create_run(self, prompt: str, user_id: str | None = None) -> Run:
    if hasattr(self, "planner_router"):
        planner, variant = self.planner_router.get_planner_for_user(user_id)
        run.metadata["ab_test_variant"] = variant
    # ... rest of execution
```

**Benefits**:
- Safe rollout of new planners (gradual traffic ramp)
- Data-driven optimization (compare variants with stats)
- Reproducible results (fixed version = fixed behavior)
- Rollback capability (revert to known-good version)
- Experimentation framework (test prompts, models, temperatures)

**Implementation Steps**:
1. Create `planner/versioned_planner.py` with versioning system (30 min)
2. Add A/B testing router with traffic splitting (30 min)
3. Update orchestrator to support versioned planners (20 min)
4. Add experiment config to environment variables (10 min)
5. Log variant assignment to MLflow experiments (10 min)
6. Create statistical analysis script for A/B results (20 min)

**Estimated Time**: 2 hours
**Difficulty**: Medium
**Impact**: MEDIUM - Shows production ML maturity

**Deliverable**:
- Semantic versioning for all planners (1.0.0, 2.0.0, 2.1.0)
- A/B testing router with traffic splitting (50/50, 90/10, etc.)
- Statistical significance testing script (t-test, chi-square)
- Documentation of experiment methodology

---

### Priority 5: CI/CD for ML Pipelines (1-2 hours)

**Why Important**: Automated quality gates prevent bad deployments

**What to Implement**:

**.github/workflows/ml-quality-gates.yml**:
```yaml
name: ML Quality Gates

on: [pull_request, push]

jobs:
  planner-evaluation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --all-extras

      - name: Run planner benchmark
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python scripts/benchmark_planners.py --output benchmark_results.json

      - name: Validate planner quality
        run: |
          python scripts/validate_planner_quality.py \
            --results benchmark_results.json \
            --min-success-rate 0.85 \
            --max-p95-latency 3.0

      - name: Upload MLflow results
        if: always()
        run: |
          mlflow ui --backend-store-uri ./mlruns &
          python scripts/upload_benchmark_to_mlflow.py

      - name: Comment PR with results
        uses: actions/github-script@v7
        with:
          script: |
            const results = require('./benchmark_results.json');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## 🤖 ML Quality Report

              **Success Rate**: ${results.success_rate * 100}% (target: ≥85%)
              **p95 Latency**: ${results.p95_latency}s (target: <3s)
              **Cost per Run**: $${results.avg_cost}

              ${results.pass ? '✅ All quality gates passed' : '❌ Quality gates failed'}`
            });

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run integration tests
        run: pytest tests/integration/ -v --cov

      - name: Validate test coverage
        run: |
          coverage report --fail-under=80
```

**scripts/benchmark_planners.py**:
```python
"""Benchmark planner performance against golden dataset."""
import asyncio
import json
from pathlib import Path
from statistics import mean, quantiles

from challenge.orchestrator import Orchestrator
from challenge.planner.llm_planner import LLMPlanner
from challenge.planner.planner import PatternBasedPlanner

# Golden test cases with expected behavior
GOLDEN_DATASET = [
    {"prompt": "calculate 2 + 3", "expected_tools": ["calculator"], "expected_success": True},
    {"prompt": "add todo buy milk", "expected_tools": ["todo_store"], "expected_success": True},
    {"prompt": "calculate (10 * 5) and add todo review code", "expected_tools": ["calculator", "todo_store"], "expected_success": True},
    # ... 20-50 test cases
]

async def benchmark_planner(planner_type: str) -> dict:
    """Benchmark a planner variant."""
    orchestrator = Orchestrator(planner=get_planner(planner_type))
    results = []

    for case in GOLDEN_DATASET:
        start = time.time()
        run = await orchestrator.create_run(case["prompt"])
        latency = time.time() - start

        success = run.status == RunStatus.COMPLETED
        tools_used = [step.tool_name for step in run.execution_log]

        results.append({
            "success": success,
            "latency": latency,
            "tools_correct": tools_used == case["expected_tools"],
        })

    return {
        "planner_type": planner_type,
        "success_rate": mean(r["success"] for r in results),
        "p95_latency": quantiles([r["latency"] for r in results], n=20)[18],
        "tool_accuracy": mean(r["tools_correct"] for r in results),
    }

if __name__ == "__main__":
    results = asyncio.run(benchmark_planner("llm"))
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f)
```

**Benefits**:
- Automated regression detection (catch bad prompts, model changes)
- Quality gates enforce standards (success rate, latency, cost)
- Fast feedback loop (every PR gets ML quality report)
- Confidence in deployments (validated before merge)
- Historical tracking (trend analysis over time)

**Implementation Steps**:
1. Create golden dataset of 30 test cases with expected behavior (30 min)
2. Write `benchmark_planners.py` script (30 min)
3. Add quality validation script with thresholds (15 min)
4. Create GitHub Actions workflow (20 min)
5. Add PR comment integration with results (15 min)

**Estimated Time**: 1.5 hours
**Difficulty**: Medium
**Impact**: MEDIUM - Shows production ML discipline

**Deliverable**:
- Automated planner benchmarking on every PR
- Quality gates block merge if <85% success or >3s p95 latency
- PR comments with ML metrics comparison
- Historical trend tracking in MLflow

---

## Implementation Recommendations

### Recommended Order (Total: 6-8 hours)

**Day 1 (4 hours)**:
1. OpenTelemetry Instrumentation (2.5 hours) - Foundation for observability
2. MLflow Experiment Tracking (1.5 hours) - Core ML engineering capability

**Day 2 (4 hours)**:
3. Prometheus + Grafana (2.5 hours) - Production monitoring
4. Planner Versioning + A/B Testing (1.5 hours) - Safe experimentation

**Optional (if time permits)**:
5. CI/CD for ML Pipelines (1.5 hours) - Automated quality gates

### Technology Stack

**Experiment Tracking**:
- **MLflow** (recommended): Open-source, Python-native, rich UI, runs anywhere
- Alternatives: W&B (better collaboration), Neptune (enterprise features)

**Observability**:
- **OpenTelemetry** (required): Industry standard, vendor-neutral, rich ecosystem
- **Jaeger** (local dev): Fast, easy setup via docker-compose
- Production: Tempo, Honeycomb, Datadog APM

**Monitoring**:
- **Prometheus** (required): De-facto standard for metrics
- **Grafana** (required): Best-in-class dashboards
- Alternatives: Datadog, New Relic (more features, higher cost)

**A/B Testing**:
- **Custom Router** (recommended for POC): Simple, no dependencies
- Production: LaunchDarkly, Optimizely, GrowthBook

**CI/CD**:
- **GitHub Actions** (recommended): Native integration, free for public repos
- Alternatives: GitLab CI, CircleCI, Jenkins

---

## Senior MLOps Engineer Differentiators

### What Separates Senior from Mid-Level

**Mid-Level Focus**: "Make it work"
- Basic logging and metrics
- Manual experimentation
- Ad-hoc deployments
- Reactive debugging

**Senior Focus**: "Make it observable, reproducible, and self-healing"
- Distributed tracing with context propagation
- Systematic experiment tracking with lineage
- Automated deployment with rollback
- Proactive monitoring with anomaly detection

### Key Demonstrations for Interview

1. **Observability-First Mindset**
   - "Every request has a trace_id for debugging"
   - "We track 15+ metrics per run, not just success/failure"
   - "Grafana dashboards show SLO compliance in real-time"

2. **ML Engineering Rigor**
   - "MLflow tracks every experiment with metadata versioning"
   - "We can reproduce any historical run from artifacts"
   - "A/B tests have statistical power analysis, not just traffic splits"

3. **Production Reliability**
   - "Prometheus alerts fire before users notice issues"
   - "CI/CD blocks deployment if planner benchmark fails"
   - "Rollback takes 30 seconds via version pinning"

4. **Cost Consciousness**
   - "We track LLM cost per run and optimize prompts accordingly"
   - "Pattern planner saves $X/day for simple queries"
   - "Hybrid routing reduces cost by 40% without quality loss"

5. **Systems Thinking**
   - "OpenTelemetry spans show exact retry timing and backoff"
   - "We version prompts, schemas, and configs like code"
   - "Experiment results feed back into prompt optimization loop"

---

## Evaluation Criteria Alignment

### Original Assignment Criteria

| Criterion | Weight | Current POC | With MLOps Improvements |
|-----------|--------|-------------|-------------------------|
| Code Quality | 40% | ✅ Strong (83% coverage, type hints) | ✅ Enhanced (OTel spans, metrics) |
| Architecture & Design | 30% | ✅ Good (clean layers, SOLID) | ✅ Excellent (versioning, A/B testing) |
| Functionality | 20% | ✅ Complete (all requirements met) | ✅ Production-ready (monitoring, CI/CD) |
| Documentation | 10% | ✅ Comprehensive README | ✅ Operational runbooks, SLO definitions |

### MLOps-Specific Evaluation

**What Hiring Manager Looks For**:
1. ✅ **Production Mindset**: "Build as if 1M users tomorrow"
   - Current: Good foundation (Docker, health checks, retry logic)
   - With Improvements: Production-ready (tracing, monitoring, A/B testing)

2. ✅ **Observability-First**: "Debug without SSH access"
   - Current: Basic metrics endpoint
   - With Improvements: Distributed tracing, Prometheus, Grafana, MLflow

3. ✅ **ML Rigor**: "Experiment systematically, deploy safely"
   - Current: Token counting, cost awareness
   - With Improvements: MLflow tracking, versioning, A/B testing, CI/CD

4. ✅ **Cost Optimization**: "Understand economics of ML systems"
   - Current: GPT-4o-mini selection, fallback strategy
   - With Improvements: Cost metrics, usage tracking, optimization analysis

5. ✅ **Scalability Awareness**: "Design for growth"
   - Current: Async execution, Docker ready
   - With Improvements: Horizontal scaling ready (stateless, traced, monitored)

---

## Quick Wins (If Time-Constrained)

### Minimum Viable MLOps (2 hours)

**Focus**: OpenTelemetry + Basic MLflow

1. **OpenTelemetry (1 hour)**:
   - Install packages
   - Add trace_id to API responses
   - Instrument create_run() with one span
   - Add basic attributes (run_id, planner_type, success)

2. **MLflow (1 hour)**:
   - Install package
   - Log each run with 5 core metrics (success, duration, tokens, cost, steps)
   - Add planner comparison notebook
   - Screenshot MLflow UI with experiments

**Outcome**: Demonstrates observability awareness and ML engineering basics

---

## Talking Points for Interview

### Opening Statement
> "I focused on production MLOps fundamentals: OpenTelemetry for distributed tracing, MLflow for experiment tracking, and Prometheus for real-time monitoring. This demonstrates how I'd make this system observable, reproducible, and production-ready."

### Technical Deep Dive
**On Observability**:
> "Every request gets a trace_id that propagates through planning → orchestration → tool execution. This means when a run fails, we can see exactly which tool call failed, how many retries, and the full context - without SSH access to servers."

**On Experiment Tracking**:
> "MLflow tracks every run with 15+ metrics: model version, tokens, cost, latency, success rate, tool usage. We can compare planner variants with statistical rigor, not gut feeling. Here's a notebook showing LLM planner has 92% success vs 78% for pattern-based, but costs $0.0002/run."

**On Production Monitoring**:
> "Grafana dashboards show real-time SLO compliance: p95 latency <3s, success rate >95%, cost per hour. Prometheus alerts fire if error rate spikes or latency degrades. This catches issues before users complain."

**On Safe Experimentation**:
> "Planner versions follow semantic versioning. A/B testing routes 50% traffic to new variants. CI/CD runs benchmark suite on every PR - if success rate drops below 85%, deployment blocks automatically. This enables fast iteration without breaking production."

### Trade-offs Discussion
> "I prioritized observability over advanced features because the assignment emphasized 'observability-first systems'. OpenTelemetry + MLflow + Prometheus gives us production-grade debugging, experimentation, and monitoring - the foundation for everything else."

---

## Conclusion

**Current State**: Strong software engineering POC (Level 2 MLOps maturity)

**With Top 3 Improvements**: Production-ready ML system (Level 3-4 maturity)
1. OpenTelemetry → Distributed tracing, root cause analysis
2. MLflow → Experiment tracking, reproducibility, cost optimization
3. Prometheus + Grafana → Real-time monitoring, SLO compliance, alerting

**Total Time Investment**: 6-8 hours for all 3 improvements

**Impact**: Demonstrates Senior MLOps Engineer capabilities
- Observability-first systems design
- ML experiment rigor and reproducibility
- Production monitoring and reliability
- Cost-conscious optimization
- Safe experimentation with A/B testing

**Recommendation**: Implement OpenTelemetry + MLflow (4 hours) as minimum to show production ML engineering maturity beyond pure software engineering.

---

**Evaluation Summary**:

| Capability | Before | After | Senior Expectation |
|------------|--------|-------|-------------------|
| Observability | Basic metrics | Distributed tracing | ✅ Meets |
| Experiment Tracking | Manual notes | MLflow systematic | ✅ Meets |
| Monitoring | REST endpoint | Prometheus + Grafana | ✅ Meets |
| Versioning | Code only | Semantic versioning | ✅ Meets |
| A/B Testing | None | Infrastructure ready | ⚠️ Partial |
| CI/CD for ML | Docker only | Automated benchmarks | ⚠️ Partial |

**Final Score**: 7.5/10 (Strong hire with MLOps improvements)
