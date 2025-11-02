# ML Engineering Evaluation: Crane AI Agent Runtime

**Evaluator Perspective**: Senior ML Engineer (Production ML Systems)
**Evaluation Date**: 2025-01-29
**Current State**: Hybrid planner (LLM + Pattern-based fallback) ✅
**Test Coverage**: 83% ✅
**Architecture**: Clean 4-layer separation ✅

---

## Executive Summary

**Overall Assessment**: **Strong foundation with excellent production patterns already in place**

The candidate has demonstrated solid ML engineering fundamentals by implementing:
- ✅ **LLM Integration**: GPT-4o-mini with structured outputs (JSON schema enforcement)
- ✅ **Fallback Strategy**: Graceful degradation to pattern-based planner
- ✅ **Cost Tracking**: Token counting and cost estimation
- ✅ **Production Patterns**: Low temperature (0.1), async support, error handling

**Key Strengths**:
1. **Structured Outputs**: Using OpenAI's JSON schema mode (strict enforcement)
2. **Hybrid Approach**: LLM + rule-based fallback shows production thinking
3. **Observability**: Token tracking, logging, and cost estimation built-in
4. **Clean Architecture**: Tool abstraction allows easy extension

**Critical Gaps** (Expected at Senior Level):
1. **No Model Monitoring**: Missing latency, success rate, error tracking
2. **No Caching Strategy**: Every request hits API (high cost + latency)
3. **No Prompt Engineering**: Basic system prompt, no few-shot examples
4. **No Model Selection Logic**: Single model, no dynamic routing
5. **No Feature Engineering**: Doesn't learn from execution history

**Recommendation**: This is **Tier 2 work** (solid implementation). To reach **Tier 1 (Senior)**, candidate should demonstrate advanced ML production skills in the interview discussion.

---

## 1. Current ML Integration Assessment

### What's Implemented ✅

**LLM Planner** (`llm_planner.py`):
```python
class LLMPlanner:
    """Production-grade patterns already present"""

    # ✅ Structured outputs with JSON schema
    response_format={"type": "json_schema", "json_schema": PLAN_SCHEMA}

    # ✅ Low temperature for consistency
    temperature=0.1

    # ✅ Automatic fallback on failure
    except Exception:
        return self.fallback.create_plan(prompt)

    # ✅ Cost tracking
    self.last_token_count = response.usage.total_tokens

    # ✅ Async support
    async def create_plan(self, prompt: str) -> Plan
```

**Strengths**:
- **Reliability**: Structured outputs prevent malformed JSON (OpenAI strict mode)
- **Cost Efficiency**: GPT-4o-mini ($0.15/1M tokens) is appropriate for planning
- **Resilience**: Fallback chain ensures system never fails due to API issues
- **Observability**: Token tracking enables cost monitoring

**What's Missing** (Senior-level expectations):
- ❌ **Prompt Engineering**: No few-shot examples, no chain-of-thought
- ❌ **Caching**: Repeat prompts hit API every time
- ❌ **Monitoring**: No latency tracking, error categorization, success metrics
- ❌ **Model Selection**: No logic to choose between models based on complexity
- ❌ **Batch Processing**: No batching for concurrent requests
- ❌ **Rate Limiting**: No circuit breaker or exponential backoff for API calls

---

## 2. Top 5 ML Engineering Improvements (Prioritized by Impact)

### **Priority 1: Semantic Caching Layer** (Highest Impact, 60 min)
**Problem**: Every request costs money and adds latency, even for similar prompts.

**Current Cost**:
- GPT-4o-mini: ~200 tokens/request @ $0.15/1M tokens = $0.00003/request
- At 10,000 requests/day: $3.65/year (low but wasteful)
- **Latency**: 200-500ms per request (bigger issue)

**Solution**: Implement semantic caching with embedding similarity

```python
from openai import AsyncOpenAI
import numpy as np
from functools import lru_cache

class SemanticCache:
    """
    Cache LLM responses based on semantic similarity of prompts.

    Uses embeddings to detect similar prompts, serving cached responses
    for semantically equivalent requests.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.client = AsyncOpenAI()
        self.cache: dict[str, tuple[np.ndarray, Plan]] = {}  # prompt_hash -> (embedding, plan)
        self.similarity_threshold = similarity_threshold

    async def get_or_generate(
        self,
        prompt: str,
        generate_fn: Callable[[str], Awaitable[Plan]]
    ) -> tuple[Plan, bool]:
        """
        Get cached plan or generate new one.

        Returns:
            (plan, cache_hit: bool)
        """
        # Generate embedding for prompt (cheap: ~$0.0001 per 1M tokens)
        embedding = await self._get_embedding(prompt)

        # Check for semantic match in cache
        for cached_prompt, (cached_embedding, cached_plan) in self.cache.items():
            similarity = self._cosine_similarity(embedding, cached_embedding)
            if similarity >= self.similarity_threshold:
                logger.info(f"Cache HIT (similarity: {similarity:.3f})")
                return cached_plan, True

        # Cache MISS - generate new plan
        logger.info("Cache MISS - generating plan")
        plan = await generate_fn(prompt)
        self.cache[prompt] = (embedding, plan)
        return plan, False

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get text-embedding-3-small embedding (1536 dims)."""
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Integration**:
```python
class LLMPlanner:
    def __init__(self, ...):
        self.semantic_cache = SemanticCache(similarity_threshold=0.95)

    async def create_plan(self, prompt: str) -> Plan:
        # Try cache first
        plan, cache_hit = await self.semantic_cache.get_or_generate(
            prompt=prompt,
            generate_fn=self._generate_plan_with_llm
        )

        if cache_hit:
            self.last_token_count = 0  # No LLM call

        return plan
```

**Expected Results**:
- **Cache Hit Rate**: 40-60% for typical workloads (similar prompts)
- **Latency Reduction**: 200-500ms → 10-20ms (25x faster on cache hits)
- **Cost Reduction**: 40-60% savings on LLM calls
- **Demo Value**: Shows production ML thinking (cost + latency optimization)

**Why This First**: Biggest bang for buck - improves both user experience and cost efficiency.

---

### **Priority 2: Model Performance Monitoring** (Critical for Production, 45 min)
**Problem**: No visibility into model behavior, errors, or degradation.

**What's Missing**:
- Latency tracking (p50, p95, p99)
- Success/failure rates by error type
- Token usage trends
- Fallback frequency
- Plan quality metrics

**Solution**: Comprehensive metrics collection

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import statistics

@dataclass
class PlanningMetrics:
    """Track LLM planner performance metrics."""

    # Latency metrics (milliseconds)
    latencies: list[float] = field(default_factory=list)

    # Success tracking
    total_requests: int = 0
    llm_successes: int = 0
    fallback_used: int = 0
    failures: int = 0

    # Token usage
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Error categorization
    errors_by_type: dict[str, int] = field(default_factory=dict)

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0

    def record_request(
        self,
        latency_ms: float,
        success: bool,
        source: Literal["llm", "fallback", "cache"],
        tokens: int = 0,
        error_type: str | None = None
    ):
        """Record a planning request."""
        self.total_requests += 1
        self.latencies.append(latency_ms)

        if success:
            if source == "llm":
                self.llm_successes += 1
                self.total_tokens += tokens
                self.total_cost_usd += (tokens / 1_000_000) * 0.15
            elif source == "fallback":
                self.fallback_used += 1
            elif source == "cache":
                self.cache_hits += 1
        else:
            self.failures += 1
            if error_type:
                self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def get_summary(self) -> dict:
        """Get metrics summary."""
        return {
            "latency": {
                "p50_ms": statistics.median(self.latencies) if self.latencies else 0,
                "p95_ms": statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) >= 20 else 0,
                "p99_ms": statistics.quantiles(self.latencies, n=100)[98] if len(self.latencies) >= 100 else 0,
                "avg_ms": statistics.mean(self.latencies) if self.latencies else 0,
            },
            "success_rate": self.llm_successes / self.total_requests if self.total_requests else 0,
            "fallback_rate": self.fallback_used / self.total_requests if self.total_requests else 0,
            "cache_hit_rate": self.cache_hits / self.total_requests if self.total_requests else 0,
            "cost": {
                "total_tokens": self.total_tokens,
                "total_cost_usd": self.total_cost_usd,
                "avg_tokens_per_request": self.total_tokens / self.llm_successes if self.llm_successes else 0,
            },
            "errors": {
                "total": self.failures,
                "by_type": self.errors_by_type,
            }
        }


class LLMPlanner:
    def __init__(self, ...):
        self.metrics = PlanningMetrics()

    async def create_plan(self, prompt: str) -> Plan:
        start = time.perf_counter()

        try:
            # Try cache
            plan, cache_hit = await self.semantic_cache.get_or_generate(...)

            latency_ms = (time.perf_counter() - start) * 1000

            if cache_hit:
                self.metrics.record_request(
                    latency_ms=latency_ms,
                    success=True,
                    source="cache"
                )
            else:
                self.metrics.record_request(
                    latency_ms=latency_ms,
                    success=True,
                    source="llm",
                    tokens=self.last_token_count
                )

            return plan

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            # Try fallback
            try:
                plan = self.fallback.create_plan(prompt)
                self.metrics.record_request(
                    latency_ms=latency_ms,
                    success=True,
                    source="fallback"
                )
                return plan
            except Exception as fallback_error:
                self.metrics.record_request(
                    latency_ms=latency_ms,
                    success=False,
                    source="llm",
                    error_type=type(e).__name__
                )
                raise
```

**New API Endpoint** (`api/routes/metrics.py`):
```python
@router.get("/metrics/planner")
async def get_planner_metrics(orchestrator: Orchestrator = Depends(get_orchestrator)):
    """Get LLM planner performance metrics."""
    return orchestrator.planner.metrics.get_summary()
```

**Expected Results**:
- **Alerting**: Detect when fallback rate > 10% (API issues)
- **Cost Tracking**: Monitor spend trends, set budget alerts
- **Latency SLOs**: Ensure p95 < 500ms for user experience
- **Error Analysis**: Identify and fix common failure modes
- **Demo Value**: Shows production ops experience

**Why This Second**: Critical for operating ML systems reliably - can't manage what you don't measure.

---

### **Priority 3: Advanced Prompt Engineering** (Quality Improvement, 90 min)
**Problem**: Current system prompt is basic, no examples, no reasoning guidance.

**Current Prompt** (Basic):
```
You are a task planning agent. Convert user requests into structured execution plans.

Available Tools: [list]

Output Format: [schema]

Rules: [4 simple rules]
```

**Solution**: Multi-stage prompt with few-shot examples and chain-of-thought

```python
class LLMPlanner:
    def _system_prompt(self) -> str:
        """Production-grade system prompt with examples."""
        return """You are an expert task planning agent that decomposes user requests into optimal execution plans.

Your goal: Create the SIMPLEST plan that fully accomplishes the user's goal.

## Available Tools

### 1. calculator
**Purpose**: Evaluate arithmetic expressions
**Input Schema**:
  - expression (string): Math expression using +, -, *, /, (), e.g., "(10 + 5) * 3"
**Capabilities**: Basic arithmetic only (no sqrt, sin, etc.)
**Example**:
  Input: {"expression": "((100 - 20) * 2) / 4"}
  Output: 40.0

### 2. todo_store
**Purpose**: Manage todo items with CRUD operations
**Input Schema**:
  - action (string): One of ["add", "list", "get", "complete", "delete"]
  - text (string, optional): Todo text (required for "add")
  - todo_id (string, optional): Todo UUID (required for "get", "complete", "delete")
**State**: In-memory, persists within session
**Examples**:
  Add:      {"action": "add", "text": "Review PR #123"}
  List:     {"action": "list"}
  Complete: {"action": "complete", "todo_id": "550e8400-e29b-41d4-a716-446655440000"}

## Planning Strategy

Follow this reasoning process:
1. **Understand Goal**: What is the user's ultimate objective?
2. **Decompose**: What are the minimal steps to achieve it?
3. **Order Steps**: What dependencies exist? (e.g., need ID before completing todo)
4. **Select Tools**: Which tool best serves each step?
5. **Validate**: Can this plan actually accomplish the goal?

## Example Plans

### Example 1: Simple Calculation
**User**: "what is 15% of 200?"
**Reasoning**: Need to calculate 200 * 0.15
**Plan**:
```json
{
  "steps": [
    {
      "step_number": 1,
      "tool_name": "calculator",
      "tool_input": {"expression": "200 * 0.15"},
      "reasoning": "Calculate 15% of 200 (200 * 0.15 = 30)"
    }
  ],
  "final_goal": "Calculate 15% of 200"
}
```

### Example 2: Multi-step Todo Management
**User**: "add a todo to call dentist and then show me my tasks"
**Reasoning**:
  1. First add the todo (creates new item)
  2. Then list all todos (shows result including new one)
**Plan**:
```json
{
  "steps": [
    {
      "step_number": 1,
      "tool_name": "todo_store",
      "tool_input": {"action": "add", "text": "call dentist"},
      "reasoning": "Create new todo item for calling dentist"
    },
    {
      "step_number": 2,
      "tool_name": "todo_store",
      "tool_input": {"action": "list"},
      "reasoning": "Display all todos including the newly created one"
    }
  ],
  "final_goal": "Add dentist reminder and view all tasks"
}
```

### Example 3: Complex Calculation
**User**: "calculate the average of 45, 67, and 23"
**Reasoning**: Average = sum / count = (45 + 67 + 23) / 3
**Plan**:
```json
{
  "steps": [
    {
      "step_number": 1,
      "tool_name": "calculator",
      "tool_input": {"expression": "(45 + 67 + 23) / 3"},
      "reasoning": "Calculate average: sum all numbers and divide by count"
    }
  ],
  "final_goal": "Calculate average of three numbers"
}
```

## Output Requirements

Return valid JSON matching this exact schema:
- steps: Array of step objects (step_number, tool_name, tool_input, reasoning)
- final_goal: Clear description of what the plan accomplishes

## Quality Guidelines

✅ DO:
- Use specific tool inputs (exact action names, complete expressions)
- Provide clear reasoning explaining WHY each step is needed
- Order steps logically (list comes AFTER add if showing new item)
- Keep plans minimal (don't add unnecessary steps)

❌ DON'T:
- Invent new tools or actions not listed above
- Create circular dependencies
- Add steps that don't contribute to the goal
- Use vague reasoning like "perform action"

Now convert the user's request into an execution plan."""
```

**Expected Results**:
- **Accuracy**: 95%+ correct plans (vs ~80% with basic prompt)
- **Edge Cases**: Better handling of ambiguous requests
- **Reasoning Quality**: Clear, specific explanations
- **Fewer Retries**: Structured examples reduce validation failures
- **Demo Value**: Shows prompt engineering expertise

**Why This Third**: Improves quality significantly with minimal infrastructure changes.

---

### **Priority 4: Intelligent Model Routing** (Cost Optimization, 60 min)
**Problem**: Using same model (GPT-4o-mini) for all requests regardless of complexity.

**Current Cost Structure**:
- Simple prompts: "calculate 2+3" → 150 tokens @ $0.15/1M = $0.000023
- Complex prompts: "add 3 todos then list" → 400 tokens @ $0.15/1M = $0.00006
- **Waste**: Overpaying for simple requests by ~2-3x

**Solution**: Route based on complexity heuristics

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    cost_per_1m_tokens: float
    max_tokens: int
    temperature: float

    # Performance characteristics
    avg_latency_ms: float
    reliability: float  # 0.0 to 1.0

# Model registry with cost/performance trade-offs
MODELS = {
    "fast": ModelConfig(
        name="gpt-4o-mini",
        cost_per_1m_tokens=0.15,
        max_tokens=16000,
        temperature=0.0,
        avg_latency_ms=200,
        reliability=0.95
    ),
    "smart": ModelConfig(
        name="gpt-4o",
        cost_per_1m_tokens=2.50,  # 16x more expensive
        max_tokens=128000,
        temperature=0.1,
        avg_latency_ms=400,
        reliability=0.98
    ),
    "reasoning": ModelConfig(
        name="o1-mini",
        cost_per_1m_tokens=3.00,
        max_tokens=65000,
        temperature=1.0,  # o1 doesn't support temp
        avg_latency_ms=2000,
        reliability=0.99
    )
}

class ComplexityAnalyzer:
    """Analyze prompt complexity to select optimal model."""

    @staticmethod
    def analyze(prompt: str) -> Literal["simple", "moderate", "complex"]:
        """
        Classify prompt complexity using heuristics.

        Returns:
            Complexity level determining model selection
        """
        # Heuristics for complexity
        word_count = len(prompt.split())
        has_multi_step = any(word in prompt.lower() for word in ["and", "then", "after"])
        has_conditionals = any(word in prompt.lower() for word in ["if", "when", "unless"])
        has_variables = any(word in prompt.lower() for word in ["each", "all", "every"])

        # Scoring
        complexity_score = 0

        if word_count > 50:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1

        if has_multi_step:
            complexity_score += 1
        if has_conditionals:
            complexity_score += 2
        if has_variables:
            complexity_score += 2

        # Classify
        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "moderate"
        else:
            return "simple"


class LLMPlanner:
    def __init__(self, ...):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.models = MODELS

    async def create_plan(self, prompt: str) -> Plan:
        """Route to appropriate model based on complexity."""

        # Analyze complexity
        complexity = self.complexity_analyzer.analyze(prompt)

        # Select model
        if complexity == "simple":
            model_config = self.models["fast"]  # GPT-4o-mini
        elif complexity == "moderate":
            model_config = self.models["fast"]  # Still use fast for moderate
        else:
            model_config = self.models["smart"]  # GPT-4o for complex

        logger.info(f"Routing to {model_config.name} (complexity: {complexity})")

        # Make LLM call with selected model
        response = await self.client.chat.completions.create(
            model=model_config.name,
            messages=[...],
            temperature=model_config.temperature,
            ...
        )

        # Track model usage
        self.metrics.record_model_usage(
            model=model_config.name,
            complexity=complexity,
            tokens=response.usage.total_tokens,
            cost=(response.usage.total_tokens / 1_000_000) * model_config.cost_per_1m_tokens
        )

        return Plan.model_validate(...)
```

**Expected Results**:
- **Cost Savings**: 20-30% reduction (most prompts are simple)
- **Latency**: Simple requests get 200ms vs 400ms
- **Quality**: Complex requests get smarter model
- **Scalability**: Easy to add new models (o1, Claude, Llama)
- **Demo Value**: Shows cost-conscious ML engineering

**Why This Fourth**: Demonstrates cost optimization thinking - critical for production at scale.

---

### **Priority 5: Execution History Learning** (Advanced ML, 120 min)
**Problem**: System has no memory - doesn't learn from successful/failed executions.

**Opportunity**: Build execution history feature store for:
1. **Pattern Detection**: Which prompts → which plans work well?
2. **Failure Analysis**: Why do certain plans fail?
3. **Auto-Improvement**: Suggest plan optimizations based on history
4. **Few-shot Examples**: Use successful runs as dynamic examples

**Solution**: Feature engineering + lightweight learning

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
import json

@dataclass
class ExecutionFeatures:
    """Features extracted from a completed run."""

    # Input features
    prompt_text: str
    prompt_embedding: list[float]  # From text-embedding-3-small
    word_count: int
    multi_step: bool

    # Plan features
    num_steps: int
    tools_used: list[str]
    plan_complexity: float  # Custom metric

    # Outcome features
    success: bool
    execution_time_ms: float
    retry_count: int
    error_type: str | None

    # Metadata
    run_id: str
    created_at: datetime


class ExecutionHistoryStore:
    """
    Store and analyze execution history for learning.

    Capabilities:
    - Store successful execution patterns
    - Detect common failure modes
    - Suggest plan improvements
    - Generate dynamic few-shot examples
    """

    def __init__(self, max_history: int = 1000):
        self.history: list[ExecutionFeatures] = []
        self.max_history = max_history

    def record_execution(self, run: Run, features: ExecutionFeatures):
        """Record completed execution."""
        self.history.append(features)

        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def find_similar_successful_runs(
        self,
        prompt_embedding: list[float],
        top_k: int = 3
    ) -> list[ExecutionFeatures]:
        """
        Find similar prompts that executed successfully.

        Used for dynamic few-shot examples.
        """
        successful_runs = [f for f in self.history if f.success]

        # Compute similarities
        similarities = []
        for run in successful_runs:
            sim = self._cosine_similarity(prompt_embedding, run.prompt_embedding)
            similarities.append((sim, run))

        # Return top-k most similar
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [run for _, run in similarities[:top_k]]

    def analyze_failure_patterns(self) -> dict[str, Any]:
        """
        Analyze common failure modes.

        Returns:
            Dict with failure statistics and patterns
        """
        failed_runs = [f for f in self.history if not f.success]

        if not failed_runs:
            return {"total_failures": 0}

        # Group by error type
        errors_by_type = {}
        for run in failed_runs:
            error = run.error_type or "unknown"
            errors_by_type[error] = errors_by_type.get(error, 0) + 1

        # Find tools associated with failures
        tools_in_failures = {}
        for run in failed_runs:
            for tool in run.tools_used:
                tools_in_failures[tool] = tools_in_failures.get(tool, 0) + 1

        return {
            "total_failures": len(failed_runs),
            "failure_rate": len(failed_runs) / len(self.history),
            "errors_by_type": errors_by_type,
            "tools_in_failures": tools_in_failures,
            "avg_retries": sum(r.retry_count for r in failed_runs) / len(failed_runs)
        }

    def get_dynamic_examples(self, prompt: str) -> list[dict]:
        """
        Generate dynamic few-shot examples from history.

        Args:
            prompt: Current user prompt

        Returns:
            List of example dicts for prompt engineering
        """
        # Get embedding for current prompt
        # (In real impl, call OpenAI embeddings API)
        prompt_embedding = self._mock_embedding(prompt)

        # Find similar successful runs
        similar_runs = self.find_similar_successful_runs(prompt_embedding, top_k=2)

        # Convert to few-shot examples
        examples = []
        for run in similar_runs:
            examples.append({
                "user": run.prompt_text,
                "plan": f"Successfully executed with {run.num_steps} steps using {', '.join(run.tools_used)}"
            })

        return examples

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))

    @staticmethod
    def _mock_embedding(text: str) -> list[float]:
        """Mock embedding for demo (use OpenAI in production)."""
        import hashlib
        import numpy as np

        # Generate deterministic embedding from text hash
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.random(1536).tolist()


class LLMPlanner:
    def __init__(self, ...):
        self.execution_history = ExecutionHistoryStore()

    async def create_plan(self, prompt: str) -> Plan:
        """Create plan with dynamic few-shot examples from history."""

        # Get dynamic examples from execution history
        examples = self.execution_history.get_dynamic_examples(prompt)

        # Enhance system prompt with examples
        enhanced_prompt = self._system_prompt()
        if examples:
            enhanced_prompt += "\n\n## Recent Successful Examples\n"
            for ex in examples:
                enhanced_prompt += f"\nUser: {ex['user']}\nResult: {ex['plan']}\n"

        # Generate plan with enhanced context
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": prompt}
            ],
            ...
        )

        return Plan.model_validate(...)

    async def record_execution_outcome(self, run: Run):
        """Record execution for learning."""
        features = await self._extract_features(run)
        self.execution_history.record_execution(run, features)

        # Log insights
        patterns = self.execution_history.analyze_failure_patterns()
        if patterns["total_failures"] > 10:
            logger.warning(f"High failure rate: {patterns['failure_rate']:.1%}")
```

**Integration with Orchestrator**:
```python
class Orchestrator:
    async def _execute_run(self, run_id: str):
        """Execute run and record outcome for learning."""
        run = self.runs[run_id]

        try:
            # Execute plan
            await self._execute_steps(run)
            run.status = RunStatus.COMPLETED
        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = str(e)
        finally:
            # Record execution for learning
            await self.planner.record_execution_outcome(run)
```

**Expected Results**:
- **Self-Improving**: Plans get better over time as history grows
- **Failure Prevention**: Detect and avoid common failure patterns
- **Dynamic Examples**: Few-shot learning from actual executions
- **Observability**: Rich failure analysis and insights
- **Demo Value**: Shows advanced ML engineering (feature eng + learning)

**Why This Fifth**: Most advanced improvement - demonstrates true ML engineering beyond API calls.

---

## 3. Implementation Guidance

### Quick Wins (Can Demo in Interview)

**Option A: Semantic Caching + Metrics (90 min total)**
- Implement `SemanticCache` class (60 min)
- Add `PlanningMetrics` tracking (30 min)
- Add `/metrics/planner` endpoint (15 min)
- **Impact**: 40-60% latency reduction, full observability

**Option B: Advanced Prompt + Model Routing (120 min total)**
- Enhance system prompt with examples (60 min)
- Implement `ComplexityAnalyzer` (30 min)
- Add model routing logic (30 min)
- **Impact**: Better quality + 20-30% cost savings

### Full Implementation (4-6 hours)

**Phase 1: Foundation** (90 min)
1. Semantic caching with embeddings
2. Comprehensive metrics collection
3. Metrics API endpoint

**Phase 2: Quality** (90 min)
1. Advanced prompt engineering
2. Few-shot example system
3. Chain-of-thought reasoning

**Phase 3: Optimization** (60 min)
1. Model complexity routing
2. Cost tracking per model
3. Dynamic model selection

**Phase 4: Learning** (120 min)
1. Execution history store
2. Feature extraction pipeline
3. Dynamic example generation
4. Failure pattern analysis

---

## 4. Senior ML Engineer Differentiators

### What Separates Senior from Mid-level

**Mid-Level** (Current Implementation ✅):
- Calls OpenAI API correctly
- Uses structured outputs
- Has fallback logic
- Tracks basic metrics

**Senior Level** (Gaps to Address):
- **Production Thinking**: Caching, rate limiting, circuit breakers
- **Cost Consciousness**: Model routing, batch optimization
- **Observability**: Comprehensive metrics, failure analysis
- **Learning Systems**: Feature stores, pattern detection
- **Prompt Engineering**: Few-shot, chain-of-thought, dynamic examples

### Interview Discussion Points

**Strong Answers** (Shows Senior Experience):

1. **"How would you optimize LLM costs?"**
   - ✅ "Semantic caching reduces duplicate calls 40-60%"
   - ✅ "Route simple requests to cheap models (GPT-4o-mini vs GPT-4o)"
   - ✅ "Batch similar requests to reduce per-request overhead"
   - ✅ "Monitor token usage by endpoint, set budget alerts"
   - ❌ "Just use a cheaper model" (too simple)

2. **"How would you handle LLM failures in production?"**
   - ✅ "Multi-layer fallback: cache → LLM → pattern-based → error"
   - ✅ "Circuit breaker after 3 consecutive failures, exponential backoff"
   - ✅ "Categorize errors (rate limit vs timeout vs invalid), different strategies"
   - ✅ "Alert on fallback rate > 10%, indicates API issues"
   - ❌ "Try again" (not sufficient)

3. **"How would you measure model quality?"**
   - ✅ "Plan validity rate (schema compliance)"
   - ✅ "Execution success rate (plan actually works)"
   - ✅ "Latency percentiles (p50, p95, p99)"
   - ✅ "User satisfaction (implicit: retry rate, explicit: feedback)"
   - ✅ "Cost per successful plan (efficiency metric)"
   - ❌ "Just look at success rate" (incomplete)

4. **"How would you improve the system over time?"**
   - ✅ "Store execution history as feature store"
   - ✅ "Use successful runs as dynamic few-shot examples"
   - ✅ "Analyze failure patterns, update prompts to prevent"
   - ✅ "A/B test prompt variations, iterate on winners"
   - ✅ "Fine-tune on high-quality examples if volume justifies"
   - ❌ "Manually update prompts sometimes" (not systematic)

### Red Flags in Interview

**Avoid These Answers** (Shows Lack of Production Experience):
- ❌ "Just use the latest GPT model" (no cost awareness)
- ❌ "LLMs are unreliable, use rules" (not understanding hybrid approach)
- ❌ "Retry until it works" (no backoff/circuit breaker)
- ❌ "Can't measure LLM quality" (defeatist, not true)
- ❌ "Fine-tuning solves everything" (expensive, often unnecessary)

---

## 5. Model Selection Deep Dive

### When to Use Each Model Type

**GPT-4o-mini** (Current Choice ✅):
- **Use For**: Simple planning, high-frequency requests
- **Cost**: $0.15 per 1M tokens (very cheap)
- **Latency**: 200-300ms
- **Quality**: Good for structured tasks
- **Best For**: Production at scale (current choice is correct)

**GPT-4o**:
- **Use For**: Complex planning, multi-step reasoning
- **Cost**: $2.50 per 1M tokens (16x more expensive)
- **Latency**: 400-500ms
- **Quality**: Significantly better reasoning
- **Best For**: Complex requests that justify cost

**o1-mini**:
- **Use For**: Very complex reasoning, novel scenarios
- **Cost**: $3.00 per 1M tokens
- **Latency**: 2-5 seconds (much slower)
- **Quality**: Best reasoning, novel problem solving
- **Best For**: Edge cases where GPT-4o fails

**Local Models (Llama 3.1, Mistral)**:
- **Use For**: Sensitive data, no API dependency
- **Cost**: Infrastructure only (~$50/month GPU)
- **Latency**: 500-2000ms (depends on hardware)
- **Quality**: Good but below GPT-4o
- **Best For**: Privacy requirements, on-prem deployment

**When to Fine-tune**:
- ❌ **DON'T** for this use case (too expensive, overkill)
- ✅ **DO** if:
  - Have >10,000 high-quality examples
  - Domain-specific terminology/patterns
  - Latency-critical (fine-tuned smaller model can be faster)
  - Cost >$1000/month (amortize fine-tuning cost)

**Recommendation**: Current GPT-4o-mini choice is correct. Add GPT-4o routing for complex requests only.

---

## 6. Structured Output Best Practices

### Current Implementation ✅

```python
# Using OpenAI's strict JSON schema mode (correct approach)
response_format={"type": "json_schema", "json_schema": PLAN_SCHEMA}
```

**Strengths**:
- ✅ Enforces exact schema match (no invalid JSON)
- ✅ Rejects responses that don't conform
- ✅ No post-processing needed

### Alternative Approaches (Don't Use)

**❌ JSON Mode Without Schema**:
```python
response_format={"type": "json_object"}  # Weak - no structure enforcement
```
- Problem: Gets valid JSON but wrong structure
- Requires validation layer and retry logic

**❌ Prompt-Based JSON**:
```python
messages=[{"role": "user", "content": "Return JSON with..."}]  # Unreliable
```
- Problem: 10-20% failure rate even with good prompts
- No guarantee of parseable JSON

**✅ Current Approach is Best**: Stick with strict JSON schema mode.

### When Schema Enforcement Fails

**Fallback Strategy** (Already Implemented ✅):
```python
try:
    plan = await llm_generate(...)  # With schema
except Exception:
    return pattern_based_planner.create_plan(prompt)  # Guaranteed structure
```

**Additional Hardening** (If Needed):
```python
# Add Pydantic validation as extra safety layer
try:
    plan_dict = json.loads(response.choices[0].message.content)
    plan = Plan.model_validate(plan_dict)  # Pydantic validation
except ValidationError as e:
    logger.error(f"Schema validation failed: {e}")
    return fallback.create_plan(prompt)
```

---

## 7. Inference Optimization Strategies

### Current State
- Latency: 200-500ms per request
- No caching
- No batching
- Sequential processing

### Optimization Techniques

**1. Semantic Caching** (Covered in Priority 1)
- **Impact**: 40-60% latency reduction on cache hits
- **Implementation**: 60 minutes
- **ROI**: Highest

**2. Request Batching**
```python
class BatchedLLMPlanner:
    """Batch multiple planning requests for efficiency."""

    def __init__(self, batch_size: int = 10, batch_timeout_ms: int = 100):
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.pending_requests: list[tuple[str, asyncio.Future]] = []

    async def create_plan(self, prompt: str) -> Plan:
        """Add to batch and wait for result."""
        future = asyncio.Future()
        self.pending_requests.append((prompt, future))

        # Trigger batch if full
        if len(self.pending_requests) >= self.batch_size:
            asyncio.create_task(self._process_batch())
        else:
            # Or trigger on timeout
            asyncio.create_task(self._process_batch_after_timeout())

        return await future

    async def _process_batch(self):
        """Process accumulated requests in single API call."""
        if not self.pending_requests:
            return

        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]

        # Make single API call with multiple prompts
        # (OpenAI doesn't support this directly, but can parallelize)
        tasks = [self._generate_plan(prompt) for prompt, _ in batch]
        results = await asyncio.gather(*tasks)

        # Resolve futures
        for (_, future), result in zip(batch, results):
            future.set_result(result)
```

**Impact**: Reduce API overhead, improve throughput 2-3x

**3. Streaming Responses**
```python
# For long plans, start execution before full response
async def create_plan_streaming(self, prompt: str) -> AsyncIterator[PlanStep]:
    """Stream plan steps as they're generated."""
    async for chunk in await self.client.chat.completions.create(
        model=self.model,
        messages=[...],
        stream=True
    ):
        if step := self._parse_step(chunk):
            yield step
```

**Impact**: Start execution 200-300ms earlier (perceived latency reduction)

**4. Speculative Execution**
```python
# Start both LLM and pattern-based, use whichever finishes first
async def create_plan_speculative(self, prompt: str) -> Plan:
    """Race LLM vs pattern-based, use fastest."""
    llm_task = asyncio.create_task(self._llm_plan(prompt))
    pattern_task = asyncio.create_task(self._pattern_plan(prompt))

    # Wait for first to complete
    done, pending = await asyncio.wait(
        [llm_task, pattern_task],
        return_when=asyncio.FIRST_COMPLETED
    )

    # Cancel slower task
    for task in pending:
        task.cancel()

    # Return fastest result
    return done.pop().result()
```

**Impact**: Eliminate tail latency, guaranteed fast response

---

## 8. Fallback Strategy Analysis

### Current Implementation ✅

```python
try:
    # Primary: LLM with structured outputs
    plan = await llm.create_plan(prompt)
except Exception:
    # Fallback: Pattern-based
    plan = pattern_based.create_plan(prompt)
```

**Strengths**:
- ✅ Simple and reliable
- ✅ Never fails due to API issues
- ✅ Graceful degradation

**Limitations**:
- Pattern-based has limited coverage (~10-15 patterns)
- No intermediate fallback options
- Treats all LLM errors the same

### Enhanced Fallback Chain

```python
class MultiLayerPlanner:
    """
    Multi-layer fallback strategy with error-specific routing.

    Layers:
    1. Cache (instant, free)
    2. LLM with strict schema (200-500ms, ~$0.00003)
    3. LLM without schema (200-500ms, more permissive)
    4. Pattern-based (instant, free, limited)
    5. Error (with helpful message)
    """

    async def create_plan(self, prompt: str) -> Plan:
        # Layer 1: Check cache
        if cached := self.cache.get(prompt):
            return cached

        # Layer 2: Try LLM with strict schema
        try:
            plan = await self._llm_with_schema(prompt)
            self.cache.set(prompt, plan)
            return plan
        except RateLimitError:
            # Don't retry on rate limit, skip to Layer 4
            logger.warning("Rate limit hit, using pattern-based")
            return self._pattern_based(prompt)
        except TimeoutError:
            # Retry without schema (Layer 3)
            logger.warning("LLM timeout, trying without schema")
            pass
        except ValidationError:
            # Schema validation failed, retry without schema
            logger.warning("Schema validation failed, trying without schema")
            pass

        # Layer 3: LLM without schema (more permissive)
        try:
            plan = await self._llm_without_schema(prompt)
            # Validate and fix if needed
            plan = self._fix_plan_errors(plan)
            self.cache.set(prompt, plan)
            return plan
        except Exception as e:
            logger.warning(f"LLM failed: {e}, using pattern-based")

        # Layer 4: Pattern-based fallback
        try:
            plan = self._pattern_based(prompt)
            return plan
        except ValueError as e:
            # Layer 5: Helpful error message
            raise ValueError(
                f"Could not generate plan: {e}\n"
                f"LLM failed, pattern matching failed.\n"
                f"Try rephrasing your request to be more specific."
            )
```

**Impact**: 99.9%+ availability, intelligent error handling

---

## Final Recommendations

### For Interview Discussion

**What to Emphasize**:
1. ✅ "Already implemented hybrid LLM + rule-based approach"
2. ✅ "Using structured outputs for reliability"
3. ✅ "Cost-optimized with GPT-4o-mini"
4. 💡 "Would add semantic caching for 40-60% latency reduction"
5. 💡 "Would add comprehensive metrics for observability"
6. 💡 "Would implement model routing based on complexity"
7. 💡 "Would build execution history feature store for learning"

**Talking Points for Each Improvement**:

**Semantic Caching**:
- "Most agent requests are similar - caching embeddings saves 40-60% latency"
- "At scale, this also saves significant cost"
- "Simple to implement with OpenAI embeddings API"

**Model Monitoring**:
- "Can't operate what you don't measure"
- "Track p50/p95/p99 latency, cost per request, fallback rate"
- "Alert when fallback > 10% indicates API issues"

**Prompt Engineering**:
- "Few-shot examples dramatically improve accuracy"
- "Chain-of-thought reasoning reduces edge case failures"
- "Dynamic examples from execution history = self-improving system"

**Model Routing**:
- "Simple requests don't need expensive models"
- "Route based on complexity heuristics"
- "20-30% cost savings with same quality"

**Execution History**:
- "Store successful patterns as feature store"
- "Detect and prevent common failure modes"
- "Use execution history for dynamic few-shot examples"
- "True ML engineering: feature extraction + learning"

### Priority Order for Demo

1. **Semantic Caching** (60 min) - Biggest impact, easy to demo
2. **Metrics Dashboard** (45 min) - Shows production thinking
3. **Advanced Prompts** (60 min) - Quality improvement visible immediately

**Total**: 2.5-3 hours for substantial upgrade

### What NOT to Do

❌ **Don't** suggest fine-tuning (overkill for this scale)
❌ **Don't** add complex ML models (unnecessary complexity)
❌ **Don't** change to local models (API is correct choice)
❌ **Don't** remove fallback logic (critical for reliability)

### Assessment: Is This Senior-Level Work?

**Current Implementation**: **Solid Mid-to-Senior** (Tier 2)
- Strong fundamentals
- Production patterns present
- Missing advanced optimizations

**With Improvements**: **Clear Senior** (Tier 1)
- Cost optimization
- Comprehensive observability
- Learning systems
- Production operations expertise

**Verdict**: Candidate has good foundation. Interview should focus on:
1. How would they add these improvements?
2. What trade-offs would they make?
3. How would they prioritize?
4. What metrics would they track?

Answers to these questions will reveal senior-level thinking better than code implementation.
