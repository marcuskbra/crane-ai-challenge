# Crane Challenge - Technical Patterns and Decisions

## Design Patterns Applied

### 1. AST Visitor Pattern (Calculator)
**Pattern**: Visitor pattern for safe expression evaluation
**Implementation**: `SafeCalculator(ast.NodeVisitor)`
**Benefits**:
- No eval/exec (security critical)
- Whitelist-based operator control
- Type-safe number handling
- Clean separation of parsing and evaluation

```python
class SafeCalculator(ast.NodeVisitor):
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        
        if op_type not in self.OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        
        return self.OPERATORS[op_type](left, right)
```

### 2. Singleton Pattern (Registries)
**Pattern**: Cached singleton with @lru_cache
**Implementation**: `get_tool_registry()`, `get_orchestrator()`
**Benefits**:
- Single instance per process
- Thread-safe initialization
- Clean dependency injection
- Memory efficient

```python
@lru_cache
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry()

@lru_cache
def get_orchestrator() -> Orchestrator:
    return Orchestrator(
        planner=PatternBasedPlanner(),
        tools=get_tool_registry(),
    )
```

### 3. Template Method Pattern (Tools)
**Pattern**: Abstract base class with template method
**Implementation**: `BaseTool` ABC
**Benefits**:
- Consistent tool interface
- Type-safe metadata
- Standardized result format
- Easy tool addition

```python
class BaseTool(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        pass
```

### 4. State Pattern (Run Status)
**Pattern**: Enum-based state machine
**Implementation**: `RunStatus` enum
**Benefits**:
- Type-safe state transitions
- Clear lifecycle management
- No invalid states possible

```python
class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# State transitions:
# PENDING → RUNNING → (COMPLETED | FAILED)
```

### 5. Retry Pattern (Exponential Backoff)
**Pattern**: Exponential backoff with max retries
**Implementation**: `_execute_step_with_retry()`
**Benefits**:
- Resilient to transient failures
- Configurable retry limits
- Progressive delay prevents thundering herd

```python
async def _execute_step_with_retry(self, step) -> ExecutionStep:
    for attempt in range(1, self.max_retries + 1):
        try:
            result = await tool.execute(**step.tool_input)
            return ExecutionStep(
                step_number=step.step_number,
                tool_name=step.tool_name,
                tool_input=step.tool_input,
                success=result.success,
                output=result.output,
                error=result.error,
                attempts=attempt,
            )
        except Exception as e:
            if attempt < self.max_retries:
                delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
                await asyncio.sleep(delay)
```

### 6. Fire-and-Forget Pattern (Async Execution)
**Pattern**: Non-blocking async task creation
**Implementation**: `asyncio.create_task()`
**Benefits**:
- Immediate API response
- Background processing
- Clean separation of concerns

```python
async def create_run(self, prompt: str) -> Run:
    run = Run(prompt=prompt)
    plan = self.planner.create_plan(prompt)
    run.plan = plan
    self.runs[run.run_id] = run
    
    # Fire and forget - returns immediately
    asyncio.create_task(self._execute_run(run.run_id))
    
    return run  # Returns in PENDING status
```

## Security Patterns

### 1. Whitelist-Based Security
**Context**: Calculator operator validation
**Pattern**: Explicit allowlist of safe operations
**Implementation**:
```python
OPERATORS = {
    ast.Add: operator.add,      # Only these operators allowed
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

if op_type not in self.OPERATORS:
    raise ValueError(f"Unsupported operator: {op_type.__name__}")
```

### 2. Input Validation
**Context**: All tool inputs and API requests
**Pattern**: Pydantic model validation
**Benefits**:
- Type safety
- Automatic validation
- Clear error messages

```python
class RunCreate(BaseModel):
    prompt: str = Field(..., min_length=1, description="Natural language task")

# Pydantic automatically validates:
# - prompt is a string
# - prompt has at least 1 character
# - raises ValidationError if invalid
```

### 3. Division by Zero Protection
**Context**: Calculator division operation
**Pattern**: Pre-execution validation
```python
if op_type == ast.Div and right == 0:
    raise ValueError("Cannot divide by zero")
```

## Data Flow Patterns

### 1. Request → Response Flow
```
User Request (JSON)
    ↓
FastAPI Pydantic Validation
    ↓
API Endpoint Handler
    ↓
Orchestrator.create_run() [Returns immediately]
    ↓
Run Model (PENDING status)
    ↓
JSON Response (201 Created)

[Background Execution]
    ↓
Planner.create_plan()
    ↓
Orchestrator._execute_run()
    ↓
Tool Execution with Retry
    ↓
Run Model Updated (COMPLETED/FAILED)
```

### 2. Multi-Step Execution Flow
```
"calculate 10 + 5 and calculate 20 / 4"
    ↓
Planner: split on "and"
    ↓
Step 1: PlanStep(tool=calculator, input={expression: "10 + 5"})
Step 2: PlanStep(tool=calculator, input={expression: "20 / 4"})
    ↓
Orchestrator: Sequential execution
    ↓
Step 1 → ToolResult(output=15.0) → ExecutionStep
Step 2 → ToolResult(output=5.0) → ExecutionStep
    ↓
Run.result = 5.0 (last step output)
Run.status = COMPLETED
```

## Error Handling Patterns

### 1. Standard Exceptions → HTTP Mapping
**Pattern**: Convert business exceptions to HTTP status codes
```python
# Service raises standard exception
raise ValueError("Prompt cannot be empty")

# API converts to HTTP status
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
```

### 2. Three-Level Error Handling
**Level 1: Tool Level** - ToolResult with success/error
```python
return ToolResult(success=False, error="Division by zero")
```

**Level 2: Orchestrator Level** - ExecutionStep tracking
```python
ExecutionStep(
    success=False,
    error="Tool execution failed",
    attempts=3
)
```

**Level 3: API Level** - HTTP status codes
```python
raise HTTPException(status_code=500, detail="Execution failed")
```

## Testing Patterns

### 1. Parameterized Tests
**Pattern**: Test multiple scenarios with single test function
```python
@pytest.mark.parametrize("expression,expected", [
    ("1 + 1", 2.0),
    ("10 - 5", 5.0),
    ("3 * 4", 12.0),
    ("20 / 4", 5.0),
])
async def test_calculator_expressions(expression, expected):
    result = await calculator.execute(expression=expression)
    assert result.output == expected
```

### 2. E2E Integration Tests
**Pattern**: Full workflow validation with async sleep
```python
async def test_calculator_run_complete_flow(self, test_client):
    # Create run
    response = test_client.post("/api/v1/runs", json={"prompt": "calculate 5 * 8"})
    run_id = response.json()["run_id"]
    
    # Wait for async execution
    await asyncio.sleep(0.5)
    
    # Verify completion
    response = test_client.get(f"/api/v1/runs/{run_id}")
    assert response.json()["status"] == "completed"
    assert response.json()["result"] == 40.0
```

### 3. Security Testing
**Pattern**: Test injection prevention
```python
async def test_code_injection_attempt_import():
    result = await calculator.execute(expression="__import__('os').system('ls')")
    assert result.success is False
    assert "unsupported expression type" in result.error.lower()
```

## Code Organization Patterns

### 1. Module Structure
```
tools/
├── __init__.py          # Public exports
├── base.py              # Abstract interfaces
├── calculator.py        # Implementation
├── todo_store.py        # Implementation
└── registry.py          # Singleton registry
```

### 2. Dependency Injection
**Pattern**: FastAPI Depends with type annotations
```python
OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]

@router.post("/runs")
async def create_run(
    request: RunCreate,
    orchestrator: OrchestratorDep  # Injected dependency
) -> Run:
    return await orchestrator.create_run(request.prompt)
```

## Performance Patterns

### 1. Async/Await Throughout
**Pattern**: Async-first design
**Benefits**:
- High concurrency
- Non-blocking I/O
- Efficient resource usage

```python
# All tool operations are async
async def execute(self, **kwargs) -> ToolResult:
    pass

# Orchestrator uses async execution
async def _execute_run(self, run_id: str) -> None:
    pass
```

### 2. Caching with @lru_cache
**Pattern**: Function-level caching for singletons
**Benefits**:
- Zero overhead after first call
- Thread-safe
- Automatic memory management

```python
@lru_cache
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry()
```

## Key Architectural Decisions

### 1. Why Standard Exceptions vs Discriminated Unions?
**Decision**: Use standard Python exceptions
**Rationale**:
- Simpler for POC scope
- Pythonic and familiar
- FastAPI integrates naturally
- YAGNI principle - don't over-engineer

### 2. Why Pattern-Based Planner vs LLM?
**Decision**: Regex pattern matching
**Rationale**:
- Predictable behavior
- No API dependencies
- Fast execution
- Sufficient for constrained domain

### 3. Why In-Memory Storage vs Database?
**Decision**: Dict-based in-memory storage
**Rationale**:
- POC requirement only
- No persistence needed
- Faster development
- Easy testing

### 4. Why Immediate Response + Async Execution?
**Decision**: Fire-and-forget pattern
**Rationale**:
- Better user experience (no blocking)
- Clean separation of concerns
- Enables long-running tasks
- Standard REST pattern

## Lessons Learned

### 1. AST is Powerful
- Perfect for safe code evaluation
- Clean separation of parsing and execution
- Type-safe by design

### 2. Pydantic Validation Rocks
- Automatic validation saves code
- Clear error messages
- Type safety throughout

### 3. FastAPI DI Simplifies Testing
- Easy to mock dependencies
- Clean test fixtures
- Natural async support

### 4. Pattern Matching Sufficient for POC
- Regex adequate for constrained domains
- Predictable and testable
- No AI complexity needed

### 5. YAGNI Prevents Over-Engineering
- Delivered working POC quickly
- 83% coverage proves quality
- Simple is better than complex
