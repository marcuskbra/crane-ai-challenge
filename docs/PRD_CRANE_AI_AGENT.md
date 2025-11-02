# Product Requirements Document: Crane AI Agent Runtime

**Project**: Minimal AI Agent Runtime POC
**Target Tier**: Tier 2 (75-85% score)
**Time Budget**: 6-8 hours
**Language**: Python 3.12+ with FastAPI
**Status**: Implementation Ready

---

## 1. Executive Summary

### 1.1 Project Vision

Build a minimal, production-quality AI agent runtime that demonstrates:

- **Intelligent task planning** from natural language
- **Robust execution orchestration** with retry logic and state management
- **Secure tool integration** with proper error handling
- **Clean REST API** for interaction and monitoring
- **Engineering excellence** through testing, documentation, and pragmatic design decisions

This is a **proof-of-concept evaluation** focused on code clarity, architecture decisions, and problem-solving approach
over feature completeness.

### 1.2 Success Criteria

**Primary Goal: Achieve Tier 2 (75-85% score)**

| Category             | Target                | Measurement                                  |
|----------------------|-----------------------|----------------------------------------------|
| **Functional**       | Core features working | POST /runs executes calculator + todo tasks  |
| **Quality**          | >80% test coverage    | pytest --cov report                          |
| **Security**         | AST-based calculator  | No eval/exec, injection tests pass           |
| **Documentation**    | Comprehensive README  | Architecture, setup, trade-offs, limitations |
| **Tier Achievement** | Tier 2 target         | 75-85% overall score                         |

### 1.3 Key Stakeholders

**Primary Evaluator**: Crane Engineering Team
**Evaluation Focus**:

- Code Quality (40%): Clean code, error handling, type safety, separation of concerns
- Architecture & Design (30%): Logical structure, extensibility, appropriate patterns
- Functionality (20%): Requirements met, API works, tools function correctly
- Documentation (10%): Clear README, honest limitations, thoughtful trade-offs

### 1.4 Constraints

- **Time**: 6-8 hours maximum (half-day assignment)
- **Local execution**: No cloud infrastructure
- **Minimal dependencies**: Standard Python libraries preferred
- **Language**: Python (using existing FastAPI foundation)
- **Scope**: POC demonstrating concepts, not production system

---

## 2. Scope & Implementation Phases

### Phase 0: Project Setup ✅ COMPLETED

**Duration**: 15-20 minutes
**Status**: Already completed with existing project structure

**Entry Criteria**:

- Clean repository or existing crane project

**Deliverables**:

- ✅ Project structure (src/, tests/, docs/)
- ✅ Virtual environment configured
- ✅ FastAPI foundation with /health endpoint
- ✅ Git repository initialized
- ✅ Dependencies managed via uv/pip
- ✅ Development tools configured (pytest, ruff, type checker)

**Exit Criteria**:

- Can run `python -m challenge` successfully
- Can import from `src/` modules
- Health endpoint responds: `curl http://localhost:8000/health`

**Quality Gate**: ✅ Already verified

---

### Phase 1: Tool System Implementation

**Duration**: 90-120 minutes
**Priority**: CRITICAL PATH - Security Sensitive

#### 1.1 Tool Interface Design

**Base Tool Interface** (`src/tools/base.py`):

```python
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel


class ToolResult(BaseModel):
    """Standard tool execution result"""
    success: bool
    output: Any | None = None
    error: str | None = None


class Tool(ABC):
    """Base tool interface"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier (e.g., 'Calculator', 'TodoStore.add')"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of tool purpose"""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """JSON schema defining expected input parameters"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute tool with validated inputs"""
        pass
```

#### 1.2 Calculator Tool Implementation

**File**: `src/tools/calculator.py`

**Tier 1 Requirements** (Minimum Viable):

- Basic operators: `+`, `-`, `*`, `/`
- Integer arithmetic
- Simple validation

**Tier 2 Requirements** (TARGET):

- ✅ Decimal numbers: `3.14 + 2.86`
- ✅ Negative numbers: `-5 * 3`
- ✅ Parentheses: `(10 + 5) * 2`
- ✅ Order of operations: PEMDAS

**Tier 3 Requirements** (Stretch):

- Scientific functions: `sqrt`, `pow`, `sin`, `cos`, `log`
- Constants: `pi`, `e`

**🔒 CRITICAL SECURITY REQUIREMENT**:

```python
# ⛔ FORBIDDEN: NEVER use eval() or exec()
# ❌ BAD: result = eval(expression)  # SECURITY VULNERABILITY

# ✅ REQUIRED: AST-based evaluation
import ast


class SafeCalculator(ast.NodeVisitor):
    """AST-based expression evaluator - NO eval/exec"""

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        # Tier 2: Add support for negation, parentheses
        ast.USub: operator.neg,
        # Tier 3: Add support for power, etc.
    }

    def evaluate(self, expression: str) -> float:
        """Safely evaluate arithmetic expression using AST"""
        try:
            tree = ast.parse(expression, mode='eval')
            return self.visit(tree.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
```

**Input Schema**:

```json
{
  "type": "object",
  "properties": {
    "expression": {
      "type": "string",
      "description": "Arithmetic expression to evaluate",
      "examples": [
        "2 + 2",
        "(10 - 3) * 4",
        "-5.5 / 2"
      ]
    }
  },
  "required": [
    "expression"
  ]
}
```

**Error Handling**:

- Invalid syntax: "Invalid expression: unexpected character"
- Division by zero: "Cannot divide by zero"
- Unsupported operator: "Operator not supported: ^"
- Injection attempt: "Security violation: eval not allowed"

**Security Tests** (MANDATORY):

```python
async def test_calculator_blocks_eval_injection():
    calc = CalculatorTool(tier=2)

    # Test injection attempts
    injection_attempts = [
        "__import__('os').system('echo hacked')",
        "exec('print(\"hacked\")')",
        "eval('2+2')",
        "__builtins__",
    ]

    for attempt in injection_attempts:
        result = await calc.execute(expression=attempt)
        assert not result.success, f"Security failure: {attempt} not blocked"
        assert "not supported" in result.error.lower()
```

#### 1.3 TodoStore Tool Implementation

**File**: `src/tools/todo_store.py`

**State Management**:

```python
class TodoItem(BaseModel):
    id: str
    title: str
    completed: bool = False
    created_at: datetime


class TodoStore:
    """In-memory todo storage"""

    def __init__(self):
        self._todos: dict[str, TodoItem] = {}
```

**Tier 1 Operations** (Minimum):

- `TodoStore.add`: Add new todo
- `TodoStore.list`: List all todos

**Tier 2 Operations** (TARGET):

- ✅ `TodoStore.add`: Create todo with generated ID
- ✅ `TodoStore.list`: Return all todos
- ✅ `TodoStore.get`: Retrieve specific todo by ID
- ✅ `TodoStore.complete`: Mark todo as completed
- ✅ `TodoStore.delete`: Remove todo

**Tier 3 Operations** (Stretch):

- `TodoStore.update`: Modify todo title
- `TodoStore.filter`: Filter by completed status
- `TodoStore.search`: Search by title
- `TodoStore.priority`: Add priority levels

**Input Schemas**:

*TodoStore.add*:

```json
{
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "minLength": 1,
      "maxLength": 200,
      "description": "Todo item title"
    }
  },
  "required": [
    "title"
  ]
}
```

*TodoStore.complete*:

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Todo item ID"
    }
  },
  "required": [
    "id"
  ]
}
```

**Error Handling**:

- Empty title: "Title cannot be empty"
- ID not found: "Todo not found: {id}"
- Invalid ID format: "Invalid ID format"

#### 1.4 Tool Registry

**File**: `src/tools/__init__.py`

```python
class ToolRegistry:
    """Central registry for all available tools"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by its name"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Retrieve tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names"""
        return list(self._tools.keys())


# Global registry instance
def get_registry() -> ToolRegistry:
    """Get or create global tool registry"""
    if not hasattr(get_registry, "_instance"):
        registry = ToolRegistry()
        # Register all tools
        registry.register(CalculatorTool(tier=2))
        registry.register(TodoStoreAddTool())
        registry.register(TodoStoreListTool())
        # ... register other todo tools
        get_registry._instance = registry
    return get_registry._instance
```

**Exit Criteria**:

1. ✅ Calculator evaluates `(10 + 5) * 2` = `30.0`
2. ✅ Calculator rejects eval injection with clear error
3. ✅ TodoStore.add returns ID, TodoStore.list retrieves it
4. ✅ All tool operations return consistent ToolResult format
5. ✅ Input validation catches invalid parameters
6. ✅ Security check passes: `grep -r "eval\|exec" src/tools/calculator.py` returns clean

**Quality Gate**:

```bash
# Must pass before proceeding to Phase 2
python -c "from src.tools import get_registry; assert len(get_registry().list_tools()) >= 6"
```

**Time Checkpoint**: Should complete within 90-120 minutes
**Red Flag**: If >150 minutes, cut Tier 3 features immediately

---

### Phase 2: Tool Testing

**Duration**: 45-60 minutes
**Priority**: HIGH - Coverage determines tier achievement

#### 2.1 Calculator Tests

**File**: `tests/unit/test_calculator.py`

**Test Coverage Requirements**:

1. **Valid Input Tests** (Tier 1):
   ```python
   @pytest.mark.parametrize("expression,expected", [
       ("2 + 2", 4.0),
       ("10 - 3", 7.0),
       ("4 * 5", 20.0),
       ("15 / 3", 5.0),
   ])
   async def test_calculator_basic_operations(expression, expected):
       calc = CalculatorTool(tier=1)
       result = await calc.execute(expression=expression)
       assert result.success
       assert result.output == expected
   ```

2. **Tier 2 Features**:
   ```python
   async def test_calculator_decimals():
       calc = CalculatorTool(tier=2)
       result = await calc.execute(expression="3.14 + 2.86")
       assert result.success
       assert abs(result.output - 6.0) < 0.001

   async def test_calculator_negatives():
       calc = CalculatorTool(tier=2)
       result = await calc.execute(expression="-5 * 3")
       assert result.success
       assert result.output == -15.0

   async def test_calculator_parentheses():
       calc = CalculatorTool(tier=2)
       result = await calc.execute(expression="(10 + 5) * 2")
       assert result.success
       assert result.output == 30.0
   ```

3. **Error Handling Tests**:
   ```python
   async def test_calculator_division_by_zero():
       calc = CalculatorTool(tier=2)
       result = await calc.execute(expression="5 / 0")
       assert not result.success
       assert "cannot divide by zero" in result.error.lower()

   async def test_calculator_invalid_syntax():
       calc = CalculatorTool(tier=2)
       result = await calc.execute(expression="2 + + 2")
       assert not result.success
       assert "invalid" in result.error.lower()
   ```

4. **🔒 Security Tests** (MANDATORY):
   ```python
   @pytest.mark.parametrize("injection", [
       "__import__('os').system('echo hacked')",
       "exec('print(\"hacked\")')",
       "eval('2+2')",
       "__builtins__",
       "globals()",
       "locals()",
   ])
   async def test_calculator_blocks_eval_injection(injection):
       calc = CalculatorTool(tier=2)
       result = await calc.execute(expression=injection)
       assert not result.success, f"SECURITY FAILURE: {injection} not blocked"
       assert "not supported" in result.error.lower() or "invalid" in result.error.lower()
   ```

#### 2.2 TodoStore Tests

**File**: `tests/unit/test_todo_store.py`

**Test Coverage Requirements**:

1. **CRUD Operations** (Tier 2):
   ```python
   async def test_todo_add():
       store = TodoStore()
       add_tool = TodoStoreAddTool(store)
       result = await add_tool.execute(title="Buy milk")
       assert result.success
       assert "id" in result.output

   async def test_todo_list():
       store = TodoStore()
       add_tool = TodoStoreAddTool(store)
       list_tool = TodoStoreListTool(store)

       await add_tool.execute(title="Task 1")
       await add_tool.execute(title="Task 2")

       result = await list_tool.execute()
       assert result.success
       assert len(result.output) == 2

   async def test_todo_complete():
       store = TodoStore()
       add_tool = TodoStoreAddTool(store)
       complete_tool = TodoStoreCompleteTool(store)

       add_result = await add_tool.execute(title="Task 1")
       todo_id = add_result.output["id"]

       result = await complete_tool.execute(id=todo_id)
       assert result.success
       assert result.output["completed"] is True

   async def test_todo_delete():
       store = TodoStore()
       add_tool = TodoStoreAddTool(store)
       delete_tool = TodoStoreDeleteTool(store)

       add_result = await add_tool.execute(title="Task 1")
       todo_id = add_result.output["id"]

       result = await delete_tool.execute(id=todo_id)
       assert result.success
   ```

2. **Error Handling Tests**:
   ```python
   async def test_todo_add_empty_title():
       store = TodoStore()
       add_tool = TodoStoreAddTool(store)
       result = await add_tool.execute(title="")
       assert not result.success
       assert "empty" in result.error.lower()

   async def test_todo_get_not_found():
       store = TodoStore()
       get_tool = TodoStoreGetTool(store)
       result = await get_tool.execute(id="nonexistent")
       assert not result.success
       assert "not found" in result.error.lower()
   ```

#### 2.3 Test Fixtures

**File**: `tests/conftest.py`

```python
import pytest
from src.tools.todo_store import TodoStore


@pytest.fixture
def todo_store():
    """Provide clean TodoStore instance for testing"""
    return TodoStore()


@pytest.fixture
def sample_todos():
    """Provide sample todo data for testing"""
    return [
        {"title": "Buy groceries"},
        {"title": "Finish assignment"},
        {"title": "Review code"},
    ]
```

**Exit Criteria**:

1. ✅ All calculator tests pass: `pytest tests/unit/test_calculator.py -v`
2. ✅ All todo store tests pass: `pytest tests/unit/test_todo_store.py -v`
3. ✅ Security tests pass (injection attempts blocked)
4. ✅ Coverage >80% for tool modules: `pytest tests/unit/ --cov=src/tools --cov-report=term`

**Coverage Target**: >80% for Tier 2

**Quality Gate**:

```bash
# Must pass before proceeding
pytest tests/unit/test_calculator.py tests/unit/test_todo_store.py -v
[ $? -eq 0 ] && echo "✅ Tests OK" || echo "❌ Tests Failed"
```

**Time Checkpoint**: Should complete within 45-60 minutes
**Red Flag**: If tests reveal major tool bugs, allocate 30m to fix before proceeding

---

### Phase 3: Planning Component

**Duration**: 30-40 minutes
**Priority**: MEDIUM - Simpler than it seems

#### 3.1 Pattern-Based Planner

**File**: `src/planning/planner.py`

**Design Decision**: Pattern-based (Option B) chosen over LLM integration for:

- ✅ Deterministic behavior (easier to test)
- ✅ No external API dependencies
- ✅ Fast execution
- ✅ Time-efficient implementation
- ❌ Limited to predefined patterns
- ❌ Can't handle novel requests

**Pattern Matching Strategy**:

1. **Calculator Patterns**:
   ```python
   CALCULATOR_PATTERNS = [
       (r"calculate (.+)", "Calculator", "expression"),
       (r"what is (.+)", "Calculator", "expression"),
       (r"compute (.+)", "Calculator", "expression"),
       (r"solve (.+)", "Calculator", "expression"),
       (r"^([\d\+\-\*/\(\)\s\.]+)$", "Calculator", "expression"),  # Direct math
   ]
   ```

2. **Todo Patterns**:
   ```python
   TODO_PATTERNS = [
       (r"add (?:a )?todo (?:to )?(.+)", "TodoStore.add", "title"),
       (r"create (?:a )?task (?:to )?(.+)", "TodoStore.add", "title"),
       (r"(?:show|list|get) (?:all )?(?:my )?(?:todos|tasks)", "TodoStore.list", None),
       (r"complete (?:todo|task) (.+)", "TodoStore.complete", "id"),
       (r"delete (?:todo|task) (.+)", "TodoStore.delete", "id"),
   ]
   ```

3. **Multi-Step Patterns** (Tier 2):
   ```python
   # Split on "and", "then", "and then"
   MULTI_STEP_SEPARATORS = [" and then ", " then ", " and "]

   def parse_multi_step(prompt: str) -> list[str]:
       """Split prompt into multiple sub-prompts"""
       for separator in MULTI_STEP_SEPARATORS:
           if separator in prompt.lower():
               return [p.strip() for p in prompt.split(separator)]
       return [prompt]
   ```

**Plan Structure**:

```python
class PlanStep(BaseModel):
    step_number: int
    tool: str  # e.g., "Calculator", "TodoStore.add"
    input: dict[str, Any]  # Tool-specific parameters
    reasoning: str  # Human-readable explanation


class Plan(BaseModel):
    plan_id: str  # UUID
    prompt: str  # Original user prompt
    steps: list[PlanStep]
    created_at: datetime


class Planner:
    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry

    def create_plan(self, prompt: str) -> Plan:
        """Convert natural language prompt to structured plan"""
        # 1. Parse multi-step prompts
        sub_prompts = self.parse_multi_step(prompt)

        # 2. Match each sub-prompt to tool pattern
        steps = []
        for i, sub_prompt in enumerate(sub_prompts):
            step = self.match_pattern(sub_prompt, i + 1)
            if step:
                steps.append(step)
            else:
                raise ValueError(f"Cannot parse: {sub_prompt}")

        # 3. Validate tools exist
        for step in steps:
            if not self.registry.get(step.tool):
                raise ValueError(f"Tool not found: {step.tool}")

        # 4. Create plan
        return Plan(
            plan_id=str(uuid.uuid4()),
            prompt=prompt,
            steps=steps,
            created_at=datetime.utcnow()
        )
```

**Input Validation**:

- Empty prompt: "Prompt cannot be empty"
- No pattern match: "Cannot understand request: {prompt}"
- Invalid tool reference: "Tool not found: {tool_name}"
- Ambiguous input: "Request is ambiguous, please be more specific"

**Example Transformations**:

```python
# Example 1: Simple calculator
Input: "calculate 2 + 2"
Output:
{
    "plan_id": "abc-123",
    "steps": [
        {
            "step_number": 1,
            "tool": "Calculator",
            "input": {"expression": "2 + 2"},
            "reasoning": "Evaluate arithmetic expression: 2 + 2"
        }
    ]
}

# Example 2: Multi-step todo
Input: "Add a todo to buy milk and then show me all my tasks"
Output:
{
    "plan_id": "def-456",
    "steps": [
        {
            "step_number": 1,
            "tool": "TodoStore.add",
            "input": {"title": "buy milk"},
            "reasoning": "Create new todo: buy milk"
        },
        {
            "step_number": 2,
            "tool": "TodoStore.list",
            "input": {},
            "reasoning": "Retrieve all todo items"
        }
    ]
}
```

**Planner Tests**:

```python
# tests/unit/test_planner.py
async def test_planner_calculator():
    registry = get_registry()
    planner = Planner(registry)
    plan = planner.create_plan("calculate 10 + 5")
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "Calculator"


async def test_planner_multi_step():
    registry = get_registry()
    planner = Planner(registry)
    plan = planner.create_plan("add todo buy milk and then list all todos")
    assert len(plan.steps) == 2
    assert plan.steps[0].tool == "TodoStore.add"
    assert plan.steps[1].tool == "TodoStore.list"


async def test_planner_invalid_tool():
    registry = get_registry()
    planner = Planner(registry)
    with pytest.raises(ValueError, match="cannot understand"):
        planner.create_plan("do something impossible")
```

**Exit Criteria**:

1. ✅ Calculator pattern matches: "calculate 2 + 2"
2. ✅ Todo patterns match: "add todo X", "list todos"
3. ✅ Multi-step parsing: "X and then Y" → 2 steps
4. ✅ Invalid tool rejection with clear error
5. ✅ Invalid prompt rejection with clear error

**Quality Gate**:

```bash
python -c "from src.planning.planner import Planner; from src.tools import get_registry; p=Planner(get_registry()); plan=p.create_plan('add todo test'); print('✅ Planner OK' if plan.steps else '❌ Planner Failed')"
```

**Time Checkpoint**: Should complete within 30-40 minutes
**Red Flag**: If >50 minutes, lock current patterns and move on

---

### Phase 4: Execution Orchestrator

**Duration**: 60-75 minutes
**Priority**: HIGH - Second most complex component

#### 4.1 Orchestrator Design

**File**: `src/orchestration/orchestrator.py`

**State Model**:

```python
class StepExecution(BaseModel):
    step_number: int
    tool: str
    input: dict[str, Any]
    output: Any | None = None
    status: Literal["pending", "running", "completed", "failed"]
    error: str | None = None
    attempts: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunState(BaseModel):
    run_id: str
    prompt: str
    status: Literal["pending", "running", "completed", "failed"]
    plan: Plan
    execution_log: list[StepExecution]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None  # Overall run error
```

**Retry Configuration** (Tier 2):

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3  # Total attempts per step
    initial_delay: float = 1.0  # Seconds
    backoff_multiplier: float = 2.0  # Exponential backoff
    max_delay: float = 10.0  # Maximum delay between retries

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt using exponential backoff"""
        delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_delay)
```

**Orchestrator Implementation**:

```python
class Orchestrator:
    def __init__(
            self,
            tool_registry: ToolRegistry,
            retry_config: RetryConfig | None = None
    ):
        self.registry = tool_registry
        self.retry_config = retry_config or RetryConfig()
        self._runs: dict[str, RunState] = {}  # In-memory state

    async def execute_run(self, run_id: str, plan: Plan) -> RunState:
        """Execute a plan and track state"""
        # 1. Initialize run state
        run_state = RunState(
            run_id=run_id,
            prompt=plan.prompt,
            status="pending",
            plan=plan,
            execution_log=[],
            created_at=datetime.utcnow()
        )
        self._runs[run_id] = run_state

        # 2. Execute steps sequentially
        run_state.status = "running"
        run_state.started_at = datetime.utcnow()

        try:
            for plan_step in plan.steps:
                step_execution = await self._execute_step_with_retry(plan_step)
                run_state.execution_log.append(step_execution)

                if step_execution.status == "failed":
                    run_state.status = "failed"
                    run_state.error = f"Step {step_execution.step_number} failed"
                    break
            else:
                run_state.status = "completed"

        except Exception as e:
            run_state.status = "failed"
            run_state.error = f"Execution error: {str(e)}"

        finally:
            run_state.completed_at = datetime.utcnow()

        return run_state

    async def _execute_step_with_retry(self, plan_step: PlanStep) -> StepExecution:
        """Execute single step with retry logic"""
        step_exec = StepExecution(
            step_number=plan_step.step_number,
            tool=plan_step.tool,
            input=plan_step.input,
            status="pending"
        )

        tool = self.registry.get(plan_step.tool)
        if not tool:
            step_exec.status = "failed"
            step_exec.error = f"Tool not found: {plan_step.tool}"
            return step_exec

        # Retry loop
        for attempt in range(1, self.retry_config.max_attempts + 1):
            step_exec.attempts = attempt
            step_exec.status = "running"
            step_exec.started_at = datetime.utcnow()

            try:
                result = await tool.execute(**plan_step.input)

                if result.success:
                    step_exec.status = "completed"
                    step_exec.output = result.output
                    step_exec.completed_at = datetime.utcnow()
                    return step_exec
                else:
                    # Tool returned error
                    step_exec.error = result.error
                    if attempt < self.retry_config.max_attempts:
                        delay = self.retry_config.get_delay(attempt)
                        await asyncio.sleep(delay)
                    else:
                        step_exec.status = "failed"

            except Exception as e:
                step_exec.error = str(e)
                if attempt < self.retry_config.max_attempts:
                    delay = self.retry_config.get_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    step_exec.status = "failed"

        step_exec.completed_at = datetime.utcnow()
        return step_exec

    def get_run(self, run_id: str) -> RunState | None:
        """Retrieve run state by ID"""
        return self._runs.get(run_id)
```

**Timeout Handling** (Tier 2):

```python
# Per-step timeout with asyncio
async def _execute_step_with_timeout(self, plan_step: PlanStep, timeout: float = 30.0) -> StepExecution:
    """Execute step with timeout"""
    try:
        return await asyncio.wait_for(
            self._execute_step_with_retry(plan_step),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return StepExecution(
            step_number=plan_step.step_number,
            tool=plan_step.tool,
            input=plan_step.input,
            status="failed",
            error=f"Step timed out after {timeout}s"
        )
```

**Idempotency Support** (Tier 3 - OPTIONAL):

```python
# Skip if behind schedule
class IdempotentOrchestrator(Orchestrator):
    """Orchestrator with idempotent retry capability"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_results_cache: dict[str, Any] = {}

    async def retry_failed_run(self, run_id: str) -> RunState:
        """Re-execute failed run, reusing completed steps"""
        run_state = self.get_run(run_id)
        if not run_state:
            raise ValueError(f"Run not found: {run_id}")

        # Resume from first failed step
        for i, step_exec in enumerate(run_state.execution_log):
            if step_exec.status == "failed":
                # Re-execute from this step
                remaining_steps = run_state.plan.steps[i:]
                # ... implementation

        return run_state
```

**Exit Criteria**:

1. ✅ Sequential execution: Steps execute in order
2. ✅ State tracking: Run status updated correctly
3. ✅ Retry logic: Exponential backoff works
4. ✅ Error handling: Failed steps don't crash orchestrator
5. ✅ Run retrieval: get_run() returns correct state

**Quality Gate**:

```bash
# Manual test of orchestration
python << 'EOF'
import asyncio
from src.orchestration.orchestrator import Orchestrator
from src.planning.planner import Planner
from src.tools import get_registry

async def test():
    registry = get_registry()
    planner = Planner(registry)
    orchestrator = Orchestrator(registry)

    plan = planner.create_plan("calculate 2 + 2")
    run_state = await orchestrator.execute_run("test-run", plan)

    assert run_state.status == "completed"
    print("✅ Orchestrator OK")

asyncio.run(test())
EOF
```

**Time Checkpoint**: Should complete within 60-75 minutes
**Red Flag**: If >100 minutes, skip idempotency, finalize retry logic

---

### Phase 5: REST API Implementation

**Duration**: 30-45 minutes
**Priority**: MEDIUM - FastAPI foundation exists

#### 5.1 API Endpoints

**File**: `src/api/routes/runs.py`

**Request/Response Schemas**:

```python
from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)


class CreateRunResponse(BaseModel):
    run_id: str
    status: str


class RunStepResponse(BaseModel):
    step_number: int
    tool: str
    input: dict
    output: Any | None
    status: str
    error: str | None
    attempts: int


class GetRunResponse(BaseModel):
    run_id: str
    prompt: str
    status: str
    execution_log: list[RunStepResponse]
    created_at: datetime
    completed_at: datetime | None
    error: str | None
```

**Endpoint Implementation**:

```python
from fastapi import APIRouter, HTTPException, status, Depends

router = APIRouter(prefix="/runs", tags=["runs"])


def get_orchestrator() -> Orchestrator:
    """Dependency injection for orchestrator"""
    if not hasattr(get_orchestrator, "_instance"):
        registry = get_registry()
        get_orchestrator._instance = Orchestrator(registry)
    return get_orchestrator._instance


def get_planner() -> Planner:
    """Dependency injection for planner"""
    if not hasattr(get_planner, "_instance"):
        registry = get_registry()
        get_planner._instance = Planner(registry)
    return get_planner._instance


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=CreateRunResponse)
async def create_run(
        request: CreateRunRequest,
        planner: Planner = Depends(get_planner),
        orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """
    Create and execute a new run from natural language prompt.

    **Status Codes**:
    - 201: Run created and execution started
    - 400: Invalid prompt or planning error
    - 500: Unexpected server error
    """
    try:
        # 1. Create plan from prompt
        plan = planner.create_plan(request.prompt)

        # 2. Generate run ID
        run_id = str(uuid.uuid4())

        # 3. Start execution (async, non-blocking)
        asyncio.create_task(orchestrator.execute_run(run_id, plan))

        return CreateRunResponse(
            run_id=run_id,
            status="pending"
        )

    except ValueError as e:
        # Planning error (invalid prompt, unknown tool, etc.)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{run_id}", response_model=GetRunResponse)
async def get_run(
        run_id: str,
        orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """
    Get complete state of a run including execution log.

    **Status Codes**:
    - 200: Run found and returned
    - 404: Run not found
    """
    run_state = orchestrator.get_run(run_id)

    if not run_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}"
        )

    return GetRunResponse(
        run_id=run_state.run_id,
        prompt=run_state.prompt,
        status=run_state.status,
        execution_log=[
            RunStepResponse(
                step_number=step.step_number,
                tool=step.tool,
                input=step.input,
                output=step.output,
                status=step.status,
                error=step.error,
                attempts=step.attempts
            )
            for step in run_state.execution_log
        ],
        created_at=run_state.created_at,
        completed_at=run_state.completed_at,
        error=run_state.error
    )
```

**Health Endpoint** (Already exists ✅):

```python
# src/api/routes/health.py
@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
```

**Main App Integration**:

```python
# src/api/main.py
from fastapi import FastAPI
from src.api.routes import health, runs

app = FastAPI(
    title="Crane AI Agent Runtime",
    description="Minimal AI agent runtime with tool execution",
    version="1.0.0"
)

app.include_router(health.router)
app.include_router(runs.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Error Handling Summary**:

| Error Type     | Status Code | Example                            |
|----------------|-------------|------------------------------------|
| Invalid prompt | 400         | "Cannot understand request"        |
| Planning error | 400         | "Tool not found: InvalidTool"      |
| Run not found  | 404         | "Run not found: {run_id}"          |
| Internal error | 500         | "Internal server error: {details}" |

**Exit Criteria**:

1. ✅ POST /runs creates run: Returns 201 with run_id
2. ✅ GET /runs/{run_id} retrieves state: Returns 200 with full state
3. ✅ GET /health works: Returns 200 with status
4. ✅ Invalid prompt returns 400: Clear error message
5. ✅ Invalid run_id returns 404: Clear error message

**Quality Gate**:

```bash
# Start server
uvicorn src.api.main:app --port 8000 &
sleep 2

# Test health
curl -s http://localhost:8000/health | grep -q "healthy" && echo "✅ Health OK" || echo "❌ Health Failed"

# Test create run
curl -s -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "calculate 2 + 2"}' | grep -q "run_id" && echo "✅ Create Run OK" || echo "❌ Create Run Failed"

# Cleanup
pkill -f "uvicorn src.api.main:app"
```

**Time Checkpoint**: Should complete within 30-45 minutes

---

### Phase 6: Integration Testing

**Duration**: 45-60 minutes
**Priority**: HIGH - Coverage gate for tier achievement

#### 6.1 Integration Test Suite

**File**: `tests/integration/test_full_flow.py`

**End-to-End Test Cases**:

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import time


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


async def test_full_calculator_flow(client):
    """Test complete flow: create run with calculator, retrieve result"""
    # 1. Create run
    response = client.post("/runs", json={"prompt": "calculate (10 + 5) * 2"})
    assert response.status_code == 201
    data = response.json()
    assert "run_id" in data
    run_id = data["run_id"]

    # 2. Wait for execution (simple polling)
    max_attempts = 10
    for _ in range(max_attempts):
        time.sleep(0.5)
        response = client.get(f"/runs/{run_id}")
        assert response.status_code == 200
        run_data = response.json()

        if run_data["status"] in ["completed", "failed"]:
            break

    # 3. Verify result
    assert run_data["status"] == "completed"
    assert len(run_data["execution_log"]) == 1
    step = run_data["execution_log"][0]
    assert step["status"] == "completed"
    assert step["output"] == 30.0


async def test_full_todo_flow(client):
    """Test complete flow: add todo, then list todos"""
    # 1. Create run with multi-step prompt
    response = client.post("/runs", json={
        "prompt": "add a todo to buy milk and then show me all my tasks"
    })
    assert response.status_code == 201
    run_id = response.json()["run_id"]

    # 2. Wait for execution
    max_attempts = 10
    for _ in range(max_attempts):
        time.sleep(0.5)
        response = client.get(f"/runs/{run_id}")
        run_data = response.json()
        if run_data["status"] in ["completed", "failed"]:
            break

    # 3. Verify multi-step execution
    assert run_data["status"] == "completed"
    assert len(run_data["execution_log"]) == 2

    # Step 1: add todo
    add_step = run_data["execution_log"][0]
    assert add_step["tool"] == "TodoStore.add"
    assert add_step["status"] == "completed"
    assert "id" in add_step["output"]

    # Step 2: list todos
    list_step = run_data["execution_log"][1]
    assert list_step["tool"] == "TodoStore.list"
    assert list_step["status"] == "completed"
    assert len(list_step["output"]) >= 1
    assert any(todo["title"] == "buy milk" for todo in list_step["output"])


async def test_retry_on_failure(client):
    """Test retry logic with simulated failure"""
    # This test requires a mock tool that fails initially then succeeds
    # Skip if time-constrained, or implement with a test-only tool
    pass


async def test_invalid_prompt_error(client):
    """Test error handling for invalid prompt"""
    response = client.post("/runs", json={"prompt": "do something impossible"})
    assert response.status_code == 400
    assert "cannot" in response.json()["detail"].lower()


async def test_run_not_found_error(client):
    """Test error handling for invalid run_id"""
    response = client.get("/runs/nonexistent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
```

#### 6.2 Coverage Analysis

**Coverage Command**:

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

**Coverage Targets**:

| Module             | Tier 1   | Tier 2   | Tier 3   |
|--------------------|----------|----------|----------|
| src/tools/         | >70%     | >80%     | >90%     |
| src/planning/      | >70%     | >80%     | >90%     |
| src/orchestration/ | >70%     | >80%     | >90%     |
| src/api/           | >60%     | >75%     | >85%     |
| **Overall**        | **>70%** | **>80%** | **>90%** |

**Coverage Report Location**: `htmlcov/index.html`

**Coverage Improvement Strategy** (if <80%):

1. Identify uncovered lines: `pytest --cov=src --cov-report=term-missing`
2. Add tests for error paths (common gap)
3. Test edge cases (empty inputs, boundary conditions)
4. Test exception handling (network errors, timeouts)
5. Backfill missing tests for Phase 1-4 components

**Exit Criteria**:

1. ✅ Calculator flow test passes: Create run → retrieve result
2. ✅ Todo flow test passes: Multi-step execution verified
3. ✅ Error handling tests pass: 400/404 responses correct
4. ✅ Overall coverage >80%: Meets Tier 2 requirement
5. ✅ All tests pass: `pytest tests/ -v` shows 100% pass rate

**Quality Gate**:

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term | grep "TOTAL" | awk '{if ($4 >= 80) print "✅ Coverage OK ("$4")"; else print "⚠️ Coverage Low ("$4")"}'
```

**Time Checkpoint**: Should complete within 45-60 minutes
**Red Flag**: If coverage <80% at 7h mark, accept current coverage and document in README

---

### Phase 7: Documentation

**Duration**: 30-45 minutes
**Priority**: MEDIUM - 10% of evaluation, tier differentiator

#### 7.1 README Structure

**File**: `README.md`

**Required Sections**:

```markdown
# Crane AI Agent Runtime

Minimal AI agent runtime that accepts natural language tasks, generates structured execution plans, and executes them
with robust error handling and retry logic.

## System Architecture

### Component Overview

```

┌─────────────┐
│ User │
│   (curl)    │
└──────┬──────┘
│ POST /runs {"prompt": "..."}
▼
┌─────────────────────────────────────┐
│ FastAPI REST API │
│  (/runs, /health)                   │
└──────┬──────────────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ Planner │
│  (Pattern-based NL → Plan)          │
└──────┬──────────────────────────────┘
│ Plan (steps)
▼
┌─────────────────────────────────────┐
│ Orchestrator │
│  (Sequential Execution + Retry)     │
└──────┬──────────────────────────────┘
│
▼
┌─────────────────────────────────────┐
│ Tool Registry │
│ - Calculator (AST-based)           │
│ - TodoStore (in-memory CRUD)       │
└─────────────────────────────────────┘

```

### Design Decisions & Trade-offs

1. **Pattern-Based Planner vs LLM Integration**
   - **Chosen**: Pattern-based regex matching
   - **Rationale**: Deterministic, fast, no external dependencies, easier to test
   - **Trade-off**: Limited to predefined patterns, can't handle novel requests
   - **Alternative**: LLM integration would be more flexible but adds complexity and unpredictability

2. **In-Memory State vs Persistent Storage**
   - **Chosen**: In-memory dict for run state
   - **Rationale**: Simple, fast, sufficient for POC demonstration
   - **Trade-off**: State lost on restart, no scalability to multiple instances
   - **Alternative**: Redis/database would enable persistence and scaling

3. **Sequential Execution vs Parallel**
   - **Chosen**: Sequential step-by-step execution
   - **Rationale**: Simpler orchestration, easier debugging, deterministic order
   - **Trade-off**: Slower for independent operations
   - **Alternative**: Parallel execution of independent steps would improve performance

4. **AST-Based Calculator vs eval()**
   - **Chosen**: AST parsing with explicit node visitor
   - **Rationale**: Security-first approach, prevents code injection
   - **Trade-off**: More complex implementation, limited to whitelisted operations
   - **Alternative**: eval() would be simpler but catastrophically insecure

## Setup Instructions

### Prerequisites
- Python 3.12+
- uv (or pip)
- Virtual environment tool

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd crane-challenge

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
uv sync --no-dev  # or: pip install -r requirements.txt

# 4. Verify installation
python -c "from src.tools import get_registry; print('✅ Setup OK')"
```

## Running the Application

### Start Server

```bash
# Option 1: Using uvicorn directly
uvicorn src.api.main:app --reload --port 8000

# Option 2: Using Python module
python -m challenge

# Option 3: Using make (if available)
make run
```

Server will start at: `http://localhost:8000`

## Example API Usage

### Health Check

```bash
curl http://localhost:8000/health

# Response:
# {"status": "healthy", "timestamp": "2025-10-29T12:00:00"}
```

### Create Run: Calculator

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "calculate (10 + 5) * 2"}'

# Response:
# {"run_id": "abc-123-...", "status": "pending"}
```

### Create Run: Multi-Step Todo

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "add a todo to buy milk and then show me all my tasks"}'

# Response:
# {"run_id": "def-456-...", "status": "pending"}
```

### Get Run State

```bash
curl http://localhost:8000/runs/abc-123

# Response:
# {
#   "run_id": "abc-123",
#   "prompt": "calculate (10 + 5) * 2",
#   "status": "completed",
#   "execution_log": [
#     {
#       "step_number": 1,
#       "tool": "Calculator",
#       "input": {"expression": "(10 + 5) * 2"},
#       "output": 30.0,
#       "status": "completed",
#       "error": null,
#       "attempts": 1
#     }
#   ],
#   "created_at": "2025-10-29T12:00:00",
#   "completed_at": "2025-10-29T12:00:01",
#   "error": null
# }
```

## Testing Instructions

### Run All Tests

```bash
# All tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_calculator.py -v

# Single test function
pytest tests/unit/test_calculator.py::test_calculator_basic_operations -v
```

### View Coverage Report

```bash
# Generate HTML report
pytest tests/ --cov=src --cov-report=html

# Open in browser (Mac)
open htmlcov/index.html

# Open in browser (Linux)
xdg-open htmlcov/index.html
```

### Security Verification

```bash
# Verify no eval/exec in calculator
grep -r "eval\|exec" src/tools/calculator.py

# Should return clean (only comments) or no results
```

## Known Limitations

### Current Implementation (Tier 2)

1. **Planning Limitations**
    - Pattern-based matching limited to ~10-15 predefined patterns
    - Cannot handle complex, novel, or ambiguous requests
    - Multi-step parsing limited to "and", "then" separators
    - No context awareness between steps

2. **State Management**
    - In-memory only: state lost on server restart
    - No persistence layer
    - Not scalable to multiple server instances
    - No state cleanup (memory leak potential for long-running servers)

3. **Execution Orchestration**
    - Sequential execution only (no parallel steps)
    - Simple retry logic (exponential backoff but no jitter)
    - No idempotency support for retry safety
    - No cancellation mechanism for running operations

4. **Tool Limitations**
    - Calculator: Limited to Tier 2 operators (no scientific functions)
    - TodoStore: No search, filter, or priority features
    - No tool versioning or hot-reload capability
    - No tool execution timeout per-tool customization

5. **API Limitations**
    - No authentication or rate limiting
    - No pagination for large execution logs
    - Polling required for run status (no webhooks/SSE)
    - No run cancellation endpoint

## Potential Improvements

### If I Had More Time

1. **LLM Integration** (2-3 hours)
    - Replace pattern-based planner with LLM (e.g., Ollama, OpenAI)
    - Structured output generation for better tool selection
    - Natural language reasoning explanations
    - Context awareness and multi-turn conversations

2. **Persistent State** (2-3 hours)
    - SQLite or PostgreSQL for run state persistence
    - Redis for fast state access and caching
    - State cleanup policies and archival
    - Database migrations for schema evolution

3. **Advanced Orchestration** (3-4 hours)
    - Parallel execution of independent steps
    - DAG-based execution planning (not just sequential)
    - Conditional branching based on step outcomes
    - Step dependencies and data passing between steps
    - Idempotency via step result caching

4. **Production Hardening** (4-5 hours)
    - Authentication and authorization (API keys, OAuth)
    - Rate limiting and request throttling
    - Comprehensive logging and observability (structured logs, metrics)
    - Health checks with dependency validation
    - Graceful shutdown and connection draining

5. **Enhanced Testing** (2-3 hours)
    - Property-based testing (Hypothesis)
    - Load testing and performance benchmarks
    - Chaos testing for retry logic validation
    - Mutation testing for test suite quality

6. **Developer Experience** (1-2 hours)
    - OpenAPI/Swagger UI for API exploration
    - WebSocket or SSE for real-time run updates
    - CLI tool for easier interaction
    - Docker containerization for easy deployment

## Test Coverage Summary

**Current Coverage**: >80% (Tier 2 target met)

| Module                            | Coverage | Status |
|-----------------------------------|----------|--------|
| src/tools/calculator.py           | >85%     | ✅      |
| src/tools/todo_store.py           | >85%     | ✅      |
| src/planning/planner.py           | >80%     | ✅      |
| src/orchestration/orchestrator.py | >80%     | ✅      |
| src/api/routes/                   | >75%     | ✅      |
| **Overall**                       | **>80%** | **✅**  |

**Uncovered Areas**:

- Edge cases in error recovery paths
- Some timeout handling branches
- Rare exception scenarios

## Technology Stack

- **Python**: 3.12+
- **Framework**: FastAPI (async REST API)
- **Testing**: pytest, pytest-cov
- **Validation**: Pydantic
- **Code Quality**: ruff (linting), type checker
- **Package Management**: uv (fast dependency management)

## Project Structure

```
crane-challenge/
├── src/
│   ├── tools/              # Tool implementations
│   │   ├── base.py         # Tool interface
│   │   ├── calculator.py   # AST-based calculator
│   │   └── todo_store.py   # In-memory todo CRUD
│   ├── planning/           # Planning component
│   │   └── planner.py      # Pattern-based planner
│   ├── orchestration/      # Execution orchestration
│   │   └── orchestrator.py # Sequential executor with retry
│   └── api/                # FastAPI application
│       ├── main.py         # App entry point
│       └── routes/         # API endpoints
├── tests/
│   ├── unit/               # Unit tests
│   │   ├── test_calculator.py
│   │   ├── test_todo_store.py
│   │   ├── test_planner.py
│   │   └── test_orchestrator.py
│   ├── integration/        # Integration tests
│   │   └── test_full_flow.py
│   └── conftest.py         # Shared fixtures
├── docs/                   # Documentation
├── README.md               # This file
└── pyproject.toml          # Project configuration
```

## License

[Your License]

## Author

[Your Name]

```

#### 7.2 Code Comments

**Critical Areas Requiring Comments**:

1. **Calculator AST Visitor**: Explain security rationale
2. **Retry Logic**: Explain exponential backoff calculation
3. **Pattern Matching**: Document regex patterns and why they exist
4. **State Management**: Document in-memory limitations
5. **Async Execution**: Explain asyncio.create_task() usage

**Exit Criteria**:
1. ✅ README complete with all required sections
2. ✅ Architecture diagram included (can be ASCII art)
3. ✅ Setup instructions tested and verified
4. ✅ Example curl commands work as documented
5. ✅ Trade-offs section honest and thoughtful
6. ✅ Known limitations documented
7. ✅ Potential improvements realistic and specific

**Quality Gate**:
```bash
# Verify README sections exist
grep -q "System Architecture" README.md && \
grep -q "Design Decisions" README.md && \
grep -q "Known Limitations" README.md && \
grep -q "Potential Improvements" README.md && \
echo "✅ README complete" || echo "❌ README incomplete"
```

**Time Checkpoint**: Should complete within 30-45 minutes
**Red Flag**: If >60 minutes, finalize current content and move to Phase 8

---

### Phase 8: Verification & Submission

**Duration**: 20-30 minutes
**Priority**: CRITICAL - Final quality gate

#### 8.1 Pre-Submission Checklist

**Functional Verification**:

```bash
# 1. Start server
uvicorn src.api.main:app --port 8000 &
sleep 2

# 2. Test health endpoint
curl -s http://localhost:8000/health | grep -q "healthy" && echo "✅ Health OK" || echo "❌ Health Failed"

# 3. Test calculator
CALC_RESULT=$(curl -s -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "calculate 2 + 2"}' | jq -r '.run_id')
sleep 1
curl -s http://localhost:8000/runs/$CALC_RESULT | grep -q "completed" && echo "✅ Calculator OK" || echo "❌ Calculator Failed"

# 4. Test todo flow
TODO_RESULT=$(curl -s -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "add todo buy milk and then list todos"}' | jq -r '.run_id')
sleep 1
curl -s http://localhost:8000/runs/$TODO_RESULT | grep -q "completed" && echo "✅ Todo OK" || echo "❌ Todo Failed"

# 5. Cleanup
pkill -f "uvicorn src.api.main:app"
```

**Security Verification**:

```bash
# Verify no eval/exec in calculator
grep -r "eval\|exec" src/tools/calculator.py && echo "🚨 SECURITY RISK" || echo "✅ Security OK"

# Run security tests
pytest tests/unit/test_calculator.py::test_calculator_blocks_eval_injection -v
```

**Test Coverage Verification**:

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term | tee coverage_report.txt

# Extract coverage percentage
COVERAGE=$(grep "TOTAL" coverage_report.txt | awk '{print $4}' | sed 's/%//')
if [ "$COVERAGE" -ge 80 ]; then
    echo "✅ Coverage OK ($COVERAGE%)"
else
    echo "⚠️ Coverage Low ($COVERAGE%)"
fi
```

**Code Quality Verification**:

```bash
# Run linter
ruff check src/ tests/

# Run type checker
ty check src/ tests/

# Check for common issues
find src/ -name "*.py" -exec grep -l "print(" {} \; | grep -v "__pycache__" && echo "⚠️ Debug prints found" || echo "✅ No debug prints"
```

#### 8.2 Submission Package Creation

**Git Repository Checklist**:

```bash
# 1. Verify all changes committed
git status  # Should show "working tree clean"

# 2. Verify commit history is clean
git log --oneline  # Check for clear commit messages

# 3. Create submission tag
git tag -a v1.0.0 -m "Crane AI Agent - Submission Ready"
git push origin v1.0.0
```

**Zip Package Creation** (if not using git):

```bash
# Create submission zip
zip -r crane-ai-agent-submission.zip \
  src/ \
  tests/ \
  docs/ \
  README.md \
  CLAUDE.md \
  pyproject.toml \
  pytest.ini \
  requirements.txt \
  .env.example \
  -x "*.pyc" "*.pyo" "__pycache__/*" ".pytest_cache/*" ".ruff_cache/*" "*.egg-info/*" "htmlcov/*" ".venv/*"

# Verify zip contents
unzip -l crane-ai-agent-submission.zip | head -20
```

**Final Documentation Check**:

```bash
# Verify README sections
cat README.md | grep -E "^##" | sort

# Expected sections:
# ## System Architecture
# ## Setup Instructions
# ## Running the Application
# ## Example API Usage
# ## Testing Instructions
# ## Known Limitations
# ## Potential Improvements
# ## Test Coverage Summary
```

#### 8.3 Submission Email/Upload

**Repository Submission** (Preferred):

1. Create private GitHub repository
2. Push all code with clean commit history
3. Add evaluator as collaborator
4. Include README.md as repository home page
5. Send repository URL via email

**Zip File Submission** (Alternative):

1. Create submission zip package
2. Upload to file sharing service (Google Drive, Dropbox, etc.)
3. Share link with evaluator via email
4. Ensure link has proper access permissions

**Submission Email Template**:

```
Subject: Crane AI Agent Take-Home Submission - [Your Name]

Hello,

Please find my submission for the AI Agent take-home assignment:

Repository: [GitHub URL] (or) Zip File: [Download Link]

**Implementation Summary**:
- Time spent: ~7 hours
- Tier achieved: Tier 2 (>80% test coverage)
- Language: Python 3.12 with FastAPI
- Key features: AST-based calculator, in-memory todo store, pattern-based planner, retry logic

**Testing**:
- Overall test coverage: 82%
- All unit and integration tests passing
- Security tests verify no eval/exec vulnerabilities

**Documentation**:
- Comprehensive README with architecture, setup, examples
- Honest trade-offs and limitations documented
- Potential improvements outlined

I'm looking forward to discussing the design decisions and architecture in the technical interview.

Best regards,
[Your Name]
```

**Exit Criteria**:

1. ✅ All tests pass: `pytest tests/ -v`
2. ✅ Coverage >80%: Verified in report
3. ✅ Security check passes: No eval/exec, injection tests pass
4. ✅ Server starts and responds: Health and execute endpoints work
5. ✅ README complete: All sections present
6. ✅ Submission package created: Zip or repository ready
7. ✅ Clean commit history: Clear, meaningful commit messages

**Quality Gate**: Final verification script

```bash
#!/bin/bash
# verify_submission.sh

echo "=== CRANE AI AGENT SUBMISSION VERIFICATION ==="
echo ""

# Test execution
echo "1. Running tests..."
pytest tests/ -q --tb=no
TEST_RESULT=$?

# Coverage check
echo "2. Checking coverage..."
COVERAGE=$(pytest tests/ --cov=src --cov-report=term 2>/dev/null | grep TOTAL | awk '{print $4}' | sed 's/%//')
if [ "$COVERAGE" -ge 80 ]; then
    echo "   ✅ Coverage: $COVERAGE%"
else
    echo "   ⚠️ Coverage: $COVERAGE% (target: 80%)"
fi

# Security check
echo "3. Security verification..."
if grep -r "eval\|exec" src/tools/calculator.py 2>/dev/null | grep -v "^#" | grep -v "docstring"; then
    echo "   🚨 SECURITY RISK: eval/exec found in calculator"
else
    echo "   ✅ Security: No eval/exec in calculator"
fi

# Documentation check
echo "4. Documentation check..."
if [ -f "README.md" ] && grep -q "Design Decisions" README.md && grep -q "Known Limitations" README.md; then
    echo "   ✅ README complete"
else
    echo "   ⚠️ README incomplete"
fi

# Final verdict
echo ""
echo "=== FINAL VERDICT ==="
if [ $TEST_RESULT -eq 0 ] && [ "$COVERAGE" -ge 80 ]; then
    echo "✅ READY TO SUBMIT"
else
    echo "⚠️ REVIEW REQUIRED"
fi
```

**Time Checkpoint**: Should complete within 20-30 minutes

---

## 3. Technical Requirements by Tier

### 3.1 Tool System

#### Tier 1 Requirements (Minimum Viable - 60-70%)

**Calculator**:

- Basic operators: `+`, `-`, `*`, `/`
- Integer arithmetic only
- Simple error messages
- Basic AST implementation (even if limited)

**TodoStore**:

- `add`: Create todo (returns ID)
- `list`: Retrieve all todos
- In-memory storage with basic dict

**Tool Interface**:

- name, description, input_schema defined
- execute() method returns success/output/error
- Basic error handling

**Evaluation Focus**: "Does it work?"

#### Tier 2 Requirements (TARGET - 75-85%)

**Calculator** (Enhanced):

- ✅ All Tier 1 features
- ✅ Decimal numbers: `3.14 + 2.86`
- ✅ Negative numbers: `-5 * 3`
- ✅ Parentheses: `(10 + 5) * 2`
- ✅ AST-based evaluation (NO eval/exec)
- ✅ Comprehensive error messages
- ✅ Input validation

**TodoStore** (Full CRUD):

- ✅ All Tier 1 features
- ✅ `get`: Retrieve todo by ID
- ✅ `complete`: Mark todo as done
- ✅ `delete`: Remove todo
- ✅ Proper state management
- ✅ Error handling (ID not found, etc.)

**Tool Interface**:

- ✅ Consistent error handling across all tools
- ✅ Type-safe input validation
- ✅ Comprehensive input schemas
- ✅ Tool registry with dynamic lookup

**Testing**:

- ✅ >80% test coverage
- ✅ Unit tests for all operations
- ✅ Security tests (injection attempts blocked)
- ✅ Error path testing

**Evaluation Focus**: "Is it well-engineered?"

#### Tier 3 Requirements (Stretch - 85-95%)

**Calculator** (Scientific):

- ✅ All Tier 2 features
- ✅ Scientific functions: `sqrt`, `pow`, `sin`, `cos`, `log`
- ✅ Constants: `pi`, `e`
- ✅ >90% test coverage
- ✅ Edge case handling (NaN, infinity, etc.)

**TodoStore** (Advanced):

- ✅ All Tier 2 features
- ✅ `update`: Modify todo title
- ✅ `filter`: Filter by completed status
- ✅ `search`: Search by title substring
- ✅ `priority`: Priority levels (high/medium/low)
- ✅ Sorting and pagination

**Tool System**:

- ✅ Tool versioning support
- ✅ Hot-reload capability
- ✅ Per-tool timeout configuration
- ✅ Tool execution metrics

**Testing**:

- ✅ >90% test coverage
- ✅ Property-based testing (Hypothesis)
- ✅ Comprehensive edge cases
- ✅ Performance benchmarks

**Evaluation Focus**: "Is this production-grade?"

### 3.2 Planning Component

#### Tier 1 (Minimum)

**Pattern Matching**:

- 5-7 basic patterns
- Calculator: "calculate X"
- Todo: "add todo X", "list todos"
- Single-step only

**Validation**:

- Basic tool existence check
- Simple error messages

#### Tier 2 (TARGET)

**Pattern Matching**:

- ✅ 10-15 comprehensive patterns
- ✅ Multiple variations per operation
- ✅ Calculator: calculate, compute, solve, what is, direct math
- ✅ Todo: add, create, list, show, complete, delete
- ✅ Multi-step parsing: "X and then Y", "X then Y", "X and Y"

**Validation**:

- ✅ Tool existence validation
- ✅ Input schema validation against tool requirements
- ✅ Clear error messages for invalid prompts
- ✅ Edge case handling (empty, ambiguous, impossible)

**Plan Quality**:

- ✅ Reasoning field explains each step
- ✅ Proper step numbering and sequencing
- ✅ Structured JSON output

#### Tier 3 (Stretch)

**Advanced Features**:

- ✅ Context awareness between steps
- ✅ Variable passing (use output from step N in step N+1)
- ✅ Conditional branching
- ✅ Loop detection and prevention

### 3.3 Orchestrator

#### Tier 1 (Minimum)

**Execution**:

- Sequential execution (one step at a time)
- Basic state tracking (run status)
- Simple error handling

**State Management**:

- In-memory dict
- run_id, status, execution_log

#### Tier 2 (TARGET)

**Execution**:

- ✅ Sequential execution with proper ordering
- ✅ Comprehensive state tracking
- ✅ Retry logic with exponential backoff
    - max_attempts: 3
    - initial_delay: 1.0s
    - backoff_multiplier: 2.0
    - max_delay: 10.0s
- ✅ Timeout handling per step
- ✅ Detailed execution logs

**State Management**:

- ✅ Complete RunState model
- ✅ StepExecution with attempt tracking
- ✅ Timestamps (created_at, started_at, completed_at)
- ✅ Error tracking at run and step levels

**Error Handling**:

- ✅ Graceful failure handling
- ✅ Clear error messages
- ✅ State preserved on failure
- ✅ No crashes on tool errors

#### Tier 3 (Stretch)

**Advanced Features**:

- ✅ Idempotency support (safe retry of failed runs)
- ✅ Parallel execution of independent steps
- ✅ Cancellation mechanism
- ✅ Progress callbacks/streaming
- ✅ Step result caching

### 3.4 REST API

#### Tier 1 (Minimum)

**Endpoints**:

- POST /runs: Create run (basic)
- GET /runs/{run_id}: Get state (basic)
- GET /health: Health check

**Error Handling**:

- Basic status codes (200, 400, 500)

#### Tier 2 (TARGET)

**Endpoints**:

- ✅ POST /runs: Create run with validation
    - Request: {"prompt": "..."}
    - Response: {"run_id": "...", "status": "pending"}
    - Status codes: 201, 400, 500
- ✅ GET /runs/{run_id}: Complete run state
    - Response includes full execution log
    - Status codes: 200, 404
- ✅ GET /health: Health check
    - Response: {"status": "healthy", "timestamp": "..."}
    - Status code: 200

**Request/Response**:

- ✅ Pydantic schemas for validation
- ✅ Type-safe models
- ✅ Clear error responses

**Error Handling**:

- ✅ Proper HTTP status codes
- ✅ Actionable error messages
- ✅ Standard exception → HTTPException mapping
- ✅ Input validation errors

**Integration**:

- ✅ FastAPI dependency injection
- ✅ Async/await throughout
- ✅ OpenAPI documentation (automatic)

#### Tier 3 (Stretch)

**Advanced Features**:

- ✅ Pagination for execution logs
- ✅ Run cancellation endpoint (DELETE /runs/{run_id})
- ✅ WebSocket or SSE for real-time updates
- ✅ Filtering and search endpoints
- ✅ Authentication and rate limiting

---

## 4. Architecture & Design

### 4.1 System Architecture

**Layered Architecture**:

```
┌────────────────────────────────────────────────┐
│              User/Client                        │
│          (curl, Postman, browser)               │
└───────────────────┬────────────────────────────┘
                    │ HTTP Requests
                    ▼
┌────────────────────────────────────────────────┐
│           API Layer (FastAPI)                   │
│  ┌──────────────────────────────────────────┐  │
│  │ Routes:  /runs, /health                   │  │
│  │ Schemas: Request/Response models          │  │
│  │ Errors:  Exception → HTTPException        │  │
│  └──────────────────────────────────────────┘  │
└───────────────────┬────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────┐
│         Planning Layer (Planner)                │
│  ┌──────────────────────────────────────────┐  │
│  │ Pattern Matching:  Regex-based           │  │
│  │ Multi-step Parsing: "and", "then"        │  │
│  │ Validation: Tool existence, schemas      │  │
│  │ Output: Structured Plan (JSON)           │  │
│  └──────────────────────────────────────────┘  │
└───────────────────┬────────────────────────────┘
                    │ Plan
                    ▼
┌────────────────────────────────────────────────┐
│    Orchestration Layer (Orchestrator)           │
│  ┌──────────────────────────────────────────┐  │
│  │ Execution: Sequential step-by-step       │  │
│  │ Retry: Exponential backoff (3 attempts)  │  │
│  │ State: Complete run tracking             │  │
│  │ Error Handling: Graceful failures        │  │
│  └──────────────────────────────────────────┘  │
└───────────────────┬────────────────────────────┘
                    │ Tool Calls
                    ▼
┌────────────────────────────────────────────────┐
│           Tool Layer (Tools)                    │
│  ┌──────────────────────────────────────────┐  │
│  │ Registry: Dynamic tool lookup             │  │
│  │ Calculator: AST-based evaluation          │  │
│  │ TodoStore: In-memory CRUD                 │  │
│  │ Interface: Consistent Tool protocol       │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
```

### 4.2 Data Flow

**Complete Request Flow**:

```
1. User → POST /runs {"prompt": "calculate 2 + 2"}
         │
         ▼
2. API   → Validate request (Pydantic)
         → Extract prompt
         │
         ▼
3. Planner → Match pattern: "calculate X"
           → Create Plan:
             {
               "plan_id": "abc-123",
               "steps": [
                 {
                   "step_number": 1,
                   "tool": "Calculator",
                   "input": {"expression": "2 + 2"},
                   "reasoning": "Evaluate arithmetic"
                 }
               ]
             }
         │
         ▼
4. API   → Generate run_id: "run-456"
         → Start async execution
         → Return: {"run_id": "run-456", "status": "pending"}
         │
         ▼
5. Orchestrator → Initialize RunState
                → For each step in plan:
                  → Execute with retry
                  → Track attempts, errors
                  → Update execution_log
                → Update status: "completed"
                │
                ▼
6. Tool (Calculator) → Parse "2 + 2" with AST
                     → Evaluate: 4.0
                     → Return: ToolResult(success=True, output=4.0)
                │
                ▼
7. Orchestrator → Record in execution_log
                → Complete RunState

8. User → GET /runs/run-456
        │
        ▼
9. API  → Retrieve RunState from orchestrator
        → Return complete state with execution_log
```

### 4.3 Design Patterns

**Factory Pattern** (Tool Registry):

```python
class ToolRegistry:
    """Central factory for tool instances"""

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)
```

**Strategy Pattern** (Retry Logic):

```python
class RetryConfig:
    """Configurable retry strategy"""

    def get_delay(self, attempt: int) -> float:
# Exponential backoff calculation
```

**State Pattern** (Run State Management):

```python
class RunState:
    status: Literal["pending", "running", "completed", "failed"]
    # State transitions tracked
```

**Dependency Injection** (FastAPI):

```python
@router.post("/runs")
async def create_run(
        request: CreateRunRequest,
        planner: Planner = Depends(get_planner),
        orchestrator: Orchestrator = Depends(get_orchestrator)
):
# Clean, testable, decoupled
```

### 4.4 Error Handling Strategy

**Layered Error Handling**:

1. **Tool Layer**:
    - Catch all exceptions
    - Return ToolResult(success=False, error="message")
    - Never raise exceptions to caller

2. **Orchestrator Layer**:
    - Retry on tool failures
    - Track errors in StepExecution
    - Mark run as failed if step fails after retries
    - Never crash on tool errors

3. **API Layer**:
    - Catch ValueError → 400 Bad Request
    - Catch KeyError → 404 Not Found
    - Catch Exception → 500 Internal Server Error
    - Return clear JSON error messages

**Error Message Quality**:

- ❌ Bad: "Error"
- ❌ Bad: "Something went wrong"
- ✅ Good: "Cannot divide by zero in expression: 5 / 0"
- ✅ Good: "Todo not found: abc-123"
- ✅ Good: "Cannot understand request: 'do something impossible'. Supported patterns: calculate, add todo, list todos"

### 4.5 State Management

**In-Memory State** (Tier 2):

```python
class Orchestrator:
    def __init__(self):
        self._runs: dict[str, RunState] = {}  # run_id → RunState
```

**Trade-offs**:

- ✅ Pros: Simple, fast, no dependencies
- ❌ Cons: Lost on restart, no persistence, not scalable

**Persistent State** (Future):

```python
# Option 1: SQLite
class PersistentOrchestrator(Orchestrator):
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)


# Option 2: Redis
class RedisOrchestrator(Orchestrator):
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
```

---

## 5. Testing Strategy

### 5.1 Test Pyramid

```
        ┌─────────┐
        │  E2E    │  (1 test: full flow)
        │  Tests  │
      ┌─┴─────────┴─┐
      │ Integration  │  (2-3 tests: API + orchestrator)
      │    Tests     │
    ┌─┴──────────────┴─┐
    │   Unit Tests      │  (15-20 tests: tools, planner)
    │   (Majority)      │
    └───────────────────┘
```

**Test Distribution**:

- **Unit Tests**: 80% of tests (fast, isolated, high coverage)
- **Integration Tests**: 15% of tests (API + orchestrator + tools)
- **E2E Tests**: 5% of tests (full user flow)

### 5.2 Unit Test Coverage by Module

**Calculator Tool** (`tests/unit/test_calculator.py`):

- ✅ Basic operations (+, -, *, /)
- ✅ Decimal numbers
- ✅ Negative numbers
- ✅ Parentheses
- ✅ Order of operations
- ✅ Division by zero error
- ✅ Invalid syntax error
- ✅ Security: eval injection blocked (MANDATORY)
- ✅ Security: exec injection blocked (MANDATORY)
- ✅ Security: __import__ blocked (MANDATORY)

**TodoStore Tool** (`tests/unit/test_todo_store.py`):

- ✅ Add todo
- ✅ List todos (empty, with items)
- ✅ Get todo by ID
- ✅ Complete todo
- ✅ Delete todo
- ✅ Get non-existent ID error
- ✅ Empty title error
- ✅ State isolation between operations

**Planner** (`tests/unit/test_planner.py`):

- ✅ Single-step calculator pattern
- ✅ Single-step todo patterns (add, list)
- ✅ Multi-step parsing ("and then")
- ✅ Invalid prompt error
- ✅ Invalid tool reference error
- ✅ Empty prompt error

**Orchestrator** (`tests/unit/test_orchestrator.py`):

- ✅ Sequential execution
- ✅ State tracking (pending → running → completed)
- ✅ Retry on failure (with mock)
- ✅ Exponential backoff calculation
- ✅ Step failure handling
- ✅ Run failure on step failure

### 5.3 Integration Test Coverage

**Full Flow Test** (`tests/integration/test_full_flow.py`):

- ✅ Calculator flow: Create run → Execute → Retrieve result
- ✅ Todo flow: Multi-step (add + list) → Retrieve results
- ✅ Error handling: Invalid prompt → 400 response
- ✅ Error handling: Invalid run_id → 404 response

**API Integration** (`tests/integration/test_api.py`):

- ✅ POST /runs with valid prompt
- ✅ POST /runs with invalid prompt
- ✅ GET /runs/{run_id} for completed run
- ✅ GET /runs/{run_id} for non-existent run
- ✅ GET /health

### 5.4 Security Test Coverage

**Calculator Security** (MANDATORY):

```python
@pytest.mark.parametrize("injection", [
    "__import__('os').system('echo hacked')",
    "exec('print(\"hacked\")')",
    "eval('2+2')",
    "__builtins__",
    "globals()",
    "locals()",
    "open('/etc/passwd')",
])
async def test_calculator_blocks_injection(injection):
    calc = CalculatorTool(tier=2)
    result = await calc.execute(expression=injection)
    assert not result.success
    assert "not supported" in result.error.lower() or "invalid" in result.error.lower()
```

### 5.5 Coverage Targets

**By Tier**:

| Tier | Overall | Tools | Planning | Orchestration | API  |
|------|---------|-------|----------|---------------|------|
| 1    | >70%    | >70%  | >60%     | >60%          | >50% |
| 2    | >80%    | >85%  | >80%     | >80%          | >75% |
| 3    | >90%    | >95%  | >90%     | >90%          | >85% |

**Priority Coverage Areas** (Must be >90%):

1. Calculator evaluation logic (security critical)
2. Retry logic (correctness critical)
3. Error handling paths (robustness critical)

**Acceptable Lower Coverage** (<80%):

1. API boilerplate (FastAPI handles much)
2. Logging and debugging code
3. Rare exception scenarios

---

## 6. Quality Gates & Checkpoints

### 6.1 Phase Checkpoints

**After Phase 3 (~3-3.5 hours)**:

```
Status Check:
- Tools implemented and tested? ✅
- Planner creating valid plans? ✅
- Coverage >75%? ✅

Decision:
- ✅ On track for Tier 2/3
- ⚠️ Behind schedule? Cut Tier 3 features
- 🚨 Major issues? Triage to Tier 1
```

**After Phase 5 (~5-6 hours)**:

```
Status Check:
- Orchestrator executing plans? ✅
- Retry logic working? ✅
- API endpoints responding? ✅
- Coverage >80%? ✅

Decision:
- ✅ On track for Tier 2
- ⚠️ Coverage <80%? Allocate 30m to testing
- 🚨 Major bugs? Fix before documentation
```

**After Phase 6 (~6-7 hours)**:

```
Status Check:
- All tests passing? ✅
- Coverage ≥80%? ✅
- Integration tests pass? ✅
- Security verified? ✅

Decision:
- ✅ Tier 2 achieved, proceed to documentation
- ⚠️ Coverage 75-80%? Accept and document
- ⚠️ Tests failing? Emergency 30m fix window
- 🚨 <1h left? Ship now, skip Phase 7
```

**After Phase 7 (~6.5-7.5 hours)**:

```
Status Check:
- README complete? ✅
- Trade-offs documented? ✅
- Examples tested? ✅

Decision:
- ✅ Proceed to Phase 8 quickly
- ⚠️ >8h total? Finalize immediately
```

**After Phase 8 (~7-8 hours)**:

```
Final Check:
- All tests pass? ✅
- Server starts? ✅
- Example works? ✅
- README complete? ✅
- Submission ready? ✅

Action:
- ✅ SUBMIT NOW
```

### 6.2 Red Flags by Phase

**Phase 1 Red Flags**:

- > 150 minutes elapsed
- Calculator still using eval/exec
- Tests not written alongside code
- **Action**: Cut Tier 3 features immediately

**Phase 3 Red Flags**:

- > 50 minutes on planner
- Trying to handle too many edge cases
- Overthinking pattern complexity
- **Action**: Lock current patterns, move on

**Phase 4 Red Flags**:

- > 100 minutes on orchestrator
- Implementing idempotency (Tier 3 feature)
- Complex state management
- **Action**: Skip idempotency, simplify state

**Phase 6 Red Flags**:

- Coverage <75%
- Multiple test failures
- Integration tests not passing
- **Action**: Emergency 30m fix window, then accept

**Phase 7 Red Flags**:

- > 60 minutes on documentation
- Trying to write perfect docs
- **Action**: Finalize current content, move to Phase 8

**Overall Red Flags**:

- > 8 hours total time
- **Action**: SHIP NOW with what you have

### 6.3 Emergency Triage Procedures

**Scenario: <4 hours remaining, Phase 1 incomplete**

```
Triage Plan:
1. STOP Tier 3 features
2. Focus: Get Tier 2 tools working
3. Skip advanced retry logic
4. Minimal documentation
5. Target: Tier 1.5 (65-70%)
```

**Scenario: <2 hours remaining, API not working**

```
Emergency Plan:
1. STOP all new development
2. Fix critical blocker ONLY
3. Get one example working (calculator)
4. Minimal tests (just security + one happy path)
5. Basic README with setup
6. SUBMIT
```

**Scenario: <1 hour remaining**

```
SHIP NOW:
1. Server starts? ✅ → Ship
2. Server fails? ❌ → 30m emergency fix, then ship anyway
3. README exists? ✅ → Ship
4. README missing? ❌ → 5m minimal README, ship
5. STOP ALL WORK at 7:30h mark
6. SUBMIT WHAT YOU HAVE
```

---

## 7. Implementation Priorities

### 7.1 Must-Have Features (ALL Tiers)

**Critical Path** (Cannot submit without):

1. **Security**: AST-based calculator (no eval/exec)
2. **Functionality**: POST /runs works with calculator AND todo
3. **Testing**: Tests exist and >75% pass
4. **Documentation**: README with setup instructions
5. **Submission**: Package created and verified

**Verification**:

```bash
# Quick verification
./verify_submission.sh | grep "READY TO SUBMIT"
```

### 7.2 Tier 1 Features (60-70%)

**Minimum Viable Product**:

- Calculator: +, -, *, / (integers)
- TodoStore: add, list
- Planner: 5 basic patterns, single-step
- Orchestrator: Sequential execution (no retry)
- API: POST /runs, GET /runs/{id}, GET /health
- Tests: Some tests passing
- README: Setup + examples

**Time Estimate**: 4-5 hours
**When to Accept**: If >7h elapsed and Tier 2 not achievable

### 7.3 Tier 2 Features (75-85%) - TARGET

**Target Achievement**:

- ✅ All Tier 1 features
- ✅ Calculator: decimals, negatives, parentheses
- ✅ TodoStore: full CRUD (add, list, get, complete, delete)
- ✅ Planner: 10-15 patterns, multi-step ("and then")
- ✅ Orchestrator: Retry with exponential backoff
- ✅ **>80% test coverage**
- ✅ Comprehensive error handling
- ✅ README: architecture, trade-offs, limitations

**Time Estimate**: 6-8 hours
**Priority**: PRIMARY TARGET

**Differentiators from Tier 1**:

- Coverage: >80% vs >70%
- Retry: Exponential backoff vs none
- Multi-step: "and then" parsing vs single-step
- Errors: Actionable messages vs basic
- Documentation: Trade-offs vs basic

### 7.4 Tier 3 Features (85-95%) - STRETCH

**Advanced Features** (Only if ahead of schedule):

- Calculator: sqrt, pow, sin, cos, log
- TodoStore: update, filter, search, priority
- Orchestrator: Idempotency support
- Tests: >90% coverage, property-based testing
- Documentation: Comprehensive with deployment guide

**Time Estimate**: 8-10 hours
**Priority**: OPTIONAL - Only if Phases 1-6 completed in <6h

**When to Skip**: If >6h elapsed and Tier 2 not complete

### 7.5 Feature Prioritization Matrix

| Feature          | Tier 1       | Tier 2          | Tier 3          | Complexity | Skip If Behind? |
|------------------|--------------|-----------------|-----------------|------------|-----------------|
| AST Calculator   | ✅ Basic      | ✅ Enhanced      | ✅ Scientific    | High       | ❌ Never         |
| TodoStore CRUD   | ✅ Add/List   | ✅ Full CRUD     | ✅ Advanced      | Medium     | ⚠️ Tier 3 only  |
| Planner Patterns | ✅ 5 patterns | ✅ 15 patterns   | ✅ Context-aware | Medium     | ⚠️ Lock at 10   |
| Retry Logic      | ❌ None       | ✅ Exponential   | ✅ + Idempotency | High       | ⚠️ Tier 3 only  |
| Test Coverage    | 70%          | 80%             | 90%             | High       | ⚠️ Accept 75%   |
| Documentation    | Basic        | Comprehensive   | Perfect         | Low        | ✅ Can compress  |
| Security Tests   | ✅ Basic      | ✅ Comprehensive | ✅ Exhaustive    | Medium     | ❌ Never         |

---

## 8. Risk Management

### 8.1 Technical Risks

**Risk 1: AST Calculator Complexity**

- **Probability**: Medium (40%)
- **Impact**: Critical (blocks submission)
- **Symptoms**: >90 minutes on Phase 1, eval/exec still present, tests failing
- **Mitigation**:
    - Start with simple NodeVisitor pattern
    - Use standard library ast module examples
    - Limit to basic operators initially (Tier 1)
    - Add Tier 2 operators incrementally
- **Fallback**: If >120m, freeze Tier 1 features, move on
- **Detection**: `grep -r "eval\|exec" src/tools/calculator.py`

**Risk 2: Test Coverage Gaps**

- **Probability**: High (60%)
- **Impact**: High (tier reduction)
- **Symptoms**: <80% coverage at Phase 6 checkpoint
- **Mitigation**:
    - Write tests alongside code (not at end)
    - Focus on critical paths first
    - Use pytest-cov to track coverage continuously
    - Prioritize error path testing
- **Fallback**: Accept 75-80% coverage, document gaps in README
- **Detection**: `pytest --cov=src --cov-report=term | grep TOTAL`

**Risk 3: Orchestrator Complexity**

- **Probability**: Medium (40%)
- **Impact**: High (Tier 2 feature loss)
- **Symptoms**: >100m on Phase 4, retry logic buggy
- **Mitigation**:
    - Keep state management simple (in-memory dict)
    - Use asyncio.sleep() for delays (not complex queue)
    - Test retry with mocked failures
    - Skip idempotency if behind
- **Fallback**: Basic sequential execution without retry
- **Detection**: Phase 4 timer >75m

**Risk 4: Pattern Planner Limitations**

- **Probability**: Low (20%)
- **Impact**: Medium (may not handle all prompts)
- **Symptoms**: Test prompts not matching patterns
- **Mitigation**:
    - Start with assignment example prompts
    - Add patterns incrementally
    - Document limitations clearly
    - Test with example prompts first
- **Fallback**: Document unsupported patterns in README
- **Detection**: Integration tests failing on prompt parsing

**Risk 5: Time Overrun**

- **Probability**: High (70%)
- **Impact**: Variable (depends on phase)
- **Symptoms**: Checkpoints exceeded (see Phase Red Flags)
- **Mitigation**:
    - Track time per phase religiously
    - Cut features proactively at red flags
    - Use emergency triage procedures
    - Accept lower tier completion over perfection
- **Fallback**: Emergency procedures (<4h, <2h, <1h)
- **Detection**: Phase timer comparisons

### 8.2 Quality Risks

**Risk 1: Error Handling Gaps**

- **Mitigation**: Test error paths explicitly, use pytest.raises
- **Fallback**: Document known error cases in README

**Risk 2: Security Vulnerabilities**

- **Mitigation**: Mandatory security tests, AST-only calculator
- **Fallback**: NONE - security is non-negotiable

**Risk 3: Documentation Rushed**

- **Mitigation**: Template-based approach, honest limitations
- **Fallback**: Basic README better than perfect README

### 8.3 Risk Monitoring

**Continuous Monitoring**:

```bash
# Time tracking
./track_time.sh summary  # Check against phase estimates

# Coverage tracking
pytest tests/ --cov=src --cov-report=term | grep TOTAL

# Security verification
grep -r "eval\|exec" src/tools/calculator.py

# Test health
pytest tests/ -q --tb=no | tail -1
```

**Decision Points**:

- After Phase 3 (~3.5h): Tier 3 features decision
- After Phase 5 (~6h): Coverage backfill decision
- After Phase 6 (~7h): Ship vs fix decision
- At 7.5h mark: Force submission decision

---

## 9. Success Metrics

### 9.1 Functional Success

**Core Functionality** (Pass/Fail):

- [ ] Server starts without errors
- [ ] GET /health returns 200
- [ ] POST /runs creates run with calculator prompt
- [ ] POST /runs creates run with todo prompt
- [ ] GET /runs/{id} retrieves run state
- [ ] Calculator evaluates: `(10 + 5) * 2` = `30.0`
- [ ] Todo flow: add → list returns added item

**Verification Command**:

```bash
./verify_submission.sh | grep "✅"
```

### 9.2 Quality Metrics

**Test Coverage** (Tier 2 Target):

- Overall: >80%
- Tools: >85%
- Planning: >80%
- Orchestration: >80%
- API: >75%

**Test Pass Rate**:

- Unit tests: 100%
- Integration tests: 100%
- Security tests: 100%

**Verification Command**:

```bash
pytest tests/ --cov=src --cov-report=term
pytest tests/ -v
```

### 9.3 Security Metrics

**Security Verification** (Pass/Fail - CRITICAL):

- [ ] No `eval()` in calculator.py
- [ ] No `exec()` in calculator.py
- [ ] Injection test: `__import__('os')` blocked
- [ ] Injection test: `exec('print')` blocked
- [ ] Injection test: `eval('2+2')` blocked

**Verification Command**:

```bash
grep -r "eval\|exec" src/tools/calculator.py && echo "🚨 SECURITY FAILURE" || echo "✅ Security OK"
pytest tests/unit/test_calculator.py::test_calculator_blocks_eval_injection -v
```

### 9.4 Documentation Metrics

**README Completeness** (Checklist):

- [ ] System architecture section
- [ ] Setup instructions tested
- [ ] Example curl commands work
- [ ] Testing instructions clear
- [ ] Design decisions documented
- [ ] Trade-offs explained
- [ ] Known limitations honest
- [ ] Potential improvements realistic

**Verification Command**:

```bash
grep -c "^##" README.md  # Should be >8 sections
```

### 9.5 Tier Achievement

**Tier 1 (60-70%)**:

- Basic features working
- Some tests passing
- Minimal documentation
- **Verdict**: Submitted but incomplete

**Tier 2 (75-85%)** - TARGET:

- All core features working
- > 80% test coverage
- Comprehensive error handling
- Trade-offs documented
- **Verdict**: Strong engineering fundamentals

**Tier 3 (85-95%)**:

- Advanced features implemented
- > 90% test coverage
- Idempotency support
- Comprehensive documentation
- **Verdict**: Production-grade implementation

**Final Tier Assessment**:

```bash
# Automated tier prediction
python << 'EOF'
import subprocess
import re

# Check coverage
cov_result = subprocess.run(
    ["pytest", "tests/", "--cov=src", "--cov-report=term"],
    capture_output=True, text=True
)
cov_match = re.search(r"TOTAL.*?(\d+)%", cov_result.stdout)
coverage = int(cov_match.group(1)) if cov_match else 0

# Check tests pass
test_result = subprocess.run(
    ["pytest", "tests/", "-q", "--tb=no"],
    capture_output=True, text=True
)
tests_pass = test_result.returncode == 0

# Check README
import os
has_readme = os.path.exists("README.md")
has_tradeoffs = False
if has_readme:
    with open("README.md") as f:
        has_tradeoffs = "Trade-offs" in f.read()

# Tier calculation
score = 0
score += 30 if tests_pass else 0
score += min(40, coverage * 0.5)  # Up to 40 points for coverage
score += 15 if has_readme else 0
score += 15 if has_tradeoffs else 0

print(f"Coverage: {coverage}%")
print(f"Tests Pass: {tests_pass}")
print(f"README: {has_readme}")
print(f"Trade-offs: {has_tradeoffs}")
print(f"\nEstimated Score: {score}%")

if score >= 85:
    print("Predicted Tier: 3 (Excellent)")
elif score >= 75:
    print("Predicted Tier: 2 (Target Achieved)")
elif score >= 60:
    print("Predicted Tier: 1 (Minimum Viable)")
else:
    print("Predicted Tier: 0 (Incomplete)")
EOF
```

---

## 10. Deliverables Checklist

### 10.1 Source Code

**Required Structure**:

```
crane-challenge/
├── src/
│   ├── __init__.py
│   ├── tools/
│   │   ├── __init__.py          # Tool registry
│   │   ├── base.py              # Tool interface
│   │   ├── calculator.py        # AST-based calculator
│   │   └── todo_store.py        # In-memory CRUD
│   ├── planning/
│   │   ├── __init__.py
│   │   └── planner.py           # Pattern-based planner
│   ├── orchestration/
│   │   ├── __init__.py
│   │   └── orchestrator.py      # Sequential executor + retry
│   └── api/
│       ├── __init__.py
│       ├── main.py              # FastAPI app
│       └── routes/
│           ├── __init__.py
│           ├── health.py        # Health check
│           └── runs.py          # /runs endpoints
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/
│   │   ├── test_calculator.py
│   │   ├── test_todo_store.py
│   │   ├── test_planner.py
│   │   └── test_orchestrator.py
│   └── integration/
│       └── test_full_flow.py
├── docs/                        # Implementation guides (optional)
├── README.md                    # Main documentation
├── CLAUDE.md                    # Project instructions (existing)
├── pyproject.toml               # Project config
├── pytest.ini                   # Test config
├── requirements.txt             # Dependencies
└── .env.example                 # Environment template
```

**Code Quality Standards**:

- [ ] Type hints on all functions
- [ ] Docstrings on all public functions
- [ ] No debug print() statements
- [ ] No commented-out code blocks
- [ ] Consistent naming (snake_case for functions/variables)
- [ ] Proper error messages (actionable, clear)
- [ ] Clean git history (meaningful commit messages)

### 10.2 Tests

**Test Files Required**:

- [ ] `tests/unit/test_calculator.py` (10+ tests)
- [ ] `tests/unit/test_todo_store.py` (8+ tests)
- [ ] `tests/unit/test_planner.py` (5+ tests)
- [ ] `tests/unit/test_orchestrator.py` (6+ tests)
- [ ] `tests/integration/test_full_flow.py` (3+ tests)
- [ ] `tests/conftest.py` (fixtures)

**Test Execution**:

```bash
# All tests must pass
pytest tests/ -v

# Coverage must be >80%
pytest tests/ --cov=src --cov-report=html
```

### 10.3 Documentation

**README.md Sections** (Required):

- [x] System Architecture (diagram + explanation)
- [x] Setup Instructions (step-by-step, tested)
- [x] Running the Application (commands that work)
- [x] Example API Usage (curl commands with expected output)
- [x] Testing Instructions (how to run tests, view coverage)
- [x] Design Decisions and Trade-offs (why choices made)
- [x] Known Limitations (honest assessment)
- [x] Potential Improvements (if more time)
- [x] Test Coverage Summary (percentage breakdown)
- [x] Technology Stack (versions, dependencies)
- [x] Project Structure (directory layout)

**Documentation Quality**:

- [ ] No broken commands (all examples tested)
- [ ] No placeholder text ("TODO", "FIXME")
- [ ] Clear, concise writing
- [ ] Code examples properly formatted
- [ ] Honest about limitations (not over-promising)

### 10.4 Submission Package

**Format Options**:

**Option 1: Git Repository** (Preferred):

```bash
# Clean repository
git status  # Should be clean
git log --oneline  # Check commit history

# Create tag
git tag -a v1.0.0 -m "Crane AI Agent - Submission"

# Push to private repository
git push origin main
git push origin v1.0.0

# Share repository URL with evaluator
```

**Option 2: Zip File**:

```bash
# Create submission package
zip -r crane-ai-agent-submission.zip \
  src/ \
  tests/ \
  docs/ \
  README.md \
  CLAUDE.md \
  pyproject.toml \
  pytest.ini \
  requirements.txt \
  .env.example \
  -x "*.pyc" "__pycache__/*" ".pytest_cache/*" "htmlcov/*" ".venv/*"

# Verify contents
unzip -l crane-ai-agent-submission.zip
```

**Excluded from Submission**:

- `.venv/` (virtual environment)
- `__pycache__/` (Python cache)
- `.pytest_cache/` (test cache)
- `htmlcov/` (coverage reports)
- `.DS_Store` (macOS artifacts)
- `*.pyc`, `*.pyo` (compiled Python)
- `.env` (local environment variables)

### 10.5 Final Verification Checklist

**Pre-Submission Checks**:

```bash
# 1. Clean environment test
rm -rf .venv __pycache__ .pytest_cache htmlcov
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Server starts
uvicorn src.api.main:app --port 8000 &
sleep 2
curl http://localhost:8000/health
pkill -f "uvicorn src.api.main:app"

# 3. Tests pass
pytest tests/ -v

# 4. Coverage verified
pytest tests/ --cov=src --cov-report=term | grep TOTAL

# 5. Security check
grep -r "eval\|exec" src/tools/calculator.py && echo "FAIL" || echo "PASS"

# 6. README exists
test -f README.md && grep -q "Design Decisions" README.md && echo "PASS" || echo "FAIL"

# 7. Clean git status
git status | grep "working tree clean" && echo "PASS" || echo "WARN"
```

**Submission Checklist**:

- [ ] All tests pass (100% pass rate)
- [ ] Coverage >80% (Tier 2 target)
- [ ] Security verified (no eval/exec)
- [ ] Server starts and responds
- [ ] README complete with all sections
- [ ] Git history clean (or zip created)
- [ ] Submission email/link prepared
- [ ] Submitted at least 1 day before interview

---

## Appendix A: Time Tracking

**Phase Timer Script** (`track_time.sh`):

```bash
#!/bin/bash
# Track time per phase

TIME_LOG=".phase_times.log"

case "$1" in
    start)
        echo "$(date +%s) $2 START" >> $TIME_LOG
        echo "⏱️ Started Phase $2 at $(date '+%H:%M:%S')"
        ;;
    end)
        LAST=$(tail -1 $TIME_LOG | awk '{print $1" "$2}')
        START_TIME=$(echo $LAST | cut -d' ' -f1)
        PHASE=$(echo $LAST | cut -d' ' -f2)
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        MINUTES=$((DURATION / 60))
        echo "$(date +%s) $PHASE END $DURATION" >> $TIME_LOG
        echo "⏱️ Completed Phase $PHASE in ${MINUTES}m"
        ;;
    summary)
        echo "=== TIME SUMMARY ==="
        awk '/START/{phase=$2; start=$1} /END/{if($2==phase) print phase": "int(($3)/60)"m"}' $TIME_LOG
        TOTAL=$(awk '/END/{sum+=$3} END{print int(sum/60)}' $TIME_LOG)
        echo "TOTAL: ${TOTAL}m"
        ;;
esac
```

**Usage**:

```bash
./track_time.sh start 1    # Start Phase 1
# ... work on phase ...
./track_time.sh end         # End current phase
./track_time.sh summary     # View time report
```

---

## Appendix B: Emergency Procedures

### Emergency Procedure 1: Phase Overrun

**Trigger**: Any phase >125% of estimated time

**Procedure**:

1. STOP current work
2. Assess completeness: 80% done? 50% done? 20% done?
3. Decision:
    - If >80%: Finish current phase (max +15m)
    - If 50-80%: Simplify to minimum viable
    - If <50%: Cut feature, move to next phase
4. Update tier target if needed
5. Continue with adjusted scope

### Emergency Procedure 2: Coverage <80% at Phase 6

**Trigger**: Coverage report shows <80% at Phase 6 checkpoint

**Procedure**:

1. Run: `pytest --cov=src --cov-report=term-missing`
2. Identify uncovered lines (prioritize critical paths)
3. Allocate 30 minutes for backfill:
    - 0-15m: Cover critical paths (calculator eval, retry logic)
    - 15-25m: Cover common error paths
    - 25-30m: Re-check coverage
4. If still <80%:
    - Accept current coverage
    - Document gaps in README Known Limitations
    - Proceed to Phase 7
5. If >75%: Acceptable for Tier 1.5-2

### Emergency Procedure 3: Tests Failing at Phase 6

**Trigger**: Multiple test failures at integration test phase

**Procedure**:

1. Categorize failures:
    - Critical (calculator, todo CRUD, API): FIX IMMEDIATELY
    - Important (error handling, edge cases): FIX IF TIME
    - Nice-to-have (advanced features): SKIP
2. Allocate 30 minutes for fixes:
    - 0-20m: Fix critical failures
    - 20-30m: Fix important failures if critical done
3. If still failing:
    - Mark failing tests with `@pytest.mark.skip(reason="...")`
    - Document in README Known Limitations
    - Proceed to Phase 7 if >75% tests pass
4. If <75% pass: EMERGENCY MODE (see below)

### Emergency Procedure 4: <2 Hours Remaining

**Trigger**: <2 hours to deadline, significant work remaining

**EMERGENCY TRIAGE**:

1. STOP ALL NEW FEATURES
2. Assess current state:
    - Server starts? If NO → 30m emergency fix
    - One example works? If NO → 45m get calculator working
    - Tests exist? If NO → 30m write critical tests
3. Minimal documentation:
    - 15m: Basic README (setup + one example)
4. SUBMIT at 1:30 remaining

### Emergency Procedure 5: <1 Hour Remaining

**TRIGGER**: <1 hour to deadline

**SHIP NOW PROTOCOL**:

1. STOP ALL WORK
2. Quick verification:
    - Server starts: YES → Ship | NO → 15m fix attempt
    - One curl command works: YES → Ship | NO → Not shippable
3. 5-minute README:
    - Copy .env.example → README
    - Add: "Run: uvicorn src.api.main:app"
    - Add: "Test: curl http://localhost:8000/health"
4. Create submission package (5m)
5. SUBMIT IMMEDIATELY

**No more work after this point - SUBMIT WHAT EXISTS**

---

## Appendix C: Example Prompts for Testing

**Calculator Prompts**:

```
"calculate 2 + 2"
"what is (10 + 5) * 2"
"compute -5 * 3"
"solve 3.14 + 2.86"
"2 + 2"  # Direct math
```

**Todo Prompts**:

```
"add a todo to buy milk"
"add todo finish assignment"
"list all my todos"
"show me all my tasks"
```

**Multi-Step Prompts**:

```
"add a todo to buy milk and then show me all my tasks"
"create a task to study then list todos"
"add todo call mom and list all todos"
```

**Error Cases**:

```
"do something impossible"
"calculate xyz + abc"
"add todo "  # Empty title
```

---

## Document Control

**Version**: 1.0
**Status**: Implementation Ready
**Last Updated**: 2025-10-29
**Target Tier**: Tier 2 (75-85% score, >80% coverage)
**Time Budget**: 6-8 hours

**Prepared For**: Crane Engineering Team Take-Home Assignment
**Document Purpose**: Comprehensive implementation guide for AI Agent Runtime POC

**References**:

- `take-home-requirements.md`: Official assignment requirements
- `QUICK_REFERENCE_CHECKLIST.md`: One-page implementation checklist
- `CLAUDE.md`: Project-specific development guidelines

---

**END OF PRODUCT REQUIREMENTS DOCUMENT**
