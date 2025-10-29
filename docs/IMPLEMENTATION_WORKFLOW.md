# Implementation Workflow: Crane AI Agent Runtime

**Project**: Minimal AI Agent Runtime POC
**Target**: Tier 2 (75-85% score, >80% coverage)
**Time Budget**: 6-8 hours
**Current Status**: Phase 0 Complete ✅

---

## 📋 Quick Reference

| Phase | Duration | Status | Deliverable |
|-------|----------|--------|-------------|
| **0. Setup** | 15-20m | ✅ COMPLETE | FastAPI foundation, health endpoint |
| **1. Tools** | 90-120m | ⬜ TODO | Calculator (AST), TodoStore (CRUD) |
| **2. Tests** | 45-60m | ⬜ TODO | Unit tests, >80% coverage |
| **3. Planner** | 30-40m | ⬜ TODO | Pattern-based NL→Plan |
| **4. Orchestrator** | 60-75m | ⬜ TODO | Sequential execution, retry logic |
| **5. API** | 30-45m | ⬜ TODO | POST /runs, GET /runs/{id} |
| **6. Integration** | 45-60m | ⬜ TODO | E2E tests, coverage verification |
| **7. Documentation** | 30-45m | ⬜ TODO | README with trade-offs |
| **8. Verification** | 20-30m | ⬜ TODO | Submission package |
| **TOTAL** | **6-8h** | - | Complete AI Agent Runtime |

---

## 🎯 Success Criteria (Tier 2)

✅ **Functional**: POST /runs executes calculator + todo tasks
✅ **Quality**: >80% test coverage verified
✅ **Security**: AST-based calculator (NO eval/exec)
✅ **Documentation**: Comprehensive README with trade-offs
✅ **Submission**: All verification checks pass

---

## 🚀 Phase 1: Tool System Implementation

**Duration**: 90-120 minutes
**Priority**: CRITICAL PATH - Security Sensitive

### Entry Criteria
- [x] Phase 0 complete (project structure exists)
- [ ] Virtual environment activated: `source .venv/bin/activate`
- [ ] Server can start: Health endpoint responding

### Implementation Tasks

#### Task 1.1: Create Base Tool Interface

**File**: `src/challenge/tools/base.py`

```python
"""Base tool interfaces and models."""

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standard tool execution result."""

    success: bool = Field(..., description="Whether execution succeeded")
    output: Any | None = Field(None, description="Result value if successful")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] | None = Field(None, description="Additional execution metadata")


class ToolMetadata(BaseModel):
    """Tool capability description."""

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Human-readable description")
    input_schema: dict[str, Any] = Field(..., description="JSON schema for inputs")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "calculator",
                "description": "Evaluate arithmetic expressions",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]
                }
            }
        }


class BaseTool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute tool with validated inputs."""
        pass
```

**Verification**:
```bash
python -c "from src.challenge.tools.base import BaseTool, ToolResult, ToolMetadata; print('✅ Base interfaces OK')"
```

---

#### Task 1.2: Create Calculator Tool (SECURITY CRITICAL)

**File**: `src/challenge/tools/calculator.py`

```python
"""Calculator tool with AST-based expression evaluation.

🔒 SECURITY CRITICAL: Uses AST parsing to prevent code injection.
NEVER use eval() or exec() - they enable arbitrary code execution.
"""

import ast
import operator
from typing import Any

from src.challenge.tools.base import BaseTool, ToolMetadata, ToolResult


class SafeCalculator(ast.NodeVisitor):
    """AST-based expression evaluator - NO eval/exec."""

    # Tier 2: Basic operators + decimals + negatives + parentheses
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,  # Unary minus for negative numbers
    }

    def visit_BinOp(self, node: ast.BinOp) -> float:
        """Visit binary operation node."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)

        if op_type not in self.OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        operator_func = self.OPERATORS[op_type]

        # Handle division by zero
        if op_type == ast.Div and right == 0:
            raise ValueError("Cannot divide by zero")

        return operator_func(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        """Visit unary operation node (e.g., -5)."""
        operand = self.visit(node.operand)
        op_type = type(node.op)

        if op_type not in self.OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

        operator_func = self.OPERATORS[op_type]
        return operator_func(operand)

    def visit_Constant(self, node: ast.Constant) -> float:
        """Visit constant node (numbers)."""
        value = node.value
        if not isinstance(value, (int, float)):
            raise ValueError(f"Only numeric constants allowed, got {type(value).__name__}")
        return float(value)

    def visit_Num(self, node: ast.Num) -> float:
        """Visit num node (for older Python versions)."""
        return float(node.n)

    def generic_visit(self, node: ast.AST) -> Any:
        """Catch-all for unsupported nodes."""
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """Calculator tool using safe AST evaluation."""

    @property
    def metadata(self) -> ToolMetadata:
        """Get calculator metadata."""
        return ToolMetadata(
            name="calculator",
            description="Safely evaluate arithmetic expressions (Tier 2: +, -, *, /, decimals, negatives, parentheses)",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate",
                        "examples": ["2 + 2", "(10 - 3) * 4", "-5.5 / 2"]
                    }
                },
                "required": ["expression"]
            }
        )

    async def execute(self, expression: str) -> ToolResult:
        """Execute calculator with safe AST evaluation.

        Args:
            expression: Arithmetic expression string

        Returns:
            ToolResult with calculated value or error
        """
        try:
            # Parse expression into AST
            tree = ast.parse(expression, mode='eval')

            # Evaluate using safe visitor
            calculator = SafeCalculator()
            result = calculator.visit(tree.body)

            return ToolResult(
                success=True,
                output=result,
                metadata={"expression": expression}
            )

        except SyntaxError as e:
            return ToolResult(
                success=False,
                error=f"Invalid syntax: {str(e)}"
            )
        except ValueError as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Calculation error: {str(e)}"
            )
```

**Security Verification** (MANDATORY):
```bash
# CRITICAL: Verify no eval/exec in calculator
grep -rn "eval\|exec" src/challenge/tools/calculator.py
# Expected: No matches (or only in comments/docstrings)

# Verify AST usage
grep -rn "ast.parse\|ast.NodeVisitor" src/challenge/tools/calculator.py
# Expected: Should find matches
```

---

#### Task 1.3: Create TodoStore Tool

**File**: `src/challenge/tools/todo_store.py`

```python
"""TodoStore tool for in-memory todo management."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.challenge.tools.base import BaseTool, ToolMetadata, ToolResult


class TodoStoreTool(BaseTool):
    """In-memory todo storage with CRUD operations."""

    def __init__(self):
        """Initialize empty todo store."""
        self.todos: dict[str, dict[str, Any]] = {}

    @property
    def metadata(self) -> ToolMetadata:
        """Get TodoStore metadata."""
        return ToolMetadata(
            name="todo_store",
            description="Manage todos with CRUD operations (Tier 2: add, list, get, complete, delete)",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "get", "complete", "delete"],
                        "description": "Action to perform"
                    },
                    "text": {
                        "type": "string",
                        "description": "Todo text (for 'add' action)"
                    },
                    "todo_id": {
                        "type": "string",
                        "description": "Todo ID (for get/complete/delete actions)"
                    }
                },
                "required": ["action"]
            }
        )

    async def execute(
        self,
        action: str,
        text: str | None = None,
        todo_id: str | None = None,
        **kwargs
    ) -> ToolResult:
        """Execute todo operation.

        Args:
            action: Operation to perform (add/list/get/complete/delete)
            text: Todo text for 'add' action
            todo_id: Todo ID for get/complete/delete actions

        Returns:
            ToolResult with operation result or error
        """
        try:
            if action == "add":
                return await self._add_todo(text)
            elif action == "list":
                return await self._list_todos()
            elif action == "get":
                return await self._get_todo(todo_id)
            elif action == "complete":
                return await self._complete_todo(todo_id)
            elif action == "delete":
                return await self._delete_todo(todo_id)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"TodoStore error: {str(e)}"
            )

    async def _add_todo(self, text: str | None) -> ToolResult:
        """Add new todo."""
        if not text:
            return ToolResult(success=False, error="Text is required for 'add' action")

        todo_id = str(uuid4())
        todo = {
            "id": todo_id,
            "text": text,
            "completed": False,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        self.todos[todo_id] = todo

        return ToolResult(
            success=True,
            output=todo,
            metadata={"action": "add", "count": len(self.todos)}
        )

    async def _list_todos(self) -> ToolResult:
        """List all todos."""
        todos_list = list(self.todos.values())
        return ToolResult(
            success=True,
            output=todos_list,
            metadata={"action": "list", "count": len(todos_list)}
        )

    async def _get_todo(self, todo_id: str | None) -> ToolResult:
        """Get specific todo."""
        if not todo_id:
            return ToolResult(success=False, error="todo_id is required for 'get' action")

        if todo_id not in self.todos:
            return ToolResult(success=False, error=f"Todo not found: {todo_id}")

        return ToolResult(
            success=True,
            output=self.todos[todo_id],
            metadata={"action": "get"}
        )

    async def _complete_todo(self, todo_id: str | None) -> ToolResult:
        """Mark todo as completed."""
        if not todo_id:
            return ToolResult(success=False, error="todo_id is required for 'complete' action")

        if todo_id not in self.todos:
            return ToolResult(success=False, error=f"Todo not found: {todo_id}")

        self.todos[todo_id]["completed"] = True
        self.todos[todo_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

        return ToolResult(
            success=True,
            output=self.todos[todo_id],
            metadata={"action": "complete"}
        )

    async def _delete_todo(self, todo_id: str | None) -> ToolResult:
        """Delete todo."""
        if not todo_id:
            return ToolResult(success=False, error="todo_id is required for 'delete' action")

        if todo_id not in self.todos:
            return ToolResult(success=False, error=f"Todo not found: {todo_id}")

        deleted_todo = self.todos.pop(todo_id)

        return ToolResult(
            success=True,
            output=deleted_todo,
            metadata={"action": "delete", "remaining": len(self.todos)}
        )
```

---

#### Task 1.4: Create Tool Registry

**File**: `src/challenge/tools/__init__.py`

```python
"""Tool system exports."""

from src.challenge.tools.base import BaseTool, ToolMetadata, ToolResult
from src.challenge.tools.calculator import CalculatorTool
from src.challenge.tools.todo_store import TodoStoreTool

__all__ = [
    "BaseTool",
    "ToolMetadata",
    "ToolResult",
    "CalculatorTool",
    "TodoStoreTool",
    "get_tool_registry",
]


def get_tool_registry() -> dict[str, BaseTool]:
    """Get registry of all available tools.

    Returns:
        Dictionary mapping tool names to tool instances
    """
    calculator = CalculatorTool()
    todo_store = TodoStoreTool()

    return {
        "calculator": calculator,
        "todo_store": todo_store,
    }
```

**Verification**:
```bash
python -c "from src.challenge.tools import get_tool_registry; tools = get_tool_registry(); print(f'✅ {len(tools)} tools registered: {list(tools.keys())}')"
```

### Quality Gate: Phase 1 Exit Criteria

Run ALL verification commands:

```bash
# 1. Security check (CRITICAL - must pass)
echo "🔒 Security Verification..."
grep -rn "eval\|exec" src/challenge/tools/calculator.py
# Expected: No matches

# 2. Import check
echo "📦 Import Verification..."
python -c "from src.challenge.tools import CalculatorTool, TodoStoreTool, get_tool_registry; print('✅ All imports OK')"

# 3. Tool registry check
echo "🔧 Tool Registry Verification..."
python -c "from src.challenge.tools import get_tool_registry; tools = get_tool_registry(); assert len(tools) == 2, 'Expected 2 tools'; print('✅ Tool registry OK')"

# 4. Basic functionality check
echo "⚡ Functionality Verification..."
python << 'EOF'
import asyncio
from src.challenge.tools import get_tool_registry

async def test_tools():
    tools = get_tool_registry()

    # Test calculator
    calc_result = await tools["calculator"].execute(expression="2 + 2")
    assert calc_result.success, "Calculator failed"
    assert calc_result.output == 4.0, f"Expected 4.0, got {calc_result.output}"
    print("✅ Calculator works")

    # Test todo store
    todo_result = await tools["todo_store"].execute(action="add", text="Test todo")
    assert todo_result.success, "TodoStore add failed"
    print("✅ TodoStore works")

asyncio.run(test_tools())
EOF
```

**All checks must pass before proceeding to Phase 2.**

### Time Checkpoint

```bash
# Expected: ~90-120 minutes elapsed
# If >120 minutes: Consider emergency fallback
```

### Emergency Fallback (if >120 min)

**ACTION**: Simplify TodoStore
**CUT**: Remove get, complete, delete methods
**KEEP**: Only add and list
**TIME SAVED**: ~20-30 minutes

---

## 🧪 Phase 2: Tool Testing

**Duration**: 45-60 minutes
**Priority**: CRITICAL - Coverage Target >80%

### Entry Criteria
- [x] Phase 1 complete and verified
- [ ] All Phase 1 quality gates passed
- [ ] Tools importable and functional

### Implementation Tasks

#### Task 2.1: Create Calculator Tests

**File**: `tests/unit/test_calculator.py`

```python
"""Unit tests for Calculator tool."""

import pytest
from src.challenge.tools.calculator import CalculatorTool


@pytest.fixture
def calculator():
    """Create calculator instance."""
    return CalculatorTool()


# Tier 1: Basic arithmetic
@pytest.mark.asyncio
async def test_addition(calculator):
    """Test basic addition."""
    result = await calculator.execute(expression="2 + 2")
    assert result.success
    assert result.output == 4.0


@pytest.mark.asyncio
async def test_subtraction(calculator):
    """Test basic subtraction."""
    result = await calculator.execute(expression="5 - 3")
    assert result.success
    assert result.output == 2.0


@pytest.mark.asyncio
async def test_multiplication(calculator):
    """Test basic multiplication."""
    result = await calculator.execute(expression="3 * 4")
    assert result.success
    assert result.output == 12.0


@pytest.mark.asyncio
async def test_division(calculator):
    """Test basic division."""
    result = await calculator.execute(expression="8 / 2")
    assert result.success
    assert result.output == 4.0


# Tier 2: Decimals, negatives, parentheses
@pytest.mark.asyncio
async def test_decimal_numbers(calculator):
    """Test decimal number support."""
    result = await calculator.execute(expression="3.14 + 2.86")
    assert result.success
    assert abs(result.output - 6.0) < 0.001  # Float comparison


@pytest.mark.asyncio
async def test_negative_numbers(calculator):
    """Test negative number support."""
    result = await calculator.execute(expression="-5 * 3")
    assert result.success
    assert result.output == -15.0


@pytest.mark.asyncio
async def test_parentheses(calculator):
    """Test parentheses for order of operations."""
    result = await calculator.execute(expression="(10 + 5) * 2")
    assert result.success
    assert result.output == 30.0


@pytest.mark.asyncio
async def test_order_of_operations(calculator):
    """Test PEMDAS order of operations."""
    result = await calculator.execute(expression="2 + 3 * 4")
    assert result.success
    assert result.output == 14.0  # Not 20


# Error handling
@pytest.mark.asyncio
async def test_division_by_zero(calculator):
    """Test division by zero error handling."""
    result = await calculator.execute(expression="5 / 0")
    assert not result.success
    assert "zero" in result.error.lower()


@pytest.mark.asyncio
async def test_invalid_syntax(calculator):
    """Test invalid syntax error handling."""
    result = await calculator.execute(expression="2 + + 2")
    assert not result.success
    assert "syntax" in result.error.lower()


# SECURITY TESTS (MANDATORY)
@pytest.mark.asyncio
async def test_blocks_eval_injection(calculator):
    """Test that eval injection attempts are blocked."""
    malicious_inputs = [
        "__import__('os').system('ls')",
        "exec('print(1)')",
        "eval('1+1')",
        "__builtins__",
    ]

    for malicious in malicious_inputs:
        result = await calculator.execute(expression=malicious)
        assert not result.success, f"Security vulnerability: {malicious} was not blocked!"


@pytest.mark.asyncio
async def test_blocks_attribute_access(calculator):
    """Test that attribute access is blocked."""
    result = await calculator.execute(expression="(2).__class__")
    assert not result.success
```

**Run Tests**:
```bash
pytest tests/unit/test_calculator.py -v
# Expected: All tests pass
```

---

#### Task 2.2: Create TodoStore Tests

**File**: `tests/unit/test_todo_store.py`

```python
"""Unit tests for TodoStore tool."""

import pytest
from src.challenge.tools.todo_store import TodoStoreTool


@pytest.fixture
def todo_store():
    """Create fresh TodoStore instance."""
    return TodoStoreTool()


# Tier 2: Full CRUD operations
@pytest.mark.asyncio
async def test_add_todo(todo_store):
    """Test adding a todo."""
    result = await todo_store.execute(action="add", text="Buy milk")

    assert result.success
    assert result.output["text"] == "Buy milk"
    assert result.output["completed"] is False
    assert "id" in result.output
    assert "created_at" in result.output


@pytest.mark.asyncio
async def test_list_empty_todos(todo_store):
    """Test listing when no todos exist."""
    result = await todo_store.execute(action="list")

    assert result.success
    assert result.output == []
    assert result.metadata["count"] == 0


@pytest.mark.asyncio
async def test_list_todos(todo_store):
    """Test listing todos."""
    # Add two todos
    await todo_store.execute(action="add", text="First")
    await todo_store.execute(action="add", text="Second")

    result = await todo_store.execute(action="list")

    assert result.success
    assert len(result.output) == 2
    assert result.metadata["count"] == 2


@pytest.mark.asyncio
async def test_get_todo(todo_store):
    """Test getting specific todo."""
    # Add a todo
    add_result = await todo_store.execute(action="add", text="Test")
    todo_id = add_result.output["id"]

    # Get the todo
    result = await todo_store.execute(action="get", todo_id=todo_id)

    assert result.success
    assert result.output["id"] == todo_id
    assert result.output["text"] == "Test"


@pytest.mark.asyncio
async def test_complete_todo(todo_store):
    """Test completing a todo."""
    # Add a todo
    add_result = await todo_store.execute(action="add", text="Task")
    todo_id = add_result.output["id"]

    # Complete it
    result = await todo_store.execute(action="complete", todo_id=todo_id)

    assert result.success
    assert result.output["completed"] is True
    assert "completed_at" in result.output


@pytest.mark.asyncio
async def test_delete_todo(todo_store):
    """Test deleting a todo."""
    # Add a todo
    add_result = await todo_store.execute(action="add", text="Delete me")
    todo_id = add_result.output["id"]

    # Delete it
    result = await todo_store.execute(action="delete", todo_id=todo_id)

    assert result.success
    assert result.output["id"] == todo_id

    # Verify it's gone
    list_result = await todo_store.execute(action="list")
    assert len(list_result.output) == 0


# Error handling
@pytest.mark.asyncio
async def test_add_without_text(todo_store):
    """Test error when adding todo without text."""
    result = await todo_store.execute(action="add")

    assert not result.success
    assert "required" in result.error.lower()


@pytest.mark.asyncio
async def test_get_nonexistent_todo(todo_store):
    """Test error when getting nonexistent todo."""
    result = await todo_store.execute(action="get", todo_id="nonexistent")

    assert not result.success
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_invalid_action(todo_store):
    """Test error with invalid action."""
    result = await todo_store.execute(action="invalid_action")

    assert not result.success
    assert "unknown" in result.error.lower()
```

**Run Tests**:
```bash
pytest tests/unit/test_todo_store.py -v
# Expected: All tests pass
```

---

#### Task 2.3: Verify Coverage

```bash
# Run all unit tests with coverage
pytest tests/unit/ --cov=src/challenge/tools --cov-report=term --cov-report=html

# Expected output should show >80% coverage
# Example:
# src/challenge/tools/base.py          100%
# src/challenge/tools/calculator.py    95%
# src/challenge/tools/todo_store.py    92%
# TOTAL                                 94%

# Open HTML coverage report
open htmlcov/index.html  # Mac
# xdg-open htmlcov/index.html  # Linux
```

### Quality Gate: Phase 2 Exit Criteria

```bash
# 1. All tests pass
pytest tests/unit/ -v
# Expected: 100% passing (e.g., "20 passed")

# 2. Coverage >80% (Tier 2 requirement)
pytest tests/unit/ --cov=src/challenge/tools --cov-report=term | grep "TOTAL"
# Expected: TOTAL >80%

# 3. Security tests pass
pytest tests/unit/test_calculator.py::test_blocks_eval_injection -v
# Expected: PASSED
```

**All checks must pass before proceeding to Phase 3.**

### Time Checkpoint

```bash
# Expected total elapsed: ~2.5-3 hours (Phase 1 + Phase 2)
# If total >3.5 hours: Consider emergency fallback
```

### Emergency Fallback (if coverage <80%)

**If coverage 75-80%**: Acceptable for Tier 1/2 boundary, proceed
**If coverage <75%**: Add basic edge case tests:
- Calculator: test complex expressions
- TodoStore: test complete/delete on completed todos
**TIME**: Add 15-20 minutes for edge cases

---

## 📋 Phase 3: Planner Implementation

**Duration**: 30-40 minutes
**Priority**: MEDIUM - Simplified pattern matching

### Entry Criteria
- [x] Phase 2 complete and verified
- [ ] Tool tests passing with >80% coverage
- [ ] Security tests verified

### Implementation Tasks

#### Task 3.1: Create Plan Models

**File**: `src/challenge/models/__init__.py`

```python
"""Domain models exports."""

# Will be populated as models are created
```

**File**: `src/challenge/models/plan.py`

```python
"""Plan and execution models."""

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """Single step in execution plan."""

    step_number: int = Field(..., description="Sequential step number", ge=1)
    tool_name: str = Field(..., description="Tool to execute")
    tool_input: dict[str, str | int | float | bool | None] = Field(..., description="Tool input parameters")
    reasoning: str = Field(..., description="Why this step is needed")


class Plan(BaseModel):
    """Execution plan with ordered steps."""

    steps: list[PlanStep] = Field(default_factory=list, description="Ordered execution steps")
    final_goal: str = Field(..., description="What the plan aims to achieve")
```

---

#### Task 3.2: Create Pattern-Based Planner

**File**: `src/challenge/planner/__init__.py`

```python
"""Planner exports."""

from src.challenge.planner.planner import PatternBasedPlanner

__all__ = ["PatternBasedPlanner"]
```

**File**: `src/challenge/planner/planner.py`

```python
"""Pattern-based planner for converting prompts to plans.

Tier 2 Implementation:
- Pattern matching for calculator and todo operations
- Multi-step support with 'and' / 'then' operators
- Simple validation
"""

import re
from src.challenge.models.plan import Plan, PlanStep


class PatternBasedPlanner:
    """Convert natural language prompts to structured plans using pattern matching."""

    def create_plan(self, prompt: str) -> Plan:
        """Create execution plan from prompt.

        Args:
            prompt: Natural language task description

        Returns:
            Plan with ordered steps

        Raises:
            ValueError: If prompt cannot be parsed
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        prompt_lower = prompt.lower().strip()
        steps: list[PlanStep] = []

        # Split on multi-step operators
        # Handle both "X and Y" and "X then Y"
        sub_prompts = re.split(r'\s+(?:and|then)\s+', prompt_lower)

        for sub_prompt in sub_prompts:
            step = self._parse_single_step(sub_prompt.strip(), len(steps) + 1)
            if step:
                steps.append(step)

        if not steps:
            raise ValueError(f"Could not parse prompt: {prompt}")

        return Plan(
            steps=steps,
            final_goal=prompt
        )

    def _parse_single_step(self, prompt: str, step_number: int) -> PlanStep | None:
        """Parse single operation from prompt.

        Args:
            prompt: Single operation prompt
            step_number: Sequential step number

        Returns:
            PlanStep if pattern matched, None otherwise
        """
        # Pattern 1: Calculator operations
        # Matches: "calculate 2+2", "compute (10-3)*4", "evaluate -5/2"
        calc_pattern = r'(?:calculate|compute|evaluate|math)\s+(.+)'
        if match := re.search(calc_pattern, prompt):
            expression = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="calculator",
                tool_input={"expression": expression},
                reasoning=f"Calculate: {expression}"
            )

        # Pattern 2: Add todo
        # Matches: "add todo buy milk", "create task finish report"
        add_pattern = r'(?:add|create)\s+(?:todo|task)\s+(.+)'
        if match := re.search(add_pattern, prompt):
            text = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="todo_store",
                tool_input={"action": "add", "text": text},
                reasoning=f"Add todo: {text}"
            )

        # Pattern 3: List todos
        # Matches: "list todos", "show tasks", "show all todos"
        list_pattern = r'(?:list|show|get|display)\s+(?:all\s+)?(?:todos|tasks)'
        if re.search(list_pattern, prompt):
            return PlanStep(
                step_number=step_number,
                tool_name="todo_store",
                tool_input={"action": "list"},
                reasoning="List all todos"
            )

        # No pattern matched
        return None
```

**Verification**:
```bash
python << 'EOF'
from src.challenge.planner import PatternBasedPlanner

planner = PatternBasedPlanner()

# Test single calculator operation
plan1 = planner.create_plan("calculate 2 + 2")
assert len(plan1.steps) == 1
assert plan1.steps[0].tool_name == "calculator"
print("✅ Calculator pattern works")

# Test single todo operation
plan2 = planner.create_plan("add todo buy milk")
assert len(plan2.steps) == 1
assert plan2.steps[0].tool_name == "todo_store"
print("✅ Todo add pattern works")

# Test multi-step
plan3 = planner.create_plan("add todo buy milk then list todos")
assert len(plan3.steps) == 2
print("✅ Multi-step pattern works")

print("✅ All planner patterns working")
EOF
```

### Quality Gate: Phase 3 Exit Criteria

```bash
# 1. Planner imports
python -c "from src.challenge.planner import PatternBasedPlanner; print('✅ Planner OK')"

# 2. Pattern matching works
python << 'EOF'
from src.challenge.planner import PatternBasedPlanner
planner = PatternBasedPlanner()

# Test each pattern type
tests = [
    ("calculate 2+2", 1, "calculator"),
    ("add todo test", 1, "todo_store"),
    ("list todos", 1, "todo_store"),
    ("add todo X and list todos", 2, None),
]

for prompt, expected_steps, expected_tool in tests:
    plan = planner.create_plan(prompt)
    assert len(plan.steps) == expected_steps, f"Failed: {prompt}"
    if expected_tool:
        assert plan.steps[0].tool_name == expected_tool
    print(f"✅ {prompt}")

print("✅ All patterns verified")
EOF
```

### Time Checkpoint

```bash
# Expected total elapsed: ~3-3.5 hours
# On track for Tier 2 if within this range
```

### Emergency Fallback (if >40 min)

**ACTION**: Simplify to single-operation only
**CUT**: Multi-step support (no "and"/"then")
**KEEP**: Basic calculator and todo patterns
**TIME SAVED**: ~10-15 minutes

---

## ⚙️ Phase 4: Execution Orchestrator

**Duration**: 60-75 minutes
**Priority**: HIGH - Core execution logic

### Entry Criteria
- [x] Phase 3 complete and verified
- [ ] Planner creates valid plans
- [ ] Tools and planner integrated

### Implementation Tasks

#### Task 4.1: Create Run Models

**File**: `src/challenge/models/run.py`

```python
"""Run execution models."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.challenge.models.plan import Plan


class RunStatus(str, Enum):
    """Run execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionStep(BaseModel):
    """Record of step execution."""

    step_number: int = Field(..., description="Step sequence number")
    tool_name: str = Field(..., description="Tool that was executed")
    tool_input: dict[str, Any] = Field(..., description="Input provided to tool")
    success: bool = Field(..., description="Whether step succeeded")
    output: Any | None = Field(None, description="Step output if successful")
    error: str | None = Field(None, description="Error message if failed")
    attempts: int = Field(1, description="Number of execution attempts")
    duration_ms: int | None = Field(None, description="Execution duration in milliseconds")


class Run(BaseModel):
    """Complete run execution record."""

    run_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique run identifier")
    prompt: str = Field(..., description="Original user prompt")
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current execution status")
    plan: Plan | None = Field(None, description="Generated execution plan")
    execution_log: list[ExecutionStep] = Field(default_factory=list, description="Step execution history")
    result: Any | None = Field(None, description="Final result")
    error: str | None = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
```

Update `src/challenge/models/__init__.py`:
```python
"""Domain models exports."""

from src.challenge.models.plan import Plan, PlanStep
from src.challenge.models.run import Run, RunStatus, ExecutionStep

__all__ = [
    "Plan",
    "PlanStep",
    "Run",
    "RunStatus",
    "ExecutionStep",
]
```

---

#### Task 4.2: Create Orchestrator

**File**: `src/challenge/orchestrator/__init__.py`

```python
"""Orchestrator exports."""

from src.challenge.orchestrator.orchestrator import Orchestrator

__all__ = ["Orchestrator"]
```

**File**: `src/challenge/orchestrator/orchestrator.py`

```python
"""Execution orchestrator with retry logic and state management."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from src.challenge.models.plan import Plan
from src.challenge.models.run import Run, RunStatus, ExecutionStep
from src.challenge.planner import PatternBasedPlanner
from src.challenge.tools import BaseTool, get_tool_registry


class Orchestrator:
    """Orchestrate plan execution with retry logic."""

    def __init__(
        self,
        planner: PatternBasedPlanner | None = None,
        tools: dict[str, BaseTool] | None = None,
        max_retries: int = 3,
    ):
        """Initialize orchestrator.

        Args:
            planner: Planner instance (creates default if None)
            tools: Tool registry (creates default if None)
            max_retries: Maximum retry attempts per step
        """
        self.planner = planner or PatternBasedPlanner()
        self.tools = tools or get_tool_registry()
        self.max_retries = max_retries

        # In-memory run storage
        self.runs: dict[str, Run] = {}

    async def create_run(self, prompt: str) -> Run:
        """Create and start execution run.

        Args:
            prompt: Natural language task description

        Returns:
            Run with pending/running status
        """
        # Create run
        run = Run(prompt=prompt)

        try:
            # Generate plan
            plan = self.planner.create_plan(prompt)
            run.plan = plan

            # Store run
            self.runs[run.run_id] = run

            # Start async execution (don't await - run in background)
            asyncio.create_task(self._execute_run(run.run_id))

            return run

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = f"Planning failed: {str(e)}"
            self.runs[run.run_id] = run
            return run

    def get_run(self, run_id: str) -> Run | None:
        """Get run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run if found, None otherwise
        """
        return self.runs.get(run_id)

    async def _execute_run(self, run_id: str) -> None:
        """Execute run asynchronously.

        Args:
            run_id: Run to execute
        """
        run = self.runs[run_id]

        try:
            # Mark as running
            run.status = RunStatus.RUNNING
            run.started_at = datetime.now(timezone.utc)

            # Execute each step
            for step in run.plan.steps:
                step_result = await self._execute_step_with_retry(step)
                run.execution_log.append(step_result)

                # Stop on failure (optional: could continue for other steps)
                if not step_result.success:
                    run.status = RunStatus.FAILED
                    run.error = f"Step {step.step_number} failed: {step_result.error}"
                    run.completed_at = datetime.now(timezone.utc)
                    return

            # All steps succeeded
            run.status = RunStatus.COMPLETED

            # Set final result (last step's output)
            if run.execution_log:
                run.result = run.execution_log[-1].output

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = f"Execution error: {str(e)}"

        finally:
            run.completed_at = datetime.now(timezone.utc)

    async def _execute_step_with_retry(self, step: Any) -> ExecutionStep:
        """Execute step with exponential backoff retry.

        Args:
            step: PlanStep to execute

        Returns:
            ExecutionStep with result
        """
        start_time = time.time()
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Get tool
                tool = self.tools.get(step.tool_name)
                if not tool:
                    return ExecutionStep(
                        step_number=step.step_number,
                        tool_name=step.tool_name,
                        tool_input=step.tool_input,
                        success=False,
                        error=f"Tool not found: {step.tool_name}",
                        attempts=attempt,
                    )

                # Execute tool
                result = await tool.execute(**step.tool_input)

                # Calculate duration
                duration_ms = int((time.time() - start_time) * 1000)

                # Return execution record
                return ExecutionStep(
                    step_number=step.step_number,
                    tool_name=step.tool_name,
                    tool_input=step.tool_input,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    attempts=attempt,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                last_error = str(e)

                # Exponential backoff: 1s, 2s, 4s
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))

        # All retries failed
        duration_ms = int((time.time() - start_time) * 1000)
        return ExecutionStep(
            step_number=step.step_number,
            tool_name=step.tool_name,
            tool_input=step.tool_input,
            success=False,
            error=f"Failed after {self.max_retries} attempts: {last_error}",
            attempts=self.max_retries,
            duration_ms=duration_ms,
        )
```

**Verification**:
```bash
python << 'EOF'
import asyncio
from src.challenge.orchestrator import Orchestrator

async def test_orchestrator():
    orchestrator = Orchestrator()

    # Create run
    run = await orchestrator.create_run("calculate 2 + 2")
    print(f"✅ Run created: {run.run_id}")
    print(f"   Status: {run.status}")
    print(f"   Plan steps: {len(run.plan.steps)}")

    # Wait for execution (max 5 seconds)
    for _ in range(50):
        await asyncio.sleep(0.1)
        updated_run = orchestrator.get_run(run.run_id)
        if updated_run.status in [RunStatus.COMPLETED, RunStatus.FAILED]:
            break

    # Check final state
    final_run = orchestrator.get_run(run.run_id)
    print(f"   Final status: {final_run.status}")
    print(f"   Result: {final_run.result}")

    assert final_run.status == RunStatus.COMPLETED
    assert final_run.result == 4.0
    print("✅ Orchestrator works correctly")

from src.challenge.models.run import RunStatus
asyncio.run(test_orchestrator())
EOF
```

### Quality Gate: Phase 4 Exit Criteria

```bash
# 1. Orchestrator imports
python -c "from src.challenge.orchestrator import Orchestrator; print('✅ Orchestrator OK')"

# 2. Run execution works
python << 'EOF'
import asyncio
from src.challenge.orchestrator import Orchestrator
from src.challenge.models.run import RunStatus

async def verify():
    orch = Orchestrator()

    # Test calculator
    run1 = await orch.create_run("calculate 2+2")
    await asyncio.sleep(0.5)
    final1 = orch.get_run(run1.run_id)
    assert final1.status == RunStatus.COMPLETED
    assert final1.result == 4.0
    print("✅ Calculator execution works")

    # Test todo
    run2 = await orch.create_run("add todo test")
    await asyncio.sleep(0.5)
    final2 = orch.get_run(run2.run_id)
    assert final2.status == RunStatus.COMPLETED
    print("✅ Todo execution works")

    # Test multi-step
    run3 = await orch.create_run("add todo milk and list todos")
    await asyncio.sleep(0.5)
    final3 = orch.get_run(run3.run_id)
    assert final3.status == RunStatus.COMPLETED
    assert len(final3.execution_log) == 2
    print("✅ Multi-step execution works")

asyncio.run(verify())
EOF
```

### Time Checkpoint

```bash
# Expected total elapsed: ~4.5-5.5 hours
# On track for Tier 2 completion
```

### Emergency Fallback (if >90 min)

**ACTION**: Simplify retry logic
**CUT**: Sophisticated backoff, just do 3 basic retries
**KEEP**: Sequential execution, state tracking
**TIME SAVED**: ~15-20 minutes

---

## 🌐 Phase 5: REST API Integration

**Duration**: 30-45 minutes
**Priority**: MEDIUM - Build on existing FastAPI

### Entry Criteria
- [x] Phase 4 complete and verified
- [ ] Orchestrator executes plans correctly
- [ ] Existing FastAPI server running

### Implementation Tasks

#### Task 5.1: Update Dependencies

**File**: `src/challenge/api/dependencies.py`

Add after existing `SettingsDep`:

```python
from typing import Annotated
from fastapi import Depends

from challenge.core.config import Settings, get_settings
from challenge.orchestrator import Orchestrator

# Existing SettingsDep remains unchanged

def get_orchestrator(settings: SettingsDep) -> Orchestrator:
    """Get orchestrator instance.

    Args:
        settings: Application settings

    Returns:
        Configured orchestrator
    """
    return Orchestrator()

OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
```

---

#### Task 5.2: Create Runs Routes

**File**: `src/challenge/api/routes/runs.py`

```python
"""Run execution endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from challenge.api.dependencies import OrchestratorDep
from challenge.models.run import Run

logger = logging.getLogger(__name__)

router = APIRouter()


class RunCreate(BaseModel):
    """Request to create new run."""

    prompt: str = Field(..., description="Natural language task to execute", min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "calculate 2 + 2 and add todo buy milk"
            }
        }


@router.post("/runs", status_code=status.HTTP_201_CREATED, response_model=Run)
async def create_run(
    request: RunCreate,
    orchestrator: OrchestratorDep,
) -> Run:
    """Create and start execution run.

    Args:
        request: Run creation request with prompt
        orchestrator: Orchestrator dependency

    Returns:
        Run with pending/running status

    Raises:
        HTTPException: 400 if prompt is invalid
    """
    try:
        run = await orchestrator.create_run(request.prompt)
        logger.info("Created run %s for prompt: %s", run.run_id, request.prompt)
        return run

    except ValueError as e:
        logger.warning("Invalid prompt: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid prompt: {str(e)}"
        )
    except Exception as e:
        logger.error("Run creation failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Run creation failed"
        )


@router.get("/runs/{run_id}", response_model=Run)
async def get_run(
    run_id: str,
    orchestrator: OrchestratorDep,
) -> Run:
    """Get run status and results.

    Args:
        run_id: Run identifier
        orchestrator: Orchestrator dependency

    Returns:
        Complete run state

    Raises:
        HTTPException: 404 if run not found
    """
    run = orchestrator.get_run(run_id)

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}"
        )

    return run
```

---

#### Task 5.3: Register Routes

**File**: `src/challenge/api/main.py`

Update the `_register_routes` function:

```python
def _register_routes(app: FastAPI) -> None:
    """Register API routes and routers."""
    # Existing health router
    app.include_router(health.router, prefix="/api/v1", tags=["health"])

    # Add runs router
    from challenge.api.routes import runs
    app.include_router(runs.router, prefix="/api/v1", tags=["runs"])
```

---

#### Task 5.4: Verify API Endpoints

**Start the server**:
```bash
uv run python -m challenge
# Server should start on http://localhost:8000
```

**Test endpoints**:
```bash
# 1. Health check (existing)
curl http://localhost:8000/api/v1/health
# Expected: {"status":"healthy", ...}

# 2. Create run
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt":"calculate 2 + 2"}'
# Expected: {"run_id":"...", "status":"pending", ...}
# Save the run_id from response

# 3. Get run (replace RUN_ID with actual ID from step 2)
curl http://localhost:8000/api/v1/runs/RUN_ID
# Expected: {"run_id":"...", "status":"completed", "result":4.0, ...}

# 4. Test invalid prompt
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt":""}'
# Expected: 400 error

# 5. Test nonexistent run
curl http://localhost:8000/api/v1/runs/nonexistent
# Expected: 404 error
```

### Quality Gate: Phase 5 Exit Criteria

```bash
# Server must be running for these checks

# 1. Health endpoint works
curl -s http://localhost:8000/api/v1/health | grep -q "healthy"
echo "✅ Health endpoint OK"

# 2. Create run works
RUN_ID=$(curl -s -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt":"calculate 2+2"}' | python3 -c "import sys, json; print(json.load(sys.stdin)['run_id'])")
echo "✅ Create run OK: $RUN_ID"

# 3. Get run works
sleep 1  # Wait for execution
curl -s http://localhost:8000/api/v1/runs/$RUN_ID | grep -q "completed"
echo "✅ Get run OK"

# 4. Error handling works
curl -s -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt":""}' | grep -q "400"
echo "✅ Error handling OK"
```

### Time Checkpoint

```bash
# Expected total elapsed: ~5-6 hours
# On track for Tier 2 if within this range
```

### Emergency Fallback (if >45 min)

**ACTION**: Minimal endpoint implementation
**CUT**: Advanced error handling, detailed responses
**KEEP**: Core POST /runs and GET /runs/{id}
**TIME SAVED**: ~10-15 minutes

---

## ✅ Phase 6: Integration Testing

**Duration**: 45-60 minutes
**Priority**: HIGH - Coverage verification

### Entry Criteria
- [x] Phase 5 complete and verified
- [ ] All API endpoints working
- [ ] Server starts successfully

### Implementation Tasks

#### Task 6.1: Create E2E Tests

**File**: `tests/integration/test_e2e_flow.py`

```python
"""End-to-end integration tests."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from challenge.api.main import create_app
    from challenge.core.config import Settings

    settings = Settings(environment="test")
    app = create_app(settings=settings)

    with TestClient(app) as test_client:
        yield test_client


def test_calculator_flow(client):
    """Test complete calculator execution flow."""
    # Create run
    response = client.post(
        "/api/v1/runs",
        json={"prompt": "calculate 2 + 2"}
    )

    assert response.status_code == 201
    data = response.json()
    assert "run_id" in data
    assert data["status"] in ["pending", "running", "completed"]

    run_id = data["run_id"]

    # Get run (may need to retry for async execution)
    import time
    for _ in range(10):
        time.sleep(0.1)
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()

        if data["status"] in ["completed", "failed"]:
            break

    # Verify completion
    assert data["status"] == "completed"
    assert data["result"] == 4.0
    assert len(data["execution_log"]) == 1
    assert data["execution_log"][0]["success"] is True


def test_todo_add_list_flow(client):
    """Test complete todo add+list flow."""
    # Create run
    response = client.post(
        "/api/v1/runs",
        json={"prompt": "add todo buy milk and list todos"}
    )

    assert response.status_code == 201
    run_id = response.json()["run_id"]

    # Wait for completion
    import time
    for _ in range(10):
        time.sleep(0.1)
        response = client.get(f"/api/v1/runs/{run_id}")
        data = response.json()

        if data["status"] in ["completed", "failed"]:
            break

    # Verify completion
    assert data["status"] == "completed"
    assert len(data["execution_log"]) == 2

    # Verify add step
    add_step = data["execution_log"][0]
    assert add_step["success"] is True
    assert add_step["tool_name"] == "todo_store"

    # Verify list step
    list_step = data["execution_log"][1]
    assert list_step["success"] is True
    assert len(list_step["output"]) >= 1  # At least one todo


def test_invalid_prompt(client):
    """Test error handling for invalid prompt."""
    response = client.post(
        "/api/v1/runs",
        json={"prompt": ""}
    )

    assert response.status_code == 422  # Pydantic validation error


def test_get_nonexistent_run(client):
    """Test 404 for nonexistent run."""
    response = client.get("/api/v1/runs/nonexistent-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_health_endpoint(client):
    """Test health endpoint still works."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
```

**Run Integration Tests**:
```bash
pytest tests/integration/test_e2e_flow.py -v
# Expected: All tests pass
```

---

#### Task 6.2: Verify Full Coverage

```bash
# Run ALL tests (unit + integration) with coverage
pytest tests/ --cov=src/challenge --cov-report=term --cov-report=html -v

# Expected coverage targets:
# src/challenge/tools/           >80%
# src/challenge/models/          >80%
# src/challenge/planner/         >75%
# src/challenge/orchestrator/    >75%
# src/challenge/api/             >70%
# TOTAL                          >80% (Tier 2 requirement)

# Open HTML report
open htmlcov/index.html
```

### Quality Gate: Phase 6 Exit Criteria

```bash
# 1. All tests pass
pytest tests/ -v
# Expected: 100% passing

# 2. Integration tests pass
pytest tests/integration/ -v
# Expected: All E2E tests pass

# 3. Coverage >80% (CRITICAL for Tier 2)
COVERAGE=$(pytest tests/ --cov=src/challenge --cov-report=term | grep "TOTAL" | awk '{print $4}' | sed 's/%//')
echo "Coverage: $COVERAGE%"
if [ "$COVERAGE" -ge 80 ]; then
  echo "✅ Coverage >80% (Tier 2 achieved)"
else
  echo "⚠️ Coverage $COVERAGE% (Tier 1/2 boundary)"
fi
```

### Time Checkpoint

```bash
# Expected total elapsed: ~6-7 hours
# Decision point: Tier 2 achieved or accept current state
```

### Emergency Fallback (if >60 min or coverage <80%)

**If coverage 75-80%**: Acceptable for Tier 1/2, proceed to docs
**If coverage <75%**: Add 2-3 quick edge case tests, then proceed
**TIME LIMIT**: Do not spend more than 20 extra minutes on coverage

---

## 📝 Phase 7: Documentation

**Duration**: 30-45 minutes
**Priority**: MEDIUM - Quality submission requirement

### Entry Criteria
- [x] Phase 6 complete
- [ ] All tests passing
- [ ] Coverage verified (>75%)

### Implementation Tasks

#### Task 7.1: Update README

**File**: `README.md`

Update with the following sections (merge with existing content):

```markdown
# Crane AI Agent Runtime

A minimal AI agent runtime POC demonstrating intelligent task planning, tool execution, and robust orchestration.

## 🎯 Project Overview

**Time Spent**: [Record actual time]
**Target**: Tier 2 (75-85% score)
**Test Coverage**: [Record actual coverage]%

This project implements a production-quality AI agent runtime with:
- **Secure tool integration**: AST-based calculator (no eval/exec)
- **Pattern-based planning**: Natural language to structured execution plans
- **Robust orchestration**: Sequential execution with exponential backoff retry
- **Clean REST API**: FastAPI endpoints for interaction and monitoring

## 🏗️ System Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP (POST /runs, GET /runs/{id})
       ▼
┌─────────────────────────┐
│     FastAPI Server      │
│   (API Layer)           │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│    Orchestrator         │
│  (Execution Engine)     │
│  - Create runs          │
│  - Execute plans        │
│  - Retry logic          │
│  - State management     │
└──────┬──────────────────┘
       │
       ├─────────────┬──────────────┐
       ▼             ▼              ▼
┌─────────────┐ ┌─────────┐ ┌──────────────┐
│   Planner   │ │  Tools  │ │   Models     │
│ (Pattern    │ │         │ │ (Domain)     │
│  Matching)  │ │         │ │              │
└─────────────┘ └─────────┘ └──────────────┘
```

**Component Responsibilities**:
- **API Layer**: HTTP interface, request/response handling, error translation
- **Orchestrator**: Plan execution, retry logic, state tracking, async coordination
- **Planner**: Natural language → structured plans via pattern matching
- **Tools**: Calculator (AST-based), TodoStore (in-memory CRUD)
- **Models**: Domain entities (Run, Plan, Step) with Pydantic validation

## 🚀 Setup and Installation

### Prerequisites
- Python 3.12+
- uv package manager (or pip)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd crane-challenge

# Create virtual environment and install dependencies
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
```

## 🏃 Running the Application

### Start Server

```bash
# Using uv
uv run python -m challenge

# Server starts at http://localhost:8000
# API docs available at http://localhost:8000/api/docs
```

### Example Usage

```bash
# 1. Health check
curl http://localhost:8000/api/v1/health

# 2. Create calculator run
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt":"calculate (10 + 5) * 2"}'

# Response: {"run_id":"abc-123", "status":"pending", ...}

# 3. Get run status (replace abc-123 with actual run_id)
curl http://localhost:8000/api/v1/runs/abc-123

# Response: Complete run state with execution_log and result

# 4. Multi-step todo flow
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt":"add todo buy milk and list todos"}'

# 5. Get run results
curl http://localhost:8000/api/v1/runs/<run-id>
```

## 🧪 Testing

### Run All Tests

```bash
# Run complete test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/challenge --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Test Categories

- **Unit Tests** (`tests/unit/`):
  - Calculator: arithmetic operations, security tests
  - TodoStore: CRUD operations
  - Each tool tested in isolation

- **Integration Tests** (`tests/integration/`):
  - End-to-end calculator flow
  - Todo add+list flow
  - Error handling scenarios

### Security Testing

```bash
# CRITICAL: Verify no eval/exec in calculator
grep -rn "eval\|exec" src/challenge/tools/calculator.py
# Expected: No matches (security requirement)

# Run security tests
pytest tests/unit/test_calculator.py::test_blocks_eval_injection -v
```

## 🎯 Design Decisions and Trade-offs

### 1. Pattern-Based Planner vs LLM

**Decision**: Pattern-based planner using regex
**Rationale**:
- ✅ **Time efficient**: 30-40 min implementation vs 2+ hours for LLM integration
- ✅ **Deterministic**: Predictable behavior, easier to test
- ✅ **No external dependencies**: Runs locally, no API keys needed
- ✅ **Sufficient for POC**: Handles calculator and todo operations effectively

**Trade-off**: Limited natural language understanding
**Alternative**: LLM-based planner would handle more complex prompts but requires:
- OpenAI/Anthropic API integration
- Prompt engineering and structured output parsing
- Error handling for API failures
- Significantly more implementation time

### 2. AST-Based Calculator vs eval()

**Decision**: AST-based evaluation using `ast.NodeVisitor`
**Rationale**:
- ✅ **Security**: Prevents arbitrary code execution (CRITICAL requirement)
- ✅ **Safety**: No risk of `__import__`, `exec`, or system access
- ✅ **Controlled**: Whitelist approach for allowed operations

**Trade-off**: More complex implementation, limited operator support
**Alternative**: `eval()` would be simpler but creates severe security vulnerability

### 3. In-Memory State vs Persistent Storage

**Decision**: In-memory dictionaries for run storage
**Rationale**:
- ✅ **POC appropriate**: Demonstrates concepts without infrastructure complexity
- ✅ **Fast implementation**: 15-20 min vs 2+ hours for database
- ✅ **No dependencies**: No PostgreSQL, Redis, or ORM needed
- ✅ **Testable**: Easy to reset state in tests

**Trade-off**: State lost on restart, no multi-instance support
**Alternative**: PostgreSQL with SQLAlchemy would enable:
- Persistence across restarts
- Multi-instance deployments
- Query capabilities
- But requires database setup, migrations, connection management

### 4. Exponential Backoff Retry vs Fixed Retry

**Decision**: Exponential backoff (1s, 2s, 4s) with 3 attempts
**Rationale**:
- ✅ **Industry standard**: Common pattern for transient failures
- ✅ **Prevents overload**: Gives systems time to recover
- ✅ **Configurable**: max_retries parameter

**Trade-off**: Slower failure detection on persistent errors
**Alternative**: Fixed retry or no retry would be faster but less robust

### 5. Async Execution vs Synchronous

**Decision**: Async orchestration with `asyncio.create_task`
**Rationale**:
- ✅ **Non-blocking**: API returns immediately, execution in background
- ✅ **Scalable**: Can handle multiple concurrent runs
- ✅ **FastAPI aligned**: Leverages async/await throughout

**Trade-off**: Added complexity in testing, state management
**Alternative**: Synchronous execution would be simpler but block requests

## ⚠️ Known Limitations

### Functional Limitations
1. **Pattern Matching Only**: Cannot handle complex natural language variations
   - Works: "calculate 2+2", "add todo buy milk"
   - Fails: "What's two plus two?", "Remind me to buy milk"

2. **In-Memory State**: Runs lost on server restart, no persistence layer

3. **Single Instance**: No support for horizontal scaling or distributed execution

4. **Limited Tool Set**: Only calculator and todo store implemented

5. **No Authentication**: API is completely open, no user management

### Technical Limitations
1. **Calculator**:
   - Tier 2: Basic operators (+, -, *, /), decimals, negatives, parentheses
   - Missing: Scientific functions (sqrt, pow, trig), constants (pi, e)

2. **TodoStore**:
   - Tier 2: CRUD operations (add, list, get, complete, delete)
   - Missing: Update, filter, search, priority, due dates

3. **Planner**:
   - Limited to "calculate X", "add todo X", "list todos" patterns
   - No support for conditional logic or complex dependencies

4. **Orchestrator**:
   - Sequential execution only (no parallel steps)
   - Basic retry logic (no circuit breaker or rate limiting)
   - No idempotency support

## 🚀 Potential Improvements

### If More Time Available

**High Priority** (Next 2-4 hours):
1. **LLM-Based Planner**:
   - OpenAI function calling for robust plan generation
   - Handles broader range of natural language inputs
   - Structured output parsing with Pydantic

2. **Persistent Storage**:
   - PostgreSQL for run history
   - SQLAlchemy ORM with migrations
   - Query capabilities for analytics

3. **Enhanced Testing**:
   - >90% coverage target
   - Property-based testing with Hypothesis
   - Performance testing with locust

**Medium Priority** (4-8 hours):
4. **Advanced Features**:
   - Parallel step execution where possible
   - Conditional logic in plans
   - Tool chaining and data passing between steps

5. **Operational Excellence**:
   - Structured logging with correlation IDs
   - Metrics and monitoring (Prometheus)
   - Health checks with dependency validation

6. **Tool Expansion**:
   - Web search tool
   - File system operations
   - API integration tool

**Lower Priority** (Future):
7. **Enterprise Features**:
   - Authentication and authorization
   - Rate limiting and quotas
   - Multi-tenancy support
   - Webhook notifications

8. **Advanced Orchestration**:
   - Idempotency with execution fingerprinting
   - Circuit breaker patterns
   - Distributed execution with Celery

## 📊 Test Coverage Report

**Overall Coverage**: [Record actual]%

```
Module                          Coverage
────────────────────────────────────────
tools/calculator.py             [X]%
tools/todo_store.py             [X]%
planner/planner.py              [X]%
orchestrator/orchestrator.py    [X]%
api/routes/runs.py              [X]%
────────────────────────────────────────
TOTAL                           [X]%
```

**Tier Achievement**: Tier 2 ✅ (Target: >80% coverage)

## 🔒 Security

### Security Measures Implemented
1. **AST-Based Calculator**: No eval/exec, prevents code injection
2. **Input Validation**: Pydantic models validate all inputs
3. **Error Handling**: No sensitive data in error messages
4. **CORS Configuration**: Specific origins only (no wildcard with credentials)

### Security Testing
```bash
# Verify calculator security
pytest tests/unit/test_calculator.py::test_blocks_eval_injection -v

# Check for eval/exec usage
grep -rn "eval\|exec" src/challenge/tools/
```

## 📚 API Documentation

Interactive API documentation available when server is running:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## 🙏 Acknowledgments

Built with:
- **FastAPI**: Modern async web framework
- **Pydantic**: Data validation and settings management
- **pytest**: Testing framework
- **uv**: Fast Python package installer

## 📄 License

[Specify license or "Proprietary"]
```

---

### Quality Gate: Phase 7 Exit Criteria

```bash
# 1. README exists and is comprehensive
grep -q "System Architecture" README.md && \
grep -q "Design Decisions" README.md && \
grep -q "Known Limitations" README.md && \
echo "✅ README comprehensive"

# 2. Trade-offs documented
grep -q "Trade-off" README.md && echo "✅ Trade-offs documented"

# 3. Limitations documented
grep -q "Limitations" README.md && echo "✅ Limitations documented"
```

### Time Checkpoint

```bash
# Expected total elapsed: ~6.5-7.5 hours
# Final push to completion
```

### Emergency Fallback (if >45 min)

**ACTION**: Use README template, fill minimally
**CUT**: Detailed trade-offs, comprehensive improvements
**KEEP**: Setup instructions, basic usage, architecture overview
**TIME SAVED**: ~15-20 minutes

---

## 🎯 Phase 8: Final Verification & Submission

**Duration**: 20-30 minutes
**Priority**: CRITICAL - Final quality check

### Entry Criteria
- [x] Phase 7 complete
- [ ] README comprehensive
- [ ] All previous phases verified

### Final Verification Checklist

```bash
#!/bin/bash
# Run all final verification checks

echo "======================================"
echo "  FINAL VERIFICATION CHECKLIST"
echo "======================================"

# 1. Server starts
echo ""
echo "1. Server startup..."
timeout 5 uv run python -m challenge &
SERVER_PID=$!
sleep 3
if curl -s http://localhost:8000/api/v1/health > /dev/null; then
  echo "✅ Server starts successfully"
else
  echo "❌ Server failed to start"
fi
kill $SERVER_PID 2>/dev/null

# 2. Health check
echo ""
echo "2. Health endpoint..."
timeout 5 uv run python -m challenge &
SERVER_PID=$!
sleep 3
if curl -s http://localhost:8000/api/v1/health | grep -q "healthy"; then
  echo "✅ Health check responds"
else
  echo "❌ Health check failed"
fi
kill $SERVER_PID 2>/dev/null

# 3. All tests pass
echo ""
echo "3. Test suite..."
if pytest tests/ -q --tb=no; then
  echo "✅ All tests pass"
else
  echo "❌ Some tests failed"
fi

# 4. Coverage >80%
echo ""
echo "4. Test coverage..."
COVERAGE=$(pytest tests/ --cov=src/challenge --cov-report=term | grep "TOTAL" | awk '{print $4}' | sed 's/%//')
if [ "$COVERAGE" -ge 80 ]; then
  echo "✅ Coverage: $COVERAGE% (Tier 2 achieved)"
elif [ "$COVERAGE" -ge 75 ]; then
  echo "⚠️ Coverage: $COVERAGE% (Tier 1/2 boundary)"
else
  echo "❌ Coverage: $COVERAGE% (Below Tier 2)"
fi

# 5. Security verified
echo ""
echo "5. Security check..."
if grep -rq "eval\|exec" src/challenge/tools/calculator.py; then
  echo "❌ SECURITY RISK: eval/exec found in calculator"
else
  echo "✅ Security: No eval/exec in calculator"
fi

# 6. No linting errors
echo ""
echo "6. Code quality..."
if ruff check src/ tests/ --quiet; then
  echo "✅ No linting errors"
else
  echo "⚠️ Linting errors found"
fi

# 7. README complete
echo ""
echo "7. Documentation..."
if grep -q "Design Decisions" README.md && grep -q "Known Limitations" README.md; then
  echo "✅ README comprehensive"
else
  echo "⚠️ README incomplete"
fi

# 8. Git status
echo ""
echo "8. Version control..."
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ "$UNCOMMITTED" -gt 0 ]; then
  echo "⚠️ $UNCOMMITTED uncommitted files"
else
  echo "✅ All files committed"
fi

echo ""
echo "======================================"
echo "  VERIFICATION COMPLETE"
echo "======================================"
```

Save as `verify_submission.sh`, make executable, and run:
```bash
chmod +x verify_submission.sh
./verify_submission.sh
```

---

### Final Commit

```bash
# Review what will be committed
git status
git diff

# Commit everything
git add .
git commit -m "Complete AI Agent Runtime POC

Tier 2 Implementation:
- Tools: Calculator (AST-based), TodoStore (CRUD)
- Planner: Pattern-based NL→Plan conversion
- Orchestrator: Sequential execution with retry logic
- API: POST /runs, GET /runs/{id}
- Tests: >80% coverage with security verification
- Docs: Comprehensive README with trade-offs

Security: AST-based calculator (no eval/exec)
Coverage: [Record actual]%
Time: [Record actual] hours
"

# Verify commit
git log --oneline -5
```

---

### Create Submission Package

```bash
# Create submission directory
mkdir -p submission

# Copy source code
cp -r src/ submission/
cp -r tests/ submission/

# Copy documentation
cp README.md submission/
cp pyproject.toml submission/
cp uv.lock submission/

# Copy additional docs if they exist
cp -r docs/ submission/ 2>/dev/null || true

# Create archive
zip -r crane-ai-agent-submission.zip submission/

# Verify archive
unzip -l crane-ai-agent-submission.zip | head -20

echo "✅ Submission package created: crane-ai-agent-submission.zip"
```

---

### Quality Gate: Phase 8 Exit Criteria

**ALL checks must pass:**

- [x] Server starts successfully
- [x] Health endpoint responds
- [x] All tests pass
- [x] Coverage ≥75% (Tier 1/2) or ≥80% (Tier 2)
- [x] No eval/exec in calculator (CRITICAL)
- [x] README comprehensive with trade-offs and limitations
- [x] Git commits clean and descriptive
- [x] Submission package created

### Time Checkpoint

```bash
# Expected total elapsed: ~7-8 hours
# Ready to submit
```

---

## 🎯 Success Metrics Summary

### Tier 2 Achievement Checklist

✅ **Functional Requirements**:
- [x] Calculator tool with AST evaluation (NO eval/exec)
- [x] TodoStore tool with CRUD operations
- [x] Pattern-based planner converts prompts to plans
- [x] Orchestrator executes plans with retry logic
- [x] POST /runs creates and executes runs
- [x] GET /runs/{id} returns run state
- [x] GET /health responds correctly

✅ **Quality Requirements**:
- [x] Test coverage >80%
- [x] All tests passing
- [x] Security tests verify injection blocking
- [x] No linting errors

✅ **Documentation Requirements**:
- [x] README with architecture overview
- [x] Setup and installation instructions
- [x] Example usage with curl commands
- [x] Design decisions and trade-offs documented
- [x] Known limitations documented
- [x] Potential improvements listed

✅ **Submission Requirements**:
- [x] Clean git history
- [x] Submission package created
- [x] All verification checks pass

---

## 📞 Emergency Decision Tree

```
Time Remaining?
│
├─ ≥4h → Continue normally with Tier 2 target
│
├─ 2-4h → Tier 2 Focus Mode
│   ├─ Skip Tier 3 features
│   ├─ Accept 75-80% coverage if achieved
│   └─ Simplify documentation
│
├─ 1-2h → Tier 1 Triage
│   ├─ Simplify tools (basic ops only)
│   ├─ Single-step planner only
│   ├─ Basic tests only
│   └─ Minimal README
│
└─ <1h → SHIP NOW
    ├─ Does server start? Yes → Document and submit
    ├─ Does one example work? Yes → Document and submit
    └─ Any blockers? Fix minimum viable, then submit
```

---

## 🎓 Lessons Learned Template

After completion, document:

**What Went Well**:
- [Record what worked efficiently]

**What Was Challenging**:
- [Record unexpected difficulties]

**Time Management**:
- Actual vs estimated time per phase
- Where time was lost/gained

**Technical Insights**:
- AST evaluation approach
- Pattern matching strategies
- FastAPI integration techniques

**Process Improvements**:
- What would you do differently?
- What should be emphasized in documentation?

---

## ✅ WORKFLOW COMPLETE

This workflow provides a systematic, step-by-step implementation path for the Crane AI Agent Runtime assignment. Follow each phase sequentially, verify at quality gates, and use emergency procedures if behind schedule.

**Key Success Factors**:
1. ✅ Security: AST-based calculator (no eval/exec)
2. ✅ Quality: >80% test coverage
3. ✅ Functionality: All endpoints working
4. ✅ Documentation: Comprehensive README with trade-offs
5. ✅ Time Management: 6-8 hours total

**Target Achievement**: Tier 2 (75-85% score)

Good luck with your implementation! 🚀
