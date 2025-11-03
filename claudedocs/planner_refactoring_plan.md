# Planner Module Refactoring Plan

**Created**: 2025-11-02
**Status**: Ready for Implementation
**Scope**: src/challenge/planner/ module quality improvements
**Overall Grade**: B+ → A (target)

## Executive Summary

This refactoring plan addresses critical security vulnerabilities, performance issues, and architectural improvements in the planner module based on comprehensive analysis by three specialized agents (refactoring-expert, backend-architect, python-expert).

**Key Achievements from ExecutionContext Implementation** ✅:
- Fixed "Todo not found: <first_todo_id>" error
- Enabled inter-step state management
- Implemented variable resolution with {placeholder} syntax
- 281-line implementation with comprehensive docstrings

**Remaining Work**: 3 phases covering security, performance, and architecture

---

## Phase 1: Critical Security & Performance (1-2 Days)

### 1.1 Fix ReDoS Vulnerability 🚨 IMMEDIATE

**Priority**: CRITICAL
**Effort**: 2 hours
**Impact**: Security vulnerability → Production-ready
**File**: `src/challenge/planner/planner.py:129`

**Problem**: Catastrophic backtracking in regex pattern with nested quantifiers:
```python
# VULNERABLE - ReDoS attack vector
operation_pattern = r"(?:multiply|divide|subtract)\s+(.+)|(?:add)\s+(?!(?:a\s+)?(?:todo|task)\b)(.+)"

# Attack payload: "add " + "x" * 1000 + "!" → exponential time
```

**Before Code** (planner.py:129):
```python
def _parse_single_step(self, prompt: str, step_number: int) -> PlanStep | None:
    """Parse single step from prompt."""
    prompt_lower = prompt.lower().strip()

    # Calculator operations - VULNERABLE
    operation_pattern = r"(?:multiply|divide|subtract)\s+(.+)|(?:add)\s+(?!(?:a\s+)?(?:todo|task)\b)(.+)"
    if match := re.search(operation_pattern, prompt_lower):
        expression = match.group(1) or match.group(2)
        return PlanStep(
            step_number=step_number,
            tool_name="calculator",
            tool_input={"expression": expression.strip()},
            reasoning=f"Performing calculation: {expression.strip()}",
        )
```

**After Code**:
```python
# Module-level constant with length limit
_MAX_EXPRESSION_LENGTH = 200
_OPERATION_PATTERN = re.compile(
    r"(?:multiply|divide|subtract)\s+(.{1,200})"
    r"|(?:add)\s+(?!(?:a\s+)?(?:todo|task)\b)(.{1,200})"
)

def _parse_single_step(self, prompt: str, step_number: int) -> PlanStep | None:
    """Parse single step from prompt with input validation."""
    prompt_lower = prompt.lower().strip()

    # Input validation - prevent ReDoS
    if len(prompt_lower) > 1000:
        logger.warning(f"Prompt too long: {len(prompt_lower)} chars")
        return None

    # Calculator operations - SAFE
    if match := _OPERATION_PATTERN.search(prompt_lower):
        expression = match.group(1) or match.group(2)
        if expression:
            return PlanStep(
                step_number=step_number,
                tool_name="calculator",
                tool_input={"expression": expression.strip()},
                reasoning=f"Performing calculation: {expression.strip()}",
            )
```

**Implementation Steps**:
1. Add input length validation at method start
2. Limit capturing groups to 200 characters using `{1,200}`
3. Move pattern to module-level compiled constant
4. Add logging for rejected inputs
5. Add unit test with 1000-character payload

**Security Testing**:
```python
# tests/unit/planner/test_planner_security.py
def test_redos_protection():
    """Test ReDoS attack protection."""
    planner = PatternBasedPlanner()

    # Attack payload - should complete quickly
    attack = "add " + "x" * 1000 + "!"
    start = time.perf_counter()
    plan = planner.create_plan(attack)
    duration = time.perf_counter() - start

    assert duration < 0.1, f"Potential ReDoS: took {duration}s"
    assert len(plan.steps) == 0  # Should reject invalid input
```

---

### 1.2 Compile Regex Patterns 📊 HIGH ROI

**Priority**: HIGH
**Effort**: 2 hours
**Impact**: 30-50% performance improvement
**File**: `src/challenge/planner/planner.py`

**Problem**: All 7 regex patterns compiled on every method call (124-line method)

**Before Code** (patterns defined inside method):
```python
def _parse_single_step(self, prompt: str, step_number: int) -> PlanStep | None:
    """Parse single step - recompiles patterns every call."""
    prompt_lower = prompt.lower().strip()

    # Pattern 1 - compiled every time
    if match := re.search(r"(?:calculate|compute|evaluate|math|solve|what\s+is)\s+(.+)", prompt_lower):
        expression = match.group(1).strip()
        return PlanStep(...)

    # Pattern 2 - compiled every time
    if match := re.search(r"(?:multiply|divide|subtract)\s+(.+)|(?:add)\s+(?!(?:a\s+)?(?:todo|task)\b)(.+)", prompt_lower):
        expression = (match.group(1) or match.group(2)).strip()
        return PlanStep(...)

    # ... 5 more patterns compiled every call
```

**After Code** (module-level compiled patterns):
```python
# Module-level compiled patterns with length limits
_CALC_PATTERN = re.compile(
    r"(?:calculate|compute|evaluate|math|solve|what\s+is)\s+(.{1,200})"
)
_OPERATION_PATTERN = re.compile(
    r"(?:multiply|divide|subtract)\s+(.{1,200})"
    r"|(?:add)\s+(?!(?:a\s+)?(?:todo|task)\b)(.{1,200})"
)
_TODO_LIST_PATTERN = re.compile(
    r"(?:list|show|display|get|view|fetch)\s+(?:all\s+)?(?:my\s+)?todos?"
)
_TODO_CREATE_PATTERN = re.compile(
    r"(?:add|create|make|new)\s+(?:a\s+)?(?:todo|task)\s+['\"](.{1,200})['\"]"
)
_TODO_COMPLETE_PATTERN = re.compile(
    r"(?:complete|finish|done|mark\s+as\s+done)\s+(?:todo|task)\s+(.{1,100})"
)
_TODO_DELETE_PATTERN = re.compile(
    r"(?:delete|remove)\s+(?:todo|task)\s+(.{1,100})"
)
_NATURAL_MATH_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)"
)

def _parse_single_step(self, prompt: str, step_number: int) -> PlanStep | None:
    """Parse single step - uses pre-compiled patterns."""
    prompt_lower = prompt.lower().strip()

    # Input validation
    if len(prompt_lower) > 1000:
        logger.warning(f"Prompt exceeds max length: {len(prompt_lower)}")
        return None

    # Pattern 1 - reuses compiled pattern (fast)
    if match := _CALC_PATTERN.search(prompt_lower):
        expression = match.group(1).strip()
        return PlanStep(
            step_number=step_number,
            tool_name="calculator",
            tool_input={"expression": expression},
            reasoning=f"Performing calculation: {expression}",
        )

    # Pattern 2 - reuses compiled pattern (fast)
    if match := _OPERATION_PATTERN.search(prompt_lower):
        expression = (match.group(1) or match.group(2)).strip()
        return PlanStep(
            step_number=step_number,
            tool_name="calculator",
            tool_input={"expression": expression},
            reasoning=f"Performing operation: {expression}",
        )

    # ... rest uses pre-compiled patterns
```

**Performance Benchmark**:
```python
# tests/unit/planner/test_planner_performance.py
def test_pattern_compilation_performance():
    """Verify regex compilation performance improvement."""
    planner = PatternBasedPlanner()
    prompts = [
        "calculate 2 + 2",
        "list all todos",
        "add a todo 'test'",
        "multiply 5 by 3",
    ]

    # Warm-up
    for prompt in prompts:
        planner.create_plan(prompt)

    # Measure performance
    start = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        for prompt in prompts:
            planner.create_plan(prompt)
    duration = time.perf_counter() - start

    avg_time = duration / (iterations * len(prompts))
    assert avg_time < 0.001, f"Average time {avg_time}s exceeds target 1ms"
```

**Implementation Steps**:
1. Extract all 7 regex patterns to module level
2. Add `re.compile()` for each pattern with length limits
3. Update `_parse_single_step` to use compiled patterns
4. Add input length validation before pattern matching
5. Benchmark before/after performance
6. Update module docstring with pattern documentation

---

### 1.3 Fix Module Exports ✅ QUICK WIN

**Priority**: MEDIUM
**Effort**: 30 minutes
**Impact**: Improves type hints and public API clarity
**File**: `src/challenge/planner/__init__.py`

**Problem**: Module only exports `PatternBasedPlanner`, hiding `Planner` Protocol and `LLMPlanner`

**Before Code**:
```python
"""
Planner module for converting natural language to execution plans.
"""

from challenge.planner.planner import PatternBasedPlanner

__all__ = ["PatternBasedPlanner"]
```

**After Code**:
```python
"""
Planner module for converting natural language to execution plans.

This module provides:
- Planner: Protocol defining the planner interface
- PatternBasedPlanner: Regex-based pattern matching implementation (default)
- LLMPlanner: LLM-powered planning with structured output (requires API key)

Example:
    >>> from challenge.planner import Planner, PatternBasedPlanner
    >>> planner: Planner = PatternBasedPlanner()
    >>> plan = planner.create_plan("calculate 2 + 2")
"""

from challenge.planner.llm_planner import LLMPlanner
from challenge.planner.planner import PatternBasedPlanner
from challenge.planner.protocol import Planner

__all__ = [
    "Planner",              # Protocol for type hints
    "PatternBasedPlanner",  # Default implementation
    "LLMPlanner",          # LLM-powered implementation
]
```

**Usage Impact**:
```python
# Before - requires internal imports
from challenge.planner.protocol import Planner
from challenge.planner.llm_planner import LLMPlanner

# After - clean public API
from challenge.planner import Planner, LLMPlanner, PatternBasedPlanner

def create_orchestrator(planner: Planner) -> Orchestrator:
    """Type hints work correctly with exported Protocol."""
    return Orchestrator(planner=planner)
```

---

### 1.4 Add Input Length Validation 🛡️

**Priority**: HIGH
**Effort**: 1 hour
**Impact**: Prevents resource exhaustion attacks
**File**: `src/challenge/planner/planner.py`

**Before Code**:
```python
def create_plan(self, prompt: str) -> Plan:
    """Create plan from prompt - no validation."""
    steps: list[PlanStep] = []
    # ... continues without checking prompt length
```

**After Code**:
```python
# Module-level constants
_MAX_PROMPT_LENGTH = 2000
_MAX_EXPRESSION_LENGTH = 200

def create_plan(self, prompt: str) -> Plan:
    """
    Create plan from prompt with input validation.

    Args:
        prompt: Natural language task (max 2000 chars)

    Returns:
        Plan with validated steps

    Raises:
        ValueError: If prompt exceeds max length
    """
    # Input validation
    if not prompt or not prompt.strip():
        logger.warning("Empty prompt provided")
        return Plan(steps=[])

    if len(prompt) > _MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Prompt too long: {len(prompt)} chars (max {_MAX_PROMPT_LENGTH})"
        )

    steps: list[PlanStep] = []
    # ... rest of implementation
```

---

## Phase 2: Architecture Improvements (3-5 Days)

### 2.1 Extract Pattern Objects (Strategy Pattern) 🎯

**Priority**: MEDIUM
**Effort**: 8 hours
**Impact**: Reduces cyclomatic complexity 11 → 4, improves testability
**Files**: Create `src/challenge/planner/patterns/`

**Problem**: 124-line `_parse_single_step` method with 7 inline patterns (complexity 11)

**Current Architecture**:
```
PatternBasedPlanner._parse_single_step()  [124 lines, complexity 11]
├── if calculator_pattern: return PlanStep
├── if operation_pattern: return PlanStep
├── if todo_list_pattern: return PlanStep
├── if todo_create_pattern: return PlanStep
├── if todo_complete_pattern: return PlanStep
├── if todo_delete_pattern: return PlanStep
└── if natural_math_pattern: return PlanStep
```

**Target Architecture** (Strategy Pattern):
```
PatternBasedPlanner.create_plan()  [30 lines, complexity 2]
└── for pattern in self._patterns:
        if step := pattern.match(prompt, step_number):
            return step

Pattern Implementations:
├── CalculatorPattern
├── TodoListPattern
├── TodoCreatePattern
├── TodoCompletePattern
├── TodoDeletePattern
└── NaturalMathPattern
```

**New Pattern Protocol**:
```python
# src/challenge/planner/patterns/protocol.py
from typing import Protocol

from challenge.models.plan import PlanStep


class Pattern(Protocol):
    """Protocol for pattern matching strategies."""

    def match(self, prompt: str, step_number: int) -> PlanStep | None:
        """
        Attempt to match pattern against prompt.

        Args:
            prompt: Natural language prompt (preprocessed)
            step_number: Step number for PlanStep

        Returns:
            PlanStep if pattern matches, None otherwise
        """
        ...

    @property
    def priority(self) -> int:
        """
        Pattern matching priority (lower = higher priority).

        Used to order patterns for matching. More specific patterns
        should have lower priority values to match first.
        """
        ...
```

**Example Pattern Implementation**:
```python
# src/challenge/planner/patterns/calculator.py
import re
from challenge.models.plan import PlanStep
from challenge.planner.patterns.protocol import Pattern

# Compiled pattern with length limit
_CALC_PATTERN = re.compile(
    r"(?:calculate|compute|evaluate|math|solve|what\s+is)\s+(.{1,200})"
)
_OPERATION_PATTERN = re.compile(
    r"(?:multiply|divide|subtract)\s+(.{1,200})"
    r"|(?:add)\s+(?!(?:a\s+)?(?:todo|task)\b)(.{1,200})"
)


class CalculatorPattern:
    """Matches calculator operations with expression extraction."""

    @property
    def priority(self) -> int:
        """High priority - should match before generic patterns."""
        return 10

    def match(self, prompt: str, step_number: int) -> PlanStep | None:
        """Match calculator operation patterns."""
        prompt_lower = prompt.lower().strip()

        # Try explicit calculator keywords first
        if match := _CALC_PATTERN.search(prompt_lower):
            expression = match.group(1).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="calculator",
                tool_input={"expression": expression},
                reasoning=f"Performing calculation: {expression}",
            )

        # Try operation keywords
        if match := _OPERATION_PATTERN.search(prompt_lower):
            expression = (match.group(1) or match.group(2)).strip()
            return PlanStep(
                step_number=step_number,
                tool_name="calculator",
                tool_input={"expression": expression},
                reasoning=f"Performing operation: {expression}",
            )

        return None
```

**Refactored PatternBasedPlanner**:
```python
# src/challenge/planner/planner.py
from challenge.planner.patterns.calculator import CalculatorPattern
from challenge.planner.patterns.natural_math import NaturalMathPattern
from challenge.planner.patterns.todo_complete import TodoCompletePattern
from challenge.planner.patterns.todo_create import TodoCreatePattern
from challenge.planner.patterns.todo_delete import TodoDeletePattern
from challenge.planner.patterns.todo_list import TodoListPattern


class PatternBasedPlanner:
    """Pattern-based planner using Strategy pattern for extensibility."""

    def __init__(self):
        """Initialize with pattern matching strategies."""
        self._patterns = [
            CalculatorPattern(),
            NaturalMathPattern(),
            TodoListPattern(),
            TodoCreatePattern(),
            TodoCompletePattern(),
            TodoDeletePattern(),
        ]
        # Sort by priority (lower = higher priority)
        self._patterns.sort(key=lambda p: p.priority)

    def create_plan(self, prompt: str) -> Plan:
        """
        Create plan from prompt using pattern strategies.

        Reduced from 124 lines to 30 lines.
        Reduced complexity from 11 to 2.
        """
        # Input validation
        if not prompt or not prompt.strip():
            return Plan(steps=[])

        if len(prompt) > 2000:
            raise ValueError(f"Prompt too long: {len(prompt)} chars")

        steps: list[PlanStep] = []

        # Sequential planning
        if "then" in prompt.lower() or " and " in prompt.lower():
            parts = self._split_sequential(prompt)
            for i, part in enumerate(parts, start=1):
                if step := self._match_single_step(part, i):
                    steps.append(step)
        else:
            # Single step
            if step := self._match_single_step(prompt, step_number=1):
                steps.append(step)

        return Plan(steps=steps)

    def _match_single_step(self, prompt: str, step_number: int) -> PlanStep | None:
        """
        Match prompt against patterns in priority order.

        Complexity: 2 (was 11)
        """
        for pattern in self._patterns:  # Already sorted by priority
            if step := pattern.match(prompt, step_number):
                return step

        logger.info(f"No pattern matched for: {prompt[:50]}")
        return None

    def _split_sequential(self, prompt: str) -> list[str]:
        """Split prompt into sequential steps."""
        # Simple splitting on keywords
        parts = []
        for separator in [" then ", " and then ", " and "]:
            if separator in prompt.lower():
                parts = [p.strip() for p in prompt.lower().split(separator)]
                break
        return parts if parts else [prompt]
```

**Benefits**:
- **Complexity**: 11 → 2 (81% reduction)
- **Testability**: Each pattern testable in isolation
- **Extensibility**: New patterns added without modifying core logic
- **SOLID**: Single Responsibility (each pattern), Open/Closed (new patterns via extension)

**Testing Pattern**:
```python
# tests/unit/planner/patterns/test_calculator.py
def test_calculator_pattern_explicit_keyword():
    """Test calculator pattern with explicit keywords."""
    pattern = CalculatorPattern()

    step = pattern.match("calculate 2 + 2", step_number=1)

    assert step is not None
    assert step.tool_name == "calculator"
    assert step.tool_input == {"expression": "2 + 2"}
    assert "calculation" in step.reasoning.lower()


def test_calculator_pattern_operation_keyword():
    """Test calculator pattern with operation keywords."""
    pattern = CalculatorPattern()

    step = pattern.match("multiply 5 by 3", step_number=1)

    assert step is not None
    assert step.tool_name == "calculator"
    assert step.tool_input == {"expression": "5 by 3"}


def test_calculator_pattern_no_match():
    """Test calculator pattern rejects non-math prompts."""
    pattern = CalculatorPattern()

    step = pattern.match("list all todos", step_number=1)

    assert step is None
```

---

### 2.2 Fix Protocol Definition 🔧

**Priority**: LOW
**Effort**: 2 hours
**Impact**: Type safety for sync/async implementations
**File**: `src/challenge/planner/protocol.py`

**Problem**: Protocol only defines async method but PatternBasedPlanner is sync

**Before Code**:
```python
from typing import Protocol
from challenge.models.plan import Plan


class Planner(Protocol):
    """Protocol defining the planner interface."""

    async def create_plan(self, prompt: str) -> Plan:
        """
        Create execution plan from natural language prompt.

        Args:
            prompt: Natural language task description

        Returns:
            Plan with executable steps
        """
        ...
```

**After Code**:
```python
from typing import Awaitable, Protocol, Union
from challenge.models.plan import Plan


class Planner(Protocol):
    """
    Protocol defining the planner interface.

    Supports both synchronous and asynchronous implementations:
    - PatternBasedPlanner: Synchronous (regex matching)
    - LLMPlanner: Asynchronous (API calls)
    """

    def create_plan(self, prompt: str) -> Union[Plan, Awaitable[Plan]]:
        """
        Create execution plan from natural language prompt.

        Args:
            prompt: Natural language task description (max 2000 chars)

        Returns:
            Plan with executable steps, or awaitable Plan for async implementations

        Raises:
            ValueError: If prompt exceeds max length or is invalid

        Example (sync):
            >>> planner = PatternBasedPlanner()
            >>> plan = planner.create_plan("calculate 2 + 2")

        Example (async):
            >>> planner = LLMPlanner()
            >>> plan = await planner.create_plan("calculate 2 + 2")
        """
        ...
```

**Usage with inspect.iscoroutinefunction** (already implemented in orchestrator.py:98):
```python
# Orchestrator already handles both correctly
if inspect.iscoroutinefunction(self.planner.create_plan):
    plan = await self.planner.create_plan(prompt)
else:
    plan = self.planner.create_plan(prompt)
```

---

### 2.3 Add ExecutionContext Tests 🧪

**Priority**: HIGH
**Effort**: 2 hours
**Impact**: Validates critical inter-step state management
**File**: `tests/unit/orchestrator/test_execution_context.py` (NEW)

**Implementation**:
```python
# tests/unit/orchestrator/test_execution_context.py
import pytest

from challenge.models.run import ExecutionStep
from challenge.orchestrator.execution_context import ExecutionContext


class TestExecutionContext:
    """Test ExecutionContext variable resolution."""

    def test_record_step_basic(self):
        """Test basic step recording."""
        context = ExecutionContext()
        step = ExecutionStep(
            step_number=1,
            tool_name="calculator",
            tool_input={"expression": "2+2"},
            success=True,
            output=4.0,
            attempts=1,
        )

        context.record_step(step)

        assert context.step_outputs[1] == 4.0
        assert context.variables["step_1_output"] == 4.0
        assert context.variables["step_1_value"] == 4.0

    def test_extract_variables_from_list(self):
        """Test variable extraction from list output."""
        context = ExecutionContext()
        todos = [
            {"id": "abc-123", "text": "Buy milk"},
            {"id": "def-456", "text": "Walk dog"},
        ]
        step = ExecutionStep(
            step_number=1,
            tool_name="todo_store",
            tool_input={"action": "list"},
            success=True,
            output=todos,
            attempts=1,
        )

        context.record_step(step)

        assert context.variables["first_todo_id"] == "abc-123"
        assert context.variables["last_todo_id"] == "def-456"
        assert context.variables["step_1_count"] == 2
        assert context.variables["step_1_first_id"] == "abc-123"

    def test_resolve_variables_simple(self):
        """Test simple variable resolution."""
        context = ExecutionContext()
        context.variables["user_id"] = "abc-123"

        tool_input = {"action": "get", "id": "{user_id}"}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"action": "get", "id": "abc-123"}

    def test_resolve_variables_nested(self):
        """Test nested dict variable resolution."""
        context = ExecutionContext()
        context.variables["todo_id"] = "xyz-789"

        tool_input = {
            "action": "update",
            "params": {
                "id": "{todo_id}",
                "status": "complete"
            }
        }
        resolved = context.resolve_variables(tool_input)

        assert resolved["params"]["id"] == "xyz-789"
        assert resolved["params"]["status"] == "complete"

    def test_resolve_variables_list(self):
        """Test list variable resolution."""
        context = ExecutionContext()
        context.variables["id1"] = "aaa"
        context.variables["id2"] = "bbb"

        tool_input = {"ids": ["{id1}", "{id2}", "ccc"]}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"ids": ["aaa", "bbb", "ccc"]}

    def test_resolve_variables_partial_replacement(self):
        """Test partial string replacement."""
        context = ExecutionContext()
        context.variables["user"] = "alice"

        tool_input = {"message": "Hello {user}!"}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"message": "Hello alice!"}

    def test_resolve_variables_type_preservation(self):
        """Test type preservation for full variable replacement."""
        context = ExecutionContext()
        context.variables["count"] = 42
        context.variables["items"] = ["a", "b", "c"]

        tool_input = {
            "count": "{count}",
            "items": "{items}"
        }
        resolved = context.resolve_variables(tool_input)

        assert resolved["count"] == 42  # int preserved
        assert resolved["items"] == ["a", "b", "c"]  # list preserved

    def test_resolve_variables_not_found_error(self):
        """Test error on undefined variable."""
        context = ExecutionContext()
        context.variables["foo"] = "bar"

        tool_input = {"key": "{undefined}"}

        with pytest.raises(ValueError, match="Variable 'undefined' not found"):
            context.resolve_variables(tool_input)

    def test_resolve_variables_angle_brackets(self):
        """Test alternative <var> syntax."""
        context = ExecutionContext()
        context.variables["user_id"] = "abc-123"

        tool_input = {"action": "get", "id": "<user_id>"}
        resolved = context.resolve_variables(tool_input)

        assert resolved == {"action": "get", "id": "abc-123"}

    def test_get_step_output(self):
        """Test direct step output retrieval."""
        context = ExecutionContext()
        step = ExecutionStep(
            step_number=1,
            tool_name="test",
            tool_input={},
            success=True,
            output="result",
            attempts=1,
        )
        context.record_step(step)

        assert context.get_step_output(1) == "result"

    def test_get_step_output_not_found(self):
        """Test error on missing step."""
        context = ExecutionContext()

        with pytest.raises(KeyError, match="No output recorded for step 99"):
            context.get_step_output(99)

    def test_clear(self):
        """Test context clearing."""
        context = ExecutionContext()
        context.variables["foo"] = "bar"
        context.step_outputs[1] = "test"

        context.clear()

        assert len(context.variables) == 0
        assert len(context.step_outputs) == 0
        assert len(context.get_execution_log()) == 0
```

---

## Phase 3: Maintainability Improvements (2-3 Days)

### 3.1 Extract Tool Schemas (DRY Principle) 🔄

**Priority**: LOW
**Effort**: 1 day
**Impact**: Eliminates duplication, improves maintainability
**Files**: Create `src/challenge/planner/schemas.py`

**Problem**: Tool schemas duplicated across planner.py, llm_planner.py, examples.py

**Current Duplication**:
```python
# In planner.py
{"tool_name": "calculator", "tool_input": {"expression": "..."}}

# In llm_planner.py (PLAN_SCHEMA)
"tool_name": {"type": "string", "enum": ["calculator", "todo_store"]}

# In examples.py
tool_input={"action": "list"}
tool_input={"action": "create", "text": "..."}
```

**After Code**:
```python
# src/challenge/planner/schemas.py
"""Shared tool schemas for planners."""

from typing import Literal
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Calculator tool input schema."""
    expression: str = Field(..., max_length=200, description="Mathematical expression")


class TodoListInput(BaseModel):
    """Todo list tool input schema."""
    action: Literal["list"] = "list"


class TodoCreateInput(BaseModel):
    """Todo create tool input schema."""
    action: Literal["create"] = "create"
    text: str = Field(..., min_length=1, max_length=200, description="Todo text")


class TodoCompleteInput(BaseModel):
    """Todo complete tool input schema."""
    action: Literal["complete"] = "complete"
    todo_id: str = Field(..., description="Todo ID to complete")


class TodoDeleteInput(BaseModel):
    """Todo delete tool input schema."""
    action: Literal["delete"] = "delete"
    todo_id: str = Field(..., description="Todo ID to delete")


# Tool name enum
TOOL_NAMES = Literal["calculator", "todo_store"]

# Tool input union
TodoInput = TodoListInput | TodoCreateInput | TodoCompleteInput | TodoDeleteInput
ToolInput = CalculatorInput | TodoInput
```

**Usage in Pattern**:
```python
# In calculator.py pattern
from challenge.planner.schemas import CalculatorInput

class CalculatorPattern:
    def match(self, prompt: str, step_number: int) -> PlanStep | None:
        if match := _CALC_PATTERN.search(prompt):
            expression = match.group(1).strip()

            # Validate using schema
            tool_input = CalculatorInput(expression=expression)

            return PlanStep(
                step_number=step_number,
                tool_name="calculator",
                tool_input=tool_input.model_dump(),
                reasoning=f"Performing calculation: {expression}",
            )
```

---

### 3.2 Split PatternBasedPlanner Responsibilities 📦

**Priority**: MEDIUM
**Effort**: 2 days
**Impact**: Better Single Responsibility compliance
**Files**: Refactor `planner.py` → separate parsing and planning

**Current Responsibilities** (SRP violation):
```
PatternBasedPlanner:
├── Pattern matching (_parse_single_step)
├── Sequential parsing (_split_sequential)
├── Plan construction (create_plan)
└── UUID detection (_is_likely_uuid)
```

**Target Architecture**:
```
SequentialPromptParser:
└── parse(prompt: str) -> list[str]  # Split into parts

PatternMatcher:
└── match_step(prompt: str, step: int) -> PlanStep | None  # Pattern strategies

PatternBasedPlanner:
├── __init__(parser, matcher)
└── create_plan(prompt: str) -> Plan  # Orchestration only
```

**Implementation**:
```python
# src/challenge/planner/parsers.py
class SequentialPromptParser:
    """Parses prompts into sequential parts."""

    SEPARATORS = [" then ", " and then ", " and "]

    def parse(self, prompt: str) -> list[str]:
        """
        Split prompt into sequential parts.

        Args:
            prompt: Natural language prompt

        Returns:
            List of prompt parts for individual steps
        """
        prompt_lower = prompt.lower()

        for separator in self.SEPARATORS:
            if separator in prompt_lower:
                return [p.strip() for p in prompt.split(separator)]

        return [prompt]


# src/challenge/planner/matcher.py
from challenge.planner.patterns.protocol import Pattern

class PatternMatcher:
    """Matches prompts against pattern strategies."""

    def __init__(self, patterns: list[Pattern]):
        """Initialize with patterns sorted by priority."""
        self._patterns = sorted(patterns, key=lambda p: p.priority)

    def match_step(self, prompt: str, step_number: int) -> PlanStep | None:
        """
        Match prompt against patterns in priority order.

        Args:
            prompt: Natural language prompt part
            step_number: Step number for PlanStep

        Returns:
            PlanStep if any pattern matches, None otherwise
        """
        for pattern in self._patterns:
            if step := pattern.match(prompt, step_number):
                return step
        return None


# src/challenge/planner/planner.py (refactored)
from challenge.planner.matcher import PatternMatcher
from challenge.planner.parsers import SequentialPromptParser
from challenge.planner.patterns.calculator import CalculatorPattern
# ... other imports

class PatternBasedPlanner:
    """
    Pattern-based planner with composition.

    Responsibilities:
    - Input validation
    - Orchestrate parsing and matching
    - Construct Plan from steps
    """

    def __init__(
        self,
        parser: SequentialPromptParser | None = None,
        matcher: PatternMatcher | None = None,
    ):
        """Initialize with parser and matcher (dependency injection)."""
        self._parser = parser or SequentialPromptParser()

        if matcher is None:
            patterns = [
                CalculatorPattern(),
                NaturalMathPattern(),
                TodoListPattern(),
                TodoCreatePattern(),
                TodoCompletePattern(),
                TodoDeletePattern(),
            ]
            matcher = PatternMatcher(patterns)

        self._matcher = matcher

    def create_plan(self, prompt: str) -> Plan:
        """
        Create plan from prompt (orchestration only).

        Reduced to ~20 lines with clear responsibilities.
        """
        # Input validation
        if not prompt or not prompt.strip():
            return Plan(steps=[])

        if len(prompt) > 2000:
            raise ValueError(f"Prompt too long: {len(prompt)} chars")

        # Parse into parts
        parts = self._parser.parse(prompt)

        # Match each part to pattern
        steps: list[PlanStep] = []
        for i, part in enumerate(parts, start=1):
            if step := self._matcher.match_step(part, step_number=i):
                steps.append(step)

        return Plan(steps=steps)
```

---

### 3.3 Improve Error Handling 🔍

**Priority**: LOW
**Effort**: 2 hours
**Impact**: Better debugging and error diagnostics
**Files**: `llm_planner.py`, `planner.py`

**Problem**: Broad exception catching without distinguishing error types

**Before Code** (llm_planner.py):
```python
async def create_plan(self, prompt: str) -> Plan:
    try:
        response = await self.client.chat.completions.create(...)
        return Plan.model_validate(json.loads(response.choices[0].message.content))
    except Exception as e:
        logger.warning(f"LLM planning failed, using pattern-based fallback: {e}")
        return self.fallback.create_plan(prompt)
```

**After Code**:
```python
async def create_plan(self, prompt: str) -> Plan:
    """Create plan with detailed error handling."""
    try:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt),
            response_format={"type": "json_schema", "json_schema": PLAN_SCHEMA},
            temperature=self.temperature,
        )

        # Track token usage
        self.last_token_count = response.usage.total_tokens

        # Parse and validate response
        content = response.choices[0].message.content
        plan_data = json.loads(content)
        plan = Plan.model_validate(plan_data)

        logger.info(f"LLM planning succeeded ({self.last_token_count} tokens)")
        return plan

    except asyncio.TimeoutError:
        logger.warning("LLM API timeout, using fallback")
        return self.fallback.create_plan(prompt)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from LLM: {e}")
        return self.fallback.create_plan(prompt)

    except ValidationError as e:
        logger.error(f"LLM response validation failed: {e}")
        return self.fallback.create_plan(prompt)

    except Exception as e:
        logger.error(f"Unexpected LLM error: {e}", exc_info=True)
        return self.fallback.create_plan(prompt)
```

---

## Implementation Timeline

### Week 1: Critical Fixes
- **Day 1**: ReDoS fix + input validation + tests
- **Day 2**: Compile regex patterns + performance benchmarks
- **Day 3**: Module exports + ExecutionContext tests

### Week 2: Architecture
- **Day 4-5**: Extract Pattern Objects (Strategy pattern)
- **Day 6**: Fix Protocol definition + split responsibilities
- **Day 7**: Extract tool schemas + improve error handling

### Week 3: Testing & Documentation
- **Day 8**: Comprehensive test suite (≥80% coverage)
- **Day 9**: Performance benchmarks + security audit
- **Day 10**: Documentation + migration guide

---

## Success Metrics

### Code Quality Targets
- **Cyclomatic Complexity**: 11 → 4 (63% reduction)
- **Test Coverage**: Current → ≥80%
- **Performance**: 30-50% improvement in pattern matching
- **Security**: 0 critical vulnerabilities (ReDoS eliminated)
- **Maintainability**: Grade B+ → A

### Validation Checklist
- [ ] All security tests pass
- [ ] Performance benchmarks meet targets
- [ ] Test coverage ≥80%
- [ ] No ReDoS vulnerabilities
- [ ] Cyclomatic complexity <10
- [ ] Module exports complete
- [ ] Protocol correctly defines sync/async
- [ ] All patterns extracted and tested
- [ ] Documentation updated

---

## Risk Assessment

### Low Risk ✅
- Module exports fix (isolated change)
- Regex compilation (backward compatible)
- Input validation (additive only)
- ExecutionContext tests (new tests)

### Medium Risk ⚠️
- Pattern extraction (structural change, extensive testing required)
- Protocol fix (affects type hints, verify usages)
- Tool schema extraction (impacts multiple files)

### High Risk 🚨
- ReDoS fix (MUST validate security, add regression tests)
- Split responsibilities (architectural change, integration testing)

---

## Dependencies & Assumptions

### External Dependencies
- None (all stdlib or existing dependencies)

### Internal Dependencies
- **ExecutionContext**: Already implemented ✅
- **ExecutionEngine**: Already updated for context integration ✅
- **Test infrastructure**: pytest fixtures available

### Assumptions
- Backward compatibility maintained for public API
- Existing tests continue passing after refactoring
- Performance improvements validated with benchmarks
- Pattern extraction doesn't change behavior (verified by tests)

---

## Appendix: File Structure After Refactoring

```
src/challenge/planner/
├── __init__.py                    # Complete exports
├── protocol.py                     # Fixed Protocol (sync/async)
├── planner.py                      # Orchestration only (~50 lines)
├── llm_planner.py                  # Better error handling
├── examples.py                     # Existing examples
├── schemas.py                      # NEW: Shared tool schemas
├── parsers.py                      # NEW: Sequential parsing
├── matcher.py                      # NEW: Pattern matching
└── patterns/
    ├── __init__.py                # Export all patterns
    ├── protocol.py                # Pattern Protocol
    ├── calculator.py              # Calculator patterns
    ├── natural_math.py            # Natural math parsing
    ├── todo_list.py               # Todo list pattern
    ├── todo_create.py             # Todo create pattern
    ├── todo_complete.py           # Todo complete pattern
    └── todo_delete.py             # Todo delete pattern

tests/unit/planner/
├── test_planner.py                # Existing tests (keep)
├── test_llm_planner.py            # Existing tests (keep)
├── test_planner_security.py       # NEW: ReDoS tests
├── test_planner_performance.py    # NEW: Performance benchmarks
├── test_parsers.py                # NEW: Parser tests
├── test_matcher.py                # NEW: Matcher tests
└── patterns/
    ├── test_calculator.py         # NEW: Calculator pattern tests
    ├── test_natural_math.py       # NEW: Natural math tests
    ├── test_todo_list.py          # NEW: Todo list tests
    ├── test_todo_create.py        # NEW: Todo create tests
    ├── test_todo_complete.py      # NEW: Todo complete tests
    └── test_todo_delete.py        # NEW: Todo delete tests

tests/unit/orchestrator/
└── test_execution_context.py      # NEW: Context tests
```

---

## Questions for Review

1. **Priority Order**: Do you agree with Critical → High → Medium → Low prioritization?
2. **Pattern Extraction**: Should we extract patterns in Phase 2 or earlier?
3. **Testing Strategy**: Unit tests first or concurrent with implementation?
4. **Breaking Changes**: Is backward compatibility required for internal APIs?
5. **Performance Targets**: Are 30-50% improvements acceptable for compiled regex?

---

**Next Steps**: Review this plan, provide feedback, then proceed with Phase 1 implementation starting with ReDoS fix (highest priority).
