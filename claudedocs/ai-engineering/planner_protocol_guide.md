# Planner Protocol Guide

## Overview

The orchestrator now uses Python's **Protocol** for planner type hints, enabling structural subtyping and following the **Dependency Inversion Principle** (SOLID).

## What Changed

### Before (String Type Hints)
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from challenge.planner.llm_planner import LLMPlanner

class Orchestrator:
    def __init__(
        self,
        planner: "PatternBasedPlanner | LLMPlanner | None" = None,
        ...
    ):
```

**Problems:**
- ❌ String literals (forward references) are fragile
- ❌ Tight coupling to specific implementations
- ❌ Need to update for every new planner type
- ❌ Violates Dependency Inversion Principle

### After (Protocol)
```python
from challenge.planner.protocol import Planner

class Orchestrator:
    def __init__(
        self,
        planner: Planner | None = None,
        ...
    ):
```

**Benefits:**
- ✅ Clean, straightforward type hint
- ✅ Depends on abstraction, not concrete classes
- ✅ Any compatible planner works automatically
- ✅ Follows SOLID principles
- ✅ Better IDE support and type checking

## The Planner Protocol

### Definition
```python
from typing import Protocol
from challenge.models.plan import Plan

class Planner(Protocol):
    """
    Protocol defining the planner interface.

    Any class implementing a create_plan method with this signature
    can be used as a planner, regardless of inheritance hierarchy.
    """

    async def create_plan(self, prompt: str) -> Plan:
        """
        Create execution plan from natural language prompt.

        Args:
            prompt: Natural language task description

        Returns:
            Plan with ordered execution steps

        Raises:
            ValueError: If prompt is invalid or cannot be parsed
        """
        ...
```

### Key Features

1. **Structural Subtyping**: Classes don't need to inherit from anything
2. **Duck Typing with Type Safety**: If it has `create_plan()`, it's a planner
3. **Sync and Async Support**: Orchestrator handles both automatically
4. **No Registration Required**: Works immediately with any compatible class

## Usage Examples

### Using Existing Planners

```python
from challenge.orchestrator.orchestrator import Orchestrator
from challenge.planner.planner import PatternBasedPlanner
from challenge.planner.llm_planner import LLMPlanner

# Pattern-based planner (sync)
orchestrator = Orchestrator(planner=PatternBasedPlanner())

# LLM planner (async)
orchestrator = Orchestrator(planner=LLMPlanner(api_key="..."))

# Default (creates PatternBasedPlanner)
orchestrator = Orchestrator()
```

### Creating Custom Planners

#### Sync Custom Planner
```python
from challenge.models.plan import Plan, PlanStep

class SimplePlanner:
    """No inheritance needed - just implement the method."""

    def create_plan(self, prompt: str) -> Plan:
        """Create a simple plan."""
        return Plan(
            steps=[
                PlanStep(
                    step_number=1,
                    tool_name="calculator",
                    tool_input={"expression": "1+1"},
                    reasoning="Simple calculation",
                )
            ],
            final_goal=prompt,
        )

# Works immediately!
orchestrator = Orchestrator(planner=SimplePlanner())
```

#### Async Custom Planner
```python
class AsyncPlanner:
    """Async planner - also no inheritance needed."""

    async def create_plan(self, prompt: str) -> Plan:
        """Create plan asynchronously."""
        # Could call external API, database, etc.
        await some_async_operation()

        return Plan(
            steps=[...],
            final_goal=prompt,
        )

# Orchestrator handles async automatically
orchestrator = Orchestrator(planner=AsyncPlanner())
```

#### Advanced Custom Planner
```python
class DatabasePlanner:
    """Planner that loads plans from database."""

    def __init__(self, db_connection):
        self.db = db_connection

    async def create_plan(self, prompt: str) -> Plan:
        """Load plan from database based on prompt pattern."""
        # Find matching plan template
        template = await self.db.find_plan_template(prompt)

        # Generate plan from template
        return self._generate_from_template(template, prompt)

    def _generate_from_template(self, template, prompt) -> Plan:
        # Custom logic here
        ...

# Just works!
db_planner = DatabasePlanner(db_connection=my_db)
orchestrator = Orchestrator(planner=db_planner)
```

## How Orchestrator Handles Both Sync and Async

The orchestrator automatically detects whether a planner is sync or async:

```python
import inspect

# In Orchestrator.create_run():
if inspect.iscoroutinefunction(self.planner.create_plan):
    plan = await self.planner.create_plan(prompt)  # Async
else:
    plan = self.planner.create_plan(prompt)  # Sync
```

This means:
- ✅ Both sync and async planners work seamlessly
- ✅ No performance overhead for sync planners
- ✅ Proper async handling for async planners
- ✅ No special configuration needed

## Type Safety

### Type Checking
```bash
# Type checker accepts any compatible planner
ty check src/challenge/orchestrator/orchestrator.py
```

### IDE Support
Modern IDEs understand Protocol and provide:
- Autocomplete for `create_plan` method
- Type hints for parameters and return values
- Warning if class doesn't implement required method
- Refactoring support

### Runtime Behavior
Protocol is checked at **static analysis time**, not runtime:
- No runtime overhead
- No metaclass magic
- Pure duck typing with type safety
- Works like normal Python classes

## Design Principles (SOLID)

### Dependency Inversion Principle ✅
```
High-level module (Orchestrator) depends on abstraction (Planner Protocol)
Low-level modules (PatternBasedPlanner, LLMPlanner) implement abstraction
Both depend on the Protocol, not on each other
```

### Open/Closed Principle ✅
```
Orchestrator is open for extension (new planners)
Orchestrator is closed for modification (no code changes needed)
```

### Liskov Substitution Principle ✅
```
Any planner implementing the protocol can substitute for any other
Behavior is defined by the protocol contract
```

## Comparison: Protocol vs ABC

### Protocol (Structural Subtyping)
```python
from typing import Protocol

class Planner(Protocol):
    async def create_plan(self, prompt: str) -> Plan: ...

# No inheritance needed!
class MyPlanner:
    async def create_plan(self, prompt: str) -> Plan:
        ...
```

**Pros:**
- ✅ No inheritance required
- ✅ Works with existing code
- ✅ More flexible
- ✅ Better for third-party integrations
- ✅ Pythonic and modern

**Cons:**
- ⚠️ No runtime enforcement
- ⚠️ Less explicit (no inheritance tree)

### Abstract Base Class (Nominal Subtyping)
```python
from abc import ABC, abstractmethod

class PlannerBase(ABC):
    @abstractmethod
    async def create_plan(self, prompt: str) -> Plan: ...

# Requires explicit inheritance
class MyPlanner(PlannerBase):
    async def create_plan(self, prompt: str) -> Plan:
        ...
```

**Pros:**
- ✅ Explicit inheritance tree
- ✅ Runtime enforcement
- ✅ Traditional OOP approach

**Cons:**
- ❌ Requires inheritance
- ❌ Must modify existing classes
- ❌ Less flexible
- ❌ Harder to integrate third-party code

### Why Protocol Was Chosen

For this codebase, Protocol is superior because:
1. **Existing code works without modification**: PatternBasedPlanner and LLMPlanner already have `create_plan()` methods
2. **No refactoring needed**: No need to add inheritance
3. **Future-proof**: Any new planner automatically works
4. **Third-party friendly**: External planner libraries can be used without modification
5. **Modern Python**: Embraces PEP 544 structural subtyping

## Testing Protocol Compliance

### Test Custom Planner
```python
import pytest
from challenge.orchestrator.orchestrator import Orchestrator

class MyCustomPlanner:
    async def create_plan(self, prompt: str) -> Plan:
        # Implementation
        ...

@pytest.mark.asyncio
async def test_custom_planner():
    """Test custom planner works with orchestrator."""
    planner = MyCustomPlanner()
    orchestrator = Orchestrator(planner=planner)

    run = await orchestrator.create_run("test prompt")

    assert run.plan is not None
    # More assertions...
```

### Test Type Safety
```python
def test_type_checking():
    """Verify type checker accepts custom planner."""
    planner = MyCustomPlanner()

    # This should pass type checking
    orchestrator = Orchestrator(planner=planner)

    # Type checker knows planner has create_plan
    assert hasattr(orchestrator.planner, 'create_plan')
```

## Migration Guide

### For Existing Code
**No changes needed!** All existing planners work without modification:
- `PatternBasedPlanner` ✅
- `LLMPlanner` ✅
- Custom planners ✅

### For New Planners
1. Create class with `create_plan(prompt: str) -> Plan` method
2. Make it sync or async (both work)
3. Use with orchestrator - that's it!

### Example Migration
```python
# Old approach (would require this with ABC)
class NewPlanner(PlannerBase):  # Must inherit
    async def create_plan(self, prompt: str) -> Plan:
        ...

# New approach (Protocol - no inheritance)
class NewPlanner:  # Just implement the method!
    async def create_plan(self, prompt: str) -> Plan:
        ...
```

## Best Practices

### 1. Follow Protocol Signature
```python
# ✅ Correct - matches protocol
async def create_plan(self, prompt: str) -> Plan:
    ...

# ❌ Wrong - different signature
async def create_plan(self, input_text: str) -> dict:
    ...
```

### 2. Handle Errors Properly
```python
async def create_plan(self, prompt: str) -> Plan:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Implementation
    ...
```

### 3. Document Custom Planners
```python
class CustomPlanner:
    """
    Custom planner for specific use case.

    Implements the Planner protocol for compatibility with Orchestrator.

    Example:
        >>> planner = CustomPlanner()
        >>> orchestrator = Orchestrator(planner=planner)
    """

    async def create_plan(self, prompt: str) -> Plan:
        """
        Create execution plan from prompt.

        Args:
            prompt: Natural language task description

        Returns:
            Plan with execution steps

        Raises:
            ValueError: If prompt is invalid
        """
        ...
```

### 4. Test Protocol Compliance
Always test that custom planners work with orchestrator:
```python
@pytest.mark.asyncio
async def test_custom_planner_with_orchestrator():
    planner = CustomPlanner()
    orchestrator = Orchestrator(planner=planner)

    run = await orchestrator.create_run("test")
    assert run.plan is not None
```

## References

- **PEP 544**: Protocol (Structural Subtyping) - https://peps.python.org/pep-0544/
- **Python typing docs**: https://docs.python.org/3/library/typing.html#typing.Protocol
- **SOLID Principles**: https://en.wikipedia.org/wiki/SOLID
- **Duck Typing**: https://en.wikipedia.org/wiki/Duck_typing

## Summary

The Protocol-based approach provides:
- ✅ **Type safety** without inheritance
- ✅ **Flexibility** for any compatible planner
- ✅ **SOLID compliance** (Dependency Inversion)
- ✅ **Backward compatibility** with existing code
- ✅ **Future-proof** design for extensibility
- ✅ **Modern Python** best practices

Any class with a `create_plan(prompt: str) -> Plan` method can now be used as a planner, making the system more flexible, maintainable, and extensible.
