# Variable Resolution Fix for LLM Planner

## Problem

When running the application with a complex multi-step prompt like:
```
"Calculate (42 * 8) + 15, then use the result and multiply by 2, and add the result as a todo"
```

The LLM was generating plans with **custom variable names** like `{calculation_result}` instead of the **standard format** `{step_N_output}`, causing variable resolution failures:

```
ERROR: Variable 'calculation_result' not found in execution context.
Available variables: step_1_output, step_1_value
```

## Root Cause

Two issues in the LLM system prompt and examples:

1. **Ambiguous prompt wording** (line 186): "you can reference outputs from previous steps using {variable_name} syntax"
   - LLM interpreted this as "use any variable name you want"
   - Should have been explicit: "use ONLY these specific patterns"

2. **Incorrect few-shot example** (`EXAMPLE_CALCULATION_THEN_TODO`):
   - Showed hardcoded result: `"Result of 3 * 4 = 12"`
   - Should have demonstrated: `"Result: {step_1_output}"`
   - Teaching the LLM the WRONG pattern!

3. **Missing chained calculation example**:
   - No example showing calc → calc → todo pattern
   - Exactly what user was trying to do

## Solution

### 1. Updated System Prompt (llm_planner.py)

**Before**:
```python
- For multi-step workflows, you can reference outputs from previous steps using {variable_name} syntax

Variable Resolution:
When a step needs data from a previous step's output, use these variable references:
- {first_todo_id} - ID of first item in a list output
- {step_N_output} - Full output from step N (replace N with step number)
...
```

**After**:
```python
Variable Resolution:
CRITICAL: When a step needs data from a previous step's output, you MUST use ONLY these exact variable patterns:
- {step_N_output} - Full output from step N (replace N with step number, e.g., {step_1_output}, {step_2_output})
- {first_todo_id} - ID of first item in a list output
...

DO NOT use custom variable names like {calculation_result}, {result}, {value}, etc.
ALWAYS use {step_N_output} where N is the step number.

Examples:
1. Calculator workflow:
   User: "Calculate (42 * 8) + 15, then multiply by 2"
   Step 1: {"expression": "(42 * 8) + 15"}
   Step 2: {"expression": "{step_1_output} * 2"} ✓ CORRECT - uses {step_1_output}
   Step 2: {"expression": "{calculation_result} * 2"} ✗ WRONG - custom variable name
```

### 2. Fixed Example 3 (examples.py)

**Before**:
```python
PlanStep(
    step_number=2,
    tool_name="todo_store",
    tool_input={"action": "add", "text": "Result of 3 * 4 = 12"},  # ❌ WRONG
    reasoning="Store calculation result as a todo item for reference",
)
```

**After**:
```python
PlanStep(
    step_number=2,
    tool_name="todo_store",
    tool_input={"action": "add", "text": "Result: {step_1_output}"},  # ✅ CORRECT
    reasoning="Store calculation result as todo using {step_1_output} variable",
)
```

### 3. Added New Example 8: Chained Calculations

```python
EXAMPLE_CHAINED_CALCULATIONS = FewShotExample(
    prompt="Calculate 10 + 5, then multiply the result by 2, and add it as a todo",
    reasoning="""
    This requires chaining calculator operations with variable resolution:
    1. First, calculate 10 + 5 using calculator
    2. Then, multiply the result by 2 using {step_1_output}
    3. Finally, add the final result as a todo using {step_2_output}
    CRITICAL: Each step references the previous step's output using {step_N_output} syntax.
    """,
    plan=Plan(
        steps=[
            PlanStep(
                step_number=1,
                tool_name="calculator",
                tool_input={"expression": "10 + 5"},
                reasoning="Calculate initial value 10 + 5",
            ),
            PlanStep(
                step_number=2,
                tool_name="calculator",
                tool_input={"expression": "{step_1_output} * 2"},
                reasoning="Multiply result from step 1 by 2 using {step_1_output}",
            ),
            PlanStep(
                step_number=3,
                tool_name="todo_store",
                tool_input={"action": "add", "text": "Calculation result: {step_2_output}"},
                reasoning="Store final result as todo using {step_2_output} variable",
            ),
        ],
        final_goal="Perform chained calculations and store final result as todo",
    ),
    complexity="complex",
)
```

## Files Modified

1. `src/challenge/planner/llm_planner.py` - Updated system prompt with explicit variable naming rules
2. `src/challenge/planner/examples.py` - Fixed Example 3, added Example 8, updated ALL_EXAMPLES
3. `tests/unit/planner/test_examples.py` - Updated test to expect 8 examples instead of 7
4. `tests/unit/planner/test_llm_planner.py` - Moved imports to top-level (linter fix)

## Testing

### Unit Tests
All 97 planner unit tests pass:
```bash
uv run pytest tests/unit/planner/ -v
# Result: 97 passed in 0.33s ✅
```

### Manual Testing

To verify the fix works with the real application:

1. **Start the application**:
   ```bash
   uv run python -m challenge
   ```

2. **Test the problematic prompt**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/runs \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Calculate (42 * 8) + 15, then use the result and multiply by 2, and add the result as a todo"}'
   ```

3. **Expected behavior**:
   - ✅ LLM generates plan with `{step_1_output}` and `{step_2_output}`
   - ✅ ExecutionContext resolves variables correctly
   - ✅ Step 1 calculates: (42 * 8) + 15 = 351
   - ✅ Step 2 calculates: 351 * 2 = 702
   - ✅ Step 3 adds todo: "Result: 702"

4. **Check logs**:
   Look for successful variable resolution instead of errors:
   ```
   INFO - LLM planning succeeded
   INFO - Step 1 completed successfully: 351
   INFO - Step 2 completed successfully: 702
   INFO - Step 3 completed successfully
   ```

## Impact

- **Immediate**: Fixes variable resolution failures in multi-step calculator workflows
- **Broader**: Improves LLM prompt engineering with clearer instructions and better examples
- **Future**: Reduces similar issues by establishing clear variable naming conventions

## Key Learnings

1. **LLM prompts must be EXPLICIT**: Vague wording like "you can use {variable_name}" leads to creative interpretations
2. **Few-shot examples are critical**: Bad examples teach bad patterns, even with good instructions
3. **Examples should match real use cases**: User's actual prompt matched Example 8 (now added)
4. **Test with real LLMs**: Unit tests passed but real LLM behavior revealed prompt ambiguity
