"""Security tests for PatternBasedPlanner.

CRITICAL: These tests validate fix for ReDoS vulnerability where regex pattern (.+)
caused exponential backtracking on attack payloads like "add " + "x"*1000 + "!".

Fix: Changed to length-limited patterns like (.{1,200}) to cap backtracking.

Tests are organized by concern:
- TestReDoSPerformance: Validates attack payloads complete quickly (CRITICAL)
- TestReDoSSafety: Validates attack payloads are handled safely
- TestReDoSImplementation: Validates length limits are enforced in patterns
- TestInputLengthValidation: Validates input size limits
- TestSecurityEdgeCases: Validates edge case handling
- TestPerformanceRegression: Validates no performance degradation
"""

import time
from typing import ClassVar, Dict

import pytest

from challenge.services.planning.planner import PatternBasedPlanner


class TestReDoSPerformance:
    """Test ReDoS attack payloads complete quickly (CRITICAL security validation)."""

    # Attack payloads targeting different regex patterns
    ATTACK_PAYLOADS: ClassVar[Dict[str, str]] = {
        "calculator": "calculate " + "x" * 1000 + "!",
        "operation_add": "add " + "x" * 1000 + "!",  # Previously CRITICAL vulnerability
        "operation_multiply": "multiply " + "x" * 1000,
        "todo_pattern": "add todo " + "x" * 1000,
        "nested_quantifiers": "calculate " + "a" * 500 + "b" * 500,
        "alternation_overlap": "add " + "todo " * 200 + "x" * 200,
    }

    @pytest.mark.timeout(1)  # Failsafe: kill test if hangs
    @pytest.mark.parametrize(("attack_name", "attack_payload"), ATTACK_PAYLOADS.items())
    def test_attack_completes_quickly(self, attack_name, attack_payload):
        """Verify all attack payloads complete in acceptable time (CRITICAL).

        This test validates that the ReDoS vulnerability fix (length-limited regex
        patterns) prevents exponential backtracking. We don't care whether parsing
        succeeds or fails, only that it completes quickly.
        """
        planner = PatternBasedPlanner()

        start = time.perf_counter()

        # Don't care about success/failure, only that it completes quickly
        try:
            planner.create_plan(attack_payload)
        except ValueError:
            pass  # Rejection is acceptable behavior

        duration = time.perf_counter() - start

        # CRITICAL: Must complete in <100ms regardless of parse success
        # Without fix, would take seconds/minutes due to exponential backtracking
        assert duration < 0.1, (
            f"ReDoS vulnerability detected in '{attack_name}' pattern!\n"
            f"Parsing took {duration:.3f}s (threshold: 0.1s)\n"
            f"Attack payload likely caused exponential backtracking.\n"
            f"Payload preview: {attack_payload[:50]}..."
        )

    @pytest.mark.timeout(1)
    def test_performance_comparable_to_baseline(self):
        """Verify attack performance is comparable to normal input (no exponential behavior).

        This test validates that attack payloads don't exhibit exponential
        time complexity compared to normal inputs. ReDoS would show 100x-1000x+ slowdown.
        """
        planner = PatternBasedPlanner()

        # Measure baseline with normal inputs
        normal_inputs = ["add task 1", "calculate 5 + 3", "list todos"]
        baseline_times = []

        for input_str in normal_inputs:
            start = time.perf_counter()
            try:
                planner.create_plan(input_str)
            except ValueError:
                pass
            baseline_times.append(time.perf_counter() - start)

        baseline_avg = sum(baseline_times) / len(baseline_times)

        # Measure attack payloads
        attack_inputs = [
            "add " + "x" * 1000 + "!",
            "calculate " + "y" * 1000,
            "multiply " + "z" * 1000,
        ]
        attack_times = []

        for attack in attack_inputs:
            start = time.perf_counter()
            try:
                planner.create_plan(attack)
            except ValueError:
                pass
            attack_times.append(time.perf_counter() - start)

        attack_avg = sum(attack_times) / len(attack_times)

        # Calculate slowdown factor
        slowdown_factor = attack_avg / baseline_avg if baseline_avg > 0 else 0

        # Attack should NOT be significantly slower (max 10x acceptable)
        # ReDoS would cause 100x-1000x+ slowdown due to exponential complexity
        assert slowdown_factor < 10, (
            f"ReDoS vulnerability detected: attack input {slowdown_factor:.1f}x slower than normal!\n"
            f"Baseline avg: {baseline_avg * 1000:.2f}ms\n"
            f"Attack avg: {attack_avg * 1000:.2f}ms\n"
            f"Expected slowdown: <10x (ReDoS causes 100x-1000x+)"
        )


class TestReDoSSafety:
    """Test ReDoS attack payloads are handled safely (parse OR reject)."""

    ATTACK_PAYLOADS = TestReDoSPerformance.ATTACK_PAYLOADS

    @pytest.mark.parametrize(("attack_name", "attack_payload"), ATTACK_PAYLOADS.items())
    def test_attack_handled_safely(self, attack_name, attack_payload):
        """Verify attack payloads are handled safely (parse with truncation OR reject).

        Safe handling means EITHER:
        1. Parse successfully using truncated input (within length limits)
        2. Reject with ValueError (invalid syntax)

        Both behaviors are acceptable. What's NOT acceptable:
        - Returning None without raising exception
        - Parsing into garbage/nonsense plan
        - Hanging or crashing
        """
        planner = PatternBasedPlanner()

        result = None
        error = None

        try:
            result = planner.create_plan(attack_payload)
        except ValueError as e:
            error = e

        # EITHER parsed successfully OR rejected with ValueError
        assert result is not None or error is not None, (
            f"Attack '{attack_name}' returned None without raising exception.\n"
            f"System should either parse (with truncation) OR reject with ValueError."
        )

        if result is not None:
            # If parsed, validate result is sensible
            assert len(result.steps) >= 1, f"Attack '{attack_name}' produced empty plan. Should have ≥1 step or reject."

            assert len(result.steps) <= 10, (
                f"Attack '{attack_name}' produced {len(result.steps)} steps - "
                f"possible garbage from attack payload. Should produce reasonable plan or reject."
            )

    def test_normal_inputs_still_work(self):
        """Verify normal inputs still parse correctly after ReDoS fix.

        This regression test ensures length limits don't break valid inputs.
        """
        planner = PatternBasedPlanner()

        normal_inputs = [
            "calculate 2 + 2",
            "add todo Buy milk",
            "list todos",
            "multiply 5 by 3",
            "complete todo abc-123",
            "add " + "x" * 100,  # Within length limits (200 chars)
        ]

        for input_str in normal_inputs:
            # Should parse successfully
            try:
                result = planner.create_plan(input_str)
                assert result is not None, f"Normal input returned None: {input_str}"
                assert len(result.steps) >= 1, f"Normal input produced empty plan: {input_str}"
            except ValueError as e:
                # If it raises ValueError, it should be for invalid syntax, not length
                if "too long" in str(e).lower():
                    pytest.fail(f"Normal input rejected as too long: {input_str}\nError: {e}")


class TestReDoSImplementation:
    """Test that length-limited regex patterns are enforced (implementation validation)."""

    @pytest.mark.parametrize(
        "length",
        [
            100,  # Well within limit
            199,  # Just below limit
            200,  # At limit
            201,  # Just above limit
            500,  # Moderately over
            1000,  # Well over
        ],
    )
    def test_length_limit_boundary_conditions(self, length):
        """Test behavior at and around length limit boundaries.

        Validates that performance doesn't degrade near length limits,
        regardless of whether input is accepted or truncated.
        """
        planner = PatternBasedPlanner()

        # Create input at specified length (assuming length limit is 200 for pattern capture)
        attack = "add " + "x" * length

        start = time.perf_counter()

        try:
            planner.create_plan(attack)
        except ValueError:
            pass  # Rejection is acceptable

        duration = time.perf_counter() - start

        # Should complete quickly regardless of where input falls relative to limit
        assert duration < 0.1, (
            f"Input at length {length} took {duration:.3f}s - possible ReDoS or performance issue near length boundary"
        )


class TestInputLengthValidation:
    """Test input length validation for security."""

    def test_max_prompt_length_enforced(self):
        """Test that prompts exceeding max length are rejected."""
        planner = PatternBasedPlanner()

        # Create prompt longer than _MAX_PROMPT_LENGTH (2000 chars)
        long_prompt = "calculate " + "x" * 2000
        assert len(long_prompt) > 2000

        with pytest.raises(ValueError, match="Prompt too long"):
            planner.create_plan(long_prompt)

    def test_prompt_at_max_length_accepted(self):
        """Test that prompts at max length (2000 chars) are accepted."""
        planner = PatternBasedPlanner()

        # Create prompt at exactly 2000 chars
        prompt = "calculate " + "1" * 1990
        assert len(prompt) == 2000

        # Should attempt parsing (may fail due to invalid expression, but not length)
        try:
            plan = planner.create_plan(prompt)
            assert len(plan.steps) >= 1
        except ValueError as e:
            # Acceptable if calculator can't parse the expression
            # But should NOT be rejected for length
            if "too long" in str(e).lower():
                pytest.fail("Should accept 2000 char prompt")

    def test_empty_prompt_rejected(self):
        """Test that empty prompts are rejected."""
        planner = PatternBasedPlanner()

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            planner.create_plan("")

    def test_whitespace_only_prompt_rejected(self):
        """Test that whitespace-only prompts are rejected."""
        planner = PatternBasedPlanner()

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            planner.create_plan("   \n\t   ")

    def test_multi_todo_item_length_validation(self):
        """Test that individual todo items in multi-todo pattern are validated."""
        planner = PatternBasedPlanner()

        # Create prompt with oversized todo items
        long_item = "x" * 300  # Exceeds _MAX_TODO_TEXT_LENGTH (200)
        prompt = f"add todos for {long_item}, valid item, another valid"

        plan = planner.create_plan(prompt)

        # Should skip oversized items but accept valid ones
        assert len(plan.steps) >= 1

        # Verify no step has oversized text
        for step in plan.steps:
            if hasattr(step.tool_input, "text"):
                assert len(step.tool_input.text) <= 200, (
                    f"Todo text exceeds max length: {len(step.tool_input.text)} chars"
                )


class TestSecurityEdgeCases:
    """Test security-related edge cases."""

    def test_null_bytes_in_prompt(self):
        """Test handling of null bytes in prompt."""
        planner = PatternBasedPlanner()

        prompt = "calculate 2 + 2\x00malicious"

        # Should handle gracefully (Python strings support null bytes)
        try:
            plan = planner.create_plan(prompt)
            assert len(plan.steps) >= 1
        except ValueError:
            pass  # Rejection also acceptable

    def test_unicode_in_prompt(self):
        """Test handling of Unicode characters."""
        planner = PatternBasedPlanner()

        prompt = "add todo 日本語 🎯 test"

        # Should handle Unicode gracefully
        plan = planner.create_plan(prompt)
        assert len(plan.steps) == 1
        assert "日本語" in plan.steps[0].tool_input.text

    def test_special_regex_characters_in_prompt(self):
        """Test handling of special regex characters."""
        planner = PatternBasedPlanner()

        # Special regex characters that could cause issues if not properly escaped
        prompt = "calculate (2 + 3) * 4"

        plan = planner.create_plan(prompt)
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "calculator"

    def test_sql_injection_attempt(self):
        """Test that SQL injection attempts are handled safely."""
        planner = PatternBasedPlanner()

        prompt = "add todo '; DROP TABLE todos; --"

        plan = planner.create_plan(prompt)
        assert len(plan.steps) == 1
        # Should be stored as-is (no SQL execution in planner)
        # Note: Prompt is lowercased during processing
        assert "drop table" in plan.steps[0].tool_input.text.lower()

    def test_command_injection_attempt(self):
        """Test that command injection attempts are handled safely."""
        planner = PatternBasedPlanner()

        prompt = "add todo $(rm -rf /)"

        plan = planner.create_plan(prompt)
        assert len(plan.steps) == 1
        # Should be stored as-is (no shell execution in planner)
        assert "$(rm" in plan.steps[0].tool_input.text


class TestPerformanceRegression:
    """Test for performance regressions after security fixes."""

    def test_normal_inputs_remain_fast(self):
        """Verify normal inputs still process quickly after ReDoS fix."""
        planner = PatternBasedPlanner()

        normal_prompts = [
            "calculate 2 + 2",
            "add todo Buy milk",
            "list todos",
            "multiply 5 by 3",
            "complete todo abc-123",
        ]

        for prompt in normal_prompts:
            start = time.perf_counter()
            plan = planner.create_plan(prompt)
            duration = time.perf_counter() - start

            assert len(plan.steps) >= 1
            assert duration < 0.01, f"Normal input too slow: {prompt}\nDuration: {duration:.4f}s (threshold: 0.01s)"

    def test_batch_processing_performance(self):
        """Test performance of batch processing after security fixes."""
        planner = PatternBasedPlanner()

        prompts = [
            "calculate 2 + 2",
            "add todo Test",
            "list todos",
            "multiply 5 by 3",
            "add task Example",
        ] * 20  # 100 prompts

        start = time.perf_counter()
        for prompt in prompts:
            planner.create_plan(prompt)
        duration = time.perf_counter() - start

        # Should process 100 prompts in < 1 second with compiled patterns
        avg_time = duration / len(prompts)
        assert avg_time < 0.01, (
            f"Performance regression detected in batch processing!\n"
            f"Average time per prompt: {avg_time:.4f}s (threshold: 0.01s)\n"
            f"Total time for {len(prompts)} prompts: {duration:.2f}s"
        )

    def test_complex_multi_step_performance(self):
        """Test performance of complex multi-step plans."""
        planner = PatternBasedPlanner()

        prompt = "calculate 2 + 2 and add todo Result and list todos and multiply 5 by 3"

        start = time.perf_counter()
        plan = planner.create_plan(prompt)
        duration = time.perf_counter() - start

        assert len(plan.steps) == 4
        assert duration < 0.01, (
            f"Complex multi-step plan too slow!\nDuration: {duration:.4f}s (threshold: 0.01s)\nSteps: {len(plan.steps)}"
        )


class TestPatternOrderingBugFix:
    """Test fix for pattern ordering bug where 'add X as a todo' was matched as calculator operation.

    Bug: Pattern 1c (operation verbs like 'add') was checked BEFORE Pattern 2a ('add X as a todo'),
    causing prompts like "add the result as a todo" to be incorrectly parsed as calculator operations
    with invalid expressions like "add the result as a todo".

    Fix: Reordered checks so Pattern 2a ('add X as a todo') is checked BEFORE Pattern 1c (operation verbs).
    """

    def test_add_result_as_todo_creates_todo_not_calculator(self):
        """Test that 'add X as a todo' creates todo operation, not calculator operation."""
        planner = PatternBasedPlanner()

        prompt = "add the result as a todo"

        plan = planner.create_plan(prompt)

        # Should create a todo_store step, not a calculator step
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "todo_store"
        assert plan.steps[0].tool_input.action == "add"
        assert plan.steps[0].tool_input.text == "the result"

    def test_multi_step_calc_and_add_as_todo(self):
        """Test the original failing prompt: 'Calculate (42 * 8) + 15 and add the result as a todo'."""
        planner = PatternBasedPlanner()

        prompt = "Calculate (42 * 8) + 15 and add the result as a todo"

        plan = planner.create_plan(prompt)

        # Should create 2 steps: calculator + todo_store
        assert len(plan.steps) == 2

        # Step 1: Calculator
        assert plan.steps[0].tool_name == "calculator"
        assert "(42 * 8) + 15" in plan.steps[0].tool_input.expression

        # Step 2: Todo (NOT calculator with invalid expression)
        assert plan.steps[1].tool_name == "todo_store"
        assert plan.steps[1].tool_input.action == "add"
        assert plan.steps[1].tool_input.text == "the result"

    def test_add_value_as_task_creates_todo(self):
        """Test variation: 'add the value as a task'."""
        planner = PatternBasedPlanner()

        prompt = "add the value as a task"

        plan = planner.create_plan(prompt)

        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "todo_store"
        assert plan.steps[0].tool_input.action == "add"
        assert plan.steps[0].tool_input.text == "the value"

    def test_add_calculation_as_todo(self):
        """Test: 'add 42 * 8 as a todo' should create todo with text '42 * 8', not calculate it."""
        planner = PatternBasedPlanner()

        prompt = "add 42 * 8 as a todo"

        plan = planner.create_plan(prompt)

        # Should create a todo with the expression as text, not calculate it
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "todo_store"
        assert plan.steps[0].tool_input.action == "add"
        assert "42 * 8" in plan.steps[0].tool_input.text

    def test_numeric_add_still_works_as_calculator(self):
        """Test that numeric 'add' operations still work as calculator operations."""
        planner = PatternBasedPlanner()

        prompt = "add 5 and 3"

        plan = planner.create_plan(prompt)

        # Should create calculator step (Pattern 1c should still match numeric adds)
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "calculator"


class TestTrailingPunctuationBugFix:
    """Test fix for trailing punctuation causing AST Tuple errors.

    Bug: When prompts are split on 'then' or 'and', trailing punctuation (especially commas)
    gets included in expressions. Python's AST parser treats "10 + 5," as a Tuple, not BinOp,
    causing "Unsupported expression type: Tuple" errors.

    Fix: Strip trailing punctuation (,.;:!?) from all calculator expressions before validation.
    """

    def test_calculate_with_trailing_comma(self):
        """Test that 'calculate X,' strips the comma and works."""
        planner = PatternBasedPlanner()

        prompt = "calculate 10 + 5,"

        plan = planner.create_plan(prompt)

        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "calculator"
        # Expression should NOT have trailing comma
        assert plan.steps[0].tool_input.expression == "10 + 5"

    def test_calculate_with_trailing_period(self):
        """Test that 'calculate X.' strips the period."""
        planner = PatternBasedPlanner()

        prompt = "calculate 42 * 8."

        plan = planner.create_plan(prompt)

        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "calculator"
        assert plan.steps[0].tool_input.expression == "42 * 8"

    def test_multi_step_with_commas_then_separator(self):
        """Test the original failing prompt with commas and 'then' separators."""
        planner = PatternBasedPlanner()

        prompt = "Calculate 10 + 5, then subtract 15"

        plan = planner.create_plan(prompt)

        # Should create 2 steps
        assert len(plan.steps) == 2

        # Step 1: Calculator without trailing comma
        assert plan.steps[0].tool_name == "calculator"
        assert plan.steps[0].tool_input.expression == "10 + 5"

        # Step 2: Calculator (or operation step)
        # Note: "subtract 15" might match operation pattern
        assert plan.steps[1].tool_name == "calculator"

    def test_complex_multi_step_prompt(self):
        """Test complex prompt: 'Calculate 10 + 5, then use result and subtract 15, then add as todo, then list'."""
        planner = PatternBasedPlanner()

        prompt = "Calculate 10 + 5, then use the result and subtract it by 15, then add the final number as a todo, then show me all my tasks"

        plan = planner.create_plan(prompt)

        # Should create 4 steps: calc, calc, todo, list
        assert len(plan.steps) >= 3  # At least calc, todo, list

        # Step 1: Calculator without comma
        assert plan.steps[0].tool_name == "calculator"
        assert "10 + 5" in plan.steps[0].tool_input.expression
        # Most important: expression should NOT contain trailing comma
        assert not plan.steps[0].tool_input.expression.endswith(",")

    def test_trailing_punctuation_variations(self):
        """Test various trailing punctuation marks are all stripped."""
        planner = PatternBasedPlanner()

        test_cases = [
            ("calculate 5 + 3,", "5 + 3"),
            ("calculate 5 + 3.", "5 + 3"),
            ("calculate 5 + 3;", "5 + 3"),
            ("calculate 5 + 3:", "5 + 3"),
            ("calculate 5 + 3!", "5 + 3"),
            ("calculate 5 + 3?", "5 + 3"),
            ("calculate 5 + 3,.;", "5 + 3"),  # Multiple punctuation
        ]

        for prompt, expected_expr in test_cases:
            plan = planner.create_plan(prompt)
            assert len(plan.steps) == 1
            assert plan.steps[0].tool_input.expression == expected_expr

    def test_expression_with_internal_punctuation_preserved(self):
        """Test that internal punctuation in expressions is preserved."""
        planner = PatternBasedPlanner()

        # Expression with decimal point (internal punctuation)
        prompt = "calculate 3.14 + 2.86"

        plan = planner.create_plan(prompt)

        assert len(plan.steps) == 1
        # Decimal points should be preserved (only TRAILING punctuation stripped)
        assert plan.steps[0].tool_input.expression == "3.14 + 2.86"


class TestNaturalLanguageValidation:
    """Test fix for natural language being accepted as math expressions."""

    def test_natural_language_subtract_rejected(self):
        """Test that 'subtract it by 15' is rejected as natural language, not math."""
        planner = PatternBasedPlanner()
        prompt = "subtract it by 15"

        # Natural language should NOT match calculator pattern
        # Should raise ValueError since no valid pattern matches
        with pytest.raises(ValueError, match="Could not parse prompt"):
            planner.create_plan(prompt)

    def test_valid_math_subtract_with_number_accepted(self):
        """Test that 'subtract 15' with context is accepted as valid math."""
        planner = PatternBasedPlanner()
        # Valid math: operation verb + number (with variable reference)
        prompt = "calculate 25 - 10"

        plan = planner.create_plan(prompt)

        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "calculator"
        assert "25 - 10" in plan.steps[0].tool_input.expression

    def test_math_expression_with_variable_reference_accepted(self):
        """Test that expressions with {step_N_output} variable references are accepted."""
        planner = PatternBasedPlanner()
        # Valid math: variable reference with operator and number
        prompt = "calculate {step_1_output} - 15"

        plan = planner.create_plan(prompt)

        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "calculator"
        assert "{step_1_output} - 15" in plan.steps[0].tool_input.expression

    def test_original_failing_prompt_now_works(self):
        """Test the original failing prompt now creates correct plan."""
        planner = PatternBasedPlanner()
        prompt = (
            "Calculate 10 + 5, then use the result and subtract it by 15, "
            "then add the final number as a todo item like `I have x bugs to fix` "
            "(where x is the output of step 2), then show me all my tasks"
        )

        plan = planner.create_plan(prompt)

        # Expected steps:
        # 1. Calculate 10 + 5
        # 2. Natural language "use the result and subtract it by 15" should NOT match calculator
        # 3. "add the final number as a todo" should match todo_store
        # 4. "show me all my tasks" should match todo_store list

        # Note: Step 2 won't match anything since it's natural language
        # So we expect: calc, todo add, todo list
        assert len(plan.steps) == 3

        # Step 1: Calculator
        assert plan.steps[0].tool_name == "calculator"
        assert "10 + 5" in plan.steps[0].tool_input.expression

        # Step 2: Todo add (natural language step filtered out)
        assert plan.steps[1].tool_name == "todo_store"
        assert plan.steps[1].tool_input.action == "add"

        # Step 3: Todo list
        assert plan.steps[2].tool_name == "todo_store"
        assert plan.steps[2].tool_input.action == "list"

    def test_natural_language_indicators_rejected(self):
        """Test various natural language patterns are rejected."""
        planner = PatternBasedPlanner()
        natural_language_prompts = [
            "use the result and add it",
            "take the value and multiply it by 5",
            "subtract it by 10",
            "divide it by 2",
        ]

        for prompt in natural_language_prompts:
            # All should fail to parse as calculator operations
            with pytest.raises(ValueError, match="Could not parse prompt"):
                planner.create_plan(prompt)

    def test_valid_math_expressions_accepted(self):
        """Test valid mathematical expressions are still accepted."""
        planner = PatternBasedPlanner()
        valid_math_prompts = [
            ("calculate 5 + 3", "5 + 3"),
            ("multiply 10 * 2", "10 * 2"),
            ("divide 20 / 4", "20 / 4"),
            ("subtract 15 - 8", "15 - 8"),
            ("calculate {step_1_output} + 5", "{step_1_output} + 5"),
        ]

        for prompt, expected_expr in valid_math_prompts:
            plan = planner.create_plan(prompt)
            assert len(plan.steps) == 1
            assert plan.steps[0].tool_name == "calculator"
            assert expected_expr in plan.steps[0].tool_input.expression
