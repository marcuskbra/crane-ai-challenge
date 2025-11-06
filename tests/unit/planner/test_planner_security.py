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
