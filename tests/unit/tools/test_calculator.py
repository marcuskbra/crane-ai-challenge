"""
Unit tests for Calculator tool.

Tests cover:
- Basic arithmetic operations
- Expression parsing and evaluation
- Error handling (syntax, division by zero, invalid operations)
- Security (injection prevention)
"""

import pytest

from challenge.tools.calculator import CalculatorTool


class TestCalculatorTool:
    """Test suite for CalculatorTool."""

    @pytest.fixture
    def calculator(self):
        """Provide calculator tool instance."""
        return CalculatorTool()

    def test_metadata(self, calculator):
        """Test calculator metadata."""
        metadata = calculator.metadata
        assert metadata.name == "calculator"
        assert "arithmetic" in metadata.description.lower()
        assert "expression" in metadata.input_schema["properties"]

    @pytest.mark.asyncio
    async def test_addition(self, calculator):
        """Test simple addition."""
        result = await calculator.execute(expression="2 + 3")
        assert result.success is True
        assert result.output == 5.0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_subtraction(self, calculator):
        """Test simple subtraction."""
        result = await calculator.execute(expression="10 - 4")
        assert result.success is True
        assert result.output == 6.0

    @pytest.mark.asyncio
    async def test_multiplication(self, calculator):
        """Test simple multiplication."""
        result = await calculator.execute(expression="6 * 7")
        assert result.success is True
        assert result.output == 42.0

    @pytest.mark.asyncio
    async def test_division(self, calculator):
        """Test simple division."""
        result = await calculator.execute(expression="15 / 3")
        assert result.success is True
        assert result.output == 5.0

    @pytest.mark.asyncio
    async def test_negative_numbers(self, calculator):
        """Test negative numbers."""
        result = await calculator.execute(expression="-5 + 3")
        assert result.success is True
        assert result.output == -2.0

    @pytest.mark.asyncio
    async def test_complex_expression(self, calculator):
        """Test operator precedence (PEMDAS)."""
        result = await calculator.execute(expression="2 + 3 * 4")
        assert result.success is True
        assert result.output == 14.0  # Not 20

    @pytest.mark.asyncio
    async def test_complex_expression_with_parentheses(self, calculator):
        """Test expression with parentheses."""
        result = await calculator.execute(expression="(2 + 3) * 4")
        assert result.success is True
        assert result.output == 20.0

    @pytest.mark.asyncio
    async def test_float_numbers(self, calculator):
        """Test floating point operations."""
        result = await calculator.execute(expression="2.5 + 3.7")
        assert result.success is True
        assert abs(result.output - 6.2) < 0.001  # Float comparison

    @pytest.mark.asyncio
    async def test_multiple_operations(self, calculator):
        """Test expression with multiple operations."""
        result = await calculator.execute(expression="10 + 5 - 3 * 2")
        assert result.success is True
        assert result.output == 9.0  # 10 + 5 - 6

    @pytest.mark.asyncio
    async def test_division_by_zero(self, calculator):
        """Test division by zero error handling."""
        result = await calculator.execute(expression="5 / 0")
        assert result.success is False
        assert "zero" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_expression(self, calculator):
        """Test empty expression error handling."""
        result = await calculator.execute(expression="")
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_expression(self, calculator):
        """Test whitespace-only expression."""
        result = await calculator.execute(expression="   ")
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_syntax(self, calculator):
        """Test invalid syntax error handling."""
        result = await calculator.execute(expression="2 + * 3")
        assert result.success is False
        assert "syntax" in result.error.lower()

    @pytest.mark.asyncio
    async def test_power_operator(self, calculator):
        """Test power/exponentiation operator."""
        result = await calculator.execute(expression="2 ** 3")
        assert result.success is True
        assert result.output == 8.0

    @pytest.mark.asyncio
    async def test_power_operator_with_negative_exponent(self, calculator):
        """Test power operator with negative exponent."""
        result = await calculator.execute(expression="2 ** -1")
        assert result.success is True
        assert result.output == 0.5

    @pytest.mark.asyncio
    async def test_power_operator_precedence(self, calculator):
        """Test power operator precedence (higher than multiplication)."""
        result = await calculator.execute(expression="2 + 3 ** 2")
        assert result.success is True
        assert result.output == 11.0  # 2 + 9, not (2+3)**2=25

    @pytest.mark.asyncio
    async def test_power_operator_with_parentheses(self, calculator):
        """Test power operator with parentheses."""
        result = await calculator.execute(expression="(2 + 3) ** 2")
        assert result.success is True
        assert result.output == 25.0

    @pytest.mark.asyncio
    async def test_unsupported_operator_modulo(self, calculator):
        """Test rejection of modulo operator."""
        result = await calculator.execute(expression="10 % 3")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_code_injection_attempt_import(self, calculator):
        """Test prevention of import injection."""
        result = await calculator.execute(expression="__import__('os')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_code_injection_attempt_function_call(self, calculator):
        """Test prevention of function call injection."""
        result = await calculator.execute(expression="print('hello')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_code_injection_attempt_variable(self, calculator):
        """Test prevention of variable access."""
        result = await calculator.execute(expression="x = 5")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_code_injection_attempt_eval(self, calculator):
        """Test prevention of eval injection."""
        result = await calculator.execute(expression="eval('2 + 2')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_metadata_includes_security_info(self, calculator):
        """Test that result metadata includes security information."""
        result = await calculator.execute(expression="2 + 2")
        assert result.success is True
        assert result.metadata is not None
        assert "security" in result.metadata
        assert "AST" in result.metadata["security"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("1 + 1", 2.0),
            ("10 - 5", 5.0),
            ("3 * 4", 12.0),
            ("20 / 4", 5.0),
            ("-10 + 5", -5.0),
            ("2 + 3 * 4", 14.0),
            ("(2 + 3) * 4", 20.0),
        ],
    )
    async def test_parameterized_expressions(self, calculator, expression, expected):
        """Test multiple expressions with parameterization."""
        result = await calculator.execute(expression=expression)
        assert result.success is True
        assert abs(result.output - expected) < 0.001
