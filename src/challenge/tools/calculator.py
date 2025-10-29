"""
Calculator tool with AST-based safe evaluation.

SECURITY CRITICAL: This module uses Abstract Syntax Tree (AST) parsing
to safely evaluate mathematical expressions without using eval() or exec().
"""

import ast
import operator
from typing import Any

from challenge.tools.base import BaseTool, ToolMetadata, ToolResult


class SafeCalculator(ast.NodeVisitor):
    """
    AST-based expression evaluator with whitelist security.

    This class safely evaluates arithmetic expressions by parsing them into
    an Abstract Syntax Tree and only allowing specific mathematical operations.
    No eval() or exec() is used, preventing code injection attacks.

    Supported operations:
        - Addition (+)
        - Subtraction (-)
        - Multiplication (*)
        - Division (/)
        - Unary negation (-)
        - Numbers (integers and floats)
    """

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,  # Unary minus for negative numbers
        ast.UAdd: operator.pos,  # Unary plus for positive numbers
    }

    def visit_BinOp(self, node: ast.BinOp) -> float:
        """
        Visit binary operation node (e.g., 2 + 3, 5 * 4).

        Args:
            node: AST binary operation node

        Returns:
            Result of the binary operation

        Raises:
            ValueError: If operator is not in whitelist or division by zero

        """
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)

        if op_type not in self.OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        operator_func = self.OPERATORS[op_type]

        # Check for division by zero
        if op_type == ast.Div and right == 0:
            raise ValueError("Cannot divide by zero")

        return operator_func(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        """
        Visit unary operation node (e.g., -5).

        Args:
            node: AST unary operation node

        Returns:
            Result of the unary operation

        Raises:
            ValueError: If operator is not in whitelist

        """
        operand = self.visit(node.operand)
        op_type = type(node.op)

        if op_type not in self.OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        operator_func = self.OPERATORS[op_type]
        return operator_func(operand)

    def visit_Constant(self, node: ast.Constant) -> float:
        """
        Visit constant node (Python 3.8+).

        Args:
            node: AST constant node

        Returns:
            The numeric value

        Raises:
            ValueError: If constant is not a number

        """
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    def generic_visit(self, node: ast.AST) -> Any:
        """
        Catch-all for unsupported node types.

        Args:
            node: AST node

        Raises:
            ValueError: Always, as only whitelisted operations are allowed

        """
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """
    Calculator tool for safe arithmetic evaluation.

    This tool safely evaluates mathematical expressions using AST parsing
    instead of eval(), preventing code injection attacks.

    Example:
        >>> calculator = CalculatorTool()
        >>> result = await calculator.execute(expression="2 + 3 * 4")
        >>> print(result.output)
        14.0

    """

    @property
    def metadata(self) -> ToolMetadata:
        """Get calculator tool metadata."""
        return ToolMetadata(
            name="calculator",
            description="Safely evaluate arithmetic expressions (addition, subtraction, multiplication, division)",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate (e.g., '2 + 3 * 4')",
                    }
                },
                "required": ["expression"],
            },
        )

    async def execute(self, expression: str) -> ToolResult:
        """
        Execute calculator with arithmetic expression.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            ToolResult with calculated value or error

        """
        try:
            # Validate input
            if not expression or not expression.strip():
                return ToolResult(success=False, error="Expression cannot be empty")

            # Parse expression into AST
            tree = ast.parse(expression, mode="eval")

            # Evaluate using safe visitor
            calculator = SafeCalculator()
            result = calculator.visit(tree.body)

            return ToolResult(
                success=True,
                output=result,
                metadata={"expression": expression, "security": "AST-based evaluation"},
            )

        except SyntaxError as e:
            return ToolResult(success=False, error=f"Invalid expression syntax: {e!s}")
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"Calculation error: {e!s}")
