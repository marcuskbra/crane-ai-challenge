"""
Placeholder test to verify test infrastructure works.

This file can be deleted once real tests are added.
"""

import pytest

import challenge


def test_placeholder():
    """Verify pytest is working correctly."""
    assert True


def test_imports():
    """Verify package can be imported."""
    assert challenge.__version__ == "0.1.0"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        ("test", "test"),
        ([1, 2, 3], [1, 2, 3]),
    ],
)
def test_parameterized_example(value, expected):
    """Example of parameterized test."""
    assert value == expected
