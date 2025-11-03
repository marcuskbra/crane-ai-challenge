#!/usr/bin/env python3
"""
Verify Local LLM Configuration.

This script tests that the OpenAI API key validator is working correctly
for different configuration scenarios (local LLM, OpenAI, pattern-based fallback).
"""

import os
import sys
import traceback
from pathlib import Path

from challenge.core.config import Settings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_local_llm_config():
    """Test configuration with local LLM (base_url set)."""
    print("Test 1: Local LLM Configuration")
    print("=" * 50)

    # Set local LLM environment
    os.environ["OPENAI_BASE_URL"] = "http://localhost:4000"
    os.environ["OPENAI_MODEL"] = "qwen2.5:3b"
    os.environ.pop("OPENAI_API_KEY", None)

    settings = Settings()

    print(f"  OPENAI_BASE_URL: {settings.openai_base_url}")
    print(f"  OPENAI_MODEL: {settings.openai_model}")
    print(f"  OPENAI_API_KEY: {settings.openai_api_key}")
    print()

    if settings.openai_api_key == "sk-local-llm-dummy-key":
        print("  ✅ PASS: Dummy API key automatically set for local LLM")
        return True
    else:
        print(f"  ❌ FAIL: Expected 'sk-local-llm-dummy-key', got '{settings.openai_api_key}'")
        return False


def test_pattern_fallback_config():
    """Test configuration with no API key or base_url (pattern-based fallback)."""
    print("\nTest 2: Pattern-Based Fallback Configuration")
    print("=" * 50)

    # Clear all LLM configuration
    os.environ.pop("OPENAI_BASE_URL", None)
    os.environ.pop("OPENAI_MODEL", None)
    os.environ.pop("OPENAI_API_KEY", None)

    settings = Settings()

    print(f"  OPENAI_BASE_URL: {settings.openai_base_url}")
    print(f"  OPENAI_MODEL: {settings.openai_model}")
    print(f"  OPENAI_API_KEY: {settings.openai_api_key}")
    print()

    if settings.openai_api_key == "sk-no-key-pattern-fallback":
        print("  ✅ PASS: Dummy API key set for pattern-based fallback")
        return True
    else:
        print(f"  ❌ FAIL: Expected 'sk-no-key-pattern-fallback', got '{settings.openai_api_key}'")
        return False


def test_explicit_api_key():
    """Test configuration with explicit API key."""
    print("\nTest 3: Explicit API Key Configuration")
    print("=" * 50)

    # Set explicit API key
    os.environ["OPENAI_API_KEY"] = "sk-test-explicit-key-123"
    os.environ.pop("OPENAI_BASE_URL", None)

    settings = Settings()

    print(f"  OPENAI_API_KEY: {settings.openai_api_key}")
    print()

    if settings.openai_api_key == "sk-test-explicit-key-123":
        print("  ✅ PASS: Explicit API key preserved")
        return True
    else:
        print(f"  ❌ FAIL: Expected 'sk-test-explicit-key-123', got '{settings.openai_api_key}'")
        return False


def main():
    """Run all configuration tests."""
    print("\n🔍 LLM Configuration Verification")
    print("=" * 50)
    print()

    results = []

    try:
        results.append(("Local LLM", test_local_llm_config()))
        results.append(("Pattern Fallback", test_pattern_fallback_config()))
        results.append(("Explicit API Key", test_explicit_api_key()))
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        traceback.print_exc()
        return 1

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    all_passed = all(result for _, result in results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    print()

    if all_passed:
        print("✅ All tests passed!")
        print()
        print("Your configuration is working correctly:")
        print("  • Local LLM (OPENAI_BASE_URL set) → Auto dummy key")
        print("  • No config → Pattern-based fallback with dummy key")
        print("  • Explicit key → Uses your provided key")
        print()
        print("You can now test with local LLM:")
        print("  export OPENAI_BASE_URL=http://localhost:4000")
        print("  export OPENAI_MODEL=qwen2.5:3b")
        print("  make llm-local-test")
        return 0
    else:
        print("❌ Some tests failed. Check the configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
