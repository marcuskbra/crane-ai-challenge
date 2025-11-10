#!/usr/bin/env python3
"""
Verify LLM Configuration.

This script checks that LiteLLM is properly configured with valid credentials.
The application NO LONGER uses pattern-based fallback by default - you MUST
configure LLM credentials properly.

Pattern-based fallback is ONLY used for transient errors (rate limits, service
outages), NOT for missing configuration.

Valid configurations:
1. Local LLM: OPENAI_BASE_URL set (e.g., http://localhost:4000)
2. Cloud Provider: Real OPENAI_API_KEY set (starts with sk-...)

Exit codes:
  0 - Configuration valid
  1 - Configuration invalid or missing
"""

import os
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from challenge.core.config import Settings


def check_local_llm_config() -> tuple[bool, str]:
    """
    Check if local LLM is configured.

    Returns:
        Tuple of (is_valid, message)

    """
    settings = Settings()

    # Local LLM requires OPENAI_BASE_URL to be set
    if settings.openai_base_url:
        # Dummy key is acceptable for local LLM
        api_key = settings.openai_api_key
        if api_key and (api_key == "sk-local-llm-dummy-key" or api_key.startswith("sk-")):
            return True, (
                f"✅ Local LLM configured:\n"
                f"   BASE_URL: {settings.openai_base_url}\n"
                f"   MODEL: {settings.openai_model}\n"
                f"   API_KEY: {'(dummy key for local LLM)' if api_key == 'sk-local-llm-dummy-key' else '(set)'}"
            )
        else:
            return False, (
                f"❌ Local LLM BASE_URL set but API_KEY invalid:\n"
                f"   BASE_URL: {settings.openai_base_url}\n"
                f"   API_KEY: {settings.openai_api_key}\n"
                f"   Expected: API_KEY starting with 'sk-' or 'sk-local-llm-dummy-key'"
            )

    return False, "Local LLM not configured (no OPENAI_BASE_URL)"


def check_cloud_provider_config() -> tuple[bool, str]:
    """
    Check if cloud provider (OpenAI/Anthropic) is configured.

    Returns:
        Tuple of (is_valid, message)

    """
    settings = Settings()

    # Cloud provider requires real API key (not dummy)
    if not settings.openai_base_url:  # No base_url means cloud provider
        if settings.openai_api_key and settings.openai_api_key.startswith("sk-"):
            # Check it's not a dummy key
            if settings.openai_api_key in ["sk-local-llm-dummy-key", "sk-no-key-pattern-fallback"]:
                return False, (
                    f"❌ Cloud provider configured but API_KEY is a dummy key:\n"
                    f"   API_KEY: {settings.openai_api_key}\n"
                    f"   You must set a REAL API key from OpenAI, Anthropic, etc."
                )
            else:
                return True, (
                    f"✅ Cloud provider configured:\n"
                    f"   MODEL: {settings.openai_model}\n"
                    f"   API_KEY: sk-...{settings.openai_api_key[-4:]} (last 4 chars)"
                )
        else:
            return False, (
                f"❌ Cloud provider not configured:\n"
                f"   API_KEY: {settings.openai_api_key or '(not set)'}\n"
                f"   Expected: Real API key starting with 'sk-'"
            )

    return False, "Cloud provider not configured (OPENAI_BASE_URL is set)"


def check_current_config() -> tuple[bool, str]:
    """
    Check current LLM configuration.

    Returns:
        Tuple of (is_valid, message)

    """
    # Try local LLM first
    local_valid, local_msg = check_local_llm_config()
    if local_valid:
        return True, local_msg

    # Try cloud provider
    cloud_valid, cloud_msg = check_cloud_provider_config()
    if cloud_valid:
        return True, cloud_msg

    # Neither configured properly
    return False, (
        f"❌ LLM not configured properly!\n\nLocal LLM check:\n  {local_msg}\n\nCloud provider check:\n  {cloud_msg}\n"
    )


def print_setup_instructions():
    """Print instructions for setting up LLM configuration."""
    print("\n" + "=" * 70)
    print("📋 SETUP INSTRUCTIONS")
    print("=" * 70)
    print()
    print("You must configure EITHER local LLM OR cloud provider:")
    print()
    print("Option 1: Local LLM (e.g., Ollama)")
    print("-" * 70)
    print("1. Start local LLM server:")
    print("   docker run -d -p 4000:4000 litellm/litellm")
    print()
    print("2. Set environment variables in .env:")
    print("   OPENAI_BASE_URL=http://localhost:4000")
    print("   OPENAI_MODEL=qwen2.5:3b")
    print("   # OPENAI_API_KEY not required for local")
    print()
    print("Option 2: Cloud Provider (OpenAI, Anthropic, etc.)")
    print("-" * 70)
    print("1. Get API key from your provider:")
    print("   • OpenAI: https://platform.openai.com/api-keys")
    print("   • Anthropic: https://console.anthropic.com/settings/keys")
    print()
    print("2. Set environment variables in .env:")
    print("   OPENAI_API_KEY=sk-your-real-api-key-here")
    print("   OPENAI_MODEL=gpt-4o-mini")
    print("   # OPENAI_BASE_URL not needed for cloud")
    print()
    print("=" * 70)
    print()


def main():
    """Run LLM configuration verification."""
    print("\n🔍 LLM Configuration Verification")
    print("=" * 70)
    print()

    # Save current env to restore later
    original_env = os.environ.copy()

    try:
        # Check current configuration
        is_valid, message = check_current_config()

        print(message)
        print()

        if is_valid:
            print("=" * 70)
            print("✅ SUCCESS: LLM is properly configured!")
            print("=" * 70)
            print()
            print("You can now:")
            print("  • Run the application: make run")
            print("  • Run tests: make test-all")
            print("  • Make API requests to create plans")
            print()
            return 0
        else:
            print("=" * 70)
            print("❌ FAILURE: LLM configuration is invalid or missing!")
            print("=" * 70)
            print_setup_instructions()
            print("After configuration, run this script again to verify:")
            print("  uv run python scripts/verify_llm_config.py")
            print()
            return 1

    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


if __name__ == "__main__":
    sys.exit(main())
