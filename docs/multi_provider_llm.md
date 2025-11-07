# Multi-Provider LLM Support

This project uses **LiteLLM** for unified multi-provider LLM support, enabling you to use OpenAI, Anthropic Claude, or
local models like Qwen2.5/Llama through a single interface. Choose the best model for your use case - from fast local
models for development to powerful cloud models for production.

## Why Multi-Provider Support?

| Benefit                   | Impact                                                       |
|---------------------------|--------------------------------------------------------------|
| ✅ **Provider Choice**     | OpenAI, Anthropic, Ollama - use what works best              |
| ✅ **Zero API costs**      | Local models (Ollama) for development/CI testing             |
| ✅ **Cost Optimization**   | Route simple tasks to cheap models, complex to powerful ones |
| ✅ **Offline development** | Work without internet using local Ollama models              |
| ✅ **Easy switching**      | Toggle between providers via environment variables           |
| ✅ **Fallback resilience** | Automatic degradation to pattern-based on LLM failure        |

## Quick Start with Local Models (Ollama)

**1. Install Ollama:**

```bash
brew install ollama  # macOS
# Or visit https://ollama.com for other platforms
```

**2. Pull Local Model:**

```bash
ollama pull qwen2.5:3b  # Recommended for development (97% accuracy vs GPT-4o-mini)
```

**3. Configure Environment:**

```bash
# Edit .env file
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:3b
LLM_PROVIDER=ollama
LLM_API_KEY=dummy-key  # Not needed for Ollama but required by LiteLLM
```

**Note**: Ollama model names are automatically prefixed with `ollama_chat/` by the planner for LiteLLM compatibility.
Just use the standard Ollama notation (e.g., `qwen2.5:3b`) in your configuration.

**4. Run Application:**

```bash
# Ollama server starts automatically on macOS
# Application will use local model automatically
make run
```

## Quick Start with OpenAI

```bash
# Edit .env file
LLM_API_KEY=sk-your-openai-key-here
LLM_MODEL=gpt-4o-mini
LLM_PROVIDER=openai
# LLM_BASE_URL not needed for OpenAI

# Run application
make run
```

## Quick Start with Anthropic Claude

```bash
# Edit .env file
LLM_API_KEY=sk-ant-your-anthropic-key-here
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_PROVIDER=anthropic
# LLM_BASE_URL not needed for Anthropic

# Run application
make run
```

## Model Recommendations

**Cloud Models (API Key Required):**
| Model | Provider | Speed | Quality | Cost/1M tokens | Use Case |
|------------------------------|------------|--------------|-----------|----------------|-----------------------|
| **gpt-4o-mini** ⭐ | OpenAI | ⚡⚡⚡ Fast | Excellent | $0.15 | **Production default** |
| gpt-4o | OpenAI | ⚡⚡ Moderate | Best | $2.50 | Complex tasks |
| claude-3-5-sonnet-20241022 | Anthropic | ⚡⚡ Moderate | Best | $3.00 | Advanced reasoning |
| claude-3-5-haiku-20241022 | Anthropic | ⚡⚡⚡ Fast | Excellent | $0.80 | Cost-efficient |

**Local Models (Free, Offline):**
| Model | Size | Speed | Quality | Use Case |
|------------------|-------|----------------|--------------------|---------------------------|
| **qwen2.5:3b** ⭐ | 2.3GB | ⚡⚡⚡ Fast | 97% vs GPT-4o-mini | **Development/testing**   |
| llama3.2:3b | 2.0GB | ⚡⚡⚡ Fast | 95% vs GPT-4o-mini | Alternative local option |
| phi3:mini | 2.4GB | ⚡⚡⚡ Fast | 97% vs GPT-4o-mini | Microsoft research model |
| qwen2.5:1.5b | 1.2GB | ⚡⚡⚡⚡ Very fast | 91% vs GPT-4o-mini | Resource-constrained CI |

## Intelligent Model Routing

The planner includes **automatic complexity-based model routing** to optimize cost and performance:

```python
from challenge.services.planning.routing import ModelRouter

router = ModelRouter(
    simple_model="gpt-4o-mini",  # For simple calculations
    moderate_model="gpt-4o-mini",  # For standard multi-step tasks
    complex_model="gpt-4o",  # For complex reasoning
    fallback_models=["claude-3-5-haiku-20241022", "qwen2.5:3b"]
)

# Automatically selects appropriate model based on prompt complexity
model = router.select_model("Calculate 2+2")  # → gpt-4o-mini (simple)
model = router.select_model("Analyze multi-step comprehensive workflow")  # → gpt-4o (complex)
```

**Benefits:**

- **Cost Optimization**: Route simple tasks to cheap models ($0.15/1M tokens)
- **Quality Assurance**: Use powerful models only when needed
- **Automatic Fallback**: Graceful degradation on provider failures

## Docker Testing

Full containerized setup with Ollama + LiteLLM:

```bash
# Start all services (Ollama + LiteLLM + App)
docker-compose -f docker-compose.litellm.yml up -d

# Run tests
docker-compose -f docker-compose.litellm.yml exec app pytest tests/ -v

# View logs
docker-compose -f docker-compose.litellm.yml logs -f litellm
```

## Performance Comparison

| Metric           | GPT-4o-mini | Qwen2.5-3B (Local) |
|------------------|-------------|--------------------|
| Planning time    | 1-2s        | 2-3s               |
| Cost per 1K runs | $0.30       | $0                 |
| Accuracy         | 99%         | 97%                |
| Offline capable  | ❌           | ✅                  |

## Complete Guide

For detailed setup instructions, troubleshooting, and advanced configuration:

**📖 [Local LLM Testing Guide](../claudedocs/local_llm_testing_guide.md)**

## Related Documentation

- **Architecture**: See [System Architecture](./architecture.md) for planning layer details
- **Design Decisions**: See [Design Decisions](./design_decisions.md) for LLM integration rationale
- **Deployment**: See [Deployment Guide](./deployment.md) for production configuration
