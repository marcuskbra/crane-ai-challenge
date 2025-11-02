# LangChain/LangGraph Architecture Implementation Plan

## Overview

This document outlines the architecture for upgrading the Crane AI Agent Runtime from a pattern-based system to a
production-grade LangChain/LangGraph implementation with Claude Sonnet 4.5 and Voyage AI embeddings.

## Current System Analysis

### Existing Components

1. **Pattern-Based Planner** (`src/challenge/planner/planner.py`)
    - Regex-based natural language understanding
    - ~15 predefined patterns
    - Deterministic but limited flexibility

2. **Sequential Orchestrator** (`src/challenge/orchestrator/orchestrator.py`)
    - Step-by-step execution
    - Exponential backoff retry (1s, 2s, 4s)
    - In-memory state management

3. **Tool System** (`src/challenge/tools/`)
    - `BaseTool` abstract interface
    - Calculator tool (AST-based, security-first)
    - TodoStore tool (in-memory CRUD)

4. **REST API** (`src/challenge/api/`)
    - FastAPI async endpoints
    - POST /runs, GET /runs/{id}
    - Pydantic validation

### Limitations to Address

- No LLM reasoning capabilities
- Limited to predefined patterns
- No conversation history/memory
- No streaming responses
- No observability/tracing
- No multi-agent orchestration

---

## Target Architecture

### 1. LangGraph State Machine

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, Annotated


class AgentState(TypedDict):
    """Enhanced state for agent execution."""
    messages: Annotated[list, "conversation history"]
    plan: Annotated[dict | None, "execution plan"]
    tool_results: Annotated[list, "tool execution results"]
    session_id: Annotated[str, "session identifier"]
    context: Annotated[dict, "additional context"]
```

### 2. LLM Integration

**Primary LLM**: Claude Sonnet 4.5 (`claude-sonnet-4-5`)

- Best reasoning capabilities
- Excellent tool use
- 200K context window

**Embeddings**: Voyage AI (`voyage-3-large`)

- Officially recommended by Anthropic
- Superior performance with Claude
- Specialized models available

### 3. Tool System Migration

Transform existing tools to LangChain `StructuredTool`:

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")


calculator_tool = StructuredTool.from_function(
    func=calculator.execute,
    name="calculator",
    description="Evaluate mathematical expressions safely",
    args_schema=CalculatorInput,
    coroutine=calculator.execute
)
```

### 4. Agent Types

#### A. ReAct Agent (Primary)

- Multi-step reasoning with tool usage
- Uses `create_react_agent(llm, tools, state_modifier)`
- Best for general-purpose tasks

#### B. Plan-and-Execute (Advanced)

- Separate planning and execution nodes
- Better for complex multi-step workflows
- Tracks progress through state

### 5. Memory System

**Short-term Memory**: `ConversationTokenBufferMemory`

- Token-based windowing
- Maintains recent context

**Summarization**: `ConversationSummaryMemory`

- Compress long conversation histories
- Extract key points

**Vector Memory** (Future):

- Semantic search across past conversations
- Pinecone + Voyage embeddings

### 6. Streaming Architecture

```python
async def stream_agent_response(request: AgentRequest):
    async for event in agent.astream_events(
            {"messages": [HumanMessage(content=request.prompt)]},
            version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            yield event["data"]["chunk"].content
```

### 7. Observability with LangSmith

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "crane-agent-runtime"
```

---

## Implementation Phases

### Phase 1: Dependencies & Setup (30 min)

- [ ] Add LangChain dependencies to pyproject.toml
- [ ] Configure environment variables
- [ ] Setup LangSmith tracing
- [ ] Add Voyage AI embeddings

### Phase 2: Tool Migration (45 min)

- [ ] Create LangChain tool wrappers
- [ ] Implement tool schema validation
- [ ] Test tool execution with LLM
- [ ] Maintain backward compatibility

### Phase 3: LLM-Based Planner (60 min)

- [ ] Implement Claude Sonnet 4.5 integration
- [ ] Create structured output parser
- [ ] Implement fallback to pattern-based planner
- [ ] Add prompt engineering for tool selection

### Phase 4: LangGraph State Machine (60 min)

- [ ] Define AgentState schema
- [ ] Implement state nodes (plan, execute, validate)
- [ ] Add conditional edges for routing
- [ ] Implement checkpointing for reliability

### Phase 5: Memory Integration (30 min)

- [ ] Add conversation buffer memory
- [ ] Implement memory persistence
- [ ] Add session management

### Phase 6: Streaming Support (45 min)

- [ ] Implement SSE streaming endpoint
- [ ] Add async event handling
- [ ] Update FastAPI routes

### Phase 7: Testing & Validation (45 min)

- [ ] Unit tests for LangChain tools
- [ ] Integration tests for agent flow
- [ ] Evaluation with LangSmith
- [ ] Performance benchmarking

### Phase 8: Documentation (30 min)

- [ ] Update README with LangChain architecture
- [ ] Add API examples with streaming
- [ ] Document configuration options

**Total Estimated Time**: 5-6 hours

---

## Technical Stack Additions

```toml
[project.dependencies]
# LangChain Core
langchain = ">=0.3.0"
langchain-core = ">=0.3.0"
langgraph = ">=0.2.0"

# LLM Providers
langchain-anthropic = ">=0.2.0"  # Claude Sonnet 4.5

# Embeddings
langchain-voyageai = ">=0.1.0"  # Voyage AI

# Observability
langsmith = ">=0.1.0"

# Optional: Vector Stores (Future)
langchain-pinecone = ">=0.1.0"
pinecone-client = ">=3.0.0"

# Async support
aiostream = ">=0.5.0"
```

---

## API Changes

### Current Endpoint

```python
POST / api / v1 / runs
{
    "prompt": "calculate 2 + 3"
}

GET / api / v1 / runs / {run_id}
```

### New Endpoints

#### 1. Standard Execution (Async)

```python
POST / api / v1 / agent / invoke
{
    "prompt": "calculate 2 + 3",
    "session_id": "user-123",  # Optional
    "stream": false
}

Response:
{
    "run_id": "abc-123",
    "result": 5.0,
    "plan": {...},
    "execution_log": [...]
}
```

#### 2. Streaming Execution

```python
POST / api / v1 / agent / stream
{
    "prompt": "calculate 2 + 3 and explain the steps",
    "session_id": "user-123"
}

Response: Server - Sent
Events(SSE)
data: {"type": "thinking", "content": "I'll calculate..."}
data: {"type": "tool_call", "tool": "calculator", "input": {...}}
data: {"type": "tool_result", "output": 5.0}
data: {"type": "final_answer", "content": "The result is 5.0"}
```

#### 3. Session Management

```python
GET / api / v1 / agent / sessions / {session_id} / history
DELETE / api / v1 / agent / sessions / {session_id}
```

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5

# Embeddings
VOYAGE_API_KEY=pa-...
VOYAGE_MODEL=voyage-3-large

# LangSmith (Observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=crane-agent-runtime

# Agent Configuration
MAX_RETRIES=3
RETRY_BACKOFF=exponential
AGENT_TYPE=react  # or plan-and-execute
MEMORY_TYPE=buffer  # or summary, vector

# Performance
ENABLE_STREAMING=true
ENABLE_CACHING=true
MAX_TOKENS=4096
TEMPERATURE=0.0
```

---

## Production Considerations

### 1. Cost Optimization

- **Caching**: Redis for response caching with TTL
- **Token Management**: Track and limit token usage
- **Prompt Optimization**: Minimize prompt size
- **Model Selection**: Use cheaper models for simple tasks

### 2. Performance

- **Connection Pooling**: Reuse HTTP connections
- **Async Everything**: Full async/await support
- **Parallel Tool Calls**: Execute independent tools concurrently
- **Streaming**: Reduce perceived latency

### 3. Reliability

- **Retry Logic**: Exponential backoff with jitter
- **Circuit Breaker**: Prevent cascading failures
- **Fallbacks**: Pattern-based planner as backup
- **Health Checks**: Validate LLM availability

### 4. Monitoring

- **LangSmith**: Trace all executions
- **Prometheus**: Track metrics (latency, errors, tokens)
- **Structured Logging**: `structlog` for consistency
- **Alerts**: Notify on failures

### 5. Security

- **API Keys**: Environment variables only
- **Rate Limiting**: Protect against abuse
- **Input Validation**: Pydantic schemas
- **Sandboxing**: AST-based calculator (maintained)

---

## Migration Strategy

### Phase 1: Parallel Implementation

- Keep existing pattern-based system
- Add LangChain implementation alongside
- Use feature flags to toggle between systems

### Phase 2: Gradual Rollout

- Start with 10% of traffic
- Monitor metrics and error rates
- Increase gradually to 100%

### Phase 3: Deprecation

- Remove pattern-based planner
- Clean up old code
- Update documentation

---

## Success Metrics

### Functional

- ✅ All existing tests passing
- ✅ New LangChain integration tests
- ✅ 80%+ test coverage maintained
- ✅ Backward compatible API

### Performance

- ✅ <2s response time (p95)
- ✅ <5s for complex multi-step tasks
- ✅ <100ms streaming first token
- ✅ 99.9% uptime

### Quality

- ✅ Better plan quality vs pattern-based
- ✅ Handle novel/ambiguous prompts
- ✅ Comprehensive observability
- ✅ Production-grade error handling

---

## Next Steps

1. Review and approve architecture
2. Set up development environment
3. Begin Phase 1: Dependencies & Setup
4. Follow implementation phases sequentially
5. Test thoroughly at each phase
6. Deploy with monitoring

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Claude Sonnet 4.5 Guide](https://docs.anthropic.com/en/docs/models-overview)
- [Voyage AI Documentation](https://docs.voyageai.com/)
- [LangSmith Guide](https://docs.smith.langchain.com/)
