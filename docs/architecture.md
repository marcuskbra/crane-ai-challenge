# System Architecture

This agent runtime uses a **4-layer architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER / CLIENT                            │
│                    (Natural Language Input)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP Request
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      🌐 REST API LAYER                           │
│  FastAPI Routes: POST /runs, GET /runs/{id}, GET /health        │
│  • Request validation (Pydantic)                                │
│  • HTTP error mapping (400/404/500)                             │
│  • Async request handling                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ create_run(prompt)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🧠 PLANNING LAYER                            │
│  Hybrid Planner: LLM + Pattern-Based Fallback                   │
│  • Multi-Provider LLM (OpenAI, Anthropic, Ollama) via LiteLLM   │
│  • Intelligent model routing based on prompt complexity         │
│  • Automatic fallback to pattern-based on failures              │
│  • Multi-step decomposition with structured outputs             │
│  • Tool validation & cost tracking                              │
└────────────────────────────┬────────────────────────────────────┘
                             │ Plan(steps)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ⚙️  ORCHESTRATION LAYER                       │
│  Sequential Executor with State Management                      │
│  • Step-by-step execution with timeout protection               │
│  • Exponential backoff retry (3 attempts)                       │
│  • Configurable per-step timeout (default: 30s)                 │
│  • Complete execution history                                   │
│  • Error tracking and recovery                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ execute(tool, input)
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                     🔧 TOOL LAYER                              │
│  ┌──────────────────┐          ┌──────────────────┐            │
│  │  Calculator      │          │  TodoStore       │            │
│  │  • AST-based ✅  │          │  • In-memory     │            │
│  │  • No eval/exec  │          │  • CRUD ops      │            │
│  │  • +, -, *, /    │          │  • UUID IDs      │            │
│  │  • ( ) grouping  │          │  • Timestamps    │            │
│  └──────────────────┘          └──────────────────┘            │
└────────────────────────────────────────────────────────────────┘
```

## Request Flow Example

```
1. User sends: "Add a todo to buy milk, then show me all my tasks"
   ↓
2. API validates and creates run → returns run_id
   ↓
3. Planner analyzes prompt:
   - Detects: "add todo" pattern → todo_store.add tool
   - Detects: "show all" pattern → todo_store.list tool
   - Generates: 2-step plan
   ↓
4. Orchestrator executes sequentially:
   Step 1: todo_store.add(text="buy milk") → {id: "abc-123"}
   Step 2: todo_store.list() → [{id: "abc-123", text: "buy milk", completed: false}]
   ↓
5. User polls: GET /runs/{run_id} → Complete execution log
```

## Related Documentation

- **Design Decisions**: See [Design Decisions](./design_decisions.md) for rationale behind architectural choices
- **API Reference**: See [API Examples](./api_examples.md) for detailed API usage
- **LLM Integration**: See [Multi-Provider LLM Setup](./multi_provider_llm.md) for planning layer details
- **Deployment**: See [Deployment Guide](./deployment.md) for running the system
