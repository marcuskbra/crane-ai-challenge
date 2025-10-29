# Crane Challenge - Assignment Requirements

## Assignment Overview
Build a minimal **AI agent runtime** for technical assessment.
**Time:** 2-4 hours (proof-of-concept, not production)
**Focus:** Code clarity, architecture decisions, problem-solving approach

## Core Requirements

### 1. Tool System
Implement a simple **Tool interface** and **two concrete tools**:

#### Required Tools:
**Calculator:**
- Safely evaluate arithmetic expressions like "(41*7)+13"
- Reject unsafe input with clear error messages

**TodoStore (in-memory):**
- CRUD operations: add, list, complete, delete
- Persist state within the session

#### Tool Interface Requirements:
Each tool must define:
- `name` (string identifier)
- `description` (what the tool does)
- `input_schema` (structured parameter definition)
- `execute(input)` returning `{ success, output, error? }`
- All tools handle errors gracefully with actionable messages

### 2. Planning Component
Create a **Planner** that converts a prompt into a structured plan.

#### Planner Options (Choose one):
**Option A: Open-Source LLM Integration** (Recommended)
- Use any open-source model or API (Ollama, LM Studio, Hugging Face)
- Implement structured output generation for tool calls
- Include fallback logic for model failures

**Option B: Rule-Based Planner**
- Pattern matching and keyword extraction
- Predefined templates for common task types
- Handle ambiguous inputs gracefully

#### Plan Format (JSON):
```json
{
  "plan_id": "unique-id",
  "steps": [
    {
      "step_number": 1,
      "tool": "TodoStore.add",
      "input": {"title": "Buy milk"},
      "reasoning": "explanation for this step"
    }
  ]
}
```

#### Validation Requirements:
- Verify all referenced tools exist
- Validate inputs against each tool's schema
- Reject invalid plans with specific error details
- Handle edge cases (empty prompt, impossible tasks, etc.)

### 3. Execution Orchestrator
The **Orchestrator** runs the plan and records progress.

#### Core Functionality:
- **Sequential Execution:** Execute plan steps in order
- **State Tracking:** Maintain complete execution history
- **Retry Logic:** Configurable retry policy (2 attempts with exponential backoff)
- **Timeout Handling:** Per-step timeout simulation
- **Idempotency:** Safe re-execution of failed runs

#### State Model (Example):
```json
{
  "run_id": "unique-id",
  "prompt": "original user prompt",
  "status": "pending|running|completed|failed",
  "plan": { /* plan object */ },
  "execution_log": [
    {
      "step_number": 1,
      "tool": "TodoStore.add",
      "input": {"title": "Buy milk"},
      "output": {"id": "todo-1"},
      "status": "completed",
      "error": null
    }
  ],
  "created_at": "2025-01-15T10:00:00Z",
  "completed_at": "2025-01-15T10:00:05Z"
}
```

### 4. REST API
Implement minimal HTTP API:

| Endpoint | Method | Description | Status Codes |
|----------|--------|-------------|--------------|
| /runs | POST | Create new run with prompt | 201, 400, 500 |
| /runs/{run_id} | GET | Get complete run state | 200, 404 |
| /health | GET | Health check | 200 |

## Testing Requirements

### Unit Tests:
- Calculator (valid/invalid inputs)
- TodoStore (add/list flow)
- Planner (invalid tool or prompt)

### Integration Test:
- One full "add + list" prompt flow

**Document skipped areas due to time limits in README**

## Language Options
**Preference:** Python, TypeScript, or Go
**Chosen:** Python 3.12+

## Constraints
- Runs locally
- Minimal external dependencies
- No cloud infrastructure deployments
- Minimal implementation to focus on concepts

## Example Scenario

### User Prompt:
"Add a todo to buy milk, then show me all my tasks"

### Generated Plan:
```json
{
  "plan_id": "plan-123",
  "steps": [
    {
      "step_number": 1,
      "tool": "TodoStore.add",
      "input": {"title": "Buy milk"},
      "reasoning": "User wants to add a new task"
    },
    {
      "step_number": 2,
      "tool": "TodoStore.list",
      "input": {},
      "reasoning": "User wants to see all tasks"
    }
  ]
}
```

### Execution:
1. TodoStore.add executes successfully → todo-1 created
2. TodoStore.list returns current todos → [{"id": "todo-1", "title": "Buy milk", "completed": false}]
3. Run completes with status "completed"

### API Interaction:
```bash
# Create run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Add a todo to buy milk, then show me all my tasks"}'

# Response
{"run_id": "abc-123", "status": "pending"}

# Check status
curl http://localhost:8000/runs/abc-123

# Response: Complete run state with execution log
```

## Deliverables

### 1. Source Code
- Well-organized project structure
- Clear separation of concerns
- Type hints/annotations where applicable

### 2. README.md including:
- System architecture overview
- Setup and installation instructions
- How to run the application
- Example API usage (curl commands)
- Testing instructions
- Design decisions and trade-offs
- Known limitations
- **Potential improvements (if you had more time)**

### 3. Tests
- Unit tests with clear naming
- At least one integration test
- Instructions to run test suite

## Evaluation Criteria

| Criterion | Weight | Focus Areas |
|-----------|--------|-------------|
| Code Quality | 40% | Clean, readable code; proper error handling; type safety; separation of concerns |
| Architecture & Design | 30% | Logical structure; clear interfaces; extensibility; appropriate design patterns |
| Functionality | 20% | Core requirements met; API works correctly; tools function properly; reliable state management |
| Documentation | 10% | Clear README; helpful code comments; thoughtful trade-off discussion; honest limitations assessment |

## Key Notes
- This is a **proof-of-concept evaluation**, not a production system
- Focus on **code clarity, architecture decisions, and problem-solving approach**
- We understand you won't build production system in 2-4 hours
- **"You open it, you own it"** - AI assistance is fine, but you own the output
