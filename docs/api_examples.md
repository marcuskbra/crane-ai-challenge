# API Examples

## 1. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-01-29T10:00:00.000Z",
  "version": "1.0.0"
}
```

---

## 2. Simple Calculator Example

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "calculate (10 + 5) * 2"}'
```

**Response (Immediate):**

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

**Check Status:**

```bash
curl http://localhost:8000/api/v1/runs/550e8400-e29b-41d4-a716-446655440000
```

**Response (After Completion):**

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "calculate (10 + 5) * 2",
  "status": "completed",
  "plan": {
    "steps": [
      {
        "step_number": 1,
        "tool_name": "calculator",
        "tool_input": {
          "expression": "(10 + 5) * 2"
        },
        "reasoning": "Evaluate arithmetic expression: (10 + 5) * 2"
      }
    ],
    "final_goal": "Calculate the result of (10 + 5) * 2"
  },
  "execution_log": [
    {
      "step_number": 1,
      "tool_name": "calculator",
      "tool_input": {
        "expression": "(10 + 5) * 2"
      },
      "success": true,
      "output": 30.0,
      "error": null,
      "attempts": 1,
      "duration_ms": 0.27
    }
  ],
  "result": 30.0,
  "error": null,
  "created_at": "2025-01-29T10:00:00.000Z",
  "started_at": "2025-01-29T10:00:00.100Z",
  "completed_at": "2025-01-29T10:00:00.150Z"
}
```

---

## 3. Multi-Step Todo Example

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "add a todo to buy milk and then show me all my tasks"}'
```

**Response (After Completion):**

```json
{
  "run_id": "660e9500-f39c-52e5-b827-557766551111",
  "prompt": "add a todo to buy milk and then show me all my tasks",
  "status": "completed",
  "plan": {
    "steps": [
      {
        "step_number": 1,
        "tool_name": "todo_store",
        "tool_input": {
          "action": "add",
          "text": "buy milk"
        },
        "reasoning": "Create new todo: buy milk"
      },
      {
        "step_number": 2,
        "tool_name": "todo_store",
        "tool_input": {
          "action": "list"
        },
        "reasoning": "Retrieve all todo items"
      }
    ],
    "final_goal": "Show all tasks after adding a todo for buying milk"
  },
  "execution_log": [
    {
      "step_number": 1,
      "tool_name": "todo_store",
      "tool_input": {
        "action": "add",
        "text": "buy milk"
      },
      "success": true,
      "output": {
        "todo": {
          "id": "todo-abc-123",
          "text": "buy milk",
          "completed": false,
          "created_at": "2025-01-29T10:01:00.100Z",
          "completed_at": null
        }
      },
      "error": null,
      "attempts": 1,
      "duration_ms": 0.17
    },
    {
      "step_number": 2,
      "tool_name": "todo_store",
      "tool_input": {
        "action": "list"
      },
      "success": true,
      "output": {
        "todos": [
          {
            "id": "todo-abc-123",
            "text": "buy milk",
            "completed": false,
            "created_at": "2025-01-29T10:01:00.100Z",
            "completed_at": null
          }
        ],
        "total_count": 1,
        "completed_count": 0,
        "pending_count": 1
      },
      "error": null,
      "attempts": 1,
      "duration_ms": 0.08
    }
  ],
  "result": {
    "todos": [
      {
        "id": "todo-abc-123",
        "text": "buy milk",
        "completed": false,
        "created_at": "2025-01-29T10:01:00.100Z",
        "completed_at": null
      }
    ],
    "total_count": 1,
    "completed_count": 0,
    "pending_count": 1
  },
  "error": null,
  "created_at": "2025-01-29T10:01:00.000Z",
  "started_at": "2025-01-29T10:01:08.702666Z",
  "completed_at": "2025-01-29T10:01:00.250Z"
}
```

---

## 4. Error Handling Example

**Invalid Prompt:**

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "do something impossible"}'
```

**Response:**

```json
{
  "detail": "Cannot parse prompt: no matching pattern found for 'do something impossible'"
}
```

**Status Code:** 400 Bad Request

**Non-Existent Run:**

```bash
curl http://localhost:8000/api/v1/runs/nonexistent-id
```

**Response:**

```json
{
  "detail": "Run not found: nonexistent-id",
  "details": {
    "run_id": "nonexistent-id"
  },
  "error_type": "run_not_found"
}
```

**Status Code:** 404 Not Found

---

## 5. System Metrics (Observability)

**Request:**

```bash
curl http://localhost:8000/api/v1/metrics
```

**Response:**

```json
{
  "timestamp": "2025-01-29T15:30:00.000Z",
  "runs": {
    "total": 150,
    "by_status": {
      "pending": 2,
      "running": 1,
      "completed": 140,
      "failed": 7
    },
    "success_rate": 0.952
  },
  "execution": {
    "avg_duration_seconds": 1.25,
    "total_steps_executed": 450
  },
  "tools": {
    "total_executions": 450,
    "by_tool": {
      "calculator": 280,
      "todo_store": 170
    }
  },
  "planner": {
    "total_plans_generated": 150,
    "llm_plans": 145,
    "pattern_plans": 5,
    "cached_plans": 0,
    "fallback_rate": 0.033,
    "avg_tokens_per_plan": 2935.5,
    "avg_latency_ms": 3795.3,
    "cache_hit_rate": 0.0
  }
}
```

**Use Cases:**

- Monitor system health and performance
- Track success rates and failure patterns
- Identify most-used tools for optimization
- Detect performance degradation over time
- Monitor LLM token usage and costs
- Track planner performance and latency
- Measure cache effectiveness
- Analyze fallback patterns for reliability tuning

## Related Documentation

- **Architecture**: See [System Architecture](./architecture.md) for request flow details
- **Deployment**: See [Deployment Guide](./deployment.md) for running the API
- **Known Limitations**: See [Known Limitations](./limitations.md) for API constraints
