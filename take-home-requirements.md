**Technical Assignment**

**AI Engineer Position**

**OVERVIEW**

This take-home assignment evaluates your ability to design and implement a simplified agent system with proper tooling,
planning, and execution capabilities. You'll build a minimal agent runtime that demonstrates your understanding of:

* Agent architecture and workflow orchestration

* Tool integration and API design

* State management and error handling

* Software engineering best practices

**Time Allocation:** Half-day (2-4 hours; a perfect solution is not expected)

Please submit either a zip file or a private repo at least 1 day in advance of your technical interview.

Expect to discuss the assignment as if you were an engineer on the team demonstrating a POC or design concept to your
colleagues.

Note that we expect most people will use AI to assist in development—this is software engineering in 2025, after all\!
At Crane we follow a simple “you open it, you own it” rule for pull requests, which applies to any work output in
general: if your name is on it, your work is evaluated the same whether AI generated 99% or you handcrafted 100% of it.

**ASSIGNMENT**

Build a minimal **AI agent runtime** that can:

1. Accept natural language tasks from users

2. Generate structured execution plans using available tools

3. Execute plans with robust state management, error handling, and retry logic

4. Expose a clean REST API for interaction and monitoring

This is a **proof-of-concept evaluation**, not a production system. We value **code clarity, architecture decisions, and
problem-solving approach** over feature completeness.

**TECHNICAL REQUIREMENTS**

**1\. Tool System**

Implement a simple **Tool interface** and **two concrete tools**:

**Required Tools:**

* **Calculator**

    * Safely evaluate arithmetic expressions like "(41\*7)+13"

    * Reject unsafe input with clear error messages

* **TodoStore (in-memory)**

    * CRUD operations: add, list, complete, delete

    * Persist state within the session

**Tool Interface Requirements:**

Each tool must define:

* name (string identifier)

* description (what the tool does)

* input\_schema (structured parameter definition)

* execute(input) returning { success, output, error? }

All tools should handle errors gracefully and provide actionable messages.

**2\. Planning Component**

Create a **Planner** that converts a prompt into a structured plan.

**Planner Options (Choose one):**

**Option A: Open-Source LLM Integration** (Recommended)

* Use any open-source model or API (e.g., Ollama, LM Studio, Hugging Face inference API)

* Implement structured output generation for tool calls

* Include fallback logic for model failures

**Option B: Rule-Based Planner**

* Pattern matching and keyword extraction

* Predefined templates for common task types

* Should handle ambiguous inputs gracefully

**Plan Format:**

The plan should be a JSON structure containing:

* **plan\_id:** unique identifier

* **steps:** array of step objects, each containing:

    * step\_number: sequential number

    * tool: tool name (e.g., "TodoStore.add")

    * input: parameters object

    * reasoning: explanation for this step

**Validation Requirements:**

* Verify all referenced tools exist

* Validate inputs against each tool's schema

* Reject invalid plans with specific error details

* Handle edge cases (empty prompt, impossible tasks, etc.)

**3\. Execution Orchestrator**

The **Orchestrator** should run the plan and record progress.

**Core Functionality:**

* **Sequential Execution:** Execute plan steps in order

* **State Tracking:** Maintain complete execution history

* **Retry Logic:** Configurable retry policy (e.g., 2 attempts with exponential backoff)

* **Timeout Handling:** Per-step timeout simulation

* **Idempotency:** Safe re-execution of failed runs

**State Model:**

Runs should be tracked—any workflow system, agents included, need persistence, and at Crane we believe in building
observability-first systems. Here is an example data model for run tracking, but of course you can adjust or use your
own:

* **run\_id:** unique identifier

* **prompt:** original user prompt

* **status:** pending, running, completed, or failed

* **plan:** the generated plan object

* **execution\_log:** array of step executions with details

* **created\_at:** timestamp

* **completed\_at:** timestamp

Each execution log entry should include: step number, tool, input, output, status, and any error.

**4\. REST API**

Implement a minimal HTTP API with the following endpoints:

| Endpoint        | Method | Description                | Status Codes  |
|:----------------|:-------|:---------------------------|:--------------|
| /runs           | POST   | Create new run with prompt | 201, 400, 500 |
| /runs/{run\_id} | GET    | Get complete run state     | 200, 404      |
| /health         | GET    | Health check               | 200           |

**TESTING REQUIREMENTS**

Include minimal but meaningful tests:

* **Unit tests:**

    * Calculator (valid/invalid inputs)

    * TodoStore (add/list flow)

    * Planner (invalid tool or prompt)

* **Integration test:** one full “add \+ list” prompt flow

**Document skipped areas due to time limits in your README.**

**Language Options:**

This is your preference, but we strongly prefer that you use one of the following languages:

* Python

* TypeScript

* Go

**Constraints:**

* Runs locally

* Minimal external dependencies

* No cloud infrastructure deployments—there are a lot of components, but it should be a minimal implementation to focus
  on concepts

**DELIVERABLES**

**Repository Contents:**

1. **Source Code**

    * Well-organized project structure

    * Clear separation of concerns

    * Type hints/annotations where applicable

2. **README.md** including:

    * System architecture overview

    * Setup and installation instructions

    * How to run the application

    * Example API usage (curl commands)

    * Testing instructions

    * Design decisions and trade-offs

    * Known limitations

    * **Potential improvements (if you had more time)**

3. **Tests**

    * Unit tests with clear naming

    * At least one integration test

    * Instructions to run test suite

**EVALUATION CRITERIA**

We will assess your submission based on:

| Criterion             | Weight | Focus Areas                                                                                         |
|:----------------------|:-------|:----------------------------------------------------------------------------------------------------|
| Code Quality          | 40%    | Clean, readable code; proper error handling; type safety; separation of concerns                    |
| Architecture & Design | 30%    | Logical structure; clear interfaces; extensibility; appropriate design patterns                     |
| Functionality         | 20%    | Core requirements met; API works correctly; tools function properly; reliable state management      |
| Documentation         | 10%    | Clear README; helpful code comments; thoughtful trade-off discussion; honest limitations assessment |

**EXAMPLE SCENARIO**

To help you understand the expected system behavior:

**User Prompt:** "Add a todo to buy milk, then show me all my tasks"

**Generated Plan:**  
The system should generate a plan with two steps:

1. Step 1: Use TodoStore.add with input containing title "Buy milk"

2. Step 2: Use TodoStore.list with empty input to retrieve all tasks

**Execution:**

1. TodoStore.add executes successfully

2. TodoStore.list returns current todos

3. Run completes with status "completed"

**API Interaction Example:**

Create a run:  
curl \-X POST http://localhost:8000/runs \-H "Content-Type: application/json" \-d '{"prompt": "Add a todo to buy milk,
then show me all my tasks"}'

Response: {"run\_id": "abc-123", "status": "pending"}

Check status:  
curl http://localhost:8000/runs/abc-123

Response: Complete run state with execution log

**FINAL NOTES**

This assignment is designed to showcase your engineering skills in a realistic but time-boxed scenario. We understand
you won't build a production system in 2-4 hours. Focus on clear code and thoughtful design to show off your thought
process.