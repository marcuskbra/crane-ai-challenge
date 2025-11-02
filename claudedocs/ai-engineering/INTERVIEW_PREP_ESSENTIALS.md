# Interview Prep Essentials - Quick Reference Card

**Print this and keep it handy during your interview**

---

## Your 2-Minute Pitch

> "I built a production-quality AI agent runtime that exceeds the assignment scope. The key innovations are: a hybrid LLM planner with GPT-4o-mini that automatically falls back to pattern matching for reliability, a Protocol-based architecture following SOLID principles, and an AST-based calculator with comprehensive security testing. I achieved 84% test coverage with 103 tests, including 5 security injection tests. The system demonstrates production patterns like structured outputs, cost tracking, and exponential backoff retry. Time investment was about 6 hours, focused on AI engineering quality over feature quantity."

---

## Your Differentiators (What Makes You Stand Out)

### 1. Hybrid LLM System ⭐⭐⭐⭐⭐
**Most candidates do EITHER LLM OR patterns. You did BOTH with automatic fallback.**

**What to say**:
> "I implemented both planner approaches to demonstrate different trade-offs, then built a production system with automatic LLM → pattern fallback. This provides intelligent planning for complex requests while guaranteeing reliability even if the OpenAI API fails. The LLM tracks tokens for cost monitoring, and I documented the cost/performance trade-offs explicitly."

### 2. Protocol Pattern ⭐⭐⭐⭐
**Shows advanced Python and SOLID principles most won't demonstrate.**

**What to say**:
> "I used Python's Protocol for the planner interface instead of abstract base classes. This follows the Dependency Inversion Principle - the Orchestrator depends on an abstraction, not concrete implementations. Anyone can plug in a custom planner by just implementing create_plan(), no inheritance required. This is structural subtyping, a modern Python best practice."

### 3. Security-First ⭐⭐⭐⭐⭐
**5 injection tests show you think about attack surfaces.**

**What to say**:
> "The calculator uses AST parsing instead of eval() because AI systems are attack surfaces. I wrote 5 specific injection tests covering imports, eval, builtins, function calls, and variable manipulation. This security-first approach is critical for agent systems that execute arbitrary user input."

### 4. Production Patterns ⭐⭐⭐⭐
**Structured outputs, cost tracking, fallback chains - real-world patterns.**

**What to say**:
> "I used OpenAI's structured output feature with JSON schema enforcement for reliable plan generation. The system tracks LLM token usage for cost monitoring, implements retry with exponential backoff, and has comprehensive error handling. These aren't academic patterns - they're production LLM engineering."

---

## Demo Script (Know This Cold)

```bash
# 1. Start server
make run

# 2. Health check
curl http://localhost:8000/api/v1/health

# 3. Simple calculator
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "calculate (10 + 5) * 2"}'

# Get run_id from response, then:
curl http://localhost:8000/api/v1/runs/{run_id}

# 4. Multi-step workflow
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "add todo to buy milk and then show all my tasks"}'

# 5. Show execution log with LLM reasoning
curl http://localhost:8000/api/v1/runs/{run_id} | jq '.plan.steps[].reasoning'
```

---

## Top 5 Questions You'll Be Asked

### Q1: "Walk me through the architecture"
**Answer** (use README diagram):
> "4-layer architecture: API layer with FastAPI routes, Planning layer with hybrid LLM/pattern system, Orchestration layer with retry logic, and Tool layer with Calculator and TodoStore. The API receives prompts, planner generates structured plans, orchestrator executes steps with exponential backoff retry, and tools perform actual operations. State is currently in-memory for POC, but I documented the production approach: Redis for active runs, PostgreSQL for historical data."

### Q2: "Why did you choose GPT-4o-mini over GPT-4?"
**Answer**:
> "Three reasons: Cost (33x cheaper at $0.15 per 1M tokens), Latency (~200ms vs 500ms+), and Sufficient Capability. Planning tasks don't need GPT-4's advanced reasoning. With structured outputs enforcing JSON schema, even smaller models produce reliable plans. I implemented token tracking so we can measure actual costs and optimize if needed."

### Q3: "What security considerations did you address?"
**Answer**:
> "Three main attack surfaces: Code injection via Calculator (mitigated with AST parsing and 5 injection tests), Prompt injection for LLM (mitigated with structured outputs and tool whitelisting), and Resource exhaustion (would add rate limiting and budget caps in production). The AST approach is critical - it prevents users from executing arbitrary Python code like __import__('os').system('rm -rf /')."

### Q4: "How would you scale this for production?"
**Answer**:
> "Three bottlenecks: State management (move to Redis+PostgreSQL), LLM API limits (add request queueing and caching), and Sequential execution (implement DAG-based parallel execution). I'd also add horizontal pod autoscaling in Kubernetes based on queue depth and P95 latency. The current architecture supports these changes - Protocol pattern makes swapping implementations easy."

### Q5: "What would you add if you had more time?"
**Answer**:
> "Priority 1: Observability - structured logging with correlation IDs, Prometheus metrics, OpenTelemetry tracing. Priority 2: State persistence - Redis for active runs, PostgreSQL for historical. Priority 3: Agent evaluation suite - gold standard test cases with accuracy metrics. The timeout implementation is a quick add I'd do immediately - just wrapping the retry logic with asyncio.wait_for()."

---

## What You're Missing (Acknowledge Proactively)

### Timeout Implementation
**Say**: "I didn't implement per-step timeouts, which the requirements mentioned. I have the design - use asyncio.wait_for() around the retry logic - but didn't add it due to time constraints. It's a 30-minute addition."

### No Persistent Storage
**Say**: "I used in-memory state for POC simplicity. For production, I documented the exact approach: Redis for active runs with TTL-based cleanup, PostgreSQL for historical data. This is a deliberate POC vs production trade-off."

### Limited Observability
**Say**: "Logging is basic stdout. For production, I'd implement structured logging with correlation IDs, Prometheus metrics, and OpenTelemetry tracing. I prioritized the LLM integration as better use of limited time."

---

## Key Metrics to Know

- **Test Coverage**: 84% (103 tests total)
- **Test Breakdown**: 73 unit, 8 integration, 22 health/smoke
- **Security Tests**: 5 injection tests for Calculator
- **Time Investment**: ~6 hours
- **Target Tier**: Tier 2 (75-85%)
- **Estimated Score**: 80-85% (solid Tier 2)

---

## File Walkthrough Order

If asked to walk through code:

1. **API Layer**: `src/challenge/api/routes/runs.py` - Entry point
2. **Orchestrator**: `src/challenge/orchestrator/orchestrator.py` - Core logic
3. **LLM Planner**: `src/challenge/planner/llm_planner.py` - AI integration
4. **Protocol**: `src/challenge/planner/protocol.py` - Architecture
5. **Calculator**: `src/challenge/tools/calculator.py` - Security
6. **Tests**: `tests/unit/tools/test_calculator.py` - Quality

---

## Technical Depth Questions - Prepare These

### LLM Integration
- **Structured Outputs**: How they work, why they're better than JSON mode
- **Fallback Chain**: When/why it triggers, success rate
- **Cost Tracking**: Token counting, cost estimation formula
- **Temperature**: Why 0.1 for planning vs higher for generation

### Protocol Pattern
- **Structural vs Nominal Subtyping**: Protocol vs ABC differences
- **SOLID Principles**: Which ones and how
- **Duck Typing**: Type safety without runtime overhead
- **Sync/Async Handling**: How orchestrator handles both

### Security
- **AST Parsing**: How it prevents injection
- **Operator Whitelisting**: What's allowed and why
- **Attack Vectors**: The 5 injection tests you wrote
- **Production Security**: Additional measures needed

### Testing
- **Coverage Strategy**: Why 84% is good but not 100%
- **Test Pyramid**: Unit vs integration balance
- **Security Testing**: Why injection tests matter
- **Missing Tests**: What you'd add with more time

---

## DO's and DON'Ts

### ✅ DO:
- Lead with differentiators (LLM integration, Protocol pattern)
- Acknowledge gaps proactively before they ask
- Explain trade-offs, not just features
- Show production thinking ("POC I chose X, production I'd use Y")
- Use metrics (84% coverage, 103 tests, 5 security tests)

### ❌ DON'T:
- Apologize for missing Tier 3 features
- Oversell capabilities ("production-ready" → "production-quality POC")
- Ignore gaps hoping they won't notice
- Focus only on what vs why
- Use marketing language ("blazingly fast", "100% secure")

---

## Your Competitive Edge

**What hiring managers are looking for**:
1. Can this person write production-quality code? → YES (84% coverage, type safety, error handling)
2. Do they understand AI engineering? → YES (structured outputs, fallback chains, cost tracking)
3. Can they make good trade-offs? → YES (documented in README)
4. Will they fit our engineering culture? → YES (SOLID principles, Protocol pattern, security-first)
5. Can they ship quality under time pressure? → YES (6 hours, Tier 2 target achieved)

---

## Last-Minute Checklist

### Before Interview:
- [ ] Run `make test-all` - verify 103 tests pass
- [ ] Run `make coverage` - verify 84%
- [ ] Test demo flow manually
- [ ] Have terminal ready with commands
- [ ] Print this sheet

### During Interview:
- [ ] Lead with 2-minute pitch
- [ ] Use demo to show capabilities
- [ ] Acknowledge gaps proactively
- [ ] Explain trade-offs
- [ ] Ask intelligent questions about their production setup

### After Demo:
- [ ] Ask: "What production challenges do you face with agent systems?"
- [ ] Ask: "How do you evaluate LLM performance in production?"
- [ ] Ask: "What's your observability stack for AI systems?"

---

## Emergency Answers

**If you blank on a technical question**:
> "I haven't implemented that specifically, but here's how I'd approach it based on production LLM patterns I've studied..."

**If they ask about something you don't know**:
> "I'm not familiar with that technology, but I'd love to learn more. How are you using it in your stack?"

**If they ask why you spent 6 hours vs 2-4**:
> "I focused on quality over speed. The assignment said the perfect solution isn't expected, but I wanted to demonstrate production-ready AI engineering skills. The LLM integration and Protocol pattern weren't required, but they showcase how I think about production systems."

---

## Confidence Boosters

**You built**:
- A hybrid AI system with automatic fallback (production pattern)
- Type-safe architecture following SOLID principles (senior engineer)
- Security-first calculator with comprehensive tests (security-aware)
- 84% test coverage in 6 hours (quality-focused)
- Honest documentation of trade-offs (mature judgment)

**You demonstrated**:
- Real-world LLM engineering patterns
- Advanced Python knowledge (Protocol, async, type hints)
- Production systems thinking (retry, fallback, cost tracking)
- Security awareness (AST, injection tests)
- Professional documentation (README, trade-offs, limitations)

**You're ready.** This is a strong submission. Trust your work and explain your thinking.

---

**Good luck! 🚀**
