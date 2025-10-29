# 🚀 Crane AI Agent - Quick Reference Checklist

**One-Page Implementation Guide | Target: Tier 2 | Time Budget: 6-8 hours**

---

## ⏱️ MASTER PHASE CHECKLIST

| Phase | Duration | Core Deliverable | Status |
|-------|----------|------------------|--------|
| **0. Setup** | 15-20m | Project structure, git, venv, dependencies | ⬜ |
| **1. Tools** | 90-120m | Calculator (AST), TodoStore (CRUD), tests | ⬜ |
| **2. Tests** | 45-60m | Unit tests, >80% coverage | ⬜ |
| **3. Planner** | 30-40m | Pattern-based NL→Plan converter | ⬜ |
| **4. Orchestrator** | 60-75m | Sequential execution, retry logic | ⬜ |
| **5. API** | 30-45m | FastAPI: /execute, /runs, /health | ⬜ |
| **6. Testing** | 45-60m | Integration tests, final coverage | ⬜ |
| **7. Documentation** | 30-45m | README with trade-offs | ⬜ |
| **8. Verification** | 20-30m | Submission package, final checks | ⬜ |
| **TOTAL** | 6-8h | Complete AI Agent Runtime POC | ⬜ |

---

## 🔒 CRITICAL SECURITY CHECKPOINTS

### Calculator Tool Security (PHASE 1)
```bash
# ⛔ RED FLAG: If these patterns exist in calculator.py
grep -n "eval\|exec" src/tools/calculator.py
# ✅ MUST SEE: "No eval or exec found" or only in comments

# ✅ VERIFY: AST-based evaluation
grep -n "ast.parse\|ast.NodeVisitor" src/tools/calculator.py
```

**Security Test Command:**
```bash
python << 'EOF'
import asyncio
from src.tools.calculator import CalculatorTool

async def security_test():
    calc = CalculatorTool(tier=2)
    result = await calc.execute(expression="__import__('os').system('echo hacked')")
    assert not result.success, "🚨 SECURITY FAILURE: eval not blocked!"
    print("✅ Security test passed")

asyncio.run(security_test())
EOF
```

---

## 🎯 TIER COMPLIANCE TRACKER

### Tier 1 (Pass ~60-70%) ⬜
- [ ] Calculator: `+`, `-`, `*`, `/`
- [ ] TodoStore: `add`, `list`
- [ ] Basic planner: single operations
- [ ] Sequential execution (no retry)
- [ ] API endpoints working
- [ ] Some tests passing
- [ ] Basic README

### Tier 2 (Target ~75-85%) ⬜
- [ ] ✅ All Tier 1 features
- [ ] Calculator: decimals, negatives, `()`
- [ ] TodoStore: `add`, `list`, `get`, `complete`, `delete`
- [ ] Multi-step planner with dependencies
- [ ] Retry logic (exponential backoff)
- [ ] **>80% test coverage**
- [ ] Comprehensive error handling
- [ ] README with trade-offs documented

### Tier 3 (Stretch ~85-95%) ⬜
- [ ] ✅ All Tier 2 features
- [ ] Calculator: `sqrt`, `pow`, `sin`, `cos`, etc.
- [ ] TodoStore: `update`, `filter`, `search`, `priority`
- [ ] Advanced retry + idempotency
- [ ] **>90% test coverage**
- [ ] Comprehensive documentation
- [ ] All edge cases tested

---

## 🚨 EMERGENCY DECISION TREE

```
┌─────────────────────────────────────┐
│  How much time remaining?           │
└────────────┬────────────────────────┘
             │
    ┌────────┴──────────┐
    │                   │
  ≥4h                 <4h
    │                   │
    ▼                   ▼
Continue      ┌─────────────────┐
normally      │   <4h remaining │
              └────┬────────────┘
                   │
         ┌─────────┴──────────┐
       ≥2h                   <2h
         │                     │
         ▼                     ▼
  Tier 2 Focus        Tier 1 Triage
  - Skip Tier 3       - Skip advanced features
  - Core features     - Get one flow working
  - 80% coverage      - Basic tests only
                      - Minimal README

                   ┌────────────┐
                   │ <1h left   │
                   └─────┬──────┘
                         │
                         ▼
                   🚨 SHIP NOW 🚨
                   - Server starts?
                   - One example works?
                   - README has setup
                   → SUBMIT
```

---

## ⚡ ESSENTIAL COMMANDS

### Setup & Environment
```bash
# Time tracking
./track_time.sh start <phase>    # Start phase timer
./track_time.sh end              # End phase timer
./track_time.sh summary          # View time report

# Environment
source venv/bin/activate         # Activate venv
pip install -r requirements.txt  # Install dependencies
```

### Testing
```bash
# Run tests
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/ -v                         # All tests

# Coverage
pytest tests/ --cov=src --cov-report=html    # Generate coverage report
open htmlcov/index.html                      # View coverage (Mac)

# Quick verification
./test.sh                                # Run full test suite
```

### Development
```bash
# Server
./run.sh                         # Start server
uvicorn src.api.main:app --reload --port 8000

# Manual testing
curl http://localhost:8000/health                        # Health check
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "calculate 2 + 2"}'                    # Execute prompt
```

### Git Workflow
```bash
# Standard workflow
git add <files>
git commit -m "message"
git log --oneline

# Verification
git status              # Check uncommitted changes
git diff                # Review changes before commit
```

### Submission
```bash
./verify_submission.sh      # Run all verification checks
./create_submission.sh      # Create submission package
```

---

## 📊 PHASE TIME CHECKPOINTS

| Checkpoint | Total Elapsed | Action Required |
|------------|---------------|-----------------|
| After Phase 3 | ~3-3.5h | ✅ On track for Tier 2/3 |
| After Phase 5 | ~5-6h | ✅ On track for Tier 2 |
| After Phase 6 | ~6-7h | ⚠️ Tier 2 or emergency mode |
| After Phase 7 | ~6.5-7.5h | ⚠️ Must finish Phase 8 quickly |
| After Phase 8 | ~7-8h | ✅ Complete and submit |

**⚠️ Red Flags:**
- Phase 1 >150 min → Cut Tier 3 features
- Phase 3 >50 min → Simplify planner patterns
- Phase 4 >100 min → Skip idempotency
- Phase 6 >90 min → Accept current coverage
- Total >8h → Submit what you have

---

## 🔍 QUALITY GATES

### Phase Exit Criteria Quick Check
```bash
# Phase 1: Tools
python -c "from src.tools import get_registry; print('✅ Tools OK')"
grep -r "eval\|exec" src/tools/calculator.py && echo "❌ SECURITY RISK" || echo "✅ Security OK"

# Phase 2: Tests
pytest tests/unit/test_calculator.py tests/unit/test_todo_store.py -v
[ $? -eq 0 ] && echo "✅ Tests OK" || echo "❌ Tests Failed"

# Phase 3: Planner
python -c "from src.planning.planner import Planner; p=Planner(); plan=p.create_plan('add todo test'); print('✅ Planner OK' if plan.steps else '❌ Planner Failed')"

# Phase 5: API
curl -s http://localhost:8000/health | grep -q "healthy" && echo "✅ API OK" || echo "❌ API Failed"

# Phase 6: Coverage
pytest tests/ --cov=src --cov-report=term | grep "TOTAL" | awk '{if ($4 >= 80) print "✅ Coverage OK ("$4")"; else print "⚠️ Coverage Low ("$4")"}'

# Phase 8: Final
./verify_submission.sh
```

---

## 💡 CRITICAL SUCCESS FACTORS

### Must Have (ALL Tiers)
1. **Security**: No `eval()` or `exec()` in calculator
2. **Functionality**: POST /execute works with calculator AND todo
3. **Testing**: Tests exist and most pass (>75% passing)
4. **Documentation**: README with setup instructions
5. **Submission**: Package created and verified

### Tier 2 Differentiators
6. **Coverage**: >80% test coverage verified
7. **Retry Logic**: Exponential backoff implemented
8. **Multi-Step**: `and`/`then` operators working
9. **Error Handling**: Comprehensive error messages
10. **Trade-offs**: Design decisions documented

### Tier 3 Stretch
11. **Advanced Features**: Scientific functions, todo filter/search
12. **Coverage**: >90% with edge cases
13. **Idempotency**: Failed run retry capability
14. **Documentation**: Comprehensive with honest limitations

---

## 🎯 ONE-LINE STATUS CHECK

**Run this to see your current tier prediction:**
```bash
echo "=== TIER STATUS ===" && \
python -c "from src.tools.calculator import CalculatorTool; print('Tools: ✅')" 2>/dev/null && \
pytest tests/ -q --tb=no 2>/dev/null | tail -1 && \
pytest tests/ --cov=src --cov-report=term 2>/dev/null | grep TOTAL | awk '{print "Coverage: "$4}' && \
[ -f "README.md" ] && grep -q "Trade-offs" README.md && echo "README: ✅" || echo "README: ⚠️" && \
echo "==================" && \
echo "Tier 1: Basic features ✅" && \
echo "Tier 2: Check coverage >80% + README" && \
echo "Tier 3: Coverage >90% + advanced features"
```

---

## 📞 PANIC MODE CHECKLIST

### If you're stuck for >30 min on any task:

1. ⏸️ **STOP** - Take a 5-min break
2. 🔄 **SIMPLIFY** - What's the minimum that works?
3. ⏭️ **SKIP** - Can this be done later or cut?
4. 📖 **GUARD-RAILS** - Check full documentation for that phase
5. 💾 **COMMIT** - Commit what you have so far
6. 🆘 **TRIAGE** - Is this blocking? Use emergency procedures

**Remember:**
- ✅ Working > Perfect
- ✅ Tier 1 > Nothing
- ✅ Submitted > Incomplete
- ✅ Honest README > Over-promising

---

**END OF QUICK REFERENCE**

*For detailed implementation, see: IMPLEMENTATION_GUARDRAILS_COMPLETE.md*

*Total Time Budget: 6-8 hours | Current Target: Tier 2 (>80% coverage, all core features)*
