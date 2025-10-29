# AI Agent Runtime - Assignment Implementation Plan

## Current Status: Ready for Implementation ✅

### Foundation Completed
- ✅ Pydantic Settings with validation
- ✅ FastAPI dependency injection
- ✅ CORS security fixed
- ✅ Health check endpoints
- ✅ Testing infrastructure
- ✅ All validation passing (tests, lint, format, types)

## Implementation Order

### Phase 1: Data Models (2-3 hours)
**Location**: `src/challenge/models/`

1. **Tool Models** (`models/tool.py`)
   ```python
   class ToolDefinition(BaseModel):
       name: str
       description: str
       parameters: dict[str, Any]
   
   class ToolResult(BaseModel):
       success: bool
       output: Any
       error: str | None = None
   ```

2. **Run Models** (`models/run.py`)
   ```python
   class RunStatus(str, Enum):
       PENDING = "pending"
       RUNNING = "running"
       COMPLETED = "completed"
       FAILED = "failed"
   
   class RunCreate(BaseModel):
       prompt: str
       max_steps: int = Field(default=10, ge=1, le=100)
   
   class RunResponse(BaseModel):
       run_id: str
       status: RunStatus
       steps: list[dict[str, Any]]
       result: Any | None = None
       error: str | None = None
       created_at: datetime
       updated_at: datetime
   ```

3. **Plan Models** (`models/plan.py`)
   ```python
   class PlanStep(BaseModel):
       step_number: int
       tool_name: str
       tool_input: dict[str, Any]
       reasoning: str
   
   class Plan(BaseModel):
       steps: list[PlanStep]
       final_goal: str
   ```

### Phase 2: Tool System (3-4 hours)
**Location**: `src/challenge/tools/`

1. **Base Tool** (`tools/base.py`)
   ```python
   class BaseTool(ABC):
       @abstractmethod
       def name(self) -> str:
           """Tool name."""
       
       @abstractmethod
       def description(self) -> str:
           """Tool description."""
       
       @abstractmethod
       def parameters(self) -> dict[str, Any]:
           """Tool parameters schema."""
       
       @abstractmethod
       async def execute(self, **kwargs) -> ToolResult:
           """Execute tool with parameters."""
   ```

2. **Calculator Tool** (`tools/calculator.py`)
   ```python
   class CalculatorTool(BaseTool):
       def name(self) -> str:
           return "calculator"
       
       def description(self) -> str:
           return "Perform arithmetic operations: add, subtract, multiply, divide"
       
       async def execute(
           self,
           operation: Literal["add", "subtract", "multiply", "divide"],
           a: float,
           b: float,
       ) -> ToolResult:
           # Implementation with error handling
   ```

3. **TodoStore Tool** (`tools/todo_store.py`)
   ```python
   class TodoStoreTool(BaseTool):
       def __init__(self):
           self.todos: dict[str, dict[str, Any]] = {}
       
       async def execute(
           self,
           action: Literal["add", "list", "complete", "delete"],
           todo_id: str | None = None,
           text: str | None = None,
       ) -> ToolResult:
           # Implementation with state management
   ```

4. **Tool Registry** (`tools/registry.py`)
   ```python
   class ToolRegistry:
       def __init__(self):
           self.tools: dict[str, BaseTool] = {}
       
       def register(self, tool: BaseTool) -> None:
           self.tools[tool.name()] = tool
       
       def get_tool(self, name: str) -> BaseTool:
           if name not in self.tools:
               raise ValueError(f"Tool not found: {name}")
           return self.tools[name]
       
       def list_tools(self) -> list[ToolDefinition]:
           return [
               ToolDefinition(
                   name=tool.name(),
                   description=tool.description(),
                   parameters=tool.parameters(),
               )
               for tool in self.tools.values()
           ]
   ```

### Phase 3: Planner (2-3 hours)
**Location**: `src/challenge/planner/`

1. **Base Planner** (`planner/base.py`)
   ```python
   class BasePlanner(ABC):
       @abstractmethod
       async def create_plan(
           self,
           prompt: str,
           available_tools: list[ToolDefinition],
       ) -> Plan:
           """Create execution plan from prompt."""
   ```

2. **Rule-Based Planner** (`planner/rule_based.py`)
   ```python
   class RuleBasedPlanner(BasePlanner):
       """Simple rule-based planning for MVP."""
       
       async def create_plan(
           self,
           prompt: str,
           available_tools: list[ToolDefinition],
       ) -> Plan:
           # Pattern matching for calculator operations
           # Pattern matching for todo operations
           # Combination handling
   ```

3. **LLM Planner** (Optional) (`planner/llm_planner.py`)
   ```python
   class LLMPlanner(BasePlanner):
       """LLM-based planning using OpenAI."""
       
       async def create_plan(
           self,
           prompt: str,
           available_tools: list[ToolDefinition],
       ) -> Plan:
           # Use OpenAI for sophisticated planning
   ```

### Phase 4: Orchestrator (4-5 hours)
**Location**: `src/challenge/orchestrator/`

1. **Orchestrator** (`orchestrator/orchestrator.py`)
   ```python
   class Orchestrator:
       def __init__(
           self,
           planner: BasePlanner,
           tool_registry: ToolRegistry,
           settings: Settings,
       ):
           self.planner = planner
           self.tool_registry = tool_registry
           self.settings = settings
           self.runs: dict[str, RunResponse] = {}
       
       async def create_run(self, request: RunCreate) -> RunResponse:
           """Create and start execution run."""
           run_id = str(uuid.uuid4())
           
           # Create plan
           plan = await self.planner.create_plan(
               prompt=request.prompt,
               available_tools=self.tool_registry.list_tools(),
           )
           
           # Create run
           run = RunResponse(
               run_id=run_id,
               status=RunStatus.PENDING,
               steps=[],
               created_at=datetime.now(timezone.utc),
               updated_at=datetime.now(timezone.utc),
           )
           
           self.runs[run_id] = run
           
           # Execute asynchronously
           asyncio.create_task(self._execute_run(run_id, plan, request.max_steps))
           
           return run
       
       async def get_run(self, run_id: str) -> RunResponse:
           """Get run by ID."""
           if run_id not in self.runs:
               raise ValueError(f"Run not found: {run_id}")
           return self.runs[run_id]
       
       async def _execute_run(
           self,
           run_id: str,
           plan: Plan,
           max_steps: int,
       ) -> None:
           """Execute run with retry logic."""
           run = self.runs[run_id]
           run.status = RunStatus.RUNNING
           
           try:
               for step in plan.steps[:max_steps]:
                   await self._execute_step(run, step)
               
               run.status = RunStatus.COMPLETED
           except Exception as e:
               run.status = RunStatus.FAILED
               run.error = str(e)
           finally:
               run.updated_at = datetime.now(timezone.utc)
       
       async def _execute_step(
           self,
           run: RunResponse,
           step: PlanStep,
       ) -> None:
           """Execute single step with retry."""
           max_retries = 3
           
           for attempt in range(max_retries):
               try:
                   tool = self.tool_registry.get_tool(step.tool_name)
                   result = await tool.execute(**step.tool_input)
                   
                   run.steps.append({
                       "step_number": step.step_number,
                       "tool_name": step.tool_name,
                       "tool_input": step.tool_input,
                       "result": result.model_dump(),
                       "attempt": attempt + 1,
                   })
                   
                   if result.success:
                       break
               except Exception as e:
                   if attempt == max_retries - 1:
                       raise
                   await asyncio.sleep(2 ** attempt)  # Exponential backoff
   ```

### Phase 5: API Integration (2-3 hours)
**Location**: `src/challenge/api/`

1. **Dependencies** (`api/dependencies.py`)
   ```python
   def get_tool_registry(settings: SettingsDep) -> ToolRegistry:
       registry = ToolRegistry()
       registry.register(CalculatorTool())
       registry.register(TodoStoreTool())
       return registry
   
   def get_planner(settings: SettingsDep) -> BasePlanner:
       return RuleBasedPlanner()
   
   def get_orchestrator(
       settings: SettingsDep,
       planner: Annotated[BasePlanner, Depends(get_planner)],
       registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
   ) -> Orchestrator:
       return Orchestrator(planner, registry, settings)
   
   OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
   ```

2. **Run Routes** (`api/routes/runs.py`)
   ```python
   router = APIRouter()
   
   @router.post("/runs", status_code=status.HTTP_201_CREATED)
   async def create_run(
       request: RunCreate,
       orchestrator: OrchestratorDep,
   ) -> RunResponse:
       """Create and start execution run."""
       return await orchestrator.create_run(request)
   
   @router.get("/runs/{run_id}")
   async def get_run(
       run_id: str,
       orchestrator: OrchestratorDep,
   ) -> RunResponse:
       """Get run status and results."""
       try:
           return await orchestrator.get_run(run_id)
       except ValueError as e:
           raise HTTPException(status_code=404, detail=str(e))
   ```

3. **Register Routes** (`api/main.py`)
   ```python
   def _register_routes(app: FastAPI) -> None:
       app.include_router(health.router, prefix="/api/v1", tags=["health"])
       app.include_router(runs.router, prefix="/api/v1", tags=["runs"])
   ```

### Phase 6: Testing (4-5 hours)
**Location**: `tests/`

1. **Tool Tests** (`tests/unit/tools/`)
   - Test calculator operations
   - Test todo store actions
   - Test error handling
   - Test tool registry

2. **Planner Tests** (`tests/unit/planner/`)
   - Test plan creation
   - Test tool selection
   - Test error cases

3. **Orchestrator Tests** (`tests/unit/orchestrator/`)
   - Test run creation
   - Test execution flow
   - Test retry logic
   - Test state management

4. **API Tests** (`tests/unit/api/routes/`)
   - Test POST /runs endpoint
   - Test GET /runs/{run_id} endpoint
   - Test error responses
   - Test status transitions

5. **Integration Tests** (`tests/integration/`)
   - End-to-end run execution
   - Multiple tool usage
   - Error recovery
   - Concurrent runs

## Test Coverage Requirements
- Unit tests: ≥80% coverage
- All error paths tested
- All status transitions tested
- Edge cases covered

## Implementation Notes

### Key Design Decisions
1. **Simple State Management**: In-memory dict for MVP (can upgrade to database)
2. **Async Execution**: Background task for long-running operations
3. **Retry Logic**: Exponential backoff for transient failures
4. **Clean Architecture**: Separation of concerns (tools, planning, orchestration, API)

### Error Handling Strategy
- Standard Python exceptions in business logic
- HTTPException at API layer
- Proper status codes (404, 400, 500)
- Detailed error messages

### Testing Strategy
- Unit tests for each component
- Integration tests for end-to-end flows
- Fixtures for test data
- Mocks for external dependencies

## Estimated Timeline
- Phase 1 (Models): 2-3 hours
- Phase 2 (Tools): 3-4 hours
- Phase 3 (Planner): 2-3 hours
- Phase 4 (Orchestrator): 4-5 hours
- Phase 5 (API): 2-3 hours
- Phase 6 (Testing): 4-5 hours

**Total**: 17-23 hours for complete implementation with comprehensive testing

## Next Immediate Steps
1. Create data models in `src/challenge/models/`
2. Implement tool system in `src/challenge/tools/`
3. Build planner in `src/challenge/planner/`
4. Develop orchestrator in `src/challenge/orchestrator/`
5. Integrate with API endpoints
6. Write comprehensive tests
