"""
Run execution endpoints for AI Agent Runtime.

This module provides endpoints for creating and retrieving run executions.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from challenge.api.dependencies import OrchestratorDep
from challenge.api.schemas.runs import RunCreate
from challenge.core.exceptions import RunNotFoundError
from challenge.domain.models.run import Run

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/runs",
    response_model=Run,
    status_code=status.HTTP_201_CREATED,
    summary="Create and execute a run",
    description="Create a new run from a natural language prompt. "
    "The run starts executing asynchronously and returns immediately.",
    tags=["runs"],
)
async def create_run(request: RunCreate, orchestrator: OrchestratorDep) -> Run:
    """
    Create and execute a new run.

    This endpoint:
    1. Creates a run in PENDING status
    2. Generates an execution plan from the prompt
    3. Starts asynchronous execution
    4. Returns immediately with run details

    The run continues executing in the background. Use GET /runs/{run_id}
    to check status and retrieve results.

    Args:
        request: Run creation request with prompt
        orchestrator: Injected orchestrator instance

    Returns:
        Run instance with run_id for status tracking

    Raises:
        InvalidPromptError: If prompt is invalid or empty
        PlanGenerationError: If planning fails

    Note:
        Exceptions are handled by centralized exception handlers.
        No try/except needed in route handlers.

    """
    run = await orchestrator.create_run(request.prompt)
    logger.info(f"Created run {run.run_id} with prompt: {request.prompt}")
    return run


@router.get(
    "/runs",
    response_model=list[Run],
    summary="List runs",
    description="Retrieve a list of runs with pagination, in reverse chronological order (most recent first).",
    tags=["runs"],
)
async def list_runs(
    orchestrator: OrchestratorDep,
    limit: int = 10,
    offset: int = 0,
) -> list[Run]:
    """
    List runs with pagination.

    Returns runs in reverse chronological order (most recent first).

    Args:
        limit: Maximum number of runs to return (default: 10, max: 100)
        offset: Number of runs to skip (default: 0)
        orchestrator: Injected orchestrator instance

    Returns:
        List of Run instances

    Raises:
        HTTPException: 400 if parameters are invalid

    """
    # Validate parameters
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 100",
        )

    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Offset must be non-negative",
        )

    runs = orchestrator.list_runs(limit=limit, offset=offset)
    logger.debug(f"Listed {len(runs)} runs (limit={limit}, offset={offset})")
    return runs


@router.get(
    "/runs/{run_id}",
    response_model=Run,
    summary="Get run status",
    description="Retrieve run status, execution history, and results.",
    tags=["runs"],
)
async def get_run(run_id: str, orchestrator: OrchestratorDep) -> Run:
    """
    Get run by ID.

    Returns complete run state including:
    - Current status (PENDING/RUNNING/COMPLETED/FAILED)
    - Execution plan
    - Step execution history
    - Final result (if completed)
    - Error details (if failed)

    Args:
        run_id: Run identifier
        orchestrator: Injected orchestrator instance

    Returns:
        Run instance with complete state

    Raises:
        RunNotFoundError: If run doesn't exist

    Note:
        Exceptions are handled by centralized exception handlers.

    """
    run: Run | None = orchestrator.get_run(run_id)

    if not run:
        raise RunNotFoundError(run_id)

    return run
