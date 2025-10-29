"""
Run execution endpoints for AI Agent Runtime.

This module provides endpoints for creating and retrieving run executions.
"""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from challenge.api.dependencies import OrchestratorDep
from challenge.models.run import Run

logger = logging.getLogger(__name__)

router = APIRouter()


class RunCreate(BaseModel):
    """
    Request model for creating a new run.

    Attributes:
        prompt: Natural language task description

    """

    prompt: str = Field(
        ...,
        min_length=1,
        description="Natural language task to execute",
        examples=["calculate 2 + 3", "add todo Buy milk"],
    )


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
        HTTPException: 400 if prompt is invalid or planning fails

    """
    try:
        run = await orchestrator.create_run(request.prompt)
        logger.info(f"Created run {run.run_id} with prompt: {request.prompt}")
        return run

    except ValueError as e:
        # Planning or validation error
        logger.warning(f"Invalid prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid prompt: {e!s}",
        )
    except Exception as e:
        # Unexpected error
        logger.error(f"Failed to create run: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create run: {e!s}",
        )


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
        HTTPException: 404 if run not found

    """
    run = orchestrator.get_run(run_id)

    if not run:
        logger.warning(f"Run not found: {run_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )

    return run
