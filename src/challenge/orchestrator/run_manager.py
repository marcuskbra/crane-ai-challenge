"""
Run storage and lifecycle management.

This module provides the RunManager class for managing run storage,
retrieval, and background task tracking.
"""

import asyncio
import logging

from challenge.models.run import Run

logger = logging.getLogger(__name__)


class RunManager:
    """
    Manages run storage and lifecycle.

    The RunManager handles:
    - Run storage and retrieval
    - Background task tracking
    - Run state management

    """

    def __init__(self):
        """Initialize run manager with empty storage."""
        self.runs: dict[str, Run] = {}
        self.tasks: dict[str, asyncio.Task] = {}

    def create_run(self, run: Run, task: asyncio.Task | None = None) -> None:
        """
        Store a new run and optionally its background task.

        Args:
            run: Run instance to store
            task: Optional background task for async execution

        """
        self.runs[run.run_id] = run
        if task:
            self.tasks[run.run_id] = task
        logger.debug(f"Created run {run.run_id}")

    def get_run(self, run_id: str) -> Run | None:
        """
        Retrieve a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run instance or None if not found

        """
        return self.runs.get(run_id)

    def get_task(self, run_id: str) -> asyncio.Task | None:
        """
        Retrieve the background task for a run.

        Args:
            run_id: Run identifier

        Returns:
            asyncio.Task or None if not found

        """
        return self.tasks.get(run_id)

    def list_runs(self) -> list[Run]:
        """
        Get all runs.

        Returns:
            List of all Run instances

        """
        return list(self.runs.values())

    def clear(self) -> None:
        """
        Clear all runs and tasks.

        Useful for testing and cleanup.

        """
        self.runs.clear()
        self.tasks.clear()
        logger.debug("Cleared all runs and tasks")
