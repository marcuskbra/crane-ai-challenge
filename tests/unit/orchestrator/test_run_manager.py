"""
Tests for RunManager.

Tests run storage, retrieval, and task tracking functionality.
"""

import asyncio

import pytest

from challenge.domain.models.run import Run, RunStatus
from challenge.services.orchestration.run_manager import RunManager


class TestRunManager:
    """Test cases for RunManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh RunManager instance."""
        return RunManager()

    @pytest.fixture
    def sample_run(self):
        """Create a sample run for testing."""
        return Run(prompt="test task")

    def test_initial_state(self, manager):
        """Test manager starts with empty storage."""
        assert len(manager.runs) == 0
        assert len(manager.tasks) == 0
        assert manager.list_runs() == []

    def test_create_run_without_task(self, manager, sample_run):
        """Test creating a run without a task."""
        manager.create_run(sample_run)

        assert len(manager.runs) == 1
        assert sample_run.run_id in manager.runs
        assert manager.get_run(sample_run.run_id) == sample_run

    @pytest.mark.asyncio
    async def test_create_run_with_task(self, manager, sample_run):
        """Test creating a run with a background task."""

        async def dummy_task():
            await asyncio.sleep(0.01)
            return "done"

        task = asyncio.create_task(dummy_task())

        manager.create_run(sample_run, task)

        assert len(manager.runs) == 1
        assert len(manager.tasks) == 1
        assert manager.get_task(sample_run.run_id) == task

        # Cleanup
        await task

    def test_get_run_exists(self, manager, sample_run):
        """Test retrieving an existing run."""
        manager.create_run(sample_run)

        retrieved = manager.get_run(sample_run.run_id)

        assert retrieved is sample_run
        assert retrieved.run_id == sample_run.run_id

    def test_get_run_not_found(self, manager):
        """Test retrieving a non-existent run returns None."""
        result = manager.get_run("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_task_exists(self, manager, sample_run):
        """Test retrieving an existing task."""

        async def dummy_task():
            return "done"

        task = asyncio.create_task(dummy_task())

        manager.create_run(sample_run, task)

        retrieved_task = manager.get_task(sample_run.run_id)

        assert retrieved_task is task

        # Cleanup
        await task

    def test_get_task_not_found(self, manager, sample_run):
        """Test retrieving a task when none exists."""
        manager.create_run(sample_run)  # No task

        result = manager.get_task(sample_run.run_id)

        assert result is None

    def test_get_task_nonexistent_run(self, manager):
        """Test retrieving task for non-existent run."""
        result = manager.get_task("nonexistent-id")

        assert result is None

    def test_list_runs_empty(self, manager):
        """Test listing runs when empty."""
        runs = manager.list_runs()

        assert runs == []
        assert isinstance(runs, list)

    def test_list_runs_single(self, manager, sample_run):
        """Test listing runs with one run."""
        manager.create_run(sample_run)

        runs = manager.list_runs()

        assert len(runs) == 1
        assert sample_run in runs

    def test_list_runs_multiple(self, manager):
        """Test listing multiple runs."""
        run1 = Run(prompt="task 1")
        run2 = Run(prompt="task 2")
        run3 = Run(prompt="task 3")

        manager.create_run(run1)
        manager.create_run(run2)
        manager.create_run(run3)

        runs = manager.list_runs()

        assert len(runs) == 3
        assert run1 in runs
        assert run2 in runs
        assert run3 in runs

    def test_create_multiple_runs(self, manager):
        """Test creating multiple runs."""
        runs = [Run(prompt=f"task {i}") for i in range(5)]

        for run in runs:
            manager.create_run(run)

        assert len(manager.runs) == 5
        for run in runs:
            assert manager.get_run(run.run_id) == run

    def test_clear(self, manager, sample_run):
        """Test clearing all runs and tasks."""
        manager.create_run(sample_run)

        manager.clear()

        assert len(manager.runs) == 0
        assert len(manager.tasks) == 0
        assert manager.list_runs() == []
        assert manager.get_run(sample_run.run_id) is None

    @pytest.mark.asyncio
    async def test_clear_with_tasks(self, manager, sample_run):
        """Test clearing runs and tasks together."""

        async def dummy_task():
            return "done"

        task = asyncio.create_task(dummy_task())

        manager.create_run(sample_run, task)

        manager.clear()

        assert len(manager.runs) == 0
        assert len(manager.tasks) == 0

        # Cleanup
        await task

    def test_run_state_modification(self, manager, sample_run):
        """Test that run state modifications are reflected."""
        manager.create_run(sample_run)

        # Modify run state
        sample_run.status = RunStatus.RUNNING

        # Retrieve and verify state changed
        retrieved = manager.get_run(sample_run.run_id)
        assert retrieved.status == RunStatus.RUNNING

    def test_multiple_runs_independent(self, manager):
        """Test that multiple runs maintain independence."""
        run1 = Run(prompt="task 1")
        run2 = Run(prompt="task 2")

        manager.create_run(run1)
        manager.create_run(run2)

        # Modify run1
        run1.status = RunStatus.COMPLETED

        # Verify run2 unaffected
        retrieved_run2 = manager.get_run(run2.run_id)
        assert retrieved_run2.status == RunStatus.PENDING

    @pytest.mark.asyncio
    async def test_task_completion_tracked(self, manager, sample_run):
        """Test that task completion can be tracked."""

        async def completable_task():
            await asyncio.sleep(0.01)
            return "completed"

        task = asyncio.create_task(completable_task())

        manager.create_run(sample_run, task)

        # Verify task is not done initially
        retrieved_task = manager.get_task(sample_run.run_id)
        assert not retrieved_task.done()

        # Wait for completion
        await task

        # Verify task is done
        assert retrieved_task.done()
        assert await retrieved_task == "completed"

    def test_duplicate_run_id_overwrites(self, manager):
        """Test that creating run with same ID overwrites."""
        run1 = Run(prompt="task 1")
        run2 = Run(prompt="task 2")

        # Force same run_id (normally auto-generated unique)
        run2.run_id = run1.run_id

        manager.create_run(run1)
        manager.create_run(run2)

        assert len(manager.runs) == 1
        retrieved = manager.get_run(run1.run_id)
        assert retrieved.prompt == "task 2"
