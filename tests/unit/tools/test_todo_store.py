"""
Unit tests for TodoStore tool.

Tests cover:
- CRUD operations (add, list, get, complete, delete)
- Input validation
- Error handling
- State management
"""

import pytest

from challenge.infrastructure.tools.implementations.todo_store import TodoStoreTool


class TestTodoStoreTool:
    """Test suite for TodoStoreTool."""

    @pytest.fixture
    def todo_store(self):
        """Provide fresh todo store instance."""
        return TodoStoreTool()

    def test_metadata(self, todo_store):
        """Test todo store metadata."""
        metadata = todo_store.metadata
        assert metadata.name == "todo_store"
        assert "CRUD" in metadata.description or "todo" in metadata.description.lower()
        assert "action" in metadata.input_schema["properties"]

    @pytest.mark.asyncio
    async def test_add_todo(self, todo_store):
        """Test adding a todo."""
        result = await todo_store.execute(action="add", text="Buy groceries")
        assert result.success is True
        assert result.output.todo.text == "Buy groceries"
        assert result.output.todo.completed is False
        assert result.output.todo.id is not None
        assert result.output.todo.created_at is not None

    @pytest.mark.asyncio
    async def test_add_todo_empty_text(self, todo_store):
        """Test adding todo with empty text fails."""
        result = await todo_store.execute(action="add", text="")
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_add_todo_whitespace_only(self, todo_store):
        """Test adding todo with whitespace-only text fails."""
        result = await todo_store.execute(action="add", text="   ")
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_add_todo_none_text(self, todo_store):
        """Test adding todo with None text fails."""
        result = await todo_store.execute(action="add", text=None)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_add_todo_strips_whitespace(self, todo_store):
        """Test that added todo text is stripped."""
        result = await todo_store.execute(action="add", text="  Buy milk  ")
        assert result.success is True
        assert result.output.todo.text == "Buy milk"

    @pytest.mark.asyncio
    async def test_list_todos_empty(self, todo_store):
        """Test listing todos when empty."""
        result = await todo_store.execute(action="list")
        assert result.success is True
        assert result.output.todos == []
        assert result.output.total_count == 0

    @pytest.mark.asyncio
    async def test_list_todos_with_items(self, todo_store):
        """Test listing todos with items."""
        # Add some todos
        await todo_store.execute(action="add", text="Task 1")
        await todo_store.execute(action="add", text="Task 2")
        await todo_store.execute(action="add", text="Task 3")

        result = await todo_store.execute(action="list")
        assert result.success is True
        assert len(result.output.todos) == 3
        assert result.output.total_count == 3
        assert result.output.pending_count == 3
        assert result.output.completed_count == 0

    @pytest.mark.asyncio
    async def test_get_todo(self, todo_store):
        """Test getting a specific todo."""
        # Add a todo
        add_result = await todo_store.execute(action="add", text="Test task")
        todo_id = add_result.output.todo.id

        # Get the todo
        result = await todo_store.execute(action="get", todo_id=todo_id)
        assert result.success is True
        assert result.output.todo.id == todo_id
        assert result.output.todo.text == "Test task"

    @pytest.mark.asyncio
    async def test_get_todo_not_found(self, todo_store):
        """Test getting non-existent todo."""
        result = await todo_store.execute(action="get", todo_id="nonexistent-id")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_todo_no_id(self, todo_store):
        """Test getting todo without ID."""
        result = await todo_store.execute(action="get", todo_id=None)
        assert result.success is False
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_complete_todo(self, todo_store):
        """Test completing a todo."""
        # Add a todo
        add_result = await todo_store.execute(action="add", text="Complete me")
        todo_id = add_result.output.todo.id

        # Complete it
        result = await todo_store.execute(action="complete", todo_id=todo_id)
        assert result.success is True
        assert result.output.todo.completed is True
        assert result.output.todo.completed_at is not None

    @pytest.mark.asyncio
    async def test_complete_todo_twice(self, todo_store):
        """Test completing already completed todo fails."""
        # Add and complete a todo
        add_result = await todo_store.execute(action="add", text="Complete me")
        todo_id = add_result.output.todo.id
        await todo_store.execute(action="complete", todo_id=todo_id)

        # Try to complete again
        result = await todo_store.execute(action="complete", todo_id=todo_id)
        assert result.success is False
        assert "already completed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_complete_todo_not_found(self, todo_store):
        """Test completing non-existent todo."""
        result = await todo_store.execute(action="complete", todo_id="nonexistent-id")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_complete_todo_no_id(self, todo_store):
        """Test completing todo without ID."""
        result = await todo_store.execute(action="complete", todo_id=None)
        assert result.success is False
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_todo(self, todo_store):
        """Test deleting a todo."""
        # Add a todo
        add_result = await todo_store.execute(action="add", text="Delete me")
        todo_id = add_result.output.todo.id

        # Delete it
        result = await todo_store.execute(action="delete", todo_id=todo_id)
        assert result.success is True
        assert result.output.todo.id == todo_id
        assert result.output.remaining_count == 0

        # Verify it's gone
        get_result = await todo_store.execute(action="get", todo_id=todo_id)
        assert get_result.success is False

    @pytest.mark.asyncio
    async def test_delete_todo_not_found(self, todo_store):
        """Test deleting non-existent todo."""
        result = await todo_store.execute(action="delete", todo_id="nonexistent-id")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_todo_no_id(self, todo_store):
        """Test deleting todo without ID."""
        result = await todo_store.execute(action="delete", todo_id=None)
        assert result.success is False
        assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self, todo_store):
        """Test unknown action returns error."""
        result = await todo_store.execute(action="unknown_action")
        assert result.success is False
        assert "unknown action" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_todos_counts(self, todo_store):
        """Test list output counts correctly."""
        # Add 3 todos
        id1 = (await todo_store.execute(action="add", text="Task 1")).output.todo.id
        id2 = (await todo_store.execute(action="add", text="Task 2")).output.todo.id
        await todo_store.execute(action="add", text="Task 3")

        # Complete 2 of them
        await todo_store.execute(action="complete", todo_id=id1)
        await todo_store.execute(action="complete", todo_id=id2)

        # Check counts
        result = await todo_store.execute(action="list")
        assert result.success is True
        assert result.output.total_count == 3
        assert result.output.completed_count == 2
        assert result.output.pending_count == 1

    @pytest.mark.asyncio
    async def test_full_workflow(self, todo_store):
        """Test complete CRUD workflow."""
        # Add todos
        id1 = (await todo_store.execute(action="add", text="Task 1")).output.todo.id
        id2 = (await todo_store.execute(action="add", text="Task 2")).output.todo.id
        id3 = (await todo_store.execute(action="add", text="Task 3")).output.todo.id

        # List all
        list_result = await todo_store.execute(action="list")
        assert len(list_result.output.todos) == 3

        # Complete one
        await todo_store.execute(action="complete", todo_id=id2)

        # Delete one
        await todo_store.execute(action="delete", todo_id=id3)

        # Final list
        final_list = await todo_store.execute(action="list")
        assert len(final_list.output.todos) == 2
        assert sorted([id1, id2]) == sorted([t.id for t in final_list.output.todos])
        assert final_list.output.completed_count == 1
        assert final_list.output.pending_count == 1

    @pytest.mark.asyncio
    async def test_todo_unique_ids(self, todo_store):
        """Test that todos get unique IDs."""
        result1 = await todo_store.execute(action="add", text="Task 1")
        result2 = await todo_store.execute(action="add", text="Task 2")

        assert result1.output.todo.id != result2.output.todo.id

    @pytest.mark.asyncio
    async def test_metadata_includes_action(self, todo_store):
        """Test that result metadata includes action type."""
        add_result = await todo_store.execute(action="add", text="Test")
        assert add_result.metadata["action"] == "add"

        list_result = await todo_store.execute(action="list")
        assert list_result.metadata["action"] == "list"
