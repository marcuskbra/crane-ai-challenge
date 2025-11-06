"""
TodoStore tool with CRUD operations.

This module provides an in-memory todo storage implementation
with create, read, update, and delete operations.
"""

from datetime import datetime, timezone
from uuid import uuid4

from challenge.domain.types import (
    TodoAddOutput,
    TodoCompleteOutput,
    TodoDeleteOutput,
    TodoGetOutput,
    TodoItem,
    TodoListOutput,
)
from challenge.infrastructure.tools.base import BaseTool, ToolMetadata, ToolResult


class TodoStoreTool(BaseTool):
    """
    In-memory todo storage with CRUD operations.

    Supported actions:
        - add: Create a new todo item
        - list: Retrieve all todos
        - get: Retrieve a specific todo by ID
        - complete: Mark a todo as completed
        - delete: Remove a todo by ID

    """

    def __init__(self):
        """Initialize empty todo storage."""
        self.todos: dict[str, TodoItem] = {}

    @property
    def metadata(self) -> ToolMetadata:
        """Get todo store tool metadata."""
        return ToolMetadata(
            name="todo_store",
            description="Manage todos with CRUD operations (add, list, get, complete, delete)",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "get", "complete", "delete"],
                        "description": "Action to perform",
                    },
                    "text": {
                        "type": "string",
                        "description": "Todo text (required for 'add' action)",
                    },
                    "todo_id": {
                        "type": "string",
                        "description": "Todo ID (required for 'get', 'complete', 'delete' actions)",
                    },
                },
                "required": ["action"],
            },
        )

    async def execute(
        self,
        action: str,
        text: str | None = None,
        todo_id: str | None = None,
        **kwargs,
    ) -> ToolResult[TodoAddOutput | TodoListOutput | TodoGetOutput | TodoCompleteOutput | TodoDeleteOutput]:
        """
        Execute todo store operation.

        Args:
            action: Action to perform (add/list/get/complete/delete)
            text: Todo text for add action
            todo_id: Todo ID for get/complete/delete actions
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolResult with strongly-typed Pydantic output:
            - add: Returns TodoAddOutput
            - list: Returns TodoListOutput
            - get: Returns TodoGetOutput
            - complete: Returns TodoCompleteOutput
            - delete: Returns TodoDeleteOutput

        """
        if action == "add":
            return await self._add_todo(text)
        elif action == "list":
            return await self._list_todos()
        elif action == "get":
            return await self._get_todo(todo_id)
        elif action == "complete":
            return await self._complete_todo(todo_id)
        elif action == "delete":
            return await self._delete_todo(todo_id)
        else:
            # For unknown actions, return error (type inferred from method signature)
            return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _add_todo(self, text: str | None) -> ToolResult[TodoAddOutput]:
        """
        Add a new todo item.

        Args:
            text: Todo text content

        Returns:
            ToolResult[TodoAddOutput] with created todo

        """
        if not text or not text.strip():
            return ToolResult[TodoAddOutput](success=False, error="Todo text cannot be empty")

        todo_id = str(uuid4())
        now = datetime.now(timezone.utc)

        todo = TodoItem(
            id=todo_id,
            text=text.strip(),
            completed=False,
            created_at=now.isoformat(),
            completed_at=None,
        )

        self.todos[todo_id] = todo

        output = TodoAddOutput(todo=todo)

        return ToolResult[TodoAddOutput](
            success=True,
            output=output,
            metadata={"action": "add", "todo_count": len(self.todos)},
        )

    async def _list_todos(self) -> ToolResult[TodoListOutput]:
        """
        List all todos.

        Returns:
            ToolResult[TodoListOutput] with list of all todos and counts

        """
        todos_list = list(self.todos.values())
        total_count = len(todos_list)
        completed_count = sum(1 for t in todos_list if t.completed)
        pending_count = total_count - completed_count

        output = TodoListOutput(
            todos=todos_list,
            total_count=total_count,
            completed_count=completed_count,
            pending_count=pending_count,
        )

        return ToolResult[TodoListOutput](
            success=True,
            output=output,
            metadata={"action": "list"},
        )

    async def _get_todo(self, todo_id: str | None) -> ToolResult[TodoGetOutput]:
        """
        Get a specific todo by ID.

        Args:
            todo_id: Todo identifier

        Returns:
            ToolResult[TodoGetOutput] with todo or error

        """
        if not todo_id:
            return ToolResult[TodoGetOutput](success=False, error="Todo ID is required")

        todo = self.todos.get(todo_id)
        if not todo:
            return ToolResult[TodoGetOutput](success=False, error=f"Todo not found: {todo_id}")

        output = TodoGetOutput(todo=todo)

        return ToolResult[TodoGetOutput](success=True, output=output, metadata={"action": "get", "todo_id": todo_id})

    async def _complete_todo(self, todo_id: str | None) -> ToolResult[TodoCompleteOutput]:
        """
        Mark a todo as completed.

        Args:
            todo_id: Todo identifier

        Returns:
            ToolResult[TodoCompleteOutput] with updated todo or error

        """
        if not todo_id:
            return ToolResult[TodoCompleteOutput](success=False, error="Todo ID is required")

        todo = self.todos.get(todo_id)
        if not todo:
            return ToolResult[TodoCompleteOutput](success=False, error=f"Todo not found: {todo_id}")

        if todo.completed:
            return ToolResult[TodoCompleteOutput](success=False, error=f"Todo already completed: {todo_id}")

        now = datetime.now(timezone.utc)

        # Create updated TodoItem with completed status
        updated_todo = TodoItem(
            id=todo.id,
            text=todo.text,
            completed=True,
            created_at=todo.created_at,
            completed_at=now.isoformat(),
        )

        self.todos[todo_id] = updated_todo
        output = TodoCompleteOutput(todo=updated_todo)

        return ToolResult[TodoCompleteOutput](
            success=True,
            output=output,
            metadata={"action": "complete", "todo_id": todo_id},
        )

    async def _delete_todo(self, todo_id: str | None) -> ToolResult[TodoDeleteOutput]:
        """
        Delete a todo by ID.

        Args:
            todo_id: Todo identifier

        Returns:
            ToolResult[TodoDeleteOutput] with deleted todo or error

        """
        if not todo_id:
            return ToolResult[TodoDeleteOutput](success=False, error="Todo ID is required")

        todo = self.todos.pop(todo_id, None)
        if not todo:
            return ToolResult[TodoDeleteOutput](success=False, error=f"Todo not found: {todo_id}")

        output = TodoDeleteOutput(todo=todo, remaining_count=len(self.todos))

        return ToolResult[TodoDeleteOutput](
            success=True,
            output=output,
            metadata={
                "action": "delete",
                "todo_id": todo_id,
            },
        )
