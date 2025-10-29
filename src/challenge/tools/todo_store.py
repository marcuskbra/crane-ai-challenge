"""
TodoStore tool with CRUD operations.

This module provides an in-memory todo storage implementation
with create, read, update, and delete operations.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from challenge.tools.base import BaseTool, ToolMetadata, ToolResult


class TodoStoreTool(BaseTool):
    """
    In-memory todo storage with CRUD operations.

    Supported actions:
        - add: Create a new todo item
        - list: Retrieve all todos
        - get: Retrieve a specific todo by ID
        - complete: Mark a todo as completed
        - delete: Remove a todo by ID

    Example:
        >>> store = TodoStoreTool()
        >>> result = await store.execute(action="add", text="Buy groceries")
        >>> print(result.output["id"])
        'uuid-string'

    """

    def __init__(self):
        """Initialize empty todo storage."""
        self.todos: dict[str, dict[str, Any]] = {}

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
    ) -> ToolResult:
        """
        Execute todo store operation.

        Args:
            action: Action to perform (add/list/get/complete/delete)
            text: Todo text for add action
            todo_id: Todo ID for get/complete/delete actions
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolResult with operation outcome

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
            return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _add_todo(self, text: str | None) -> ToolResult:
        """
        Add a new todo item.

        Args:
            text: Todo text content

        Returns:
            ToolResult with created todo

        """
        if not text or not text.strip():
            return ToolResult(success=False, error="Todo text cannot be empty")

        todo_id = str(uuid4())
        now = datetime.now(timezone.utc)

        todo = {
            "id": todo_id,
            "text": text.strip(),
            "completed": False,
            "created_at": now.isoformat(),
            "completed_at": None,
        }

        self.todos[todo_id] = todo

        return ToolResult(
            success=True,
            output=todo,
            metadata={"action": "add", "todo_count": len(self.todos)},
        )

    async def _list_todos(self) -> ToolResult:
        """
        List all todos.

        Returns:
            ToolResult with list of all todos

        """
        todos_list = list(self.todos.values())

        return ToolResult(
            success=True,
            output=todos_list,
            metadata={
                "action": "list",
                "total_count": len(todos_list),
                "completed_count": sum(1 for t in todos_list if t["completed"]),
                "pending_count": sum(1 for t in todos_list if not t["completed"]),
            },
        )

    async def _get_todo(self, todo_id: str | None) -> ToolResult:
        """
        Get a specific todo by ID.

        Args:
            todo_id: Todo identifier

        Returns:
            ToolResult with todo or error

        """
        if not todo_id:
            return ToolResult(success=False, error="Todo ID is required")

        todo = self.todos.get(todo_id)
        if not todo:
            return ToolResult(success=False, error=f"Todo not found: {todo_id}")

        return ToolResult(success=True, output=todo, metadata={"action": "get", "todo_id": todo_id})

    async def _complete_todo(self, todo_id: str | None) -> ToolResult:
        """
        Mark a todo as completed.

        Args:
            todo_id: Todo identifier

        Returns:
            ToolResult with updated todo or error

        """
        if not todo_id:
            return ToolResult(success=False, error="Todo ID is required")

        todo = self.todos.get(todo_id)
        if not todo:
            return ToolResult(success=False, error=f"Todo not found: {todo_id}")

        if todo["completed"]:
            return ToolResult(success=False, error=f"Todo already completed: {todo_id}")

        now = datetime.now(timezone.utc)
        todo["completed"] = True
        todo["completed_at"] = now.isoformat()

        return ToolResult(
            success=True,
            output=todo,
            metadata={"action": "complete", "todo_id": todo_id},
        )

    async def _delete_todo(self, todo_id: str | None) -> ToolResult:
        """
        Delete a todo by ID.

        Args:
            todo_id: Todo identifier

        Returns:
            ToolResult with deleted todo or error

        """
        if not todo_id:
            return ToolResult(success=False, error="Todo ID is required")

        todo = self.todos.pop(todo_id, None)
        if not todo:
            return ToolResult(success=False, error=f"Todo not found: {todo_id}")

        return ToolResult(
            success=True,
            output=todo,
            metadata={
                "action": "delete",
                "todo_id": todo_id,
                "remaining_count": len(self.todos),
            },
        )
