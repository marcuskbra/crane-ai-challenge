"""
End-to-end integration tests for runs API.

Tests the complete flow: API → Orchestrator → Planner → Tools
"""

import asyncio

import pytest


class TestRunsE2E:
    """End-to-end integration tests for runs API."""

    @pytest.mark.asyncio
    async def test_calculator_run_complete_flow(self, test_client, wait_for_run_completion):
        """Test complete calculator flow from API to execution."""
        # Create run
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": "calculate 5 * 8"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "run_id" in data
        assert data["prompt"] == "calculate 5 * 8"
        assert data["status"] in ["pending", "running", "completed"]
        assert data["plan"]["steps"][0]["tool_name"] == "calculator"

        run_id = data["run_id"]

        # Wait for execution to complete (polling is much faster than fixed sleep)
        data = await wait_for_run_completion(test_client, run_id)

        assert data["status"] == "completed"
        assert data["result"] == 40.0
        assert len(data["execution_log"]) == 1
        assert data["execution_log"][0]["success"] is True
        assert data["execution_log"][0]["output"] == 40.0

    @pytest.mark.asyncio
    async def test_todo_add_and_list_flow(self, test_client, wait_for_run_completion):
        """Test todo add and list flow."""
        # Create run to add todo
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": "add todo Write tests"},
        )

        assert response.status_code == 201
        run1_id = response.json()["run_id"]

        # Wait for execution (polling)
        data = await wait_for_run_completion(test_client, run1_id)

        assert data["status"] == "completed"
        assert data["result"]["todo"]["text"] == "write tests"  # Planner converts to lowercase
        assert data["result"]["todo"]["completed"] is False

        # Create run to list todos
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": "list todos"},
        )

        assert response.status_code == 201
        run2_id = response.json()["run_id"]

        # Wait for execution (polling)
        data = await wait_for_run_completion(test_client, run2_id)

        assert data["status"] == "completed"
        assert isinstance(data["result"]["todos"], list)
        assert len(data["result"]["todos"]) >= 1
        # Find our todo (case-insensitive since planner lowercases)
        assert any(t["text"] == "write tests" for t in data["result"]["todos"])

    @pytest.mark.asyncio
    async def test_multi_step_run(self, test_client, wait_for_run_completion):
        """Test multi-step run with 'and' operator."""
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": "calculate 10 + 5 and calculate 20 / 4"},
        )

        assert response.status_code == 201
        data = response.json()
        run_id = data["run_id"]

        # Verify plan has 2 steps
        assert len(data["plan"]["steps"]) == 2
        assert data["plan"]["steps"][0]["tool_input"]["expression"] == "10 + 5"
        assert data["plan"]["steps"][1]["tool_input"]["expression"] == "20 / 4"

        # Wait for execution (polling)
        data = await wait_for_run_completion(test_client, run_id)

        assert data["status"] == "completed"
        assert len(data["execution_log"]) == 2
        assert data["execution_log"][0]["output"] == 15.0
        assert data["execution_log"][1]["output"] == 5.0
        assert data["result"] == 5.0  # Result of last step

    @pytest.mark.asyncio
    async def test_complex_multi_step_workflow(self, test_client, wait_for_run_completion):
        """Test complex workflow with calculator and todos."""
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": "calculate 100 / 10 and add todo Review results then list todos"},
        )

        assert response.status_code == 201
        data = response.json()
        run_id = data["run_id"]

        # Verify plan
        assert len(data["plan"]["steps"]) == 3
        assert data["plan"]["steps"][0]["tool_name"] == "calculator"
        assert data["plan"]["steps"][1]["tool_name"] == "todo_store"
        assert data["plan"]["steps"][1]["tool_input"]["action"] == "add"
        assert data["plan"]["steps"][2]["tool_name"] == "todo_store"
        assert data["plan"]["steps"][2]["tool_input"]["action"] == "list"

        # Wait for execution (polling)
        data = await wait_for_run_completion(test_client, run_id)

        assert data["status"] == "completed"
        assert len(data["execution_log"]) == 3
        assert data["execution_log"][0]["success"] is True  # calc
        assert data["execution_log"][1]["success"] is True  # add
        assert data["execution_log"][2]["success"] is True  # list

    @pytest.mark.asyncio
    async def test_invalid_prompt(self, test_client, wait_for_run_completion):
        """Test error handling for invalid prompt."""
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": "do something completely unknown"},
        )

        assert response.status_code == 201  # Run created
        data = response.json()
        run_id = data["run_id"]

        # Wait for planning to fail (polling handles failed status too)
        data = await wait_for_run_completion(test_client, run_id)

        assert data["status"] == "failed"
        assert "could not parse" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_prompt(self, test_client):
        """Test validation for empty prompt."""
        response = test_client.post(
            "/api/v1/runs",
            json={"prompt": ""},
        )

        # Should fail validation
        assert response.status_code == 422

    def test_get_nonexistent_run(self, test_client):
        """Test 404 for nonexistent run."""
        response = test_client.get("/api/v1/runs/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_run_isolation(self, test_client, wait_for_run_completion):
        """Test that runs are isolated and don't interfere."""
        # Create two independent runs
        response1 = test_client.post(
            "/api/v1/runs",
            json={"prompt": "calculate 1 + 1"},
        )
        response2 = test_client.post(
            "/api/v1/runs",
            json={"prompt": "calculate 2 + 2"},
        )

        assert response1.status_code == 201
        assert response2.status_code == 201

        run1_id = response1.json()["run_id"]
        run2_id = response2.json()["run_id"]
        assert run1_id != run2_id

        # Wait for both runs (polling) - can wait in parallel
        data1, data2 = await asyncio.gather(
            wait_for_run_completion(test_client, run1_id), wait_for_run_completion(test_client, run2_id)
        )

        assert data1["result"] == 2.0
        assert data2["result"] == 4.0

    @pytest.mark.asyncio
    async def test_list_runs_default_pagination(self, test_client, wait_for_run_completion):
        """Test listing runs with default pagination."""
        # Create multiple runs
        run_ids = []
        for i in range(5):
            response = test_client.post(
                "/api/v1/runs",
                json={"prompt": f"calculate {i} + 1"},
            )
            assert response.status_code == 201
            run_ids.append(response.json()["run_id"])

        # Wait for at least first run to ensure runs are created (polling)
        await wait_for_run_completion(test_client, run_ids[0])

        # List runs (default: limit=10, offset=0)
        response = test_client.get("/api/v1/runs")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 5  # At least our 5 runs

        # Verify runs are in reverse chronological order (most recent first)
        # The last created run should appear first
        returned_ids = [run["run_id"] for run in data]
        assert run_ids[-1] in returned_ids[:5]  # Most recent should be in first 5

    @pytest.mark.asyncio
    async def test_list_runs_with_limit(self, test_client, wait_for_run_completion):
        """Test listing runs with custom limit."""
        # Create multiple runs
        run_ids = []
        for i in range(5):
            response = test_client.post(
                "/api/v1/runs",
                json={"prompt": f"calculate {i} + 2"},
            )
            assert response.status_code == 201
            run_ids.append(response.json()["run_id"])

        # Wait for at least first run (polling)
        await wait_for_run_completion(test_client, run_ids[0])

        # List with limit=2
        response = test_client.get("/api/v1/runs?limit=2")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 2

    @pytest.mark.asyncio
    async def test_list_runs_with_offset(self, test_client, wait_for_run_completion):
        """Test listing runs with offset."""
        # Create multiple runs
        run_ids = []
        for i in range(5):
            response = test_client.post(
                "/api/v1/runs",
                json={"prompt": f"calculate {i} + 3"},
            )
            assert response.status_code == 201
            run_ids.append(response.json()["run_id"])

        # Wait for at least first run (polling)
        await wait_for_run_completion(test_client, run_ids[0])

        # Get first page
        response1 = test_client.get("/api/v1/runs?limit=2&offset=0")
        assert response1.status_code == 200
        page1 = response1.json()

        # Get second page
        response2 = test_client.get("/api/v1/runs?limit=2&offset=2")
        assert response2.status_code == 200
        page2 = response2.json()

        # Pages should be different
        page1_ids = {run["run_id"] for run in page1}
        page2_ids = {run["run_id"] for run in page2}
        assert len(page1_ids & page2_ids) == 0  # No overlap

    def test_list_runs_invalid_limit(self, test_client):
        """Test validation for invalid limit."""
        # Limit too small
        response = test_client.get("/api/v1/runs?limit=0")
        assert response.status_code == 400
        assert "limit" in response.json()["detail"].lower()

        # Limit too large
        response = test_client.get("/api/v1/runs?limit=101")
        assert response.status_code == 400
        assert "limit" in response.json()["detail"].lower()

    def test_list_runs_invalid_offset(self, test_client):
        """Test validation for invalid offset."""
        response = test_client.get("/api/v1/runs?offset=-1")
        assert response.status_code == 400
        assert "offset" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_list_runs_reverse_chronological(self, test_client, wait_for_run_completion):
        """Test that runs are listed in reverse chronological order."""
        # Create runs sequentially to ensure order
        run_ids = []
        for i in range(3):
            response = test_client.post(
                "/api/v1/runs",
                json={"prompt": f"calculate {i} + 10"},
            )
            assert response.status_code == 201
            run_id = response.json()["run_id"]
            run_ids.append(run_id)
            # Wait for each run to complete to ensure clear temporal ordering
            await wait_for_run_completion(test_client, run_id)

        # List runs
        response = test_client.get("/api/v1/runs?limit=3")
        assert response.status_code == 200

        data = response.json()
        returned_ids = [run["run_id"] for run in data[:3]]

        # Most recent (last created) should be first
        assert run_ids[-1] in returned_ids[:3]
