"""Unit tests for API Pydantic schemas."""

import pytest
from pydantic import ValidationError
from datetime import datetime, timezone

from roma_dspy.api.schemas import (
    SolveRequest,
    CheckpointRestoreRequest,
    ExecutionResponse,
    ExecutionDetailResponse,
    CheckpointResponse,
    TaskNodeResponse,
    VisualizationRequest,
    MetricsResponse,
    HealthResponse,
    ErrorResponse,
    StatusPollingResponse,
)


class TestSolveRequest:
    """Tests for SolveRequest schema."""

    def test_valid_request(self):
        """Test valid solve request."""
        req = SolveRequest(
            goal="Test task",
            max_depth=3,
            config_profile="high_quality"
        )
        assert req.goal == "Test task"
        assert req.max_depth == 3
        assert req.config_profile == "high_quality"

    def test_default_max_depth(self):
        """Test default max_depth value."""
        req = SolveRequest(goal="Test task")
        assert req.max_depth == 2

    def test_empty_goal_fails(self):
        """Test that empty goal fails validation."""
        with pytest.raises(ValidationError):
            SolveRequest(goal="")

    def test_max_depth_bounds(self):
        """Test max_depth boundary validation."""
        # Valid boundaries
        SolveRequest(goal="Test", max_depth=0)
        SolveRequest(goal="Test", max_depth=10)

        # Invalid boundaries
        with pytest.raises(ValidationError):
            SolveRequest(goal="Test", max_depth=-1)

        with pytest.raises(ValidationError):
            SolveRequest(goal="Test", max_depth=11)

    def test_optional_fields(self):
        """Test optional fields."""
        req = SolveRequest(
            goal="Test",
            config_overrides={"key": "value"},
            metadata={"custom": "data"}
        )
        assert req.config_overrides == {"key": "value"}
        assert req.metadata == {"custom": "data"}


class TestExecutionResponse:
    """Tests for ExecutionResponse schema."""

    def test_valid_response(self):
        """Test valid execution response."""
        resp = ExecutionResponse(
            execution_id="exec-123",
            status="running",
            initial_goal="Test goal",
            max_depth=2,
            total_tasks=10,
            completed_tasks=5,
            failed_tasks=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={}
        )
        assert resp.execution_id == "exec-123"
        assert resp.status == "running"
        assert resp.total_tasks == 10

    def test_optional_config(self):
        """Test optional config field."""
        resp = ExecutionResponse(
            execution_id="exec-123",
            status="completed",
            initial_goal="Test",
            max_depth=2,
            total_tasks=1,
            completed_tasks=1,
            failed_tasks=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            config={"test": "config"},
            metadata={}
        )
        assert resp.config == {"test": "config"}


class TestTaskNodeResponse:
    """Tests for TaskNodeResponse schema."""

    def test_valid_task_node(self):
        """Test valid task node response."""
        node = TaskNodeResponse(
            task_id="task-1",
            goal="Test task",
            status="completed",
            depth=0,
            created_at=datetime.now(timezone.utc)
        )
        assert node.task_id == "task-1"
        assert node.status == "completed"
        assert node.depth == 0

    def test_optional_fields(self):
        """Test optional fields in task node."""
        node = TaskNodeResponse(
            task_id="task-1",
            goal="Test",
            status="completed",
            depth=0,
            node_type="execute",
            parent_id="task-0",
            result="test result",
            error=None,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc)
        )
        assert node.node_type == "execute"
        assert node.parent_id == "task-0"
        assert node.result == "test result"


class TestVisualizationRequest:
    """Tests for VisualizationRequest schema."""

    def test_valid_request(self):
        """Test valid visualization request."""
        req = VisualizationRequest(
            visualizer_type="tree",
            include_subgraphs=True,
            format="text"
        )
        assert req.visualizer_type == "tree"
        assert req.include_subgraphs is True
        assert req.format == "text"

    def test_default_values(self):
        """Test default values."""
        req = VisualizationRequest(visualizer_type="timeline")
        assert req.include_subgraphs is True
        assert req.format == "text"


class TestMetricsResponse:
    """Tests for MetricsResponse schema."""

    def test_valid_metrics(self):
        """Test valid metrics response."""
        metrics = MetricsResponse(
            execution_id="exec-123",
            total_lm_calls=50,
            total_tokens=10000,
            total_cost_usd=1.25,
            average_latency_ms=850.5,
            task_breakdown={}
        )
        assert metrics.total_lm_calls == 50
        assert metrics.total_cost_usd == 1.25
        assert metrics.average_latency_ms == 850.5

    def test_with_task_breakdown(self):
        """Test metrics with task breakdown."""
        breakdown = {
            "task_1": {
                "calls": 10,
                "tokens": 2000,
                "cost_usd": 0.25
            }
        }
        metrics = MetricsResponse(
            execution_id="exec-123",
            total_lm_calls=10,
            total_tokens=2000,
            total_cost_usd=0.25,
            average_latency_ms=500.0,
            task_breakdown=breakdown
        )
        assert "task_1" in metrics.task_breakdown
        assert metrics.task_breakdown["task_1"]["calls"] == 10


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_valid_health(self):
        """Test valid health response."""
        health = HealthResponse(
            status="healthy",
            version="0.1.0",
            uptime_seconds=120.5,
            active_executions=3,
            storage_connected=True,
            cache_size=10,
            timestamp=datetime.now(timezone.utc)
        )
        assert health.status == "healthy"
        assert health.uptime_seconds == 120.5
        assert health.storage_connected is True


class TestStatusPollingResponse:
    """Tests for StatusPollingResponse schema."""

    def test_valid_status(self):
        """Test valid status polling response."""
        status = StatusPollingResponse(
            execution_id="exec-123",
            status="running",
            progress=0.65,
            current_task_id="task-5",
            current_task_goal="Processing data",
            completed_tasks=13,
            total_tasks=20,
            estimated_remaining_seconds=120,
            last_updated=datetime.now(timezone.utc)
        )
        assert status.progress == 0.65
        assert status.completed_tasks == 13
        assert status.total_tasks == 20

    def test_optional_estimation(self):
        """Test optional estimation field."""
        status = StatusPollingResponse(
            execution_id="exec-123",
            status="running",
            progress=0.5,
            completed_tasks=10,
            total_tasks=20,
            last_updated=datetime.now(timezone.utc)
        )
        assert status.estimated_remaining_seconds is None


class TestCheckpointResponse:
    """Tests for CheckpointResponse schema."""

    def test_valid_checkpoint(self):
        """Test valid checkpoint response."""
        cp = CheckpointResponse(
            checkpoint_id="cp-123",
            execution_id="exec-123",
            created_at=datetime.now(timezone.utc),
            trigger="manual",
            state="saved",
            file_path="/tmp/checkpoint.json",
            file_size_bytes=1024,
            compressed=True
        )
        assert cp.checkpoint_id == "cp-123"
        assert cp.trigger == "manual"
        assert cp.compressed is True

    def test_optional_file_info(self):
        """Test optional file information."""
        cp = CheckpointResponse(
            checkpoint_id="cp-123",
            execution_id="exec-123",
            created_at=datetime.now(timezone.utc),
            trigger="automatic",
            state="saved",
            compressed=False
        )
        assert cp.file_path is None
        assert cp.file_size_bytes is None


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_valid_error(self):
        """Test valid error response."""
        error = ErrorResponse(
            error="Test error",
            detail="Detailed error message",
            execution_id="exec-123",
            timestamp=datetime.now(timezone.utc)
        )
        assert error.error == "Test error"
        assert error.detail == "Detailed error message"
        assert error.execution_id == "exec-123"

    def test_minimal_error(self):
        """Test minimal error response."""
        error = ErrorResponse(
            error="Simple error",
            timestamp=datetime.now(timezone.utc)
        )
        assert error.error == "Simple error"
        assert error.detail is None
        assert error.execution_id is None
