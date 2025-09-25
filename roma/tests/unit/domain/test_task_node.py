"""
Unit tests for TaskNode entity.

Tests the immutable TaskNode implementation including state transitions,
validation, and property methods.
"""

from datetime import UTC, datetime

import pytest

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


class TestTaskNodeImmutability:
    """Test TaskNode immutability constraints."""

    def test_task_node_is_frozen(self, sample_task_node: TaskNode):
        """Test that TaskNode is frozen (immutable)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Instance is frozen"):
            sample_task_node.goal = "Modified goal"

    def test_cannot_modify_status(self, sample_task_node: TaskNode):
        """Test that status cannot be directly modified."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Instance is frozen"):
            sample_task_node.status = TaskStatus.COMPLETED

    def test_cannot_modify_children(self, sample_task_node: TaskNode):
        """Test that children set cannot be directly modified."""
        with pytest.raises(AttributeError):
            sample_task_node.children.add("new-child")

    def test_cannot_modify_dependencies(self, sample_task_node: TaskNode):
        """Test that dependencies set cannot be directly modified."""
        with pytest.raises(AttributeError):
            sample_task_node.dependencies.add("new-dep")


class TestTaskNodeCreation:
    """Test TaskNode creation and validation."""

    def test_create_basic_task_node(self):
        """Test creating a basic task node."""
        node = TaskNode(
            goal="Test goal",
            task_type=TaskType.THINK
        )

        assert node.goal == "Test goal"
        assert node.task_type == TaskType.THINK
        assert node.status == TaskStatus.PENDING
        assert node.version == 0
        assert node.task_id is not None
        assert len(node.task_id) > 0

    def test_create_with_all_fields(self):
        """Test creating task node with all fields specified."""
        node = TaskNode(
            task_id="custom-id",
            goal="Custom goal",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.READY,
            parent_id="parent-123",
            version=5
        )

        assert node.task_id == "custom-id"
        assert node.goal == "Custom goal"
        assert node.task_type == TaskType.RETRIEVE
        assert node.node_type == NodeType.EXECUTE
        assert node.status == TaskStatus.READY
        assert node.parent_id == "parent-123"
        assert node.version == 5

    def test_empty_goal_raises_error(self):
        """Test that empty goal raises validation error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            TaskNode(goal="", task_type=TaskType.THINK)

    def test_whitespace_goal_raises_error(self):
        """Test that whitespace-only goal raises validation error."""
        with pytest.raises(ValueError, match="Task goal cannot be empty"):
            TaskNode(goal="   ", task_type=TaskType.THINK)

    def test_retrieve_task_can_be_any_node_type(self):
        """Test that RETRIEVE tasks can have any node_type (atomizer decides)."""
        # RETRIEVE tasks can be PLAN (decomposed) or EXECUTE (atomic)
        execute_node = TaskNode(
            goal="Search query",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE
        )
        assert execute_node.node_type == NodeType.EXECUTE

        plan_node = TaskNode(
            goal="Research comprehensive topic",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.PLAN
        )
        assert plan_node.node_type == NodeType.PLAN

    def test_retrieve_task_allows_execute(self):
        """Test that RETRIEVE tasks can have EXECUTE node_type."""
        node = TaskNode(
            goal="Search query",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE
        )
        assert node.node_type == NodeType.EXECUTE


class TestTaskNodeStateTransitions:
    """Test TaskNode state transition methods."""

    def test_valid_status_transition(self, sample_task_node: TaskNode):
        """Test valid status transition creates new instance."""
        new_node = sample_task_node.transition_to(TaskStatus.READY)

        # Different instances
        assert new_node is not sample_task_node

        # Original unchanged
        assert sample_task_node.status == TaskStatus.PENDING
        assert sample_task_node.version == 0

        # New instance updated
        assert new_node.status == TaskStatus.READY
        assert new_node.version == 1
        assert new_node.task_id == sample_task_node.task_id  # Same ID

    def test_invalid_status_transition_raises_error(self, sample_task_node: TaskNode):
        """Test invalid status transition raises error."""
        with pytest.raises(ValueError, match="Invalid transition"):
            sample_task_node.transition_to(TaskStatus.COMPLETED)

    def test_transition_with_additional_updates(self, sample_task_node: TaskNode):
        """Test transition with additional field updates."""
        ready_node = sample_task_node.transition_to(TaskStatus.READY)
        new_node = ready_node.transition_to(
            TaskStatus.EXECUTING,
            result="Partial result",
            metadata={"key": "value"}
        )

        assert new_node.status == TaskStatus.EXECUTING
        assert new_node.result == "Partial result"
        assert new_node.metadata == {"key": "value"}
        assert new_node.version == 2  # Two transitions: PENDING->READY->EXECUTING

    def test_executing_status_sets_started_timestamp(self, sample_task_node: TaskNode):
        """Test that EXECUTING status automatically sets started_at."""
        ready_node = sample_task_node.transition_to(TaskStatus.READY)
        executing_node = ready_node.transition_to(TaskStatus.EXECUTING)

        assert executing_node.started_at is not None
        assert isinstance(executing_node.started_at, datetime)

    def test_completed_status_sets_completed_timestamp(self, sample_task_node: TaskNode):
        """Test that COMPLETED status automatically sets completed_at."""
        ready_node = sample_task_node.transition_to(TaskStatus.READY)
        executing_node = ready_node.transition_to(TaskStatus.EXECUTING)
        completed_node = executing_node.transition_to(TaskStatus.COMPLETED)

        assert completed_node.completed_at is not None
        assert isinstance(completed_node.completed_at, datetime)

    def test_failed_status_sets_completed_timestamp(self, sample_task_node: TaskNode):
        """Test that FAILED status automatically sets completed_at."""
        ready_node = sample_task_node.transition_to(TaskStatus.READY)
        executing_node = ready_node.transition_to(TaskStatus.EXECUTING)
        failed_node = executing_node.transition_to(TaskStatus.FAILED)

        assert failed_node.completed_at is not None
        assert isinstance(failed_node.completed_at, datetime)


class TestTaskNodeResultHandling:
    """Test TaskNode result and error handling methods."""

    def test_with_result_creates_completed_node(self, sample_task_node: TaskNode):
        """Test with_result creates completed node."""
        ready_node = sample_task_node.transition_to(TaskStatus.READY)
        executing_node = ready_node.transition_to(TaskStatus.EXECUTING)
        completed_node = executing_node.with_result("Task result")

        assert completed_node.status == TaskStatus.COMPLETED
        assert completed_node.result == "Task result"
        assert completed_node.completed_at is not None
        assert completed_node.version == executing_node.version + 1

    def test_with_result_and_metadata(self, sample_task_node: TaskNode):
        """Test with_result can include metadata."""
        ready_node = sample_task_node.transition_to(TaskStatus.READY)
        executing_node = ready_node.transition_to(TaskStatus.EXECUTING)
        completed_node = executing_node.with_result(
            "Task result",
            metadata={"execution_time": 123.45}
        )

        assert completed_node.result == "Task result"
        assert "execution_time" in completed_node.metadata
        assert completed_node.metadata["execution_time"] == 123.45

    def test_with_error_creates_failed_node(self, sample_task_node: TaskNode):
        """Test with_error creates failed node."""
        ready_node = sample_task_node.transition_to(TaskStatus.READY)
        executing_node = ready_node.transition_to(TaskStatus.EXECUTING)
        failed_node = executing_node.with_error("Error message")

        assert failed_node.status == TaskStatus.FAILED
        assert failed_node.error == "Error message"
        assert failed_node.completed_at is not None
        assert failed_node.version == executing_node.version + 1

    def test_with_error_and_metadata(self, sample_task_node: TaskNode):
        """Test with_error can include metadata."""
        ready_node = sample_task_node.transition_to(TaskStatus.READY)
        executing_node = ready_node.transition_to(TaskStatus.EXECUTING)
        failed_node = executing_node.with_error(
            "Error message",
            metadata={"error_code": 500}
        )

        assert failed_node.error == "Error message"
        assert "error_code" in failed_node.metadata
        assert failed_node.metadata["error_code"] == 500


class TestTaskNodeRelationships:
    """Test TaskNode relationship management."""

    def test_add_child(self, sample_task_node: TaskNode):
        """Test adding child to node."""
        child_id = "child-123"
        parent_with_child = sample_task_node.add_child(child_id)

        assert child_id in parent_with_child.children
        assert child_id not in sample_task_node.children  # Original unchanged
        assert parent_with_child.version == 1

    def test_add_duplicate_child_no_change(self, sample_task_node: TaskNode):
        """Test adding duplicate child returns same instance."""
        child_id = "child-123"
        parent_with_child = sample_task_node.add_child(child_id)
        parent_same = parent_with_child.add_child(child_id)

        assert parent_same is parent_with_child  # Same instance
        assert len(parent_same.children) == 1

    def test_remove_child(self, sample_task_node: TaskNode):
        """Test removing child from node."""
        child_id = "child-123"
        parent_with_child = sample_task_node.add_child(child_id)
        parent_without_child = parent_with_child.remove_child(child_id)

        assert child_id not in parent_without_child.children
        assert child_id in parent_with_child.children  # Previous version unchanged
        assert parent_without_child.version == 2

    def test_remove_nonexistent_child_no_change(self, sample_task_node: TaskNode):
        """Test removing nonexistent child returns same instance."""
        parent_same = sample_task_node.remove_child("nonexistent-child")

        assert parent_same is sample_task_node  # Same instance

    def test_add_dependency(self, sample_task_node: TaskNode):
        """Test adding dependency to node."""
        dep_id = "dep-123"
        node_with_dep = sample_task_node.add_dependency(dep_id)

        assert dep_id in node_with_dep.dependencies
        assert dep_id not in sample_task_node.dependencies  # Original unchanged
        assert node_with_dep.version == 1

    def test_remove_dependency(self, sample_task_node: TaskNode):
        """Test removing dependency from node."""
        dep_id = "dep-123"
        node_with_dep = sample_task_node.add_dependency(dep_id)
        node_without_dep = node_with_dep.remove_dependency(dep_id)

        assert dep_id not in node_without_dep.dependencies
        assert dep_id in node_with_dep.dependencies  # Previous version unchanged
        assert node_without_dep.version == 2


class TestTaskNodeNodeTypeHandling:
    """Test TaskNode node_type setting and validation."""

    def test_set_node_type(self, sample_task_node: TaskNode):
        """Test setting node_type on task."""
        node_with_type = sample_task_node.set_node_type(NodeType.EXECUTE)

        assert node_with_type.node_type == NodeType.EXECUTE
        assert sample_task_node.node_type is None  # Original unchanged
        assert node_with_type.version == 1

    def test_retrieve_task_allows_any_node_type(self):
        """Test that RETRIEVE tasks can be set to any node_type."""
        retrieve_node = TaskNode(
            goal="Search query",
            task_type=TaskType.RETRIEVE
        )

        # Both should work - atomizer decides the appropriate type
        execute_node = retrieve_node.set_node_type(NodeType.EXECUTE)
        assert execute_node.node_type == NodeType.EXECUTE

        plan_node = retrieve_node.set_node_type(NodeType.PLAN)
        assert plan_node.node_type == NodeType.PLAN


class TestTaskNodeProperties:
    """Test TaskNode convenience properties."""

    def test_is_atomic_property(self):
        """Test is_atomic property."""
        node = TaskNode(goal="Test", task_type=TaskType.THINK)
        assert node.is_atomic is False  # node_type is None

        atomic_node = node.set_node_type(NodeType.EXECUTE)
        assert atomic_node.is_atomic is True

        composite_node = node.set_node_type(NodeType.PLAN)
        assert composite_node.is_atomic is False

    def test_is_composite_property(self):
        """Test is_composite property."""
        node = TaskNode(goal="Test", task_type=TaskType.THINK)
        assert node.is_composite is False  # node_type is None

        plan_node = node.set_node_type(NodeType.PLAN)
        assert plan_node.is_composite is True

        execute_node = node.set_node_type(NodeType.EXECUTE)
        assert execute_node.is_composite is False

    def test_is_root_property(self, sample_task_node: TaskNode):
        """Test is_root property."""
        assert sample_task_node.is_root is True  # No parent

        child_node = TaskNode(
            goal="Child task",
            task_type=TaskType.THINK,
            parent_id="parent-123"
        )
        assert child_node.is_root is False

    def test_is_leaf_property(self, sample_task_node: TaskNode):
        """Test is_leaf property."""
        assert sample_task_node.is_leaf is True  # No children

        parent_with_child = sample_task_node.add_child("child-123")
        assert parent_with_child.is_leaf is False

    def test_has_dependencies_property(self, sample_task_node: TaskNode):
        """Test has_dependencies property."""
        assert sample_task_node.has_dependencies is False

        node_with_dep = sample_task_node.add_dependency("dep-123")
        assert node_with_dep.has_dependencies is True

    def test_execution_duration_property(self, sample_task_node: TaskNode):
        """Test execution_duration property calculation."""
        assert sample_task_node.execution_duration is None  # No timestamps

        # Set started_at
        ready_node = sample_task_node.transition_to(TaskStatus.READY)
        started_node = ready_node.transition_to(
            TaskStatus.EXECUTING,
            started_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        )
        assert started_node.execution_duration is None  # No completed_at

        # Set completed_at
        completed_node = started_node.transition_to(
            TaskStatus.COMPLETED,
            completed_at=datetime(2024, 1, 1, 10, 0, 5, tzinfo=UTC)
        )
        assert completed_node.execution_duration == 5.0  # 5 seconds


class TestTaskNodeStringRepresentation:
    """Test TaskNode string representations."""

    def test_str_representation(self, sample_task_node: TaskNode):
        """Test __str__ method."""
        str_repr = str(sample_task_node)
        assert sample_task_node.task_id[:8] in str_repr
        assert "Test goal for sample task" in str_repr

    def test_str_with_node_type(self, sample_task_node: TaskNode):
        """Test __str__ method with node_type."""
        execute_node = sample_task_node.set_node_type(NodeType.EXECUTE)
        str_repr = str(execute_node)
        assert "(EXECUTE)" in str_repr

    def test_repr_representation(self, sample_task_node: TaskNode):
        """Test __repr__ method."""
        repr_str = repr(sample_task_node)
        assert "TaskNode(" in repr_str
        assert f"task_id='{sample_task_node.task_id}'" in repr_str
        assert f"task_type={sample_task_node.task_type}" in repr_str
        assert f"status={sample_task_node.status}" in repr_str


class TestTaskNodeMetadata:
    """Test TaskNode metadata handling."""

    def test_update_metadata(self, sample_task_node: TaskNode):
        """Test updating metadata."""
        updated_node = sample_task_node.update_metadata(
            key1="value1",
            key2=42
        )

        assert updated_node.metadata["key1"] == "value1"
        assert updated_node.metadata["key2"] == 42
        assert len(sample_task_node.metadata) == 0  # Original unchanged
        assert updated_node.version == 1

    def test_update_metadata_merge(self, sample_task_node: TaskNode):
        """Test metadata merging."""
        node_with_meta = sample_task_node.update_metadata(existing="old")
        updated_node = node_with_meta.update_metadata(
            existing="new",
            additional="extra"
        )

        assert updated_node.metadata["existing"] == "new"  # Overwritten
        assert updated_node.metadata["additional"] == "extra"  # Added
        assert updated_node.version == 2
