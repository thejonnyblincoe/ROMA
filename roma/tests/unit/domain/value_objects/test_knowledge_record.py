"""
Tests for KnowledgeRecord value object.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from roma.domain.value_objects.knowledge_record import KnowledgeRecord
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus


class TestKnowledgeRecord:
    """Test KnowledgeRecord value object."""

    def test_create_knowledge_record(self):
        """Test creating a new knowledge record."""
        task_id = str(uuid4())
        goal = "Test goal"
        task_type = TaskType.THINK
        status = TaskStatus.PENDING

        record = KnowledgeRecord.create(
            task_id=task_id,
            goal=goal,
            task_type=task_type,
            status=status
        )

        assert record.task_id == task_id
        assert record.goal == goal
        assert record.task_type == task_type
        assert record.status == status
        assert record.artifacts == []
        assert record.parent_task_id is None
        assert record.child_task_ids == []
        assert record.created_at is not None
        assert record.updated_at == record.created_at
        assert record.completed_at is None

    def test_create_with_parent_and_artifacts(self):
        """Test creating record with parent and artifacts."""
        task_id = str(uuid4())
        parent_id = str(uuid4())
        artifacts = ["artifact1", "artifact2"]

        record = KnowledgeRecord.create(
            task_id=task_id,
            goal="Test goal",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.PENDING,
            artifacts=artifacts,
            parent_task_id=parent_id
        )

        assert record.parent_task_id == parent_id
        assert record.artifacts == artifacts
        assert len(record.artifacts) == 2

    def test_create_completed_record(self):
        """Test creating a completed record sets completion time."""
        record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test goal",
            task_type=TaskType.WRITE,
            status=TaskStatus.COMPLETED
        )

        assert record.status == TaskStatus.COMPLETED
        assert record.completed_at is not None
        assert record.completed_at == record.updated_at

    def test_update_status(self):
        """Test updating record status."""
        record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test goal",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        # Update to executing
        updated = record.update_status(TaskStatus.EXECUTING)

        assert updated.status == TaskStatus.EXECUTING
        assert updated.updated_at > record.updated_at
        assert updated.completed_at is None
        assert updated.version > record.version

        # Update to completed
        completed = updated.update_status(TaskStatus.COMPLETED)

        assert completed.status == TaskStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.updated_at == completed.completed_at

    def test_add_child(self):
        """Test adding child task ID."""
        record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Parent task",
            task_type=TaskType.THINK,
            status=TaskStatus.EXECUTING
        )

        child_id = str(uuid4())
        updated = record.add_child(child_id)

        assert child_id in updated.child_task_ids
        assert len(updated.child_task_ids) == 1
        assert updated.updated_at > record.updated_at

    def test_add_duplicate_child(self):
        """Test adding duplicate child returns same instance."""
        child_id = str(uuid4())
        record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Parent task",
            task_type=TaskType.THINK,
            status=TaskStatus.EXECUTING
        )

        updated1 = record.add_child(child_id)
        updated2 = updated1.add_child(child_id)  # Duplicate

        assert updated1 == updated2  # Should be same instance
        assert len(updated2.child_task_ids) == 1

    def test_add_artifacts(self):
        """Test adding artifacts to record."""
        record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test task",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.COMPLETED,
            artifacts=["existing1"]
        )

        new_artifacts = ["new1", "new2"]
        updated = record.add_artifacts(new_artifacts)

        assert "existing1" in updated.artifacts
        assert "new1" in updated.artifacts
        assert "new2" in updated.artifacts
        assert len(updated.artifacts) == 3
        assert updated.updated_at > record.updated_at

    def test_add_duplicate_artifacts(self):
        """Test adding duplicate artifacts doesn't create duplicates."""
        record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test task",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.COMPLETED,
            artifacts=["artifact1"]
        )

        # Add existing artifact again
        updated = record.add_artifacts(["artifact1", "artifact2"])

        assert updated.artifacts.count("artifact1") == 1
        assert "artifact2" in updated.artifacts
        assert len(updated.artifacts) == 2

    def test_add_empty_artifacts(self):
        """Test adding empty artifacts list returns same instance."""
        record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test task",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED
        )

        updated = record.add_artifacts([])

        assert updated == record  # Should be same instance

    def test_is_completed(self):
        """Test completion check."""
        pending_record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        completed_record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test task",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED
        )

        assert not pending_record.is_completed()
        assert completed_record.is_completed()

    def test_has_artifacts(self):
        """Test artifact check."""
        empty_record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test task",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED
        )

        with_artifacts = empty_record.add_artifacts(["artifact1"])

        assert not empty_record.has_artifacts()
        assert with_artifacts.has_artifacts()

    def test_has_children(self):
        """Test children check."""
        parent_record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Parent task",
            task_type=TaskType.THINK,
            status=TaskStatus.EXECUTING
        )

        with_child = parent_record.add_child(str(uuid4()))

        assert not parent_record.has_children()
        assert with_child.has_children()

    def test_immutability(self):
        """Test that records are immutable."""
        record = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        # Field reassignment should raise errors due to frozen=True
        with pytest.raises(Exception):
            record.status = TaskStatus.COMPLETED

        with pytest.raises(Exception):
            record.task_id = "new_id"

    def test_thread_safety_isolation(self):
        """Test that updates create new instances (thread safety)."""
        original = KnowledgeRecord.create(
            task_id=str(uuid4()),
            goal="Test task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        # Multiple updates should create separate instances
        updated1 = original.update_status(TaskStatus.EXECUTING)
        updated2 = original.add_child(str(uuid4()))
        updated3 = original.add_artifacts(["artifact1"])

        # Original should be unchanged
        assert original.status == TaskStatus.PENDING
        assert len(original.child_task_ids) == 0
        assert len(original.artifacts) == 0

        # Updates should be different instances
        assert updated1 is not original
        assert updated2 is not original
        assert updated3 is not original
        assert updated1.status == TaskStatus.EXECUTING
        assert len(updated2.child_task_ids) == 1
        assert len(updated3.artifacts) == 1