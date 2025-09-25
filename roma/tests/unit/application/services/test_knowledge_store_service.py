"""
Tests for KnowledgeStoreService.
"""

import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from roma.application.services.artifact_service import ArtifactService
from roma.application.services.knowledge_store_service import KnowledgeStoreService
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.result_envelope import ResultEnvelope
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


class TestKnowledgeStoreService:
    """Test KnowledgeStoreService."""

    @pytest.fixture
    def mock_artifact_service(self):
        """Create mock artifact service."""
        service = Mock(spec=ArtifactService)
        service.store_envelope_artifacts = AsyncMock(return_value=["artifact1", "artifact2"])
        return service

    @pytest.fixture
    def knowledge_store(self, mock_artifact_service):
        """Create knowledge store service."""
        return KnowledgeStoreService(mock_artifact_service)

    @pytest.fixture
    def sample_task_node(self):
        """Create sample task node."""
        return TaskNode(
            task_id=str(uuid4()),
            goal="Test task goal",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

    @pytest.mark.asyncio
    async def test_add_record_without_envelope(self, knowledge_store, sample_task_node):
        """Test adding record without result envelope."""
        record = await knowledge_store.add_or_update_record(sample_task_node)

        assert record.task_id == sample_task_node.task_id
        assert record.goal == sample_task_node.goal
        assert record.task_type == sample_task_node.task_type
        assert record.status == sample_task_node.status
        assert len(record.artifacts) == 0

        # Verify it's stored
        retrieved = await knowledge_store.get_record(sample_task_node.task_id)
        assert retrieved == record

    @pytest.mark.asyncio
    async def test_add_record_with_envelope(self, knowledge_store, sample_task_node, mock_artifact_service):
        """Test adding record with result envelope."""
        # Create a real ResultEnvelope instead of Mock
        from roma.domain.value_objects.agent_responses import ExecutorResult
        from roma.domain.value_objects.agent_type import AgentType
        from roma.domain.value_objects.result_envelope import ExecutionMetrics

        # Create execution metrics
        metrics = ExecutionMetrics(
            execution_time=1.0,
            tokens_used=100,
            model_calls=1,
            cost_estimate=0.01
        )

        # Create executor result
        result = ExecutorResult(result="Test result")

        # Create some mock artifacts
        import tempfile

        from roma.domain.entities.artifacts.file_artifact import FileArtifact

        # Create temporary files for artifacts
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp1:
            temp1.write("Mock artifact 1")
            temp1_path = temp1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp2:
            temp2.write("Mock artifact 2")
            temp2_path = temp2.name

        # Create file artifacts
        artifact1 = FileArtifact.from_path(
            name="artifact1",
            file_path=temp1_path,
            task_id=sample_task_node.task_id,
            metadata={"source": "test"}
        )

        artifact2 = FileArtifact.from_path(
            name="artifact2",
            file_path=temp2_path,
            task_id=sample_task_node.task_id,
            metadata={"source": "test"}
        )

        # Create envelope with artifacts
        envelope = ResultEnvelope.create_success(
            result=result,
            task_id=sample_task_node.task_id,
            execution_id="test_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=[artifact1, artifact2]
        )

        # Mock the artifact service to return expected refs
        mock_artifact_service.store_envelope_artifacts.return_value = ["artifact1", "artifact2"]

        record = await knowledge_store.add_or_update_record(sample_task_node, envelope)

        # Verify artifacts were stored
        mock_artifact_service.store_envelope_artifacts.assert_called_once_with(
            envelope
        )
        assert "artifact1" in record.artifacts
        assert "artifact2" in record.artifacts

        # Cleanup temp files
        import os
        os.unlink(temp1_path)
        os.unlink(temp2_path)

    @pytest.mark.asyncio
    async def test_update_existing_record(self, knowledge_store, sample_task_node):
        """Test updating existing record."""
        # Add initial record
        initial_record = await knowledge_store.add_or_update_record(sample_task_node)

        # Update task status and add again - use valid transition
        updated_node = sample_task_node.transition_to(TaskStatus.READY).transition_to(TaskStatus.EXECUTING).transition_to(TaskStatus.COMPLETED)
        updated_record = await knowledge_store.add_or_update_record(updated_node)

        assert updated_record.status == TaskStatus.COMPLETED
        assert updated_record.updated_at > initial_record.updated_at
        assert updated_record.task_id == initial_record.task_id

        # Verify only one record exists
        retrieved = await knowledge_store.get_record(sample_task_node.task_id)
        assert retrieved.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_nonexistent_record(self, knowledge_store):
        """Test getting non-existent record returns None."""
        result = await knowledge_store.get_record("nonexistent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_child_records(self, knowledge_store):
        """Test getting child records."""
        parent_id = str(uuid4())
        child1_id = str(uuid4())
        child2_id = str(uuid4())

        # Create parent task
        parent_node = TaskNode(
            task_id=parent_id,
            goal="Parent task",
            task_type=TaskType.THINK,
            status=TaskStatus.EXECUTING
        )

        # Create child tasks
        child1_node = TaskNode(
            task_id=child1_id,
            goal="Child task 1",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.COMPLETED,
            parent_id=parent_id
        )

        child2_node = TaskNode(
            task_id=child2_id,
            goal="Child task 2",
            task_type=TaskType.WRITE,
            status=TaskStatus.COMPLETED,
            parent_id=parent_id
        )

        # Add records
        await knowledge_store.add_or_update_record(parent_node)
        await knowledge_store.add_or_update_record(child1_node)
        await knowledge_store.add_or_update_record(child2_node)

        # Get child records
        children = await knowledge_store.get_child_records(parent_id)

        assert len(children) == 2
        child_ids = {child.task_id for child in children}
        assert child1_id in child_ids
        assert child2_id in child_ids

    @pytest.mark.asyncio
    async def test_get_records_by_status(self, knowledge_store):
        """Test getting records by status."""
        # Create tasks with different statuses
        pending_node = TaskNode(
            task_id=str(uuid4()),
            goal="Pending task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        completed_node = TaskNode(
            task_id=str(uuid4()),
            goal="Completed task",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.COMPLETED
        )

        failed_node = TaskNode(
            task_id=str(uuid4()),
            goal="Failed task",
            task_type=TaskType.WRITE,
            status=TaskStatus.FAILED
        )

        # Add records
        await knowledge_store.add_or_update_record(pending_node)
        await knowledge_store.add_or_update_record(completed_node)
        await knowledge_store.add_or_update_record(failed_node)

        # Test getting by status
        pending_records = await knowledge_store.get_records_by_status(TaskStatus.PENDING)
        completed_records = await knowledge_store.get_records_by_status(TaskStatus.COMPLETED)
        failed_records = await knowledge_store.get_records_by_status(TaskStatus.FAILED)

        assert len(pending_records) == 1
        assert len(completed_records) == 1
        assert len(failed_records) == 1

        assert pending_records[0].task_id == pending_node.task_id
        assert completed_records[0].task_id == completed_node.task_id
        assert failed_records[0].task_id == failed_node.task_id

    @pytest.mark.asyncio
    async def test_add_child_relationship(self, knowledge_store):
        """Test adding child relationship."""
        parent_id = str(uuid4())
        child_id = str(uuid4())

        # Create parent record
        parent_node = TaskNode(
            task_id=parent_id,
            goal="Parent task",
            task_type=TaskType.THINK,
            status=TaskStatus.EXECUTING
        )
        await knowledge_store.add_or_update_record(parent_node)

        # Add child relationship
        success = await knowledge_store.add_child_relationship(parent_id, child_id)
        assert success

        # Verify relationship was added
        parent_record = await knowledge_store.get_record(parent_id)
        assert parent_record is not None
        assert child_id in parent_record.child_task_ids

    @pytest.mark.asyncio
    async def test_add_child_relationship_nonexistent_parent(self, knowledge_store):
        """Test adding child relationship with non-existent parent."""
        success = await knowledge_store.add_child_relationship("nonexistent", "child_id")
        assert not success

    @pytest.mark.asyncio
    async def test_get_completed_records(self, knowledge_store):
        """Test getting completed records."""
        # Create completed task
        completed_node = TaskNode(
            task_id=str(uuid4()),
            goal="Completed task",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED
        )
        await knowledge_store.add_or_update_record(completed_node)

        # Create pending task
        pending_node = TaskNode(
            task_id=str(uuid4()),
            goal="Pending task",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.PENDING
        )
        await knowledge_store.add_or_update_record(pending_node)

        completed_records = await knowledge_store.get_completed_records()
        assert len(completed_records) == 1
        assert completed_records[0].task_id == completed_node.task_id

    @pytest.mark.asyncio
    async def test_get_failed_records(self, knowledge_store):
        """Test getting failed records."""
        # Create failed task
        failed_node = TaskNode(
            task_id=str(uuid4()),
            goal="Failed task",
            task_type=TaskType.THINK,
            status=TaskStatus.FAILED
        )
        await knowledge_store.add_or_update_record(failed_node)

        # Create completed task
        completed_node = TaskNode(
            task_id=str(uuid4()),
            goal="Completed task",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.COMPLETED
        )
        await knowledge_store.add_or_update_record(completed_node)

        failed_records = await knowledge_store.get_failed_records()
        assert len(failed_records) == 1
        assert failed_records[0].task_id == failed_node.task_id

    @pytest.mark.asyncio
    async def test_lru_cache_functionality(self, knowledge_store):
        """Test LRU cache works correctly."""
        # Add record
        node = TaskNode(
            task_id=str(uuid4()),
            goal="Test task",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED
        )
        await knowledge_store.add_or_update_record(node)

        # First access - should be cache miss
        record1 = await knowledge_store.get_record(node.task_id)
        assert record1 is not None

        # Second access - should be cache hit
        record2 = await knowledge_store.get_record(node.task_id)
        assert record2 == record1

        # Verify cache statistics
        stats = await knowledge_store.get_summary_stats()
        assert stats["cache_hit_rate"] > 0

    @pytest.mark.asyncio
    async def test_get_summary_stats(self, knowledge_store):
        """Test getting summary statistics."""
        # Add various records
        completed_node = TaskNode(
            task_id=str(uuid4()),
            goal="Completed task",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED
        )

        pending_node = TaskNode(
            task_id=str(uuid4()),
            goal="Pending task",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.PENDING
        )

        await knowledge_store.add_or_update_record(completed_node)
        await knowledge_store.add_or_update_record(pending_node)

        # Get some records to generate cache activity
        await knowledge_store.get_record(completed_node.task_id)
        await knowledge_store.get_record(pending_node.task_id)

        stats = await knowledge_store.get_summary_stats()

        assert stats["total_records"] == 2
        assert stats["cache_size"] <= 2
        assert "status_breakdown" in stats
        assert "task_type_breakdown" in stats
        assert "cache_hit_rate" in stats

    @pytest.mark.asyncio
    async def test_clear(self, knowledge_store, sample_task_node):
        """Test clearing all records."""
        # Add record
        await knowledge_store.add_or_update_record(sample_task_node)

        # Verify it exists
        record = await knowledge_store.get_record(sample_task_node.task_id)
        assert record is not None

        # Clear all records
        await knowledge_store.clear()

        # Verify it's gone
        record = await knowledge_store.get_record(sample_task_node.task_id)
        assert record is None

        # Verify stats are reset
        stats = await knowledge_store.get_summary_stats()
        assert stats["total_records"] == 0
        assert stats["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self, knowledge_store):
        """Test concurrent access is thread-safe."""
        node_ids = [str(uuid4()) for _ in range(10)]

        async def add_record(task_id: str):
            node = TaskNode(
                task_id=task_id,
                goal=f"Task {task_id}",
                task_type=TaskType.THINK,
                status=TaskStatus.COMPLETED
            )
            return await knowledge_store.add_or_update_record(node)

        # Add records concurrently
        records = await asyncio.gather(*[add_record(task_id) for task_id in node_ids])

        # Verify all records were added
        assert len(records) == 10
        for i, record in enumerate(records):
            assert record.task_id == node_ids[i]

        # Verify all records can be retrieved
        retrieved_records = await asyncio.gather(*[
            knowledge_store.get_record(task_id) for task_id in node_ids
        ])

        assert len(retrieved_records) == 10
        assert all(record is not None for record in retrieved_records)

    @pytest.mark.asyncio
    async def test_artifact_service_error_handling(self, knowledge_store, sample_task_node, mock_artifact_service):
        """Test error handling when artifact service fails."""
        # Make artifact service fail
        mock_artifact_service.store_envelope_artifacts.side_effect = Exception("Storage failed")

        # Create a real ResultEnvelope with artifacts
        import tempfile

        from roma.domain.entities.artifacts.file_artifact import FileArtifact
        from roma.domain.value_objects.agent_responses import ExecutorResult
        from roma.domain.value_objects.agent_type import AgentType
        from roma.domain.value_objects.result_envelope import ExecutionMetrics

        # Create temporary file for artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp:
            temp.write("Mock artifact content")
            temp_path = temp.name

        # Create file artifact
        artifact = FileArtifact.from_path(
            name="mock_artifact",
            file_path=temp_path,
            task_id=sample_task_node.task_id,
            metadata={"source": "test"}
        )

        # Create execution metrics and result
        metrics = ExecutionMetrics(
            execution_time=1.0,
            tokens_used=50,
            model_calls=1,
            cost_estimate=0.005
        )
        result = ExecutorResult(result="Test result")

        # Create envelope with artifact
        envelope = ResultEnvelope.create_success(
            result=result,
            task_id=sample_task_node.task_id,
            execution_id="test_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=[artifact]
        )

        # Should still create record without artifacts due to storage error
        record = await knowledge_store.add_or_update_record(sample_task_node, envelope)

        assert record.task_id == sample_task_node.task_id
        assert len(record.artifacts) == 0  # No artifacts due to error

        # Cleanup temp file
        import os
        os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_cache_eviction(self, mock_artifact_service):
        """Test cache eviction when max size is reached."""
        # Create store with small cache size
        knowledge_store = KnowledgeStoreService(mock_artifact_service)
        knowledge_store._cache_max_size = 2

        # Add 3 records (should evict first one)
        for i in range(3):
            node = TaskNode(
                task_id=str(uuid4()),
                goal=f"Task {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.COMPLETED
            )
            await knowledge_store.add_or_update_record(node)

        # Cache should only contain 2 items
        assert len(knowledge_store._cache) <= 2
