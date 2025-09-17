"""
Unit tests for ArtifactService.

Tests artifact operations and ResultEnvelope integration.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

from src.roma.application.services.artifact_service import ArtifactService
from src.roma.domain.entities.artifacts.file_artifact import FileArtifact
from src.roma.domain.value_objects.result_envelope import ResultEnvelope, ExecutionMetrics
from src.roma.domain.value_objects.agent_type import AgentType
from src.roma.domain.value_objects.agent_responses import ExecutorResult
from src.roma.domain.value_objects.media_type import MediaType
from src.roma.infrastructure.storage.local_storage import LocalFileStorage
from src.roma.infrastructure.storage.storage_interface import StorageConfig


class TestArtifactService:
    """Test ArtifactService functionality."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def storage_config(self, temp_storage_dir):
        """Create storage configuration."""
        return StorageConfig.from_mount_path(temp_storage_dir)

    @pytest.fixture
    def storage(self, storage_config):
        """Create storage instance."""
        return LocalFileStorage(storage_config)

    @pytest.fixture
    def artifact_service(self, storage):
        """Create artifact service."""
        return ArtifactService(storage)

    @pytest.fixture
    async def initialized_service(self, artifact_service):
        """Create initialized artifact service."""
        await artifact_service.initialize()
        return artifact_service

    @pytest.fixture
    def sample_file_artifact(self, temp_storage_dir):
        """Create sample file artifact."""
        # Create a test file
        test_file = Path(temp_storage_dir) / "test.txt"
        test_file.write_text("Sample test content")

        return FileArtifact.from_path(
            name="test_document",
            file_path=str(test_file),
            task_id="test-task-123",
            metadata={"source": "test"}
        )

    @pytest.fixture
    def sample_result_envelope(self, sample_file_artifact):
        """Create sample result envelope with artifacts."""
        metrics = ExecutionMetrics(
            execution_time=1.5,
            tokens_used=150,
            model_calls=1,
            cost_estimate=0.01
        )

        executor_result = ExecutorResult(
            result="Task completed successfully",
            sources=["test_source"],
            success=True,
            confidence=0.95
        )

        return ResultEnvelope.create_success(
            result=executor_result,
            task_id="test-task-123",
            execution_id="test-exec-456",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=[sample_file_artifact],
            output_text="Sample output"
        )

    @pytest.mark.asyncio
    async def test_initialization(self, artifact_service, temp_storage_dir):
        """Test artifact service initialization."""
        assert not artifact_service._initialized

        await artifact_service.initialize()

        assert artifact_service._initialized

        # Check that directories were created
        artifacts_dir = Path(temp_storage_dir) / "artifacts"
        temp_dir = Path(temp_storage_dir) / "temp"

        assert artifacts_dir.parent.exists()
        assert temp_dir.parent.exists()

    @pytest.mark.asyncio
    async def test_double_initialization(self, artifact_service):
        """Test that double initialization is safe."""
        await artifact_service.initialize()
        assert artifact_service._initialized

        # Second initialization should be safe
        await artifact_service.initialize()
        assert artifact_service._initialized

    @pytest.mark.asyncio
    async def test_store_envelope_artifacts_empty(self, initialized_service):
        """Test storing envelope with no artifacts."""
        metrics = ExecutionMetrics(execution_time=1.0)
        result = ExecutorResult(result="No artifacts")

        envelope = ResultEnvelope.create_success(
            result=result,
            task_id="test-task",
            execution_id="test-exec",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=[]  # No artifacts
        )

        storage_refs = await initialized_service.store_envelope_artifacts("exec-123", envelope)

        assert storage_refs == []

    @pytest.mark.asyncio
    async def test_store_envelope_artifacts_success(self, initialized_service, sample_result_envelope):
        """Test successful artifact storage from envelope."""
        storage_refs = await initialized_service.store_envelope_artifacts(
            "exec-123", sample_result_envelope
        )

        assert len(storage_refs) == 1
        assert "executions/exec-123/tasks/test-task-123/" in storage_refs[0]
        assert "test_document" in storage_refs[0]

    @pytest.mark.asyncio
    async def test_store_single_artifact(self, initialized_service, sample_file_artifact):
        """Test storing a single artifact."""
        storage_ref = await initialized_service._store_single_artifact(
            "exec-123", "task-456", sample_file_artifact
        )

        assert "executions/exec-123/tasks/task-456/" in storage_ref
        assert sample_file_artifact.artifact_id in storage_ref
        assert "test_document" in storage_ref

    @pytest.mark.asyncio
    async def test_artifact_key_generation(self, initialized_service, sample_file_artifact):
        """Test artifact storage key generation."""
        key = initialized_service._generate_artifact_key(
            "exec-123", "task-456", sample_file_artifact
        )

        expected_parts = [
            "executions/exec-123/tasks/task-456/",
            sample_file_artifact.artifact_id,
            "test_document"
        ]

        for part in expected_parts:
            assert part in key

    def test_sanitize_filename(self, artifact_service):
        """Test filename sanitization."""
        # Test normal filename
        assert artifact_service._sanitize_filename("test_file.txt") == "test_file.txt"

        # Test unsafe characters
        assert artifact_service._sanitize_filename("test/file\\name?.txt") == "test_file_name_.txt"

        # Test long filename
        long_name = "a" * 100
        sanitized = artifact_service._sanitize_filename(long_name)
        assert len(sanitized) <= 50

        # Test empty filename
        assert artifact_service._sanitize_filename("") == "artifact"

    def test_get_extension_for_media_type(self, artifact_service):
        """Test media type to extension mapping."""
        assert artifact_service._get_extension_for_media_type(MediaType.TEXT) == ".txt"
        assert artifact_service._get_extension_for_media_type(MediaType.IMAGE) == ".png"
        assert artifact_service._get_extension_for_media_type(MediaType.FILE) == ""

    @pytest.mark.asyncio
    async def test_retrieve_artifact(self, initialized_service, sample_result_envelope):
        """Test artifact retrieval."""
        # First store artifacts
        storage_refs = await initialized_service.store_envelope_artifacts(
            "exec-123", sample_result_envelope
        )

        assert len(storage_refs) == 1
        storage_ref = storage_refs[0]

        # Retrieve as bytes
        content = await initialized_service.retrieve_artifact(storage_ref)
        assert content is not None
        assert isinstance(content, bytes)

        # Retrieve as text
        text_content = await initialized_service.retrieve_artifact(storage_ref, as_text=True)
        assert text_content is not None
        assert isinstance(text_content, str)

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_artifact(self, initialized_service):
        """Test retrieving nonexistent artifact."""
        content = await initialized_service.retrieve_artifact("nonexistent/path")
        assert content is None

        text_content = await initialized_service.retrieve_artifact("nonexistent/path", as_text=True)
        assert text_content is None

    @pytest.mark.asyncio
    async def test_list_execution_artifacts(self, initialized_service, sample_result_envelope):
        """Test listing artifacts for execution."""
        # Initially empty
        artifacts = await initialized_service.list_execution_artifacts("exec-123")
        assert artifacts == []

        # Store artifacts
        await initialized_service.store_envelope_artifacts("exec-123", sample_result_envelope)

        # List artifacts
        artifacts = await initialized_service.list_execution_artifacts("exec-123")
        assert len(artifacts) == 1
        assert "executions/exec-123/" in artifacts[0]

    @pytest.mark.asyncio
    async def test_get_artifact_metadata(self, initialized_service, sample_result_envelope):
        """Test getting artifact metadata."""
        # Store artifact
        storage_refs = await initialized_service.store_envelope_artifacts(
            "exec-123", sample_result_envelope
        )
        storage_ref = storage_refs[0]

        # Get metadata
        metadata = await initialized_service.get_artifact_metadata(storage_ref)

        assert metadata["exists"] is True
        assert "storage_key" in metadata
        assert "size_bytes" in metadata
        assert "full_path" in metadata

        # Test nonexistent artifact
        metadata = await initialized_service.get_artifact_metadata("nonexistent")
        assert metadata["exists"] is False

    @pytest.mark.asyncio
    async def test_cleanup_execution_artifacts(self, initialized_service, sample_result_envelope):
        """Test cleaning up execution artifacts."""
        # Store artifacts
        await initialized_service.store_envelope_artifacts("exec-123", sample_result_envelope)

        # Verify artifacts exist
        artifacts_before = await initialized_service.list_execution_artifacts("exec-123")
        assert len(artifacts_before) == 1

        # Cleanup
        deleted_count = await initialized_service.cleanup_execution_artifacts("exec-123")
        assert deleted_count == 1

        # Verify artifacts are gone
        artifacts_after = await initialized_service.list_execution_artifacts("exec-123")
        assert len(artifacts_after) == 0

    @pytest.mark.asyncio
    async def test_create_file_artifact_from_storage(self, initialized_service, sample_result_envelope, temp_storage_dir):
        """Test creating FileArtifact from storage."""
        # Store artifacts first
        storage_refs = await initialized_service.store_envelope_artifacts(
            "exec-123", sample_result_envelope
        )
        storage_ref = storage_refs[0]

        # Create artifact from storage
        artifact = await initialized_service.create_file_artifact_from_storage(
            storage_ref, "recreated_artifact", "task-456"
        )

        assert artifact is not None
        assert artifact.name == "recreated_artifact"
        assert artifact.task_id == "task-456"
        assert storage_ref in artifact.metadata.get("storage_key", "")

    @pytest.mark.asyncio
    async def test_create_file_artifact_from_nonexistent_storage(self, initialized_service):
        """Test creating FileArtifact from nonexistent storage."""
        artifact = await initialized_service.create_file_artifact_from_storage(
            "nonexistent/path", "test", "task"
        )
        assert artifact is None

    def test_get_storage_stats(self, artifact_service, temp_storage_dir):
        """Test getting storage statistics."""
        stats = artifact_service.get_storage_stats()

        assert "storage_type" in stats
        assert "mount_path" in stats
        assert "artifacts_path" in stats
        assert "temp_path" in stats
        assert "initialized" in stats
        assert "config" in stats

        assert stats["storage_type"] == "LocalFileStorage"
        assert temp_storage_dir in stats["mount_path"]
        assert not stats["initialized"]  # Not initialized yet

    @pytest.mark.asyncio
    async def test_artifact_storage_error_handling(self, initialized_service):
        """Test error handling during artifact storage."""
        # Create mock artifact that will fail
        mock_artifact = Mock()
        mock_artifact.name = "failing_artifact"
        mock_artifact.artifact_id = "fail-123"
        mock_artifact.task_id = "task-123"
        mock_artifact.media_type = MediaType.TEXT
        mock_artifact.metadata = {}
        mock_artifact.created_at.isoformat.return_value = "2023-01-01T00:00:00"
        mock_artifact.get_content = AsyncMock(return_value=None)  # Will cause error

        metrics = ExecutionMetrics(execution_time=1.0)
        result = ExecutorResult(result="Test")

        envelope = ResultEnvelope.create_success(
            result=result,
            task_id="task-123",
            execution_id="exec-123",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=[mock_artifact]
        )

        # Should handle error gracefully
        storage_refs = await initialized_service.store_envelope_artifacts("exec-123", envelope)
        assert storage_refs == []  # Failed to store, but didn't crash

    @pytest.mark.asyncio
    async def test_multiple_artifacts_in_envelope(self, initialized_service, temp_storage_dir):
        """Test storing multiple artifacts in one envelope."""
        # Create multiple test files
        test_files = []
        artifacts = []

        for i in range(3):
            test_file = Path(temp_storage_dir) / f"test_{i}.txt"
            test_file.write_text(f"Content {i}")
            test_files.append(test_file)

            artifact = FileArtifact.from_path(
                name=f"document_{i}",
                file_path=str(test_file),
                task_id="task-123"
            )
            artifacts.append(artifact)

        # Create envelope with multiple artifacts
        metrics = ExecutionMetrics(execution_time=2.0)
        result = ExecutorResult(result="Multiple artifacts test")

        envelope = ResultEnvelope.create_success(
            result=result,
            task_id="task-123",
            execution_id="exec-123",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=artifacts
        )

        # Store all artifacts
        storage_refs = await initialized_service.store_envelope_artifacts("exec-123", envelope)

        assert len(storage_refs) == 3

        # Verify all artifacts can be retrieved
        for storage_ref in storage_refs:
            content = await initialized_service.retrieve_artifact(storage_ref, as_text=True)
            assert content is not None
            assert "Content" in content


class TestArtifactServiceEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage for testing edge cases."""
        storage = Mock()
        storage.mount_path = Path("/mock/path")
        storage.config = Mock()
        storage.config.model_dump.return_value = {"mount_path": "/mock/path"}
        return storage

    @pytest.fixture
    def artifact_service_with_mock(self, mock_storage):
        """Create artifact service with mock storage."""
        return ArtifactService(mock_storage)

    def test_constructor(self, mock_storage):
        """Test artifact service constructor."""
        service = ArtifactService(mock_storage)
        assert service.storage == mock_storage
        assert not service._initialized

    @pytest.mark.asyncio
    async def test_initialization_directory_creation(self):
        """Test initialization creates directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig.from_mount_path(temp_dir)
            storage = LocalFileStorage(config)
            service = ArtifactService(storage)

            await service.initialize()

            artifacts_path = storage.get_artifacts_path("")
            temp_path = storage.get_temp_path("")

            assert artifacts_path.parent.exists()
            assert temp_path.parent.exists()

    @pytest.mark.asyncio
    async def test_store_artifacts_partial_failure(self, artifact_service_with_mock, mock_storage):
        """Test partial failure when storing multiple artifacts."""
        # Mock storage methods
        mock_storage.get_artifacts_path.return_value = Path("/mock/artifacts")
        mock_storage.get_temp_path.return_value = Path("/mock/temp")

        # Create mock artifacts - one succeeds, one fails
        good_artifact = Mock()
        good_artifact.name = "good"
        good_artifact.artifact_id = "good-123"
        good_artifact.task_id = "task"
        good_artifact.media_type = MediaType.TEXT
        good_artifact.metadata = {}
        good_artifact.created_at.isoformat.return_value = "2023-01-01"
        good_artifact.get_content = AsyncMock(return_value=b"content")

        bad_artifact = Mock()
        bad_artifact.name = "bad"
        bad_artifact.artifact_id = "bad-123"
        bad_artifact.task_id = "task"
        bad_artifact.media_type = MediaType.TEXT
        bad_artifact.metadata = {}
        bad_artifact.created_at.isoformat.return_value = "2023-01-01"
        bad_artifact.get_content = AsyncMock(side_effect=Exception("Fail"))

        # Mock storage operations
        mock_storage.put_text = AsyncMock()
        mock_storage.put_text.side_effect = ["good_path", Exception("Storage fail")]

        # Create envelope with both artifacts
        metrics = ExecutionMetrics(execution_time=1.0)
        result = ExecutorResult(result="Test")

        envelope = ResultEnvelope.create_success(
            result=result,
            task_id="task-123",
            execution_id="exec-123",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=[good_artifact, bad_artifact]
        )

        # Initialize and store
        await artifact_service_with_mock.initialize()
        storage_refs = await artifact_service_with_mock.store_envelope_artifacts("exec", envelope)

        # Should continue despite failure
        assert len(storage_refs) == 1  # Only the good one succeeded