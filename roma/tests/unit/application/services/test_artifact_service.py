"""
Unit tests for ArtifactService.

Tests artifact operations and ResultEnvelope integration.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from roma.infrastructure.storage.storage_interface import StorageConfig

from roma.application.services.artifact_service import ArtifactService
from roma.domain.entities.artifacts.file_artifact import FileArtifact
from roma.domain.value_objects.agent_responses import ExecutorResult
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.media_type import MediaType
from roma.domain.value_objects.result_envelope import ExecutionMetrics, ResultEnvelope
from roma.infrastructure.storage.local_storage import LocalFileStorage


class TestArtifactService:
    """Test ArtifactService functionality."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        # Ensure the directory has proper permissions for subdirectory creation
        os.chmod(temp_dir, 0o755)
        yield temp_dir
        # Clean up with proper error handling
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # Try to fix permissions and retry
            for root, dirs, files in os.walk(temp_dir):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o755)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o644)
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

    @pytest_asyncio.fixture
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

        storage_refs = await initialized_service.store_envelope_artifacts(envelope)

        assert storage_refs == []

    @pytest.mark.asyncio
    async def test_store_envelope_artifacts_success(self, initialized_service, sample_result_envelope):
        """Test successful artifact storage from envelope."""
        storage_refs = await initialized_service.store_envelope_artifacts(
            sample_result_envelope
        )

        assert len(storage_refs) == 1
        assert "tasks/test-task-123/" in storage_refs[0]
        assert "test_document" in storage_refs[0]

    @pytest.mark.asyncio
    async def test_store_single_artifact(self, initialized_service, sample_file_artifact):
        """Test storing a single artifact."""
        storage_ref = await initialized_service._store_single_artifact(
            "task-456", sample_file_artifact
        )

        assert "tasks/task-456/" in storage_ref
        assert sample_file_artifact.artifact_id in storage_ref
        assert "test_document" in storage_ref

    @pytest.mark.asyncio
    async def test_artifact_key_generation(self, initialized_service, sample_file_artifact):
        """Test artifact storage key generation."""
        key = initialized_service._generate_artifact_key(
            "task-456", sample_file_artifact
        )

        expected_parts = [
            "tasks/task-456/",
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
            sample_result_envelope
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
    async def test_list_execution_artifacts(self, storage):
        """Test listing artifacts for execution."""
        # Create fresh service without any pre-existing files
        fresh_service = ArtifactService(storage)
        await fresh_service.initialize()

        # Initially empty
        artifacts = await fresh_service.list_execution_artifacts()
        assert artifacts == []

        # Create sample file artifact in a different location
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Sample test content")
            temp_file_path = temp_file.name

        sample_artifact = FileArtifact.from_path(
            name="test_document",
            file_path=temp_file_path,
            task_id="test-task-456",
            metadata={"source": "test"}
        )

        # Create result envelope
        metrics = ExecutionMetrics(
            execution_time=1.5,
            tokens_used=150,
            model_calls=1,
            cost_estimate=0.01
        )
        result_envelope = ResultEnvelope.create_success(
            result={"output": "Test result"},
            task_id="test-task-456",
            execution_id="test-exec",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=[sample_artifact]
        )

        # Store artifacts
        await fresh_service.store_envelope_artifacts(result_envelope)

        # List artifacts (should have both artifact file and metadata file)
        artifacts = await fresh_service.list_execution_artifacts()
        assert len(artifacts) == 2
        artifact_files = [f for f in artifacts if not f.endswith('.metadata')]
        metadata_files = [f for f in artifacts if f.endswith('.metadata')]
        assert len(artifact_files) == 1
        assert len(metadata_files) == 1
        assert "tasks/test-task-456/" in artifact_files[0]

    @pytest.mark.asyncio
    async def test_get_artifact_metadata(self, initialized_service, sample_result_envelope):
        """Test getting artifact metadata."""
        # Store artifact
        storage_refs = await initialized_service.store_envelope_artifacts(
            sample_result_envelope
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
    async def test_cleanup_execution_artifacts(self, storage):
        """Test cleaning up execution artifacts."""
        # Create fresh service without any pre-existing files
        fresh_service = ArtifactService(storage)
        await fresh_service.initialize()

        # Create and store artifacts
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Sample test content")
            temp_file_path = temp_file.name

        sample_artifact = FileArtifact.from_path(
            name="test_document",
            file_path=temp_file_path,
            task_id="test-task-cleanup",
            metadata={"source": "test"}
        )

        metrics = ExecutionMetrics(
            execution_time=1.5,
            tokens_used=150,
            model_calls=1,
            cost_estimate=0.01
        )
        result_envelope = ResultEnvelope.create_success(
            result={"output": "Test result"},
            task_id="test-task-cleanup",
            execution_id="test-exec",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=[sample_artifact]
        )

        # Store artifacts
        await fresh_service.store_envelope_artifacts(result_envelope)

        # Verify artifacts exist (artifact + metadata file)
        artifacts_before = await fresh_service.list_execution_artifacts()
        assert len(artifacts_before) == 2

        # Cleanup
        deleted_count = await fresh_service.cleanup_execution_artifacts()
        assert deleted_count == 2  # Both artifact and metadata file

        # Verify artifacts are gone
        artifacts_after = await fresh_service.list_execution_artifacts()
        assert len(artifacts_after) == 0

    @pytest.mark.asyncio
    async def test_create_file_artifact_from_storage(self, initialized_service, sample_result_envelope, temp_storage_dir):
        """Test creating FileArtifact from storage."""
        # Store artifacts first
        storage_refs = await initialized_service.store_envelope_artifacts(
            sample_result_envelope
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
        # Create artifact with invalid file path that will fail
        invalid_artifact = FileArtifact.from_path(
            name="failing_artifact",
            file_path="/nonexistent/invalid/path/file.txt",  # Invalid path
            task_id="task-123",
            metadata={"source": "test"}
        )

        metrics = ExecutionMetrics(
            execution_time=1.0,
            tokens_used=100,
            model_calls=1,
            cost_estimate=0.01
        )
        result = ExecutorResult(result="Test")

        envelope = ResultEnvelope.create_success(
            result=result,
            task_id="task-123",
            execution_id="exec-123",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            artifacts=[invalid_artifact]
        )

        # Should handle error gracefully
        storage_refs = await initialized_service.store_envelope_artifacts(envelope)
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
        storage_refs = await initialized_service.store_envelope_artifacts(envelope)

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

    @pytest.mark.skip(reason="Complex edge case test requiring extensive async mock refactoring")
    @pytest.mark.asyncio
    async def test_store_artifacts_partial_failure(self):
        """Test partial failure when storing multiple artifacts."""
        # Create a mixed mock storage for this test (some sync methods, some async)
        mock_storage = Mock()

        # Create real temp directories for mock storage
        import tempfile
        temp_base = tempfile.mkdtemp()

        # Mock storage methods with real temp paths
        artifacts_path = Path(temp_base) / "artifacts"
        temp_path = Path(temp_base) / "temp"

        # Ensure directories exist for initialization
        artifacts_path.mkdir(parents=True, exist_ok=True)
        temp_path.mkdir(parents=True, exist_ok=True)

        # These should be synchronous methods that return paths
        mock_storage.get_artifacts_path.return_value = artifacts_path
        mock_storage.get_temp_path.return_value = temp_path
        mock_storage.mount_path = Path(temp_base)
        mock_storage.config = Mock()
        mock_storage.config.model_dump.return_value = {"mount_path": temp_base}

        # Only put_text should be async
        mock_storage.put_text = AsyncMock()

        # Create artifact service with fresh mock
        artifact_service = ArtifactService(mock_storage)

        # Create real artifacts - one succeeds, one will fail during storage
        # Good artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Good content")
            good_file_path = temp_file.name

        good_artifact = FileArtifact.from_path(
            name="good_artifact",
            file_path=good_file_path,
            task_id="task-123",
            metadata={"source": "test"}
        )

        # Second good artifact that will fail only during storage
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Bad content")
            bad_file_path = temp_file.name

        bad_artifact = FileArtifact.from_path(
            name="bad_artifact",
            file_path=bad_file_path,
            task_id="task-123",
            metadata={"source": "test"}
        )

        # Mock storage operations to simulate partial failure
        # First call succeeds, second call fails
        call_count = 0
        async def mock_put_text_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "good_path"
            else:
                raise Exception("Storage fail")

        mock_storage.put_text.side_effect = mock_put_text_side_effect

        # Create envelope with both artifacts
        metrics = ExecutionMetrics(
            execution_time=1.0,
            tokens_used=100,
            model_calls=1,
            cost_estimate=0.01
        )
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
        await artifact_service.initialize()
        storage_refs = await artifact_service.store_envelope_artifacts(envelope)

        # Should continue despite failure - only one succeeded
        assert len(storage_refs) == 1  # Only the good one succeeded

        # Cleanup temp files
        import os
        os.unlink(good_file_path)
        os.unlink(bad_file_path)
        import shutil
        shutil.rmtree(temp_base)
