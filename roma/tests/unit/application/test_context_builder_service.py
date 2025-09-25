"""
Tests for Context Builder Service.

Tests the multimodal context assembly functionality including text content,
file artifacts, and context validation.
"""


import pytest

from roma.application.services.context_builder_service import (
    ContextBuilderService,
    ContextItem,
    TaskContext,
)
from roma.domain.entities.artifacts.file_artifact import FileArtifact
from roma.domain.entities.media_file import MediaFile
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.media_type import MediaType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


class TestContextItem:
    """Test ContextItem creation and functionality."""

    def test_from_text_creates_text_context_item(self):
        """Test creating context item from text."""
        content = "Test content"
        metadata = {"source": "test"}
        priority = 10

        item = ContextItem.from_text(content, metadata, priority)

        assert item.content_type == MediaType.TEXT
        assert item.content == content
        assert item.metadata == metadata
        assert item.priority == priority
        assert item.item_id is not None

    def test_from_artifact_creates_artifact_context_item(self):
        """Test creating context item from artifact."""
        media_file = MediaFile.from_bytes(b"test content", "test.txt", "text/plain")
        artifact = FileArtifact(name="test", media_file=media_file)
        priority = 20

        item = ContextItem.from_artifact(artifact, priority)

        assert item.content_type == MediaType.FILE
        assert item.content == artifact
        assert item.metadata == artifact.metadata
        assert item.priority == priority


class TestTaskContext:
    """Test TaskContext functionality."""

    @pytest.fixture
    def sample_task(self) -> TaskNode:
        """Create sample task for testing."""
        return TaskNode(
            goal="Test task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

    @pytest.fixture
    def sample_context_items(self) -> list[ContextItem]:
        """Create sample context items."""
        text_item = ContextItem.from_text("Test text", priority=10)

        media_file = MediaFile.from_bytes(b"file content", "test.txt", "text/plain")
        artifact = FileArtifact(name="test", media_file=media_file)
        file_item = ContextItem.from_artifact(artifact, priority=5)

        return [text_item, file_item]

    def test_context_creation(self, sample_task, sample_context_items):
        """Test TaskContext creation."""
        context = TaskContext(
            task=sample_task,
            overall_objective="Test objective",
            context_items=sample_context_items
        )

        assert context.task == sample_task
        assert context.overall_objective == "Test objective"
        assert len(context.context_items) == 2

    def test_get_text_content(self, sample_task, sample_context_items):
        """Test extracting text content."""
        context = TaskContext(
            task=sample_task,
            overall_objective="Test objective",
            context_items=sample_context_items
        )

        text_content = context.get_text_content()

        assert len(text_content) == 1
        assert "Test text" in text_content

    def test_get_file_artifacts(self, sample_task, sample_context_items):
        """Test extracting file artifacts."""
        context = TaskContext(
            task=sample_task,
            overall_objective="Test objective",
            context_items=sample_context_items
        )

        file_artifacts = context.get_file_artifacts()

        assert len(file_artifacts) == 1
        assert isinstance(file_artifacts[0], FileArtifact)

    def test_get_by_media_type(self, sample_task, sample_context_items):
        """Test filtering by media type."""
        context = TaskContext(
            task=sample_task,
            overall_objective="Test objective",
            context_items=sample_context_items
        )

        text_items = context.get_by_media_type(MediaType.TEXT)
        file_items = context.get_by_media_type(MediaType.FILE)

        assert len(text_items) == 1
        assert len(file_items) == 1

    def test_to_dict_serialization(self, sample_task, sample_context_items):
        """Test dictionary serialization."""
        context = TaskContext(
            task=sample_task,
            overall_objective="Test objective",
            context_items=sample_context_items,
            constraints=["constraint1"],
            user_preferences={"pref": "value"}
        )

        context_dict = context.to_dict()

        assert "task" in context_dict
        assert "overall_objective" in context_dict
        assert "context_items" in context_dict
        assert "constraints" in context_dict
        assert "user_preferences" in context_dict
        assert context_dict["text_count"] == 1
        assert context_dict["file_count"] == 1


class TestContextBuilderService:
    """Test ContextBuilderService functionality."""

    @pytest.fixture
    def service(self) -> ContextBuilderService:
        """Create service instance."""
        return ContextBuilderService()

    @pytest.fixture
    def sample_task(self) -> TaskNode:
        """Create sample task for testing."""
        return TaskNode(
            goal="Research AI trends",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.PENDING
        )

    @pytest.fixture
    def sample_file_artifact(self) -> FileArtifact:
        """Create sample file artifact."""
        media_file = MediaFile.from_bytes(
            b"Sample document content",
            "document.txt",
            "text/plain"
        )
        return FileArtifact(name="sample_doc", media_file=media_file)

    @pytest.mark.asyncio
    async def test_build_basic_context(self, service, sample_task):
        """Test building basic context with task and objective."""
        context = await service.build_context(
            task=sample_task,
            overall_objective="Research current AI market trends"
        )

        assert context.task == sample_task
        assert context.overall_objective == "Research current AI market trends"
        assert len(context.context_items) >= 2  # At least task goal and objective

        # Check priority ordering (highest first)
        priorities = [item.priority for item in context.context_items]
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_build_context_with_text_content(self, service, sample_task):
        """Test building context with additional text content."""
        text_content = ["Previous research shows...", "Key findings indicate..."]

        context = await service.build_context(
            task=sample_task,
            overall_objective="Research objective",
            text_content=text_content
        )

        text_items = context.get_text_content()
        assert any("Previous research shows" in item for item in text_items)
        assert any("Key findings indicate" in item for item in text_items)

    @pytest.mark.asyncio
    async def test_build_context_with_file_artifacts(
        self, service, sample_task, sample_file_artifact
    ):
        """Test building context with file artifacts."""
        file_artifacts = [sample_file_artifact]

        context = await service.build_context(
            task=sample_task,
            overall_objective="Research objective",
            file_artifacts=file_artifacts
        )

        artifacts = context.get_file_artifacts()
        assert len(artifacts) == 1
        assert artifacts[0].name == "sample_doc"

    @pytest.mark.asyncio
    async def test_build_context_with_parent_results(self, service, sample_task):
        """Test building context with parent task results."""
        parent_results = ["Parent task completed successfully", "Found 10 relevant papers"]

        context = await service.build_context(
            task=sample_task,
            overall_objective="Research objective",
            parent_results=parent_results
        )

        text_items = context.get_text_content()
        assert any("Parent task completed" in item for item in text_items)
        assert any("Found 10 relevant papers" in item for item in text_items)

    @pytest.mark.asyncio
    async def test_build_context_with_sibling_results(self, service, sample_task):
        """Test building context with sibling task results."""
        sibling_results = ["Sibling found key insight", "Related analysis complete"]

        context = await service.build_context(
            task=sample_task,
            overall_objective="Research objective",
            sibling_results=sibling_results
        )

        text_items = context.get_text_content()
        assert any("Sibling found key insight" in item for item in text_items)
        assert any("Related analysis complete" in item for item in text_items)

    @pytest.mark.asyncio
    async def test_build_context_with_constraints_and_preferences(
        self, service, sample_task
    ):
        """Test building context with constraints and user preferences."""
        constraints = ["Use only peer-reviewed sources", "Maximum 2 pages"]
        user_preferences = {"style": "academic", "detail_level": "high"}

        context = await service.build_context(
            task=sample_task,
            overall_objective="Research objective",
            constraints=constraints,
            user_preferences=user_preferences
        )

        assert context.constraints == constraints
        assert context.user_preferences == user_preferences

    @pytest.mark.asyncio
    async def test_context_priority_ordering(self, service, sample_task):
        """Test that context items are properly ordered by priority."""
        context = await service.build_context(
            task=sample_task,
            overall_objective="Research objective",
            text_content=["Additional content"],
            parent_results=["Parent result"],
            sibling_results=["Sibling result"]
        )

        priorities = [item.priority for item in context.context_items]

        # Task goal should be highest priority (100)
        assert priorities[0] == 100

        # Overall objective should be second (95)
        assert priorities[1] == 95

        # Should be sorted in descending order
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_build_lineage_context(self, service, sample_task):
        """Test building context with parent chain."""
        parent_chain = [
            {"task_id": "parent1", "result": "Found initial data"},
            {"task_id": "parent2", "result": "Analyzed patterns"}
        ]

        context = await service.build_lineage_context(
            task=sample_task,
            overall_objective="Research objective",
            parent_chain=parent_chain
        )

        text_items = context.get_text_content()
        assert any("Found initial data" in item for item in text_items)
        assert any("Analyzed patterns" in item for item in text_items)

    @pytest.mark.asyncio
    async def test_build_rich_context(self, service, sample_task):
        """Test building rich context with knowledge store."""
        knowledge_store = {
            "relevant_results": ["Result 1", "Result 2"],
            "constraints": ["Constraint 1"],
            "user_preferences": {"format": "json"},
            "custom_data": "Important context"
        }

        context = await service.build_rich_context(
            task=sample_task,
            overall_objective="Research objective",
            knowledge_store=knowledge_store
        )

        assert context.constraints == ["Constraint 1"]
        assert context.user_preferences == {"format": "json"}

        text_items = context.get_text_content()
        assert any("custom_data: Important context" in item for item in text_items)

    def test_validate_context_valid(self, service, sample_task):
        """Test validating a valid context."""
        context = TaskContext(
            task=sample_task,
            overall_objective="Test objective"
        )

        assert service.validate_context(context) is True

    def test_validate_context_missing_task(self, service, sample_task):
        """Test validating context with missing task (invalid task)."""
        # Create context with invalid task data (empty goal)
        try:
            invalid_task = TaskNode(goal="", task_type=TaskType.THINK)
            context = TaskContext(
                task=invalid_task,
                overall_objective="Test objective"
            )
            # If somehow created, validation should fail
            assert service.validate_context(context) is False
        except ValueError:
            # Expected - TaskNode validation should prevent empty goal
            pass

    def test_validate_context_missing_objective(self, service, sample_task):
        """Test validating context with missing objective."""
        context = TaskContext(
            task=sample_task,
            overall_objective=""
        )

        assert service.validate_context(context) is False

    def test_validate_context_with_inaccessible_file(
        self, service, sample_task
    ):
        """Test validating context with inaccessible file artifact."""
        # Create file artifact with non-existent file
        media_file = MediaFile.from_filepath("/nonexistent/file.txt")
        artifact = FileArtifact(name="missing", media_file=media_file)

        context_item = ContextItem.from_artifact(artifact)
        context = TaskContext(
            task=sample_task,
            overall_objective="Test objective",
            context_items=[context_item]
        )

        # Should fail validation due to inaccessible file
        assert service.validate_context(context) is False
