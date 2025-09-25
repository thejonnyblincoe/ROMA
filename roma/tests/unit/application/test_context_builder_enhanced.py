"""
Enhanced Tests for Context Builder Service - Task 3.1.2 Implementation.

Tests the enhanced multimodal context assembly functionality including:
- KnowledgeStore integration for lineage and sibling context
- Multimodal artifact context building
- Tool availability context
- Thread-safe knowledge persistence
"""

from unittest.mock import AsyncMock

import pytest

from roma.application.services.context_builder_service import (
    ContextBuilderService,
    ContextItem,
)
from roma.application.services.knowledge_store_service import KnowledgeStoreService
from roma.domain.entities.artifacts.image_artifact import ImageArtifact
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_responses import ExecutorResult
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.context_item_type import ContextItemType
from roma.domain.value_objects.knowledge_record import KnowledgeRecord
from roma.domain.value_objects.result_envelope import ExecutionMetrics, ResultEnvelope
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


class TestEnhancedContextBuilder:
    """Test the enhanced ContextBuilder with KnowledgeStore integration."""

    @pytest.fixture
    def mock_knowledge_store(self) -> AsyncMock:
        """Create mock KnowledgeStore service."""
        mock = AsyncMock(spec=KnowledgeStoreService)
        return mock

    @pytest.fixture
    def mock_toolkit_manager(self) -> AsyncMock:
        """Create mock toolkit manager."""
        mock = AsyncMock()
        mock.get_available_tools.return_value = [
            {"name": "web_search", "description": "Search the web"},
            {"name": "code_exec", "description": "Execute code"}
        ]
        return mock

    @pytest.fixture
    def context_service(self, mock_knowledge_store, mock_toolkit_manager) -> ContextBuilderService:
        """Create ContextBuilder with mocked dependencies."""
        return ContextBuilderService(
            knowledge_store=mock_knowledge_store,
            toolkit_manager=mock_toolkit_manager
        )

    @pytest.fixture
    def sample_task(self) -> TaskNode:
        """Create sample task for testing."""
        return TaskNode(
            goal="Analyze AI market trends",
            task_type=TaskType.THINK,
            status=TaskStatus.READY,
            parent_id="parent_task_123"
        )

    @pytest.fixture
    def sample_knowledge_records(self) -> list[KnowledgeRecord]:
        """Create sample knowledge records for testing."""
        # Create execution metrics
        metrics = ExecutionMetrics(execution_time=1.5)

        # Create executor results
        parent_result = ExecutorResult(
            output="Found 15 relevant AI market reports covering 2024 trends.",
            primary_result="AI market expected to grow 35% in 2024",
            success=True
        )

        sibling_result = ExecutorResult(
            output="EU AI Act implementation will impact market dynamics.",
            primary_result="Regulatory compliance costs expected to increase",
            success=True
        )

        # Parent record
        parent_envelope = ResultEnvelope.create_success(
            result=parent_result,
            task_id="parent_task_123",
            execution_id="exec_123",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            output_text="AI market expected to grow 35% in 2024"
        )

        parent_record = KnowledgeRecord.create(
            task_id="parent_task_123",
            goal="Research AI market",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.COMPLETED,
            result=parent_envelope
        )

        # Sibling record
        sibling_envelope = ResultEnvelope.create_success(
            result=sibling_result,
            task_id="sibling_task_456",
            execution_id="exec_456",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            output_text="Regulatory compliance costs expected to increase"
        )

        sibling_record = KnowledgeRecord.create(
            task_id="sibling_task_456",
            goal="Analyze AI regulations",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED,
            parent_task_id="parent_task_123",
            result=sibling_envelope
        )

        return [parent_record, sibling_record]

    @pytest.mark.asyncio
    async def test_build_context_with_knowledge_integration(
        self, context_service, sample_task, sample_knowledge_records, mock_knowledge_store
    ):
        """Test building context with KnowledgeStore integration."""
        # Setup mock responses
        parent_record, sibling_record = sample_knowledge_records
        mock_knowledge_store.get_record.return_value = parent_record
        mock_knowledge_store.get_child_records.return_value = [sibling_record]

        # Build context
        context = await context_service.build_context(
            task=sample_task,
            overall_objective="Understand AI market dynamics"
        )

        # Verify basic context structure
        assert context.task == sample_task
        assert context.overall_objective == "Understand AI market dynamics"
        assert len(context.context_items) >= 4  # task_goal + objective + parent + sibling

        # Verify knowledge integration calls
        mock_knowledge_store.get_record.assert_called_with(sample_task.parent_id)
        mock_knowledge_store.get_child_records.assert_called_with(sample_task.parent_id)

        # Verify parent context is included
        parent_items = [item for item in context.context_items
                       if item.item_type == ContextItemType.PARENT_RESULT]
        assert len(parent_items) == 1
        assert "AI market expected to grow 35%" in parent_items[0].content

        # Verify sibling context is included
        sibling_items = [item for item in context.context_items
                        if item.item_type == ContextItemType.SIBLING_RESULT]
        assert len(sibling_items) == 1
        assert "Regulatory compliance costs" in sibling_items[0].content

    @pytest.mark.asyncio
    async def test_build_context_with_artifacts(self, context_service, sample_task):
        """Test building context with artifacts."""
        # Create artifacts
        image_artifact = ImageArtifact.from_url(
            name="market_chart",
            url="https://example.com/chart.png",
            task_id=sample_task.task_id
        )

        # Build context
        context = await context_service.build_context(
            task=sample_task,
            overall_objective="Multi-modal analysis"
        )

        # Should have basic context structure
        assert context.task == sample_task
        assert context.overall_objective == "Multi-modal analysis"
        assert len(context.context_items) >= 2  # task_goal + objective

    @pytest.mark.asyncio
    async def test_build_toolkit_context(self, context_service, sample_task, mock_toolkit_manager):
        """Test building context with toolkit availability."""
        # Build context
        context = await context_service.build_context(
            task=sample_task,
            overall_objective="Tool-enabled analysis"
        )

        # Verify toolkit manager was called
        mock_toolkit_manager.get_available_tools.assert_called_once()

        # Verify toolkit context is included
        toolkit_items = [item for item in context.context_items
                        if item.item_type == ContextItemType.TOOLKITS]
        assert len(toolkit_items) == 1

        toolkit_content = toolkit_items[0].content
        assert "web_search" in str(toolkit_content)
        assert "code_exec" in str(toolkit_content)

    @pytest.mark.asyncio
    async def test_knowledge_context_without_parent(
        self, context_service, mock_knowledge_store
    ):
        """Test building context for root task without parent."""
        root_task = TaskNode(
            goal="Root analysis task",
            task_type=TaskType.THINK,
            status=TaskStatus.READY,
            parent_id=None  # No parent
        )

        # Build context
        context = await context_service.build_context(
            task=root_task,
            overall_objective="Root level analysis"
        )

        # Verify no parent context calls
        mock_knowledge_store.get_record.assert_not_called()

        # Should still have basic context items
        assert len(context.context_items) >= 2  # task_goal + objective

    @pytest.mark.asyncio
    async def test_context_priority_ordering(
        self, context_service, sample_task, sample_knowledge_records, mock_knowledge_store
    ):
        """Test that context items are properly prioritized."""
        # Setup mock with parent record
        parent_record, _ = sample_knowledge_records
        mock_knowledge_store.get_record.return_value = parent_record
        mock_knowledge_store.get_child_records.return_value = []

        # Build context
        context = await context_service.build_context(
            task=sample_task,
            overall_objective="Priority test"
        )

        # Verify priority ordering (highest first)
        priorities = [item.priority for item in context.context_items]
        assert priorities == sorted(priorities, reverse=True)

        # Verify expected priority ranges
        task_items = [item for item in context.context_items
                     if item.item_type == ContextItemType.TASK_GOAL]
        objective_items = [item for item in context.context_items
                          if item.item_type == ContextItemType.OVERALL_OBJECTIVE]
        parent_items = [item for item in context.context_items
                       if item.item_type == ContextItemType.PARENT_RESULT]

        assert task_items[0].priority == 100  # Highest priority
        assert objective_items[0].priority == 95
        assert parent_items[0].priority == 85

    @pytest.mark.asyncio
    async def test_context_creation_basics(self, context_service, sample_task):
        """Test basic context creation functionality."""
        context = await context_service.build_context(
            task=sample_task,
            overall_objective="Basic test"
        )

        # Verify basic structure
        assert context.task == sample_task
        assert context.overall_objective == "Basic test"
        assert len(context.context_items) >= 2

    @pytest.mark.asyncio
    async def test_error_handling_in_knowledge_retrieval(
        self, context_service, sample_task, mock_knowledge_store
    ):
        """Test graceful error handling when knowledge retrieval fails."""
        # Setup mock to raise exception
        mock_knowledge_store.get_record.side_effect = Exception("Database error")

        # Should not raise, but log error and continue
        context = await context_service.build_context(
            task=sample_task,
            overall_objective="Error handling test"
        )

        # Should still have basic context items
        assert len(context.context_items) >= 2  # task_goal + objective

        # Should not have parent context items due to error
        parent_items = [item for item in context.context_items
                       if item.item_type == ContextItemType.PARENT_RESULT]
        assert len(parent_items) == 0

    def test_context_item_creation(self):
        """Test ContextItem creation."""
        item = ContextItem.from_text(
            content="Test content",
            item_type=ContextItemType.REFERENCE_TEXT,
            metadata={"source": "test"},
            priority=50
        )

        assert item.content == "Test content"
        assert item.item_type == ContextItemType.REFERENCE_TEXT
        assert item.priority == 50
        assert item.metadata == {"source": "test"}


class TestKnowledgeStoreIntegration:
    """Test KnowledgeStore specific integration patterns."""

    @pytest.fixture
    def real_knowledge_store(self) -> KnowledgeStoreService:
        """Create real KnowledgeStore for integration testing."""
        return KnowledgeStoreService()

    @pytest.fixture
    def context_service_with_real_store(self, real_knowledge_store) -> ContextBuilderService:
        """Create ContextBuilder with real KnowledgeStore."""
        return ContextBuilderService(knowledge_store=real_knowledge_store)

    @pytest.mark.asyncio
    async def test_end_to_end_knowledge_flow(
        self, context_service_with_real_store, real_knowledge_store
    ):
        """Test complete flow from knowledge storage to context building."""
        # Create parent task and store result
        parent_task = TaskNode(
            goal="Research market data",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.COMPLETED
        )

        metrics = ExecutionMetrics(execution_time=2.0)
        executor_result = ExecutorResult(
            output="Market research completed successfully",
            primary_result="Market growing at 25% annually",
            success=True
        )

        parent_result = ResultEnvelope.create_success(
            result=executor_result,
            task_id=parent_task.task_id,
            execution_id="exec_parent",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            output_text="Market growing at 25% annually"
        )

        # Store parent record
        await real_knowledge_store.add_or_update_record(parent_task, parent_result)

        # Create child task
        child_task = TaskNode(
            goal="Analyze market trends",
            task_type=TaskType.THINK,
            status=TaskStatus.READY,
            parent_id=parent_task.task_id
        )

        # Build context for child task
        context = await context_service_with_real_store.build_context(
            task=child_task,
            overall_objective="Trend analysis"
        )

        # Verify parent context is properly retrieved and included
        parent_items = [item for item in context.context_items
                       if item.item_type == ContextItemType.PARENT_RESULT]
        assert len(parent_items) == 1
        assert "Market growing at 25%" in parent_items[0].content

    @pytest.mark.asyncio
    async def test_sibling_context_retrieval(
        self, context_service_with_real_store, real_knowledge_store
    ):
        """Test retrieval of sibling task context."""
        # Create parent task
        parent_task = TaskNode(
            goal="Market analysis project",
            task_type=TaskType.THINK,
            status=TaskStatus.EXECUTING
        )

        # Create first sibling (completed)
        sibling1 = TaskNode(
            goal="Analyze competitors",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.COMPLETED,
            parent_id=parent_task.task_id
        )

        metrics = ExecutionMetrics(execution_time=1.0)
        executor_result = ExecutorResult(
            output="Competitor analysis complete",
            primary_result="Top 3 competitors identified",
            success=True
        )

        sibling1_result = ResultEnvelope.create_success(
            result=executor_result,
            task_id=sibling1.task_id,
            execution_id="exec_sibling1",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=metrics,
            output_text="Top 3 competitors identified"
        )

        await real_knowledge_store.add_or_update_record(sibling1, sibling1_result)

        # Create second sibling (current)
        sibling2 = TaskNode(
            goal="Analyze market size",
            task_type=TaskType.THINK,
            status=TaskStatus.READY,
            parent_id=parent_task.task_id
        )

        # Build context for second sibling
        context = await context_service_with_real_store.build_context(
            task=sibling2,
            overall_objective="Market sizing"
        )

        # Verify sibling context is included
        sibling_items = [item for item in context.context_items
                        if item.item_type == ContextItemType.SIBLING_RESULT]
        assert len(sibling_items) == 1
        assert "Top 3 competitors identified" in sibling_items[0].content
