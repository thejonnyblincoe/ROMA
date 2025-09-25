"""
Tests for AgentRuntimeService - Agent Execution and Lifecycle Management.

Tests the agent runtime service including agent creation, execution with proper
agent_type parameter, ResultEnvelope creation, and event emission.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.event_publisher import EventPublisher
from roma.application.services.event_store import InMemoryEventStore
from roma.domain.context import ContextItem, TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_responses import ExecutorResult
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.result_envelope import ResultEnvelope
from roma.domain.value_objects.task_type import TaskType
from roma.infrastructure.agents.agent_factory import AgentFactory
from roma.infrastructure.agents.configurable_agent import ConfigurableAgent


@pytest.fixture
def mock_event_store():
    """Mock InMemoryEventStore."""
    return AsyncMock(spec=InMemoryEventStore)


@pytest.fixture
def mock_agent_factory():
    """Mock AgentFactory."""
    return AsyncMock(spec=AgentFactory)


@pytest.fixture
def mock_event_publisher(mock_event_store):
    """Mock EventPublisher."""
    return EventPublisher(event_store=mock_event_store)


@pytest.fixture
def mock_agent():
    """Mock ConfigurableAgent."""
    agent = MagicMock(spec=ConfigurableAgent)
    agent.name = "test_agent"
    agent.run = AsyncMock()
    return agent


@pytest.fixture
def agent_runtime_service(mock_event_publisher, mock_agent_factory):
    """AgentRuntimeService instance with mocked dependencies."""
    return AgentRuntimeService(
        agent_factory=mock_agent_factory,
        event_publisher=mock_event_publisher
    )


@pytest.fixture
def sample_task():
    """Sample TaskNode for testing."""
    return TaskNode(
        task_id="test_task",
        goal="Test goal",
        task_type=TaskType.THINK
    )


@pytest.fixture
def sample_context():
    """Sample TaskContext for testing."""
    return TaskContext(
        task=None,
        overall_objective="Test objective",
        execution_id="test-execution-id",
        context_items=[],
        execution_metadata={}
    )


class TestAgentRuntimeService:
    """Test cases for AgentRuntimeService."""

    def test_initialization(self, mock_event_publisher, mock_agent_factory):
        """Test service initialization."""
        service = AgentRuntimeService(
            agent_factory=mock_agent_factory,
            event_publisher=mock_event_publisher
        )

        assert service._agent_factory == mock_agent_factory
        assert service._event_publisher == mock_event_publisher
        assert service._initialized is False
        assert len(service._runtime_agents) == 0
        assert service._runtime_metrics["agents_created"] == 0
        assert service._runtime_metrics["agents_executed"] == 0
        assert service._runtime_metrics["runtime_errors"] == 0

    @pytest.mark.asyncio
    async def test_initialization_with_dependencies(self, agent_runtime_service, mock_agent_factory):
        """Test service initialization with dependencies."""
        # Test uninitialized state
        assert not agent_runtime_service._initialized

        # Initialize
        await agent_runtime_service.initialize()

        # Verify initialization
        assert agent_runtime_service._initialized is True
        mock_agent_factory.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_agent_lazy_creation(self, agent_runtime_service, mock_agent_factory, mock_agent):
        """Test lazy agent creation."""
        await agent_runtime_service.initialize()

        # Setup agent factory to return config and create agent
        config_dict = {
            "name": "test_agent",
            "type": "executor",
            "task_type": TaskType.THINK,
            "enabled": True,
            "model": {"provider": "litellm", "model_id": "gpt-4o", "temperature": 0.7}
        }
        mock_agent_factory.get_agent_config.return_value = config_dict
        mock_agent_factory.create_agent.return_value = mock_agent

        # Get agent (should create on first call)
        agent = await agent_runtime_service.get_agent(TaskType.THINK, AgentType.EXECUTOR)

        assert agent == mock_agent
        assert agent_runtime_service._runtime_metrics["agents_created"] == 1

        # Verify agent was cached
        agent_key = f"{TaskType.THINK.value}_{AgentType.EXECUTOR.value}"
        assert agent_key in agent_runtime_service._runtime_agents
        assert agent_runtime_service._runtime_agents[agent_key] == mock_agent

    @pytest.mark.asyncio
    async def test_get_agent_cached(self, agent_runtime_service, mock_agent):
        """Test getting cached agent."""
        await agent_runtime_service.initialize()

        # Manually cache agent
        agent_key = f"{TaskType.THINK.value}_{AgentType.EXECUTOR.value}"
        agent_runtime_service._runtime_agents[agent_key] = mock_agent

        # Get agent (should return cached)
        agent = await agent_runtime_service.get_agent(TaskType.THINK, AgentType.EXECUTOR)

        assert agent == mock_agent
        # Metrics should not increase for cached retrieval
        assert agent_runtime_service._runtime_metrics["agents_created"] == 0

    @pytest.mark.asyncio
    async def test_get_agent_not_initialized(self, agent_runtime_service):
        """Test getting agent when service not initialized."""
        with pytest.raises(RuntimeError, match="Agent runtime service not initialized"):
            await agent_runtime_service.get_agent(TaskType.THINK, AgentType.EXECUTOR)

    @pytest.mark.asyncio
    async def test_execute_agent_success(self, agent_runtime_service, mock_agent, sample_task, sample_context):
        """Test successful agent execution with proper ResultEnvelope creation."""
        await agent_runtime_service.initialize()

        # Setup agent to return ExecutorResult
        executor_result = ExecutorResult(
            result="Task completed successfully",
            sources=["source1", "source2"],
            success=True,
            confidence=0.95,
            tokens_used=150
        )
        mock_agent.run.return_value = executor_result

        # Execute agent with explicit agent_type
        result_envelope = await agent_runtime_service.execute_agent(
            agent=mock_agent,
            task=sample_task,
            context=sample_context,
            agent_type=AgentType.EXECUTOR
        )

        # Verify ResultEnvelope structure
        assert isinstance(result_envelope, ResultEnvelope)
        assert result_envelope.success is True
        assert result_envelope.task_id == sample_task.task_id
        assert result_envelope.agent_type == AgentType.EXECUTOR
        assert result_envelope.metadata["agent_name"] == "test_agent"
        assert result_envelope.metadata["agent_type"] == "executor"
        assert result_envelope.metadata["response_type"] == "ExecutorResult"
        assert result_envelope.metadata["framework"] == "agno"

        # Verify execution metrics
        assert result_envelope.execution_metrics.tokens_used == 150
        assert result_envelope.execution_metrics.execution_time > 0
        assert result_envelope.execution_metrics.model_calls == 1

        # Verify agent was called correctly
        mock_agent.run.assert_called_once_with(sample_task, sample_context)

        # Verify metrics updated
        assert agent_runtime_service._runtime_metrics["agents_executed"] == 1

    @pytest.mark.asyncio
    async def test_execute_agent_without_agent_type(self, agent_runtime_service, mock_agent, sample_task, sample_context):
        """Test agent execution without explicit agent_type parameter."""
        await agent_runtime_service.initialize()

        # Setup agent to return simple result
        mock_agent.run.return_value = {"output": "test result"}

        # Execute agent without agent_type (should default to EXECUTOR)
        result_envelope = await agent_runtime_service.execute_agent(
            agent=mock_agent,
            task=sample_task,
            context=sample_context
        )

        # Verify default agent_type was used
        assert result_envelope.agent_type == AgentType.EXECUTOR
        assert result_envelope.metadata["agent_type"] == "executor"

    @pytest.mark.asyncio
    async def test_execute_agent_with_different_agent_types(self, agent_runtime_service, mock_agent, sample_task, sample_context):
        """Test agent execution with different agent types."""
        await agent_runtime_service.initialize()

        mock_agent.run.return_value = {"output": "test result"}

        # Test different agent types
        agent_types = [AgentType.ATOMIZER, AgentType.PLANNER, AgentType.AGGREGATOR]

        for agent_type in agent_types:
            result_envelope = await agent_runtime_service.execute_agent(
                agent=mock_agent,
                task=sample_task,
                context=sample_context,
                agent_type=agent_type
            )

            assert result_envelope.agent_type == agent_type
            assert result_envelope.metadata["agent_type"] == agent_type.value

    @pytest.mark.asyncio
    async def test_execute_agent_with_none_result(self, agent_runtime_service, mock_agent, sample_task, sample_context):
        """Test agent execution when agent returns None - should raise error."""
        await agent_runtime_service.initialize()

        # Agent returns None
        mock_agent.run.return_value = None

        # Should raise an error for None result since it indicates a problem
        with pytest.raises(Exception):
            await agent_runtime_service.execute_agent(
                agent=mock_agent,
                task=sample_task,
                context=sample_context,
                agent_type=AgentType.EXECUTOR
            )

    @pytest.mark.asyncio
    async def test_execute_agent_failure(self, agent_runtime_service, mock_agent, sample_task, sample_context):
        """Test agent execution failure handling."""
        await agent_runtime_service.initialize()

        # Setup agent to raise exception
        mock_agent.run.side_effect = Exception("Agent execution failed")

        # Execute should raise the exception
        with pytest.raises(Exception, match="Agent execution failed"):
            await agent_runtime_service.execute_agent(
                agent=mock_agent,
                task=sample_task,
                context=sample_context,
                agent_type=AgentType.EXECUTOR
            )

        # Verify error metrics updated
        assert agent_runtime_service._runtime_metrics["runtime_errors"] == 1

    @pytest.mark.asyncio
    async def test_execute_agent_event_emission(self, agent_runtime_service, mock_agent, sample_task,
                                                sample_context, mock_event_store):
        """Test that execution events are properly emitted."""
        await agent_runtime_service.initialize()

        # Setup successful execution
        mock_agent.run.return_value = {"output": "test result"}

        # Execute agent
        await agent_runtime_service.execute_agent(
            agent=mock_agent,
            task=sample_task,
            context=sample_context,
            agent_type=AgentType.EXECUTOR
        )

        # Verify events were emitted
        assert mock_event_store.append.call_count >= 2  # start and completion events

        # Check event types
        emitted_events = [call[0][0] for call in mock_event_store.append.call_args_list]
        event_types = [event.event_type for event in emitted_events]

        assert "runtime.agent_execution_started" in event_types
        assert "runtime.agent_execution_completed" in event_types

    @pytest.mark.asyncio
    async def test_shutdown(self, agent_runtime_service, mock_agent, mock_event_store):
        """Test service shutdown."""
        await agent_runtime_service.initialize()

        # Add some runtime state
        agent_runtime_service._runtime_agents["test"] = mock_agent
        agent_runtime_service._runtime_metrics["agents_executed"] = 5

        # Shutdown
        await agent_runtime_service.shutdown()

        # Verify cleanup
        assert len(agent_runtime_service._runtime_agents) == 0
        assert agent_runtime_service._initialized is False

        # Verify shutdown event was emitted
        mock_event_store.append.assert_called()
        shutdown_events = [call for call in mock_event_store.append.call_args_list
                          if "shutdown" in str(call)]
        assert len(shutdown_events) > 0

    def test_get_runtime_metrics(self, agent_runtime_service):
        """Test runtime metrics retrieval."""
        # Set some metrics
        agent_runtime_service._runtime_metrics["agents_executed"] = 10
        agent_runtime_service._runtime_agents["test"] = "agent"
        agent_runtime_service._initialized = True

        metrics = agent_runtime_service.get_runtime_metrics()

        assert metrics["agents_executed"] == 10
        assert metrics["runtime_agents_available"] == 1
        assert metrics["framework"] == "agno"
        assert metrics["initialized"] is True

    def test_get_framework_name(self, agent_runtime_service):
        """Test framework name retrieval."""
        assert agent_runtime_service.get_framework_name() == "agno"

    def test_is_initialized(self, agent_runtime_service):
        """Test initialization state check."""
        assert agent_runtime_service.is_initialized() is False

        agent_runtime_service._initialized = True
        assert agent_runtime_service.is_initialized() is True

    @pytest.mark.asyncio
    async def test_context_with_files(self, agent_runtime_service, mock_agent, sample_task):
        """Test execution with context containing files."""
        await agent_runtime_service.initialize()

        # Create context with file items
        from roma.domain.value_objects.context_item_type import ContextItemType

        context_with_files = TaskContext(
            task=sample_task,
            overall_objective="Test objective",
            execution_id="test-execution-with-files",
            context_items=[
                ContextItem(
                    content="test content",
                    item_type=ContextItemType.FILE_ARTIFACT,
                    metadata={"filename": "test.txt"}
                ),
                ContextItem(
                    content="image data",
                    item_type=ContextItemType.IMAGE_ARTIFACT,
                    metadata={"filename": "test.jpg"}
                )
            ],
            execution_metadata={}
        )

        mock_agent.run.return_value = {"output": "processed files"}

        # Execute with file context
        result_envelope = await agent_runtime_service.execute_agent(
            agent=mock_agent,
            task=sample_task,
            context=context_with_files,
            agent_type=AgentType.EXECUTOR
        )

        # Verify execution was successful
        assert result_envelope.success is True

        # Verify agent was called with the file context
        mock_agent.run.assert_called_once_with(sample_task, context_with_files)

    @pytest.mark.asyncio
    async def test_concurrent_executions(self, agent_runtime_service, mock_agent, sample_task, sample_context):
        """Test concurrent agent executions don't interfere."""
        await agent_runtime_service.initialize()

        # Setup agent to simulate some processing time
        async def slow_execution(task, context):
            import asyncio
            await asyncio.sleep(0.1)
            return {"output": f"Result for {task.task_id}"}

        mock_agent.run.side_effect = slow_execution

        # Create multiple tasks
        tasks = [
            TaskNode(task_id=f"task_{i}", goal=f"Goal {i}", task_type=TaskType.THINK)
            for i in range(3)
        ]

        # Execute concurrently
        import asyncio
        results = await asyncio.gather(*[
            agent_runtime_service.execute_agent(mock_agent, task, sample_context, AgentType.EXECUTOR)
            for task in tasks
        ])

        # Verify all executions completed
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.task_id == f"task_{i}"
            assert result.success is True

        # Verify metrics reflect all executions
        assert agent_runtime_service._runtime_metrics["agents_executed"] == 3
