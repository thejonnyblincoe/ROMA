"""
Tests for AgentRuntimeService - ROMA v2 Application Layer.

Tests the application service for managing agent runtime lifecycle.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.roma.application.services.agent_runtime_service import AgentRuntimeService
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.value_objects.agent_type import AgentType
from src.roma.domain.entities.task_node import TaskNode


@pytest.fixture
def mock_runtime():
    """Create mock agent runtime."""
    runtime = Mock()
    runtime.initialize = AsyncMock()
    runtime.create_agent = AsyncMock()
    runtime.execute_agent = AsyncMock()
    runtime.emit_event = AsyncMock()
    runtime.register_toolkit = AsyncMock()
    runtime.get_registered_toolkits.return_value = []
    runtime.get_framework_name.return_value = "agno"
    runtime.resolve_agent_config.return_value = {"name": "test_agent", "model": "gpt-4o"}
    return runtime


@pytest.fixture
def mock_event_store():
    """Create mock event store."""
    event_store = Mock()
    event_store.initialize = AsyncMock()
    event_store.shutdown = AsyncMock()
    event_store.append = AsyncMock()  # For event emissions
    return event_store


@pytest.fixture
def mock_toolkit_manager():
    """Create mock toolkit manager."""
    toolkit_manager = Mock()
    toolkit_manager.initialize = AsyncMock()
    return toolkit_manager


@pytest.fixture
def mock_agent_factory():
    """Create mock agent factory."""
    factory = Mock()
    factory.initialize = AsyncMock()
    factory.create_agent = AsyncMock()

    # Return complete agent config
    complete_config = {
        "name": "test_agent",
        "type": "atomizer",
        "task_type": "THINK",
        "description": "Test agent for atomization",
        "model": {
            "provider": "litellm",
            "name": "gpt-4o",
            "temperature": 0.7
        },
        "enabled": True
    }
    factory.get_agent_config = Mock(return_value=complete_config)
    factory.get_available_agents.return_value = ["test_agent"]
    return factory


@pytest.fixture
def agent_runtime_service(mock_event_store, mock_agent_factory):
    """Create AgentRuntimeService instance for testing."""
    return AgentRuntimeService(
        event_store=mock_event_store,
        agent_factory=mock_agent_factory
    )


class TestAgentRuntimeService:
    """Test cases for AgentRuntimeService."""
    
    def test_initialization(self, agent_runtime_service):
        """Test service initialization."""
        assert not agent_runtime_service._initialized
        assert len(agent_runtime_service._runtime_agents) == 0
        assert agent_runtime_service._runtime_metrics["agents_created"] == 0
        
    @pytest.mark.asyncio
    async def test_initialize(self, agent_runtime_service):
        """Test service initialization process."""
        await agent_runtime_service.initialize()

        # Verify initialization state
        assert agent_runtime_service._initialized
        assert agent_runtime_service.is_initialized()
        
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, agent_runtime_service):
        """Test initialization when already initialized."""
        agent_runtime_service._initialized = True

        # Should not fail when already initialized
        await agent_runtime_service.initialize()
        assert agent_runtime_service._initialized
            
    @pytest.mark.asyncio
    async def test_get_agent_with_enums(self, agent_runtime_service, mock_agent_factory):
        """Test agent retrieval with enum parameters."""
        await agent_runtime_service.initialize()

        mock_agent = Mock()
        mock_agent_factory.create_agent.return_value = mock_agent

        result = await agent_runtime_service.get_agent(TaskType.THINK, AgentType.ATOMIZER)
        
        # Verify agent creation
        assert result == mock_agent
        assert agent_runtime_service._runtime_metrics["agents_created"] == 1

        # Verify agent factory was called
        mock_agent_factory.create_agent.assert_called_once()

        # Verify agent is cached
        agent_key = f"{TaskType.THINK.value}_{AgentType.ATOMIZER.value}"
        assert agent_key in agent_runtime_service._runtime_agents
        
    @pytest.mark.asyncio
    async def test_get_agent_with_strings(self, agent_runtime_service, mock_agent_factory):
        """Test agent retrieval with string parameters converted to enums."""
        await agent_runtime_service.initialize()

        mock_agent = Mock()
        mock_agent_factory.create_agent.return_value = mock_agent

        # Test string conversion to enums
        result = await agent_runtime_service.get_agent(TaskType.WRITE, AgentType.PLANNER)

        # Verify agent creation
        assert result == mock_agent
        
    @pytest.mark.asyncio
    async def test_get_agent_from_cache(self, agent_runtime_service, mock_agent_factory):
        """Test agent retrieval from cache."""
        await agent_runtime_service.initialize()

        # Get agent first time
        mock_agent = Mock()
        mock_agent_factory.create_agent.return_value = mock_agent

        first_result = await agent_runtime_service.get_agent(TaskType.RETRIEVE, AgentType.EXECUTOR)

        # Get same agent second time (should come from cache)
        second_result = await agent_runtime_service.get_agent(TaskType.RETRIEVE, AgentType.EXECUTOR)
        
        # Verify same instance returned
        assert first_result == second_result
        
        # Verify both results are the same (cached)
        assert first_result == second_result

        # Verify factory only called once (first time, second from cache)
        assert mock_agent_factory.create_agent.call_count == 1
        
    @pytest.mark.asyncio
    async def test_get_agent_not_initialized(self, agent_runtime_service):
        """Test agent retrieval when service not initialized."""
        with pytest.raises(RuntimeError):
            await agent_runtime_service.get_agent(TaskType.THINK, AgentType.ATOMIZER)
            
    @pytest.mark.asyncio
    async def test_execute_agent(self, agent_runtime_service):
        """Test agent execution."""
        await agent_runtime_service.initialize()

        # Create test task
        task = TaskNode(
            task_id="test_task",
            goal="test goal",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent.run = AsyncMock(return_value={"result": "success", "confidence": 0.9})

        result = await agent_runtime_service.execute_agent(mock_agent, task)

        # Verify execution
        mock_agent.run.assert_called_once_with(task, {})
        assert result["success"] is True
        assert agent_runtime_service._runtime_metrics["agents_executed"] == 1
        
    @pytest.mark.asyncio
    async def test_execute_agent_failure(self, agent_runtime_service):
        """Test agent execution failure handling."""
        await agent_runtime_service.initialize()

        task = TaskNode(
            task_id="test_task",
            goal="test goal",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent.run = AsyncMock(side_effect=Exception("Execution failed"))

        with pytest.raises(Exception, match="Execution failed"):
            await agent_runtime_service.execute_agent(mock_agent, task)

        # Verify error metrics updated
        assert agent_runtime_service._runtime_metrics["runtime_errors"] == 1
        
    def test_get_runtime_metrics(self, agent_runtime_service):
        """Test getting runtime metrics."""
        metrics = agent_runtime_service.get_runtime_metrics()

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "agents_created" in metrics
        assert "agents_executed" in metrics
        assert "runtime_errors" in metrics
        assert "framework" in metrics

        # Verify initial values
        assert metrics["agents_created"] == 0
        assert metrics["agents_executed"] == 0
        assert metrics["runtime_errors"] == 0
        
    def test_get_framework_name(self, agent_runtime_service):
        """Test getting framework name."""
        result = agent_runtime_service.get_framework_name()
        assert result == "agno"
        
    def test_is_initialized(self, agent_runtime_service):
        """Test initialization status check."""
        assert not agent_runtime_service.is_initialized()
        
        agent_runtime_service._initialized = True
        assert agent_runtime_service.is_initialized()
        
    @pytest.mark.asyncio
    async def test_shutdown(self, agent_runtime_service):
        """Test service shutdown."""
        agent_runtime_service._initialized = True

        # Add some cached agents
        agent_runtime_service._runtime_agents["test"] = Mock()

        await agent_runtime_service.shutdown()

        # Verify cleanup
        assert len(agent_runtime_service._runtime_agents) == 0
        assert not agent_runtime_service._initialized
        
    @pytest.mark.asyncio
    async def test_shutdown_not_initialized(self, agent_runtime_service):
        """Test shutdown when not initialized."""
        # Should not raise exception
        await agent_runtime_service.shutdown()
        
    @pytest.mark.asyncio
    async def test_get_agent_factory_error(self, agent_runtime_service, mock_agent_factory):
        """Test agent retrieval when factory fails."""
        await agent_runtime_service.initialize()

        # Mock factory to raise error
        mock_agent_factory.get_agent_config.side_effect = Exception("Config not found")

        with pytest.raises(RuntimeError, match="Agent THINK_atomizer not available"):
            await agent_runtime_service.get_agent(TaskType.THINK, AgentType.ATOMIZER)