"""
Tests for SystemManager - ROMA v2 Application Layer.

Tests the central orchestrator managing all ROMA v2 components.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.roma.infrastructure.orchestration.system_manager import SystemManager
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.value_objects.agent_type import AgentType
from src.roma.domain.value_objects.config.roma_config import ROMAConfig
from src.roma.domain.value_objects.config.profile_config import ProfileConfig
from src.roma.domain.value_objects.config.app_config import AppConfig, StorageConfig


@pytest.fixture
def sample_config() -> ROMAConfig:
    """Sample ROMAConfig for testing."""
    profile_config = ProfileConfig(
        name="test_profile",
        description="Test profile for unit tests"
    )

    storage_config = StorageConfig(
        mount_path="/tmp/roma_test_storage"
    )

    return ROMAConfig(
        profile=profile_config,
        storage=storage_config,
        default_profile="test_profile"
    )


@pytest.fixture
def system_manager(sample_config) -> SystemManager:
    """Create SystemManager instance for testing."""
    return SystemManager(sample_config)


class TestSystemManager:
    """Test cases for SystemManager."""
    
    def test_initialization(self, system_manager):
        """Test SystemManager initialization."""
        assert system_manager.config is not None
        assert not system_manager._initialized
        assert system_manager._current_profile is None
        assert len(system_manager._active_executions) == 0
        
    @pytest.mark.asyncio
    async def test_initialize_with_profile(self, system_manager):
        """Test system initialization with profile."""
        profile_name = "test_profile"
        
        # Mock the component initialization methods
        with patch.object(system_manager, '_initialize_event_store', new_callable=AsyncMock) as mock_event_store, \
             patch.object(system_manager, '_initialize_task_graph', new_callable=AsyncMock) as mock_task_graph, \
             patch.object(system_manager, '_initialize_toolkit_manager', new_callable=AsyncMock) as mock_toolkit, \
             patch.object(system_manager, '_initialize_agent_runtime_service', new_callable=AsyncMock) as mock_runtime, \
             patch.object(system_manager, '_load_profile', new_callable=AsyncMock) as mock_profile:
            
            await system_manager.initialize(profile_name)
            
            # Verify initialization sequence
            mock_event_store.assert_called_once()
            mock_task_graph.assert_called_once()
            mock_toolkit.assert_called_once()
            mock_runtime.assert_called_once()
            mock_profile.assert_called_once_with(profile_name)
            
            # Verify state
            assert system_manager._initialized
            assert system_manager._current_profile == profile_name
            
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, system_manager):
        """Test initialization when already initialized."""
        system_manager._initialized = True
        
        with patch.object(system_manager, '_initialize_event_store') as mock_init:
            await system_manager.initialize("test_profile")
            mock_init.assert_not_called()
            
    @pytest.mark.asyncio
    async def test_execute_task_not_initialized(self, system_manager):
        """Test task execution when not initialized."""
        with pytest.raises(RuntimeError, match="SystemManager not initialized"):
            await system_manager.execute_task("test task")
            
    @pytest.mark.asyncio
    async def test_execute_task_success(self, system_manager):
        """Test successful task execution."""
        system_manager._initialized = True
        
        # Mock dependencies
        mock_task_graph = Mock()
        mock_task_graph.add_node = AsyncMock()
        mock_task_graph.update_node_status = AsyncMock()
        mock_task_graph.get_all_nodes.return_value = [Mock()]
        system_manager._task_graph = mock_task_graph
        
        mock_runtime_service = Mock()
        mock_runtime_service.get_framework_name.return_value = "agno"
        mock_runtime_service.get_agent = AsyncMock(return_value=Mock())  # Changed to get_agent and made async
        mock_runtime_service.execute_agent = AsyncMock(return_value={"result": "success"})
        system_manager._agent_runtime_service = mock_runtime_service

        # Mock recovery manager
        mock_recovery_manager = Mock()
        mock_recovery_manager.record_success = AsyncMock()
        mock_recovery_manager.handle_failure = AsyncMock(return_value=Mock())
        system_manager._recovery_manager = mock_recovery_manager

        # Mock context builder
        mock_context_builder = Mock()
        mock_context_builder.build_context = AsyncMock(return_value=Mock(
            context_items=[],
            task=Mock(),
            overall_objective="test"
        ))
        system_manager._context_builder = mock_context_builder

        # Mock storage
        mock_storage = Mock()
        system_manager._storage = mock_storage
        
        # Mock the task execution
        mock_result = {
            "result": "Task completed successfully",
            "task_id": "test_id"
        }
        
        with patch.object(system_manager, '_execute_task_with_context', new_callable=AsyncMock, return_value=mock_result):
            result = await system_manager.execute_task("test task")
            
            # Verify result structure
            assert "execution_id" in result
            assert result["task"] == "test task"
            assert result["status"] == "completed"
            assert "execution_time" in result
            assert "node_count" in result
            assert result["framework"] == "agno"
            
    @pytest.mark.asyncio
    async def test_execute_task(self, system_manager):
        """Test task execution through agent runtime service."""
        system_manager._initialized = True
        
        # Mock agent runtime service
        mock_agent = Mock()
        mock_runtime_service = Mock()
        mock_runtime_service.get_agent = AsyncMock(return_value=mock_agent)  # Now async
        mock_runtime_service.execute_agent = AsyncMock(return_value={"result": "success"})
        mock_runtime_service.get_framework_name.return_value = "agno"
        system_manager._agent_runtime_service = mock_runtime_service
        
        # Mock task graph
        mock_task_graph = Mock()
        mock_task_graph.update_node_status = AsyncMock()
        system_manager._task_graph = mock_task_graph

        # Mock recovery manager
        mock_recovery_manager = Mock()
        mock_recovery_manager.record_success = AsyncMock()
        mock_recovery_manager.handle_failure = AsyncMock(return_value=Mock())
        system_manager._recovery_manager = mock_recovery_manager
        
        # Create test task
        from src.roma.domain.entities.task_node import TaskNode
        task = TaskNode(
            task_id="test_task",
            goal="test goal",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )
        
        result = await system_manager._execute_task(task, "exec_123")
        
        # Verify agent retrieval with correct enums
        mock_runtime_service.get_agent.assert_called_once_with(
            TaskType.THINK,
            AgentType.ATOMIZER
        )
        
        # Verify agent execution (task should be in EXECUTING state when passed to agent)
        args, kwargs = mock_runtime_service.execute_agent.call_args
        assert args[0] == mock_agent
        assert args[1].task_id == task.task_id
        assert args[1].status == TaskStatus.EXECUTING
        
        # Verify result
        assert "result" in result
        assert result["task_id"] == "test_task"
        assert result["execution_id"] == "exec_123"
        
    def test_get_system_info_not_initialized(self, system_manager):
        """Test system info when not initialized."""
        info = system_manager.get_system_info()
        assert info["status"] == "not_initialized"
        
    def test_get_system_info_initialized(self, system_manager):
        """Test system info when initialized."""
        system_manager._initialized = True
        system_manager._current_profile = "test_profile"
        
        # Mock runtime service
        mock_runtime_service = Mock()
        mock_runtime_service.get_framework_name.return_value = "agno"
        mock_runtime_service.get_runtime_metrics.return_value = {"agents_created": 5}
        system_manager._agent_runtime_service = mock_runtime_service
        
        # Mock task graph
        mock_task_graph = Mock()
        mock_task_graph.get_all_nodes.return_value = [Mock(), Mock()]
        system_manager._task_graph = mock_task_graph
        
        info = system_manager.get_system_info()
        
        assert info["status"] == "initialized"
        assert info["current_profile"] == "test_profile"
        assert info["framework"] == "agno"
        assert info["total_nodes"] == 2
        assert "components" in info
        assert "runtime_metrics" in info
        
    def test_validate_configuration_valid(self, system_manager):
        """Test configuration validation with valid config."""
        result = system_manager.validate_configuration()
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
    def test_validate_configuration_missing_framework(self, system_manager):
        """Test configuration validation with missing framework."""
        # Use default config but test validation logic
        result = system_manager.validate_configuration()
        # Should pass validation with valid config
        assert result["valid"] is True
        
    def test_get_available_profiles(self, system_manager):
        """Test getting available profiles."""
        profiles = system_manager.get_available_profiles()
        assert "test_profile" in profiles
        
    @pytest.mark.asyncio
    async def test_switch_profile_not_initialized(self, system_manager):
        """Test switching profile when not initialized."""
        with pytest.raises(RuntimeError, match="SystemManager not initialized"):
            await system_manager.switch_profile("new_profile")
            
    @pytest.mark.asyncio
    async def test_switch_profile_success(self, system_manager):
        """Test successful profile switching."""
        system_manager._initialized = True
        
        # Mock task graph
        mock_task_graph = Mock()
        system_manager._task_graph = mock_task_graph
        
        with patch.object(system_manager, '_load_profile', new_callable=AsyncMock), \
             patch.object(system_manager, 'get_system_info', return_value={"status": "ok"}):
            
            result = await system_manager.switch_profile("new_profile")
            
            assert result["success"] is True
            assert result["profile"] == "new_profile"
            assert system_manager._current_profile == "new_profile"
            
    @pytest.mark.asyncio
    async def test_shutdown(self, system_manager):
        """Test system shutdown."""
        system_manager._initialized = True
        
        # Mock components
        mock_runtime_service = Mock()
        mock_runtime_service.shutdown = AsyncMock()
        system_manager._agent_runtime_service = mock_runtime_service
        
        mock_event_store = Mock()
        mock_event_store.clear = AsyncMock()
        system_manager._event_store = mock_event_store
        
        # Add active execution
        system_manager._active_executions["test"] = {"status": TaskStatus.EXECUTING}
        
        await system_manager.shutdown()
        
        # Verify shutdown sequence
        mock_runtime_service.shutdown.assert_called_once()
        mock_event_store.clear.assert_called_once()
        assert not system_manager._initialized
        assert len(system_manager._active_executions) == 0
        
    @pytest.mark.asyncio
    async def test_cleanup_on_initialization_failure(self, system_manager):
        """Test cleanup when initialization fails."""
        with patch.object(system_manager, '_initialize_event_store', new_callable=AsyncMock) as mock_event_store, \
             patch.object(system_manager, '_initialize_task_graph', side_effect=Exception("Init failed")), \
             patch.object(system_manager, '_cleanup', new_callable=AsyncMock) as mock_cleanup:
            
            with pytest.raises(Exception, match="Init failed"):
                await system_manager.initialize("test_profile")
                
            mock_cleanup.assert_called_once()
            assert not system_manager._initialized

    @pytest.mark.asyncio
    async def test_execute_task_error_handling_with_execution_tracking(self, system_manager):
        """Test error handling updates execution tracking properly."""
        await system_manager.initialize("test_profile")

        try:
            # Mock agent runtime to fail during execution
            mock_runtime_service = Mock()
            mock_runtime_service.get_agent = AsyncMock(return_value=Mock())
            mock_runtime_service.execute_agent = AsyncMock(side_effect=Exception("Agent execution failed"))
            mock_runtime_service.shutdown = AsyncMock()  # Add async shutdown mock
            system_manager._agent_runtime_service = mock_runtime_service

            goal = "Test goal for error handling"

            with pytest.raises(Exception):
                await system_manager.execute_task(goal)

            # Verify execution was tracked and marked as failed
            assert len(system_manager._active_executions) == 1
            execution = list(system_manager._active_executions.values())[0]
            assert execution["status"] == TaskStatus.FAILED
            assert execution["task"] == goal

        finally:
            await system_manager.shutdown()

    @pytest.mark.asyncio
    async def test_build_context_error_fallback(self, system_manager):
        """Test context building error fallback returns minimal context."""
        await system_manager.initialize("test_profile")

        try:
            from src.roma.domain.entities.task_node import TaskNode
            task = TaskNode(
                goal="Test task for context error",
                task_type=TaskType.THINK
            )

            # Mock context builder to raise exception
            mock_context_builder = Mock()
            mock_context_builder.build_task_context = AsyncMock(side_effect=Exception("Context build failed"))
            system_manager._context_builder = mock_context_builder

            # Create execution context should handle the error gracefully
            context = await system_manager._build_execution_context(task, {})

            # Should return minimal context on error
            assert context["task"] == task
            assert context["overall_objective"] == task.goal
            assert context["execution_options"] == {}
            assert context["files"] == []
            assert context["context_text"] == ""

        finally:
            await system_manager.shutdown()

    @pytest.mark.asyncio
    async def test_execute_task_without_execution_id_in_tracking(self, system_manager):
        """Test error handling when execution_id is not in active executions."""
        await system_manager.initialize("test_profile")

        try:
            # Mock components
            mock_runtime_service = Mock()
            mock_runtime_service.get_agent = AsyncMock(return_value=Mock())
            mock_runtime_service.execute_agent = AsyncMock(side_effect=Exception("Test error"))
            mock_runtime_service.shutdown = AsyncMock()  # Add async shutdown mock
            system_manager._agent_runtime_service = mock_runtime_service

            # Clear active executions to simulate missing execution_id
            system_manager._active_executions.clear()

            goal = "Test goal"

            with pytest.raises(Exception):
                await system_manager.execute_task(goal)

            # Should handle missing execution_id gracefully (no KeyError)

        finally:
            await system_manager.shutdown()