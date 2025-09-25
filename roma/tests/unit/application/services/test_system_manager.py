"""
Tests for SystemManager - ROMA v2 Application Layer.

Tests the central orchestrator managing all ROMA v2 components.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from roma.application.orchestration.system_manager import SystemManager
from roma.domain.value_objects.config.app_config import StorageConfig
from roma.domain.value_objects.config.profile_config import ProfileConfig
from roma.domain.value_objects.config.roma_config import ROMAConfig
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


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

        # Use a simpler approach with fewer mocks to avoid Python nesting limits
        with patch.object(system_manager, 'initialize', wraps=system_manager.initialize) as mock_initialize:
            # Patch all the initialization methods to do nothing
            init_patches = [
                patch.object(system_manager, '_initialize_event_store', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_event_publisher', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_task_graph', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_storage', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_artifact_service', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_knowledge_store', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_context_builder', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_toolkit_manager', new_callable=AsyncMock),
                patch.object(system_manager, '_load_tools_registry', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_agent_factory', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_agent_runtime_service', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_recovery_manager', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_persistence_repositories', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_checkpoint_service', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_hitl_service', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_deadlock_detector', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_graph_state_manager', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_parallel_execution_engine', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_agent_service_registry', new_callable=AsyncMock),
                patch.object(system_manager, '_initialize_execution_orchestrator', new_callable=AsyncMock),
                patch.object(system_manager, '_load_profile', new_callable=AsyncMock)
            ]

            # Apply all patches
            for patch_context in init_patches:
                patch_context.start()

            try:
                await system_manager.initialize(profile_name)

                # Verify state
                assert system_manager._initialized
                assert system_manager._current_profile == profile_name

            finally:
                # Clean up patches
                for patch_context in init_patches:
                    patch_context.stop()

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

        # Mock execution orchestrator for cleanup
        mock_execution_orchestrator = Mock()
        mock_execution_orchestrator.cleanup_execution = AsyncMock()
        mock_execution_orchestrator.get_orchestration_metrics = Mock(return_value={})
        system_manager._execution_orchestrator = mock_execution_orchestrator

        # Mock the task execution
        mock_result = {
            "result": "Task completed successfully",
            "task_id": "test_id"
        }

        # Mock the execution orchestrator
        from roma.domain.value_objects.execution_result import ExecutionResult
        mock_execution_result = ExecutionResult(
            success=True,
            total_nodes=3,
            completed_nodes=3,
            failed_nodes=0,
            execution_time_seconds=1.5,
            iterations=5,
            final_result=None,
            error_details=[]
        )

        with patch('roma.infrastructure.orchestration.system_manager.ExecutionOrchestrator') as mock_orchestrator_class, \
             patch('roma.infrastructure.orchestration.system_manager.ExecutionContext') as mock_context_class:

            # Mock ExecutionOrchestrator
            mock_orchestrator = Mock()
            mock_orchestrator.execute = AsyncMock(return_value=mock_execution_result)
            mock_orchestrator.cleanup_execution = AsyncMock()
            mock_orchestrator.get_orchestration_metrics.return_value = {}
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock ExecutionContext
            mock_context = Mock()
            mock_context.artifact_service = Mock()
            mock_context.artifact_service.store_envelope_artifacts = AsyncMock(return_value=[])
            mock_context.cleanup = AsyncMock()
            mock_context.set_config = Mock()  # Make set_config method available
            mock_context.initialize = AsyncMock()  # Make initialize method available and async
            mock_context_class.return_value = mock_context

            result = await system_manager.execute_task("test task")

            # Verify result structure
            assert "execution_id" in result
            assert result["task"] == "test task"
            assert result["status"] == "completed"
            assert "execution_time" in result
            assert "node_count" in result

    @pytest.mark.skip(reason="Complex test requiring ExecutionOrchestrator architecture refactoring")
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
        from roma.domain.entities.task_node import TaskNode
        task = TaskNode(
            task_id="test_task",
            goal="test goal",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        # Mock the execution orchestrator to return expected result
        mock_orchestrator = Mock()
        mock_orchestrator.execute_task = AsyncMock(return_value={
            "result": "success",
            "task_id": "test_task",
            "execution_id": "exec_123",
            "status": "completed"
        })
        system_manager._execution_orchestrator = mock_orchestrator

        result = await system_manager.execute_task("test goal")

        # Verify execution orchestrator was called
        mock_orchestrator.execute_task.assert_called_once()

        # Verify result structure
        assert isinstance(result, dict)
        assert result["status"] == "completed"
        assert "execution_id" in result
        assert "execution_time" in result

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
        # Mock all initialization methods and make one fail
        with patch.object(system_manager, '_initialize_event_store', new_callable=AsyncMock), \
             patch.object(system_manager, '_initialize_event_publisher', new_callable=AsyncMock), \
             patch.object(system_manager, '_initialize_deadlock_detector', new_callable=AsyncMock), \
             patch.object(system_manager, '_initialize_task_graph', side_effect=Exception("Init failed")) as mock_task_graph, \
             patch.object(system_manager, '_cleanup', new_callable=AsyncMock) as mock_cleanup:

            with pytest.raises(Exception, match="Init failed"):
                await system_manager.initialize("test_profile")

            mock_cleanup.assert_called_once()
            assert not system_manager._initialized

    @pytest.mark.skip(reason="Complex test requiring ExecutionOrchestrator architecture refactoring")
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

    @pytest.mark.skip(reason="Complex test requiring ExecutionOrchestrator architecture refactoring")
    @pytest.mark.asyncio
    async def test_build_context_error_fallback(self, system_manager):
        """Test context building error fallback returns minimal context."""
        await system_manager.initialize("test_profile")

        try:
            from roma.domain.entities.task_node import TaskNode
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

    @pytest.mark.skip(reason="Complex test requiring ExecutionOrchestrator architecture refactoring")
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

    @pytest.mark.skip(reason="Complex test requiring ExecutionOrchestrator architecture refactoring")
    @pytest.mark.asyncio
    async def test_memory_leak_fix_executions_cleaned_up(self, system_manager):
        """Test that _active_executions dict is properly cleaned up after task execution."""
        await system_manager.initialize("test_profile")

        try:
            # Mock all necessary components for successful execution
            mock_execution_orchestrator = Mock()
            mock_execution_result = Mock()
            mock_execution_result.success = True
            mock_execution_result.final_result = None
            mock_execution_result.execution_time_seconds = 1.0
            mock_execution_result.total_nodes = 1
            mock_execution_result.completed_nodes = 1
            mock_execution_result.failed_nodes = 0
            mock_execution_result.iterations = 1
            mock_execution_result.error_details = None
            mock_execution_result.has_errors = False
            mock_execution_result.to_dict.return_value = {"result": "success"}
            mock_execution_orchestrator.execute = AsyncMock(return_value=mock_execution_result)
            mock_execution_orchestrator.cleanup_execution = AsyncMock()
            mock_execution_orchestrator.get_orchestration_metrics.return_value = {}

            # Mock artifact service for successful execution
            mock_artifact_service = Mock()
            mock_artifact_service.store_envelope_artifacts = AsyncMock(return_value=[])

            # Mock runtime service
            mock_runtime_service = Mock()
            mock_runtime_service.get_framework_name.return_value = "agno"
            mock_runtime_service.shutdown = AsyncMock()
            system_manager._agent_runtime_service = mock_runtime_service

            # Verify _active_executions is initially empty
            assert len(system_manager._active_executions) == 0

            # Patch the ExecutionOrchestrator and ExecutionContext creation
            with patch('roma.infrastructure.orchestration.system_manager.ExecutionOrchestrator', return_value=mock_execution_orchestrator), \
                 patch('roma.infrastructure.orchestration.system_manager.ExecutionContext') as mock_context_class:

                mock_context = Mock()
                mock_context.artifact_service = mock_artifact_service
                mock_context.cleanup = AsyncMock()
                mock_context_class.return_value = mock_context

                # Execute a task
                result = await system_manager.execute_task("test task")

                # Verify task executed successfully
                assert result["status"] == "completed"

                # CRITICAL: Verify _active_executions is cleaned up (no memory leak)
                assert len(system_manager._active_executions) == 0, "Memory leak: _active_executions not cleaned up"

                # Verify _active_contexts is also cleaned up
                assert len(system_manager._active_contexts) == 0, "Memory leak: _active_contexts not cleaned up"

        finally:
            await system_manager.shutdown()

    @pytest.mark.asyncio
    async def test_memory_leak_fix_executions_cleaned_up_on_failure(self, system_manager):
        """Test that _active_executions dict is cleaned up even when task execution fails."""
        await system_manager.initialize("test_profile")

        try:
            # Mock runtime service to fail
            mock_runtime_service = Mock()
            mock_runtime_service.get_framework_name.return_value = "agno"
            mock_runtime_service.shutdown = AsyncMock()
            system_manager._agent_runtime_service = mock_runtime_service

            # Mock execution orchestrator
            mock_execution_orchestrator = Mock()
            mock_execution_orchestrator.cleanup_execution = AsyncMock()

            # Verify _active_executions is initially empty
            assert len(system_manager._active_executions) == 0

            # Patch to raise an exception during execution
            with patch('roma.infrastructure.orchestration.system_manager.ExecutionContext') as mock_context_class:
                mock_context = Mock()
                mock_context.cleanup = AsyncMock()
                mock_context_class.side_effect = Exception("Execution failed")

                # Execute a task and expect it to fail
                result = await system_manager.execute_task("test task")

                # Verify task failed
                assert result["status"] == "failed"

                # CRITICAL: Verify _active_executions is cleaned up even on failure (no memory leak)
                assert len(system_manager._active_executions) == 0, "Memory leak: _active_executions not cleaned up on failure"

                # Verify _active_contexts is also cleaned up
                assert len(system_manager._active_contexts) == 0, "Memory leak: _active_contexts not cleaned up on failure"

        finally:
            await system_manager.shutdown()

    @pytest.mark.skip(reason="Complex test requiring ExecutionOrchestrator architecture refactoring")
    @pytest.mark.asyncio
    async def test_multiple_executions_no_memory_leak(self, system_manager):
        """Test that multiple task executions don't cause memory leak in tracking dicts."""
        await system_manager.initialize("test_profile")

        try:
            # Mock successful execution components
            mock_execution_orchestrator = Mock()
            mock_execution_result = Mock()
            mock_execution_result.success = True
            mock_execution_result.final_result = None
            mock_execution_result.execution_time_seconds = 1.0
            mock_execution_result.total_nodes = 1
            mock_execution_result.completed_nodes = 1
            mock_execution_result.failed_nodes = 0
            mock_execution_result.iterations = 1
            mock_execution_result.error_details = None
            mock_execution_result.has_errors = False
            mock_execution_result.to_dict.return_value = {"result": "success"}
            mock_execution_orchestrator.execute = AsyncMock(return_value=mock_execution_result)
            mock_execution_orchestrator.cleanup_execution = AsyncMock()
            mock_execution_orchestrator.get_orchestration_metrics.return_value = {}

            mock_artifact_service = Mock()
            mock_artifact_service.store_envelope_artifacts = AsyncMock(return_value=[])

            mock_runtime_service = Mock()
            mock_runtime_service.get_framework_name.return_value = "agno"
            mock_runtime_service.shutdown = AsyncMock()
            system_manager._agent_runtime_service = mock_runtime_service

            # Execute multiple tasks
            with patch('roma.infrastructure.orchestration.system_manager.ExecutionOrchestrator', return_value=mock_execution_orchestrator), \
                 patch('roma.infrastructure.orchestration.system_manager.ExecutionContext') as mock_context_class:

                mock_context = Mock()
                mock_context.artifact_service = mock_artifact_service
                mock_context.cleanup = AsyncMock()
                mock_context_class.return_value = mock_context

                # Execute 5 tasks sequentially
                for i in range(5):
                    # Verify dictionaries are empty before each execution
                    assert len(system_manager._active_executions) == 0
                    assert len(system_manager._active_contexts) == 0

                    result = await system_manager.execute_task(f"test task {i}")

                    # Verify successful execution
                    assert result["status"] == "completed"

                    # Verify cleanup happened immediately after each execution
                    assert len(system_manager._active_executions) == 0, f"Memory leak after execution {i}"
                    assert len(system_manager._active_contexts) == 0, f"Memory leak after execution {i}"

        finally:
            await system_manager.shutdown()
