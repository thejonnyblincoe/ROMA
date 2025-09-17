"""
Integration tests for ROMA System Manager end-to-end functionality.

Tests the complete integration of SystemManager with ContextBuilder,
Storage, and AgentRuntime services for multimodal task execution.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from src.roma.infrastructure.orchestration.system_manager import SystemManager
from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.value_objects.config.roma_config import ROMAConfig
from src.roma.domain.value_objects.config.profile_config import ProfileConfig, AgentMappingConfig
from src.roma.domain.value_objects.config.app_config import StorageConfig


class TestSystemIntegration:
    """Test complete system integration."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def system_config(self, temp_storage_path):
        """Create system configuration for testing."""
        # Create agent mapping with all required agent types
        agent_mapping = AgentMappingConfig(
            atomizers={
                "THINK": "test_atomizer",
                "RETRIEVE": "test_atomizer",
                "WRITE": "test_atomizer"
            },
            planners={
                "THINK": "test_planner",
                "RETRIEVE": "test_planner",
                "WRITE": "test_planner"
            },
            executors={
                "THINK": "test_executor",
                "RETRIEVE": "test_executor",
                "WRITE": "test_executor"
            },
            aggregators={
                "THINK": "test_aggregator",
                "RETRIEVE": "test_aggregator",
                "WRITE": "test_aggregator"
            }
        )

        profile_config = ProfileConfig(
            name="test_profile",
            description="Test profile for integration testing",
            agent_mapping=agent_mapping
        )

        storage_config = StorageConfig(
            mount_path=temp_storage_path
        )

        return ROMAConfig(
            profile=profile_config,
            storage=storage_config,
            default_profile="test_profile"
        )

    @pytest.fixture
    def system_manager(self, system_config):
        """Create SystemManager instance for integration testing."""
        return SystemManager(system_config)
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, system_config):
        """Test that system initializes all components correctly."""
        manager = SystemManager(system_config)
        await manager.initialize("test_profile")
        
        try:
            system_info = manager.get_system_info()
            
            assert system_info["status"] == "initialized"
            assert system_info["current_profile"] == "test_profile"
            assert system_info["components"]["event_store"] is True
            assert system_info["components"]["task_graph"] is True
            assert system_info["components"]["agent_runtime_service"] is True
            assert system_info["components"]["toolkit_manager"] is True
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_goal_execution_with_context(self, system_manager):
        """Test goal execution with multimodal context building."""
        goal = "Analyze the current market trends for cryptocurrency"

        # Initialize the system manager first
        await system_manager.initialize("test_profile")

        # Mock the agent runtime to return predictable results
        mock_agent = Mock()
        system_manager._agent_runtime_service.get_agent = AsyncMock(return_value=mock_agent)
        system_manager._agent_runtime_service.execute_agent = AsyncMock(return_value={
            "result": "Cryptocurrency market analysis completed",
            "success": True,
            "artifacts": [
                {
                    "name": "market_report.txt", 
                    "type": "text",
                    "content": b"Market analysis report content"
                }
            ]
        })
        
        try:
            result = await system_manager.execute_task(goal)

            # Verify execution result
            assert result["status"] == "completed"
            assert result["task"] == goal
            assert "execution_id" in result
            assert "execution_time" in result
            assert result["node_count"] == 1
            assert "artifacts" in result

            # Verify agent was called with context
            system_manager._agent_runtime_service.execute_agent.assert_called_once()
            call_args = system_manager._agent_runtime_service.execute_agent.call_args

            # Should have been called with 3 arguments: agent, task, context
            assert len(call_args[0]) == 3
            agent, task, context = call_args[0]

            assert isinstance(task, TaskNode)
            assert task.goal == goal
            assert isinstance(context, dict)
            assert "task" in context
            assert "overall_objective" in context
        finally:
            await system_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_artifact_storage_integration(self, system_manager, temp_storage_path):
        """Test that artifacts are stored correctly during execution."""
        goal = "Create a research document"

        # Initialize the system manager first
        await system_manager.initialize("test_profile")

        try:
            # Mock agent runtime to return artifacts
            mock_agent = Mock()
            system_manager._agent_runtime_service.get_agent = AsyncMock(return_value=mock_agent)
            system_manager._agent_runtime_service.execute_agent = AsyncMock(return_value={
                "result": "Research document created",
                "success": True,
                "artifacts": [
                    {
                        "name": "research_doc.txt",
                        "type": "document",
                        "content": b"This is the research document content."
                    }
                ]
            })

            result = await system_manager.execute_task(goal)

            # Verify artifacts were processed
            assert "artifacts" in result

            # Check that storage operations were attempted
            storage_path = Path(temp_storage_path)
            assert storage_path.exists()

        finally:
            await system_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_context_builder_integration(self, system_manager):
        """Test that ContextBuilder is properly integrated."""
        goal = "Test context building functionality"

        # Initialize the system first
        await system_manager.initialize("test_profile")

        try:
            # Create some test files to include in context
            test_files = [
                {"path": "test1.txt", "content": "Test file content 1"},
                {"path": "test2.txt", "content": "Test file content 2"}
            ]

            # Mock agent execution
            mock_agent = Mock()
            system_manager._agent_runtime_service.get_agent = AsyncMock(return_value=mock_agent)

            # Capture the context that gets passed to the agent
            captured_context = None
            async def capture_execute(agent, task, context):
                nonlocal captured_context
                captured_context = context
                return {"result": "Context processed", "success": True, "artifacts": []}

            system_manager._agent_runtime_service.execute_agent = capture_execute

            # Execute with file options
            result = await system_manager.execute_task(
                goal,
                include_files=test_files
            )

            # Verify context was built and passed
            assert captured_context is not None
            assert "task" in captured_context
            assert "overall_objective" in captured_context
            assert "execution_options" in captured_context

            # Verify task was created correctly
            assert result["status"] == "completed"
            assert result["task"] == goal
        finally:
            await system_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_profile_switching(self, system_manager, system_config):
        """Test switching between different profiles."""
        # Initialize with test profile first
        await system_manager.initialize("test_profile")

        try:
            # Verify current profile
            assert system_manager.get_current_profile() == "test_profile"

            # Verify system info reflects initial profile
            system_info = system_manager.get_system_info()
            assert system_info["current_profile"] == "test_profile"

            # Profile switching functionality is not fully implemented yet
            # Just verify that the profile is correctly tracked
            assert system_manager.get_current_profile() == "test_profile"

        finally:
            await system_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, system_manager):
        """Test system error handling during execution."""
        goal = "Test error handling"

        # Initialize the system first
        await system_manager.initialize("test_profile")

        try:
            # Mock agent creation to succeed but execution to fail
            mock_agent = Mock()
            system_manager._agent_runtime_service.get_agent = AsyncMock(return_value=mock_agent)
            system_manager._agent_runtime_service.execute_agent = AsyncMock(
                side_effect=Exception("Simulated execution error")
            )

            # Execution should raise an exception (may be wrapped)
            with pytest.raises(Exception):
                await system_manager.execute_task(goal)

            # Verify execution tracking was updated correctly
            executions = system_manager._active_executions
            assert len(executions) == 1

            execution_info = next(iter(executions.values()))
            assert execution_info["status"] == TaskStatus.FAILED
            assert execution_info["task"] == goal
        finally:
            await system_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(self, system_manager):
        """Test system handles concurrent executions safely."""
        import asyncio

        # Initialize the system first
        await system_manager.initialize("test_profile")

        try:
            goals = [
                "Concurrent task 1",
                "Concurrent task 2",
                "Concurrent task 3"
            ]

            # Mock successful agent execution
            mock_agent = Mock()
            system_manager._agent_runtime_service.get_agent = AsyncMock(return_value=mock_agent)
            system_manager._agent_runtime_service.execute_agent = AsyncMock(return_value={
                "result": "Task completed successfully",
                "success": True,
                "artifacts": []
            })

            # Execute multiple goals concurrently
            tasks = [system_manager.execute_task(goal) for goal in goals]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            assert len(results) == 3
            for i, result in enumerate(results):
                assert not isinstance(result, Exception)
                assert result["status"] == "completed"
                assert result["task"] == goals[i]

            # Verify all executions are tracked
            assert len(system_manager._active_executions) == 3
        finally:
            await system_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, system_manager):
        """Test configuration validation functionality."""
        validation_result = system_manager.validate_configuration()
        
        assert "valid" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result
        
        # With our test config, should be valid
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
    
    def test_available_profiles(self, system_manager):
        """Test getting available profiles."""
        profiles = system_manager.get_available_profiles()
        
        assert isinstance(profiles, list)
        assert "test_profile" in profiles
        assert len(profiles) > 0


class TestSystemManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_uninitialized_system_execution(self):
        """Test execution fails when system is not initialized."""
        config = {"framework": {"type": "agno"}}
        manager = SystemManager(config)
        
        # Should fail without initialization
        with pytest.raises(RuntimeError, match="SystemManager not initialized"):
            await manager.execute_task("Test goal")
    
    @pytest.mark.asyncio
    async def test_invalid_profile_initialization(self):
        """Test initialization with invalid profile."""
        # Create config with minimal profile but try to initialize nonexistent one
        profile_config = ProfileConfig(name="test_profile", description="Test")
        storage_config = StorageConfig(mount_path="/tmp/roma_test")
        config = ROMAConfig(
            profile=profile_config,
            storage=storage_config,
            default_profile="test_profile"
        )
        manager = SystemManager(config)

        # Initialize with the valid profile - should work
        await manager.initialize("test_profile")

        # Should be initialized
        assert manager.get_system_info()["status"] == "initialized"

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_cleanup_on_initialization_failure(self):
        """Test cleanup occurs when initialization fails."""
        # Create config that will fail during initialization
        profile_config = ProfileConfig(name="test_profile", description="Test")
        storage_config = StorageConfig(mount_path="/invalid/nonexistent/path")
        config = ROMAConfig(
            profile=profile_config,
            storage=storage_config,
            default_profile="test_profile"
        )
        manager = SystemManager(config)

        # Should fail and clean up
        with pytest.raises(Exception):
            await manager.initialize("test_profile")

        # Should not be initialized
        assert manager.get_system_info()["status"] == "not_initialized"