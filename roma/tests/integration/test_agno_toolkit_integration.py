"""
Integration tests for Agno Toolkit system in ROMA v2.

Tests both default Agno toolkits and custom toolkit implementations
to validate end-to-end integration with the SystemManager.
"""

import tempfile
from unittest.mock import patch

import pytest

from roma.application.orchestration.system_manager import SystemManager
from roma.domain.value_objects.config.roma_config import ROMAConfig
from roma.infrastructure.toolkits.agno_toolkit_manager import AgnoToolkitManager
from roma.infrastructure.toolkits.base_agno_toolkit import BaseAgnoToolkit, ToolkitValidationMixin
from roma.infrastructure.toolkits.custom.crypto.binance_toolkit import BinanceToolkit


class TestAgnoToolkitIntegration:
    """Test Agno toolkit integration end-to-end."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def roma_config(self, temp_storage_path):
        """Create ROMA configuration for toolkit testing."""
        # Use the same format as working system integration tests
        from roma.domain.value_objects.config.app_config import StorageConfig
        from roma.domain.value_objects.config.profile_config import (
            AgentMappingConfig,
            ProfileConfig,
        )

        # Create agent mapping with all required agent types
        agent_mapping = AgentMappingConfig(
            atomizers={
                "THINK": "toolkit_test_atomizer",
                "RETRIEVE": "toolkit_test_atomizer",
                "WRITE": "toolkit_test_atomizer"
            },
            planners={
                "THINK": "toolkit_test_planner",
                "RETRIEVE": "toolkit_test_planner",
                "WRITE": "toolkit_test_planner"
            },
            executors={
                "THINK": "toolkit_test_executor",
                "RETRIEVE": "toolkit_test_executor",
                "WRITE": "toolkit_test_executor"
            },
            aggregators={
                "THINK": "toolkit_test_aggregator",
                "RETRIEVE": "toolkit_test_aggregator",
                "WRITE": "toolkit_test_aggregator"
            }
        )

        profile_config = ProfileConfig(
            name="toolkit_test_profile",
            description="Profile for testing toolkit integration",
            agent_mapping=agent_mapping
        )

        storage_config = StorageConfig(
            mount_path=temp_storage_path
        )

        return ROMAConfig(
            profile=profile_config,
            storage=storage_config,
            default_profile="toolkit_test_profile"
        )

    @pytest.mark.asyncio
    async def test_agno_toolkit_manager_initialization(self):
        """Test that AgnoToolkitManager initializes correctly."""
        manager = AgnoToolkitManager()
        await manager.initialize()

        assert manager.is_connected() is True

        # Test basic CRUD operations
        available_toolkits = await manager.list_available_toolkits()
        assert isinstance(available_toolkits, list)

    @pytest.mark.asyncio
    async def test_custom_binance_toolkit_creation(self):
        """Test creating custom BinanceToolkit through AgnoToolkitManager."""
        manager = AgnoToolkitManager()
        await manager.initialize()

        # Test custom toolkit creation
        binance_spec = {
            "name": "binance_test",
            "type": "crypto",
            "implementation": "src.roma.infrastructure.toolkits.custom.crypto.binance_toolkit.BinanceToolkit",
            "config": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "default_market_type": "spot"
            }
        }

        toolkit = await manager.create_toolkit(binance_spec)

        assert toolkit is not None
        assert isinstance(toolkit, BinanceToolkit)
        assert isinstance(toolkit, BaseAgnoToolkit)
        assert isinstance(toolkit, ToolkitValidationMixin)  # Should support validation

        # Test toolkit functionality
        tools = toolkit.get_available_tools()
        assert "get_current_price" in tools
        assert "get_order_book" in tools
        assert "get_klines" in tools

        # Test validation
        health_result = await toolkit.run_full_validation()
        assert "toolkit_name" in health_result
        assert "overall_healthy" in health_result
        assert "checks" in health_result

    @pytest.mark.asyncio
    async def test_default_agno_toolkit_wrapper(self):
        """Test DefaultAgnoToolkitWrapper for standard Agno toolkits."""
        manager = AgnoToolkitManager()
        await manager.initialize()

        # Test default toolkit creation (would import from Agno in real scenario)
        default_spec = {
            "name": "web_search_toolkit",
            "type": "web_search",
            "config": {
                "search_engine": "google",
                "api_key": "test_key"
            }
        }

        try:
            toolkit = await manager.create_toolkit(default_spec)
            assert toolkit is not None

            # Verify it's a wrapper
            from roma.infrastructure.toolkits.base_agno_toolkit import DefaultAgnoToolkitWrapper
            assert isinstance(toolkit, DefaultAgnoToolkitWrapper)

            # Test basic functionality
            assert toolkit.is_default_agno_toolkit() is True
            assert toolkit.is_custom_toolkit() is False

        except ImportError:
            # Expected if Agno not available - this is fine for testing
            pytest.skip("Agno not available - skipping default toolkit test")

    @pytest.mark.asyncio
    async def test_toolkit_integration_with_system_manager(self, roma_config):
        """Test toolkit integration through SystemManager."""
        system_manager = SystemManager(roma_config)
        await system_manager.initialize("toolkit_test_profile")

        try:
            # Verify toolkit manager is available
            system_info = system_manager.get_system_info()
            assert system_info["components"]["toolkit_manager"] is True

            # Test toolkit creation through system
            toolkit_manager = system_manager._toolkit_manager
            assert isinstance(toolkit_manager, AgnoToolkitManager)

            # Create a test toolkit
            binance_spec = {
                "name": "binance_integration_test",
                "type": "crypto",
                "implementation": "src.roma.infrastructure.toolkits.custom.crypto.binance_toolkit.BinanceToolkit",
                "config": {
                    "symbols": ["BTCUSDT"],
                    "default_market_type": "spot"
                }
            }

            toolkit = await toolkit_manager.create_toolkit(binance_spec)
            assert toolkit is not None

            # Test toolkit is available
            retrieved_toolkit = await toolkit_manager.read_toolkit("binance_integration_test")
            assert retrieved_toolkit is not None
            assert retrieved_toolkit == toolkit

            # Test toolkit for agent use
            agent_toolkit = toolkit_manager.get_toolkit_for_agent("binance_integration_test")
            assert agent_toolkit is not None

        finally:
            await system_manager.shutdown()

    @pytest.mark.asyncio
    async def test_toolkit_registry_integration(self, roma_config):
        """Test toolkit integration with enhanced registry."""
        system_manager = SystemManager(roma_config)
        await system_manager.initialize("toolkit_test_profile")

        try:
            # Get the enhanced registry (should be available through system manager)
            # This tests the integration between AgnoToolkitManager and EnhancedToolkitRegistry

            toolkit_manager = system_manager._toolkit_manager

            # Create a custom toolkit
            binance_spec = {
                "name": "binance_registry_test",
                "type": "crypto",
                "implementation": "src.roma.infrastructure.toolkits.custom.crypto.binance_toolkit.BinanceToolkit",
                "config": {
                    "symbols": ["BTCUSDT"],
                    "default_market_type": "spot"
                }
            }

            toolkit = await toolkit_manager.create_toolkit(binance_spec)

            # Test toolkit is properly managed
            toolkits_list = await toolkit_manager.list_available_toolkits()
            assert len(toolkits_list) > 0

            # Find our toolkit
            our_toolkit = next((t for t in toolkits_list if t["name"] == "binance_registry_test"), None)
            assert our_toolkit is not None
            assert our_toolkit["type"] == "crypto"
            assert our_toolkit["is_custom"] is True

        finally:
            await system_manager.shutdown()

    @pytest.mark.asyncio
    async def test_toolkit_execution_simulation(self, roma_config):
        """Test simulated toolkit execution through agent framework."""
        system_manager = SystemManager(roma_config)
        await system_manager.initialize("toolkit_test_profile")

        try:
            # Mock agent framework execution to test toolkit integration

            # Create toolkit through system manager
            toolkit_manager = system_manager._toolkit_manager
            binance_spec = {
                "name": "binance_execution_test",
                "type": "crypto",
                "implementation": "src.roma.infrastructure.toolkits.custom.crypto.binance_toolkit.BinanceToolkit",
                "config": {
                    "symbols": ["BTCUSDT"],
                    "default_market_type": "spot"
                }
            }

            toolkit = await toolkit_manager.create_toolkit(binance_spec)

            # Simulate agent execution using the toolkit
            # This tests that the toolkit can be accessed by agents
            agent_toolkit = toolkit_manager.get_toolkit_for_agent("binance_execution_test")
            assert agent_toolkit is not None

            # Test a toolkit tool execution
            result = await agent_toolkit.get_current_price("BTCUSDT")
            assert result is not None
            assert isinstance(result, dict)
            assert result.get("success") is True
            assert "symbol" in result

        finally:
            await system_manager.shutdown()

    @pytest.mark.asyncio
    async def test_toolkit_validation_integration(self, roma_config):
        """Test toolkit validation through the complete system."""
        system_manager = SystemManager(roma_config)
        await system_manager.initialize("toolkit_test_profile")

        try:
            toolkit_manager = system_manager._toolkit_manager

            # Create toolkit with validation support
            binance_spec = {
                "name": "binance_validation_test",
                "type": "crypto",
                "implementation": "src.roma.infrastructure.toolkits.custom.crypto.binance_toolkit.BinanceToolkit",
                "config": {
                    "symbols": ["BTCUSDT"],
                    "default_market_type": "spot"
                }
            }

            toolkit = await toolkit_manager.create_toolkit(binance_spec)

            # Test validation through the system
            assert isinstance(toolkit, ToolkitValidationMixin)

            # Run comprehensive validation
            validation_result = await toolkit.run_full_validation()

            assert validation_result["toolkit_name"] == "binance_validation_test"
            assert "overall_healthy" in validation_result
            assert "checks" in validation_result

            # Verify validation categories
            checks = validation_result["checks"]
            assert "configuration" in checks
            assert "tools" in checks
            assert "health" in checks

            # All checks should pass for our test setup
            for check_name, check_result in checks.items():
                assert check_result["passed"] is True, f"Validation check {check_name} failed"

        finally:
            await system_manager.shutdown()

    @pytest.mark.asyncio
    async def test_end_to_end_goal_execution_with_toolkits(self, roma_config):
        """Test end-to-end goal execution that would use toolkits."""
        system_manager = SystemManager(roma_config)
        await system_manager.initialize("toolkit_test_profile")

        try:
            # This simulates a full execution that would use toolkits
            # In a real scenario, the agent would access the toolkits

            goal = "Get current price of Bitcoin"

            # Mock the agent execution to return toolkit-based results
            with patch.object(system_manager._agent_runtime_service, 'execute_agent') as mock_execute:
                mock_execute.return_value = {
                    "result": "Bitcoin (BTCUSDT) current price: $45,000.00",
                    "success": True,
                    "toolkit_used": "binance_toolkit",
                    "artifacts": []
                }

                result = await system_manager.execute_task(goal)

                assert result["status"] == "completed"
                assert "Bitcoin" in str(result.get("result", result.get("task", "")))
                assert result["execution_time"] > 0

                # Verify the agent execution was called with context
                mock_execute.assert_called_once()
                call_args = mock_execute.call_args[0]
                assert len(call_args) >= 2  # agent, task, and possibly context

        finally:
            await system_manager.shutdown()


class TestAgnoToolkitPerformance:
    """Test toolkit performance and resource management."""

    @pytest.mark.asyncio
    async def test_toolkit_concurrent_creation(self):
        """Test concurrent toolkit creation performance."""
        import asyncio

        manager = AgnoToolkitManager()
        await manager.initialize()

        # Create multiple toolkits concurrently
        specs = [
            {
                "name": f"binance_concurrent_{i}",
                "type": "crypto",
                "implementation": "src.roma.infrastructure.toolkits.custom.crypto.binance_toolkit.BinanceToolkit",
                "config": {"symbols": ["BTCUSDT"], "default_market_type": "spot"}
            }
            for i in range(5)
        ]

        # Create toolkits concurrently
        start_time = asyncio.get_event_loop().time()
        toolkits = await asyncio.gather(*[
            manager.create_toolkit(spec) for spec in specs
        ])
        end_time = asyncio.get_event_loop().time()

        # Verify all were created
        assert len(toolkits) == 5
        for toolkit in toolkits:
            assert toolkit is not None
            assert isinstance(toolkit, BinanceToolkit)

        # Should complete in reasonable time (< 5 seconds for 5 toolkits)
        assert (end_time - start_time) < 5.0

        # Cleanup
        for spec in specs:
            await manager.delete_toolkit(spec["name"])

    @pytest.mark.asyncio
    async def test_toolkit_memory_cleanup(self):
        """Test that toolkits are properly cleaned up."""
        manager = AgnoToolkitManager()
        await manager.initialize()

        spec = {
            "name": "binance_cleanup_test",
            "type": "crypto",
            "implementation": "src.roma.infrastructure.toolkits.custom.crypto.binance_toolkit.BinanceToolkit",
            "config": {"symbols": ["BTCUSDT"], "default_market_type": "spot"}
        }

        # Create toolkit
        toolkit = await manager.create_toolkit(spec)
        assert await manager.toolkit_exists("binance_cleanup_test") is True

        # Delete toolkit
        deleted = await manager.delete_toolkit("binance_cleanup_test")
        assert deleted is True
        assert await manager.toolkit_exists("binance_cleanup_test") is False

        # Verify cleanup
        available_toolkits = await manager.list_available_toolkits()
        cleanup_toolkit = next((t for t in available_toolkits if t["name"] == "binance_cleanup_test"), None)
        assert cleanup_toolkit is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
