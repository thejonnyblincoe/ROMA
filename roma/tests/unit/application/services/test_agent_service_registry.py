"""
Tests for AgentServiceRegistry.

Tests the registration, management, and retrieval of agent services.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.agent_service_registry import AgentServiceRegistry
from roma.application.services.hitl_service import HITLService
from roma.application.services.recovery_manager import RecoveryManager
from roma.domain.interfaces.agent_service import BaseAgentServiceInterface
from roma.domain.value_objects.agent_type import AgentType


@pytest.fixture
def mock_agent_runtime_service():
    """Mock agent runtime service."""
    mock = Mock(spec=AgentRuntimeService)
    mock.get_agent = AsyncMock()
    mock.execute_agent = AsyncMock()
    return mock


@pytest.fixture
def mock_recovery_manager():
    """Mock recovery manager."""
    return Mock(spec=RecoveryManager)


@pytest.fixture
def mock_hitl_service():
    """Mock HITL service."""
    mock = Mock(spec=HITLService)
    mock.enabled = True
    mock.request_replanning_approval = AsyncMock()
    return mock


@pytest.fixture
def agent_service_registry(mock_agent_runtime_service, mock_recovery_manager, mock_hitl_service):
    """Create agent service registry with mocked dependencies."""
    return AgentServiceRegistry(
        agent_runtime_service=mock_agent_runtime_service,
        recovery_manager=mock_recovery_manager,
        hitl_service=mock_hitl_service
    )


class TestAgentServiceRegistry:
    """Test AgentServiceRegistry functionality."""

    def test_initialization_creates_all_services(self, agent_service_registry):
        """Test that registry initializes with all standard agent services."""
        # Verify all agent types are registered
        expected_types = {
            AgentType.ATOMIZER,
            AgentType.PLANNER,
            AgentType.EXECUTOR,
            AgentType.AGGREGATOR,
            AgentType.PLAN_MODIFIER
        }

        registered_types = set(agent_service_registry._services.keys())
        assert registered_types == expected_types

    def test_get_service_returns_correct_service(self, agent_service_registry):
        """Test getting service by agent type."""
        atomizer_service = agent_service_registry.get_service(AgentType.ATOMIZER)
        assert atomizer_service is not None
        assert atomizer_service.agent_type == AgentType.ATOMIZER

        planner_service = agent_service_registry.get_service(AgentType.PLANNER)
        assert planner_service is not None
        assert planner_service.agent_type == AgentType.PLANNER

    def test_get_service_nonexistent_raises_error(self, agent_service_registry):
        """Test getting non-existent service raises KeyError."""
        # Create a fake agent type
        fake_type = "FAKE_AGENT"
        with pytest.raises(KeyError, match="No service registered for agent type"):
            agent_service_registry.get_service(fake_type)

    def test_register_service_adds_new_service(self, agent_service_registry):
        """Test registering a new service."""
        # Create a mock service
        mock_service = Mock(spec=BaseAgentServiceInterface)
        mock_service.agent_type = "CUSTOM_AGENT"

        # Register the service
        agent_service_registry.register_service(mock_service)

        # Verify it was registered
        retrieved_service = agent_service_registry.get_service("CUSTOM_AGENT")
        assert retrieved_service == mock_service

    def test_register_service_overwrites_existing(self, agent_service_registry):
        """Test that registering overwrites existing service."""
        # Get original atomizer service
        original_service = agent_service_registry.get_service(AgentType.ATOMIZER)

        # Create a new mock service
        mock_service = Mock(spec=BaseAgentServiceInterface)
        mock_service.agent_type = AgentType.ATOMIZER

        # Register the new service
        agent_service_registry.register_service(mock_service)

        # Verify it was replaced
        retrieved_service = agent_service_registry.get_service(AgentType.ATOMIZER)
        assert retrieved_service == mock_service
        assert retrieved_service != original_service

    def test_unregister_service_removes_service(self, agent_service_registry):
        """Test unregistering a service."""
        # Verify atomizer exists
        assert agent_service_registry.get_service(AgentType.ATOMIZER) is not None

        # Unregister it
        result = agent_service_registry.unregister_service(AgentType.ATOMIZER)
        assert result is True

        # Verify it's gone
        with pytest.raises(KeyError):
            agent_service_registry.get_service(AgentType.ATOMIZER)

    def test_unregister_nonexistent_service_returns_false(self, agent_service_registry):
        """Test unregistering non-existent service returns False."""
        result = agent_service_registry.unregister_service("FAKE_AGENT")
        assert result is False

    def test_get_all_services_returns_all_registered(self, agent_service_registry):
        """Test getting all registered services."""
        services = agent_service_registry.get_all_services()

        # Should have all 5 standard agent types
        assert len(services) == 5

        # Verify all expected types are present
        service_types = set(services.keys())
        expected_types = {
            AgentType.ATOMIZER,
            AgentType.PLANNER,
            AgentType.EXECUTOR,
            AgentType.AGGREGATOR,
            AgentType.PLAN_MODIFIER
        }
        assert service_types == expected_types

    def test_hitl_service_passed_to_plan_modifier(self, mock_agent_runtime_service, mock_recovery_manager, mock_hitl_service):
        """Test that HITL service is properly passed to PlanModifierService."""
        registry = AgentServiceRegistry(
            agent_runtime_service=mock_agent_runtime_service,
            recovery_manager=mock_recovery_manager,
            hitl_service=mock_hitl_service
        )

        plan_modifier_service = registry.get_service(AgentType.PLAN_MODIFIER)
        assert plan_modifier_service is not None
        assert plan_modifier_service.hitl_service == mock_hitl_service

    def test_initialization_without_hitl_service(self, mock_agent_runtime_service, mock_recovery_manager):
        """Test initialization without HITL service."""
        registry = AgentServiceRegistry(
            agent_runtime_service=mock_agent_runtime_service,
            recovery_manager=mock_recovery_manager,
            hitl_service=None
        )

        # Should still create all services
        assert len(registry.get_all_services()) == 5

        # Plan modifier should have None HITL service
        plan_modifier_service = registry.get_service(AgentType.PLAN_MODIFIER)
        assert plan_modifier_service.hitl_service is None

    def test_get_registry_stats_returns_info(self, agent_service_registry):
        """Test getting registry statistics."""
        stats = agent_service_registry.get_registry_stats()

        assert "total_services" in stats
        assert "registered_types" in stats
        assert stats["total_services"] == 5
        assert len(stats["registered_types"]) == 5

    def test_clear_all_services_removes_all(self, agent_service_registry):
        """Test clearing all services."""
        # Verify services exist
        assert len(agent_service_registry.get_all_services()) == 5

        # Clear all services
        agent_service_registry.clear_all_services()

        # Verify all are gone
        assert len(agent_service_registry.get_all_services()) == 0
        with pytest.raises(KeyError):
            agent_service_registry.get_service(AgentType.ATOMIZER)
