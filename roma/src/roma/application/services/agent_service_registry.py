"""
Agent Service Registry Implementation.

Manages all agent services and provides CRUD operations for agent service management.
Handles service lifecycle, dependency injection, and service lookup.
"""

import logging
from typing import Dict, Optional

from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.task_type import TaskType
from roma.domain.interfaces.agent_service import (
    BaseAgentServiceInterface,
    AtomizerServiceInterface,
    PlannerServiceInterface,
    ExecutorServiceInterface,
    AggregatorServiceInterface,
    PlanModifierServiceInterface
)
from roma.application.services.atomizer_service import AtomizerService
from roma.application.services.planner_service import PlannerService
from roma.application.services.executor_service import ExecutorService
from roma.application.services.aggregator_service import AggregatorService
from roma.application.services.plan_modifier_service import PlanModifierService
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.recovery_manager import RecoveryManager
from roma.application.services.hitl_service import HITLService
from roma.domain.interfaces.persistence import CheckpointRepository, RecoveryRepository, ExecutionHistoryRepository

logger = logging.getLogger(__name__)


class AgentServiceRegistry:
    """
    Registry for managing all agent services.

    Provides CRUD operations and service lookup by agent type.
    Handles service lifecycle and dependency injection.
    """

    def __init__(
        self,
        agent_runtime_service: AgentRuntimeService,
        recovery_manager: RecoveryManager,
        hitl_service: Optional[HITLService] = None,
        checkpoint_repository: Optional[CheckpointRepository] = None,
        recovery_repository: Optional[RecoveryRepository] = None,
        execution_history_repository: Optional[ExecutionHistoryRepository] = None
    ):
        """
        Initialize registry with core dependencies.

        Args:
            agent_runtime_service: Runtime service for agent execution
            recovery_manager: Recovery manager for error handling
            hitl_service: Optional HITL service for human interaction
            checkpoint_repository: Optional checkpoint repository for persistence
            recovery_repository: Optional recovery repository for persistence
            execution_history_repository: Optional execution history repository
        """
        self.agent_runtime_service = agent_runtime_service
        self.recovery_manager = recovery_manager
        self.hitl_service = hitl_service
        self.checkpoint_repository = checkpoint_repository
        self.recovery_repository = recovery_repository
        self.execution_history_repository = execution_history_repository
        self._services: Dict[AgentType, BaseAgentServiceInterface] = {}

        # Initialize all standard services
        self._initialize_services()

    def _initialize_services(self) -> None:
        """Initialize all standard agent services."""
        # Create all services with shared dependencies
        services = {
            AgentType.ATOMIZER: AtomizerService(
                self.agent_runtime_service,
                self.recovery_manager
            ),
            AgentType.PLANNER: PlannerService(
                self.agent_runtime_service,
                self.recovery_manager
            ),
            AgentType.EXECUTOR: ExecutorService(
                self.agent_runtime_service,
                self.recovery_manager
            ),
            AgentType.AGGREGATOR: AggregatorService(
                self.agent_runtime_service,
                self.recovery_manager
            ),
            AgentType.PLAN_MODIFIER: PlanModifierService(
                self.agent_runtime_service,
                self.recovery_manager,
                self.hitl_service
            )
        }

        # Register all services
        for agent_type, service in services.items():
            self.register_service(service)

        logger.info(f"Initialized {len(services)} agent services")

    def register_service(self, service: BaseAgentServiceInterface) -> None:
        """
        Register a new agent service.

        Args:
            service: Service to register

        Raises:
            ValueError: If service is invalid or agent type already registered
        """
        if not isinstance(service, BaseAgentServiceInterface):
            raise ValueError(f"Service must implement BaseAgentServiceInterface")

        agent_type = service.agent_type
        if agent_type in self._services:
            logger.warning(f"Replacing existing service for {agent_type}")

        self._services[agent_type] = service
        logger.info(f"Registered {service.__class__.__name__} for {agent_type}")

    def unregister_service(self, agent_type: AgentType) -> bool:
        """
        Unregister a service.

        Args:
            agent_type: Type of agent service to unregister

        Returns:
            True if service was removed, False if not found
        """
        if agent_type in self._services:
            service = self._services.pop(agent_type)
            logger.info(f"Unregistered {service.__class__.__name__} for {agent_type}")
            return True
        return False

    def get_service(self, agent_type: AgentType) -> BaseAgentServiceInterface:
        """
        Get service for specific agent type.

        Args:
            agent_type: Type of agent service needed

        Returns:
            Agent service instance

        Raises:
            KeyError: If agent type not registered
        """
        if agent_type not in self._services:
            raise KeyError(f"No service registered for agent type: {agent_type}")

        return self._services[agent_type]

    def get_atomizer_service(self) -> AtomizerServiceInterface:
        """Get atomizer service."""
        return self.get_service(AgentType.ATOMIZER)

    def get_planner_service(self) -> PlannerServiceInterface:
        """Get planner service."""
        return self.get_service(AgentType.PLANNER)

    def get_executor_service(self) -> ExecutorServiceInterface:
        """Get executor service."""
        return self.get_service(AgentType.EXECUTOR)

    def get_aggregator_service(self) -> AggregatorServiceInterface:
        """Get aggregator service."""
        return self.get_service(AgentType.AGGREGATOR)

    def get_plan_modifier_service(self) -> PlanModifierServiceInterface:
        """Get plan modifier service."""
        return self.get_service(AgentType.PLAN_MODIFIER)

    # Persistence repository accessors
    def get_checkpoint_repository(self) -> Optional[CheckpointRepository]:
        """Get checkpoint repository if available."""
        return self.checkpoint_repository

    def get_recovery_repository(self) -> Optional[RecoveryRepository]:
        """Get recovery repository if available."""
        return self.recovery_repository

    def get_execution_history_repository(self) -> Optional[ExecutionHistoryRepository]:
        """Get execution history repository if available."""
        return self.execution_history_repository

    def has_persistence(self) -> bool:
        """Check if persistence repositories are available."""
        return (self.checkpoint_repository is not None and
                self.recovery_repository is not None and
                self.execution_history_repository is not None)

    def get_all_services(self) -> Dict[AgentType, BaseAgentServiceInterface]:
        """
        Get all registered services.

        Returns:
            Dictionary mapping agent types to services
        """
        return self._services.copy()

    def list_registered_types(self) -> list[AgentType]:
        """
        List all registered agent types.

        Returns:
            List of registered agent types
        """
        return list(self._services.keys())

    def is_registered(self, agent_type: AgentType) -> bool:
        """
        Check if agent type is registered.

        Args:
            agent_type: Agent type to check

        Returns:
            True if registered, False otherwise
        """
        return agent_type in self._services

    def update_service(
        self,
        agent_type: AgentType,
        service: BaseAgentServiceInterface
    ) -> BaseAgentServiceInterface:
        """
        Update an existing service.

        Args:
            agent_type: Agent type to update
            service: New service instance

        Returns:
            Previous service instance

        Raises:
            KeyError: If agent type not registered
            ValueError: If service is invalid
        """
        if agent_type not in self._services:
            raise KeyError(f"No service registered for agent type: {agent_type}")

        if not isinstance(service, BaseAgentServiceInterface):
            raise ValueError("Service must implement BaseAgentServiceInterface")

        if service.agent_type != agent_type:
            raise ValueError(f"Service agent type {service.agent_type} doesn't match {agent_type}")

        old_service = self._services[agent_type]
        self._services[agent_type] = service

        logger.info(f"Updated service for {agent_type}: {old_service.__class__.__name__} → {service.__class__.__name__}")
        return old_service

    def clear_all_services(self) -> None:
        """Clear all registered services."""
        count = len(self._services)
        self._services.clear()
        logger.warning(f"Cleared all {count} registered services")

    def get_registry_stats(self) -> Dict[str, any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        stats = {
            "total_services": len(self._services),
            "registered_types": [agent_type.value for agent_type in self._services.keys()],
            "services": {}
        }

        # Get stats from each service
        for agent_type, service in self._services.items():
            try:
                service_stats = service.get_stats() if hasattr(service, 'get_stats') else {}
                stats["services"][agent_type.value] = service_stats
            except Exception as e:
                logger.warning(f"Failed to get stats from {agent_type}: {e}")
                stats["services"][agent_type.value] = {"error": str(e)}

        return stats

    def health_check(self) -> Dict[str, any]:
        """
        Perform health check on all services.

        Returns:
            Health check results
        """
        health = {
            "status": "healthy",
            "services_count": len(self._services),
            "missing_services": [],
            "service_health": {}
        }

        # Check if all expected services are registered
        expected_types = [
            AgentType.ATOMIZER,
            AgentType.PLANNER,
            AgentType.EXECUTOR,
            AgentType.AGGREGATOR,
            AgentType.PLAN_MODIFIER
        ]

        for agent_type in expected_types:
            if agent_type not in self._services:
                health["missing_services"].append(agent_type.value)
                health["status"] = "degraded"

        # Basic health check for each service
        for agent_type, service in self._services.items():
            try:
                # Basic check - service exists and has required methods
                if hasattr(service, 'run') and hasattr(service, 'agent_type'):
                    health["service_health"][agent_type.value] = "healthy"
                else:
                    health["service_health"][agent_type.value] = "invalid"
                    health["status"] = "unhealthy"
            except Exception as e:
                health["service_health"][agent_type.value] = f"error: {e}"
                health["status"] = "unhealthy"

        return health