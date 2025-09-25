"""
Execution context providing complete resource isolation except agents.

This module implements the ExecutionContext class which provides isolated
resources per execution while reusing agents across executions for efficiency.
"""

import logging
from typing import Any

from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.application.services.artifact_service import ArtifactService
from roma.application.services.context_builder_service import ContextBuilderService
from roma.application.services.event_publisher import EventPublisher
from roma.application.services.event_store import InMemoryEventStore
from roma.application.services.knowledge_store_service import KnowledgeStoreService
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.interfaces.storage import IStorage
from roma.domain.value_objects.config.roma_config import ROMAConfig

logger = logging.getLogger(__name__)


class ExecutionContext:
    """
    Isolated execution context providing dedicated resources per execution.

    This class creates isolated instances of all resources except agents,
    which are reused across executions for efficiency with session isolation.

    Resources isolated per execution:
    - Task Graph (DynamicTaskGraph)
    - Event Store (InMemoryEventStore)
    - Knowledge Store (KnowledgeStoreService)
    - Context Builder (with isolated knowledge store)
    - Graph State Manager (with isolated components)
    - Artifact Service (with execution namespacing)

    Resources shared across executions:
    - Agents (with Agno session isolation)
    - Storage instance (with execution namespacing)
    - Recovery Manager (stateless)
    """

    def __init__(self, execution_id: str, storage: IStorage):
        """
        Initialize execution context with isolated resources.

        Args:
            execution_id: Unique identifier for this execution
            storage: Storage implementation (injected via dependency inversion)
        """
        self.execution_id = execution_id

        # Use injected storage instance
        self.storage = storage

        # Create isolated resources
        self.task_graph = DynamicTaskGraph()

        # Limited event store per execution to prevent memory growth
        self.event_store = InMemoryEventStore(max_events_per_task=100, max_total_events=10000)

        # Event publisher for event emission using isolated event store
        self.event_publisher = EventPublisher(self.event_store)

        # Isolated knowledge store per execution
        self.knowledge_store = KnowledgeStoreService()

        # Artifact service with execution context
        self.artifact_service = ArtifactService(self.storage)
        # Set execution context for namespacing
        if hasattr(self.artifact_service, "__dict__"):
            self.artifact_service.execution_id = execution_id

        # Context builder using isolated knowledge store
        self.context_builder = ContextBuilderService(
            knowledge_store=self.knowledge_store,
            storage_manager=self.storage,
            roma_config=None,  # Will be set by SystemManager
        )

        # Note: Event publishing is handled by GraphStateManager to avoid duplicates

        # Graph state manager with isolated components - using EventPublisher not EventStore
        self.graph_state_manager = GraphStateManager(self.task_graph, self.event_publisher)

        logger.info(f"ExecutionContext created for execution {execution_id}")

    def set_config(self, config: ROMAConfig) -> None:
        """
        Set configuration for services that need it.

        Args:
            config: ROMA configuration object
        """
        if hasattr(self.context_builder, "roma_config"):
            self.context_builder.roma_config = config

    async def initialize(self) -> None:
        """
        Initialize all isolated services.

        Currently no services require async initialization,
        but this method is provided for future extensibility.
        """
        # Initialize artifact service if needed
        if hasattr(self.artifact_service, "initialize"):
            await self.artifact_service.initialize()

        logger.debug(f"ExecutionContext initialized for execution {self.execution_id}")

    async def cleanup(self) -> None:
        """
        Clean up execution-specific resources.

        This method cleans up all resources associated with this execution:
        - Execution-specific temporary files in storage
        - Event store contents
        - Knowledge store contents
        - Task graph (cleaned by Python GC)
        """
        try:
            # Clean up execution-specific temp files using storage method
            if hasattr(self.storage, "cleanup_execution_temp_files"):
                cleaned_files = await self.storage.cleanup_execution_temp_files()
                logger.debug(
                    f"Cleaned {cleaned_files} temp files for execution {self.execution_id}"
                )

            # Clear isolated event store
            await self.event_store.clear()

            # Clear isolated knowledge store
            await self.knowledge_store.clear()

            # Task graph and other components will be cleaned by Python GC
            logger.info(f"ExecutionContext cleanup completed for execution {self.execution_id}")

        except Exception as e:
            logger.error(f"Error during ExecutionContext cleanup for {self.execution_id}: {e}")
            # Don't re-raise to avoid masking the original exception

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about this execution context.

        Returns:
            Dictionary with execution context statistics
        """
        return {
            "execution_id": self.execution_id,
            "task_graph_nodes": len(self.task_graph.get_all_nodes()),
            "event_store_events": len(self.event_store._global_events)
            if hasattr(self.event_store, "_global_events")
            else 0,
            "knowledge_records": len(self.knowledge_store._records)
            if hasattr(self.knowledge_store, "_records")
            else 0,
        }
