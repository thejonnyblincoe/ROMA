"""
Graph State Manager Interface - ROMA v2.0 Domain Interface.

Abstract interface for graph state management operations that application must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus


class IGraphStateManager(ABC):
    """
    Domain interface for graph state manager operations.

    Abstract interface that defines all graph state management operations required by the domain.
    Application layer must implement this interface to provide state management capabilities.
    """

    @property
    @abstractmethod
    def is_locked(self) -> bool:
        """Check if state manager is currently locked."""

    @abstractmethod
    async def transition_node_status(self, task_id: str, new_status: TaskStatus) -> TaskNode:
        """
        Transition node status with validation and event emission.

        Args:
            task_id: Task identifier
            new_status: Target status

        Returns:
            Updated TaskNode

        Raises:
            ValueError: If task not found or invalid transition
        """

    @abstractmethod
    async def add_node(self, node: TaskNode) -> None:
        """
        Add new node to graph with event emission.

        Args:
            node: TaskNode to add

        Raises:
            ValueError: If node already exists
        """

    @abstractmethod
    async def add_dependency_edge(self, from_id: str, to_id: str) -> None:
        """
        Add dependency edge between nodes.

        Args:
            from_id: Source task ID
            to_id: Target task ID

        Raises:
            ValueError: If nodes don't exist or cycle would be created
        """

    @abstractmethod
    async def get_ready_nodes(self) -> list[TaskNode]:
        """
        Get nodes ready for execution (READY status with dependencies met).

        Returns:
            List of ready TaskNode objects
        """

    @abstractmethod
    def get_node_by_id(self, task_id: str) -> TaskNode | None:
        """
        Get node by task ID.

        Args:
            task_id: Task identifier

        Returns:
            TaskNode if found, None otherwise
        """

    @abstractmethod
    def get_all_nodes(self) -> list[TaskNode]:
        """
        Get all nodes in graph.

        Returns:
            List of all TaskNode objects
        """

    @abstractmethod
    def get_children_nodes(self, task_id: str) -> list[TaskNode]:
        """
        Get child nodes for given parent.

        Args:
            task_id: Parent task ID

        Returns:
            List of child TaskNode objects
        """

    @abstractmethod
    def has_cycles(self) -> bool:
        """
        Check if graph contains cycles.

        Returns:
            True if cycles detected
        """

    @abstractmethod
    def get_execution_statistics(self) -> dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """

    @abstractmethod
    async def remove_node(self, task_id: str) -> None:
        """
        Remove node from graph.

        Args:
            task_id: Task ID to remove

        Raises:
            ValueError: If node not found or has dependencies
        """

    @abstractmethod
    async def update_node_metadata(
        self, task_id: str, metadata_updates: dict[str, Any]
    ) -> TaskNode:
        """
        Update node metadata.

        Args:
            task_id: Task ID
            metadata_updates: Metadata updates to apply

        Returns:
            Updated TaskNode

        Raises:
            ValueError: If node not found
        """

    @abstractmethod
    async def increment_node_retry_count(self, task_id: str) -> TaskNode:
        """
        Increment node retry count.

        Args:
            task_id: Task ID

        Returns:
            Updated TaskNode

        Raises:
            ValueError: If node not found
        """
