"""
Knowledge Store Service Interface - ROMA v2.0 Domain Interface.

Abstract interface for knowledge store operations that application must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.knowledge_record import KnowledgeRecord
from roma.domain.value_objects.result_envelope import ResultEnvelope
from roma.domain.value_objects.task_status import TaskStatus


class IKnowledgeStoreService(ABC):
    """
    Domain interface for knowledge store operations.

    Abstract interface that defines all knowledge management operations required by the domain.
    Application layer must implement this interface to provide knowledge store capabilities.
    """

    @abstractmethod
    async def add_or_update_record(
        self, node: TaskNode, envelope: ResultEnvelope[Any] | None = None
    ) -> KnowledgeRecord:
        """
        Add or update a knowledge record from TaskNode.

        Args:
            node: TaskNode to store
            envelope: Optional ResultEnvelope with artifacts

        Returns:
            Created or updated KnowledgeRecord
        """

    @abstractmethod
    async def get_record(self, task_id: str) -> KnowledgeRecord | None:
        """
        Get knowledge record by task ID.

        Args:
            task_id: Task identifier

        Returns:
            KnowledgeRecord if found, None otherwise
        """

    @abstractmethod
    async def get_child_records(self, parent_id: str) -> list[KnowledgeRecord]:
        """
        Get all child records for a parent task.

        Args:
            parent_id: Parent task identifier

        Returns:
            List of child KnowledgeRecord objects
        """

    @abstractmethod
    async def get_records_by_status(self, status: TaskStatus) -> list[KnowledgeRecord]:
        """
        Get all records with specific status.

        Args:
            status: TaskStatus to filter by

        Returns:
            List of KnowledgeRecord objects with matching status
        """

    @abstractmethod
    async def add_child_relationship(self, parent_id: str, child_id: str) -> bool:
        """
        Add child relationship between tasks.

        Args:
            parent_id: Parent task identifier
            child_id: Child task identifier

        Returns:
            True if relationship added successfully
        """

    @abstractmethod
    async def get_completed_records(self) -> list[KnowledgeRecord]:
        """
        Get all completed records.

        Returns:
            List of completed KnowledgeRecord objects
        """

    @abstractmethod
    async def get_failed_records(self) -> list[KnowledgeRecord]:
        """
        Get all failed records.

        Returns:
            List of failed KnowledgeRecord objects
        """

    @abstractmethod
    async def clear(self) -> None:
        """Clear all records from the store."""

    @abstractmethod
    async def get_summary_stats(self) -> dict[str, Any]:
        """
        Get summary statistics about the knowledge store.

        Returns:
            Dictionary with statistics (record counts, cache stats, etc.)
        """

    @abstractmethod
    async def get_records_with_artifacts(
        self, task_id: str, include_siblings: bool = True
    ) -> list[str]:
        """
        Get storage keys for artifacts associated with task and optionally siblings.

        Args:
            task_id: Task identifier
            include_siblings: Whether to include sibling artifacts

        Returns:
            List of artifact storage keys
        """
