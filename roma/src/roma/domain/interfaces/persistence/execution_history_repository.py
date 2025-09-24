"""
Execution History Repository Interface for Clean Architecture.

Defines the repository interface for execution history persistence operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.persistence import (
    TaskRelationshipType,
    ExecutionRecord,
    ExecutionTreeNode,
    ExecutionAnalytics,
)


class ExecutionHistoryRepository(ABC):
    """
    Abstract repository interface for execution history persistence.

    Defines operations for tracking task execution history, relationships,
    and analytics without exposing persistence implementation details.
    """

    @abstractmethod
    async def create_execution(
        self,
        task: TaskNode,
        execution_context: Optional[dict] = None,
        agent_config: Optional[dict] = None
    ) -> str:
        """
        Create a new execution record.

        Args:
            task: Task node to track
            execution_context: Execution context
            agent_config: Agent configuration used

        Returns:
            Execution ID
        """
        pass

    @abstractmethod
    async def update_execution_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[dict] = None,
        error_info: Optional[dict] = None,
        execution_duration_ms: Optional[int] = None
    ) -> None:
        """
        Update execution status and result.

        Args:
            task_id: Task ID to update
            status: New task status
            result: Execution result
            error_info: Error information if failed
            execution_duration_ms: Execution duration in milliseconds
        """
        pass

    @abstractmethod
    async def add_task_relationship(
        self,
        parent_task_id: str,
        child_task_id: str,
        relationship_type: TaskRelationshipType,
        order_index: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Add a relationship between tasks.

        Args:
            parent_task_id: Parent task ID
            child_task_id: Child task ID
            relationship_type: Type of relationship
            order_index: Order index for ordered relationships
            metadata: Relationship metadata
        """
        pass

    @abstractmethod
    async def get_execution_history(
        self,
        task_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        include_children: bool = False
    ) -> Optional[ExecutionRecord]:
        """
        Get execution history for a task.

        Args:
            task_id: Task ID to get history for
            execution_id: Execution ID to get history for
            include_children: Whether to include child task history

        Returns:
            Execution history or None if not found
        """
        pass

    @abstractmethod
    async def get_execution_tree(self, root_task_id: str) -> ExecutionTreeNode:
        """
        Get complete execution tree for a root task.

        Args:
            root_task_id: Root task ID

        Returns:
            Complete execution tree
        """
        pass

    @abstractmethod
    async def get_child_executions(self, parent_task_id: str) -> List[ExecutionRecord]:
        """
        Get child executions for a parent task.

        Args:
            parent_task_id: Parent task ID

        Returns:
            List of child execution data
        """
        pass

    @abstractmethod
    async def get_execution_analytics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> ExecutionAnalytics:
        """
        Get execution analytics and performance metrics.

        Args:
            start_time: Start time for analysis
            end_time: End time for analysis

        Returns:
            Analytics data
        """
        pass

    @abstractmethod
    async def cleanup_old_executions(self, days: int = 90) -> int:
        """
        Clean up old execution records.

        Args:
            days: Number of days to keep

        Returns:
            Number of records cleaned up
        """
        pass