"""
Execution History Service with Clean Architecture.

Application layer service that orchestrates execution history operations using repository interfaces.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from uuid import uuid4

from roma.domain.interfaces.persistence import ExecutionHistoryRepository
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.persistence import (
    TaskRelationshipType,
    ExecutionRecord,
    ExecutionTreeNode,
    ExecutionAnalytics,
)

logger = logging.getLogger(__name__)


class ExecutionHistoryService:
    """
    Application service for tracking and managing task execution history.

    Uses repository interfaces for persistence operations following Clean Architecture.
    Provides high-level execution history orchestration without persistence implementation details.
    """

    def __init__(self, execution_history_repository: ExecutionHistoryRepository):
        """
        Initialize execution history service with repository dependency.

        Args:
            execution_history_repository: Repository for execution history persistence
        """
        self.execution_repo = execution_history_repository
        self._stats = {
            "executions_tracked": 0,
            "relationships_created": 0,
            "queries_executed": 0,
        }

    async def start_execution(
        self,
        task: TaskNode,
        execution_context: Optional[Dict[str, Any]] = None,
        agent_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking a new task execution.

        Args:
            task: Task node to track
            execution_context: Execution context
            agent_config: Agent configuration used

        Returns:
            Execution ID
        """
        execution_id = await self.execution_repo.create_execution(
            task=task,
            execution_context=execution_context,
            agent_config=agent_config
        )

        self._stats["executions_tracked"] += 1
        logger.debug(f"Started tracking execution for task {task.task_id}")

        return execution_id

    async def update_execution_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None,
        error_info: Optional[Dict[str, Any]] = None,
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
        await self.execution_repo.update_execution_status(
            task_id=task_id,
            status=status,
            result=result,
            error_info=error_info,
            execution_duration_ms=execution_duration_ms
        )

        self._stats["queries_executed"] += 1
        logger.debug(f"Updated execution status for task {task_id} to {status}")

    async def add_task_relationship(
        self,
        parent_task_id: str,
        child_task_id: str,
        relationship_type: TaskRelationshipType = TaskRelationshipType.PARENT_CHILD,
        order_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
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
        try:
            async with self.session_factory() as session:
                relationship = TaskRelationshipModel(
                    parent_task_id=parent_task_id,
                    child_task_id=child_task_id,
                    relationship_type=relationship_type,
                    order_index=order_index,
                    metadata=metadata
                )

                session.add(relationship)
                await session.commit()

                self._stats["relationships_created"] += 1
                logger.debug(f"Added {relationship_type} relationship: {parent_task_id} -> {child_task_id}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to add task relationship: {e}")
            raise

    async def get_execution_history(
        self,
        task_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        include_children: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get execution history for a task.

        Args:
            task_id: Task ID to get history for
            execution_id: Execution ID to get history for
            include_children: Whether to include child task history

        Returns:
            Execution history or None if not found
        """
        try:
            async with self.session_factory() as session:
                # Build query
                query = select(TaskExecutionModel)

                if execution_id:
                    query = query.where(TaskExecutionModel.id == execution_id)
                elif task_id:
                    query = query.where(TaskExecutionModel.task_id == task_id)
                else:
                    raise ValueError("Either task_id or execution_id must be provided")

                # Execute query
                result = await session.execute(query)
                execution = result.scalar_one_or_none()

                if not execution:
                    return None

                # Convert to dictionary
                history = {
                    "execution_id": execution.id,
                    "task_id": execution.task_id,
                    "goal": execution.goal,
                    "task_type": execution.task_type,
                    "node_type": execution.node_type,
                    "status": execution.status.value,
                    "parent_task_id": execution.parent_task_id,
                    "root_task_id": execution.root_task_id,
                    "depth_level": execution.depth_level,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "execution_duration_ms": execution.execution_duration_ms,
                    "result": execution.result,
                    "metadata": execution.metadata,
                    "error_info": execution.error_info,
                    "agent_config": execution.agent_config,
                    "execution_context": execution.execution_context,
                    "retry_count": execution.retry_count,
                    "max_retries": execution.max_retries,
                    "token_usage": execution.token_usage,
                    "cost_info": execution.cost_info,
                    "created_at": execution.created_at.isoformat(),
                    "updated_at": execution.updated_at.isoformat(),
                }

                # Include children if requested
                if include_children:
                    children = await self._get_child_executions(session, execution.task_id)
                    history["children"] = children

                self._stats["queries_executed"] += 1
                return history

        except SQLAlchemyError as e:
            logger.error(f"Failed to get execution history: {e}")
            raise

    async def _get_child_executions(
        self,
        session: AsyncSession,
        parent_task_id: str
    ) -> List[Dict[str, Any]]:
        """Get child executions for a parent task."""
        # Get child relationships
        rel_query = select(TaskRelationshipModel).where(
            TaskRelationshipModel.parent_task_id == parent_task_id
        ).order_by(TaskRelationshipModel.order_index)

        rel_result = await session.execute(rel_query)
        relationships = rel_result.scalars().all()

        children = []
        for rel in relationships:
            # Get child execution
            exec_query = select(TaskExecutionModel).where(
                TaskExecutionModel.task_id == rel.child_task_id
            )
            exec_result = await session.execute(exec_query)
            child_exec = exec_result.scalar_one_or_none()

            if child_exec:
                child_data = {
                    "execution_id": child_exec.id,
                    "task_id": child_exec.task_id,
                    "goal": child_exec.goal,
                    "task_type": child_exec.task_type,
                    "status": child_exec.status.value,
                    "relationship_type": rel.relationship_type.value,
                    "order_index": rel.order_index,
                    "started_at": child_exec.started_at.isoformat() if child_exec.started_at else None,
                    "completed_at": child_exec.completed_at.isoformat() if child_exec.completed_at else None,
                    "execution_duration_ms": child_exec.execution_duration_ms,
                }
                children.append(child_data)

        return children

    async def get_execution_tree(self, root_task_id: str) -> Dict[str, Any]:
        """
        Get complete execution tree for a root task.

        Args:
            root_task_id: Root task ID

        Returns:
            Complete execution tree
        """
        try:
            async with self.session_factory() as session:
                # Get all executions in the tree
                exec_query = select(TaskExecutionModel).where(
                    or_(
                        TaskExecutionModel.task_id == root_task_id,
                        TaskExecutionModel.root_task_id == root_task_id
                    )
                ).order_by(TaskExecutionModel.depth_level, TaskExecutionModel.created_at)

                exec_result = await session.execute(exec_query)
                executions = exec_result.scalars().all()

                # Get all relationships
                task_ids = [exec.task_id for exec in executions]
                rel_query = select(TaskRelationshipModel).where(
                    TaskRelationshipModel.parent_task_id.in_(task_ids)
                ).order_by(TaskRelationshipModel.order_index)

                rel_result = await session.execute(rel_query)
                relationships = rel_result.scalars().all()

                # Build tree structure
                tree = self._build_execution_tree(executions, relationships, root_task_id)

                self._stats["queries_executed"] += 2
                return tree

        except SQLAlchemyError as e:
            logger.error(f"Failed to get execution tree: {e}")
            raise

    def _build_execution_tree(
        self,
        executions: List[TaskExecutionModel],
        relationships: List[TaskRelationshipModel],
        root_task_id: str
    ) -> Dict[str, Any]:
        """Build hierarchical execution tree from flat data."""
        # Create execution lookup
        exec_lookup = {exec.task_id: exec for exec in executions}

        # Create relationship lookup
        children_lookup = {}
        for rel in relationships:
            if rel.parent_task_id not in children_lookup:
                children_lookup[rel.parent_task_id] = []
            children_lookup[rel.parent_task_id].append(rel)

        def build_node(task_id: str) -> Dict[str, Any]:
            exec = exec_lookup[task_id]
            node = {
                "execution_id": exec.id,
                "task_id": exec.task_id,
                "goal": exec.goal,
                "task_type": exec.task_type,
                "node_type": exec.node_type,
                "status": exec.status.value,
                "depth_level": exec.depth_level,
                "started_at": exec.started_at.isoformat() if exec.started_at else None,
                "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                "execution_duration_ms": exec.execution_duration_ms,
                "children": []
            }

            # Add children
            if task_id in children_lookup:
                for rel in children_lookup[task_id]:
                    if rel.child_task_id in exec_lookup:
                        child_node = build_node(rel.child_task_id)
                        child_node["relationship_type"] = rel.relationship_type.value
                        child_node["order_index"] = rel.order_index
                        node["children"].append(child_node)

            return node

        return build_node(root_task_id)

    async def get_execution_analytics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get execution analytics and performance metrics.

        Args:
            start_time: Start time for analysis
            end_time: End time for analysis

        Returns:
            Analytics data
        """
        try:
            async with self.session_factory() as session:
                # Build time filter
                time_filter = []
                if start_time:
                    time_filter.append(TaskExecutionModel.created_at >= start_time)
                if end_time:
                    time_filter.append(TaskExecutionModel.created_at <= end_time)

                base_filter = and_(*time_filter) if time_filter else True

                # Total executions
                total_query = select(func.count(TaskExecutionModel.id)).where(base_filter)
                total_result = await session.execute(total_query)
                total_executions = total_result.scalar()

                # Status distribution
                status_query = (
                    select(TaskExecutionModel.status, func.count(TaskExecutionModel.id))
                    .where(base_filter)
                    .group_by(TaskExecutionModel.status)
                )
                status_result = await session.execute(status_query)
                status_distribution = {status.value: count for status, count in status_result.fetchall()}

                # Task type distribution
                type_query = (
                    select(TaskExecutionModel.task_type, func.count(TaskExecutionModel.id))
                    .where(base_filter)
                    .group_by(TaskExecutionModel.task_type)
                )
                type_result = await session.execute(type_query)
                type_distribution = dict(type_result.fetchall())

                # Performance metrics
                perf_query = select(
                    func.avg(TaskExecutionModel.execution_duration_ms),
                    func.min(TaskExecutionModel.execution_duration_ms),
                    func.max(TaskExecutionModel.execution_duration_ms)
                ).where(
                    and_(base_filter, TaskExecutionModel.execution_duration_ms.isnot(None))
                )
                perf_result = await session.execute(perf_query)
                avg_duration, min_duration, max_duration = perf_result.fetchone()

                self._stats["queries_executed"] += 4

                return {
                    "total_executions": total_executions,
                    "status_distribution": status_distribution,
                    "task_type_distribution": type_distribution,
                    "performance_metrics": {
                        "avg_duration_ms": float(avg_duration) if avg_duration else None,
                        "min_duration_ms": min_duration,
                        "max_duration_ms": max_duration,
                    },
                    "analysis_period": {
                        "start_time": start_time.isoformat() if start_time else None,
                        "end_time": end_time.isoformat() if end_time else None,
                    }
                }

        except SQLAlchemyError as e:
            logger.error(f"Failed to get execution analytics: {e}")
            raise

    def _map_status_to_db(self, status: TaskStatus) -> TaskExecutionStatus:
        """Map domain status to database enum."""
        mapping = {
            TaskStatus.PENDING: TaskExecutionStatus.PENDING,
            TaskStatus.READY: TaskExecutionStatus.READY,
            TaskStatus.EXECUTING: TaskExecutionStatus.EXECUTING,
            TaskStatus.AGGREGATING: TaskExecutionStatus.AGGREGATING,
            TaskStatus.COMPLETED: TaskExecutionStatus.COMPLETED,
            TaskStatus.FAILED: TaskExecutionStatus.FAILED,
            TaskStatus.NEEDS_REPLAN: TaskExecutionStatus.NEEDS_REPLAN,
            TaskStatus.CANCELLED: TaskExecutionStatus.CANCELLED,
            TaskStatus.PAUSED: TaskExecutionStatus.PAUSED,
        }
        return mapping.get(status, TaskExecutionStatus.PENDING)

    async def cleanup_old_executions(self, days: int = 90) -> int:
        """
        Clean up old execution records.

        Args:
            days: Number of days to keep

        Returns:
            Number of records cleaned up
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            async with self.session_factory() as session:
                # Mark as archived instead of deleting
                result = await session.execute(
                    update(TaskExecutionModel)
                    .where(
                        and_(
                            TaskExecutionModel.completed_at < cutoff_date,
                            TaskExecutionModel.is_archived == False
                        )
                    )
                    .values(is_archived=True)
                )

                await session.commit()
                cleaned_count = result.rowcount

                logger.info(f"Archived {cleaned_count} execution records older than {days} days")
                return cleaned_count

        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old executions: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return dict(self._stats)