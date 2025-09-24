"""
SQLAlchemy Implementation of Execution History Repository.

Infrastructure layer implementation of execution history persistence using SQLAlchemy.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from uuid import uuid4

from sqlalchemy import select, delete, update, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from roma.domain.interfaces.persistence import ExecutionHistoryRepository
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.persistence import (
    TaskRelationshipType,
    ExecutionRecord,
    ExecutionTreeNode,
    ExecutionAnalytics,
    PerformanceMetrics,
    AnalysisPeriod,
)
from roma.infrastructure.persistence.models.task_execution_model import (
    TaskExecutionModel,
    TaskRelationshipModel,
)

logger = logging.getLogger(__name__)


class SQLAlchemyExecutionHistoryRepository(ExecutionHistoryRepository):
    """
    SQLAlchemy implementation of execution history repository.

    Handles all execution history persistence operations using PostgreSQL.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """
        Initialize repository with session factory.

        Args:
            session_factory: SQLAlchemy async session factory
        """
        self.session_factory = session_factory

    async def create_execution(
        self,
        task: TaskNode,
        execution_context: Optional[dict] = None,
        agent_config: Optional[dict] = None
    ) -> str:
        """Create a new execution record."""
        try:
            async with self.session_factory() as session:
                try:
                    execution = TaskExecutionModel(
                        task_id=task.task_id,
                        goal=task.goal,
                        task_type=task.task_type.value if task.task_type else "UNKNOWN",
                        node_type=task.node_type.value if task.node_type else None,
                        status=task.status,
                        parent_task_id=task.parent_id,
                        root_task_id=task.root_task_id if hasattr(task, 'root_task_id') else task.task_id,
                        depth_level=getattr(task, 'depth_level', 0),
                        task_metadata=task.metadata if hasattr(task, 'metadata') else {},
                        agent_config=agent_config or {},
                        execution_context=execution_context or {}
                    )

                    session.add(execution)
                    await session.commit()

                    logger.info(f"Created execution record for task {task.task_id}")
                    return execution.id

                except Exception as e:
                    await session.rollback()
                    raise

        except SQLAlchemyError as e:
            logger.error(f"Failed to create execution record: {e}")
            raise

    async def update_execution_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[dict] = None,
        error_info: Optional[dict] = None,
        execution_duration_ms: Optional[int] = None
    ) -> None:
        """Update execution status and result."""
        try:
            async with self.session_factory() as session:
                try:
                    update_values = {
                        "status": status,
                        "updated_at": datetime.now(timezone.utc),
                    }

                    if result is not None:
                        update_values["result"] = result

                    if error_info is not None:
                        update_values["error_info"] = error_info

                    if execution_duration_ms is not None:
                        update_values["execution_duration_ms"] = execution_duration_ms

                    if status in [TaskStatus.EXECUTING]:
                        update_values["started_at"] = datetime.now(timezone.utc)
                    elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        update_values["completed_at"] = datetime.now(timezone.utc)

                    await session.execute(
                        update(TaskExecutionModel)
                        .where(TaskExecutionModel.task_id == task_id)
                        .values(**update_values)
                    )

                    await session.commit()

                    logger.debug(f"Updated execution status for task {task_id} to {status}")

                except Exception as e:
                    await session.rollback()
                    raise

        except SQLAlchemyError as e:
            logger.error(f"Failed to update execution status: {e}")
            raise

    async def add_task_relationship(
        self,
        parent_task_id: str,
        child_task_id: str,
        relationship_type: TaskRelationshipType,
        order_index: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Add a relationship between tasks."""
        try:
            async with self.session_factory() as session:
                try:
                    relationship = TaskRelationshipModel(
                        parent_task_id=parent_task_id,
                        child_task_id=child_task_id,
                        relationship_type=relationship_type,
                        order_index=order_index,
                        relationship_metadata=metadata or {}
                    )

                    session.add(relationship)
                    await session.commit()

                    logger.debug(f"Added relationship {parent_task_id} -> {child_task_id}")

                except Exception as e:
                    await session.rollback()
                    raise

        except SQLAlchemyError as e:
            logger.error(f"Failed to add task relationship: {e}")
            raise

    async def get_execution_history(
        self,
        task_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        include_children: bool = False
    ) -> Optional[ExecutionRecord]:
        """Get execution history for a task."""
        try:
            async with self.session_factory() as session:
                query = select(TaskExecutionModel)

                if task_id:
                    query = query.where(TaskExecutionModel.task_id == task_id)
                elif execution_id:
                    query = query.where(TaskExecutionModel.id == execution_id)
                else:
                    return None

                result = await session.execute(query)
                execution = result.scalar_one_or_none()

                if not execution:
                    return None

                # Convert to ExecutionRecord
                record = ExecutionRecord(
                    execution_id=execution.id,
                    task_id=execution.task_id,
                    goal=execution.goal,
                    task_type=execution.task_type,
                    node_type=execution.node_type,
                    status=execution.status.value,
                    parent_task_id=execution.parent_task_id,
                    root_task_id=execution.root_task_id,
                    depth_level=execution.depth_level,
                    started_at=execution.started_at.isoformat() if execution.started_at else None,
                    completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
                    execution_duration_ms=execution.execution_duration_ms,
                    result=execution.result,
                    metadata=execution.task_metadata or {},
                    error_info=execution.error_info,
                    agent_config=execution.agent_config,
                    execution_context=execution.execution_context,
                    retry_count=execution.retry_count,
                    max_retries=execution.max_retries,
                    token_usage=execution.token_usage,
                    cost_info=execution.cost_info,
                    created_at=execution.created_at.isoformat(),
                    updated_at=execution.updated_at.isoformat(),
                )

                # Add children if requested
                if include_children:
                    children = await self.get_child_executions(execution.task_id)
                    record = record.model_copy(update={"children": children})

                return record

        except SQLAlchemyError as e:
            logger.error(f"Failed to get execution history: {e}")
            raise

    async def get_execution_tree(self, root_task_id: str) -> ExecutionTreeNode:
        """Get complete execution tree for a root task using optimized batch loading."""
        try:
            async with self.session_factory() as session:
                # First, get all executions in the tree with a single query
                # This eliminates the N+1 problem by fetching everything at once
                all_executions_query = (
                    select(TaskExecutionModel)
                    .where(TaskExecutionModel.root_task_id == root_task_id)
                    .order_by(TaskExecutionModel.depth_level, TaskExecutionModel.task_id)
                )

                all_executions_result = await session.execute(all_executions_query)
                all_executions = all_executions_result.scalars().all()

                if not all_executions:
                    raise ValueError(f"Root task {root_task_id} not found")

                # Get all relationships in a single query
                relationships_query = (
                    select(TaskRelationshipModel)
                    .join(TaskExecutionModel, TaskRelationshipModel.child_task_id == TaskExecutionModel.task_id)
                    .where(TaskExecutionModel.root_task_id == root_task_id)
                    .order_by(TaskRelationshipModel.order_index, TaskRelationshipModel.created_at)
                )

                relationships_result = await session.execute(relationships_query)
                relationships = relationships_result.scalars().all()

                # Build lookup maps
                executions_by_id = {exec.task_id: exec for exec in all_executions}
                children_by_parent = {}

                for rel in relationships:
                    if rel.parent_task_id not in children_by_parent:
                        children_by_parent[rel.parent_task_id] = []
                    children_by_parent[rel.parent_task_id].append(rel.child_task_id)

                # Build tree recursively using pre-loaded data with defensive checks
                def build_tree_node(task_id: str) -> Optional[ExecutionTreeNode]:
                    if task_id not in executions_by_id:
                        logger.warning(f"Task {task_id} not found in execution tree, skipping")
                        return None

                    execution = executions_by_id[task_id]

                    # Get children from pre-loaded relationships with defensive filtering
                    child_ids = children_by_parent.get(task_id, [])
                    children = []
                    for child_id in child_ids:
                        child_node = build_tree_node(child_id)
                        if child_node is not None:
                            children.append(child_node)

                    return ExecutionTreeNode(
                        execution_id=execution.id,
                        task_id=execution.task_id,
                        goal=execution.goal,
                        task_type=execution.task_type,
                        node_type=execution.node_type,
                        status=execution.status.value,
                        parent_task_id=execution.parent_task_id,
                        depth_level=execution.depth_level,
                        started_at=execution.started_at.isoformat() if execution.started_at else None,
                        completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
                        execution_duration_ms=execution.execution_duration_ms,
                        result=execution.result,
                        error_info=execution.error_info,
                        children=children
                    )

                # Build tree with defensive check for root task
                root_node = build_tree_node(root_task_id)
                if root_node is None:
                    raise ValueError(f"Root task {root_task_id} not found in execution tree")
                return root_node

        except SQLAlchemyError as e:
            logger.error(f"Failed to get execution tree: {e}")
            raise

    async def get_child_executions(self, parent_task_id: str) -> List[ExecutionRecord]:
        """Get child executions for a parent task."""
        try:
            async with self.session_factory() as session:
                query = (
                    select(TaskExecutionModel)
                    .join(TaskRelationshipModel, TaskExecutionModel.task_id == TaskRelationshipModel.child_task_id)
                    .where(TaskRelationshipModel.parent_task_id == parent_task_id)
                    .order_by(TaskRelationshipModel.order_index, TaskRelationshipModel.created_at)
                )

                result = await session.execute(query)
                executions = result.scalars().all()

                records = []
                for execution in executions:
                    record = ExecutionRecord(
                        execution_id=execution.id,
                        task_id=execution.task_id,
                        goal=execution.goal,
                        task_type=execution.task_type,
                        node_type=execution.node_type,
                        status=execution.status.value,
                        parent_task_id=execution.parent_task_id,
                        root_task_id=execution.root_task_id,
                        depth_level=execution.depth_level,
                        started_at=execution.started_at.isoformat() if execution.started_at else None,
                        completed_at=execution.completed_at.isoformat() if execution.completed_at else None,
                        execution_duration_ms=execution.execution_duration_ms,
                        result=execution.result,
                        metadata=execution.task_metadata or {},
                        error_info=execution.error_info,
                        agent_config=execution.agent_config,
                        execution_context=execution.execution_context,
                        retry_count=execution.retry_count,
                        max_retries=execution.max_retries,
                        token_usage=execution.token_usage,
                        cost_info=execution.cost_info,
                        created_at=execution.created_at.isoformat(),
                        updated_at=execution.updated_at.isoformat(),
                    )
                    records.append(record)

                return records

        except SQLAlchemyError as e:
            logger.error(f"Failed to get child executions: {e}")
            raise

    async def get_execution_analytics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> ExecutionAnalytics:
        """Get execution analytics and performance metrics."""
        try:
            async with self.session_factory() as session:
                # Base query with time filtering
                base_query = select(TaskExecutionModel)
                if start_time:
                    base_query = base_query.where(TaskExecutionModel.created_at >= start_time)
                if end_time:
                    base_query = base_query.where(TaskExecutionModel.created_at <= end_time)

                # Total executions
                total_query = select(func.count(TaskExecutionModel.id)).select_from(base_query.subquery())
                total_result = await session.execute(total_query)
                total_executions = total_result.scalar()

                # Status distribution
                status_query = (
                    select(TaskExecutionModel.status, func.count(TaskExecutionModel.id))
                    .select_from(base_query.subquery())
                    .group_by(TaskExecutionModel.status)
                )
                status_result = await session.execute(status_query)
                status_distribution = {status.value: count for status, count in status_result.fetchall()}

                # Task type distribution
                type_query = (
                    select(TaskExecutionModel.task_type, func.count(TaskExecutionModel.id))
                    .select_from(base_query.subquery())
                    .group_by(TaskExecutionModel.task_type)
                )
                type_result = await session.execute(type_query)
                task_type_distribution = dict(type_result.fetchall())

                # Performance metrics
                perf_query = select(
                    func.avg(TaskExecutionModel.execution_duration_ms),
                    func.min(TaskExecutionModel.execution_duration_ms),
                    func.max(TaskExecutionModel.execution_duration_ms),
                    func.count(TaskExecutionModel.id).filter(TaskExecutionModel.status == TaskStatus.COMPLETED),
                    func.count(TaskExecutionModel.id).filter(TaskExecutionModel.status == TaskStatus.FAILED),
                ).select_from(base_query.subquery())

                perf_result = await session.execute(perf_query)
                avg_duration, min_duration, max_duration, completed_count, failed_count = perf_result.fetchone()

                success_rate = (completed_count / total_executions * 100) if total_executions > 0 else 0.0

                performance_metrics = PerformanceMetrics(
                    avg_execution_time_ms=float(avg_duration) if avg_duration else 0.0,
                    min_execution_time_ms=min_duration or 0,
                    max_execution_time_ms=max_duration or 0,
                    success_rate_percent=success_rate,
                    total_failed=failed_count,
                    total_completed=completed_count
                )

                # Analysis period
                period = AnalysisPeriod(
                    start_time=start_time.isoformat() if start_time else None,
                    end_time=end_time.isoformat() if end_time else None,
                    duration_hours=((end_time - start_time).total_seconds() / 3600) if start_time and end_time else None
                )

                return ExecutionAnalytics(
                    total_executions=total_executions,
                    status_distribution=status_distribution,
                    task_type_distribution=task_type_distribution,
                    performance_metrics=performance_metrics,
                    analysis_period=period,
                    service_stats={}  # No instance-level stats
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to get execution analytics: {e}")
            raise

    async def cleanup_old_executions(self, days: int = 90) -> int:
        """Clean up old execution records."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            async with self.session_factory() as session:
                # First delete relationships
                rel_result = await session.execute(
                    delete(TaskRelationshipModel)
                    .where(TaskRelationshipModel.created_at < cutoff_date)
                )

                # Then delete executions
                exec_result = await session.execute(
                    delete(TaskExecutionModel)
                    .where(TaskExecutionModel.created_at < cutoff_date)
                )

                await session.commit()

                total_deleted = rel_result.rowcount + exec_result.rowcount
                logger.info(f"Cleaned up {total_deleted} records older than {days} days")
                return total_deleted

        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old executions: {e}")
            raise