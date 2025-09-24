"""
Task Node Processor - State Management Handlers.

This module implements state management handlers that process NodeResult actions
from services, maintaining clear separation between business logic (services)
and state management (handlers).

Services own business logic with run() as entry points.
Handlers manage state transitions based on service results.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.task_status import TaskStatus
from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.domain.value_objects.execution_state import ExecutionState

logger = logging.getLogger(__name__)


class NodeActionHandler(ABC):
    """Base handler for managing state transitions based on service results."""

    def __init__(self,
                 graph_manager: GraphStateManager,
                 execution_state: ExecutionState):
        """Initialize handler with state management dependencies.

        Args:
            graph_manager: Graph state management
            execution_state: Execution state tracking
        """
        self.graph = graph_manager
        self.state = execution_state

    @abstractmethod
    async def handle(self, result: NodeResult) -> None:
        """Handle state transition based on service result.

        Args:
            result: NodeResult returned by service

        Note: execution_id is available via self.state.execution_id
        """
        raise NotImplementedError

    def _log_action(self, action: str, task_id: str) -> None:
        """Log handler action for observability."""
        logger.info(f"[{self.__class__.__name__}] {action} for task {task_id}")


class AddSubtasksHandler(NodeActionHandler):
    """Handles ADD_SUBTASKS action by adding nodes to graph and updating state."""

    async def handle(self, result: NodeResult) -> None:
        """Add subtasks to graph - handles both planning and replanning."""
        self._log_action("Processing subtasks", result.task_id)

        if not result.new_nodes:
            logger.warning(f"ADD_SUBTASKS result for {result.task_id} has no new_nodes")
            return

        # Check if this is replanning (parent was NEEDS_REPLAN)
        parent = self.graph.get_node_by_id(result.task_id)
        is_replanning = parent and parent.status == TaskStatus.NEEDS_REPLAN

        if is_replanning:
            # Remove ALL old children first
            old_children = self.graph.get_children_nodes(result.task_id)
            for child in old_children:
                await self.graph.remove_node(child.task_id)
            logger.info(f"Removed {len(old_children)} old children for replanning")

        # Add new subtasks
        for subtask in result.new_nodes:
            await self.graph.add_node(subtask)

        # Transition parent to appropriate status
        if is_replanning:
            # From NEEDS_REPLAN to READY for re-execution
            await self.graph.transition_node_status(
                result.task_id,
                TaskStatus.READY
            )
            logger.info(f"Replanned {result.task_id} with {len(result.new_nodes)} new tasks")
        else:
            # Normal planning - to WAITING_FOR_CHILDREN
            await self.graph.transition_node_status(
                result.task_id,
                TaskStatus.WAITING_FOR_CHILDREN
            )
            logger.info(f"Added {len(result.new_nodes)} subtasks for {result.task_id}")


class CompleteHandler(NodeActionHandler):
    """Handles COMPLETE action by marking node complete and caching result."""

    async def handle(self, result: NodeResult) -> None:
        """Mark node complete and cache result."""
        self._log_action("Completing", result.task_id)

        # Cache the result if available
        if result.envelope:
            await self.state.cache_result(result.task_id, result.envelope)

        # Transition to completed
        await self.graph.transition_node_status(
            result.task_id,
            TaskStatus.COMPLETED
        )

        # Mark as completed in execution state
        await self.state.mark_node_completed(result.task_id)


class AggregateHandler(NodeActionHandler):
    """Handles AGGREGATE action by completing aggregation and transitioning to COMPLETED."""

    async def handle(self, result: NodeResult) -> None:
        """Complete aggregation lifecycle - aggregation work is already done by service."""
        self._log_action("Completing aggregation", result.task_id)

        # The actual aggregation logic is already handled by AggregatorService
        # This handler completes the aggregation lifecycle

        # First transition to aggregating status briefly
        await self.graph.transition_node_status(
            result.task_id,
            TaskStatus.AGGREGATING
        )

        # Cache the aggregated result if provided
        if result.envelope:
            await self.state.cache_result(result.task_id, result.envelope)

        # Complete the aggregation by transitioning to COMPLETED
        await self.graph.transition_node_status(
            result.task_id,
            TaskStatus.COMPLETED
        )

        # Mark as completed in execution state
        await self.state.mark_node_completed(result.task_id)


class ReplanHandler(NodeActionHandler):
    """Handles REPLAN action by transitioning node to needs replan status."""

    async def handle(self, result: NodeResult) -> None:
        """Mark node as needing replanning."""
        self._log_action("Marking for replan", result.task_id)

        # Update metadata with replan reason if provided
        if result.metadata and 'replan_reason' in result.metadata:
            await self.graph.update_node_metadata(
                result.task_id,
                {'replan_reason': result.metadata['replan_reason']}
            )

        # Transition to needs replan
        await self.graph.transition_node_status(
            result.task_id,
            TaskStatus.NEEDS_REPLAN
        )


class RetryHandler(NodeActionHandler):
    """Handles RETRY action by resetting node for retry."""

    async def handle(self, result: NodeResult) -> None:
        """Reset node for retry with incremented retry count."""
        self._log_action("Retrying", result.task_id)

        # Increment actual TaskNode retry_count field
        updated_node = await self.graph.increment_node_retry_count(result.task_id)

        # Update metadata with error information
        await self.graph.update_node_metadata(
            result.task_id,
            {
                'last_error': result.metadata.get('error', 'Unknown') if result.metadata else 'Unknown'
            }
        )

        # Reset to ready state for retry
        await self.graph.transition_node_status(
            result.task_id,
            TaskStatus.READY
        )


class FailHandler(NodeActionHandler):
    """Handles FAIL action by marking node as failed."""

    async def handle(self, result: NodeResult) -> None:
        """Mark node as failed with error information."""
        self._log_action("Failing", result.task_id)

        # Store error in metadata
        error_info = {
            'error': str(result.error) if result.error else 'Unknown error',
            'failure_timestamp': self.state.execution_metadata.get("execution_timestamp")
        }

        # Add any additional metadata from result
        if result.metadata:
            error_info.update(result.metadata)

        await self.graph.update_node_metadata(result.task_id, error_info)

        # Transition to failed
        await self.graph.transition_node_status(
            result.task_id,
            TaskStatus.FAILED
        )

        # Mark as failed in execution state
        await self.state.mark_node_failed(result.task_id)


class NoopHandler(NodeActionHandler):
    """Handles NOOP action by logging and continuing."""

    async def handle(self, result: NodeResult) -> None:
        """Log no-operation and continue without state changes."""
        self._log_action("No-op", result.task_id)

        # Log any metadata for debugging
        if result.metadata:
            logger.debug(f"NOOP metadata for {result.task_id}: {result.metadata}")


class TaskNodeProcessor:
    """Processes NodeResult actions using appropriate state management handlers.

    This class coordinates state management based on service results,
    maintaining clear separation between business logic (services) and
    state management (handlers).
    """

    def __init__(self,
                 graph_manager: GraphStateManager,
                 execution_state: ExecutionState):
        """Initialize processor with state management dependencies.

        Args:
            graph_manager: Graph state management
            execution_state: Execution state tracking
        """
        self.graph = graph_manager
        self.state = execution_state

        # Initialize handlers for each action type
        self.handlers = {
            NodeAction.ADD_SUBTASKS: AddSubtasksHandler(graph_manager, execution_state),
            NodeAction.COMPLETE: CompleteHandler(graph_manager, execution_state),
            NodeAction.AGGREGATE: AggregateHandler(graph_manager, execution_state),
            NodeAction.REPLAN: ReplanHandler(graph_manager, execution_state),
            NodeAction.RETRY: RetryHandler(graph_manager, execution_state),
            NodeAction.FAIL: FailHandler(graph_manager, execution_state),
            NodeAction.NOOP: NoopHandler(graph_manager, execution_state),
        }

    async def process_result(self, result: NodeResult) -> None:
        """Process NodeResult using appropriate handler for state management.

        Args:
            result: NodeResult returned by service

        Note: execution_id is available via self.state.execution_id

        Raises:
            ValueError: If no handler found for action
            Exception: If handler execution fails
        """
        if not result or not result.action:
            logger.error("Invalid NodeResult: missing action")
            return

        handler = self.handlers.get(result.action)
        if not handler:
            logger.error(f"No handler found for action {result.action}")
            raise ValueError(f"No handler found for action {result.action}")

        try:
            await handler.handle(result)
            logger.debug(f"Successfully processed {result.action} for {result.task_id}")
        except Exception as e:
            logger.error(f"Handler failed for {result.task_id} with action {result.action}: {e}")
            # Re-raise to let orchestrator handle the error
            raise

    def get_handler_stats(self) -> Dict[str, Any]:
        """Get statistics about handler usage."""
        return {
            "handlers_available": list(self.handlers.keys()),
            "handler_count": len(self.handlers)
        }