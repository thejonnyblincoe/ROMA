"""
Recovery Manager Service for ROMA

Provides centralized error recovery and failure handling with circuit breaker pattern.
Prevents infinite loops and manages retry strategies for task execution failures.
"""

from enum import Enum
from typing import Dict, Optional, Set
from datetime import datetime, timedelta
import logging

from pydantic import BaseModel, Field, ConfigDict

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.child_evaluation_result import ChildEvaluationResult
from roma.domain.value_objects.config.execution_config import ExecutionConfig


logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    """Possible recovery actions for failed tasks."""
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    FORCE_ATOMIC = "FORCE_ATOMIC"
    FAIL_PERMANENTLY = "FAIL_PERMANENTLY"
    CIRCUIT_BREAK = "CIRCUIT_BREAK"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class RecoveryResult(BaseModel):
    """Result of recovery attempt."""
    model_config = ConfigDict(frozen=True)
    
    action: RecoveryAction = Field(..., description="Recovery action to take")
    reasoning: str = Field(..., description="Reasoning for the action")
    updated_node: Optional[TaskNode] = Field(default=None, description="Updated task node if applicable")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""
    model_config = ConfigDict(frozen=True)
    
    failure_threshold: int = Field(default=5, description="Failures needed to open circuit")
    timeout_seconds: int = Field(default=300, description="How long circuit stays open")
    success_threshold: int = Field(default=2, description="Successes needed to close circuit from half-open")


class CircuitBreaker(BaseModel):
    """Circuit breaker implementation for task failure protection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    failure_count: int = Field(default=0, description="Current failure count")
    success_count: int = Field(default=0, description="Current success count") 
    state: CircuitBreakerState = Field(default=CircuitBreakerState.CLOSED, description="Current circuit state")
    last_failure_time: Optional[datetime] = Field(default=None, description="Last failure timestamp")
    
    @property
    def timeout(self) -> timedelta:
        """Get timeout as timedelta."""
        return timedelta(seconds=self.config.timeout_seconds)
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        current_time = datetime.now()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and current_time - self.last_failure_time >= self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker moved to HALF_OPEN state")
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record successful execution."""
        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker moved to CLOSED state")
    
    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker returned to OPEN state after failure in HALF_OPEN")


class RecoveryManagerConfig(BaseModel):
    """Configuration for RecoveryManager."""
    model_config = ConfigDict(frozen=True)
    
    max_retries: int = Field(default=3, description="Default maximum retries per task")
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)


class RecoveryManager(BaseModel):
    """
    Centralized recovery manager with circuit breaker pattern.

    Manages retry strategies and prevents infinite loops by:
    1. Tracking retry counts per task
    2. Circuit breaker for system-wide failure protection
    3. Escalating to parent replanning when retries exhausted
    4. Permanent failure for critical errors
    5. Evaluating child failures for parent replanning decisions
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: RecoveryManagerConfig = Field(default_factory=RecoveryManagerConfig)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    circuit_breaker: CircuitBreaker = Field(default_factory=CircuitBreaker)
    permanent_failures: Set[str] = Field(default_factory=set, description="Permanently failed task IDs")
    
    def __init__(self, **data):
        """Initialize recovery manager."""
        super().__init__(**data)
        
        # Configure circuit breaker from config
        self.circuit_breaker = CircuitBreaker(config=self.config.circuit_breaker)
        
        logger.info(
            f"RecoveryManager initialized with max_retries={self.config.max_retries}, "
            f"circuit_breaker_threshold={self.config.circuit_breaker.failure_threshold}"
        )
    
    async def handle_failure(
        self,
        failed_node: TaskNode,
        error: Exception,
        context: Optional[Dict] = None
    ) -> RecoveryResult:
        """
        Handle task failure and determine recovery action.
        
        Args:
            failed_node: The task that failed
            error: The error that occurred
            context: Additional context for recovery decision
            
        Returns:
            RecoveryResult with recommended action
        """
        task_id = failed_node.task_id
        
        # Check for permanent failure
        if task_id in self.permanent_failures:
            return RecoveryResult(
                action=RecoveryAction.FAIL_PERMANENTLY,
                reasoning="Task previously marked as permanently failed"
            )
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN for task {task_id}")
            return RecoveryResult(
                action=RecoveryAction.CIRCUIT_BREAK,
                reasoning="Circuit breaker is open, system under stress"
            )
        
        # Record failure for circuit breaker
        self.circuit_breaker.record_failure()
        
        # Check if task can be retried
        if failed_node.can_retry:
            # Retry the task - increment retry count but keep in FAILED status
            # The SystemManager will handle the transition back to PENDING when ready
            updated_node = failed_node.increment_retry()
            
            logger.info(
                f"Retrying task {task_id} (attempt {updated_node.retry_count}/{updated_node.max_retries})"
            )
            
            return RecoveryResult(
                action=RecoveryAction.RETRY,
                reasoning=f"Retrying task (attempt {updated_node.retry_count}/{updated_node.max_retries})",
                updated_node=updated_node,
                metadata={"error": str(error), "retry_attempt": updated_node.retry_count}
            )
        
        # Retries exhausted - decide next action based on error type and context
        return await self._escalate_failure(failed_node, error, context)
    
    async def _escalate_failure(
        self,
        failed_node: TaskNode,
        error: Exception,
        context: Optional[Dict] = None
    ) -> RecoveryResult:
        """
        Escalate failure when retries are exhausted.
        
        Args:
            failed_node: Task that exhausted retries
            error: The error that occurred
            context: Additional context
            
        Returns:
            RecoveryResult with escalation action
        """
        error_str = str(error).lower()
        task_id = failed_node.task_id
        
        # Critical errors lead to permanent failure
        critical_errors = [
            "invalid_configuration",
            "authentication_failed", 
            "insufficient_permissions",
            "resource_not_found"
        ]
        
        if any(critical in error_str for critical in critical_errors):
            self.permanent_failures.add(task_id)
            
            logger.error(f"Permanent failure for task {task_id}: critical error {error}")
            
            return RecoveryResult(
                action=RecoveryAction.FAIL_PERMANENTLY,
                reasoning=f"Critical error detected: {error}",
                metadata={"error_type": "critical", "error": str(error)}
            )
        
        # If task has parent, trigger replanning
        if failed_node.parent_id:
            logger.info(f"Escalating task {task_id} failure to parent for replanning")
            
            return RecoveryResult(
                action=RecoveryAction.REPLAN,
                reasoning="Retries exhausted, requesting parent to replan",
                metadata={
                    "failed_task_id": task_id,
                    "parent_id": failed_node.parent_id,
                    "error": str(error),
                    "retry_count": failed_node.retry_count
                }
            )
        
        # Root task with no parent - try forcing atomic execution as last resort
        if failed_node.is_composite:
            logger.warning(f"Forcing atomic execution for root task {task_id}")
            
            # Reset retry count for atomic attempt - keep in FAILED status
            updated_node = failed_node.model_copy(update={
                "retry_count": 0,
                "version": failed_node.version + 1
            })
            
            return RecoveryResult(
                action=RecoveryAction.FORCE_ATOMIC,
                reasoning="Root task failed - forcing atomic execution as last resort",
                updated_node=updated_node,
                metadata={"original_error": str(error), "force_atomic": True}
            )
        
        # Atomic root task failed - permanent failure
        self.permanent_failures.add(task_id)
        
        logger.error(f"Permanent failure for atomic root task {task_id}")
        
        return RecoveryResult(
            action=RecoveryAction.FAIL_PERMANENTLY,
            reasoning="Atomic root task failed - no recovery options available",
            metadata={"error": str(error), "task_type": "atomic_root"}
        )
    
    async def record_success(self, task_id: str) -> None:
        """
        Record successful task execution.
        
        Args:
            task_id: ID of successful task
        """
        self.circuit_breaker.record_success()
        
        # Remove from permanent failures if it was there (shouldn't happen, but defensive)
        self.permanent_failures.discard(task_id)
        
        logger.debug(f"Recorded success for task {task_id}")
    
    def is_permanently_failed(self, task_id: str) -> bool:
        """Check if task is permanently failed."""
        return task_id in self.permanent_failures
    
    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.circuit_breaker.state
    
    def get_stats(self) -> Dict:
        """Get recovery manager statistics."""
        return {
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_failure_count": self.circuit_breaker.failure_count,
            "circuit_breaker_success_count": self.circuit_breaker.success_count,
            "permanent_failures_count": len(self.permanent_failures),
            "max_retries": self.config.max_retries
        }
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state (admin operation)."""
        self.circuit_breaker.state = CircuitBreakerState.CLOSED
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.success_count = 0
        self.circuit_breaker.last_failure_time = None
        
        logger.warning("Circuit breaker manually reset to CLOSED state")
    
    def clear_permanent_failures(self) -> None:
        """Clear all permanent failures (admin operation)."""
        failure_count = len(self.permanent_failures)
        self.permanent_failures.clear()
        
        logger.warning(f"Cleared {failure_count} permanent failures")

    def evaluate_terminal_children(
        self,
        parent_node: TaskNode,
        child_nodes: list[TaskNode]
    ) -> ChildEvaluationResult:
        """
        Evaluate terminal children to determine aggregation decision.

        Args:
            parent_node: The parent task node
            child_nodes: List of child task nodes

        Returns:
            ChildEvaluationResult indicating the decision
        """
        if not child_nodes:
            logger.debug(f"Parent {parent_node.task_id} has no children, proceeding with aggregation")
            return ChildEvaluationResult.AGGREGATE_ALL

        # Count terminal statuses
        completed_count = sum(1 for child in child_nodes if child.status == TaskStatus.COMPLETED)
        failed_count = sum(1 for child in child_nodes if child.status == TaskStatus.FAILED)
        total_terminal = completed_count + failed_count
        total_children = len(child_nodes)

        # Check if all children are terminal (completed or failed)
        if total_terminal < total_children:
            # Not all children are terminal yet, continue waiting
            logger.debug(
                f"Parent {parent_node.task_id}: {total_terminal}/{total_children} children terminal, waiting"
            )
            return ChildEvaluationResult.AGGREGATE_ALL  # Will be filtered out by caller

        logger.info(
            f"Parent {parent_node.task_id}: All children terminal. "
            f"Completed: {completed_count}, Failed: {failed_count}"
        )

        # All children are terminal, evaluate failure rate
        if failed_count == 0:
            # No failures, proceed with normal aggregation
            logger.debug(f"Parent {parent_node.task_id}: No failures, proceeding with aggregation")
            return ChildEvaluationResult.AGGREGATE_ALL

        # If replanning is disabled, always aggregate (even with failures)
        if not self.execution_config.replanning_enabled:
            logger.debug(f"Parent {parent_node.task_id}: Replanning disabled, aggregating all")
            return ChildEvaluationResult.AGGREGATE_ALL

        # If partial aggregation is disabled and we have failures, we must replan
        # (otherwise we'd be stuck - can't aggregate with failures, can't proceed)
        if not self.execution_config.enable_partial_aggregation and failed_count > 0:
            logger.warning(
                f"Parent {parent_node.task_id}: Partial aggregation disabled with {failed_count} failures, "
                f"must replan to proceed"
            )
            return ChildEvaluationResult.REPLAN

        # Calculate failure rate and apply threshold logic
        failure_rate = failed_count / total_children
        threshold = self.execution_config.get_failure_threshold(parent_node.task_type)

        logger.info(
            f"Parent {parent_node.task_id}: Failure rate {failure_rate:.2%} "
            f"vs threshold {threshold:.2%} for {parent_node.task_type}"
        )

        if failure_rate > threshold:
            # Too many failures, trigger replanning
            logger.warning(
                f"Parent {parent_node.task_id}: Failure rate {failure_rate:.2%} exceeds "
                f"threshold {threshold:.2%}, triggering replan"
            )
            return ChildEvaluationResult.REPLAN
        else:
            # Below threshold and partial aggregation is enabled, proceed with partial
            logger.info(
                f"Parent {parent_node.task_id}: Failure rate {failure_rate:.2%} below threshold, "
                f"proceeding with partial aggregation"
            )
            return ChildEvaluationResult.AGGREGATE_PARTIAL