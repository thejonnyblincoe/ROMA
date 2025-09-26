"""Enhanced error types for better error propagation and context."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Categories of errors for better handling."""
    NETWORK = "network"  # Connection issues, timeouts
    VALIDATION = "validation"  # Input validation failures
    RESOURCE = "resource"  # Memory, disk, quota limits
    LOGIC = "logic"  # Business logic errors
    EXTERNAL = "external"  # External service failures
    CONFIGURATION = "configuration"  # Setup/config issues
    TIMEOUT = "timeout"  # Operation timeouts
    AUTHENTICATION = "authentication"  # Auth failures
    RATE_LIMIT = "rate_limit"  # Rate limiting
    UNKNOWN = "unknown"  # Unclassified errors


class TaskHierarchyError(Exception):
    """Enhanced error with task hierarchy context."""

    def __init__(
        self,
        message: str,
        task_id: str,
        task_goal: Optional[str] = None,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.task_id = task_id
        self.task_goal = task_goal
        self.error_category = error_category
        self.severity = severity
        self.original_error = original_error
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.now()

        # Hierarchy tracking
        self.task_path: List[str] = []  # Path from root to failed task
        self.depth = 0
        self.child_errors: List[TaskHierarchyError] = []

    def add_parent_context(self, parent_task_id: str, parent_goal: Optional[str] = None) -> TaskHierarchyError:
        """Add parent task context to error path."""
        self.task_path.insert(0, parent_task_id)
        self.depth += 1
        return self

    def add_child_error(self, child_error: TaskHierarchyError) -> TaskHierarchyError:
        """Add child error to this error's context."""
        self.child_errors.append(child_error)
        return self

    def get_root_cause(self) -> TaskHierarchyError:
        """Get the deepest error in the hierarchy."""
        if not self.child_errors:
            return self
        return max(self.child_errors, key=lambda e: e.depth).get_root_cause()

    def get_error_summary(self) -> str:
        """Get a human-readable summary of the error hierarchy."""
        root = self.get_root_cause()
        summary = f"Task '{root.task_id}' failed: {root.message}"

        if self.task_path:
            path_str = " -> ".join(self.task_path + [self.task_id])
            summary += f"\nTask path: {path_str}"

        if root.recovery_suggestions:
            summary += f"\nSuggestions: {', '.join(root.recovery_suggestions)}"

        return summary


class ModuleError(TaskHierarchyError):
    """Error specific to module execution."""

    def __init__(
        self,
        module_name: str,
        message: str,
        task_id: str,
        **kwargs
    ):
        self.module_name = module_name
        super().__init__(message, task_id, **kwargs)


class PlanningError(ModuleError):
    """Error during task planning/decomposition."""

    def __init__(self, message: str, task_id: str, **kwargs):
        super().__init__("planner", message, task_id, **kwargs)
        self.error_category = ErrorCategory.LOGIC


class ExecutionError(ModuleError):
    """Error during task execution."""

    def __init__(self, message: str, task_id: str, **kwargs):
        super().__init__("executor", message, task_id, **kwargs)


class AggregationError(ModuleError):
    """Error during result aggregation."""

    def __init__(self, message: str, task_id: str, **kwargs):
        super().__init__("aggregator", message, task_id, **kwargs)


class RetryExhaustedError(TaskHierarchyError):
    """Error when all retry attempts have been exhausted."""

    def __init__(
        self,
        task_id: str,
        attempts: int,
        last_error: Exception,
        **kwargs
    ):
        message = f"Task failed after {attempts} attempts: {str(last_error)}"
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(message, task_id, severity=ErrorSeverity.HIGH, **kwargs)