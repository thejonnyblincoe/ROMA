"""
Deadlock Detector Service for ROMA v2.0.

Lightweight deadlock detection that leverages existing graph methods.
Uses graph's own cycle detection and node filtering capabilities.
"""

import logging
from typing import List, Optional
from datetime import datetime, timezone

from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.deadlock_analysis import (
    DeadlockReport, DeadlockSummary, DeadlockType, DeadlockSeverity
)
from roma.application.orchestration.graph_state_manager import GraphStateManager


logger = logging.getLogger(__name__)


class DeadlockDetector:
    """
    Lightweight deadlock detection service.

    Uses existing graph methods and avoids duplication of analysis logic.
    Delegates all graph operations to the graph itself.
    """

    def __init__(self,
                 graph_state_manager: GraphStateManager,
                 stall_threshold_seconds: int = 600):
        """
        Initialize deadlock detector.

        Args:
            graph_state_manager: State manager with graph access
            stall_threshold_seconds: Seconds before considering execution stalled
        """
        self.graph_state_manager = graph_state_manager
        self.stall_threshold = stall_threshold_seconds

        # Simple state tracking
        self.monitoring_start_time: Optional[datetime] = None
        self.detected_deadlocks: List[DeadlockReport] = []

        logger.info(f"DeadlockDetector initialized with {stall_threshold_seconds}s stall threshold")

    def start_monitoring(self) -> None:
        """Start deadlock monitoring for execution."""
        self.monitoring_start_time = datetime.now(timezone.utc)
        self.detected_deadlocks.clear()
        logger.info("Deadlock monitoring started")

    def analyze_execution_state(self) -> List[DeadlockReport]:
        """
        Analyze current execution state for deadlocks using graph methods.

        Returns:
            List of detected deadlock reports
        """
        reports: List[DeadlockReport] = []
        current_time = datetime.now(timezone.utc)

        # 1. Check for circular dependencies using graph's cycle detection
        if self.graph_state_manager.graph.has_cycles():
            cycles = self.graph_state_manager.graph.get_cycles()
            if cycles:
                # Get the first cycle found
                cycle_nodes = cycles[0] if cycles[0] else []
                cycle_report = DeadlockReport(
                    deadlock_type=DeadlockType.CIRCULAR_DEPENDENCY,
                    affected_nodes=cycle_nodes,
                    description=f"Circular dependency: {' -> '.join(cycle_nodes + [cycle_nodes[0]] if cycle_nodes else [])}",
                    severity=DeadlockSeverity.CRITICAL,
                    suggested_actions=[
                        "Break dependency cycle by restructuring subtasks",
                        "Review task decomposition for circular dependencies",
                        "Consider manual intervention to resolve dependency loop"
                    ],
                    detection_time=current_time,
                    cycle_path=cycle_nodes
                )
                reports.append(cycle_report)

        # 2. Check for infinite waits using graph's node filtering
        infinite_wait_report = self._detect_infinite_waits(current_time)
        if infinite_wait_report:
            reports.append(infinite_wait_report)

        # Store new reports (avoiding duplicates)
        for report in reports:
            if not self._is_duplicate_deadlock(report):
                self.detected_deadlocks.append(report)
                logger.warning(f"Deadlock detected: {report.deadlock_type.value} - {report.description}")

        return reports

    def _detect_infinite_waits(self, current_time: datetime) -> Optional[DeadlockReport]:
        """Detect nodes waiting for failed dependencies using graph methods."""
        # Use graph's node filtering methods
        pending_nodes = self.graph_state_manager.graph.get_nodes_by_status(TaskStatus.PENDING)
        failed_nodes = self.graph_state_manager.graph.get_nodes_by_status(TaskStatus.FAILED)
        failed_node_ids = {node.task_id for node in failed_nodes}

        infinite_wait_nodes = []
        for node in pending_nodes:
            if node.dependencies:
                # Check if waiting for any failed dependencies
                failed_deps = node.dependencies & failed_node_ids
                if failed_deps:
                    infinite_wait_nodes.append(node.task_id)

        if not infinite_wait_nodes:
            return None

        return DeadlockReport(
            deadlock_type=DeadlockType.INFINITE_WAIT,
            affected_nodes=infinite_wait_nodes,
            description=f"Infinite wait: {len(infinite_wait_nodes)} nodes waiting for failed dependencies",
            severity=DeadlockSeverity.HIGH,
            suggested_actions=[
                "Retry failed dependency nodes",
                "Remove problematic dependencies",
                "Trigger replanning for affected subtrees",
                "Mark waiting nodes as failed to unblock execution"
            ],
            detection_time=current_time
        )

    def _is_duplicate_deadlock(self, new_report: DeadlockReport) -> bool:
        """Check if this deadlock has already been reported."""
        for existing in self.detected_deadlocks:
            if (existing.deadlock_type == new_report.deadlock_type and
                set(existing.affected_nodes) == set(new_report.affected_nodes)):
                return True
        return False

    def get_deadlock_summary(self) -> DeadlockSummary:
        """Get summary of all detected deadlocks."""
        if not self.detected_deadlocks:
            return DeadlockSummary(
                total_deadlocks=0,
                status="healthy",
                monitoring_duration_seconds=self._get_monitoring_duration()
            )

        by_type = {}
        by_severity = {}
        has_critical = False

        for report in self.detected_deadlocks:
            # Count by type
            type_key = report.deadlock_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            # Count by severity
            severity_key = report.severity.value
            by_severity[severity_key] = by_severity.get(severity_key, 0) + 1

            # Check for critical deadlocks
            if report.severity == DeadlockSeverity.CRITICAL:
                has_critical = True

        status = "deadlocked" if has_critical else "degraded"
        latest_type = self.detected_deadlocks[-1].deadlock_type.value if self.detected_deadlocks else None

        return DeadlockSummary(
            total_deadlocks=len(self.detected_deadlocks),
            status=status,
            by_type=by_type,
            by_severity=by_severity,
            monitoring_duration_seconds=self._get_monitoring_duration(),
            latest_deadlock_type=latest_type,
            has_critical_deadlocks=has_critical
        )

    def _get_monitoring_duration(self) -> float:
        """Get monitoring duration in seconds."""
        if not self.monitoring_start_time:
            return 0.0
        return (datetime.now(timezone.utc) - self.monitoring_start_time).total_seconds()

    def clear_deadlocks(self) -> None:
        """Clear detected deadlocks (for recovery scenarios)."""
        cleared_count = len(self.detected_deadlocks)
        self.detected_deadlocks.clear()
        logger.info(f"Cleared {cleared_count} deadlock reports")

    def get_all_deadlocks(self) -> List[DeadlockReport]:
        """Get all detected deadlock reports."""
        return self.detected_deadlocks.copy()