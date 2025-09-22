"""
Dependency Validation Service for ROMA v2.0.

Provides pre-execution validation to ensure all dependencies are satisfied
before allowing task execution. Integrates with existing graph operations.
"""

import logging
from typing import List, Set, Optional, Dict, Any
from datetime import datetime, timezone

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.dependency_status import DependencyStatus
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph

logger = logging.getLogger(__name__)


class DependencyValidationError(Exception):
    """Raised when dependency validation fails."""

    def __init__(self, message: str, validation_details: Dict[str, Any]):
        super().__init__(message)
        self.validation_details = validation_details


class DependencyValidationResult:
    """Result of dependency validation for a task node."""

    def __init__(
        self,
        node_id: str,
        is_valid: bool,
        missing_dependencies: Optional[Set[str]] = None,
        failed_dependencies: Optional[Set[str]] = None,
        pending_dependencies: Optional[Set[str]] = None,
        validation_message: Optional[str] = None
    ):
        self.node_id = node_id
        self.is_valid = is_valid
        self.missing_dependencies = missing_dependencies or set()
        self.failed_dependencies = failed_dependencies or set()
        self.pending_dependencies = pending_dependencies or set()
        self.validation_message = validation_message or ""
        self.validated_at = datetime.now(timezone.utc)

    @property
    def has_issues(self) -> bool:
        """Check if validation found any issues."""
        return bool(
            self.missing_dependencies or
            self.failed_dependencies or
            self.pending_dependencies
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "node_id": self.node_id,
            "is_valid": self.is_valid,
            "missing_dependencies": list(self.missing_dependencies),
            "failed_dependencies": list(self.failed_dependencies),
            "pending_dependencies": list(self.pending_dependencies),
            "validation_message": self.validation_message,
            "validated_at": self.validated_at.isoformat()
        }


class DependencyValidator:
    """
    Service for validating task dependencies before execution.

    Provides comprehensive dependency validation including:
    - Existence checks for all dependencies
    - Status validation (completed vs failed)
    - Circular dependency detection
    - Dependency chain validation
    """

    def __init__(
        self,
        allow_pending_dependencies: bool = False,
        recovery_manager: Optional[Any] = None
    ):
        """
        Initialize dependency validator.

        Args:
            allow_pending_dependencies: Whether to allow execution with pending dependencies
                                      (useful for testing scenarios)
            recovery_manager: Optional recovery manager for dependency failure handling
        """
        self.allow_pending_dependencies = allow_pending_dependencies
        self.recovery_manager = recovery_manager
        logger.info(f"DependencyValidator initialized (allow_pending={allow_pending_dependencies})")

    def validate_node_dependencies(
        self,
        node: TaskNode,
        graph: DynamicTaskGraph
    ) -> DependencyValidationResult:
        """
        Validate dependencies for a single node.

        Args:
            node: TaskNode to validate
            graph: Current task graph

        Returns:
            DependencyValidationResult with validation details
        """
        node_id = node.task_id
        logger.debug(f"Validating dependencies for node {node_id}")

        # If no dependencies, validation passes
        if not node.has_dependencies:
            return DependencyValidationResult(
                node_id=node_id,
                is_valid=True,
                validation_message="No dependencies to validate"
            )

        missing_deps = set()
        failed_deps = set()
        pending_deps = set()

        # Check each dependency
        for dep_id in node.dependencies:
            dependency_node = graph.get_node(dep_id)

            if dependency_node is None:
                missing_deps.add(dep_id)
                logger.warning(f"Node {node_id} depends on missing node {dep_id}")
            else:
                # Check dependency status
                if dependency_node.status == TaskStatus.FAILED:
                    failed_deps.add(dep_id)
                    logger.warning(f"Node {node_id} depends on failed node {dep_id}")
                elif dependency_node.status in [TaskStatus.PENDING, TaskStatus.READY, TaskStatus.EXECUTING]:
                    pending_deps.add(dep_id)
                    if not self.allow_pending_dependencies:
                        logger.debug(f"Node {node_id} depends on incomplete node {dep_id} (status: {dependency_node.status})")

        # Determine overall validation result
        has_blocking_issues = bool(missing_deps or failed_deps)
        if not self.allow_pending_dependencies:
            has_blocking_issues = has_blocking_issues or bool(pending_deps)

        is_valid = not has_blocking_issues

        # Generate validation message
        issues = []
        if missing_deps:
            issues.append(f"{len(missing_deps)} missing dependencies")
        if failed_deps:
            issues.append(f"{len(failed_deps)} failed dependencies")
        if pending_deps and not self.allow_pending_dependencies:
            issues.append(f"{len(pending_deps)} incomplete dependencies")

        if issues:
            validation_message = f"Dependency validation failed: {', '.join(issues)}"
        else:
            validation_message = "All dependencies satisfied"

        result = DependencyValidationResult(
            node_id=node_id,
            is_valid=is_valid,
            missing_dependencies=missing_deps,
            failed_dependencies=failed_deps,
            pending_dependencies=pending_deps,
            validation_message=validation_message
        )

        if not is_valid:
            logger.warning(f"Dependency validation failed for {node_id}: {validation_message}")
        else:
            logger.debug(f"Dependency validation passed for {node_id}")

        return result

    def validate_ready_nodes(
        self,
        ready_nodes: List[TaskNode],
        graph: DynamicTaskGraph
    ) -> List[DependencyValidationResult]:
        """
        Validate dependencies for multiple ready nodes.

        Args:
            ready_nodes: List of nodes marked as ready
            graph: Current task graph

        Returns:
            List of validation results for each node
        """
        logger.info(f"Validating dependencies for {len(ready_nodes)} ready nodes")

        results = []
        for node in ready_nodes:
            result = self.validate_node_dependencies(node, graph)
            results.append(result)

        # Log summary
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count

        logger.info(f"Dependency validation complete: {valid_count} valid, {invalid_count} invalid")

        return results

    async def get_executable_nodes(
        self,
        ready_nodes: List[TaskNode],
        graph: DynamicTaskGraph
    ) -> List[TaskNode]:
        """
        Filter ready nodes to only include those with satisfied dependencies.

        Args:
            ready_nodes: List of nodes marked as ready
            graph: Current task graph

        Returns:
            List of nodes that can actually be executed
        """
        validation_results = self.validate_ready_nodes(ready_nodes, graph)

        executable_nodes = []
        for node, result in zip(ready_nodes, validation_results):
            if result.is_valid:
                executable_nodes.append(node)
            else:
                logger.info(f"Node {node.task_id} not executable: {result.validation_message}")

        logger.info(f"Filtered {len(ready_nodes)} ready nodes to {len(executable_nodes)} executable nodes")

        # Handle dependency failures with recovery manager
        if self.recovery_manager and len(executable_nodes) < len(ready_nodes):
            await self._handle_dependency_failures(ready_nodes, validation_results, graph)

        return executable_nodes

    def validate_graph_integrity(self, graph: DynamicTaskGraph) -> Dict[str, Any]:
        """
        Validate overall graph integrity including circular dependencies.

        Args:
            graph: Task graph to validate

        Returns:
            Dictionary with validation results and recommendations
        """
        logger.info("Performing comprehensive graph integrity validation")

        issues = []
        warnings = []
        recommendations = []

        # Check for circular dependencies
        if graph.has_cycles():
            cycles = graph.get_cycles()
            issues.append(f"Circular dependencies detected: {len(cycles)} cycles found")
            for cycle in cycles[:3]:  # Limit to first 3 cycles for readability
                issues.append(f"Cycle: {' -> '.join(cycle + [cycle[0]])}")
            recommendations.append("Break dependency cycles by restructuring tasks")

        # Check for orphaned dependencies
        all_nodes = graph.get_all_nodes()
        orphaned_dependencies = set()

        for node in all_nodes:
            for dep_id in node.dependencies:
                if graph.get_node(dep_id) is None:
                    orphaned_dependencies.add(dep_id)

        if orphaned_dependencies:
            warnings.append(f"Orphaned dependencies found: {len(orphaned_dependencies)} missing nodes")
            recommendations.append("Remove references to deleted dependency nodes")

        # Check for nodes with failed dependencies
        nodes_with_failed_deps = []
        failed_nodes = {n.task_id for n in all_nodes if n.status == TaskStatus.FAILED}

        for node in all_nodes:
            if node.status == TaskStatus.PENDING:
                failed_deps = node.dependencies & failed_nodes
                if failed_deps:
                    nodes_with_failed_deps.append(node.task_id)

        if nodes_with_failed_deps:
            warnings.append(f"Nodes waiting for failed dependencies: {len(nodes_with_failed_deps)}")
            recommendations.append("Consider recovery strategies for failed dependency chains")

        # Summary
        is_healthy = not issues
        status = "healthy" if is_healthy else "degraded" if not issues else "critical"

        return {
            "status": status,
            "is_healthy": is_healthy,
            "total_nodes": len(all_nodes),
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "orphaned_dependencies": list(orphaned_dependencies),
            "nodes_with_failed_deps": nodes_with_failed_deps,
            "validation_timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _handle_dependency_failures(
        self,
        ready_nodes: List[TaskNode],
        validation_results: List[DependencyValidationResult],
        graph: DynamicTaskGraph
    ) -> None:
        """
        Handle dependency failures using the recovery manager.

        Args:
            ready_nodes: Original list of ready nodes
            validation_results: Validation results for each node
            graph: Current task graph
        """
        if not self.recovery_manager:
            return

        # Process each failed validation
        for node, result in zip(ready_nodes, validation_results):
            if not result.is_valid and result.has_issues:
                # Handle failed dependencies
                for failed_dep in result.failed_dependencies:
                    try:
                        failed_node = graph.get_node(failed_dep)
                        if failed_node and failed_node.status == TaskStatus.FAILED:
                            # Create a generic exception for dependency failure
                            error = Exception(f"Dependency {failed_dep} failed during execution")

                            # Use recovery manager to handle the failure
                            recovery_result = await self.recovery_manager.handle_failure(
                                failed_node, error, {"dependent_task": node.task_id}
                            )

                            logger.info(
                                f"Recovery action for failed dependency {failed_dep}: {recovery_result.action}"
                            )

                            # Apply recovery result if node update is provided
                            if recovery_result.updated_node:
                                await graph.update_node(recovery_result.updated_node)

                    except Exception as e:
                        logger.error(f"Error during dependency failure recovery for {failed_dep}: {e}")

                # Handle missing dependencies
                for missing_dep in result.missing_dependencies:
                    logger.warning(
                        f"Node {node.task_id} depends on missing dependency {missing_dep}. "
                        "Consider removing the dependency or creating the missing task."
                    )