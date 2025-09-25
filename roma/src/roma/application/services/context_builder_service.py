"""
Context Builder Service - ROMA v2.0 Multimodal Context Assembly.

Assembles complete context for agent execution by combining text strings,
file artifacts, and task metadata into structured context objects.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from roma.domain.context import ContextConfig, ContextItem, TaskContext
from roma.domain.entities.artifacts.file_artifact import FileArtifact
from roma.domain.entities.task_node import TaskNode
from roma.domain.interfaces.graph_state_manager import IGraphStateManager
from roma.domain.interfaces.knowledge_store_service import IKnowledgeStoreService
from roma.domain.interfaces.storage import IStorage
from roma.domain.interfaces.toolkit_manager import IToolkitManager
from roma.domain.value_objects.agent_responses import (
    AggregatorResult,
    AtomizerResult,
    ExecutorResult,
    PlanModifierResult,
    PlannerResult,
    SubTask,
)
from roma.domain.value_objects.config.roma_config import ROMAConfig
from roma.domain.value_objects.context_item_type import ContextItemType
from roma.domain.value_objects.dependency_status import DependencyStatus
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.result_envelope import AnyResultEnvelope
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType

logger = logging.getLogger(__name__)


class ContextBuilderService:
    """
    Service for assembling complete context for agent execution.

    Builds ALL context types and exports all variables for flexible template usage.
    Templates can choose which variables to include based on their needs.
    Uses dependency injection to avoid violating clean architecture.
    """

    def __init__(
        self,
        context_config: ContextConfig,
        graph_manager: IGraphStateManager,
        knowledge_store: IKnowledgeStoreService,
        toolkit_manager: IToolkitManager,
        storage_manager: IStorage,
        roma_config: ROMAConfig,
    ):
        """
        Initialize with required dependencies.

        Args:
            context_config: Configuration for context building limits (required)
            graph_manager: Graph state manager for accessing task relationships (required)
            knowledge_store: IKnowledgeStoreService for task execution history (required)
            toolkit_manager: AgnoToolkitManager for tool capability context (required)
            storage_manager: Storage manager for storage paths and configuration (required)
            roma_config: ROMAConfig for application-level project information (required)
        """
        self.logger = logger
        self.config = context_config
        self.graph_manager = graph_manager
        self.knowledge_store = knowledge_store
        self.toolkit_manager = toolkit_manager
        self.storage_manager = storage_manager
        self.roma_config = roma_config

    async def build_context(
        self,
        task: TaskNode,
        overall_objective: str,
        text_content: list[str] | None = None,
        file_artifacts: list[FileArtifact] | None = None,
        parent_results: list[str | AnyResultEnvelope] | None = None,
        sibling_results: list[str | AnyResultEnvelope] | None = None,
        child_results: list[str | AnyResultEnvelope] | None = None,  # For aggregators
        constraints: list[str] | None = None,
        user_preferences: dict[str, Any] | None = None,
        execution_metadata: dict[str, Any] | None = None,
    ) -> TaskContext:
        """
        Build complete context for task execution.

        Args:
            task: The task node being executed
            overall_objective: Root goal/objective
            text_content: Additional text content for context
            file_artifacts: File artifacts to include
            parent_results: Results from parent tasks
            sibling_results: Results from sibling tasks
            constraints: Any execution constraints
            user_preferences: User preferences for execution
            execution_metadata: System execution metadata

        Returns:
            TaskContext: Complete assembled context
        """
        context_items = []

        # Add goal and objective using helper method
        context_items.append(
            self._create_context_item(
                content=f"Current Task: {task.goal}",
                item_type=ContextItemType.TASK_GOAL,
                priority=100,
                metadata={"task_id": task.task_id},
                source="task_goal",
            )
        )

        context_items.append(
            self._create_context_item(
                content=f"Overall Objective: {overall_objective}",
                item_type=ContextItemType.OVERALL_OBJECTIVE,
                priority=95,
                source="overall_objective",
            )
        )

        # Query KnowledgeStore for historical context if available
        try:
            knowledge_context = await self._build_knowledge_context(task)
            # Add KnowledgeStore context items
            context_items.extend(knowledge_context)
        except Exception as e:
            logger.warning(f"Failed to build knowledge context for task {task.task_id}: {e}")

        # Add parent results with high priority
        if parent_results:
            try:
                for i, result in enumerate(parent_results):
                    content = self._extract_content_safely(result)
                    if content:  # Only add non-empty content
                        context_items.append(
                            self._create_context_item(
                                content=content,
                                item_type=ContextItemType.PARENT_RESULT,
                                priority=80,
                                metadata={"index": i},
                                source="parent_result",
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to process parent results for task {task.task_id}: {e}")

        # Add sibling results with medium priority
        if sibling_results:
            try:
                for i, result in enumerate(sibling_results):
                    content = self._extract_content_safely(result)
                    if content:  # Only add non-empty content
                        context_items.append(
                            self._create_context_item(
                                content=content,
                                item_type=ContextItemType.SIBLING_RESULT,
                                priority=60,
                                metadata={"index": i},
                                source="sibling_result",
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to process sibling results for task {task.task_id}: {e}")

        # Add additional text content
        if text_content:
            try:
                for i, content in enumerate(text_content):
                    if content and content.strip():  # Only add non-empty content
                        context_items.append(
                            self._create_context_item(
                                content=content,
                                item_type=ContextItemType.REFERENCE_TEXT,
                                priority=40,
                                metadata={"index": i},
                                source="additional_text",
                            )
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to process additional text content for task {task.task_id}: {e}"
                )

        # Add file artifacts
        if file_artifacts:
            try:
                for artifact in file_artifacts:
                    if artifact:  # Ensure artifact is not None
                        context_items.append(
                            ContextItem.from_artifact(artifact=artifact, priority=50)
                        )
            except Exception as e:
                logger.warning(f"Failed to process file artifacts for task {task.task_id}: {e}")

        # Add child results with high priority for aggregation (must be before multimodal)
        if child_results:
            try:
                for i, result in enumerate(child_results):
                    content = self._extract_content_safely(result)
                    if content:  # Only add non-empty content
                        context_items.append(
                            self._create_context_item(
                                content=content,
                                item_type=ContextItemType.CHILD_RESULT,
                                priority=85,  # Higher than parent for aggregation
                                metadata={"index": i},
                                source="child_result",
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to process child results for task {task.task_id}: {e}")

        # Add multimodal artifacts from KnowledgeStore
        try:
            multimodal_artifacts = await self._build_multimodal_context(task)
            context_items.extend(multimodal_artifacts)
        except Exception as e:
            logger.warning(f"Failed to build multimodal context for task {task.task_id}: {e}")

        # Sort by priority (highest first)
        context_items.sort(key=lambda x: x.priority, reverse=True)

        # Apply context prioritization and overflow handling if enabled
        final_items = context_items
        if self.config.enable_context_prioritization:
            try:
                final_items = self._prioritize_and_limit_context_items(context_items, task)
            except Exception as e:
                logger.error(f"Context prioritization failed for task {task.task_id}: {e}")
                # Fallback: use original items but truncate if too many
                final_items = context_items[
                    : self.config.max_parent_items + self.config.max_sibling_items
                ]

        # Extract execution_id from metadata
        execution_metadata = execution_metadata or {}
        execution_id = execution_metadata.get("execution_id", "unknown")

        # Build final context
        context = TaskContext(
            task=task,
            overall_objective=overall_objective,
            execution_id=execution_id,
            context_items=final_items,
            constraints=constraints or [],
            user_preferences=user_preferences or {},
            execution_metadata=execution_metadata,
        )

        # Validate context before returning (if enabled in config)
        if self.config.enable_context_validation and not self.validate_context(context):
                self.logger.warning(
                    f"Context validation failed for task {task.task_id} - continuing with invalid context"
                )

        self.logger.info(
            f"Built context for task {task.task_id}: "
            f"{len(context.get_text_content())} text items, "
            f"{len(context.get_file_artifacts())} file artifacts "
            f"({'prioritized' if self.config.enable_context_prioritization else 'unfiltered'})"
        )

        return context

    async def export_template_variables(
        self, task: TaskNode, task_context: TaskContext
    ) -> dict[str, Any]:
        """
        Export ALL available variables for template usage - orchestrates sub-methods.

        Templates can choose which variables to include based on their needs.
        Each variable category is mutually exclusive with no overlapping keys.

        Args:
            task: The task node
            task_context: The complete task context

        Returns:
            Dictionary of all available template variables
        """
        # Gather all variable categories in parallel for better performance
        try:
            results = await asyncio.gather(
                self._export_core_variables(task, task_context),
                self._export_temporal_variables(),
                self._export_toolkit_variables(),
                self._export_artifact_variables(task_context.context_items),
                self._export_constraint_variables(task_context),
                self._export_parent_hierarchy_variables(task, task_context),
                self._export_execution_history_variables(task_context),
                self._export_dependency_details_variables(task, task_context),
                self._export_planning_context_variables(task_context),
                self._export_project_environment_variables(task_context),
                self._export_task_relationship_variables(task, task_context),
                self._export_response_model_variables(),
                self._export_execution_metadata_variables(task_context),
                return_exceptions=True,  # Don't fail if one category fails
            )

            # Merge all results safely
            all_variables = {}
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    all_variables.update(result)
                elif isinstance(result, Exception):
                    category_names = [
                        "core",
                        "temporal",
                        "toolkit",
                        "artifact",
                        "constraint",
                        "hierarchy",
                        "execution",
                        "dependency",
                        "planning",
                        "environment",
                        "relationship",
                        "response_model",
                        "metadata",
                    ]
                    category_name = (
                        category_names[i] if i < len(category_names) else f"category_{i}"
                    )
                    logger.warning(f"Variable export failed for {category_name}: {result}")

            self.logger.debug(
                f"Exported {len(all_variables)} template variables for task {task.task_id}"
            )

            return all_variables

        except Exception as e:
            logger.error(f"Template variable export failed for {task.task_id}: {e}")
            # Return minimal core variables as fallback
            return await self._export_core_variables(task, task_context)

    def _export_core_variables(self, task: TaskNode, task_context: TaskContext) -> dict[str, Any]:
        """Export essential core variables (Category 1)."""
        task_type_value = task.task_type.value if task.task_type else "UNKNOWN"
        return {
            "task": task,
            "goal": task.goal,
            "task_type": task_type_value,
            "task_status": task.status.value if task.status else "PENDING",
            "overall_objective": task_context.overall_objective,
            "task_id": task.task_id,
            "parent_id": task.parent_id,
            "is_root_task": task.parent_id is None,
            "task_layer": getattr(task, "layer", 0),
            "current_task_type_info": self._get_task_type_info(task_type_value),
        }

    def _export_execution_metadata_variables(
        self, task_context: TaskContext
    ) -> dict[str, Any]:
        """Export essential execution metadata (Category 12)."""
        execution_id = task_context.execution_metadata.get("execution_id", "unknown")

        # Check if there's any prior work available in context
        has_parent_items = any(
            item.item_type == ContextItemType.PARENT_RESULT for item in task_context.context_items
        )
        has_sibling_items = any(
            item.item_type == ContextItemType.SIBLING_RESULT for item in task_context.context_items
        )

        return {
            "execution_id": execution_id,
            "has_prior_work": has_parent_items or has_sibling_items,
        }

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to specified length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[:max_length].rsplit(" ", 1)[0] + "..."

    def _calculate_context_priority(self, item: ContextItem, task: "TaskNode") -> int:
        """Calculate priority score based on task's agent type and status (agent-aware)."""
        priority = item.priority  # Base priority from item

        # AGENT-SPECIFIC PRIORITIZATION based on what each agent type needs most

        # Aggregator nodes prioritize child results above all else
        if task.status in {TaskStatus.WAITING_FOR_CHILDREN, TaskStatus.AGGREGATING}:
            if item.item_type == ContextItemType.CHILD_RESULT:
                priority += 50  # Massive boost - children are critical for aggregation
            elif item.metadata.get("is_failed_child"):
                priority += 45  # Failed children need special attention
            elif item.item_type == ContextItemType.PARENT_RESULT:
                priority -= 10  # Deprioritize parent context for aggregators

        # Replan nodes prioritize failure information
        elif task.status == TaskStatus.NEEDS_REPLAN:
            if item.metadata.get("is_failed_child"):
                priority += 40  # Boost failed children info
            elif item.metadata.get("has_error"):
                priority += 30  # Boost error context
            elif item.item_type == ContextItemType.SIBLING_RESULT:
                priority += 20  # Learn from sibling successes

        # Planning nodes prioritize parent and sibling context for decomposition
        elif hasattr(task, "node_type") and task.node_type == NodeType.PLAN:
            if item.item_type == ContextItemType.PARENT_RESULT:
                priority += 20  # Parent critical for understanding scope
            elif item.item_type == ContextItemType.SIBLING_RESULT:
                priority += 15  # Siblings important for coordination
            elif item.item_type == ContextItemType.OVERALL_OBJECTIVE:
                priority += 10  # Keep aligned with overall goal

        # Executor nodes prioritize sibling results to avoid duplication
        elif hasattr(task, "node_type") and task.node_type == NodeType.EXECUTE:
            if item.item_type == ContextItemType.SIBLING_RESULT:
                priority += 25  # Critical to avoid duplicate work
            elif item.item_type == ContextItemType.FILE_ARTIFACT:
                priority += 10  # May need artifacts for execution
            elif item.item_type == ContextItemType.PARENT_RESULT:
                priority += 5  # Some parent context helpful
        else:
            # Default behavior for unknown agent types
            if item.item_type == ContextItemType.PARENT_RESULT:
                priority += 3  # Parent results are highly relevant
            elif item.item_type == ContextItemType.SIBLING_RESULT:
                priority += 2  # Sibling results are moderately relevant
            elif item.item_type == ContextItemType.TEXT_CONTENT:
                priority += 1  # Text content is somewhat relevant

        # Boost recent items (if timestamp available)
        timestamp = item.metadata.get("timestamp")
        if timestamp:
            try:
                # Handle various timestamp formats more robustly
                if isinstance(timestamp, str):
                    # Handle ISO format with 'Z' suffix
                    if timestamp.endswith("Z"):
                        item_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    else:
                        # Try parsing as-is for other ISO formats
                        item_time = datetime.fromisoformat(timestamp)
                else:
                    # Skip non-string timestamps
                    item_time = None

                # Process timestamp if valid
                if item_time is not None:
                    # Ensure timezone awareness
                    if item_time.tzinfo is None:
                        item_time = item_time.replace(tzinfo=UTC)

                    age_minutes = (datetime.now(UTC) - item_time).total_seconds() / 60
                    if age_minutes < 1:  # Just created
                        priority += 3
                    elif age_minutes < 5:  # Very recent
                        priority += 2
                    elif age_minutes < 10:  # Recent within task execution timeframe
                        priority += 1
            except (ValueError, AttributeError, TypeError):
                pass  # Invalid timestamp, no boost

        # Boost items related to current task type
        item_task_type = item.metadata.get("task_type")
        if item_task_type == task.task_type.value:
            priority += 1

        return max(0, priority)  # Ensure non-negative

    def _get_task_type_info(self, task_type_value: str) -> dict[str, Any]:
        """Get task type information safely with error handling."""
        try:
            if task_type_value == "UNKNOWN":
                return {}

            task_type = TaskType.from_string(task_type_value)
            return {
                "description": task_type.get_description(),
                "examples": task_type.get_examples(),
                "atomic_indicators": task_type.get_atomic_indicators(),
                "composite_indicators": task_type.get_composite_indicators(),
            }
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to get task type info for '{task_type_value}': {e}")
            return {}

    def _prioritize_and_limit_context_items(
        self, context_items: list[ContextItem], task: "TaskNode"
    ) -> list[ContextItem]:
        """Prioritize context items and apply overflow limits when context is too large."""
        if not context_items:
            return context_items

        # Calculate priorities for all items
        prioritized_items = []
        for item in context_items:
            priority_score = self._calculate_context_priority(item, task)
            prioritized_items.append((priority_score, item))

        # Sort by priority (highest first)
        prioritized_items.sort(key=lambda x: x[0], reverse=True)

        # Separate by type for type-specific limits
        parent_items = []
        sibling_items = []
        other_items = []

        for priority_score, item in prioritized_items:
            if item.item_type == ContextItemType.PARENT_RESULT:
                parent_items.append((priority_score, item))
            elif item.item_type == ContextItemType.SIBLING_RESULT:
                sibling_items.append((priority_score, item))
            else:
                other_items.append((priority_score, item))

        # Apply type-specific limits
        limited_parents = parent_items[: self.config.max_parent_items]
        limited_siblings = sibling_items[: self.config.max_sibling_items]

        # Filter other items by priority thresholds
        high_priority_others = [
            (score, item)
            for score, item in other_items
            if score >= self.config.priority_high_threshold
        ]
        medium_priority_others = [
            (score, item)
            for score, item in other_items
            if self.config.priority_medium_threshold <= score < self.config.priority_high_threshold
        ]

        # Combine results: parents/siblings first (most relevant), then high priority others, then medium priority
        final_items = []
        final_items.extend([item for _, item in limited_parents])  # Parents most important
        final_items.extend([item for _, item in limited_siblings])  # Siblings second most important
        final_items.extend([item for _, item in high_priority_others])  # High priority others third
        final_items.extend([item for _, item in medium_priority_others[:5]])  # Medium priority last

        logger.debug(
            f"Context prioritization: {len(context_items)} → {len(final_items)} items "
            f"(parents: {len(limited_parents)}, siblings: {len(limited_siblings)}, "
            f"high-priority: {len(high_priority_others)}, medium-priority: {len(medium_priority_others[:5])})"
        )

        return final_items

    def _extract_artifacts_from_context(
        self, context_items: list[ContextItem]
    ) -> list[dict[str, Any]]:
        """Extract artifact information from context items."""
        artifacts = []

        for item in context_items:
            metadata = item.metadata
            if metadata.get("type") in ["image", "audio", "video", "file"]:
                artifacts.append(
                    {"type": metadata["type"], "content": item.content, "metadata": metadata}
                )

        return artifacts

    def _extract_prior_work_from_context(self, context_items: list[ContextItem]) -> dict[str, Any]:
        """Extract prior work from context items."""
        result = {"parent": [], "sibling": [], "child": [], "has_content": False}

        for item in context_items:
            metadata = item.metadata
            if metadata.get("type") == "prior_work":
                data = metadata.get("data", {})
                result["parent"] = data.get("parent", [])
                result["sibling"] = data.get("sibling", [])
                result["child"] = data.get("child", [])
                result["has_content"] = data.get("has_content", False)
                break

        return result

    def _find_dependency_result_in_context(
        self, dep_id: str, task_context: TaskContext
    ) -> dict[str, Any]:
        """Find dependency result in context items or return default status."""
        # Try to find dependency in sibling or parent results
        for item in task_context.context_items:
            if (item.item_type in [ContextItemType.SIBLING_RESULT, ContextItemType.PARENT_RESULT]
                and item.metadata.get("task_id") == dep_id):
                    status = item.metadata.get("status", "completed")
                    return {
                        "status": status,
                        "goal": item.metadata.get("goal", f"Task {dep_id}"),
                        "result_summary": self._truncate_content(str(item.content), 200),
                        "full_result": item.content,
                        "execution_time": item.metadata.get("execution_time"),
                        "task_type": item.metadata.get("task_type", "UNKNOWN"),
                        "metadata": item.metadata,
                        "error": item.metadata.get("error"),
                    }

        # If not found in context, return missing status
        return {
            "status": DependencyStatus.MISSING.value,
            "goal": f"Task {dep_id}",
            "error": "Dependency not found in execution context",
        }

    async def build_aggregation_context(
        self,
        parent_task: TaskNode,
        child_results: list[Any],  # List of result envelopes
        overall_objective: str,
        execution_metadata: dict[str, Any] | None = None,
    ) -> TaskContext:
        """
        Build context for aggregating child results - thin wrapper around build_context.

        Args:
            parent_task: Parent task node that needs aggregation
            child_results: List of result envelopes from completed child tasks
            overall_objective: Overall execution objective
            execution_metadata: System execution metadata

        Returns:
            TaskContext: Context for aggregation with child results
        """
        # Extract text from child results using safe helper method
        child_text = []
        for i, child_result in enumerate(child_results):
            content = self._extract_content_safely(child_result)
            child_text.append(f"Child Result {i + 1}: {content}")

        # Add aggregation instruction
        child_text.append(
            f"Please aggregate the above {len(child_results)} child results to complete "
            f"the parent task: '{parent_task.goal}'. Synthesize coherently."
        )

        # Reuse existing build_context method
        return await self.build_context(
            task=parent_task,
            overall_objective=overall_objective,
            text_content=child_text,
            constraints=[f"Must aggregate {len(child_results)} child results"],
            execution_metadata=execution_metadata,
        )

    async def build_per_node_context(
        self,
        node: TaskNode,
        base_context: TaskContext,
        execution_state: Any,  # ExecutionState from orchestration layer
    ) -> TaskContext:
        """
        Build enriched context for a specific node by automatically gathering
        parent, sibling, and child results based on node status.

        This method automatically determines what context data to collect:
        - WAITING_FOR_CHILDREN/AGGREGATING nodes: collect child results
        - Other nodes: collect parent and sibling results

        Args:
            node: TaskNode to build context for
            base_context: Base execution context
            execution_state: ExecutionState for accessing cached results and graph

        Returns:
            TaskContext: Enriched context with automatically gathered data
        """

        # Start with base context data
        parent_results = None
        sibling_results = None
        child_results = None

        # Determine what context to collect based on node status
        if node.status in {TaskStatus.WAITING_FOR_CHILDREN, TaskStatus.AGGREGATING}:
            # Aggregation nodes need child results
            child_results = await self._collect_child_results(node, execution_state)
            self.logger.debug(
                f"Collected {len(child_results) if child_results else 0} child results for aggregation node {node.task_id}"
            )
        else:
            # Normal execution nodes need parent and sibling context
            parent_results = await self._collect_parent_results(node, execution_state)
            sibling_results = await self._collect_sibling_results(node, execution_state)
            self.logger.debug(
                f"Collected {len(parent_results) if parent_results else 0} parent and {len(sibling_results) if sibling_results else 0} sibling results for node {node.task_id}"
            )

        # Build enriched context using the main build_context method
        return await self.build_context(
            task=node,
            overall_objective=base_context.overall_objective,
            parent_results=parent_results,
            sibling_results=sibling_results,
            child_results=child_results,
            constraints=base_context.constraints,
            user_preferences=base_context.user_preferences,
            execution_metadata=base_context.execution_metadata,
        )

    async def _collect_child_results(
        self, parent_node: TaskNode, execution_state: Any
    ) -> list[Any] | None:
        """
        Collect results from completed child nodes.

        Args:
            parent_node: Parent node to collect children for
            execution_state: ExecutionState for accessing graph and cached results

        Returns:
            List of child result envelopes, or None if no children
        """
        try:
            # Use the injected graph manager directly
            children_nodes = self.graph_manager.get_children_nodes(parent_node.task_id)
            if not children_nodes:
                return None

            child_envelopes = []
            for child_node in children_nodes:
                if child_node.status == TaskStatus.COMPLETED:
                    child_result = await execution_state.get_cached_result(child_node.task_id)
                    if child_result:
                        child_envelopes.append(child_result)

            return child_envelopes if child_envelopes else None

        except Exception as e:
            self.logger.error(f"Error collecting child results for {parent_node.task_id}: {e}")
            return None

    async def _collect_parent_results(
        self, node: TaskNode, execution_state: Any
    ) -> list[Any] | None:
        """
        Collect results from parent nodes.

        Args:
            node: Child node to collect parent results for
            execution_state: ExecutionState for accessing cached results

        Returns:
            List of parent result envelopes, or None if no parent
        """
        if not node.parent_id:
            return None

        try:
            parent_result = await execution_state.get_cached_result(node.parent_id)
            return [parent_result] if parent_result else None
        except Exception as e:
            self.logger.error(f"Error collecting parent results for {node.task_id}: {e}")
            return None

    async def _collect_sibling_results(
        self, node: TaskNode, execution_state: Any
    ) -> list[Any] | None:
        """
        Collect results from sibling nodes (same parent).

        Args:
            node: Node to collect sibling results for
            execution_state: ExecutionState for accessing graph and cached results

        Returns:
            List of sibling result envelopes, or None if no siblings
        """
        if not node.parent_id:
            return None  # Root nodes have no siblings

        try:
            # Use the injected graph manager directly to find sibling nodes
            sibling_nodes = self.graph_manager.get_children_nodes(node.parent_id)
            if not sibling_nodes:
                return None

            sibling_envelopes = []
            for sibling_node in sibling_nodes:
                # Skip self and only include completed siblings
                if (
                    sibling_node.task_id != node.task_id
                    and sibling_node.status == TaskStatus.COMPLETED
                ):
                    sibling_result = await execution_state.get_cached_result(sibling_node.task_id)
                    if sibling_result:
                        sibling_envelopes.append(sibling_result)

            return sibling_envelopes if sibling_envelopes else None

        except Exception as e:
            self.logger.error(f"Error collecting sibling results for {node.task_id}: {e}")
            return None

    def validate_context(self, context: TaskContext) -> bool:
        """
        Validate that context is properly structured.

        Args:
            context: Context to validate

        Returns:
            bool: True if context is valid
        """
        try:
            # Check required fields
            if not context.task or not context.overall_objective:
                self.logger.warning("Context missing required task or overall_objective")
                return False

            # Check context items structure
            for item in context.context_items:
                if not item.item_id or not item.item_type:
                    self.logger.warning(f"Context item missing id or type: {item}")
                    return False

            # Agent-specific validation based on task status
            task = context.task

            # Aggregator nodes MUST have child results
            if task.status in {TaskStatus.WAITING_FOR_CHILDREN, TaskStatus.AGGREGATING}:
                child_items = context.get_by_item_type(ContextItemType.CHILD_RESULT)
                if not child_items:
                    self.logger.error(
                        f"Aggregator node {task.task_id} missing required child results!"
                    )
                    return False

            # Replan nodes should have failure context
            elif task.status == TaskStatus.NEEDS_REPLAN:
                failure_items = [
                    item
                    for item in context.context_items
                    if item.metadata.get("is_failed_child") or item.metadata.get("has_error")
                ]
                if not failure_items:
                    self.logger.warning(f"Replan node {task.task_id} has no failure context")

            # Validate file artifacts are accessible (but don't fail on inaccessible)
            for artifact in context.get_file_artifacts():
                if not artifact.is_accessible():
                    self.logger.warning(f"File artifact not accessible: {artifact.name}")

            return True

        except Exception as e:
            self.logger.error(f"Context validation failed: {e}")
            return False

    # ========================================================================
    # HELPER METHODS (eliminate duplication)
    # ========================================================================

    def _safe_dependency_status_conversion(self, status_value: Any) -> DependencyStatus:
        """
        Safely convert any status value to DependencyStatus enum.

        Args:
            status_value: Status value (string, enum, or other)

        Returns:
            DependencyStatus enum value
        """
        if isinstance(status_value, DependencyStatus):
            return status_value

        if isinstance(status_value, str):
            # Map common string values to enum values
            status_mapping = {
                "completed": DependencyStatus.COMPLETED,
                "failed": DependencyStatus.FAILED,
                "pending": DependencyStatus.PENDING,
                "executing": DependencyStatus.PENDING,  # Treat as pending
                "ready": DependencyStatus.PENDING,  # Treat as pending
                "missing": DependencyStatus.MISSING,
                "blocked": DependencyStatus.FAILED,  # Treat as failed
            }

            normalized_status = status_value.lower().strip()
            if normalized_status in status_mapping:
                return status_mapping[normalized_status]

            # Try direct enum name lookup
            try:
                return DependencyStatus[status_value.upper()]
            except KeyError:
                pass

        # Default fallback for unknown values
        logger.warning(f"Unknown dependency status value '{status_value}', treating as MISSING")
        return DependencyStatus.MISSING

    def _create_context_item(
        self,
        content: str,
        item_type: ContextItemType,
        priority: int,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> ContextItem:
        """
        Unified context item creation to eliminate duplication.

        Args:
            content: Text content for the item
            item_type: Type of context item
            priority: Priority for sorting
            metadata: Additional metadata
            source: Source identifier

        Returns:
            ContextItem instance
        """
        final_metadata = {"source": source or item_type.value}
        if metadata:
            final_metadata.update(metadata)

        return ContextItem.from_text(
            content=content, item_type=item_type, metadata=final_metadata, priority=priority
        )

    def _extract_content_safely(self, result: Any) -> str:
        """
        Unified safe content extraction for all result types.

        Args:
            result: Result of any type (envelope, string, etc.)

        Returns:
            Extracted content as string
        """
        if result is None:
            return "[No content]"

        if isinstance(result, str):
            return result

        try:
            if hasattr(result, "extract_primary_output"):
                return result.extract_primary_output()
            elif hasattr(result, "output_text"):
                return result.output_text
            elif hasattr(result, "result"):
                return str(result.result)
            else:
                return str(result)
        except Exception as e:
            logger.warning(f"Content extraction failed: {e}")
            return f"[Content extraction failed: {type(result).__name__}]"

    def _extract_result_content(self, result: AnyResultEnvelope) -> str:
        """
        Legacy method - use _extract_content_safely instead.

        Args:
            result: Result envelope of any type

        Returns:
            Extracted content as string
        """
        return self._extract_content_safely(result)

    # ========================================================================
    # MUTUALLY EXCLUSIVE VARIABLE EXPORT METHODS (v1 COMPATIBILITY)
    # ========================================================================

    async def _export_temporal_variables(self) -> dict[str, Any]:
        """Export temporal context variables (Category 2)."""
        now = datetime.now(UTC)

        temporal_context = await self.build_temporal_context()
        return {
            "temporal_context": temporal_context,
            "current_date": temporal_context["current_date"],
            "current_year": temporal_context["current_year"],
            "current_timestamp": now.isoformat(),
        }

    async def _export_toolkit_variables(self) -> dict[str, Any]:
        """Export tool capability variables (Category 3)."""
        toolkit_context = await self.build_toolkit_context()
        tools = toolkit_context.get("toolkits", [])
        return {
            "tool_context": toolkit_context,
            "available_tools": tools,
            "has_tools": toolkit_context["available"],
        }

    async def _export_artifact_variables(self, context_items: list[ContextItem]) -> dict[str, Any]:
        """Export artifact-related variables (Category 4)."""
        artifacts = self._extract_artifacts_from_context(context_items)
        return {
            "artifacts": artifacts,
            "has_artifacts": bool(artifacts),
            "artifact_types": list({art["type"] for art in artifacts}) if artifacts else [],
        }

    async def _export_constraint_variables(self, task_context: TaskContext) -> dict[str, Any]:
        """Export constraint and preference variables (Category 5) - simplified for LLM clarity."""
        constraints = task_context.constraints or []
        preferences = task_context.user_preferences or {}

        # Create readable constraint summary
        constraint_summary = (
            "No specific constraints."
            if not constraints
            else f"Constraints: {'; '.join(constraints)}"
        )
        preference_summary = (
            "No specific preferences."
            if not preferences
            else f"Preferences: {', '.join(f'{k}: {v}' for k, v in preferences.items())}"
        )

        return {
            "constraints": constraints,
            "user_preferences": preferences,
            "constraint_summary": constraint_summary,
            "preference_summary": preference_summary,
            "has_constraints": bool(constraints),
            "has_preferences": bool(preferences),
        }

    async def _export_parent_hierarchy_variables(
        self, task: TaskNode, task_context: TaskContext
    ) -> dict[str, Any]:
        """Export enhanced parent hierarchy variables (Category 6) - v1 compatible."""
        # Extract parent results from context
        parent_items = [
            item
            for item in task_context.context_items
            if item.item_type == ContextItemType.PARENT_RESULT
        ]

        parent_chain = []
        current_layer = getattr(task, "layer", 0)

        for item in parent_items:
            parent_chain.append(
                {
                    "goal": item.metadata.get("goal", "Unknown"),
                    "layer": item.metadata.get("layer", current_layer + 1),
                    "task_type": item.metadata.get("task_type", "UNKNOWN"),
                    "result_summary": self._truncate_content(
                        str(item.content), self.config.max_summary_length
                    ),
                    "key_insights": item.metadata.get("key_insights", ""),
                    "constraints_identified": item.metadata.get("constraints", ""),
                    "requirements_specified": item.metadata.get("requirements", ""),
                    "planning_reasoning": item.metadata.get("planning_reasoning", ""),
                    "coordination_notes": item.metadata.get("coordination_notes", ""),
                    "timestamp_completed": item.metadata.get("timestamp", ""),
                }
            )

        return {
            "parent_chain": parent_chain,
            "has_parent_hierarchy": bool(parent_chain),
            "current_layer": current_layer,
            "hierarchy_depth": len(parent_chain),
            "parent_results": parent_chain,  # For template convenience
        }

    async def _export_execution_history_variables(
        self, task_context: TaskContext
    ) -> dict[str, Any]:
        """Export execution history variables (Category 7) - v1 compatible."""
        sibling_items = [
            item
            for item in task_context.context_items
            if item.item_type == ContextItemType.SIBLING_RESULT
        ]

        # Build execution history
        prior_sibling_outputs = []
        for item in sibling_items:
            prior_sibling_outputs.append(
                {
                    "task_goal": item.metadata.get("goal", "Unknown"),
                    "outcome_summary": self._truncate_content(
                        str(item.content), self.config.max_outcome_summary_length
                    ),
                    "full_output_reference_id": item.item_id,
                    "execution_order": item.metadata.get("execution_order", 0),
                    "task_type": item.metadata.get("task_type", "UNKNOWN"),
                    "success": item.metadata.get("success", True),
                }
            )

        # Sort by execution order
        prior_sibling_outputs.sort(key=lambda x: x["execution_order"])

        # Create execution history summary for LLM understanding
        execution_summary = ""
        if prior_sibling_outputs:
            success_count = sum(1 for s in prior_sibling_outputs if s["success"])
            failure_count = len(prior_sibling_outputs) - success_count
            execution_summary = f"Previous execution history: {success_count} successful, {failure_count} failed tasks at this level."
        else:
            execution_summary = "No prior execution history at this level."

        return {
            "prior_sibling_outputs": prior_sibling_outputs,
            "has_execution_history": bool(prior_sibling_outputs),
            "execution_summary": execution_summary,
            "sibling_results": prior_sibling_outputs,  # For template convenience
        }

    async def _export_dependency_details_variables(
        self, task: TaskNode, task_context: TaskContext
    ) -> dict[str, Any]:
        """Export enhanced dependency validation and details (Category 8) - comprehensive dependency context."""
        if not task.has_dependencies:
            return {
                "has_dependencies": False,
                "dependency_summary": "This task has no dependencies and can be executed independently.",
                "dependency_count": 0,
                "dependency_results": [],
                "dependency_validation": {
                    "status": DependencyStatus.COMPLETED.value,
                    "message": "No validation required",
                },
            }

        # Get dependency results from context items
        dependency_results = []
        dependency_statuses = {}
        failed_dependencies = []
        completed_dependencies = []

        for dep_id in task.dependencies:
            # Try to find dependency result in context
            dep_result = self._find_dependency_result_in_context(dep_id, task_context)
            dep_status = self._safe_dependency_status_conversion(dep_result["status"])
            dependency_statuses[dep_id] = dep_status

            if dep_status.is_satisfied:
                completed_dependencies.append(dep_id)
                dependency_results.append(
                    {
                        "dependency_id": dep_id,
                        "goal": dep_result.get("goal", f"Task {dep_id}"),
                        "status": dep_status.value,
                        "result_summary": dep_result.get("result_summary", "No summary available"),
                        "full_result": dep_result.get("full_result"),
                        "execution_time": dep_result.get("execution_time"),
                        "task_type": dep_result.get("task_type", "UNKNOWN"),
                        "metadata": dep_result.get("metadata", {}),
                    }
                )
            elif dep_status.is_blocking:
                failed_dependencies.append(dep_id)
                dependency_results.append(
                    {
                        "dependency_id": dep_id,
                        "goal": dep_result.get("goal", f"Task {dep_id}"),
                        "status": dep_status.value,
                        "error": dep_result.get("error", "Unknown error"),
                        "task_type": dep_result.get("task_type", "UNKNOWN"),
                        "metadata": dep_result.get("metadata", {}),
                    }
                )
            else:
                # Pending, executing, etc.
                dependency_results.append(
                    {
                        "dependency_id": dep_id,
                        "goal": dep_result.get("goal", f"Task {dep_id}"),
                        "status": dep_status.value,
                        "task_type": dep_result.get("task_type", "UNKNOWN"),
                        "metadata": dep_result.get("metadata", {}),
                    }
                )

        # Create validation summary
        total_deps = len(task.dependencies)
        completed_count = len(completed_dependencies)
        failed_count = len(failed_dependencies)
        pending_count = total_deps - completed_count - failed_count

        if failed_count > 0:
            validation_status = DependencyStatus.FAILED
            validation_message = f"Execution blocked: {failed_count} dependencies failed"
        elif pending_count > 0:
            validation_status = DependencyStatus.PENDING
            validation_message = f"Waiting for {pending_count} dependencies to complete"
        else:
            validation_status = DependencyStatus.COMPLETED
            validation_message = "All dependencies completed successfully"

        # Create comprehensive dependency summary
        dependency_summary = (
            f"This task depends on {total_deps} other task(s). "
            f"Status: {completed_count} completed, {failed_count} failed, {pending_count} pending."
        )

        return {
            "has_dependencies": True,
            "dependency_count": total_deps,
            "dependency_ids": list(task.dependencies),
            "dependency_summary": dependency_summary,
            "dependency_results": dependency_results,
            "dependency_validation": {
                "status": validation_status.value,
                "message": validation_message,
                "completed_count": completed_count,
                "failed_count": failed_count,
                "pending_count": pending_count,
                "can_execute": validation_status.is_satisfied,
            },
            "completed_dependencies": completed_dependencies,
            "failed_dependencies": failed_dependencies,
            "dependency_chain_valid": validation_status.is_satisfied,
        }

    async def _export_planning_context_variables(
        self, task_context: TaskContext
    ) -> dict[str, Any]:
        """Export planning-specific context (Category 9) - v1 compatible."""
        planning_depth = task_context.execution_metadata.get("planning_depth", 0)

        # Check for replan context in metadata
        replan_details = task_context.execution_metadata.get("replan_details", {})
        is_replan = bool(replan_details)

        return {
            "planning_depth": planning_depth,
            "is_deep_planning": planning_depth > 2,
            "max_planning_depth": 5,  # TODO: Make configurable
            "parent_task_goal": task_context.execution_metadata.get("parent_goal", ""),
            "is_replan": is_replan,
            "replan_details": replan_details if is_replan else {},
            "replan_reason": replan_details.get("reason", "") if is_replan else "",
            "replan_guidance": replan_details.get("specific_guidance", "") if is_replan else "",
            "global_constraints": task_context.execution_metadata.get("global_constraints", []),
            "planning_preferences": task_context.execution_metadata.get("planning_preferences", []),
        }

    async def _export_project_environment_variables(
        self, task_context: TaskContext
    ) -> dict[str, Any]:
        """Export project environment context (Category 10) - v1 compatible.

        Sources information from SystemManager, StorageManager, and ROMAConfig.
        """
        # Default values
        project_info = {
            "project_id": "",
            "project_name": "ROMA",
            "project_version": "2.0.0",
            "environment": "development",
        }

        storage_info = {
            "mount_path": "",
            "artifacts_dir": "",
            "results_dir": "",
            "plots_dir": "",
            "reports_dir": "",
            "temp_dir": "",
        }

        runtime_info = {
            "active_profile": "",
            "active_executions": 0,
            "system_initialized": False,
        }

        # Get information from ROMAConfig (application-level) with safe attribute access
        if self.roma_config:
            app_config = getattr(self.roma_config, "app", None)
            if app_config:
                project_info.update(
                    {
                        "project_name": getattr(app_config, "name", "ROMA"),
                        "project_version": getattr(app_config, "version", "2.0.0"),
                        "environment": getattr(app_config, "environment", "development"),
                    }
                )
            else:
                logger.warning("ROMAConfig has no app configuration, using defaults")

        # Get information from StorageManager (storage paths)
        if self.storage_manager:
            storage_config = getattr(self.storage_manager, "config", None)
            if storage_config:
                mount_path = getattr(storage_config, "mount_path", "")
                storage_info.update(
                    {
                        "mount_path": mount_path,
                        "artifacts_dir": f"{mount_path}/{getattr(storage_config, 'artifacts_subdir', 'artifacts')}",
                        "results_dir": f"{mount_path}/results",
                        "plots_dir": f"{mount_path}/results/plots",
                        "reports_dir": f"{mount_path}/results/reports",
                        "temp_dir": f"{mount_path}/{getattr(storage_config, 'temp_subdir', 'temp')}",
                    }
                )

        # Get execution ID from task context directly (no crash risk)
        execution_id = task_context.execution_id
        project_info["project_id"] = f"roma_project_{execution_id}"

        # Build environment context string (v1 compatible)
        environment_context = f"""
Project: {project_info["project_name"]} v{project_info["project_version"]}
Environment: {project_info["environment"]}
Profile: {runtime_info["active_profile"]}

Storage Directories:
- Mount Point: {storage_info["mount_path"]}
- Artifacts: {storage_info["artifacts_dir"]}
- Results: {storage_info["results_dir"]}
  - Plots: {storage_info["plots_dir"]}
  - Reports: {storage_info["reports_dir"]}
- Temporary: {storage_info["temp_dir"]}

Runtime Status:
- System Initialized: {runtime_info["system_initialized"]}
- Active Executions: {runtime_info["active_executions"]}

Use these paths for reading data and saving execution results.
""".strip()

        return {
            # v1 compatible fields
            "project_id": project_info["project_id"],
            "has_project_isolation": bool(project_info["project_id"]),
            "environment_context": environment_context,
            # Enhanced project information
            "project_info": project_info,
            "storage_info": storage_info,
            "runtime_info": runtime_info,
            # Directory information (v1 compatible)
            "project_directories": {
                "mount_path": storage_info["mount_path"],
                "artifacts_dir": storage_info["artifacts_dir"],
                "results_dir": storage_info["results_dir"],
                "plots_dir": storage_info["plots_dir"],
                "reports_dir": storage_info["reports_dir"],
                "temp_dir": storage_info["temp_dir"],
            },
        }

    async def _export_task_relationship_variables(
        self, task: TaskNode, task_context: TaskContext
    ) -> dict[str, Any]:
        """Export simplified task relationship info (Category 11) - LLM optimized."""
        # Extract sibling information
        sibling_items = [
            item
            for item in task_context.context_items
            if item.item_type == ContextItemType.SIBLING_RESULT
        ]

        # Create simple relationship summary for LLM understanding
        if not sibling_items:
            relationship_summary = "This task has no sibling tasks at the same level."
        else:
            completed_count = len(
                [item for item in sibling_items if item.metadata.get("success", True)]
            )
            total_count = len(sibling_items)
            relationship_summary = (
                f"This task is part of a group of {total_count + 1} sibling tasks. "
                f"{completed_count} siblings have completed successfully."
            )

        return {
            "relationship_summary": relationship_summary,
            "has_siblings": bool(sibling_items),
            "position_in_plan": getattr(task, "aux_data", {}).get("position_in_plan", 0)
            if hasattr(task, "aux_data")
            else 0,
        }

    async def _export_response_model_variables(self) -> dict[str, Any]:
        """Export response model schema and examples for template compatibility (Category 13)."""
        # Agent response models are now imported at module top

        def extract_schema_and_examples(model_class):
            """Extract schema and examples from Pydantic model."""
            try:
                schema = model_class.model_json_schema()
                examples = (
                    model_class.get_examples() if hasattr(model_class, "get_examples") else []
                )

                return {"schema": schema, "examples": examples}
            except Exception as e:
                logger.warning(f"Failed to extract schema for {model_class.__name__}: {e}")
                return {"schema": {}, "examples": []}

        # Extract schemas and examples for all models
        atomizer_data = extract_schema_and_examples(AtomizerResult)
        planner_data = extract_schema_and_examples(PlannerResult)
        executor_data = extract_schema_and_examples(ExecutorResult)
        aggregator_data = extract_schema_and_examples(AggregatorResult)
        plan_modifier_data = extract_schema_and_examples(PlanModifierResult)
        subtask_data = extract_schema_and_examples(SubTask)

        return {
            # Individual model access
            "atomizer_schema": atomizer_data["schema"],
            "atomizer_examples": atomizer_data["examples"],
            "planner_schema": planner_data["schema"],
            "planner_examples": planner_data["examples"],
            "executor_schema": executor_data["schema"],
            "executor_examples": executor_data["examples"],
            "aggregator_schema": aggregator_data["schema"],
            "aggregator_examples": aggregator_data["examples"],
            "plan_modifier_schema": plan_modifier_data["schema"],
            "plan_modifier_examples": plan_modifier_data["examples"],
            "subtask_schema": subtask_data["schema"],
            "subtask_examples": subtask_data["examples"],
        }

    # ========================================================================
    # KNOWLEDGE CONTEXT BUILDING METHODS
    # ========================================================================

    async def _build_knowledge_context(self, task: TaskNode) -> list[ContextItem]:
        """
        Build historical context from KnowledgeStore.

        Args:
            task: Current task node

        Returns:
            List of ContextItem objects from KnowledgeStore
        """
        context_items = []

        if not self.knowledge_store:
            return context_items

        try:
            # Get parent context with error handling
            if task.parent_id:
                try:
                    parent_record = await self.knowledge_store.get_record(task.parent_id)
                    if parent_record and parent_record.has_result():
                        context_items.append(
                            self._convert_record_to_context_item(
                                parent_record, ContextItemType.PARENT_RESULT, priority=85
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to get parent record {task.parent_id}: {e}")

            # Get sibling contexts with error handling
            if task.parent_id:
                try:
                    sibling_records = await self.knowledge_store.get_child_records(task.parent_id)
                    for sibling in sibling_records:
                        try:
                            # Skip current task and only include completed siblings with results
                            if (
                                sibling.task_id != task.task_id
                                and sibling.is_completed()
                                and sibling.has_result()
                            ):
                                context_items.append(
                                    self._convert_record_to_context_item(
                                        sibling, ContextItemType.SIBLING_RESULT, priority=65
                                    )
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to process sibling record {getattr(sibling, 'task_id', 'unknown')}: {e}"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to get sibling records for parent {task.parent_id}: {e}"
                    )

            # Get lineage contexts (walk up the ancestry) with error handling
            try:
                lineage_items = await self._build_lineage_context(task)
                context_items.extend(lineage_items)
            except Exception as e:
                logger.warning(f"Failed to build lineage context for task {task.task_id}: {e}")

            logger.debug(
                f"Built KnowledgeStore context: {len(context_items)} items for task {task.task_id}"
            )

        except Exception as e:
            logger.error(f"Failed to build knowledge context for task {task.task_id}: {e}")

        return context_items

    async def _build_lineage_context(self, task: TaskNode) -> list[ContextItem]:
        """Build complete lineage from ancestors."""
        context_items = []

        if not self.knowledge_store or not task.parent_id:
            return context_items

        try:
            current_id = task.parent_id
            depth = 1  # Start at 1 since parent is already handled
            max_depth = 5

            while current_id and depth < max_depth:
                try:
                    record = await self.knowledge_store.get_record(current_id)
                    if record and record.has_result():
                        # Add as prior work with decreasing priority
                        context_items.append(
                            self._convert_record_to_context_item(
                                record,
                                ContextItemType.PRIOR_WORK,
                                priority=75 - depth,
                                metadata_extra={"lineage_depth": depth},
                            )
                        )
                    # Move to next ancestor
                    current_id = getattr(record, "parent_id", None) if record else None
                except Exception as e:
                    logger.warning(
                        f"Failed to get lineage record {current_id} at depth {depth}: {e}"
                    )
                    break  # Stop traversing on error to prevent infinite loop
                depth += 1

        except Exception as e:
            logger.warning(f"Failed to build lineage context for task {task.task_id}: {e}")

        return context_items

    async def _build_multimodal_context(self, task: TaskNode) -> list[ContextItem]:
        """
        Build multimodal artifact context from KnowledgeStore.

        Args:
            task: Current task node

        Returns:
            List of ContextItem objects with multimodal artifacts
        """
        context_items = []

        if not self.knowledge_store:
            return context_items

        try:
            # Get artifact references from KnowledgeStore
            artifact_refs = await self.knowledge_store.get_records_with_artifacts(
                task.task_id, include_siblings=True
            )

            if not artifact_refs:
                return context_items

            # Get ArtifactService through KnowledgeStore dependency
            artifact_service = getattr(self.knowledge_store, "_artifact_service", None)

            if not artifact_service:
                logger.warning("No artifact service available for multimodal context building")
                return context_items

            # Process artifacts with error handling
            for i, artifact_ref in enumerate(artifact_refs[: self.config.max_artifacts]):
                try:
                    # Try to create artifact from storage reference
                    artifact = await artifact_service.create_file_artifact_from_storage(
                        storage_key=artifact_ref,
                        name=f"Historical Artifact {i + 1}",
                        task_id=task.task_id,
                    )

                    if artifact:
                        # Use artifact's own method to determine context type
                        item_type = artifact.get_context_item_type()

                        context_items.append(
                            ContextItem.from_artifact(
                                artifact=artifact,
                                item_type=item_type,
                                priority=45,  # Medium priority for historical artifacts
                            )
                        )
                        logger.debug(
                            f"Added historical artifact: {artifact.name} ({artifact.media_type.value})"
                        )
                    else:
                        # Add as reference if not accessible
                        context_items.append(
                            self._create_artifact_reference_item(artifact_ref, i, "not_accessible")
                        )

                except Exception as e:
                    logger.warning(f"Failed to process artifact {artifact_ref}: {e}")
                    context_items.append(
                        self._create_artifact_reference_item(artifact_ref, i, f"error: {str(e)}")
                    )

            logger.info(f"Processed {len(artifact_refs)} historical artifacts from KnowledgeStore")

        except Exception as e:
            logger.error(f"Failed to build multimodal context for task {task.task_id}: {e}")

        return context_items

    def _convert_record_to_context_item(
        self,
        record,  # KnowledgeRecord type (avoiding import)
        item_type: ContextItemType,
        priority: int,
        metadata_extra: dict[str, Any] | None = None,
    ) -> ContextItem:
        """
        Convert KnowledgeRecord to ContextItem (DRY method).

        Args:
            record: KnowledgeRecord to convert
            item_type: Type of context item
            priority: Priority for sorting
            metadata_extra: Additional metadata

        Returns:
            ContextItem instance
        """
        # Get content summary using KnowledgeRecord's method
        summary = record.get_summary(max_length=200)

        # Build metadata using KnowledgeRecord's to_context_dict method
        context_data = record.to_context_dict()
        metadata = {
            "source": "knowledge_store",
            **context_data,  # Include all context data as metadata
        }

        if metadata_extra:
            metadata.update(metadata_extra)

        return ContextItem.from_text(
            content=f"{record.task_type.value} Task: {record.goal}\nResult: {summary}",
            item_type=item_type,
            metadata=metadata,
            priority=priority,
        )

    def _create_artifact_reference_item(
        self, artifact_ref: str, index: int, reason: str
    ) -> ContextItem:
        """Create a reference item for inaccessible artifacts."""
        return ContextItem.from_text(
            content=f"Historical Artifact Reference: {artifact_ref}",
            item_type=ContextItemType.FILE_ARTIFACT,
            metadata={
                "source": "knowledge_store_artifact_ref",
                "artifact_ref": artifact_ref,
                "index": index,
                "accessible": False,
                "reason": reason,
            },
            priority=35,  # Lower priority for inaccessible artifacts
        )

    async def build_toolkit_context(self) -> dict[str, Any]:
        """
        Build tool availability context.

        Returns:
            Dictionary with tool availability information
        """
        if not self.toolkit_manager:
            return {"available": False, "toolkits": [], "capabilities": []}

        try:
            # Get available tools from toolkit manager
            available_tools = await self.toolkit_manager.get_available_tools()

            # Extract capabilities
            capabilities = []
            toolkits = []

            for tool in available_tools:
                if hasattr(tool, "name"):
                    toolkits.append(
                        {
                            "name": tool.name,
                            "type": getattr(tool, "type", "unknown"),
                            "description": getattr(tool, "description", ""),
                        }
                    )

                if hasattr(tool, "capabilities"):
                    capabilities.extend(tool.capabilities)

            return {
                "available": len(available_tools) > 0,
                "toolkits": toolkits,
                "capabilities": capabilities,
                "tool_count": len(available_tools),
            }

        except Exception as e:
            logger.error(f"Failed to build toolkit context: {e}")
            return {"available": False, "toolkits": [], "capabilities": [], "error": str(e)}
