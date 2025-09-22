"""
Context Builder Service - ROMA v2.0 Multimodal Context Assembly.

Assembles complete context for agent execution by combining text strings,
file artifacts, and task metadata into structured context objects.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

from roma.domain.entities.task_node import TaskNode
from roma.domain.entities.artifacts.base_artifact import BaseArtifact
from roma.domain.entities.artifacts.file_artifact import FileArtifact
from roma.domain.value_objects.media_type import MediaType
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.result_envelope import ResultEnvelope, AnyResultEnvelope
from roma.domain.value_objects.dependency_status import DependencyStatus
from roma.domain.value_objects.agent_responses import (
    AtomizerResult, PlannerResult, ExecutorResult, AggregatorResult, PlanModifierResult, SubTask
)
from roma.domain.value_objects.context_item_type import ContextItemType

logger = logging.getLogger(__name__)



class ContextConfig(BaseModel):
    """Configuration for context building limits and options."""

    model_config = ConfigDict(frozen=True)

    # Result limits
    max_parent_results: int = Field(default=10, ge=1, le=50)
    max_sibling_results: int = Field(default=10, ge=1, le=50)
    max_child_results: int = Field(default=20, ge=1, le=100)  # For aggregators

    # Content limits
    max_text_content: int = Field(default=10, ge=1, le=30)
    max_artifacts: int = Field(default=10, ge=1, le=20)

    # Text length limits
    max_text_length: int = Field(default=500, ge=100, le=2000)  # Max length for individual text content
    max_result_length: int = Field(default=1000, ge=200, le=5000)  # Max length for task results
    max_summary_length: int = Field(default=200, ge=50, le=500)  # Max length for parent result summaries
    max_outcome_summary_length: int = Field(default=150, ge=50, le=300)  # Max length for sibling outcome summaries

    # Context prioritization and overflow limits
    enable_context_prioritization: bool = Field(default=True)  # Whether to apply context filtering and prioritization
    max_total_context_tokens: int = Field(default=8000, ge=1000, le=32000)  # Total context size limit to prevent LLM overflow
    max_parent_items: int = Field(default=5, ge=1, le=20)  # Maximum parent results to include when context is large
    max_sibling_items: int = Field(default=8, ge=1, le=30)  # Maximum sibling results to include when context is large
    priority_high_threshold: int = Field(default=8, ge=1, le=10)  # Priority score threshold for high-priority items
    priority_medium_threshold: int = Field(default=5, ge=1, le=10)  # Priority score threshold for medium-priority items



class ContextItem(BaseModel):
    """Single context item with content and metadata."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(default_factory=lambda: str(uuid4()))
    item_type: ContextItemType
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0

    @classmethod
    def from_text(
        cls,
        content: str,
        item_type: ContextItemType,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> "ContextItem":
        """Create context item from text content."""
        return cls(
            item_type=item_type,
            content=content,
            metadata=metadata or {},
            priority=priority
        )

    @classmethod
    def from_artifact(
        cls,
        artifact: BaseArtifact,
        item_type: ContextItemType,
        priority: int = 0
    ) -> "ContextItem":
        """Create context item from artifact."""
        return cls(
            item_type=item_type,
            content=artifact,
            metadata=artifact.metadata,
            priority=priority
        )


class TaskContext(BaseModel):
    """Complete context assembled for agent execution."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, from_attributes=True)
    
    # Core task information
    task: Any  # TaskNode - using Any to avoid Pydantic validation issues
    overall_objective: str
    
    # Context items (ordered by priority)
    context_items: List[ContextItem] = Field(default_factory=list)
    
    # System metadata
    execution_metadata: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    def get_text_content(self) -> List[str]:
        """Extract all text content from context."""
        text_types = {
            ContextItemType.TASK_GOAL,
            ContextItemType.OVERALL_OBJECTIVE,
            ContextItemType.TEMPORAL,
            ContextItemType.TOOLKITS,
            ContextItemType.PARENT_RESULT,
            ContextItemType.SIBLING_RESULT,
            ContextItemType.CHILD_RESULT,
            ContextItemType.PRIOR_WORK,
            ContextItemType.REFERENCE_TEXT
        }
        text_items = [
            item.content for item in self.context_items
            if item.item_type in text_types and isinstance(item.content, str)
        ]
        return text_items

    def get_file_artifacts(self) -> List[FileArtifact]:
        """Extract all file artifacts from context."""
        artifact_types = {
            ContextItemType.IMAGE_ARTIFACT,
            ContextItemType.AUDIO_ARTIFACT,
            ContextItemType.VIDEO_ARTIFACT,
            ContextItemType.FILE_ARTIFACT
        }
        file_items = []
        for item in self.context_items:
            if item.item_type in artifact_types and isinstance(item.content, FileArtifact):
                file_items.append(item.content)
        return file_items

    def get_by_item_type(self, item_type: ContextItemType) -> List[ContextItem]:
        """Get context items by item type."""
        return [item for item in self.context_items if item.item_type == item_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task": self.task.to_dict(),
            "overall_objective": self.overall_objective,
            "context_items": [
                {
                    "item_id": item.item_id,
                    "item_type": item.item_type.value,
                    "content": (
                        item.content.to_dict() if isinstance(item.content, BaseArtifact)
                        else str(item.content)
                    ),
                    "metadata": item.metadata,
                    "priority": item.priority
                }
                for item in self.context_items
            ],
            "execution_metadata": self.execution_metadata,
            "constraints": self.constraints,
            "user_preferences": self.user_preferences,
            "text_count": len(self.get_text_content()),
            "file_count": len(self.get_file_artifacts()),
        }


class ContextBuilderService:
    """
    Service for assembling complete context for agent execution.

    Builds ALL context types and exports all variables for flexible template usage.
    Templates can choose which variables to include based on their needs.
    Uses dependency injection to avoid violating clean architecture.
    """

    def __init__(
        self,
        toolkit_manager: Optional[Any] = None,
        context_config: Optional[ContextConfig] = None,
        system_manager: Optional[Any] = None,
        storage_manager: Optional[Any] = None,
        roma_config: Optional[Any] = None,
        knowledge_store: Optional[Any] = None
    ):
        """
        Initialize with optional components for enhanced context building.

        Args:
            toolkit_manager: Optional toolkit manager for capability context
            context_config: Configuration for context building limits
            system_manager: SystemManager for runtime project information
            storage_manager: Storage manager for storage paths and configuration
            roma_config: ROMAConfig for application-level project information
            knowledge_store: KnowledgeStoreService for task execution history
        """
        self.logger = logger
        self.toolkit_manager = toolkit_manager
        self.config = context_config or ContextConfig()
        self.system_manager = system_manager
        self.storage_manager = storage_manager
        self.roma_config = roma_config
        self.knowledge_store = knowledge_store
    
    async def build_context(
        self,
        task: TaskNode,
        overall_objective: str,
        text_content: Optional[List[str]] = None,
        file_artifacts: Optional[List[FileArtifact]] = None,
        parent_results: Optional[List[Union[str, AnyResultEnvelope]]] = None,
        sibling_results: Optional[List[Union[str, AnyResultEnvelope]]] = None,
        child_results: Optional[List[Union[str, AnyResultEnvelope]]] = None,  # For aggregators
        constraints: Optional[List[str]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        execution_metadata: Optional[Dict[str, Any]] = None
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
        
        # Add goal as highest priority text content
        context_items.append(
            ContextItem.from_text(
                content=f"Current Task: {task.goal}",
                item_type=ContextItemType.TASK_GOAL,
                metadata={"source": "task_goal", "task_id": task.task_id},
                priority=100
            )
        )
        
        # Add overall objective
        context_items.append(
            ContextItem.from_text(
                content=f"Overall Objective: {overall_objective}",
                item_type=ContextItemType.OVERALL_OBJECTIVE,
                metadata={"source": "overall_objective"},
                priority=95
            )
        )

        # Query KnowledgeStore for historical context if available
        knowledge_context = await self._build_knowledge_context(task)

        # Add KnowledgeStore context items
        context_items.extend(knowledge_context)

        # Add parent results with high priority
        if parent_results:
            for i, result in enumerate(parent_results):
                context_items.append(
                    ContextItem.from_text(
                        content=result,
                        item_type=ContextItemType.PARENT_RESULT,
                        metadata={"source": "parent_result", "index": i},
                        priority=80
                    )
                )
        
        # Add sibling results with medium priority
        if sibling_results:
            for i, result in enumerate(sibling_results):
                context_items.append(
                    ContextItem.from_text(
                        content=result,
                        item_type=ContextItemType.SIBLING_RESULT,
                        metadata={"source": "sibling_result", "index": i},
                        priority=60
                    )
                )
        
        # Add additional text content
        if text_content:
            for i, content in enumerate(text_content):
                context_items.append(
                    ContextItem.from_text(
                        content=content,
                        item_type=ContextItemType.REFERENCE_TEXT,
                        metadata={"source": "additional_text", "index": i},
                        priority=40
                    )
                )
        
        # Add file artifacts
        if file_artifacts:
            for artifact in file_artifacts:
                context_items.append(
                    ContextItem.from_artifact(
                        artifact=artifact,
                        priority=50
                    )
                )

        # Add multimodal artifacts from KnowledgeStore
        multimodal_artifacts = await self._build_multimodal_context(task)
        context_items.extend(multimodal_artifacts)

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
                final_items = context_items[:self.config.max_parent_items + self.config.max_sibling_items]

        # Build final context
        context = TaskContext(
            task=task,
            overall_objective=overall_objective,
            context_items=final_items,
            constraints=constraints or [],
            user_preferences=user_preferences or {},
            execution_metadata=execution_metadata or {}
        )
        
        self.logger.info(
            f"Built context for task {task.task_id}: "
            f"{len(context.get_text_content())} text items, "
            f"{len(context.get_file_artifacts())} file artifacts "
            f"({'prioritized' if self.config.enable_context_prioritization else 'unfiltered'})"
        )
        
        return context

    async def export_template_variables(
        self,
        task: TaskNode,
        task_context: TaskContext
    ) -> Dict[str, Any]:
        """
        Export ALL available variables for template usage.

        Templates can choose which variables to include based on their needs.
        Each variable category is mutually exclusive with no overlapping keys.

        Args:
            task: The task node
            task_context: The complete task context

        Returns:
            Dictionary of all available template variables
        """
        # CATEGORY 1: Essential Core Variables (always present)
        task_type_value = task.task_type.value if task.task_type else "UNKNOWN"
        core_variables = {
            "task": task,
            "goal": task.goal,
            "task_type": task_type_value,
            "task_status": task.status.value if task.status else "PENDING",
            "overall_objective": task_context.overall_objective,
            "task_id": task.task_id,
            "parent_id": task.parent_id,
            "is_root_task": task.parent_id is None,
            "task_layer": getattr(task, 'layer', 0),
            # Task type information for templates
            "current_task_type_info": self._get_task_type_info(task_type_value)
        }

        # CATEGORY 2: Temporal Variables
        temporal_variables = await self._export_temporal_variables()

        # CATEGORY 3: Toolkit Variables
        toolkit_variables = await self._export_toolkit_variables()

        # CATEGORY 4: Artifact Variables
        artifact_variables = await self._export_artifact_variables(task_context.context_items)

        # CATEGORY 5: Constraint Variables
        constraint_variables = await self._export_constraint_variables(task_context)

        # CATEGORY 6: Parent Hierarchy Variables (enhanced from v1)
        parent_hierarchy_variables = await self._export_parent_hierarchy_variables(task, task_context)

        # CATEGORY 7: Execution History Variables (enhanced from v1)
        execution_history_variables = await self._export_execution_history_variables(task, task_context)

        # CATEGORY 8: Dependency Variables (enhanced from v1)
        dependency_variables = await self._export_dependency_details_variables(task, task_context)

        # CATEGORY 9: Planning Context Variables (enhanced from v1)
        planning_context_variables = await self._export_planning_context_variables(task, task_context)

        # CATEGORY 10: Project Environment Variables (enhanced from v1)
        project_environment_variables = await self._export_project_environment_variables(task, task_context)

        # CATEGORY 11: Task Relationship Variables (enhanced from v1)
        task_relationship_variables = await self._export_task_relationship_variables(task, task_context)

        # CATEGORY 12: Essential Execution Metadata (only useful info for LLMs)
        execution_id = task_context.execution_metadata.get("execution_id", "unknown")

        # Check if there's any prior work (parent or sibling results)
        has_parent_work = bool(parent_hierarchy_variables.get("has_parent_hierarchy", False))
        has_sibling_work = bool(execution_history_variables.get("has_execution_history", False))

        metadata_variables = {
            "execution_id": execution_id,
            "has_prior_work": has_parent_work or has_sibling_work
        }

        # CATEGORY 13: Response Model Variables (for template compatibility)
        response_model_variables = await self._export_response_model_variables(task)

        # Combine all categories (each category has unique keys)
        all_variables = {
            **core_variables,
            **temporal_variables,
            **toolkit_variables,
            **artifact_variables,
            **constraint_variables,
            **parent_hierarchy_variables,
            **execution_history_variables,
            **dependency_variables,
            **planning_context_variables,
            **project_environment_variables,
            **task_relationship_variables,
            **metadata_variables,
            **response_model_variables,
        }

        self.logger.debug(
            f"Exported {len(all_variables)} mutually exclusive template variables for task {task.task_id}"
        )

        return all_variables

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to specified length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[:max_length].rsplit(' ', 1)[0] + "..."

    def _calculate_context_priority(self, item: ContextItem, task: "TaskNode") -> int:
        """Calculate priority score for context item based on relevance and recency."""
        priority = item.priority  # Base priority from item

        # Boost priority based on item type
        if item.item_type == ContextItemType.PARENT_RESULT:
            priority += 3  # Parent results are highly relevant
        elif item.item_type == ContextItemType.SIBLING_RESULT:
            priority += 2  # Sibling results are moderately relevant
        elif item.item_type == ContextItemType.TEXT_CONTENT:
            priority += 1  # Text content is somewhat relevant

        # Boost recent items (if timestamp available)
        timestamp = item.metadata.get("timestamp")
        if timestamp:
            from datetime import datetime, timezone
            try:
                # Handle various timestamp formats more robustly
                if isinstance(timestamp, str):
                    # Handle ISO format with 'Z' suffix
                    if timestamp.endswith('Z'):
                        item_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
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
                        item_time = item_time.replace(tzinfo=timezone.utc)

                    age_minutes = (datetime.now(timezone.utc) - item_time).total_seconds() / 60
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

    def _get_task_type_info(self, task_type_value: str) -> Dict[str, Any]:
        """Get task type information safely with error handling."""
        try:
            if task_type_value == "UNKNOWN":
                return {}

            task_type = TaskType.from_string(task_type_value)
            return {
                "description": task_type.get_description(),
                "examples": task_type.get_examples(),
                "atomic_indicators": task_type.get_atomic_indicators(),
                "composite_indicators": task_type.get_composite_indicators()
            }
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to get task type info for '{task_type_value}': {e}")
            return {}

    def _prioritize_and_limit_context_items(
        self,
        context_items: List[ContextItem],
        task: "TaskNode"
    ) -> List[ContextItem]:
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
        limited_parents = parent_items[:self.config.max_parent_items]
        limited_siblings = sibling_items[:self.config.max_sibling_items]

        # Filter other items by priority thresholds
        high_priority_others = [
            (score, item) for score, item in other_items
            if score >= self.config.priority_high_threshold
        ]
        medium_priority_others = [
            (score, item) for score, item in other_items
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

    def _extract_artifacts_from_context(self, context_items: List[ContextItem]) -> List[Dict[str, Any]]:
        """Extract artifact information from context items."""
        artifacts = []

        for item in context_items:
            metadata = item.metadata
            if metadata.get("type") in ["image", "audio", "video", "file"]:
                artifacts.append({
                    "type": metadata["type"],
                    "content": item.content,
                    "metadata": metadata
                })

        return artifacts

    def _extract_prior_work_from_context(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Extract prior work from context items."""
        result = {
            "parent": [],
            "sibling": [],
            "child": [],
            "has_content": False
        }

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

    def _find_dependency_result_in_context(self, dep_id: str, task_context: TaskContext) -> Dict[str, Any]:
        """Find dependency result in context items or return default status."""
        # Try to find dependency in sibling or parent results
        for item in task_context.context_items:
            if item.item_type in [ContextItemType.SIBLING_RESULT, ContextItemType.PARENT_RESULT]:
                if item.metadata.get("task_id") == dep_id:
                    status = item.metadata.get("status", "completed")
                    return {
                        "status": status,
                        "goal": item.metadata.get("goal", f"Task {dep_id}"),
                        "result_summary": self._truncate_content(str(item.content), 200),
                        "full_result": item.content,
                        "execution_time": item.metadata.get("execution_time"),
                        "task_type": item.metadata.get("task_type", "UNKNOWN"),
                        "metadata": item.metadata,
                        "error": item.metadata.get("error")
                    }

        # If not found in context, return missing status
        return {
            "status": DependencyStatus.MISSING.value,
            "goal": f"Task {dep_id}",
            "error": "Dependency not found in execution context"
        }

    async def build_aggregation_context(
        self,
        parent_task: TaskNode,
        child_results: List[Any],  # List of result envelopes
        overall_objective: str,
        execution_metadata: Optional[Dict[str, Any]] = None
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
        # Extract text from child results
        child_text = []
        for i, child_result in enumerate(child_results):
            try:
                if hasattr(child_result, 'extract_primary_output'):
                    content = child_result.extract_primary_output()
                elif hasattr(child_result, 'output_text'):
                    content = child_result.output_text
                else:
                    content = str(child_result)
                child_text.append(f"Child Result {i+1}: {content}")
            except Exception:
                child_text.append(f"Child Result {i+1}: [Unable to extract content]")

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
            execution_metadata=execution_metadata
        )

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
                return False
            
            # Check context items structure
            for item in context.context_items:
                if not item.item_id or not item.content_type:
                    return False
            
            # Validate file artifacts are accessible
            for artifact in context.get_file_artifacts():
                if not artifact.is_accessible():
                    self.logger.warning(f"File artifact not accessible: {artifact.name}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Context validation failed: {e}")
            return False

    # ========================================================================
    # MUTUALLY EXCLUSIVE VARIABLE EXPORT METHODS (v1 COMPATIBILITY)
    # ========================================================================

    async def _export_temporal_variables(self) -> Dict[str, Any]:
        """Export temporal context variables (Category 2)."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        temporal_context = await self.build_temporal_context()
        return {
            "temporal_context": temporal_context,
            "current_date": temporal_context["current_date"],
            "current_year": temporal_context["current_year"],
            "current_timestamp": now.isoformat(),
        }

    async def _export_toolkit_variables(self) -> Dict[str, Any]:
        """Export tool capability variables (Category 3)."""
        toolkit_context = await self.build_toolkit_context()
        tools = toolkit_context.get("toolkits", [])
        return {
            "tool_context": toolkit_context,
            "available_tools": tools,
            "has_tools": toolkit_context["available"],
        }

    async def _export_artifact_variables(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Export artifact-related variables (Category 4)."""
        artifacts = self._extract_artifacts_from_context(context_items)
        return {
            "artifacts": artifacts,
            "has_artifacts": bool(artifacts),
            "artifact_types": list(set(art["type"] for art in artifacts)) if artifacts else [],
        }

    async def _export_constraint_variables(self, task_context: TaskContext) -> Dict[str, Any]:
        """Export constraint and preference variables (Category 5) - simplified for LLM clarity."""
        constraints = task_context.constraints or []
        preferences = task_context.user_preferences or {}

        # Create readable constraint summary
        constraint_summary = "No specific constraints." if not constraints else f"Constraints: {'; '.join(constraints)}"
        preference_summary = "No specific preferences." if not preferences else f"Preferences: {', '.join(f'{k}: {v}' for k, v in preferences.items())}"

        return {
            "constraints": constraints,
            "user_preferences": preferences,
            "constraint_summary": constraint_summary,
            "preference_summary": preference_summary,
            "has_constraints": bool(constraints),
            "has_preferences": bool(preferences)
        }

    async def _export_parent_hierarchy_variables(self, task: TaskNode, task_context: TaskContext) -> Dict[str, Any]:
        """Export enhanced parent hierarchy variables (Category 6) - v1 compatible."""
        # Extract parent results from context
        parent_items = [item for item in task_context.context_items
                       if item.item_type == ContextItemType.PARENT_RESULT]

        parent_chain = []
        current_layer = getattr(task, 'layer', 0)

        for item in parent_items:
            parent_chain.append({
                "goal": item.metadata.get("goal", "Unknown"),
                "layer": item.metadata.get("layer", current_layer + 1),
                "task_type": item.metadata.get("task_type", "UNKNOWN"),
                "result_summary": self._truncate_content(str(item.content), self.config.max_summary_length),
                "key_insights": item.metadata.get("key_insights", ""),
                "constraints_identified": item.metadata.get("constraints", ""),
                "requirements_specified": item.metadata.get("requirements", ""),
                "planning_reasoning": item.metadata.get("planning_reasoning", ""),
                "coordination_notes": item.metadata.get("coordination_notes", ""),
                "timestamp_completed": item.metadata.get("timestamp", ""),
            })

        return {
            "parent_chain": parent_chain,
            "has_parent_hierarchy": bool(parent_chain),
            "current_layer": current_layer,
            "hierarchy_depth": len(parent_chain),
            "parent_results": parent_chain  # For template convenience
        }

    async def _export_execution_history_variables(self, task: TaskNode, task_context: TaskContext) -> Dict[str, Any]:
        """Export execution history variables (Category 7) - v1 compatible."""
        sibling_items = [item for item in task_context.context_items
                        if item.item_type == ContextItemType.SIBLING_RESULT]

        # Build execution history
        prior_sibling_outputs = []
        for item in sibling_items:
            prior_sibling_outputs.append({
                "task_goal": item.metadata.get("goal", "Unknown"),
                "outcome_summary": self._truncate_content(str(item.content), self.config.max_outcome_summary_length),
                "full_output_reference_id": item.item_id,
                "execution_order": item.metadata.get("execution_order", 0),
                "task_type": item.metadata.get("task_type", "UNKNOWN"),
                "success": item.metadata.get("success", True),
            })

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
            "sibling_results": prior_sibling_outputs  # For template convenience
        }

    async def _export_dependency_details_variables(self, task: TaskNode, task_context: TaskContext) -> Dict[str, Any]:
        """Export enhanced dependency validation and details (Category 8) - comprehensive dependency context."""
        if not task.has_dependencies:
            return {
                "has_dependencies": False,
                "dependency_summary": "This task has no dependencies and can be executed independently.",
                "dependency_count": 0,
                "dependency_results": [],
                "dependency_validation": {
                    "status": DependencyStatus.COMPLETED.value,
                    "message": "No validation required"
                }
            }

        # Get dependency results from context items
        dependency_results = []
        dependency_statuses = {}
        failed_dependencies = []
        completed_dependencies = []

        for dep_id in task.dependencies:
            # Try to find dependency result in context
            dep_result = self._find_dependency_result_in_context(dep_id, task_context)
            dep_status = DependencyStatus(dep_result["status"])
            dependency_statuses[dep_id] = dep_status

            if dep_status.is_satisfied:
                completed_dependencies.append(dep_id)
                dependency_results.append({
                    "dependency_id": dep_id,
                    "goal": dep_result.get("goal", f"Task {dep_id}"),
                    "status": dep_status.value,
                    "result_summary": dep_result.get("result_summary", "No summary available"),
                    "full_result": dep_result.get("full_result"),
                    "execution_time": dep_result.get("execution_time"),
                    "task_type": dep_result.get("task_type", "UNKNOWN"),
                    "metadata": dep_result.get("metadata", {})
                })
            elif dep_status.is_blocking:
                failed_dependencies.append(dep_id)
                dependency_results.append({
                    "dependency_id": dep_id,
                    "goal": dep_result.get("goal", f"Task {dep_id}"),
                    "status": dep_status.value,
                    "error": dep_result.get("error", "Unknown error"),
                    "task_type": dep_result.get("task_type", "UNKNOWN"),
                    "metadata": dep_result.get("metadata", {})
                })
            else:
                # Pending, executing, etc.
                dependency_results.append({
                    "dependency_id": dep_id,
                    "goal": dep_result.get("goal", f"Task {dep_id}"),
                    "status": dep_status.value,
                    "task_type": dep_result.get("task_type", "UNKNOWN"),
                    "metadata": dep_result.get("metadata", {})
                })

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
        dependency_summary = f"This task depends on {total_deps} other task(s). " \
                           f"Status: {completed_count} completed, {failed_count} failed, {pending_count} pending."

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
                "can_execute": validation_status.is_satisfied
            },
            "completed_dependencies": completed_dependencies,
            "failed_dependencies": failed_dependencies,
            "dependency_chain_valid": validation_status.is_satisfied
        }

    async def _export_planning_context_variables(self, task: TaskNode, task_context: TaskContext) -> Dict[str, Any]:
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

    async def _export_project_environment_variables(self, task: TaskNode, task_context: TaskContext) -> Dict[str, Any]:
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

        # Get information from ROMAConfig (application-level)
        if self.roma_config:
            project_info.update({
                "project_name": getattr(self.roma_config.app, 'name', 'ROMA'),
                "project_version": getattr(self.roma_config.app, 'version', '2.0.0'),
                "environment": getattr(self.roma_config.app, 'environment', 'development'),
            })

        # Get information from StorageManager (storage paths)
        if self.storage_manager:
            storage_config = getattr(self.storage_manager, 'config', None)
            if storage_config:
                mount_path = getattr(storage_config, 'mount_path', '')
                storage_info.update({
                    "mount_path": mount_path,
                    "artifacts_dir": f"{mount_path}/{getattr(storage_config, 'artifacts_subdir', 'artifacts')}",
                    "results_dir": f"{mount_path}/results",
                    "plots_dir": f"{mount_path}/results/plots",
                    "reports_dir": f"{mount_path}/results/reports",
                    "temp_dir": f"{mount_path}/{getattr(storage_config, 'temp_subdir', 'temp')}",
                })

        # Get information from SystemManager (runtime state)
        if self.system_manager:
            runtime_info.update({
                "active_profile": getattr(self.system_manager, '_current_profile', ''),
                "active_executions": len(getattr(self.system_manager, '_active_executions', {})),
                "system_initialized": getattr(self.system_manager, '_initialized', False),
            })

            # Get execution ID from task context metadata - this propagates from the execution session
            execution_id = task_context.execution_metadata.get("execution_id") if task_context.execution_metadata else None
            if execution_id:
                project_info["project_id"] = f"roma_project_{execution_id}"
            else:
                raise ValueError("execution_id not found in task context metadata - execution ID propagation failed")

        # Build environment context string (v1 compatible)
        environment_context = f"""
Project: {project_info['project_name']} v{project_info['project_version']}
Environment: {project_info['environment']}
Profile: {runtime_info['active_profile']}

Storage Directories:
- Mount Point: {storage_info['mount_path']}
- Artifacts: {storage_info['artifacts_dir']}
- Results: {storage_info['results_dir']}
  - Plots: {storage_info['plots_dir']}
  - Reports: {storage_info['reports_dir']}
- Temporary: {storage_info['temp_dir']}

Runtime Status:
- System Initialized: {runtime_info['system_initialized']}
- Active Executions: {runtime_info['active_executions']}

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

    async def _export_task_relationship_variables(self, task: TaskNode, task_context: TaskContext) -> Dict[str, Any]:
        """Export simplified task relationship info (Category 11) - LLM optimized."""
        # Extract sibling information
        sibling_items = [item for item in task_context.context_items
                        if item.item_type == ContextItemType.SIBLING_RESULT]

        # Create simple relationship summary for LLM understanding
        if not sibling_items:
            relationship_summary = "This task has no sibling tasks at the same level."
        else:
            completed_count = len([item for item in sibling_items if item.metadata.get("success", True)])
            total_count = len(sibling_items)
            relationship_summary = f"This task is part of a group of {total_count + 1} sibling tasks. " \
                                 f"{completed_count} siblings have completed successfully."

        return {
            "relationship_summary": relationship_summary,
            "has_siblings": bool(sibling_items),
            "position_in_plan": getattr(task, 'aux_data', {}).get('position_in_plan', 0) if hasattr(task, 'aux_data') else 0
        }

    async def _export_response_model_variables(self, task: TaskNode) -> Dict[str, Any]:
        """Export response model schema and examples for template compatibility (Category 13)."""
        # Import all response models
        from roma.domain.value_objects.agent_responses import (
            AtomizerResult, PlannerResult, ExecutorResult,
            AggregatorResult, PlanModifierResult, SubTask
        )

        def extract_schema_and_examples(model_class):
            """Extract schema and examples from Pydantic model."""
            try:
                schema = model_class.model_json_schema()
                examples = model_class.get_examples() if hasattr(model_class, 'get_examples') else []

                return {
                    "schema": schema,
                    "examples": examples
                }
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
            "subtask_examples": subtask_data["examples"]
        }

    # ========================================================================
    # KNOWLEDGE CONTEXT BUILDING METHODS
    # ========================================================================

    async def _build_knowledge_context(self, task: TaskNode) -> List[ContextItem]:
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
            # Get parent context
            if task.parent_id:
                parent_record = await self.knowledge_store.get_record(task.parent_id)
                if parent_record and parent_record.has_result():
                    context_items.append(
                        self._convert_record_to_context_item(
                            parent_record,
                            ContextItemType.PARENT_RESULT,
                            priority=85
                        )
                    )

            # Get sibling contexts
            if task.parent_id:
                sibling_records = await self.knowledge_store.get_child_records(task.parent_id)
                for sibling in sibling_records:
                    # Skip current task and only include completed siblings with results
                    if (sibling.task_id != task.task_id and
                        sibling.is_completed() and
                        sibling.has_result()):
                        context_items.append(
                            self._convert_record_to_context_item(
                                sibling,
                                ContextItemType.SIBLING_RESULT,
                                priority=65
                            )
                        )

            # Get lineage contexts (walk up the ancestry)
            lineage_items = await self._build_lineage_context(task)
            context_items.extend(lineage_items)

            logger.debug(f"Built KnowledgeStore context: {len(context_items)} items for task {task.task_id}")

        except Exception as e:
            logger.error(f"Failed to build knowledge context for task {task.task_id}: {e}")

        return context_items

    async def _build_lineage_context(self, task: TaskNode) -> List[ContextItem]:
        """Build complete lineage from ancestors."""
        context_items = []

        if not self.knowledge_store or not task.parent_id:
            return context_items

        try:
            current_id = task.parent_id
            depth = 1  # Start at 1 since parent is already handled
            max_depth = 5

            while current_id and depth < max_depth:
                record = await self.knowledge_store.get_record(current_id)
                if record and record.has_result():
                    # Add as prior work with decreasing priority
                    context_items.append(
                        self._convert_record_to_context_item(
                            record,
                            ContextItemType.PRIOR_WORK,
                            priority=75 - depth,
                            metadata_extra={"lineage_depth": depth}
                        )
                    )
                    current_id = record.parent_task_id
                    depth += 1
                else:
                    break

        except Exception as e:
            logger.error(f"Failed to build lineage context for task {task.task_id}: {e}")

        return context_items

    async def _build_multimodal_context(self, task: TaskNode) -> List[ContextItem]:
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
            artifact_service = getattr(self.knowledge_store, '_artifact_service', None)

            if not artifact_service:
                logger.warning("No artifact service available for multimodal context building")
                return context_items

            # Process artifacts with error handling
            for i, artifact_ref in enumerate(artifact_refs[:self.config.max_artifacts]):
                try:
                    # Try to create artifact from storage reference
                    artifact = await artifact_service.create_file_artifact_from_storage(
                        storage_key=artifact_ref,
                        name=f"Historical Artifact {i+1}",
                        task_id=task.task_id
                    )

                    if artifact:
                        # Use artifact's own method to determine context type
                        item_type = artifact.get_context_item_type()

                        context_items.append(
                            ContextItem.from_artifact(
                                artifact=artifact,
                                item_type=item_type,
                                priority=45  # Medium priority for historical artifacts
                            )
                        )
                        logger.debug(f"Added historical artifact: {artifact.name} ({artifact.media_type.value})")
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
        metadata_extra: Optional[Dict[str, Any]] = None
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
            **context_data  # Include all context data as metadata
        }

        if metadata_extra:
            metadata.update(metadata_extra)

        return ContextItem.from_text(
            content=f"{record.task_type.value} Task: {record.goal}\nResult: {summary}",
            item_type=item_type,
            metadata=metadata,
            priority=priority
        )

    def _create_artifact_reference_item(self, artifact_ref: str, index: int, reason: str) -> ContextItem:
        """Create a reference item for inaccessible artifacts."""
        return ContextItem.from_text(
            content=f"Historical Artifact Reference: {artifact_ref}",
            item_type=ContextItemType.FILE_ARTIFACT,
            metadata={
                "source": "knowledge_store_artifact_ref",
                "artifact_ref": artifact_ref,
                "index": index,
                "accessible": False,
                "reason": reason
            },
            priority=35  # Lower priority for inaccessible artifacts
        )

    async def build_toolkit_context(self) -> Dict[str, Any]:
        """
        Build tool availability context.

        Returns:
            Dictionary with tool availability information
        """
        if not self.toolkit_manager:
            return {
                "available": False,
                "toolkits": [],
                "capabilities": []
            }

        try:
            # Get available tools from toolkit manager
            available_tools = await self.toolkit_manager.get_available_tools()

            # Extract capabilities
            capabilities = []
            toolkits = []

            for tool in available_tools:
                if hasattr(tool, 'name'):
                    toolkits.append({
                        "name": tool.name,
                        "type": getattr(tool, 'type', 'unknown'),
                        "description": getattr(tool, 'description', '')
                    })

                if hasattr(tool, 'capabilities'):
                    capabilities.extend(tool.capabilities)

            return {
                "available": len(available_tools) > 0,
                "toolkits": toolkits,
                "capabilities": capabilities,
                "tool_count": len(available_tools)
            }

        except Exception as e:
            logger.error(f"Failed to build toolkit context: {e}")
            return {
                "available": False,
                "toolkits": [],
                "capabilities": [],
                "error": str(e)
            }