"""
Task Context Domain Objects.

Core context structures for task execution that belong in the domain layer.
"""

import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from roma.domain.entities.artifacts.base_artifact import BaseArtifact
from roma.domain.value_objects.context_item_type import ContextItemType

logger = logging.getLogger(__name__)


class ContextConfig(BaseModel):
    """Configuration for context building limits and options."""

    model_config = ConfigDict(frozen=True)

    # Result limits
    max_parent_results: int = Field(default=10, ge=1, le=50)
    max_sibling_results: int = Field(default=10, ge=1, le=50)
    max_child_results: int = Field(default=20, ge=1, le=100)

    # Content limits
    max_text_content: int = Field(default=10, ge=1, le=30)
    max_artifacts: int = Field(default=10, ge=1, le=20)

    # Text length limits
    max_text_length: int = Field(default=500, ge=100, le=2000)
    max_result_length: int = Field(default=1000, ge=200, le=5000)
    max_summary_length: int = Field(default=200, ge=50, le=500)
    max_outcome_summary_length: int = Field(default=150, ge=50, le=300)

    # Context prioritization and overflow limits
    enable_context_prioritization: bool = Field(default=True)
    enable_context_validation: bool = Field(
        default=True, description="Enable agent-aware context validation"
    )
    max_total_context_tokens: int = Field(default=8000, ge=1000, le=32000)
    max_parent_items: int = Field(default=5, ge=1, le=20)
    max_sibling_items: int = Field(default=8, ge=1, le=30)
    priority_high_threshold: int = Field(default=8, ge=1, le=10)
    priority_medium_threshold: int = Field(default=5, ge=1, le=10)


class ContextItem(BaseModel):
    """Single context item with content and metadata."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(default_factory=lambda: str(uuid4()))
    item_type: ContextItemType
    content: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0

    @classmethod
    def from_text(
        cls,
        content: str,
        item_type: ContextItemType,
        metadata: dict[str, Any] | None = None,
        priority: int = 0,
    ) -> "ContextItem":
        """Create context item from text content."""
        return cls(item_type=item_type, content=content, metadata=metadata or {}, priority=priority)

    @classmethod
    def from_artifact(
        cls, artifact: BaseArtifact, item_type: ContextItemType, priority: int = 0
    ) -> "ContextItem":
        """Create context item from artifact."""
        return cls(
            item_type=item_type, content=artifact, metadata=artifact.metadata, priority=priority
        )


class TaskContext(BaseModel):
    """Complete context assembled for agent execution."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, from_attributes=True)

    # Core task information
    task: Any  # TaskNode - using Any to avoid circular import
    overall_objective: str
    execution_id: str = Field(..., description="Execution ID for session isolation")

    # Context items (ordered by priority)
    context_items: list[ContextItem] = Field(default_factory=list)

    # System metadata
    execution_metadata: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    user_preferences: dict[str, Any] = Field(default_factory=dict)

    def get_text_content(self) -> list[str]:
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
            ContextItemType.REFERENCE_TEXT,
        }
        text_items = [
            item.content
            for item in self.context_items
            if item.item_type in text_types and isinstance(item.content, str)
        ]
        return text_items

    def get_file_artifacts(self) -> list[BaseArtifact]:
        """Extract all file artifacts from context."""
        artifact_types = {
            ContextItemType.IMAGE_ARTIFACT,
            ContextItemType.AUDIO_ARTIFACT,
            ContextItemType.VIDEO_ARTIFACT,
            ContextItemType.FILE_ARTIFACT,
        }
        file_items = []
        for item in self.context_items:
            if item.item_type in artifact_types and isinstance(item.content, BaseArtifact):
                file_items.append(item.content)
        return file_items

    def get_by_item_type(self, item_type: ContextItemType) -> list[ContextItem]:
        """Get context items by item type."""
        return [item for item in self.context_items if item.item_type == item_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task": self.task.to_dict(),
            "overall_objective": self.overall_objective,
            "execution_id": self.execution_id,
            "context_items": [
                {
                    "item_id": item.item_id,
                    "item_type": item.item_type.value,
                    "content": (
                        item.content.to_dict()
                        if isinstance(item.content, BaseArtifact)
                        else str(item.content)
                    ),
                    "metadata": item.metadata,
                    "priority": item.priority,
                }
                for item in self.context_items
            ],
            "execution_metadata": self.execution_metadata,
            "constraints": self.constraints,
            "user_preferences": self.user_preferences,
            "text_count": len(self.get_text_content()),
            "file_count": len(self.get_file_artifacts()),
        }
