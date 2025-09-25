"""
Knowledge Record Value Object - ROMA v2.0 Domain Layer.

Thread-safe immutable value object for storing task execution knowledge.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from roma.domain.value_objects.result_envelope import AnyResultEnvelope
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


class KnowledgeRecord(BaseModel):
    """
    Thread-safe immutable knowledge record value object.

    All operations are thread-safe due to immutability.
    Updates create new instances rather than mutating existing ones.
    """

    model_config = ConfigDict(frozen=True)

    # Core identification
    task_id: str = Field(..., description="Unique task identifier")
    goal: str = Field(..., min_length=1, description="Task objective")
    task_type: TaskType = Field(..., description="Type of task")
    status: TaskStatus = Field(..., description="Task status")

    # Artifacts and Results
    artifacts: list[str] = Field(default_factory=list, description="Artifact storage keys")
    result: AnyResultEnvelope | None = Field(None, description="Task execution result envelope")

    # Relationships
    parent_task_id: str | None = Field(None, description="Parent task ID")
    child_task_ids: list[str] = Field(default_factory=list, description="Child task IDs")

    # Timestamps
    created_at: datetime = Field(..., description="Creation time")
    updated_at: datetime = Field(..., description="Last update time")
    completed_at: datetime | None = Field(None, description="Completion time")

    # Version tracking for optimistic concurrency
    version: int = Field(default=0, description="Record version for updates")

    @classmethod
    def create(
        cls,
        task_id: str,
        goal: str,
        task_type: TaskType,
        status: TaskStatus,
        artifacts: list[str] | None = None,
        parent_task_id: str | None = None,
        result: AnyResultEnvelope | None = None,
    ) -> "KnowledgeRecord":
        """Create new KnowledgeRecord (immutable value objects are inherently thread-safe)."""
        now = datetime.now(UTC)

        return cls(
            task_id=task_id,
            goal=goal,
            task_type=task_type,
            status=status,
            artifacts=list(artifacts) if artifacts else [],  # Copy to ensure immutability
            result=result,
            parent_task_id=parent_task_id,
            child_task_ids=[],
            created_at=now,
            updated_at=now,
            completed_at=now if status == TaskStatus.COMPLETED else None,
            version=0,
        )

    def update_status(self, new_status: TaskStatus) -> "KnowledgeRecord":
        """Update status - returns new instance."""
        now = datetime.now(UTC)
        return self.model_copy(
            update={
                "status": new_status,
                "updated_at": now,
                "completed_at": now if new_status == TaskStatus.COMPLETED else self.completed_at,
                "version": self.version + 1,
            }
        )

    def add_child(self, child_id: str) -> "KnowledgeRecord":
        """Add child - returns new instance."""
        if child_id in self.child_task_ids:
            return self  # No change needed

        # Create new list to ensure immutability
        updated_children = list(self.child_task_ids)
        updated_children.append(child_id)

        return self.model_copy(
            update={
                "child_task_ids": updated_children,
                "updated_at": datetime.now(UTC),
                "version": self.version + 1,
            }
        )

    def add_artifacts(self, artifact_keys: list[str]) -> "KnowledgeRecord":
        """Add artifacts - returns new instance."""
        if not artifact_keys:
            return self  # No change needed

        # Create new list to ensure immutability
        updated_artifacts = list(self.artifacts)
        changes_made = False

        for key in artifact_keys:
            if key not in updated_artifacts:
                updated_artifacts.append(key)
                changes_made = True

        # Only create new instance if changes were actually made
        if not changes_made:
            return self  # No new artifacts were added

        return self.model_copy(
            update={
                "artifacts": updated_artifacts,
                "updated_at": datetime.now(UTC),
                "version": self.version + 1,
            }
        )

    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == TaskStatus.COMPLETED

    def has_artifacts(self) -> bool:
        """Check if record has artifacts."""
        return len(self.artifacts) > 0

    def has_children(self) -> bool:
        """Check if record has children."""
        return len(self.child_task_ids) > 0

    def has_result(self) -> bool:
        """Check if record has execution result."""
        return self.result is not None

    def extract_content(self) -> str | None:
        """
        Extract primary content from result envelope.

        Returns:
            Primary output string or None if no result available
        """
        if not self.result:
            return None

        # Use ResultEnvelope's extract_primary_output method if available
        if hasattr(self.result, "extract_primary_output"):
            try:
                return self.result.extract_primary_output()
            except Exception:
                # Fallback to string conversion
                return str(self.result)

        # Fallback to string conversion
        return str(self.result)

    def get_summary(self, max_length: int = 200) -> str:
        """
        Get content summary for context building.

        Args:
            max_length: Maximum length of summary

        Returns:
            Truncated content or status message
        """
        content = self.extract_content()

        if not content:
            return f"Task {self.task_id}: No result available"

        if len(content) <= max_length:
            return content

        # Find word boundary for clean truncation
        truncated = content[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:  # Only use word boundary if it's not too short
            truncated = truncated[:last_space]

        return truncated + "..."

    def to_context_dict(self) -> dict[str, Any]:
        """
        Convert to standardized context dictionary for context building.

        Returns:
            Dictionary with all context-relevant data
        """
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "content": self.extract_content(),
            "summary": self.get_summary(),
            "has_artifacts": self.has_artifacts(),
            "artifact_count": len(self.artifacts),
            "has_children": self.has_children(),
            "child_count": len(self.child_task_ids),
            "has_result": self.has_result(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
