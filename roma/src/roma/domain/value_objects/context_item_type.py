"""
Context Item Type Value Object - ROMA v2.0 Domain Layer.

Defines types of context items for categorization and prioritization.
"""

from enum import Enum


class ContextItemType(Enum):
    """Types of context items for better categorization."""

    # Core task information
    TASK_GOAL = "task_goal"
    OVERALL_OBJECTIVE = "overall_objective"

    # System context
    TEMPORAL = "temporal"
    TOOLKITS = "toolkits"

    # Results and prior work
    PARENT_RESULT = "parent_result"
    SIBLING_RESULT = "sibling_result"
    CHILD_RESULT = "child_result"  # For aggregators
    PRIOR_WORK = "prior_work"

    # Content and artifacts
    REFERENCE_TEXT = "reference_text"
    IMAGE_ARTIFACT = "image_artifact"
    AUDIO_ARTIFACT = "audio_artifact"
    VIDEO_ARTIFACT = "video_artifact"
    FILE_ARTIFACT = "file_artifact"

    # Additional content
    CONSTRAINT = "constraint"
    USER_PREFERENCE = "user_preference"
    EXECUTION_METADATA = "execution_metadata"
