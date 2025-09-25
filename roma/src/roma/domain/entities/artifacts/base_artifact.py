"""
Base Artifact Entity - ROMA v2.0 Multimodal Context.

Abstract base class for all context artifacts with common functionality.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from roma.domain.value_objects.context_item_type import ContextItemType
from roma.domain.value_objects.media_type import MediaType

logger = logging.getLogger(__name__)


class BaseArtifact(BaseModel, ABC):
    """
    Abstract base class for all context artifacts.

    Provides common functionality for identification, metadata, and serialization
    while allowing specialized implementations for different media types.
    """

    model_config = ConfigDict(frozen=True)

    # Core identifiers
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str | None = None
    name: str

    # Common metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate artifact name."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Artifact name cannot be empty")
        return v.strip()

    @property
    @abstractmethod
    def media_type(self) -> MediaType:
        """Get the media type for this artifact."""

    @abstractmethod
    async def get_content(self) -> Any | None:
        """Get the raw content of this artifact."""

    @abstractmethod
    def get_content_summary(self) -> str:
        """Get a human-readable summary of the artifact content."""

    @abstractmethod
    def get_size_bytes(self) -> int | None:
        """Get artifact size in bytes."""

    @abstractmethod
    def is_accessible(self) -> bool:
        """Check if the artifact content is accessible."""

    def get_mime_type(self) -> str | None:
        """Get MIME type of the artifact. Override in subclasses."""
        return None

    def get_file_extension(self) -> str | None:
        """Get file extension if applicable. Override in subclasses."""
        return None

    def get_context_item_type(self) -> ContextItemType:
        """
        Get appropriate ContextItemType for this artifact.

        Returns:
            ContextItemType based on the artifact's media type
        """
        media_type = self.media_type
        if media_type.value == "IMAGE":
            return ContextItemType.IMAGE_ARTIFACT
        elif media_type.value == "AUDIO":
            return ContextItemType.AUDIO_ARTIFACT
        elif media_type.value == "VIDEO":
            return ContextItemType.VIDEO_ARTIFACT
        elif media_type.value == "TEXT":
            return ContextItemType.REFERENCE_TEXT
        else:  # FILE or unknown
            return ContextItemType.FILE_ARTIFACT

    def is_text_content(self) -> bool:
        """Check if this artifact contains text content."""
        return self.media_type == MediaType.TEXT

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "name": self.name,
            "media_type": self.media_type.value,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.get_size_bytes(),
            "mime_type": self.get_mime_type(),
        }
        # Subclasses should extend this with their specific fields
        return base_dict

    @classmethod
    def from_dict_base(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Common dictionary processing for subclasses.

        Args:
            data: Dictionary data

        Returns:
            Processed data with common fields handled
        """
        from datetime import datetime

        data_copy = data.copy()

        # Handle datetime
        if "created_at" in data_copy and isinstance(data_copy["created_at"], str):
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])

        # Remove computed fields that are properties
        computed_fields = ["media_type", "size_bytes", "mime_type"]
        for field in computed_fields:
            data_copy.pop(field, None)

        return data_copy

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', type={self.media_type.value}, task_id='{self.task_id}')"
