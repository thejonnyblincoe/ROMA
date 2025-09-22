"""
Audio Artifact Entity - ROMA v2.0 Multimodal Context.

Simple audio artifact that wraps MediaFile for audio content.
"""

from typing import Dict, Any, Optional

from roma.domain.value_objects.media_type import MediaType
from roma.domain.entities.media_file import MediaFile
from .base_artifact import BaseArtifact


class AudioArtifact(BaseArtifact):
    """
    Audio content artifact.

    Simple wrapper around MediaFile for audio files.
    MediaFile already handles all audio formats and sources.
    """

    # Audio-specific field
    media_file: MediaFile
    duration_seconds: Optional[float] = None

    @property
    def media_type(self) -> MediaType:
        """Get the media type for audio artifacts."""
        return MediaType.AUDIO


    @classmethod
    def from_path(
        cls,
        name: str,
        file_path: str,
        task_id: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AudioArtifact":
        """Create AudioArtifact from local file path."""
        media_file = MediaFile.from_filepath(file_path, name=name)
        return cls(
            name=name,
            media_file=media_file,
            duration_seconds=duration_seconds,
            task_id=task_id,
            metadata=metadata or {}
        )

    @classmethod
    def from_url(
        cls,
        name: str,
        file_url: str,
        task_id: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AudioArtifact":
        """Create AudioArtifact from URL."""
        media_file = MediaFile.from_url(file_url, name=name)
        return cls(
            name=name,
            media_file=media_file,
            duration_seconds=duration_seconds,
            task_id=task_id,
            metadata=metadata or {}
        )

    @classmethod
    def from_bytes(
        cls,
        name: str,
        content: bytes,
        format: Optional[str] = None,
        task_id: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AudioArtifact":
        """Create AudioArtifact from raw bytes."""
        media_file = MediaFile.from_bytes(content, name=name, format=format)
        return cls(
            name=name,
            media_file=media_file,
            duration_seconds=duration_seconds,
            task_id=task_id,
            metadata=metadata or {}
        )

    # Required abstract method implementations
    async def get_content(self) -> Optional[bytes]:
        """Get the raw audio content as bytes."""
        return await self.media_file.get_content_bytes()

    def get_content_summary(self) -> str:
        """Get a summary of the audio content."""
        duration_str = f" ({self.duration_seconds}s)" if self.duration_seconds else ""
        return f"Audio: {self.media_file.get_content_summary()}{duration_str}"

    def get_size_bytes(self) -> Optional[int]:
        """Get audio file size in bytes."""
        return self.media_file.size

    def is_accessible(self) -> bool:
        """Check if the audio content is accessible."""
        return self.media_file.is_accessible()

    def get_mime_type(self) -> Optional[str]:
        """Get MIME type of the audio file."""
        return self.media_file.format

    def get_file_extension(self) -> Optional[str]:
        """Get file extension from the audio file."""
        if self.media_file.name:
            name_parts = self.media_file.name.split('.')
            if len(name_parts) > 1:
                return f".{name_parts[-1]}"
        return None