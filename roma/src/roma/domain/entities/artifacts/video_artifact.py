"""
Video Artifact Entity - ROMA v2.0 Multimodal Context.

Handles video content following Agno media patterns with support for various
video formats, content sources (URL, filepath, bytes), and validation.
"""

import base64
import logging
from pathlib import Path
from typing import Any

import httpx
from pydantic import ConfigDict, field_validator, model_validator

from roma.domain.entities.artifacts.base_artifact import BaseArtifact
from roma.domain.value_objects.media_type import MediaType

logger = logging.getLogger(__name__)


class VideoArtifact(BaseArtifact):
    """
    Video artifact following Agno media pattern.

    Supports multiple content sources:
    - URL: Remote video URL
    - filepath: Local file path
    - content: Raw video bytes

    Features:
    - Automatic format detection
    - Base64 encoding/decoding
    - MIME type validation
    - Content validation and accessibility checks
    """

    model_config = ConfigDict(frozen=True)

    # Video-specific fields following Agno pattern
    url: str | None = None
    filepath: str | Path | None = None
    content: bytes | None = None

    # Video metadata
    format: str | None = None  # e.g., "MP4", "AVI", "MOV"
    mime_type: str = "video/mp4"
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    bitrate: int | None = None

    # Validation metadata
    _content_loaded: bool = False

    @model_validator(mode="after")
    def validate_content_sources(self) -> "VideoArtifact":
        """Validate exactly one content source is provided (Agno pattern)."""
        sources = [self.url, self.filepath, self.content]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one content source must be provided: url, filepath, or content"
            )

        return self

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """Validate MIME type is video-related."""
        valid_mime_types = [
            "video/mp4",
            "video/avi",
            "video/mov",
            "video/wmv",
            "video/flv",
            "video/webm",
            "video/mkv",
            "video/m4v",
        ]

        if v not in valid_mime_types:
            raise ValueError(f"Invalid video MIME type: {v}. Must be one of {valid_mime_types}")

        return v

    @field_validator("filepath")
    @classmethod
    def validate_filepath(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None

        path = Path(v) if isinstance(v, str) else v

        # Check file extension for common video formats
        if path.suffix.lower() not in [
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".mkv",
            ".m4v",
        ]:
            logger.warning(f"File extension {path.suffix} may not be a supported video format")

        return path

    @property
    def media_type(self) -> MediaType:
        """Get the media type for this artifact."""
        return MediaType.VIDEO

    async def get_content(self) -> bytes | None:
        """
        Get the raw video content bytes (Agno pattern).

        Returns:
            Video content as bytes or None if not accessible
        """
        try:
            if self.content:
                return self.content
            elif self.filepath:
                return await self._load_from_file()
            elif self.url:
                return await self._load_from_url()
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to load video content: {e}")
            return None

    async def _load_from_file(self) -> bytes | None:
        """Load video content from file."""
        try:
            if self.filepath is None:
                return None

            # Convert to Path if it's a string
            path_obj = Path(self.filepath) if isinstance(self.filepath, str) else self.filepath

            if not path_obj.exists():
                logger.error(f"Video file not found: {path_obj}")
                return None

            return path_obj.read_bytes()
        except Exception as e:
            logger.error(f"Error reading video file {self.filepath}: {e}")
            return None

    async def _load_from_url(self) -> bytes | None:
        """Load video content from URL."""
        try:
            if self.url is None:
                return None

            async with httpx.AsyncClient() as client:
                response = await client.get(self.url)
                response.raise_for_status()

                # Validate content type
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("video/"):
                    logger.warning(f"URL content type {content_type} is not video")

                return response.content
        except Exception as e:
            logger.error(f"Error fetching video from URL {self.url}: {e}")
            return None

    def get_content_summary(self) -> str:
        """Get a human-readable summary of the video."""
        parts = [f"Video: {self.name}"]

        if self.format:
            parts.append(f"format={self.format}")

        if self.duration_seconds:
            minutes = int(self.duration_seconds // 60)
            seconds = int(self.duration_seconds % 60)
            parts.append(f"duration={minutes}:{seconds:02d}")

        if self.width and self.height:
            parts.append(f"resolution={self.width}x{self.height}")

        if self.fps:
            parts.append(f"fps={self.fps}")

        if self.bitrate:
            parts.append(f"bitrate={self.bitrate}bps")

        if self.url:
            parts.append("source=URL")
        elif self.filepath:
            path_obj = Path(self.filepath) if isinstance(self.filepath, str) else self.filepath
            parts.append(f"source=file({path_obj.name})")
        else:
            parts.append("source=bytes")

        return ", ".join(parts)

    def get_size_bytes(self) -> int | None:
        """Get video size in bytes."""
        if self.content:
            return len(self.content)
        elif self.filepath:
            path_obj = Path(self.filepath) if isinstance(self.filepath, str) else self.filepath
            if path_obj.exists():
                return path_obj.stat().st_size
            return None
        else:
            # Cannot determine size for URLs without fetching
            return None

    def is_accessible(self) -> bool:
        """Check if the video content is accessible."""
        if self.content:
            return True
        elif self.filepath:
            path_obj = Path(self.filepath) if isinstance(self.filepath, str) else self.filepath
            return path_obj.exists() and path_obj.is_file()
        else:
            # URL is accessible if it exists - would need async check to fully verify
            return self.url is not None

    def get_mime_type(self) -> str:
        """Get MIME type of the video."""
        return self.mime_type

    def get_file_extension(self) -> str | None:
        """Get file extension based on format."""
        if self.filepath:
            path_obj = Path(self.filepath) if isinstance(self.filepath, str) else self.filepath
            return path_obj.suffix.lower()
        elif self.format:
            format_to_ext = {
                "MP4": ".mp4",
                "MPEG4": ".mp4",
                "AVI": ".avi",
                "MOV": ".mov",
                "WMV": ".wmv",
                "FLV": ".flv",
                "WEBM": ".webm",
                "MKV": ".mkv",
                "M4V": ".m4v",
            }
            return format_to_ext.get(self.format.upper())
        else:
            # Try to infer from MIME type
            mime_to_ext = {
                "video/mp4": ".mp4",
                "video/avi": ".avi",
                "video/mov": ".mov",
                "video/wmv": ".wmv",
                "video/flv": ".flv",
                "video/webm": ".webm",
                "video/mkv": ".mkv",
                "video/m4v": ".m4v",
            }
            return mime_to_ext.get(self.mime_type)

    async def to_base64(self, include_data_url: bool = False) -> str | None:
        """
        Convert video content to base64 string (Agno pattern).

        Args:
            include_data_url: Whether to include data URL prefix

        Returns:
            Base64 encoded video or None if content not accessible
        """
        content = await self.get_content()
        if not content:
            return None

        b64_str = base64.b64encode(content).decode("utf-8")

        if include_data_url:
            return f"data:{self.mime_type};base64,{b64_str}"
        else:
            return b64_str

    @classmethod
    def from_base64(
        cls, base64_str: str, name: str, mime_type: str = "video/mp4", **kwargs: Any
    ) -> "VideoArtifact":
        """
        Create VideoArtifact from base64 string (Agno pattern).

        Args:
            base64_str: Base64 encoded video data (with or without data URL prefix)
            name: Artifact name
            mime_type: Video MIME type
            **kwargs: Additional fields

        Returns:
            New VideoArtifact instance
        """
        # Handle data URL format
        if base64_str.startswith("data:"):
            # Extract MIME type and base64 data
            header, b64_data = base64_str.split(",", 1)
            if ";base64" in header:
                mime_type = header.split(";")[0].split(":")[1]
            base64_str = b64_data

        # Decode base64 to bytes
        try:
            content_bytes = base64.b64decode(base64_str)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}") from e

        return cls(name=name, content=content_bytes, mime_type=mime_type, **kwargs)

    @classmethod
    def from_url(
        cls, url: str, name: str, mime_type: str = "video/mp4", **kwargs: Any
    ) -> "VideoArtifact":
        """
        Create VideoArtifact from URL (Agno pattern).

        Args:
            url: Video URL
            name: Artifact name
            mime_type: Expected MIME type
            **kwargs: Additional fields

        Returns:
            New VideoArtifact instance
        """
        return cls(name=name, url=url, mime_type=mime_type, **kwargs)

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        name: str | None = None,
        mime_type: str | None = None,
        **kwargs: Any,
    ) -> "VideoArtifact":
        """
        Create VideoArtifact from file path (Agno pattern).

        Args:
            filepath: Path to video file
            name: Artifact name (defaults to filename)
            mime_type: Video MIME type (auto-detected if None)
            **kwargs: Additional fields

        Returns:
            New VideoArtifact instance
        """
        path = Path(filepath)

        if name is None:
            name = path.name

        if mime_type is None:
            # Auto-detect MIME type from extension
            ext_to_mime = {
                ".mp4": "video/mp4",
                ".avi": "video/avi",
                ".mov": "video/mov",
                ".wmv": "video/wmv",
                ".flv": "video/flv",
                ".webm": "video/webm",
                ".mkv": "video/mkv",
                ".m4v": "video/m4v",
            }
            mime_type = ext_to_mime.get(path.suffix.lower(), "video/mp4")

        return cls(name=name, filepath=path, mime_type=mime_type, **kwargs)

    def to_dict(self, include_content: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary for serialization (Agno pattern).

        Args:
            include_content: Whether to include base64 encoded content

        Returns:
            Dictionary representation
        """
        base_dict = super().to_dict()

        video_dict = {
            **base_dict,
            "url": self.url,
            "filepath": str(self.filepath) if self.filepath else None,
            "format": self.format,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "bitrate": self.bitrate,
        }

        if include_content and self.content:
            video_dict["content_base64"] = base64.b64encode(self.content).decode("utf-8")

        return video_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoArtifact":
        """
        Create VideoArtifact from dictionary.

        Args:
            data: Dictionary data

        Returns:
            New VideoArtifact instance
        """
        data_copy = cls.from_dict_base(data)

        # Handle content_base64 field - this takes priority
        if "content_base64" in data_copy:
            content_b64 = data_copy.pop("content_base64")
            data_copy["content"] = base64.b64decode(content_b64)
            # Clear other content sources to avoid validation error
            data_copy.pop("url", None)
            data_copy.pop("filepath", None)

        # Convert filepath string to Path
        if "filepath" in data_copy and data_copy["filepath"]:
            data_copy["filepath"] = Path(data_copy["filepath"])

        return cls(**data_copy)
