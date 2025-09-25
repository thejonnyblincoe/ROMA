"""
Image Artifact Entity - ROMA v2.0 Multimodal Context.

Handles image content following Agno media patterns with support for various
image formats, content sources (URL, filepath, bytes), and validation.
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


class ImageArtifact(BaseArtifact):
    """
    Image artifact following Agno media pattern.

    Supports multiple content sources:
    - URL: Remote image URL
    - filepath: Local file path
    - content: Raw image bytes

    Features:
    - Automatic format detection
    - Base64 encoding/decoding
    - MIME type validation
    - Content validation and accessibility checks
    """

    model_config = ConfigDict(frozen=True)

    # Image-specific fields following Agno pattern
    url: str | None = None
    filepath: str | Path | None = None
    content: bytes | None = None

    # Image metadata
    format: str | None = None  # e.g., "PNG", "JPEG", "GIF"
    mime_type: str = "image/png"
    width: int | None = None
    height: int | None = None

    # Validation metadata
    _content_loaded: bool = False

    @model_validator(mode="after")
    def validate_content_sources(self) -> "ImageArtifact":
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
        """Validate MIME type is image-related."""
        valid_mime_types = [
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/webp",
            "image/bmp",
            "image/tiff",
            "image/svg+xml",
        ]

        if v not in valid_mime_types:
            raise ValueError(f"Invalid image MIME type: {v}. Must be one of {valid_mime_types}")

        return v

    @field_validator("filepath")
    @classmethod
    def validate_filepath(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None

        path = Path(v) if isinstance(v, str) else v

        # Check file extension for common image formats
        if path.suffix.lower() not in [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".bmp",
            ".tiff",
            ".svg",
        ]:
            logger.warning(f"File extension {path.suffix} may not be a supported image format")

        return path

    @property
    def media_type(self) -> MediaType:
        """Get the media type for this artifact."""
        return MediaType.IMAGE

    async def get_content(self) -> bytes | None:
        """
        Get the raw image content bytes (Agno pattern).

        Returns:
            Image content as bytes or None if not accessible
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
            logger.error(f"Failed to load image content: {e}")
            return None

    async def _load_from_file(self) -> bytes | None:
        """Load image content from file."""
        try:
            if not self.filepath.exists():
                logger.error(f"Image file not found: {self.filepath}")
                return None

            return self.filepath.read_bytes()
        except Exception as e:
            logger.error(f"Error reading image file {self.filepath}: {e}")
            return None

    async def _load_from_url(self) -> bytes | None:
        """Load image content from URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.url)
                response.raise_for_status()

                # Validate content type
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    logger.warning(f"URL content type {content_type} is not an image")

                return response.content
        except Exception as e:
            logger.error(f"Error fetching image from URL {self.url}: {e}")
            return None

    def get_content_summary(self) -> str:
        """Get a human-readable summary of the image."""
        parts = [f"Image: {self.name}"]

        if self.format:
            parts.append(f"format={self.format}")

        if self.width and self.height:
            parts.append(f"size={self.width}x{self.height}")

        if self.url:
            parts.append("source=URL")
        elif self.filepath:
            parts.append(f"source=file({self.filepath.name})")
        else:
            parts.append("source=bytes")

        return ", ".join(parts)

    def get_size_bytes(self) -> int | None:
        """Get image size in bytes."""
        if self.content:
            return len(self.content)
        elif self.filepath and self.filepath.exists():
            return self.filepath.stat().st_size
        else:
            # Cannot determine size for URLs without fetching
            return None

    def is_accessible(self) -> bool:
        """Check if the image content is accessible."""
        if self.content:
            return True
        elif self.filepath:
            return self.filepath.exists() and self.filepath.is_file()
        else:
            # URL is accessible if it exists - would need async check to fully verify
            return self.url is not None

    def get_mime_type(self) -> str:
        """Get MIME type of the image."""
        return self.mime_type

    def get_file_extension(self) -> str | None:
        """Get file extension based on format."""
        if self.filepath:
            return self.filepath.suffix.lower()
        elif self.format:
            format_to_ext = {
                "PNG": ".png",
                "JPEG": ".jpg",
                "JPG": ".jpg",
                "GIF": ".gif",
                "WEBP": ".webp",
                "BMP": ".bmp",
                "TIFF": ".tiff",
                "SVG": ".svg",
            }
            return format_to_ext.get(self.format.upper())
        else:
            # Try to infer from MIME type
            mime_to_ext = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "image/bmp": ".bmp",
                "image/tiff": ".tiff",
                "image/svg+xml": ".svg",
            }
            return mime_to_ext.get(self.mime_type)

    async def to_base64(self, include_data_url: bool = False) -> str | None:
        """
        Convert image content to base64 string (Agno pattern).

        Args:
            include_data_url: Whether to include data URL prefix

        Returns:
            Base64 encoded image or None if content not accessible
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
        cls, base64_str: str, name: str, mime_type: str = "image/png", **kwargs: Any
    ) -> "ImageArtifact":
        """
        Create ImageArtifact from base64 string (Agno pattern).

        Args:
            base64_str: Base64 encoded image data (with or without data URL prefix)
            name: Artifact name
            mime_type: Image MIME type
            **kwargs: Additional fields

        Returns:
            New ImageArtifact instance
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
        cls, url: str, name: str, mime_type: str = "image/png", **kwargs: Any
    ) -> "ImageArtifact":
        """
        Create ImageArtifact from URL (Agno pattern).

        Args:
            url: Image URL
            name: Artifact name
            mime_type: Expected MIME type
            **kwargs: Additional fields

        Returns:
            New ImageArtifact instance
        """
        return cls(name=name, url=url, mime_type=mime_type, **kwargs)

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        name: str | None = None,
        mime_type: str | None = None,
        **kwargs: Any,
    ) -> "ImageArtifact":
        """
        Create ImageArtifact from file path (Agno pattern).

        Args:
            filepath: Path to image file
            name: Artifact name (defaults to filename)
            mime_type: Image MIME type (auto-detected if None)
            **kwargs: Additional fields

        Returns:
            New ImageArtifact instance
        """
        path = Path(filepath)

        if name is None:
            name = path.name

        if mime_type is None:
            # Auto-detect MIME type from extension
            ext_to_mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".tiff": "image/tiff",
                ".svg": "image/svg+xml",
            }
            mime_type = ext_to_mime.get(path.suffix.lower(), "image/png")

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

        image_dict = {
            **base_dict,
            "url": self.url,
            "filepath": str(self.filepath) if self.filepath else None,
            "format": self.format,
            "width": self.width,
            "height": self.height,
        }

        if include_content and self.content:
            image_dict["content_base64"] = base64.b64encode(self.content).decode("utf-8")

        return image_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageArtifact":
        """
        Create ImageArtifact from dictionary.

        Args:
            data: Dictionary data

        Returns:
            New ImageArtifact instance
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
