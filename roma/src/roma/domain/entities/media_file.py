"""
Media File Entity - ROMA v2.0 Framework-Agnostic Implementation.

Framework-independent file handling inspired by Agno's design patterns
but with no external dependencies. Supports URL, filepath, and raw content.
"""

import logging
import mimetypes
from base64 import b64decode, b64encode
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class MediaFile(BaseModel):
    """
    Framework-agnostic media file entity.

    Inspired by Agno's File class design but completely independent.
    Supports flexible file sources: URL, filepath, or raw bytes.
    """

    model_config = ConfigDict(frozen=True)

    # Core identifiers
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str | None = None

    # Content sources (exactly one must be provided)
    url: str | None = None
    filepath: str | None = None
    content: bytes | None = None

    # Metadata
    format: str | None = None  # MIME type
    size: int | None = None
    encoding: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context: Any) -> None:
        """Validate content sources and auto-detect properties."""
        # Validate exactly one content source
        content_sources = [self.url, self.filepath, self.content]
        provided_sources = [s for s in content_sources if s is not None]

        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one content source must be provided: url, filepath, or content"
            )

        # Auto-detect MIME type if not provided
        if not self.format:
            if self.filepath:
                mime_type, _ = mimetypes.guess_type(self.filepath)
                object.__setattr__(self, "format", mime_type)
            elif self.url:
                mime_type, _ = mimetypes.guess_type(self.url)
                object.__setattr__(self, "format", mime_type)

        # Auto-detect size for filepath
        if self.filepath and not self.size:
            try:
                path = Path(self.filepath)
                if path.exists():
                    object.__setattr__(self, "size", path.stat().st_size)
            except Exception as e:
                logger.warning(f"Could not get file size for {self.filepath}: {e}")

        # Auto-detect size for content
        if self.content and not self.size:
            object.__setattr__(self, "size", len(self.content))

        # Auto-generate name if not provided
        if not self.name:
            if self.filepath:
                object.__setattr__(self, "name", Path(self.filepath).name)
            elif self.url:
                object.__setattr__(self, "name", Path(self.url).name)
            else:
                object.__setattr__(self, "name", f"file_{self.id[:8]}")

    @field_validator("filepath")
    @classmethod
    def validate_filepath(cls, v: str | None) -> str | None:
        """Validate filepath format."""
        if v is not None:
            # Normalize path separators
            return str(Path(v))
        return v

    @classmethod
    def from_filepath(
        cls, filepath: str, name: str | None = None, metadata: dict[str, Any] | None = None
    ) -> "MediaFile":
        """
        Create MediaFile from local file path.

        Args:
            filepath: Path to local file
            name: Optional custom name
            metadata: Additional metadata

        Returns:
            MediaFile instance
        """
        return cls(filepath=filepath, name=name, metadata=metadata or {})

    @classmethod
    def from_url(
        cls, url: str, name: str | None = None, metadata: dict[str, Any] | None = None
    ) -> "MediaFile":
        """
        Create MediaFile from URL.

        Args:
            url: File URL
            name: Optional custom name
            metadata: Additional metadata

        Returns:
            MediaFile instance
        """
        return cls(url=url, name=name, metadata=metadata or {})

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        name: str,
        format: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MediaFile":
        """
        Create MediaFile from raw bytes.

        Args:
            content: Raw file content
            name: File name
            format: MIME type
            metadata: Additional metadata

        Returns:
            MediaFile instance
        """
        return cls(content=content, name=name, format=format, metadata=metadata or {})

    async def get_content_bytes(self) -> bytes | None:
        """
        Retrieve file content as bytes from any source.

        Returns:
            File content as bytes, None if retrieval fails
        """
        if self.content is not None:
            return self.content

        if self.filepath:
            try:
                path = Path(self.filepath)
                if path.exists() and path.is_file():
                    return path.read_bytes()
                else:
                    logger.error(f"File does not exist: {self.filepath}")
            except Exception as e:
                logger.error(f"Failed to read file {self.filepath}: {e}")

        if self.url:
            # URL content retrieval would be handled by HTTP client
            logger.warning(f"URL content retrieval not implemented: {self.url}")

        return None

    async def get_text_content(self, encoding: str | None = None) -> str | None:
        """
        Retrieve file content as text.

        Args:
            encoding: Text encoding (defaults to utf-8)

        Returns:
            File content as string, None if retrieval/decoding fails
        """
        content_bytes = await self.get_content_bytes()
        if content_bytes is None:
            return None

        encoding = encoding or self.encoding or "utf-8"

        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode content as {encoding}: {e}")
            return None

    def to_base64(self) -> str | None:
        """
        Convert file content to base64 string (synchronous version).

        Returns:
            Base64 encoded content, None if content unavailable
        """
        if self.content is not None:
            return b64encode(self.content).decode("utf-8")

        if self.filepath:
            try:
                path = Path(self.filepath)
                if path.exists():
                    content = path.read_bytes()
                    return b64encode(content).decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to encode file to base64 {self.filepath}: {e}")

        return None

    @classmethod
    def from_base64(
        cls,
        base64_content: str,
        name: str,
        format: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MediaFile":
        """
        Create MediaFile from base64 encoded content.

        Args:
            base64_content: Base64 encoded file content
            name: File name
            format: MIME type
            metadata: Additional metadata

        Returns:
            MediaFile instance
        """
        try:
            content = b64decode(base64_content)
            return cls.from_bytes(content, name, format, metadata)
        except Exception as e:
            raise ValueError(f"Invalid base64 content: {e}") from e

    def validate_and_normalize_content(self) -> bool:
        """
        Validate content accessibility and normalize properties.

        Returns:
            True if content is valid and accessible
        """
        if self.content is not None:
            return True

        if self.filepath:
            try:
                path = Path(self.filepath)
                return path.exists() and path.is_file()
            except Exception:
                return False

        if self.url:
            # URL validation would require network check
            # For now, assume valid if properly formatted
            return self.url.startswith(("http://", "https://", "ftp://"))

        return False

    def get_content_summary(self) -> str:
        """
        Get summary description of file content.

        Returns:
            Human-readable content summary
        """
        size_str = f" ({self.size} bytes)" if self.size else ""
        format_str = f" [{self.format}]" if self.format else ""

        if self.filepath:
            return f"File: {self.name}{size_str}{format_str} @ {self.filepath}"
        elif self.url:
            return f"File: {self.name}{size_str}{format_str} @ {self.url}"
        else:
            return f"File: {self.name}{size_str}{format_str} (in-memory)"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "filepath": self.filepath,
            "content": b64encode(self.content).decode("utf-8") if self.content else None,
            "format": self.format,
            "size": self.size,
            "encoding": self.encoding,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaFile":
        """Create MediaFile from dictionary representation."""
        data_copy = data.copy()

        # Handle base64 content
        if data_copy.get("content"):
            data_copy["content"] = b64decode(data_copy["content"])

        # Handle datetime
        if "created_at" in data_copy and isinstance(data_copy["created_at"], str):
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])

        return cls(**data_copy)

    def __str__(self) -> str:
        """String representation."""
        return f"MediaFile(name='{self.name}', format='{self.format}', size={self.size})"
