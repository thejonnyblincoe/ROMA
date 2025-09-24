"""
Audio Artifact Entity - ROMA v2.0 Multimodal Context.

Handles audio content following Agno media patterns with support for various
audio formats, content sources (URL, filepath, bytes), and validation.
"""

import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from roma.domain.entities.artifacts.base_artifact import BaseArtifact
from roma.domain.value_objects.media_type import MediaType

logger = logging.getLogger(__name__)


class AudioArtifact(BaseArtifact):
    """
    Audio artifact following Agno media pattern.

    Supports multiple content sources:
    - URL: Remote audio URL
    - filepath: Local file path
    - content: Raw audio bytes

    Features:
    - Automatic format detection
    - Base64 encoding/decoding
    - MIME type validation
    - Content validation and accessibility checks
    """

    model_config = ConfigDict(frozen=True)

    # Audio-specific fields following Agno pattern
    url: Optional[str] = None
    filepath: Optional[Union[str, Path]] = None
    content: Optional[bytes] = None

    # Audio metadata
    format: Optional[str] = None  # e.g., "MP3", "WAV", "FLAC"
    mime_type: str = "audio/mpeg"
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    # Validation metadata
    _content_loaded: bool = False

    @model_validator(mode='after')
    def validate_content_sources(self) -> 'AudioArtifact':
        """Validate exactly one content source is provided (Agno pattern)."""
        sources = [self.url, self.filepath, self.content]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one content source must be provided: url, filepath, or content"
            )

        return self

    @field_validator('mime_type')
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """Validate MIME type is audio-related."""
        valid_mime_types = [
            'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/flac',
            'audio/ogg', 'audio/aac', 'audio/m4a', 'audio/webm'
        ]

        if v not in valid_mime_types:
            raise ValueError(f"Invalid audio MIME type: {v}. Must be one of {valid_mime_types}")

        return v

    @field_validator('filepath')
    @classmethod
    def validate_filepath(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        """Convert string paths to Path objects."""
        if v is None:
            return None

        path = Path(v) if isinstance(v, str) else v

        # Check file extension for common audio formats
        if path.suffix.lower() not in ['.mp3', '.wav', '.flac', '.ogg', '.aac', '.m4a', '.webm']:
            logger.warning(f"File extension {path.suffix} may not be a supported audio format")

        return path

    @property
    def media_type(self) -> MediaType:
        """Get the media type for this artifact."""
        return MediaType.AUDIO

    async def get_content(self) -> Optional[bytes]:
        """
        Get the raw audio content bytes (Agno pattern).

        Returns:
            Audio content as bytes or None if not accessible
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
            logger.error(f"Failed to load audio content: {e}")
            return None

    async def _load_from_file(self) -> Optional[bytes]:
        """Load audio content from file."""
        try:
            if not self.filepath.exists():
                logger.error(f"Audio file not found: {self.filepath}")
                return None

            with open(self.filepath, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading audio file {self.filepath}: {e}")
            return None

    async def _load_from_url(self) -> Optional[bytes]:
        """Load audio content from URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.url)
                response.raise_for_status()

                # Validate content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('audio/'):
                    logger.warning(f"URL content type {content_type} is not audio")

                return response.content
        except Exception as e:
            logger.error(f"Error fetching audio from URL {self.url}: {e}")
            return None

    def get_content_summary(self) -> str:
        """Get a human-readable summary of the audio."""
        parts = [f"Audio: {self.name}"]

        if self.format:
            parts.append(f"format={self.format}")

        if self.duration_seconds:
            minutes = int(self.duration_seconds // 60)
            seconds = int(self.duration_seconds % 60)
            parts.append(f"duration={minutes}:{seconds:02d}")

        if self.sample_rate:
            parts.append(f"rate={self.sample_rate}Hz")

        if self.channels:
            channel_desc = "mono" if self.channels == 1 else f"{self.channels}ch"
            parts.append(f"{channel_desc}")

        if self.url:
            parts.append(f"source=URL")
        elif self.filepath:
            parts.append(f"source=file({self.filepath.name})")
        else:
            parts.append(f"source=bytes")

        return ", ".join(parts)

    def get_size_bytes(self) -> Optional[int]:
        """Get audio size in bytes."""
        if self.content:
            return len(self.content)
        elif self.filepath and self.filepath.exists():
            return self.filepath.stat().st_size
        else:
            # Cannot determine size for URLs without fetching
            return None

    def is_accessible(self) -> bool:
        """Check if the audio content is accessible."""
        if self.content:
            return True
        elif self.filepath:
            return self.filepath.exists() and self.filepath.is_file()
        elif self.url:
            # Assume URL is accessible - would need async check to verify
            return True
        else:
            return False

    def get_mime_type(self) -> str:
        """Get MIME type of the audio."""
        return self.mime_type

    def get_file_extension(self) -> Optional[str]:
        """Get file extension based on format."""
        if self.filepath:
            return self.filepath.suffix.lower()
        elif self.format:
            format_to_ext = {
                'MP3': '.mp3',
                'MPEG': '.mp3',
                'WAV': '.wav',
                'WAVE': '.wav',
                'FLAC': '.flac',
                'OGG': '.ogg',
                'AAC': '.aac',
                'M4A': '.m4a',
                'WEBM': '.webm'
            }
            return format_to_ext.get(self.format.upper())
        else:
            # Try to infer from MIME type
            mime_to_ext = {
                'audio/mpeg': '.mp3',
                'audio/mp3': '.mp3',
                'audio/wav': '.wav',
                'audio/flac': '.flac',
                'audio/ogg': '.ogg',
                'audio/aac': '.aac',
                'audio/m4a': '.m4a',
                'audio/webm': '.webm'
            }
            return mime_to_ext.get(self.mime_type)

    async def to_base64(self, include_data_url: bool = False) -> Optional[str]:
        """
        Convert audio content to base64 string (Agno pattern).

        Args:
            include_data_url: Whether to include data URL prefix

        Returns:
            Base64 encoded audio or None if content not accessible
        """
        content = await self.get_content()
        if not content:
            return None

        b64_str = base64.b64encode(content).decode('utf-8')

        if include_data_url:
            return f"data:{self.mime_type};base64,{b64_str}"
        else:
            return b64_str

    @classmethod
    def from_base64(
        cls,
        base64_str: str,
        name: str,
        mime_type: str = "audio/mpeg",
        **kwargs: Any
    ) -> 'AudioArtifact':
        """
        Create AudioArtifact from base64 string (Agno pattern).

        Args:
            base64_str: Base64 encoded audio data (with or without data URL prefix)
            name: Artifact name
            mime_type: Audio MIME type
            **kwargs: Additional fields

        Returns:
            New AudioArtifact instance
        """
        # Handle data URL format
        if base64_str.startswith('data:'):
            # Extract MIME type and base64 data
            header, b64_data = base64_str.split(',', 1)
            if ';base64' in header:
                mime_type = header.split(';')[0].split(':')[1]
            base64_str = b64_data

        # Decode base64 to bytes
        try:
            content_bytes = base64.b64decode(base64_str)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")

        return cls(
            name=name,
            content=content_bytes,
            mime_type=mime_type,
            **kwargs
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        name: str,
        mime_type: str = "audio/mpeg",
        **kwargs: Any
    ) -> 'AudioArtifact':
        """
        Create AudioArtifact from URL (Agno pattern).

        Args:
            url: Audio URL
            name: Artifact name
            mime_type: Expected MIME type
            **kwargs: Additional fields

        Returns:
            New AudioArtifact instance
        """
        return cls(
            name=name,
            url=url,
            mime_type=mime_type,
            **kwargs
        )

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        name: Optional[str] = None,
        mime_type: Optional[str] = None,
        **kwargs: Any
    ) -> 'AudioArtifact':
        """
        Create AudioArtifact from file path (Agno pattern).

        Args:
            filepath: Path to audio file
            name: Artifact name (defaults to filename)
            mime_type: Audio MIME type (auto-detected if None)
            **kwargs: Additional fields

        Returns:
            New AudioArtifact instance
        """
        path = Path(filepath)

        if name is None:
            name = path.name

        if mime_type is None:
            # Auto-detect MIME type from extension
            ext_to_mime = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.flac': 'audio/flac',
                '.ogg': 'audio/ogg',
                '.aac': 'audio/aac',
                '.m4a': 'audio/m4a',
                '.webm': 'audio/webm'
            }
            mime_type = ext_to_mime.get(path.suffix.lower(), 'audio/mpeg')

        return cls(
            name=name,
            filepath=path,
            mime_type=mime_type,
            **kwargs
        )

    def to_dict(self, include_content: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization (Agno pattern).

        Args:
            include_content: Whether to include base64 encoded content

        Returns:
            Dictionary representation
        """
        base_dict = super().to_dict()

        audio_dict = {
            **base_dict,
            "url": self.url,
            "filepath": str(self.filepath) if self.filepath else None,
            "format": self.format,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }

        if include_content and self.content:
            audio_dict["content_base64"] = base64.b64encode(self.content).decode('utf-8')

        return audio_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioArtifact':
        """
        Create AudioArtifact from dictionary.

        Args:
            data: Dictionary data

        Returns:
            New AudioArtifact instance
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