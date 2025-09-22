"""
File Artifact Entity - ROMA v2.0 Multimodal Context.

Specialized artifact for file content using MediaFile.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from roma.domain.value_objects.media_type import MediaType
from roma.domain.entities.media_file import MediaFile
from .base_artifact import BaseArtifact


class FileArtifact(BaseArtifact):
    """
    File content artifact.
    
    Wraps MediaFile with context metadata for agent use.
    Handles all file types: documents, data files, images, etc.
    """
    
    # File-specific field
    media_file: MediaFile
    
    @property
    def media_type(self) -> MediaType:
        """Get the media type for file artifacts."""
        return MediaType.FILE
    
    def model_post_init(self, __context: Any) -> None:
        """Validate media file is provided."""
        if not self.media_file:
            raise ValueError("FileArtifact must have media_file")
    
    @classmethod
    def from_path(
        cls,
        name: str,
        file_path: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "FileArtifact":
        """Create FileArtifact from local file path."""
        media_file = MediaFile.from_filepath(file_path, name=name)
        return cls(
            name=name,
            media_file=media_file,
            task_id=task_id,
            metadata=metadata or {}
        )
    
    @classmethod
    def from_url(
        cls,
        name: str,
        file_url: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "FileArtifact":
        """Create FileArtifact from URL."""
        media_file = MediaFile.from_url(file_url, name=name)
        return cls(
            name=name,
            media_file=media_file,
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
        metadata: Optional[Dict[str, Any]] = None
    ) -> "FileArtifact":
        """Create FileArtifact from raw bytes."""
        media_file = MediaFile.from_bytes(content, name=name, format=format)
        return cls(
            name=name,
            media_file=media_file,
            task_id=task_id,
            metadata=metadata or {}
        )
    
    # Required abstract method implementations
    async def get_content(self) -> Optional[bytes]:
        """Get the raw file content as bytes."""
        return await self.media_file.get_content_bytes()
    
    def get_content_summary(self) -> str:
        """Get a summary of the file content."""
        return f"File: {self.media_file.get_content_summary()}"
    
    def get_size_bytes(self) -> Optional[int]:
        """Get file size in bytes."""
        return self.media_file.size
    
    def is_accessible(self) -> bool:
        """Check if the file content is accessible."""
        return self.media_file.validate_and_normalize_content()
    
    # Override base class methods for file-specific behavior
    def get_mime_type(self) -> Optional[str]:
        """Get MIME type of the file."""
        return self.media_file.format
    
    def get_file_extension(self) -> Optional[str]:
        """Get file extension."""
        if self.media_file.name:
            return Path(self.media_file.name).suffix.lower()
        return None
    
    # File-specific methods
    async def get_text_content(self, encoding: Optional[str] = None) -> Optional[str]:
        """Get file content as text (for text files)."""
        return await self.media_file.get_text_content(encoding)
    
    def is_text_file(self) -> bool:
        """Check if this is a text file based on MIME type."""
        mime_type = self.get_mime_type()
        if not mime_type:
            return False
        return mime_type.startswith('text/') or mime_type in [
            'application/json',
            'application/xml',
            'application/javascript'
        ]
    
    def is_image_file(self) -> bool:
        """Check if this is an image file."""
        mime_type = self.get_mime_type()
        return mime_type.startswith('image/') if mime_type else False
    
    def to_base64(self) -> Optional[str]:
        """Convert file content to base64 string."""
        return self.media_file.to_base64()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "media_file": self.media_file.to_dict(),
            "file_extension": self.get_file_extension(),
            "is_text_file": self.is_text_file(),
            "is_image_file": self.is_image_file(),
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileArtifact":
        """Create from dictionary."""
        data_copy = cls.from_dict_base(data)
        
        # Handle MediaFile
        if "media_file" in data_copy:
            data_copy["media_file"] = MediaFile.from_dict(data_copy["media_file"])
        
        # Remove file-specific computed fields
        computed_fields = ["file_extension", "is_text_file", "is_image_file"]
        for field in computed_fields:
            data_copy.pop(field, None)
        
        return cls(**data_copy)