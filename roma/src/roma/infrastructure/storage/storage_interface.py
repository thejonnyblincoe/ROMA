"""
Storage Interface - ROMA v2.0 Goofys-Based Storage Pattern.

Simple storage abstraction that uses goofys-mounted local directories
for transparent remote storage access. All operations are local filesystem
operations that are automatically synced to remote storage via goofys.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class StorageConfig(BaseModel):
    """Configuration for goofys-based storage."""
    
    model_config = ConfigDict(frozen=True)
    
    # Local mount point (goofys mounts remote storage here)
    mount_path: str = Field(description="Local path where goofys mounts remote storage")
    
    # Performance settings
    max_file_size: int = Field(default=100 * 1024 * 1024, description="Max file size in bytes (100MB)")
    create_subdirs: bool = Field(default=True, description="Auto-create subdirectories")
    
    # File organization
    artifacts_subdir: str = Field(default="artifacts", description="Subdirectory for artifacts")
    temp_subdir: str = Field(default="temp", description="Subdirectory for temporary files")
    
    @classmethod
    def from_mount_path(cls, mount_path: str) -> "StorageConfig":
        """Create storage config from goofys mount path."""
        return cls(mount_path=mount_path)


class StorageInterface(ABC):
    """
    Abstract interface for goofys-based storage operations.
    
    All operations work with local filesystem paths that are transparently
    synced to remote storage via goofys mounting.
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = logger
        self.mount_path = Path(config.mount_path)
    
    @abstractmethod
    async def put(
        self,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Store data at key path (relative to mount point).
        
        Args:
            key: Storage key/path relative to mount
            data: Raw data to store
            metadata: Custom metadata (stored as extended attributes)
            
        Returns:
            Full local path to stored file
        """
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """
        Retrieve data by key.
        
        Args:
            key: Storage key/path relative to mount
            
        Returns:
            Raw data, or None if not found
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if file exists at key path.
        
        Args:
            key: Storage key/path relative to mount
            
        Returns:
            True if file exists
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete file at key path.
        
        Args:
            key: Storage key/path relative to mount
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str = "") -> List[str]:
        """
        List all keys with optional prefix filter.
        
        Args:
            prefix: Key prefix to filter by
            
        Returns:
            List of matching keys (relative to mount)
        """
        pass
    
    def get_full_path(self, key: str) -> Path:
        """
        Get full local path for key.
        
        Args:
            key: Storage key relative to mount
            
        Returns:
            Full local filesystem path
        """
        normalized_key = self.normalize_key(key)
        return self.mount_path / normalized_key
    
    def get_artifacts_path(self, key: str) -> Path:
        """
        Get path in artifacts subdirectory.
        
        Args:
            key: Artifact key
            
        Returns:
            Full path in artifacts subdirectory
        """
        return self.get_full_path(f"{self.config.artifacts_subdir}/{key}")
    
    def get_temp_path(self, key: str) -> Path:
        """
        Get path in temp subdirectory.
        
        Args:
            key: Temp file key
            
        Returns:
            Full path in temp subdirectory
        """
        return self.get_full_path(f"{self.config.temp_subdir}/{key}")
    
    def generate_key(self, prefix: str = "", suffix: str = "") -> str:
        """
        Generate unique storage key.
        
        Args:
            prefix: Key prefix
            suffix: Key suffix (e.g., file extension)
            
        Returns:
            Unique storage key
        """
        unique_id = str(uuid4())
        return f"{prefix}{unique_id}{suffix}"
    
    def normalize_key(self, key: str) -> str:
        """
        Normalize storage key to consistent format.
        
        Args:
            key: Raw key
            
        Returns:
            Normalized key (no leading slashes, forward slashes only)
        """
        # Remove leading/trailing slashes, normalize path separators
        normalized = key.strip("/").replace("\\", "/")
        
        # Remove double slashes
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        
        return normalized
    
    async def put_text(
        self,
        key: str,
        text: str,
        encoding: str = "utf-8",
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Store text content.
        
        Args:
            key: Storage key/path
            text: Text content to store
            encoding: Text encoding
            metadata: Custom metadata
            
        Returns:
            Full local path to stored file
        """
        data = text.encode(encoding)
        return await self.put(key, data, metadata)
    
    async def get_text(self, key: str, encoding: str = "utf-8") -> Optional[str]:
        """
        Retrieve text content.
        
        Args:
            key: Storage key/path
            encoding: Text encoding
            
        Returns:
            Text content, or None if not found
        """
        data = await self.get(key)
        if data is None:
            return None
        
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as e:
            self.logger.error(f"Failed to decode text from {key}: {e}")
            return None
    
    async def get_size(self, key: str) -> Optional[int]:
        """
        Get file size in bytes.
        
        Args:
            key: Storage key/path
            
        Returns:
            Size in bytes, or None if not found
        """
        full_path = self.get_full_path(key)
        try:
            return full_path.stat().st_size
        except FileNotFoundError:
            return None