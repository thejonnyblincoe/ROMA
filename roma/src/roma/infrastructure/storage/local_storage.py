"""
Local Storage Implementation - ROMA v2.0 Goofys-Based Pattern.

Implements storage operations using local filesystem paths that are
transparently synced to remote storage via goofys mounting.
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

from .storage_interface import StorageInterface, StorageConfig

logger = logging.getLogger(__name__)


class LocalFileStorage(StorageInterface):
    """
    Local filesystem storage implementation for goofys-mounted directories.
    
    All operations work with local filesystem paths. When the mount point
    is managed by goofys, changes are automatically synced to remote storage.
    """
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        
        # Ensure mount point exists
        self.mount_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories if enabled
        if config.create_subdirs:
            (self.mount_path / config.artifacts_subdir).mkdir(parents=True, exist_ok=True)
            (self.mount_path / config.temp_subdir).mkdir(parents=True, exist_ok=True)
    
    async def put(
        self,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Store data at key path."""
        if len(data) > self.config.max_file_size:
            raise ValueError(f"File size {len(data)} exceeds maximum {self.config.max_file_size}")
        
        full_path = self.get_full_path(key)
        
        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write data atomically (write to temp file, then move)
        temp_path = full_path.with_suffix(full_path.suffix + ".tmp")
        
        try:
            # Use asyncio thread pool for file I/O
            await asyncio.get_event_loop().run_in_executor(
                None, temp_path.write_bytes, data
            )
            
            # Atomic move
            await asyncio.get_event_loop().run_in_executor(
                None, temp_path.rename, full_path
            )
            
            # Store metadata as extended attributes (if supported)
            if metadata:
                await self._store_metadata(full_path, metadata)
            
            self.logger.debug(f"Stored {len(data)} bytes at {key}")
            return str(full_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            
            self.logger.error(f"Failed to store data at {key}: {e}")
            raise
    
    async def get(self, key: str) -> Optional[bytes]:
        """Retrieve data by key."""
        full_path = self.get_full_path(key)
        
        if not full_path.exists():
            return None
        
        try:
            # Use asyncio thread pool for file I/O
            data = await asyncio.get_event_loop().run_in_executor(
                None, full_path.read_bytes
            )
            
            self.logger.debug(f"Retrieved {len(data)} bytes from {key}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve data from {key}: {e}")
            return None
    
    async def exists(self, key: str) -> bool:
        """Check if file exists at key path."""
        full_path = self.get_full_path(key)
        return full_path.exists() and full_path.is_file()
    
    async def delete(self, key: str) -> bool:
        """Delete file at key path."""
        full_path = self.get_full_path(key)
        
        if not full_path.exists():
            return False
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, full_path.unlink
            )
            
            self.logger.debug(f"Deleted file at {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete file at {key}: {e}")
            return False
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix filter."""
        if prefix:
            search_path = self.get_full_path(prefix)
            # If prefix is a directory, search inside it
            if search_path.is_dir():
                pattern = "**/*"
                search_root = search_path
            else:
                # Search for files matching prefix pattern
                pattern = f"{search_path.name}*"
                search_root = search_path.parent
        else:
            pattern = "**/*"
            search_root = self.mount_path
        
        try:
            files = await asyncio.get_event_loop().run_in_executor(
                None, lambda: list(search_root.glob(pattern))
            )
            
            # Convert to relative keys and filter files only
            keys = []
            for file_path in files:
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.mount_path)
                    keys.append(str(relative_path))
            
            return sorted(keys)
            
        except Exception as e:
            self.logger.error(f"Failed to list keys with prefix '{prefix}': {e}")
            return []
    
    async def copy_local(self, source_path: str, key: str) -> str:
        """
        Copy local file to storage.
        
        Args:
            source_path: Path to local file
            key: Storage key for destination
            
        Returns:
            Full path to stored file
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        dest_path = self.get_full_path(key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.copy2, str(source), str(dest_path)
            )
            
            self.logger.debug(f"Copied {source_path} to {key}")
            return str(dest_path)
            
        except Exception as e:
            self.logger.error(f"Failed to copy {source_path} to {key}: {e}")
            raise
    
    async def move_local(self, source_path: str, key: str) -> str:
        """
        Move local file to storage.
        
        Args:
            source_path: Path to local file
            key: Storage key for destination
            
        Returns:
            Full path to stored file
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        dest_path = self.get_full_path(key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.move, str(source), str(dest_path)
            )
            
            self.logger.debug(f"Moved {source_path} to {key}")
            return str(dest_path)
            
        except Exception as e:
            self.logger.error(f"Failed to move {source_path} to {key}: {e}")
            raise
    
    async def cleanup_temp_files(self, older_than_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified hours.
        
        Args:
            older_than_hours: Remove files older than this many hours
            
        Returns:
            Number of files cleaned up
        """
        temp_path = self.mount_path / self.config.temp_subdir
        if not temp_path.exists():
            return 0
        
        import time
        cutoff_time = time.time() - (older_than_hours * 3600)
        cleaned_count = 0
        
        try:
            for file_path in temp_path.rglob("*"):
                if file_path.is_file():
                    file_mtime = file_path.stat().st_mtime
                    if file_mtime < cutoff_time:
                        await asyncio.get_event_loop().run_in_executor(
                            None, file_path.unlink
                        )
                        cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} temporary files")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to clean up temp files: {e}")
            return 0
    
    async def _store_metadata(self, file_path: Path, metadata: Dict[str, str]) -> None:
        """
        Store metadata as extended attributes (platform-dependent).
        
        This is a basic implementation - extended attributes support
        varies by filesystem and platform.
        """
        try:
            # Simple implementation: store as .metadata file
            metadata_path = file_path.with_suffix(file_path.suffix + ".metadata")
            
            import json
            metadata_content = json.dumps(metadata, indent=2)
            
            await asyncio.get_event_loop().run_in_executor(
                None, metadata_path.write_text, metadata_content
            )
            
        except Exception as e:
            # Metadata storage is optional - don't fail the main operation
            self.logger.warning(f"Failed to store metadata for {file_path}: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage usage information."""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.mount_path.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith(".metadata"):
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                "mount_path": str(self.mount_path),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count,
                "artifacts_path": str(self.mount_path / self.config.artifacts_subdir),
                "temp_path": str(self.mount_path / self.config.temp_subdir),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage info: {e}")
            return {"error": str(e)}