"""
Local Storage Implementation - ROMA v2.0 Goofys-Based Pattern.

Implements storage operations using local filesystem paths that are
transparently synced to remote storage via goofys mounting.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .storage_interface import StorageInterface, StorageConfig
from ..utils.async_file_utils import (
    async_makedirs, async_path_exists, async_is_file, async_is_dir,
    async_file_size, async_unlink, async_rename, async_copy2, async_move,
    async_glob, async_rglob, async_read_bytes, async_write_bytes, async_read_text,
    async_write_text, async_cleanup_old_files, async_calculate_directory_size
)

logger = logging.getLogger(__name__)


class LocalFileStorage(StorageInterface):
    """
    Local filesystem storage implementation for goofys-mounted directories.

    All operations work with local filesystem paths. When the mount point
    is managed by goofys, changes are automatically synced to remote storage.

    If execution_id is provided, all operations are automatically namespaced
    under executions/{execution_id}/ for complete isolation.
    """

    def __init__(self, config: StorageConfig, execution_id: Optional[str] = None):
        super().__init__(config)
        self.execution_id = execution_id
        self._initialized = False

        # Determine base mount path (with execution namespace if provided)
        if execution_id:
            self.mount_path = Path(config.mount_path) / "executions" / execution_id
        else:
            self.mount_path = Path(config.mount_path)

    async def initialize(self) -> None:
        """
        Initialize storage directories asynchronously.

        Must be called after construction to set up the storage structure.
        """
        if self._initialized:
            return

        # Ensure mount point exists
        await async_makedirs(self.mount_path, parents=True, exist_ok=True)

        # Create standard subdirectories if enabled
        if self.config.create_subdirs:
            await async_makedirs(self.mount_path / self.config.artifacts_subdir, parents=True, exist_ok=True)
            await async_makedirs(self.mount_path / self.config.temp_subdir, parents=True, exist_ok=True)
            await async_makedirs(self.mount_path / self.config.results_subdir, parents=True, exist_ok=True)
            await async_makedirs(self.mount_path / self.config.plots_subdir, parents=True, exist_ok=True)
            await async_makedirs(self.mount_path / self.config.reports_subdir, parents=True, exist_ok=True)
            await async_makedirs(self.mount_path / self.config.logs_subdir, parents=True, exist_ok=True)

        self._initialized = True
    
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
        await async_makedirs(full_path.parent, parents=True, exist_ok=True)

        # Write data atomically (write to temp file, then move)
        temp_path = full_path.with_suffix(full_path.suffix + ".tmp")

        try:
            # Use async file utilities for file I/O
            await async_write_bytes(temp_path, data)

            # Atomic move
            await async_rename(temp_path, full_path)
            
            # Store metadata as extended attributes (if supported)
            if metadata:
                await self._store_metadata(full_path, metadata)
            
            self.logger.debug(f"Stored {len(data)} bytes at {key}")
            return str(full_path)
            
        except Exception as e:
            # Clean up temp file on error
            if await async_path_exists(temp_path):
                await async_unlink(temp_path)
            
            self.logger.error(f"Failed to store data at {key}: {e}")
            raise
    
    async def get(self, key: str) -> Optional[bytes]:
        """Retrieve data by key."""
        full_path = self.get_full_path(key)

        if not await async_path_exists(full_path):
            return None

        try:
            # Use async file utilities for file I/O
            data = await async_read_bytes(full_path)
            
            self.logger.debug(f"Retrieved {len(data)} bytes from {key}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve data from {key}: {e}")
            return None

    async def put_text(
        self,
        key: str,
        text: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Store text content.

        Args:
            key: Storage key
            text: Text content to store
            metadata: Optional metadata dictionary

        Returns:
            Full path to stored file
        """
        data = text.encode('utf-8')
        return await self.put(key, data, metadata)

    async def get_text(self, key: str) -> Optional[str]:
        """
        Retrieve text content.

        Args:
            key: Storage key

        Returns:
            Text content or None if not found
        """
        data = await self.get(key)
        return data.decode('utf-8') if data else None

    async def exists(self, key: str) -> bool:
        """Check if file exists at key path."""
        full_path = self.get_full_path(key)
        return await async_path_exists(full_path) and await async_is_file(full_path)
    
    async def delete(self, key: str) -> bool:
        """Delete file at key path."""
        full_path = self.get_full_path(key)

        if not await async_path_exists(full_path):
            return False

        try:
            await async_unlink(full_path)
            
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
            if await async_is_dir(search_path):
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
            if "**" in pattern:
                files = await async_rglob(search_root, pattern.replace("**/*", "*"))
            else:
                files = await async_glob(search_root, pattern)

            # Convert to relative keys and filter files only
            keys = []
            for file_path in files:
                if await async_is_file(file_path):
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
        if not await async_path_exists(source):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        dest_path = self.get_full_path(key)
        await async_makedirs(dest_path.parent, parents=True, exist_ok=True)

        try:
            await async_copy2(source, dest_path)
            
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
        if not await async_path_exists(source):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        dest_path = self.get_full_path(key)
        await async_makedirs(dest_path.parent, parents=True, exist_ok=True)

        try:
            await async_move(source, dest_path)
            
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

        try:
            cleaned_count = await async_cleanup_old_files(
                temp_path, older_than_hours, "*"
            )
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
            
            await async_write_text(metadata_path, metadata_content)
            
        except Exception as e:
            # Metadata storage is optional - don't fail the main operation
            self.logger.warning(f"Failed to store metadata for {file_path}: {e}")
    
    async def cleanup_execution_temp_files(self) -> int:
        """
        Clean up temporary files for this execution.

        Only works if storage was initialized with execution_id.

        Returns:
            Number of files cleaned
        """
        if not self.execution_id:
            logger.warning("Cannot cleanup execution temp files - no execution_id set")
            return 0

        temp_path = self.mount_path / self.config.temp_subdir
        if not await async_path_exists(temp_path):
            return 0

        cleaned_count = 0
        try:
            files = await async_rglob(temp_path, "*")
            for file_path in files:
                if await async_is_file(file_path):
                    await async_unlink(file_path)
                    cleaned_count += 1

            self.logger.info(f"Cleaned {cleaned_count} temp files for execution {self.execution_id}")
            return cleaned_count

        except Exception as e:
            self.logger.error(f"Failed to clean temp files for {self.execution_id}: {e}")
            return cleaned_count

    async def get_storage_info(self) -> Dict[str, Any]:
        """Get storage usage information."""
        try:
            size_info = await async_calculate_directory_size(self.mount_path)

            if "error" in size_info:
                return size_info

            return {
                "mount_path": str(self.mount_path),
                "total_size_bytes": size_info["total_size_bytes"],
                "total_size_mb": size_info["total_size_mb"],
                "file_count": size_info["file_count"],
                "artifacts_path": str(self.mount_path / self.config.artifacts_subdir),
                "temp_path": str(self.mount_path / self.config.temp_subdir),
            }

        except Exception as e:
            self.logger.error(f"Failed to get storage info: {e}")
            return {"error": str(e)}