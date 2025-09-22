"""
Tests for Local Storage Implementation.

Tests the goofys-based local storage functionality including file operations,
directory management, and storage information.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock

from roma.infrastructure.storage.local_storage import LocalFileStorage
from roma.infrastructure.storage.storage_interface import StorageConfig


class TestStorageConfig:
    """Test StorageConfig functionality."""
    
    def test_from_mount_path(self):
        """Test creating config from mount path."""
        mount_path = "/tmp/test_mount"
        config = StorageConfig.from_mount_path(mount_path)
        
        assert config.mount_path == mount_path
        assert config.artifacts_subdir == "artifacts"
        assert config.temp_subdir == "temp"
        assert config.create_subdirs is True
    
    def test_config_immutability(self):
        """Test that config is immutable."""
        config = StorageConfig.from_mount_path("/tmp/test")
        
        with pytest.raises(Exception):  # Should raise validation error
            config.mount_path = "/different/path"


class TestLocalFileStorage:
    """Test LocalFileStorage functionality."""
    
    @pytest.fixture
    def temp_mount_path(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def storage_config(self, temp_mount_path):
        """Create storage config for testing."""
        return StorageConfig.from_mount_path(str(temp_mount_path))
    
    @pytest_asyncio.fixture
    async def storage(self, storage_config):
        """Create storage instance for testing."""
        storage = LocalFileStorage(storage_config)
        await storage.initialize()
        return storage

    @pytest.mark.asyncio
    async def test_initialization(self, storage, temp_mount_path):
        """Test storage initialization."""
        assert storage.mount_path == temp_mount_path
        assert storage.mount_path.exists()

        # Check subdirectories were created
        artifacts_path = temp_mount_path / "artifacts"
        temp_path = temp_mount_path / "temp"
        assert artifacts_path.exists()
        assert temp_path.exists()
    
    @pytest.mark.asyncio
    async def test_put_and_get(self, storage):
        """Test storing and retrieving data."""
        key = "test/file.txt"
        data = b"Hello, World!"
        
        # Store data
        result_path = await storage.put(key, data)
        assert result_path
        assert Path(result_path).exists()
        
        # Retrieve data
        retrieved_data = await storage.get(key)
        assert retrieved_data == data
    
    @pytest.mark.asyncio
    async def test_put_text_and_get_text(self, storage):
        """Test storing and retrieving text content."""
        key = "test/text_file.txt"
        text = "Hello, World!\nThis is a test."
        
        # Store text
        result_path = await storage.put_text(key, text)
        assert result_path
        
        # Retrieve text
        retrieved_text = await storage.get_text(key)
        assert retrieved_text == text
    
    @pytest.mark.asyncio
    async def test_exists(self, storage):
        """Test file existence check."""
        key = "test/exists.txt"
        data = b"test data"
        
        # File shouldn't exist initially
        assert await storage.exists(key) is False
        
        # Store file
        await storage.put(key, data)
        
        # File should exist now
        assert await storage.exists(key) is True
    
    @pytest.mark.asyncio
    async def test_delete(self, storage):
        """Test file deletion."""
        key = "test/delete_me.txt"
        data = b"delete this"
        
        # Store file
        await storage.put(key, data)
        assert await storage.exists(key) is True
        
        # Delete file
        result = await storage.delete(key)
        assert result is True
        assert await storage.exists(key) is False
        
        # Try to delete non-existent file
        result = await storage.delete("nonexistent.txt")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_list_keys(self, storage):
        """Test listing storage keys."""
        # Store multiple files
        files = [
            ("dir1/file1.txt", b"data1"),
            ("dir1/file2.txt", b"data2"),
            ("dir2/file3.txt", b"data3"),
            ("root_file.txt", b"root_data")
        ]
        
        for key, data in files:
            await storage.put(key, data)
        
        # List all keys
        all_keys = await storage.list_keys()
        assert len(all_keys) == 4
        assert all([key in all_keys for key, _ in files])
        
        # List keys with prefix
        dir1_keys = await storage.list_keys("dir1/")
        assert len(dir1_keys) == 2
        assert "dir1/file1.txt" in dir1_keys
        assert "dir1/file2.txt" in dir1_keys
    
    @pytest.mark.asyncio
    async def test_get_full_path(self, storage):
        """Test getting full path for key."""
        key = "test/subdir/file.txt"
        full_path = storage.get_full_path(key)
        
        expected = storage.mount_path / "test" / "subdir" / "file.txt"
        assert full_path == expected
    
    @pytest.mark.asyncio
    async def test_get_artifacts_path(self, storage):
        """Test getting artifacts path."""
        key = "my_artifact.dat"
        artifacts_path = storage.get_artifacts_path(key)
        
        expected = storage.mount_path / "artifacts" / "my_artifact.dat"
        assert artifacts_path == expected
    
    @pytest.mark.asyncio
    async def test_get_temp_path(self, storage):
        """Test getting temp path."""
        key = "temp_file.tmp"
        temp_path = storage.get_temp_path(key)
        
        expected = storage.mount_path / "temp" / "temp_file.tmp"
        assert temp_path == expected
    
    def test_generate_key(self, storage):
        """Test key generation."""
        key1 = storage.generate_key()
        key2 = storage.generate_key()
        
        # Should be unique
        assert key1 != key2
        assert len(key1) == 36  # UUID4 length
        
        # With prefix and suffix
        key_with_parts = storage.generate_key("prefix_", ".txt")
        assert key_with_parts.startswith("prefix_")
        assert key_with_parts.endswith(".txt")
    
    def test_normalize_key(self, storage):
        """Test key normalization."""
        test_cases = [
            ("/leading/slash", "leading/slash"),
            ("trailing/slash/", "trailing/slash"),
            ("//double//slashes//", "double/slashes"),
            ("mixed\\slashes/path", "mixed/slashes/path"),
            ("normal/path", "normal/path")
        ]
        
        for input_key, expected in test_cases:
            normalized = storage.normalize_key(input_key)
            assert normalized == expected
    
    @pytest.mark.asyncio
    async def test_get_size(self, storage):
        """Test getting file size."""
        key = "test/size_test.dat"
        data = b"x" * 1024  # 1KB of data
        
        # Store data
        await storage.put(key, data)
        
        # Get size
        size = await storage.get_size(key)
        assert size == 1024
        
        # Non-existent file
        size = await storage.get_size("nonexistent.txt")
        assert size is None
    
    @pytest.mark.asyncio
    async def test_copy_local(self, storage, temp_mount_path):
        """Test copying local file to storage."""
        # Create source file
        source_path = temp_mount_path / "source.txt"
        source_data = b"source file content"
        source_path.write_bytes(source_data)
        
        # Copy to storage
        key = "copied/file.txt"
        result_path = await storage.copy_local(str(source_path), key)
        
        assert result_path
        assert await storage.exists(key)
        
        # Verify content
        retrieved_data = await storage.get(key)
        assert retrieved_data == source_data
        
        # Source should still exist
        assert source_path.exists()
    
    @pytest.mark.asyncio
    async def test_move_local(self, storage, temp_mount_path):
        """Test moving local file to storage."""
        # Create source file
        source_path = temp_mount_path / "move_source.txt"
        source_data = b"move file content"
        source_path.write_bytes(source_data)
        
        # Move to storage
        key = "moved/file.txt"
        result_path = await storage.move_local(str(source_path), key)
        
        assert result_path
        assert await storage.exists(key)
        
        # Verify content
        retrieved_data = await storage.get(key)
        assert retrieved_data == source_data
        
        # Source should be gone
        assert not source_path.exists()
    
    @pytest.mark.asyncio
    async def test_cleanup_temp_files(self, storage):
        """Test cleaning up temporary files."""
        # Create temp files
        temp_keys = ["temp_old.tmp", "temp_new.tmp"]
        for key in temp_keys:
            await storage.put(f"temp/{key}", b"temp data")
        
        # Mock file modification times (would need more complex setup for real test)
        # For now, test the function exists and returns a count
        count = await storage.cleanup_temp_files(older_than_hours=0)
        assert isinstance(count, int)
        assert count >= 0
    
    @pytest.mark.asyncio
    async def test_get_storage_info(self, storage):
        """Test getting storage information."""
        info = await storage.get_storage_info()

        assert "mount_path" in info
        assert "total_size_bytes" in info
        assert "total_size_mb" in info
        assert "file_count" in info
        assert "artifacts_path" in info
        assert "temp_path" in info

        assert info["mount_path"] == str(storage.mount_path)
        assert info["file_count"] >= 0
        assert info["total_size_bytes"] >= 0
    
    @pytest.mark.asyncio
    async def test_max_file_size_limit(self, storage):
        """Test file size limit enforcement."""
        key = "large_file.dat"
        
        # Create data larger than max size
        large_data = b"x" * (storage.config.max_file_size + 1)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="File size .* exceeds maximum"):
            await storage.put(key, large_data)
    
    @pytest.mark.asyncio
    async def test_atomic_write(self, storage, temp_mount_path):
        """Test that writes are atomic (using temp file + move)."""
        key = "atomic_test.txt"
        data = b"atomic write test"
        
        # Mock file operations to simulate failure during write
        original_write_bytes = Path.write_bytes
        
        def failing_write_bytes(self, data):
            if self.name.endswith(".tmp"):
                raise IOError("Simulated write failure")
            return original_write_bytes(self, data)
        
        Path.write_bytes = failing_write_bytes
        
        try:
            with pytest.raises(IOError):
                await storage.put(key, data)
            
            # File should not exist after failed write
            assert not await storage.exists(key)
            
        finally:
            # Restore original method
            Path.write_bytes = original_write_bytes
    
    @pytest.mark.asyncio
    async def test_error_handling_get_nonexistent(self, storage):
        """Test error handling when getting non-existent file."""
        result = await storage.get("nonexistent/file.txt")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_directory_creation(self, storage):
        """Test automatic directory creation."""
        key = "deep/nested/directory/structure/file.txt"
        data = b"nested file"
        
        # Should create all necessary directories
        await storage.put(key, data)
        assert await storage.exists(key)
        
        # Verify directory structure exists
        full_path = storage.get_full_path(key)
        assert full_path.parent.exists()
        assert full_path.parent.is_dir()