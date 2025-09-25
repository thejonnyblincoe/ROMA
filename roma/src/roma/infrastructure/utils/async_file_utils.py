"""
Async File Utilities - ROMA v2.0 Infrastructure

Provides async-safe file system operations to prevent blocking the event loop.
All file I/O operations are wrapped with asyncio.run_in_executor() for proper concurrency.
"""

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


async def async_makedirs(path: str | Path, parents: bool = True, exist_ok: bool = True) -> None:
    """
    Async directory creation.

    Args:
        path: Directory path to create
        parents: Create parent directories if needed
        exist_ok: Don't raise error if directory exists
    """

    def _mkdir() -> None:
        path_obj = Path(path)
        path_obj.mkdir(parents=parents, exist_ok=exist_ok)
        # Ensure the created directory has proper write permissions
        if path_obj.exists():
            path_obj.chmod(0o755)

    await asyncio.get_event_loop().run_in_executor(None, _mkdir)


async def async_path_exists(path: str | Path) -> bool:
    """
    Async path existence check.

    Args:
        path: Path to check

    Returns:
        True if path exists
    """
    path_obj = Path(path)
    return await asyncio.get_event_loop().run_in_executor(None, path_obj.exists)


async def async_is_file(path: str | Path) -> bool:
    """
    Async file check.

    Args:
        path: Path to check

    Returns:
        True if path is a file
    """
    path_obj = Path(path)
    return await asyncio.get_event_loop().run_in_executor(None, path_obj.is_file)


async def async_is_dir(path: str | Path) -> bool:
    """
    Async directory check.

    Args:
        path: Path to check

    Returns:
        True if path is a directory
    """
    path_obj = Path(path)
    return await asyncio.get_event_loop().run_in_executor(None, path_obj.is_dir)


async def async_file_stat(path: str | Path) -> os.stat_result | None:
    """
    Async file stat.

    Args:
        path: Path to stat

    Returns:
        Stat result or None if file doesn't exist
    """
    path_obj = Path(path)
    try:
        return await asyncio.get_event_loop().run_in_executor(None, path_obj.stat)
    except FileNotFoundError:
        return None


async def async_file_size(path: str | Path) -> int | None:
    """
    Async file size check.

    Args:
        path: Path to check

    Returns:
        File size in bytes or None if file doesn't exist
    """
    stat_result = await async_file_stat(path)
    return stat_result.st_size if stat_result else None


async def async_unlink(path: str | Path) -> bool:
    """
    Async file deletion.

    Args:
        path: File path to delete

    Returns:
        True if deleted, False if file didn't exist
    """
    path_obj = Path(path)
    try:
        await asyncio.get_event_loop().run_in_executor(None, path_obj.unlink)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.error(f"Failed to delete file {path}: {e}")
        raise


async def async_rename(src: str | Path, dst: str | Path) -> None:
    """
    Async file rename/move.

    Args:
        src: Source path
        dst: Destination path
    """
    src_obj = Path(src)
    dst_obj = Path(dst)
    await asyncio.get_event_loop().run_in_executor(None, src_obj.rename, dst_obj)


async def async_copy2(src: str | Path, dst: str | Path) -> None:
    """
    Async file copy with metadata preservation.

    Args:
        src: Source path
        dst: Destination path
    """
    await asyncio.get_event_loop().run_in_executor(None, shutil.copy2, str(src), str(dst))


async def async_move(src: str | Path, dst: str | Path) -> None:
    """
    Async file move.

    Args:
        src: Source path
        dst: Destination path
    """
    await asyncio.get_event_loop().run_in_executor(None, shutil.move, str(src), str(dst))


async def async_glob(path: str | Path, pattern: str) -> list[Path]:
    """
    Async glob pattern matching.

    Args:
        path: Base path to search
        pattern: Glob pattern

    Returns:
        List of matching paths
    """
    path_obj = Path(path)
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: list(path_obj.glob(pattern))
    )


async def async_rglob(path: str | Path, pattern: str) -> list[Path]:
    """
    Async recursive glob pattern matching.

    Args:
        path: Base path to search
        pattern: Glob pattern

    Returns:
        List of matching paths
    """
    path_obj = Path(path)
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: list(path_obj.rglob(pattern))
    )


async def async_read_bytes(path: str | Path) -> bytes:
    """
    Async binary file reading.

    Args:
        path: File path to read

    Returns:
        File contents as bytes
    """
    path_obj = Path(path)
    return await asyncio.get_event_loop().run_in_executor(None, path_obj.read_bytes)


async def async_write_bytes(path: str | Path, data: bytes) -> None:
    """
    Async binary file writing.

    Args:
        path: File path to write
        data: Data to write
    """
    path_obj = Path(path)
    await asyncio.get_event_loop().run_in_executor(None, path_obj.write_bytes, data)


async def async_read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """
    Async text file reading.

    Args:
        path: File path to read
        encoding: Text encoding

    Returns:
        File contents as string
    """
    path_obj = Path(path)
    return await asyncio.get_event_loop().run_in_executor(None, path_obj.read_text, encoding)


async def async_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """
    Async text file writing.

    Args:
        path: File path to write
        text: Text to write
        encoding: Text encoding
    """
    path_obj = Path(path)
    await asyncio.get_event_loop().run_in_executor(None, path_obj.write_text, text, encoding)


async def async_yaml_load(path: str | Path) -> dict[str, Any]:
    """
    Async YAML file loading.

    Args:
        path: YAML file path

    Returns:
        Parsed YAML data
    """

    def _load_yaml() -> Any:
        return yaml.safe_load(Path(path).read_text())

    return await asyncio.get_event_loop().run_in_executor(None, _load_yaml)


async def async_yaml_dump(path: str | Path, data: dict[str, Any], **kwargs: Any) -> None:
    """
    Async YAML file writing.

    Args:
        path: YAML file path
        data: Data to write
        **kwargs: Additional arguments for yaml.dump
    """

    def _dump_yaml() -> None:
        Path(path).write_text(yaml.dump(data, **kwargs))

    await asyncio.get_event_loop().run_in_executor(None, _dump_yaml)


async def async_cleanup_old_files(
    directory: str | Path, older_than_hours: int = 24, pattern: str = "*"
) -> int:
    """
    Async cleanup of old files.

    Args:
        directory: Directory to clean
        older_than_hours: Remove files older than this many hours
        pattern: File pattern to match

    Returns:
        Number of files cleaned up
    """
    directory_path = Path(directory)
    if not await async_path_exists(directory_path):
        return 0

    cutoff_time = time.time() - (older_than_hours * 3600)
    cleaned_count = 0

    # Get list of files matching pattern
    files = await async_rglob(directory_path, pattern)

    for file_path in files:
        if await async_is_file(file_path):
            stat_result = await async_file_stat(file_path)
            if stat_result and stat_result.st_mtime < cutoff_time:
                try:
                    await async_unlink(file_path)
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete old file {file_path}: {e}")

    return cleaned_count


async def async_calculate_directory_size(directory: str | Path) -> dict[str, Any]:
    """
    Async directory size calculation.

    Args:
        directory: Directory to analyze

    Returns:
        Dictionary with size information
    """
    directory_path = Path(directory)
    if not await async_path_exists(directory_path):
        return {"error": "Directory not found"}

    total_size = 0
    file_count = 0

    # Get all files recursively
    files = await async_rglob(directory_path, "*")

    for file_path in files:
        if await async_is_file(file_path):
            stat_result = await async_file_stat(file_path)
            if stat_result:
                total_size += stat_result.st_size
                file_count += 1

    return {
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "file_count": file_count,
        "directory": str(directory_path),
    }
