"""
Storage Infrastructure Package - ROMA v2.0.

Provides storage abstractions and implementations for file storage,
caching, and persistence across local filesystem and cloud services.
"""

from .local_storage import LocalFileStorage
from .storage_interface import StorageInterface, StorageConfig

__all__ = [
    "StorageInterface", 
    "StorageConfig",
    "LocalFileStorage",
]