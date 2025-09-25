"""
Storage Infrastructure Package - ROMA v2.0.

Provides storage abstractions and implementations for file storage,
caching, and persistence across local filesystem and cloud services.
"""

from roma.domain.interfaces.storage import IStorage
from roma.domain.value_objects.storage_config import StorageConfig

from .local_storage import LocalFileStorage

__all__ = [
    "IStorage",
    "StorageConfig",
    "LocalFileStorage",
]
