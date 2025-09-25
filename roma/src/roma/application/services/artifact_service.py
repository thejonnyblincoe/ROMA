"""
Artifact Service - ROMA v2.0 Application Layer.

Handles artifact operations for ResultEnvelope, bridging between
domain artifacts and storage infrastructure.
"""

import logging
from typing import Any

from roma.domain.entities.artifacts.base_artifact import BaseArtifact
from roma.domain.entities.artifacts.file_artifact import FileArtifact
from roma.domain.interfaces.artifact_service import IArtifactService
from roma.domain.interfaces.storage import IStorage
from roma.domain.value_objects.media_type import MediaType
from roma.domain.value_objects.result_envelope import AnyResultEnvelope

logger = logging.getLogger(__name__)


class ArtifactService(IArtifactService):
    """
    Service for managing artifacts in ResultEnvelopes.

    Provides high-level artifact operations using the existing storage
    infrastructure while maintaining clean separation of concerns.
    """

    def __init__(self, storage: IStorage):
        """
        Initialize artifact service.

        Args:
            storage: Storage interface implementation
        """
        self.storage = storage
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize artifact service and ensure storage directories exist."""
        if self._initialized:
            return

        # Ensure artifact directories exist in storage
        artifacts_path = self.storage.get_artifacts_path("")
        artifacts_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = self.storage.get_temp_path("")
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logger.info("ArtifactService initialized")

    async def store_envelope_artifacts(
        self, envelope: AnyResultEnvelope, execution_id: str
    ) -> list[str]:
        """
        Store all artifacts from a ResultEnvelope and return storage references.

        Args:
            envelope: ResultEnvelope containing artifacts to store

        Returns:
            List of storage keys/references for stored artifacts
        """
        if not self._initialized:
            await self.initialize()

        if not envelope.artifacts:
            logger.debug("No artifacts to store")
            return []

        stored_refs = []

        for artifact in envelope.artifacts:
            try:
                storage_ref = await self._store_single_artifact(execution_id, artifact)
                stored_refs.append(storage_ref)

            except Exception as e:
                logger.error(f"Failed to store artifact {artifact.name}: {e}")
                # Continue with other artifacts

        logger.info(f"Stored {len(stored_refs)}/{len(envelope.artifacts)} artifacts")
        return stored_refs

    async def _store_single_artifact(self, task_id: str, artifact: BaseArtifact) -> str:
        """
        Store a single artifact and return its storage reference.

        Args:
            task_id: Associated task ID
            artifact: Artifact to store

        Returns:
            Storage key/reference for the stored artifact
        """
        # Generate storage key
        storage_key = self._generate_artifact_key(task_id, artifact)

        # Get artifact content
        content = await artifact.get_content()
        if content is None:
            raise ValueError(f"Artifact {artifact.name} has no accessible content")

        # Prepare metadata
        metadata = {
            "artifact_id": artifact.artifact_id,
            "task_id": task_id,
            "media_type": artifact.media_type.value,
            "created_at": artifact.created_at.isoformat(),
            "original_name": artifact.name,
            **artifact.metadata,
        }

        # Store using storage interface (already execution-isolated)
        if isinstance(content, str):
            # Text content
            await self.storage.put_text(storage_key, content, metadata=metadata)
        else:
            # Binary content
            await self.storage.put(storage_key, content, metadata=metadata)

        logger.debug(f"Stored artifact {artifact.name} at {storage_key}")
        return storage_key

    def _generate_artifact_key(self, task_id: str, artifact: BaseArtifact) -> str:
        """
        Generate organized storage key for artifact.

        Args:
            task_id: Task identifier
            artifact: Artifact to generate key for

        Returns:
            Storage key in format: tasks/{task_id}/{artifact_id}_{name}
        """
        # Sanitize artifact name for filesystem
        safe_name = self._sanitize_filename(artifact.name)

        # Add appropriate file extension based on media type
        if hasattr(artifact, "get_file_extension") and artifact.get_file_extension():
            extension = artifact.get_file_extension()
        else:
            extension = self._get_extension_for_media_type(artifact.media_type)

        # Build hierarchical key (execution_id is handled by storage)
        filename = f"{artifact.artifact_id}_{safe_name}{extension}"
        storage_key = f"tasks/{task_id}/{filename}"

        return storage_key

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove/replace unsafe characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)

        # Limit length and ensure not empty
        sanitized = sanitized[:50] if sanitized else "artifact"

        return sanitized

    def _get_extension_for_media_type(self, media_type: MediaType) -> str:
        """Get appropriate file extension for media type."""
        extension_map = {
            MediaType.TEXT: ".txt",
            MediaType.FILE: "",  # Keep original or no extension
            MediaType.IMAGE: ".png",
            MediaType.AUDIO: ".mp3",
            MediaType.VIDEO: ".mp4",
        }
        return extension_map.get(media_type, "")

    async def retrieve_artifact(
        self, storage_key: str, as_text: bool = False
    ) -> bytes | str | None:
        """
        Retrieve artifact content by storage key.

        Args:
            storage_key: Storage key returned from store operation
            as_text: If True, return content as text string

        Returns:
            Artifact content, or None if not found
        """
        if as_text:
            return await self.storage.get_text(storage_key)
        else:
            return await self.storage.get(storage_key)

    async def list_execution_artifacts(self) -> list[str]:
        """
        List all artifact storage keys for this execution.

        Returns:
            List of storage keys for artifacts in this execution
        """
        # Storage is already execution-isolated, so list all keys
        return await self.storage.list_keys("")

    async def get_artifact_metadata(self, storage_key: str) -> dict[str, Any]:
        """
        Get metadata for stored artifact.

        Args:
            storage_key: Storage key for artifact

        Returns:
            Metadata dictionary, empty if not found
        """
        # This would need extended attributes support in StorageInterface
        # For now, return basic info
        if await self.storage.exists(storage_key):
            size = await self.storage.get_size(storage_key)
            full_path = self.storage.get_full_path(storage_key)

            return {
                "storage_key": storage_key,
                "size_bytes": size,
                "full_path": str(full_path),
                "exists": True,
            }

        return {"exists": False}

    async def cleanup_execution_artifacts(self) -> int:
        """
        Clean up all artifacts for this execution.

        Returns:
            Number of artifacts deleted
        """
        artifact_keys = await self.list_execution_artifacts()
        deleted_count = 0

        for key in artifact_keys:
            try:
                if await self.storage.delete(key):
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete artifact {key}: {e}")

        logger.info(f"Cleaned up {deleted_count}/{len(artifact_keys)} artifacts")
        return deleted_count

    async def create_file_artifact_from_storage(
        self, storage_key: str, name: str, task_id: str | None = None
    ) -> FileArtifact | None:
        """
        Create a FileArtifact from existing storage.

        Args:
            storage_key: Storage key for existing file
            name: Human-readable name for artifact
            task_id: Optional associated task ID

        Returns:
            FileArtifact instance, or None if storage key not found
        """
        if not await self.storage.exists(storage_key):
            return None

        # Get full path for MediaFile creation
        full_path = self.storage.get_full_path(storage_key)

        try:
            artifact = FileArtifact.from_path(
                name=name,
                file_path=str(full_path),
                task_id=task_id,
                metadata={"storage_key": storage_key},
            )
            return artifact

        except Exception as e:
            logger.error(f"Failed to create FileArtifact from {storage_key}: {e}")
            return None

    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get storage statistics and configuration info.

        Returns:
            Dictionary with storage stats
        """
        return {
            "storage_type": type(self.storage).__name__,
            "mount_path": str(self.storage.mount_path),
            "artifacts_path": str(self.storage.get_artifacts_path("")),
            "temp_path": str(self.storage.get_temp_path("")),
            "initialized": self._initialized,
            "config": self.storage.config.model_dump(),
        }
