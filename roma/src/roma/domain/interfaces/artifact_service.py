"""
Artifact Service Interface - ROMA v2.0 Domain Interface.

Abstract interface for artifact management operations that application must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.entities.artifacts.base_artifact import BaseArtifact
from roma.domain.value_objects.result_envelope import AnyResultEnvelope


class IArtifactService(ABC):
    """
    Domain interface for artifact service operations.

    Abstract interface that defines all artifact management operations required by the domain.
    Application layer must implement this interface to provide artifact management capabilities.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize artifact service and ensure storage directories exist."""

    @abstractmethod
    async def store_envelope_artifacts(
        self, envelope: AnyResultEnvelope, execution_id: str
    ) -> list[BaseArtifact]:
        """
        Store all artifacts from a result envelope.

        Args:
            envelope: Result envelope containing artifacts
            execution_id: Execution identifier for organization

        Returns:
            List of stored artifacts with storage keys

        Raises:
            ArtifactStorageError: If artifact storage fails
        """

    @abstractmethod
    async def retrieve_artifact(
        self, storage_key: str, include_content: bool = False
    ) -> BaseArtifact | None:
        """
        Retrieve artifact by storage key.

        Args:
            storage_key: Unique storage key for artifact
            include_content: Whether to include file content in result

        Returns:
            Artifact if found, None otherwise

        Raises:
            ArtifactRetrievalError: If artifact retrieval fails
        """

    @abstractmethod
    async def list_execution_artifacts(self) -> list[str]:
        """
        List all artifact storage keys for current execution.

        Returns:
            List of storage keys for execution artifacts

        Raises:
            ArtifactListingError: If listing fails
        """

    @abstractmethod
    async def get_artifact_metadata(self, storage_key: str) -> dict[str, Any]:
        """
        Get metadata for an artifact.

        Args:
            storage_key: Storage key for artifact

        Returns:
            Dictionary with artifact metadata

        Raises:
            ArtifactNotFoundError: If artifact not found
        """

    @abstractmethod
    async def cleanup_execution_artifacts(self) -> int:
        """
        Clean up artifacts for current execution.

        Returns:
            Number of artifacts cleaned up

        Raises:
            ArtifactCleanupError: If cleanup fails
        """

    @abstractmethod
    async def create_file_artifact_from_storage(
        self, storage_key: str, original_filename: str, media_type_hint: str | None = None
    ) -> BaseArtifact:
        """
        Create file artifact from existing storage key.

        Args:
            storage_key: Storage key for existing file
            original_filename: Original filename for the artifact
            media_type_hint: Optional media type hint

        Returns:
            File artifact with storage information

        Raises:
            ArtifactCreationError: If artifact creation fails
        """

    @abstractmethod
    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics (files, sizes, etc.)
        """
