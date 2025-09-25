"""
Storage Configuration Value Object.

Configuration for goofys-based storage systems.
"""

from pydantic import BaseModel, ConfigDict, Field


class StorageConfig(BaseModel):
    """Configuration for goofys-based storage."""

    model_config = ConfigDict(frozen=True)

    # Local mount point (goofys mounts remote storage here)
    mount_path: str = Field(description="Local path where goofys mounts remote storage")

    # Performance settings
    max_file_size: int = Field(
        default=100 * 1024 * 1024, description="Max file size in bytes (100MB)"
    )
    create_subdirs: bool = Field(default=True, description="Auto-create subdirectories")

    # File organization subdirectories
    artifacts_subdir: str = Field(default="artifacts", description="Subdirectory for artifacts")
    temp_subdir: str = Field(default="temp", description="Subdirectory for temporary files")
    results_subdir: str = Field(default="results", description="Subdirectory for execution results")
    plots_subdir: str = Field(default="results/plots", description="Subdirectory for plot outputs")
    reports_subdir: str = Field(
        default="results/reports", description="Subdirectory for report outputs"
    )
    logs_subdir: str = Field(default="logs", description="Subdirectory for execution logs")

    @classmethod
    def from_mount_path(cls, mount_path: str) -> "StorageConfig":
        """Create storage config from goofys mount path."""
        return cls(mount_path=mount_path)
