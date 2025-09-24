"""
Checkpoint Type Value Object for Persistence.

Domain value object for checkpoint types used in persistence operations.
"""

from enum import Enum


class CheckpointType(str, Enum):
    """Types of execution checkpoints."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"
    ERROR_RECOVERY = "error_recovery"
    MILESTONE = "milestone"
    BACKUP = "backup"