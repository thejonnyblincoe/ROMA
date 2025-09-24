"""
Recovery Status Value Object for Persistence.

Domain value object for recovery operation status used in persistence operations.
"""

from enum import Enum


class RecoveryStatus(str, Enum):
    """Status of recovery operations."""

    ACTIVE = "active"
    RECOVERED = "recovered"
    ABANDONED = "abandoned"
    FAILED = "failed"
    TIMEOUT = "timeout"