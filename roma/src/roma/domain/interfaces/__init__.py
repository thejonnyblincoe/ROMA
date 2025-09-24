"""
Domain Interfaces.

Centralized exports for all domain interfaces organized by category.
"""

from .persistence import (
    CheckpointRepository,
    RecoveryRepository,
    ExecutionHistoryRepository,
)

__all__ = [
    "CheckpointRepository",
    "RecoveryRepository",
    "ExecutionHistoryRepository",
]