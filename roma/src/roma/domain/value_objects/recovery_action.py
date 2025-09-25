"""
Recovery Action Value Object.

Domain value object for recovery actions that can be taken when tasks fail.
"""

from enum import Enum


class RecoveryAction(str, Enum):
    """Possible recovery actions for failed tasks."""

    RETRY = "RETRY"
    REPLAN = "REPLAN"
    FORCE_ATOMIC = "FORCE_ATOMIC"
    FAIL_PERMANENTLY = "FAIL_PERMANENTLY"
    CIRCUIT_BREAK = "CIRCUIT_BREAK"
