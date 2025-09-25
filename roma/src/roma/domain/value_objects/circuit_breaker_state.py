"""
Circuit Breaker State Value Object.

Domain value object for circuit breaker states used in recovery management.
"""

from enum import Enum


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
