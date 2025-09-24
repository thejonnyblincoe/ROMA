"""
Task Relationship Type Value Object for Persistence.

Domain value object for task relationship types used in persistence operations.
"""

from enum import Enum


class TaskRelationshipType(str, Enum):
    """Types of relationships between tasks."""

    PARENT_CHILD = "parent_child"
    DEPENDENCY = "dependency"
    SIBLING = "sibling"
    SEQUENCE = "sequence"
    PARALLEL = "parallel"