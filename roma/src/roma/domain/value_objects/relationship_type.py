"""
Task Relationship Type Value Object for ROMA v2.0
"""

from enum import Enum


class TaskRelationshipType(str, Enum):
    """
    Task relationship types for hierarchical task graph management.

    Defines the types of relationships that can exist between tasks
    in the dynamic task graph structure.
    """

    PARENT_CHILD = "PARENT_CHILD"
    """Standard parent-child decomposition relationship"""

    SIBLING = "SIBLING"
    """Tasks at the same level in hierarchy"""

    DEPENDENCY = "DEPENDENCY"
    """Task depends on completion of another task"""

    AGGREGATION = "AGGREGATION"
    """Task aggregates results from multiple child tasks"""