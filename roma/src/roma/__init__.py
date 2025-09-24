"""
ROMA - Research-Oriented Multi-Agent Architecture

Advanced Hierarchical Agent Framework with immutable state, event sourcing,
and atomizer-centric design.
"""

__version__ = "2.0.0"
__title__ = "ROMA"
__description__ = "Research-Oriented Multi-Agent Architecture - Advanced Hierarchical Agent Framework"
__author__ = "ROMA Development Team"

# Core exports
from .domain.entities.task_node import TaskNode
from .domain.value_objects.task_type import TaskType
from .domain.value_objects.task_status import TaskStatus
from .domain.value_objects.node_type import NodeType
# Domain interfaces removed - using agent-based implementations
from .domain.value_objects.agent_responses import AtomizerResult

# Application services
from .application.services.event_store import InMemoryEventStore, emit_event, get_event_store

# Framework entry imports moved to prevent circular dependency
# Import these directly from roma.framework_entry when needed

# Event types
from .domain.events.task_events import (
    BaseTaskEvent,
    TaskCreatedEvent,
    TaskStatusChangedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    AtomizerEvaluatedEvent,
    TaskDecomposedEvent,
    TaskExecutedEvent,
    ResultsAggregatedEvent,
)

__all__ = [
    # Version info
    "__version__",
    "__title__",
    "__description__",
    "__author__",

    # Core domain
    "TaskNode",
    "TaskType",
    "TaskStatus",
    "NodeType",
    "AtomizerResult",

    # Application services
    "InMemoryEventStore",
    "emit_event",
    "get_event_store",

    # Framework entry
    "SentientAgent",
    "ProfiledSentientAgent",
    "LightweightSentientAgent",

    # Events
    "BaseTaskEvent",
    "TaskCreatedEvent",
    "TaskStatusChangedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "AtomizerEvaluatedEvent",
    "TaskDecomposedEvent",
    "TaskExecutedEvent",
    "ResultsAggregatedEvent",
]