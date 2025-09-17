"""
Test configuration and fixtures for ROMA v2.0 tests.

Provides common test fixtures and utilities for all test modules.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.value_objects.node_type import NodeType
from src.roma.domain.events.task_events import TaskCreatedEvent, TaskStatusChangedEvent
from src.roma.application.services.event_store import InMemoryEventStore, get_event_store


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def clean_event_store() -> AsyncGenerator[InMemoryEventStore, None]:
    """Provide clean event store for each test."""
    store = InMemoryEventStore(max_events_per_task=100, max_total_events=1000)
    yield store
    await store.clear()


@pytest.fixture
def sample_task_node() -> TaskNode:
    """Create a sample TaskNode for testing."""
    return TaskNode(
        task_id="test-task-123",
        goal="Test goal for sample task",
        task_type=TaskType.THINK,
        parent_id=None
    )


@pytest.fixture
def atomic_task_node() -> TaskNode:
    """Create an atomic task node for testing."""
    return TaskNode(
        task_id="atomic-task-456", 
        goal="Simple search query",
        task_type=TaskType.RETRIEVE,
        node_type=NodeType.EXECUTE
    )


@pytest.fixture
def composite_task_node() -> TaskNode:
    """Create a composite task node for testing."""
    return TaskNode(
        task_id="composite-task-789",
        goal="Complex analysis requiring multiple steps",
        task_type=TaskType.THINK,
        node_type=NodeType.PLAN
    )


@pytest.fixture
def task_hierarchy() -> list[TaskNode]:
    """Create a hierarchy of related tasks for testing."""
    root = TaskNode(
        task_id="root-task",
        goal="Research cryptocurrency market trends",
        task_type=TaskType.THINK,
        parent_id=None
    )
    
    child1 = TaskNode(
        task_id="child-task-1",
        goal="Gather price data from CoinGecko",
        task_type=TaskType.RETRIEVE,
        parent_id=root.task_id
    )
    
    child2 = TaskNode(
        task_id="child-task-2", 
        goal="Analyze trading volume patterns",
        task_type=TaskType.THINK,
        parent_id=root.task_id
    )
    
    grandchild = TaskNode(
        task_id="grandchild-task",
        goal="Calculate 30-day moving average",
        task_type=TaskType.THINK,
        parent_id=child2.task_id
    )
    
    return [root, child1, child2, grandchild]


@pytest.fixture
def sample_events(sample_task_node: TaskNode) -> list:
    """Create sample events for testing."""
    created_event = TaskCreatedEvent.create(
        task_id=sample_task_node.task_id,
        goal=sample_task_node.goal,
        task_type=sample_task_node.task_type
    )
    
    status_change_event = TaskStatusChangedEvent.create(
        task_id=sample_task_node.task_id,
        old_status=TaskStatus.PENDING,
        new_status=TaskStatus.READY,
        version=1
    )
    
    return [created_event, status_change_event]



@pytest.fixture
def mock_agno_agent() -> MagicMock:
    """Create a mock Agno agent for testing."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock()
    return mock_agent


@pytest.fixture
def mock_atomizer_output() -> MagicMock:
    """Create a mock AtomizerOutput for testing."""
    mock_output = MagicMock()
    mock_output.is_atomic = True
    mock_output.updated_goal = "Refined test goal"
    mock_output.reasoning_steps = []
    return mock_output


# Test utilities
class TestEventSubscriber:
    """Test event subscriber for capturing events."""
    
    def __init__(self):
        self.received_events = []
        self.call_count = 0
    
    def __call__(self, event):
        self.received_events.append(event)
        self.call_count += 1
    
    async def async_callback(self, event):
        self.received_events.append(event) 
        self.call_count += 1


@pytest.fixture
def test_event_subscriber() -> TestEventSubscriber:
    """Create a test event subscriber."""
    return TestEventSubscriber()


# Async test utilities
@pytest_asyncio.fixture
async def running_event_store() -> AsyncGenerator[InMemoryEventStore, None]:
    """Event store with some events already added."""
    store = InMemoryEventStore()
    
    # Add some test events
    node = TaskNode(goal="Test task", task_type=TaskType.THINK)
    created = TaskCreatedEvent.create(
        task_id=node.task_id,
        goal=node.goal,
        task_type=node.task_type
    )
    await store.append(created)
    
    status_changed = TaskStatusChangedEvent.create(
        task_id=node.task_id,
        old_status=TaskStatus.PENDING,
        new_status=TaskStatus.READY,
        version=1
    )
    await store.append(status_changed)
    
    yield store
    await store.clear()


# Performance test fixtures
@pytest.fixture
def large_task_set() -> list[TaskNode]:
    """Create a large set of tasks for performance testing."""
    tasks = []
    for i in range(100):
        task = TaskNode(
            task_id=f"perf-task-{i}",
            goal=f"Performance test task {i}",
            task_type=TaskType.THINK if i % 3 == 0 else TaskType.RETRIEVE
        )
        tasks.append(task)
    return tasks


# Integration test fixtures  
@pytest.fixture
def mock_agent_registry() -> MagicMock:
    """Mock agent registry for integration tests."""
    registry = MagicMock()
    registry.get_agent_adapter = AsyncMock()
    return registry


# Assertion helpers
def assert_task_node_immutable(node: TaskNode):
    """Assert that TaskNode is properly immutable."""
    # Try to modify fields (should fail)
    try:
        node.goal = "Modified goal"
        assert False, "TaskNode should be immutable"
    except AttributeError:
        pass  # Expected
    
    try:
        node.status = TaskStatus.COMPLETED
        assert False, "TaskNode should be immutable"
    except AttributeError:
        pass  # Expected


def assert_valid_state_transition(old_node: TaskNode, new_node: TaskNode):
    """Assert that state transition is valid."""
    # Must be different instances
    assert old_node is not new_node, "State transition must create new instance"
    
    # Version must increment
    assert new_node.version == old_node.version + 1, "Version must increment"
    
    # Transition must be valid
    assert old_node.status.can_transition_to_status(new_node.status), \
        f"Invalid transition from {old_node.status} to {new_node.status}"


def assert_event_valid(event):
    """Assert that event has required fields."""
    assert event.event_id, "Event must have ID"
    assert event.task_id, "Event must have task ID"
    assert event.timestamp, "Event must have timestamp" 
    assert event.event_type, "Event must have type"
    assert isinstance(event.metadata, dict), "Event metadata must be dict"


# Parametrized fixtures for different task types
@pytest.fixture(params=[TaskType.RETRIEVE, TaskType.WRITE, TaskType.THINK])
def task_type_variant(request) -> TaskType:
    """Parametrized fixture for different task types."""
    return request.param


@pytest.fixture(params=[TaskStatus.PENDING, TaskStatus.READY, TaskStatus.EXECUTING])
def task_status_variant(request) -> TaskStatus:
    """Parametrized fixture for different task statuses."""
    return request.param


@pytest.fixture(params=[NodeType.PLAN, NodeType.EXECUTE])
def node_type_variant(request) -> NodeType:
    """Parametrized fixture for different node types.""" 
    return request.param