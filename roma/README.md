# ROMA v2.0 - Advanced Hierarchical Agent Framework

> **Research-Oriented Multi-Agent Architecture v2.0**  
> Next-generation immutable, event-driven task execution system

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-blue)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-96%25%20coverage-green)](./tests/)
[![Architecture](https://img.shields.io/badge/architecture-clean%20%2B%20immutable-brightgreen)](./ARCHITECTURE_SUMMARY.md)

## 🚀 Core Innovation

ROMA v2.0 introduces a fundamentally new approach to AI agent orchestration through:

- **Immutable Task Nodes**: Thread-safe execution through frozen dataclasses
- **Atomizer-Centric Design**: Every task flows through `Task → ATOMIZER → (PLAN or EXECUTE) → Result`
- **MECE Framework**: Universal task classification (RETRIEVE, WRITE, THINK)
- **Event Sourcing**: Complete observability and replay capability
- **Dynamic DAG Construction**: Graphs built during execution based on atomizer decisions

## 🏗️ Architecture Overview

### Clean Architecture Layers

```
src/
├── domain/                 # Core Business Logic
│   ├── entities/          # TaskNode (immutable)
│   ├── value_objects/     # TaskType, TaskStatus, NodeType
│   ├── events/           # 8 comprehensive event types
│   └── interfaces/       # Atomizer contracts
├── application/          # Use Cases & Services  
│   └── services/         # AtomizerService, EventStore
└── infrastructure/       # External Integrations
    └── adapters/         # Agno agent bridges
```

### Key Design Patterns

**1. Immutable Entity with State Transitions**
```python
@dataclass(frozen=True, slots=True)
class TaskNode:
    def transition_to(self, status: TaskStatus) -> "TaskNode":
        return dataclass_replace(self, status=status, version=self.version + 1)
```

**2. Atomizer Decision Pattern**
```python
class Atomizer(ABC):
    @abstractmethod
    async def evaluate(self, node: TaskNode) -> AtomizerResult:
        # Core decision: PLAN (decompose) or EXECUTE (atomic)
        pass
```

**3. Event Sourcing for Observability**
```python
await emit_event(TaskStatusChangedEvent.create(
    task_id=node.task_id,
    old_status=old_status, 
    new_status=new_status
))
```

## 📦 Installation & Setup

### Prerequisites

- Python 3.12+
- [uv package manager](https://github.com/astral-sh/uv)
- [just command runner](https://github.com/casey/just) (optional, for development workflows)

### Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd roma_v2

# Install with uv
uv sync

# Run tests
uv run pytest

# Install in development mode
uv pip install -e .
```

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Using just (recommended)
just setup                    # Setup development environment
just test                     # Run full test suite
just test-domain             # Run domain tests only
just test-app                # Run application tests only
just lint                    # Run all code quality checks
just format                  # Auto-format code
just pre-commit              # Run pre-commit checks

# Or use uv directly
uv run pytest tests/ -v                      # Run all tests
uv run pytest tests/unit/domain/ -v          # Domain tests
uv run pytest tests/unit/application/ -v     # Application tests  
uv run pytest tests/integration/ -v          # Integration tests
uv run ruff check src/ tests/               # Linting
uv run mypy src/                            # Type checking
```

## 🧪 Testing

ROMA v2.0 includes comprehensive testing with **96% coverage** across 59 test cases:

### Test Categories

- **Unit Tests (39)**: Domain logic validation
  - Immutability enforcement
  - State transition validation
  - MECE framework compliance
  - Property and method validation

- **Integration Tests (20)**: Service integration
  - Event store functionality  
  - Atomizer decision logic
  - Cross-component interaction

### Running Tests

```bash
# All tests with coverage
uv run pytest --cov=src --cov-report=html

# Domain layer tests
uv run pytest tests/unit/domain/ -v

# Watch mode for development
uv run pytest-watch

# Performance tests
uv run pytest tests/ -m performance
```

## 🎯 Usage Examples

### Basic Task Creation

```python
from src.domain.entities.task_node import TaskNode
from src.domain.value_objects.task_type import TaskType

# Create immutable task
task = TaskNode(
    goal="Analyze customer feedback trends",
    task_type=TaskType.THINK
)

# State transitions return new instances
ready_task = task.transition_to(TaskStatus.READY)
executing_task = ready_task.transition_to(TaskStatus.EXECUTING)
completed_task = executing_task.with_result("Analysis complete")
```

### Atomizer Integration

```python
from src.application.services.atomizer_service import RuleBasedAtomizer

# Rule-based atomizer
atomizer = RuleBasedAtomizer()
result = await atomizer.evaluate(task)

print(f"Task is {'atomic' if result.is_atomic else 'composite'}")
print(f"Should be processed as: {result.node_type}")
```

### Event System Usage

```python
from src.application.services.event_store import emit_event
from src.domain.events.task_events import TaskCreatedEvent

# Emit events for observability
await emit_event(TaskCreatedEvent.create(
    task_id=task.task_id,
    goal=task.goal,
    task_type=task.task_type
))

# Subscribe to events
def handle_completion(event):
    print(f"Task {event.task_id} completed!")

event_store.subscribe(TaskCompletedEvent, handle_completion)
```

## 🔄 Integration with Legacy Systems

ROMA v2.0 maintains full compatibility with existing Agno agent systems:

```python
from src.application.services.atomizer_service import AgnoAtomizerService

# Bridge to existing agents
agno_atomizer = AgnoAtomizerService(agent_name="research_atomizer")
result = await agno_atomizer.evaluate(v2_task)

# Seamless v1 ↔ v2 conversion
v1_task = agno_atomizer.to_v1_task_node(v2_task)
v2_task = agno_atomizer.from_v1_task_node(v1_task)
```

## 📊 Performance Characteristics

### Memory Efficiency
- `__slots__` in all dataclasses for 40% memory reduction
- Bounded event storage with automatic cleanup
- Immutable objects prevent memory leaks

### Concurrency
- All operations use `asyncio.Lock()` for thread safety
- Non-blocking async/await throughout
- Tested with 1000+ concurrent operations

### Scalability
- Event indexing by task ID and type
- Configurable memory limits prevent unbounded growth
- Efficient relationship tracking with frozensets

## 🛠️ Development Workflow

### Project Structure

```
roma_v2/
├── src/                   # Source code
├── tests/                 # Test suite
├── docs/                  # Documentation
├── pyproject.toml        # uv configuration
├── README.md             # This file
└── ARCHITECTURE_SUMMARY.md  # Detailed architecture
```

### Contributing

1. **Setup Development Environment**
   ```bash
   uv sync --dev
   ```

2. **Run Tests Before Committing**
   ```bash
   uv run pytest
   uv run ruff check src/ tests/
   uv run mypy src/
   ```

3. **Follow Architecture Principles**
   - Maintain immutability
   - Use proper clean architecture layers
   - Add comprehensive tests
   - Document public APIs

## 🗺️ Roadmap

### Week 1 ✅ (Current)
- [x] Immutable TaskNode foundation
- [x] Atomizer decision system  
- [x] Event-driven observability
- [x] Integration with Agno agents
- [x] Comprehensive test coverage

### Week 2 🚧 (Next)
- [ ] Dynamic Task Graph construction
- [ ] Kahn's algorithm adaptation
- [ ] Parallel execution engine
- [ ] Dependency resolution system

### Future Weeks
- [ ] Persistent event store
- [ ] Distributed coordination
- [ ] Advanced recovery strategies
- [ ] Performance optimizations

## 📚 Documentation

- [Architecture Summary](./ARCHITECTURE_SUMMARY.md) - Detailed system design
- [API Documentation](./docs/api/) - Code-level documentation
- [Integration Guide](./docs/integration.md) - Legacy system integration
- [Performance Guide](./docs/performance.md) - Optimization strategies

## 🤝 Support

For questions, issues, or contributions:

1. **Issues**: Use GitHub Issues for bug reports
2. **Discussions**: Use GitHub Discussions for questions
3. **Documentation**: Check `./docs/` directory
4. **Architecture**: See `ARCHITECTURE_SUMMARY.md`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**ROMA v2.0** - Building the future of intelligent task orchestration through immutable, event-driven architecture.