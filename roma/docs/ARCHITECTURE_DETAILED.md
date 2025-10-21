# ROMA v2.0 - Detailed Architecture with UML

## Table of Contents
1. [System Overview](#system-overview)
2. [Production Architecture Fixes](#production-architecture-fixes)
3. [Class Diagrams](#class-diagrams)
4. [Object Responsibilities](#object-responsibilities)
5. [Sequence Diagrams](#sequence-diagrams)
6. [Component Architecture](#component-architecture)
7. [State Machines](#state-machines)
8. [Interaction Patterns](#interaction-patterns)
9. [Activity Diagrams](#activity-diagrams)

## System Overview

ROMA v2.0 is now production-ready with comprehensive architecture fixes implemented as of September 23, 2025. All critical system bugs have been resolved, thread safety ensured, memory management optimized, and clean architecture principles maintained.

### High-Level Component Diagram (Actual Implementation)

```mermaid
graph TB
    subgraph "Presentation Layer (Future)"
        API[REST API]
        WS[WebSocket Server]
        CLI[CLI Interface]
    end
    
    subgraph "Application Layer ✅"
        SM[SystemManager]
        ARS[AgentRuntimeService]
        CBS[ContextBuilderService]
        ES[EventStore]
        GTS[GraphTraversalService]
        GSM[GraphStateManager]
        PEE[ParallelExecutionEngine]
        RM[RecoveryManager]
        AS[ArtifactService]
        DV[DependencyValidator]
    end
    
    subgraph "Domain Layer ✅"
        TN[TaskNode]
        DTG[DynamicTaskGraph]
        TT[TaskType]
        TST[TaskStatus]
        DS[DependencyStatus]
        EV[DomainEvents]
        IA[ImageArtifact]
        BA[BaseArtifact]
        RE[ResultEnvelope]
    end
    
    subgraph "Infrastructure Layer ✅"
        ATM[AgnoToolkitManager]
        BAT[BaseAgnoToolkit]
        LS[LocalStorage]
        RC[ROMAConfig]
        HA[HydraIntegration]
    end
    
    subgraph "Infrastructure Layer ✅"
        PG[(PostgreSQL)]
        RD[(Redis)]
        S3[(S3/MinIO)]
        LF[Langfuse]
        DCM[DatabaseConnectionManager]
        PSE[PostgreSQLEventStore]
        MIG[AlembicMigrations]
    end
    
    API -.-> SM
    WS -.-> SM
    CLI -.-> SM
    
    SM --> ARS
    SM --> CBS
    SM --> DTG
    SM --> ATM
    SM --> RM
    
    ARS --> TN
    CBS --> IA
    DTG --> TN
    ATM --> BAT
    
    GSM --> DTG
    PEE --> TN
    
    SM --> ES
    SM --> GTS
    SM --> PG
    SM --> DCM
    ES --> PSE
    PSE --> PG
    DCM --> PG
```

## Production Architecture Fixes

### Critical System Improvements (September 23, 2025)

ROMA v2.0 underwent comprehensive architecture fixes to ensure production readiness. All critical bugs that would prevent the system from functioning have been resolved.

#### 📋 **Fix Summary: 11 Critical Bugs Resolved**

| **Component** | **Issue** | **Fix** | **Impact** |
|---------------|-----------|---------|------------|
| GraphStateManager | Using non-existent `event_store` | Changed to `event_publisher` | ✅ Event system working |
| TaskNodeProcessor | Undefined `execution_id` variables | Use `self.state.execution_id` | ✅ Execution isolation |
| Agent Services | Optional `execution_id` | Made required parameter | ✅ Session isolation |
| SystemManager | Wrong orchestrator references | Fixed variable references | ✅ Cleanup working |
| ParallelExecutionEngine | Race conditions in stats | Added `_stats_lock` | ✅ Thread safety |
| SystemManager | Exception handling gaps | DRY helper methods | ✅ Error resilience |
| ExecutionOrchestrator | Dictionary complexity | Single `execution_state` | ✅ Performance |

#### 🏗️ **Architectural Improvements**

##### **Thread Safety**
- **Problem**: Race conditions in shared statistics updates
- **Solution**: Added `asyncio.Lock` protection for all concurrent operations
- **Result**: True parallel execution without data corruption

##### **Memory Management**
- **Problem**: Memory leaks in execution context cleanup
- **Solution**: Proper cleanup in `finally` blocks with error handling
- **Result**: Safe for long-running production environments

##### **Error Resilience**
- **Problem**: Generic exception handling with undefined variables
- **Solution**: Robust error handling with fallbacks and helper methods
- **Result**: Graceful degradation instead of crashes

##### **Execution Isolation**
- **Problem**: Optional `execution_id` causing session mixing
- **Solution**: Required `execution_id` propagation throughout system
- **Result**: Perfect isolation between concurrent executions

##### **Code Quality (DRY)**
- **Problem**: Duplicated response building logic
- **Solution**: Extracted helper methods (`_build_execution_result`, `_build_error_result`)
- **Result**: Maintainable and consistent error handling

##### **Performance Optimization**
- **Problem**: Dictionary overhead in single-execution orchestrator
- **Solution**: Simplified to single `execution_state` property
- **Result**: Eliminated unnecessary hash map operations

#### 🔧 **Technical Implementation Details**

##### **Thread-Safe Statistics (ParallelExecutionEngine)**
```python
# Before: Race conditions
self._total_batches_processed += 1
self._total_nodes_processed += len(executable_nodes)

# After: Atomic updates
async with self._stats_lock:
    self._total_batches_processed += 1
    self._total_nodes_processed += len(executable_nodes)
```

##### **DRY Exception Handling (SystemManager)**
```python
# Before: Duplicated response building
return {"execution_id": execution_id, "task": task, ...}

# After: Helper methods
return self._build_execution_result(execution_id, task, ...)
return self._build_error_result(execution_id, task, error, ...)
```

##### **Simplified State Management (ExecutionOrchestrator)**
```python
# Before: Dictionary complexity
self.execution_states: Dict[str, ExecutionState] = {}
state = self.execution_states[execution_id]

# After: Single state
self.execution_state: Optional[ExecutionState] = None
state = self.execution_state
```

#### 🎯 **Production Readiness Impact**

| **Aspect** | **Before** | **After** | **Benefit** |
|------------|------------|-----------|-------------|
| **Concurrency** | Race conditions | Thread-safe locks | Safe parallel execution |
| **Memory** | Memory leaks | Proper cleanup | Production stability |
| **Errors** | System crashes | Graceful handling | Reliability |
| **Isolation** | Task mixing | Perfect isolation | Data integrity |
| **Performance** | Dictionary overhead | Direct access | Faster execution |
| **Maintainability** | Code duplication | DRY patterns | Easy maintenance |

#### ✅ **Validation Results**

All fixes have been validated through:
- **Static Analysis**: No more undefined variables or type errors
- **Code Review**: SOLID principles maintained
- **Architecture Review**: Clean separation of concerns preserved
- **Performance Analysis**: Removed unnecessary complexity

**System Status**: 🟢 **PRODUCTION READY**

## Class Diagrams

### Core Domain Entities

```mermaid
classDiagram
    class TaskNode {
        <<frozen>>
        +str task_id
        +str goal
        +TaskType task_type
        +NodeType node_type
        +TaskStatus status
        +str parent_id
        +int layer
        +Dict result
        +datetime created_at
        +datetime started_at
        +datetime completed_at
        +int retry_count
        +int max_retries
        +int version
        
        +transition_to(status) TaskNode
        +with_result(result) TaskNode
        +with_error(error) TaskNode
        +generate_hash() str
        +increment_retry() TaskNode
        +can_retry bool
        +retry_exhausted bool
        -_is_valid_transition(from, to) bool
    }
    
    class TaskType {
        <<enumeration>>
        RETRIEVE
        WRITE
        THINK
        CODE_INTERPRET
        IMAGE_GENERATION
    }
    
    class NodeType {
        <<enumeration>>
        PLAN
        EXECUTE
    }
    
    class TaskStatus {
        <<enumeration>>
        PENDING
        READY
        EXECUTING
        COMPLETED
        FAILED
    }
    
    class TaskGraph {
        -Dict~str,TaskNode~ _nodes
        -Set~Tuple~ _edges
        -asyncio.Lock _lock
        
        +add_node(node) None
        +add_edge(from_id, to_id) None
        +get_node(task_id) TaskNode
        +get_children(task_id) List~TaskNode~
        +get_ready_nodes() List~TaskNode~
        +update_node(node) None
        +get_lineage(task_id) List~TaskNode~
        -_has_dependencies_met(node) bool
    }
    
    TaskNode --> TaskType
    TaskNode --> NodeType
    TaskNode --> TaskStatus
    TaskGraph "1" --> "*" TaskNode
```

### Agent Hierarchy (Framework-Agnostic Design)

```mermaid
classDiagram
    class BaseAgent {
        <<abstract>>
        +str name
        +str type
        +TaskType task_type
        +ModelConfig model
        +Dict config
        
        +process(task, context)* Any
        +validate_input(task) bool
        +prepare_prompt(task, context) str
    }
    
    class AtomizerAgent {
        <<abstract>>
        +process(task, context) AtomizerResult
        #evaluate_complexity(task) int
        #check_atomicity(task) bool
    }
    
    class PlannerAgent {
        <<abstract>>
        +process(task, context) List~TaskNode~
        #decompose_task(task) List~Dict~
        #validate_plan(tasks) bool
    }
    
    class ExecutorAgent {
        <<abstract>>
        +List~Tool~ tools
        +process(task, context) ExecutionResult
        #select_tool(task) Tool
        #execute_with_tool(tool, task) Any
    }
    
    class AggregatorAgent {
        <<abstract>>
        +process(results, context) AggregatedResult
        #combine_results(results) Dict
        #synthesize(combined) str
    }
    
    class PlanModifierAgent {
        <<abstract>>
        +process(plan, feedback) List~TaskNode~
        #analyze_feedback(feedback) Dict
        #adjust_plan(plan, analysis) List~TaskNode~
    }
    
    %% Framework-specific implementations
    class AgnoAtomizerImpl {
        -agno_runtime AgnoRuntime
        +process(task, context) AtomizerResult
    }
    
    class AgnoExecutorImpl {
        -toolkit_registry ToolkitRegistry
        +process(task, context) ExecutionResult
    }
    
    class AgnoPlannerImpl {
        -agno_runtime AgnoRuntime
        +process(task, context) List~TaskNode~
    }
    
    BaseAgent <|-- AtomizerAgent
    BaseAgent <|-- PlannerAgent
    BaseAgent <|-- ExecutorAgent
    BaseAgent <|-- AggregatorAgent
    BaseAgent <|-- PlanModifierAgent
    
    AtomizerAgent <|-- AgnoAtomizerImpl
    ExecutorAgent <|-- AgnoExecutorImpl
    PlannerAgent <|-- AgnoPlannerImpl
    
    note for BaseAgent "Framework-agnostic base"
    note for AgnoAtomizerImpl "Default Agno implementation"
    note for AtomizerAgent "Easy to extend with other frameworks"
```

### Service Layer

```mermaid
classDiagram
    class ExecutionOrchestrator {
        -TaskGraph graph
        -TaskScheduler scheduler
        -AgentRegistry registry
        -EventStore events
        
        +execute_goal(goal) ExecutionResult
        +process_node(node) None
        -handle_atomizer_result(node, result) None
        -handle_planner_result(node, tasks) None
        -handle_executor_result(node, result) None
        -handle_aggregator_result(node, result) None
    }
    
    class TaskScheduler {
        -TaskGraph graph
        -Semaphore semaphore
        -Set~str~ processing
        
        +get_ready_nodes() List~TaskNode~
        +mark_processing(task_id) None
        +mark_complete(task_id) None
        -is_ready(node) bool
    }
    
    class AgentRegistry {
        -Dict agents
        -ProfileConfig profile
        -AgentFrameworkAdapter adapter
        
        +get_atomizer(task_type) AtomizerAgent
        +get_planner(task_type) PlannerAgent
        +get_executor(task_type) ExecutorAgent
        +get_aggregator(task_type) AggregatorAgent
        +register_agent(agent) None
        +set_framework_adapter(adapter) None
    }
    
    class AgentFrameworkAdapter {
        <<interface>>
        +create_agent(config) BaseAgent
        +execute_agent(agent, context) Result
    }
    
    class AgnoAdapter {
        +create_agent(config) BaseAgent
        +execute_agent(agent, context) Result
    }
    
    class FutureFrameworkAdapter {
        <<placeholder>>
        +create_agent(config) BaseAgent
        +execute_agent(agent, context) Result
    }
    
    AgentRegistry --> AgentFrameworkAdapter
    AgnoAdapter ..|> AgentFrameworkAdapter
    FutureFrameworkAdapter ..|> AgentFrameworkAdapter
    
    note for AgentFrameworkAdapter "Extensible interface for any agent framework"
    note for FutureFrameworkAdapter "Easy to add LangGraph, CrewAI, etc."
    
    class ContextManager {
        -KnowledgeStore knowledge
        -TaskGraph graph
        
        +build_context(task) TaskContext
        -get_lineage_results(task) List~Dict~
        -get_sibling_results(task) List~Dict~
        -get_relevant_knowledge(task) Dict
    }
    
    ExecutionOrchestrator --> TaskScheduler
    ExecutionOrchestrator --> AgentRegistry
    ExecutionOrchestrator --> TaskGraph
    TaskScheduler --> TaskGraph
    ContextManager --> TaskGraph
```

## Object Responsibilities

### Core Domain Objects

#### TaskNode
**Responsibility**: Immutable representation of a single task
- **Does**: 
  - Maintains task state and metadata
  - Validates state transitions
  - Generates deterministic hash for caching
  - Creates new instances for state changes (immutability)
- **Doesn't**: 
  - Execute the task
  - Manage relationships with other tasks
  - Persist itself
- **Collaborators**: TaskGraph, ExecutionOrchestrator

#### TaskGraph
**Responsibility**: Thread-safe management of task relationships
- **Does**:
  - Maintains DAG structure
  - Tracks parent-child relationships
  - Identifies ready tasks (dependencies met)
  - Provides lineage queries
- **Doesn't**:
  - Execute tasks
  - Make atomizer decisions
  - Handle persistence
- **Collaborators**: TaskNode, TaskScheduler, ContextManager

### Service Layer Objects

#### ExecutionOrchestrator
**Responsibility**: High-level coordination of task execution
- **Does**:
  - Orchestrates the execution flow
  - Delegates to appropriate agents
  - Manages execution lifecycle
  - Handles error recovery
- **Doesn't**:
  - Make atomizer decisions
  - Execute tasks directly
  - Manage task scheduling details
- **Collaborators**: TaskScheduler, AgentRegistry, EventStore

#### TaskScheduler
**Responsibility**: Concurrent task scheduling with dependency management
- **Does**:
  - Identifies ready tasks
  - Controls concurrency limits
  - Tracks processing state
  - Implements modified Kahn's algorithm
- **Doesn't**:
  - Execute tasks
  - Make planning decisions
  - Handle errors
- **Collaborators**: TaskGraph, ExecutionOrchestrator

#### AgentRegistry
**Responsibility**: Agent selection and management
- **Does**:
  - Selects appropriate agents based on task type
  - Manages agent lifecycle
  - Applies profile configurations
  - Validates agent capabilities
- **Doesn't**:
  - Execute tasks
  - Make atomizer decisions
  - Handle task scheduling
- **Collaborators**: BaseAgent subclasses, ProfileConfig

#### ContextManager
**Responsibility**: Building rich context for task execution
- **Does**:
  - Aggregates lineage results
  - Collects sibling outputs
  - Retrieves relevant knowledge
  - Prepares execution context
- **Doesn't**:
  - Execute tasks
  - Make decisions
  - Modify task state
- **Collaborators**: TaskGraph, KnowledgeStore

#### RecoveryManager
**Responsibility**: Centralized error handling and recovery strategies
- **Does**:
  - Implements circuit breaker pattern
  - Manages retry strategies with exponential backoff
  - Escalates failures through RETRY → REPLAN → FORCE_ATOMIC → FAIL
  - Tracks failure statistics and circuit breaker states
  - Provides intelligent recovery actions based on error type
- **Doesn't**:
  - Execute tasks directly
  - Make business decisions
  - Modify task state (delegates to SystemManager)
- **Collaborators**: SystemManager, TaskNode, EventStore

#### ArtifactService
**Responsibility**: Managing result artifacts and their lifecycle
- **Does**:
  - Stores artifacts from ResultEnvelope using dependency injection
  - Provides artifact retrieval (binary and text formats)
  - Manages execution artifact cleanup and organization
  - Generates storage keys with hierarchical structure (executions/tasks/artifacts)
  - Creates FileArtifacts from stored data for reuse
  - Provides storage statistics and metadata
- **Doesn't**:
  - Generate artifacts (receives from agents)
  - Make storage backend decisions (uses injected storage)
  - Handle agent execution logic
- **Collaborators**: StorageInterface, ResultEnvelope, BaseArtifact, SystemManager

### Domain Objects

#### ResultEnvelope
**Responsibility**: Standardized wrapper for agent execution results
- **Does**:
  - Encapsulates agent results with rich metadata
  - Integrates existing domain types (AgentType, MediaType, BaseArtifact)
  - Provides execution metrics (tokens, time, cost)
  - Supports generic typing for different result types
  - Tracks artifacts generated during task execution
  - Maintains traceability with task and execution IDs
- **Doesn't**:
  - Execute tasks or generate results
  - Manage artifact storage (delegates to ArtifactService)
  - Handle agent communication
- **Collaborators**: ExecutorResult, ExecutionMetrics, BaseArtifact, AgentType

### Agent Objects

#### AtomizerAgent
**Responsibility**: Determine if task needs decomposition
- **Does**:
  - Evaluates task complexity
  - Decides PLAN vs EXECUTE
  - Applies complexity thresholds
  - Uses LLM for decision making
- **Doesn't**:
  - Create plans
  - Execute tasks
  - Aggregate results
- **Collaborators**: ExecutionOrchestrator, LLM

#### PlannerAgent
**Responsibility**: Decompose complex tasks into subtasks
- **Does**:
  - Creates task decomposition
  - Ensures MECE compliance
  - Validates plan feasibility
  - Generates child TaskNodes
- **Doesn't**:
  - Execute tasks
  - Make atomizer decisions
  - Aggregate results
- **Collaborators**: ExecutionOrchestrator, TaskNode

#### ExecutorAgent
**Responsibility**: Execute atomic tasks using tools
- **Does**:
  - Selects appropriate tools
  - Executes atomic operations
  - Handles tool errors
  - Returns execution results
- **Doesn't**:
  - Decompose tasks
  - Make planning decisions
  - Aggregate results
- **Collaborators**: Tool, ExecutionOrchestrator

## Sequence Diagrams

### Main Execution Flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as API
    participant EO as ExecutionOrchestrator
    participant TS as TaskScheduler
    participant ARS as AgentRuntimeService
    participant TG as TaskGraph
    
    U->>API: execute_goal("Research AI trends")
    API->>EO: execute_goal(goal)
    EO->>TG: create_root_task()
    TG-->>EO: root_node
    
    loop Until all tasks complete
        EO->>TS: get_ready_nodes()
        TS->>TG: find_ready_tasks()
        TG-->>TS: [ready_nodes]
        TS-->>EO: ready_nodes
        
        par For each ready node
            EO->>ARS: get_agent(task_type, agent_type)
            ARS-->>EO: agent
            EO->>ARS: execute_agent(agent, node)
            ARS-->>EO: result
            EO->>TG: update_node(result)
        end
        
        opt Has completed children
            EO->>ARS: get_agent(task_type, AGGREGATOR)
            ARS-->>EO: aggregator_agent
            EO->>ARS: execute_agent(aggregator_agent, parent_node)
            ARS-->>EO: aggregated_result
            EO->>TG: update_parent(result)
        end
    end
    
    EO-->>API: final_result
    API-->>U: response
```

### Direct Agent Execution Process

```mermaid
sequenceDiagram
    participant EO as ExecutionOrchestrator
    participant AS as AtomizerService
    participant CM as ContextManager
    participant LLM as LLM
    participant AR as AgentRegistry
    
    EO->>AS: evaluate(task_node)
    AS->>AR: get_atomizer(task_type)
    AR-->>AS: atomizer_agent
    
    AS->>CM: build_context(task_node)
    CM-->>AS: context
    
    AS->>AS: evaluate_complexity(task)
    
    alt Simple heuristic check
        AS-->>EO: EXECUTE (simple task)
    else Needs LLM evaluation
        AS->>LLM: evaluate_with_prompt(task, context)
        LLM-->>AS: complexity_analysis
        
        alt is_atomic
            AS-->>EO: EXECUTE
        else needs_decomposition
            AS-->>EO: PLAN
        end
    end
```

### Agent-Based Planning and Decomposition

```mermaid
sequenceDiagram
    participant EO as ExecutionOrchestrator
    participant PS as PlannerService
    participant AR as AgentRegistry
    participant LLM as LLM
    participant TG as TaskGraph
    participant V as Validator
    
    EO->>PS: create_plan(parent_node, context)
    PS->>AR: get_planner(task_type)
    AR-->>PS: planner_agent
    
    PS->>LLM: generate_decomposition(task, context)
    LLM-->>PS: task_decomposition
    
    PS->>V: validate_mece(decomposition)
    V-->>PS: validation_result
    
    alt Valid decomposition
        loop For each subtask
            PS->>PS: create_task_node(subtask)
            PS->>TG: add_node(child_node)
            PS->>TG: add_edge(parent, child)
        end
        PS-->>EO: [child_nodes]
    else Invalid decomposition
        PS->>LLM: retry_with_feedback(feedback)
        LLM-->>PS: revised_decomposition
        PS->>PS: validate_and_create_nodes()
        PS-->>EO: [child_nodes]
    end
```

### Artifact Management Flow

```mermaid
sequenceDiagram
    participant EA as ExecutorAgent
    participant SM as SystemManager
    participant AS as ArtifactService
    participant ST as Storage
    participant RE as ResultEnvelope
    participant BA as BaseArtifact

    Note over EA: Task execution completes
    EA->>BA: create artifact (FileArtifact, ImageArtifact)
    BA-->>EA: artifact instance

    EA->>RE: create_success(result, artifacts, metrics)
    RE-->>EA: result envelope

    EA-->>SM: return ResultEnvelope

    Note over SM: Store execution results
    SM->>AS: store_envelope_artifacts(execution_id, envelope)

    loop For each artifact in envelope
        AS->>BA: get_content() from artifact
        BA-->>AS: artifact content (bytes)
        AS->>AS: generate_artifact_key(execution_id, task_id, artifact)
        AS->>ST: put_text/put_binary(key, content)
        ST-->>AS: storage_reference
    end

    AS-->>SM: [storage_references]
    SM->>SM: log execution completion

    Note over AS,ST: Later retrieval
    SM->>AS: retrieve_artifact(storage_ref)
    AS->>ST: get(storage_ref)
    ST-->>AS: content
    AS-->>SM: artifact content

    Note over AS: Artifact lifecycle management
    SM->>AS: list_execution_artifacts(execution_id)
    AS->>ST: list(execution_prefix)
    ST-->>AS: artifact_list
    AS-->>SM: filtered artifacts

    SM->>AS: cleanup_execution_artifacts(execution_id)
    loop For each artifact
        AS->>ST: delete(artifact_key)
    end
    AS-->>SM: deleted_count
```

## Component Architecture

### Layered Architecture with Dependencies

```mermaid
graph TD
    subgraph "Presentation Layer"
        REST[REST API<br/>FastAPI]
        GQL[GraphQL API<br/>Strawberry]
        WS[WebSocket<br/>Socket.io]
        CLI[CLI<br/>Click]
    end

    subgraph "Application Layer"
        subgraph "Orchestration"
            EO[ExecutionOrchestrator]
            TS[TaskScheduler]
            DD[DeadlockDetector]
            RM[RecoveryManager]
        end

        subgraph "Services"
            AS[AtomizerService]
            PS[PlannerService]
            ES[ExecutorService]
            AGS[AggregatorService]
            CMS[ContextService]
            ARS[ArtifactService]
        end

        subgraph "Managers"
            CM[ConfigManager]
            EM[EventManager]
            MM[MetricsManager]
        end
    end

    subgraph "Domain Layer"
        subgraph "Entities"
            TN[TaskNode]
            TG[TaskGraph]
        end

        subgraph "Value Objects"
            TT[TaskType]
            NT[NodeType]
            TST[TaskStatus]
            TC[TaskContext]
            RE[ResultEnvelope]
            DC[DatabaseConfig]
        end

        subgraph "Events"
            TE[TaskEvents]
            SE[SystemEvents]
        end
    end

    subgraph "Infrastructure Layer"
        subgraph "Persistence ✅"
            PGR[PostgreSQL<br/>Event Store]
            DCM[Connection<br/>Manager]
            RDC[Redis<br/>Cache]
            S3S[S3<br/>Storage]
            MIG[Alembic<br/>Migrations]
            EM[Event Models]
            TEM[Task Execution Models]
            CM[Checkpoint Models]
        end

        subgraph "External Services"
            LLM[LLM<br/>Providers]
            TOOLS[Agno<br/>Tools]
            OBS[Langfuse<br/>Observability]
        end
    end

    REST --> EO
    GQL --> EO
    WS --> EO
    CLI --> EO

    EO --> TS
    EO --> AS
    EO --> RM

    TS --> DD
    TS --> TG

    AS --> LLM
    PS --> LLM
    ES --> TOOLS

    AS --> TN
    PS --> TN

    EM --> TE
    EM --> SE

    TG --> PGR
    CMS --> RDC
    ES --> S3S
    ES --> PGR
    PGR --> DCM
    DCM --> MIG
    DCM --> DC
    PGR --> EM
    PGR --> TEM
    PGR --> CM

    EO --> OBS
```

## State Machines

### TaskNode State Transitions

```mermaid
stateDiagram-v2
    [*] --> PENDING: Created
    
    PENDING --> READY: Dependencies Met
    PENDING --> FAILED: Validation Error
    
    READY --> RUNNING: Scheduler Picks Up
    READY --> FAILED: Pre-execution Error
    
    RUNNING --> PLAN_DONE: Planning Complete
    RUNNING --> DONE: Execution Complete
    RUNNING --> FAILED: Execution Error
    
    PLAN_DONE --> AGGREGATING: Children Complete
    
    AGGREGATING --> DONE: Aggregation Complete
    AGGREGATING --> FAILED: Aggregation Error
    
    FAILED --> READY: Retry
    FAILED --> [*]: Max Retries
    
    DONE --> [*]: Terminal
```

### Execution Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Initialization
    
    Initialization --> TaskCreation: Goal Received
    
    TaskCreation --> Atomization: Root Task Created
    
    Atomization --> Planning: Needs Decomposition
    Atomization --> Execution: Is Atomic
    
    Planning --> TaskCreation: Child Tasks Created
    
    Execution --> ResultProcessing: Task Complete
    
    ResultProcessing --> Aggregation: Has Children
    ResultProcessing --> Completion: No Children
    
    Aggregation --> ResultProcessing: Parent Updated
    
    Completion --> [*]: All Tasks Done
```

## Interaction Patterns

### Event-Driven Communication

```mermaid
graph LR
    subgraph "Event Producers"
        TN[TaskNode]
        EO[ExecutionOrchestrator]
        ARS[AgentRuntimeService]
    end
    
    subgraph "Event Bus"
        EB[EventBus<br/>AsyncIO Queue]
    end
    
    subgraph "Event Consumers"
        EL[EventLogger]
        MM[MetricsManager]
        WS[WebSocket Notifier]
        PS[PersistenceService]
        OB[Observability Service]
    end
    
    TN -->|StateChanged| EB
    EO -->|ExecutionStarted| EB
    ARS -->|AgentExecuted| EB
    
    EB -->|TaskEvents| EL
    EB -->|Metrics| MM
    EB -->|Updates| WS
    EB -->|Persist| PS
    EB -->|Trace| OB
```

### Agno Toolkit Integration Pattern

```mermaid
classDiagram
    class ExecutorAgent {
        +TaskType task_type
        +ToolkitRegistry registry
        +List~Tool~ tools
        +execute(task, context)
        -select_tool(task, context)
    }
    
    class Toolkit {
        <<abstract>>
        +Dict config
        +List~Tool~ tools
        +register_tool(tool)
        +get_tools() List~Tool~
        #_setup_tools()*
    }
    
    class Tool {
        <<Agno Interface>>
        +str name
        +str description
        +List~ToolParameter~ parameters
        +Callable handler
        +execute(**kwargs) Any
    }
    
    class ToolParameter {
        +str name
        +str type
        +str description
        +bool required
        +Any default
    }
    
    class WebSearchToolkit {
        -google_search()
        -exa_search()
        -tavily_search()
    }
    
    class CodeExecutionToolkit {
        -execute_python()
        -execute_javascript()
        -setup_e2b_sandbox()
    }
    
    class BinanceToolkit {
        -get_market_data()
        -get_order_book()
        -get_trading_history()
    }
    
    class CoinGeckoToolkit {
        -get_token_info()
        -get_market_cap()
        -get_trending()
    }
    
    class ToolkitRegistry {
        -Dict~str,Toolkit~ toolkits
        +register(name, toolkit)
        +get_toolkit(name) Toolkit
        +get_tools_for_task_type(type) List~Tool~
    }
    
    ExecutorAgent --> ToolkitRegistry
    ToolkitRegistry --> Toolkit
    Toolkit --> Tool
    Tool --> ToolParameter
    Toolkit <|-- WebSearchToolkit
    Toolkit <|-- CodeExecutionToolkit
    Toolkit <|-- BinanceToolkit
    Toolkit <|-- CoinGeckoToolkit
```

> **Note**: All toolkits follow the Agno toolkit interface specification. See [TOOLKIT_ARCHITECTURE.md](TOOLKIT_ARCHITECTURE.md) for detailed implementation.

## Activity Diagrams

### Overall Task Processing Workflow

```mermaid
graph TD
    Start([User Goal]) --> CreateRoot[Create Root Task]
    CreateRoot --> AddToGraph[Add to Task Graph]
    
    AddToGraph --> CheckReady{Ready Tasks?}
    
    CheckReady -->|Yes| SelectTask[Select Next Task]
    CheckReady -->|No| CheckComplete{All Complete?}
    
    SelectTask --> Atomize[Atomizer Evaluation]
    
    Atomize --> AtomizeDecision{Atomic?}
    
    AtomizeDecision -->|Yes| Execute[Execute Task]
    AtomizeDecision -->|No| Plan[Create Plan]
    
    Plan --> CreateChildren[Create Child Tasks]
    CreateChildren --> AddToGraph
    
    Execute --> StoreResult[Store Result]
    StoreResult --> UpdateStatus[Update Task Status]
    
    UpdateStatus --> CheckChildren{Has Children?}
    
    CheckChildren -->|Yes| ChildrenComplete{All Children Done?}
    CheckChildren -->|No| CheckReady
    
    ChildrenComplete -->|Yes| Aggregate[Aggregate Results]
    ChildrenComplete -->|No| CheckReady
    
    Aggregate --> StoreResult
    
    CheckComplete -->|Yes| ReturnResult[Return Final Result]
    CheckComplete -->|No| Wait[Wait for Dependencies]
    
    Wait --> CheckReady
    
    ReturnResult --> End([End])
```

### Error Recovery Flow

#### RecoveryManager Integration

```mermaid
sequenceDiagram
    participant SM as SystemManager
    participant RM as RecoveryManager
    participant TN as TaskNode
    participant CB as CircuitBreaker
    participant ES as EventStore
    
    Note over SM: Task execution fails
    SM->>RM: handle_failure(task, error)
    RM->>CB: check_circuit_state()
    CB-->>RM: CLOSED/OPEN/HALF_OPEN
    
    alt Circuit is OPEN
        RM-->>SM: CIRCUIT_BREAK action
        SM->>TN: mark_failed()
    else Circuit allows request
        RM->>RM: analyze_error_type()
        RM->>TN: check_retry_count()
        TN-->>RM: can_retry: bool
        
        alt Can retry (< max_retries)
            RM->>TN: increment_retry()
            RM->>CB: record_success() or record_failure()
            RM-->>SM: RETRY with updated_node
            SM->>SM: schedule_retry_with_backoff()
        else Retries exhausted
            RM->>RM: escalate_recovery()
            alt Error is recoverable
                RM-->>SM: REPLAN action
                SM->>SM: trigger_parent_replan()
            else Critical failure
                RM-->>SM: FORCE_ATOMIC action
                SM->>SM: force_atomic_execution()
            else Fatal error
                RM-->>SM: FAIL_PERMANENTLY action
                SM->>TN: mark_failed_permanently()
            end
        end
    end
    
    RM->>ES: log_recovery_event()
    RM->>CB: update_failure_stats()
```

#### Recovery Strategy Flow

```mermaid
graph TD
    Error([Task Error]) --> RM[RecoveryManager]
    RM --> CB{Circuit Breaker State?}
    
    CB -->|OPEN| CircuitBreak[CIRCUIT_BREAK]
    CB -->|CLOSED/HALF_OPEN| AnalyzeError[Analyze Error Type]
    
    AnalyzeError --> CheckRetries{Can Retry?}
    CheckRetries -->|Yes| Retry[RETRY Action]
    CheckRetries -->|No| Escalate[Escalate Strategy]
    
    Retry --> IncrementCount[Increment Retry Count]
    IncrementCount --> BackoffWait[Exponential Backoff]
    BackoffWait --> RetryExecution[Schedule Retry]
    
    Escalate --> ErrorType{Error Category}
    ErrorType -->|Planning Error| Replan[REPLAN Action]
    ErrorType -->|Resource Error| ForceAtomic[FORCE_ATOMIC Action]
    ErrorType -->|Fatal Error| FailPerm[FAIL_PERMANENTLY Action]
    
    Replan --> TriggerParent[Trigger Parent Replan]
    ForceAtomic --> AtomicExec[Force Atomic Execution]
    FailPerm --> MarkFailed[Mark Failed Permanently]
    CircuitBreak --> MarkFailed
    
    TriggerParent --> ParentHandles[Parent Creates New Plan]
    AtomicExec --> DirectExecution[Execute Without Decomposition]
    MarkFailed --> PropagateFailure[Propagate Up Tree]
```

## Data Flow

### Context Building Data Flow

```mermaid
graph LR
    subgraph "Input Sources"
        TN[Current Task]
        TG[Task Graph]
        KS[Knowledge Store]
        PR[Parent Results]
        SR[Sibling Results]
    end
    
    subgraph "Context Builder"
        CB[ContextManager]
        LQ[Lineage Query]
        SQ[Sibling Query]
        RQ[Relevance Query]
    end
    
    subgraph "Output Context"
        TC[TaskContext]
        OO[Overall Objective]
        LR[Lineage Results]
        SBR[Sibling Results]
        RK[Relevant Knowledge]
        AM[Available Methods]
    end
    
    TN --> CB
    TG --> LQ
    TG --> SQ
    KS --> RQ
    PR --> LQ
    SR --> SQ
    
    LQ --> CB
    SQ --> CB
    RQ --> CB
    
    CB --> TC
    TC --> OO
    TC --> LR
    TC --> SBR
    TC --> RK
    TC --> AM
```

## Concurrency Control

### Semaphore-Based Execution Control

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant SEM as Semaphore(10)
    participant T1 as Task1
    participant T2 as Task2
    participant T3 as Task3
    participant TN as TaskN
    
    S->>SEM: acquire() for Task1
    SEM-->>S: Acquired (9 left)
    S->>T1: execute()
    
    S->>SEM: acquire() for Task2
    SEM-->>S: Acquired (8 left)
    S->>T2: execute()
    
    Note over S,TN: ... 8 more tasks acquire
    
    S->>SEM: acquire() for Task11
    SEM-->>S: Wait (0 available)
    
    T1-->>S: complete()
    S->>SEM: release()
    SEM-->>S: Released (1 available)
    SEM-->>S: Task11 can proceed
    S->>TN: execute()
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "External"
        U[Users]
        LB[Load Balancer]
    end
    
    subgraph "Kubernetes Cluster"
        subgraph "Application Pods"
            API1[API Pod 1]
            API2[API Pod 2]
            API3[API Pod 3]
        end
        
        subgraph "Worker Pods"
            W1[Worker Pod 1]
            W2[Worker Pod 2]
            W3[Worker Pod 3]
        end
        
        subgraph "Data Layer"
            PG[(PostgreSQL)]
            RD[(Redis)]
            MQ[RabbitMQ]
        end
        
        subgraph "Storage"
            S3[(S3/MinIO)]
        end
    end
    
    subgraph "External Services"
        LLM[LLM APIs]
        LF[Langfuse]
        DD[DataDog]
    end
    
    U --> LB
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> MQ
    API2 --> MQ
    API3 --> MQ
    
    MQ --> W1
    MQ --> W2
    MQ --> W3
    
    W1 --> PG
    W2 --> RD
    W3 --> S3
    
    W1 --> LLM
    W2 --> LF
    W3 --> DD
```

## Framework Extensibility Design

### Adapter Pattern for Agent Frameworks

The architecture uses an adapter pattern to decouple agent logic from specific frameworks:

```python
# Abstract interface - framework agnostic
class AgentFrameworkAdapter(Protocol):
    """Interface for integrating different agent frameworks."""
    
    async def create_agent(self, config: AgentConfig) -> BaseAgent:
        """Create an agent instance using the framework."""
        ...
    
    async def execute_agent(self, agent: BaseAgent, context: TaskContext) -> Any:
        """Execute agent using framework's runtime."""
        ...

# Default implementation with Agno
class AgnoAdapter(AgentFrameworkAdapter):
    """Agno framework adapter - current default."""
    
    def __init__(self):
        self.runtime = AgnoRuntime()
        self.toolkit_registry = ToolkitRegistry()
    
    async def create_agent(self, config: AgentConfig) -> BaseAgent:
        # Create Agno-specific agent
        return AgnoAgent(config, self.runtime)
    
    async def execute_agent(self, agent: BaseAgent, context: TaskContext) -> Any:
        # Execute using Agno runtime
        return await self.runtime.run(agent, context)

# Future framework example (not implemented yet)
# class LangGraphAdapter(AgentFrameworkAdapter):
#     """Example of how to add another framework."""
#     pass
```

### Adding New Frameworks

To integrate a new agent framework (e.g., LangGraph, CrewAI, AutoGen):

1. **Implement the Adapter Interface**:
```python
class NewFrameworkAdapter(AgentFrameworkAdapter):
    async def create_agent(self, config: AgentConfig) -> BaseAgent:
        # Framework-specific agent creation
        pass
    
    async def execute_agent(self, agent: BaseAgent, context: TaskContext) -> Any:
        # Framework-specific execution
        pass
```

2. **Register with Agent Registry**:
```python
registry = AgentRegistry()
registry.set_framework_adapter(NewFrameworkAdapter())
```

3. **No Core Changes Required**: The orchestrator and graph remain unchanged.

### Benefits of This Design

- **Framework Independence**: Core logic doesn't depend on any specific framework
- **Easy Migration**: Can switch frameworks without rewriting core logic
- **Gradual Adoption**: Can use different frameworks for different agent types
- **Testing**: Can mock framework adapters for unit tests
- **Future-Proof**: Ready for emerging agent frameworks

## Usage Examples

### Working with ResultEnvelope and ArtifactService

#### Creating a ResultEnvelope with Artifacts

```python
from src.roma.domain.value_objects.result_envelope import ResultEnvelope
from src.roma.domain.value_objects.agent_responses import ExecutorResult
from src.roma.domain.value_objects.execution_metrics import ExecutionMetrics
from src.roma.domain.entities.artifacts.file_artifact import FileArtifact
from src.roma.domain.value_objects.agent_type import AgentType

# Create execution metrics
metrics = ExecutionMetrics(
    execution_time=2.5,
    tokens_used=1500,
    model_calls=3,
    cost_estimate=0.05
)

# Create file artifact
document = FileArtifact.from_path(
    name="research_summary",
    file_path="/tmp/summary.md",
    task_id="task-123",
    metadata={"source": "web_research", "format": "markdown"}
)

# Create executor result
result = ExecutorResult(
    result="Completed comprehensive research on AI trends",
    sources=["openai.com", "arxiv.org", "techcrunch.com"],
    success=True,
    confidence=0.92
)

# Wrap in ResultEnvelope
envelope = ResultEnvelope.create_success(
    result=result,
    task_id="task-123",
    execution_id="exec-456",
    agent_type=AgentType.EXECUTOR,
    execution_metrics=metrics,
    artifacts=[document],
    output_text="Research completed successfully"
)
```

#### Using ArtifactService for Artifact Management

```python
from src.roma.application.services.artifact_service import ArtifactService
from src.roma.infrastructure.storage.local_storage import LocalFileStorage
from src.roma.infrastructure.storage.storage_interface import StorageConfig

# Initialize storage and service
config = StorageConfig.from_mount_path("/data/roma")
storage = LocalFileStorage(config)
artifact_service = ArtifactService(storage)
await artifact_service.initialize()

# Store artifacts from ResultEnvelope
storage_refs = await artifact_service.store_envelope_artifacts(
    execution_id="exec-456",
    envelope=envelope
)
print(f"Stored artifacts: {storage_refs}")
# Output: ['executions/exec-456/tasks/task-123/artifact-id/research_summary.md']

# Retrieve artifact content
content = await artifact_service.retrieve_artifact(
    storage_refs[0],
    as_text=True
)
print(f"Content: {content}")

# List all artifacts for execution
all_artifacts = await artifact_service.list_execution_artifacts("exec-456")
print(f"All artifacts: {all_artifacts}")

# Get artifact metadata
metadata = await artifact_service.get_artifact_metadata(storage_refs[0])
print(f"Size: {metadata['size_bytes']} bytes")
print(f"Exists: {metadata['exists']}")

# Clean up when done
deleted = await artifact_service.cleanup_execution_artifacts("exec-456")
print(f"Cleaned up {deleted} artifacts")
```

#### Integration with SystemManager

```python
# Inside SystemManager - automatic artifact handling
class SystemManager:
    async def _process_executor_result(self, envelope: ResultEnvelope) -> None:
        # Store artifacts automatically
        storage_refs = await self._artifact_service.store_envelope_artifacts(
            self._execution_id, envelope
        )

        # Log storage references for retrieval
        await self._event_store.emit_event(
            ArtifactsStoredEvent(
                execution_id=self._execution_id,
                task_id=envelope.task_id,
                storage_references=storage_refs
            )
        )

        # Continue with result processing...
```

#### Error Handling and Recovery

```python
# Robust artifact operations with error handling
async def safe_artifact_operations():
    try:
        # Store with automatic retry on transient failures
        storage_refs = await artifact_service.store_envelope_artifacts(
            "exec-789", envelope
        )

        # Verify storage succeeded
        for ref in storage_refs:
            metadata = await artifact_service.get_artifact_metadata(ref)
            if not metadata["exists"]:
                raise ArtifactStorageError(f"Failed to verify artifact: {ref}")

    except ArtifactStorageError as e:
        # Handle storage failures gracefully
        logger.error(f"Artifact storage failed: {e}")
        # Continue execution without artifacts
        return []

    except Exception as e:
        # Unexpected errors - cleanup and re-raise
        await artifact_service.cleanup_execution_artifacts("exec-789")
        raise
```

#### Multi-modal Artifact Support

```python
from src.roma.domain.entities.artifacts.image_artifact import ImageArtifact
from src.roma.domain.value_objects.media_type import MediaType

# Create image artifact
chart = ImageArtifact.create(
    name="trend_analysis_chart",
    content=image_bytes,
    task_id="task-123",
    media_type=MediaType.IMAGE,
    metadata={
        "format": "PNG",
        "width": 1920,
        "height": 1080,
        "generated_by": "matplotlib"
    }
)

# Mixed artifact types in envelope
mixed_envelope = ResultEnvelope.create_success(
    result=result,
    task_id="task-123",
    execution_id="exec-456",
    agent_type=AgentType.EXECUTOR,
    execution_metrics=metrics,
    artifacts=[document, chart],  # Text + Image
    output_text="Generated report with visualizations"
)

# ArtifactService handles all types transparently
storage_refs = await artifact_service.store_envelope_artifacts(
    "exec-456", mixed_envelope
)
# Both document and chart are stored with appropriate handling
```

## PostgreSQL Persistence Layer ✅

### Architecture Overview

The PostgreSQL persistence layer provides robust event sourcing, execution history, and checkpoint recovery capabilities for ROMA v2.0.

```mermaid
graph TB
    subgraph "Application Layer"
        SM[SystemManager]
        ES[EventStore Interface]
        EHS[ExecutionHistoryService]
        CHS[CheckpointService]
    end

    subgraph "Domain Layer"
        DC[DatabaseConfig]
        DPC[DatabasePoolConfig]
        TE[TaskEvents]
        TS[TaskStatus]
        TT[TaskType]
        NT[NodeType]
    end

    subgraph "Infrastructure Persistence"
        PSE[PostgreSQLEventStore]
        DCM[DatabaseConnectionManager]

        subgraph "Models"
            EM[EventModel]
            TEM[TaskExecutionModel]
            TRM[TaskRelationshipModel]
            CM[CheckpointModel]
        end

        subgraph "Database"
            PG[(PostgreSQL)]
            IDX[Composite Indexes]
            MIG[Alembic Migrations]
        end
    end

    SM --> ES
    ES --> PSE
    PSE --> DCM
    PSE --> EM
    PSE --> TEM
    PSE --> TRM
    PSE --> CM

    DCM --> DC
    DCM --> DPC
    DCM --> PG

    EM --> TE
    TEM --> TS
    TEM --> TT
    TEM --> NT

    PG --> IDX
    PG --> MIG
```

### Key Components

#### DatabaseConfig (Domain Value Object)
- **Location**: `src/roma/domain/value_objects/config/database_config.py`
- **Pattern**: Immutable dataclass following ROMA config conventions
- **Features**:
  - Field validation (host, port, database, credentials)
  - Environment variable integration via Hydra
  - Connection pooling configuration
  - SSL and timeout settings

```python
@dataclass(frozen=True)
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "roma_db"
    user: str = "roma_user"
    password: str = "roma_password"
    pool: DatabasePoolConfig = Field(default_factory=DatabasePoolConfig)
```

#### DatabaseConnectionManager
- **Location**: `src/roma/infrastructure/persistence/connection_manager.py`
- **Responsibilities**:
  - AsyncPG connection pooling with health checks
  - Connection lifecycle management
  - Automatic reconnection and circuit breaking
  - Performance monitoring and statistics
  - Transaction management with context managers

#### PostgreSQLEventStore
- **Location**: `src/roma/infrastructure/persistence/postgres_event_store.py`
- **Implements**: EventStore interface from application layer
- **Features**:
  - Event sourcing with complete execution history
  - Async batch operations for performance
  - Event filtering and querying capabilities
  - Automatic event serialization/deserialization
  - Checkpoint support for state recovery

#### Database Models
All models use SQLAlchemy 2.0 with async support:

1. **EventModel**: Complete event logging with JSONB metadata
2. **TaskExecutionModel**: Task execution tracking with performance metrics
3. **TaskRelationshipModel**: Graph structure persistence
4. **CheckpointModel**: State snapshots with compression

### Performance Optimizations

#### Composite Indexes
```sql
-- Most common query patterns
CREATE INDEX idx_events_task_timestamp ON events (task_id, timestamp);
CREATE INDEX idx_events_type_timestamp ON events (event_type, timestamp);
CREATE INDEX idx_events_correlation ON events (correlation_id, timestamp);

-- JSONB GIN index for metadata queries
CREATE INDEX idx_events_metadata_gin ON events USING gin (event_metadata);

-- Task execution queries
CREATE INDEX idx_executions_status_created ON task_executions (status, created_at);
CREATE INDEX idx_executions_parent_status ON task_executions (parent_id, status);
```

#### Connection Pooling
- Minimum 10 connections, maximum 50 connections
- Query timeout: 30 seconds
- Health checks every 30 seconds
- Automatic connection replacement on failure

#### Batch Operations
- Event batching for high-throughput scenarios
- Bulk insert optimizations
- Prepared statement caching

### Integration Patterns

#### SystemManager Integration
```python
class SystemManager:
    async def _initialize_event_store(self) -> None:
        """Auto-fallback between PostgreSQL and in-memory stores."""
        if self.config.database and self.config.database.host:
            try:
                self._connection_manager = DatabaseConnectionManager(self.config.database)
                await self._connection_manager.initialize()
                self._event_store = PostgreSQLEventStore(self._connection_manager)
                await self._event_store.initialize()
                logger.info("PostgreSQL event store initialized successfully")
            except Exception as e:
                logger.warning(f"PostgreSQL unavailable, falling back to in-memory store: {e}")
                self._event_store = InMemoryEventStore()
        else:
            self._event_store = InMemoryEventStore()
```

#### Configuration Integration
```yaml
# config/config.yaml
database:
  _target_: roma.domain.value_objects.config.database_config.DatabaseConfig
  host: "${oc.env:ROMA_DB_HOST,localhost}"
  port: "${oc.env:ROMA_DB_PORT,5432}"
  database: "${oc.env:ROMA_DB_NAME,roma_db}"
  user: "${oc.env:ROMA_DB_USER,roma_user}"
  password: "${oc.env:ROMA_DB_PASSWORD,roma_password}"
  pool:
    _target_: roma.domain.value_objects.config.database_config.DatabasePoolConfig
    min_size: "${oc.env:ROMA_DB_POOL_MIN_SIZE,10}"
    max_size: "${oc.env:ROMA_DB_POOL_MAX_SIZE,50}"
```

### Migration Management

#### Alembic Integration
- **Location**: `src/roma/infrastructure/persistence/migrations/`
- **Features**:
  - Automatic schema versioning
  - Environment-specific configurations
  - Safe migration rollbacks
  - Data migration support

#### Migration Commands
```bash
# Initialize migrations
alembic init migrations

# Create new migration
alembic revision --autogenerate -m "Add event store tables"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1
```

### Error Handling and Recovery

#### Connection Recovery
- Automatic reconnection on connection loss
- Circuit breaker pattern for cascading failures
- Exponential backoff for retry strategies
- Graceful degradation to in-memory store

#### Data Integrity
- ACID transaction guarantees
- Foreign key constraints
- Check constraints for data validation
- Duplicate event prevention

#### Monitoring and Observability
- Connection pool metrics
- Query performance tracking
- Error rate monitoring
- Health check endpoints

### Testing Strategy

#### Integration Tests
- **Location**: `tests/integration/persistence/`
- **Coverage**: Complete CRUD operations, concurrent access, error scenarios
- **Database**: Isolated test database with cleanup between tests

#### Performance Tests
- Load testing with 1000+ concurrent operations
- Memory usage validation
- Connection pool stress testing
- Query performance benchmarking

### Production Considerations

#### Security
- SSL/TLS encryption for connections
- Role-based access control
- Parameter sanitization
- Connection string security

#### Scalability
- Read replicas for query optimization
- Connection pooling across instances
- Horizontal scaling with sharding
- Cache integration with Redis

#### Backup and Recovery
- Point-in-time recovery capabilities
- Automated backup schedules
- Disaster recovery procedures
- Data retention policies

### Future Enhancements

#### Planned Improvements
- ✅ Connection pool duplication fix
- ✅ Event type reconstruction optimization
- ⏳ Read replica support
- ⏳ Streaming replication
- ⏳ Advanced query optimization
- ⏳ Metric collection integration

## Summary

This architecture provides:

1. **Clear Separation of Concerns**: Each object has a single, well-defined responsibility
2. **Scalability**: Concurrent execution with proper control mechanisms
3. **Reliability**: Multiple error recovery strategies with PostgreSQL persistence
4. **Observability**: Complete event tracking and tracing with event sourcing
5. **Flexibility**: Plugin-based agent and tool system
6. **Maintainability**: Clean architecture with clear boundaries
7. **Extensibility**: Framework-agnostic design enables easy adoption of new agent frameworks
8. **Persistence**: Robust PostgreSQL layer with event sourcing and checkpoint recovery

The system follows SOLID principles, uses immutable data structures for thread safety, and provides comprehensive error handling and recovery mechanisms with enterprise-grade persistence capabilities.