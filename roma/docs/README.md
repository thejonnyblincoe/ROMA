# ROMA v2.0 Documentation

## Core Documents

### 📐 [FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md)
The complete consolidated architecture document for ROMA v2.0. This is the primary reference for understanding the system design, components, and implementation details.

### 📊 [ARCHITECTURE_DETAILED.md](ARCHITECTURE_DETAILED.md)
Comprehensive technical architecture with:
- Updated UML diagrams reflecting AgentRuntimeService pattern
- Direct agent execution flow (no AtomizerService layer)
- Component interactions and data flows
- State machines and activity diagrams
- Concurrency patterns and deployment architecture

### 📋 [ROMA_V2_TASK_PLAN.json](ROMA_V2_TASK_PLAN.json)
Detailed 12-week implementation plan with all tasks, dependencies, and milestones in JSON format.

### ✅ [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
Comprehensive implementation status tracking for ROMA v2.0 components, showing Phase 1 completion status with 99.3% domain test coverage and 92.9% orchestration test coverage including enhanced dependency resolution features.

## Key Concepts

### Core Pattern
```
Task → AgentRuntimeService → Agent (ATOMIZER/PLANNER/EXECUTOR/AGGREGATOR) → Result
```

### Task Types (MECE)
- **RETRIEVE**: Data acquisition from external sources
- **WRITE**: Content generation and synthesis
- **THINK**: Analysis and reasoning
- **CODE_INTERPRET**: Code execution
- **IMAGE_GENERATION**: Visual content creation

**Note**: AGGREGATE is an **agent type** (performed by Aggregator agents), not a task type.

### Configuration Hierarchy
1. **Level 1**: Entities (models, tools)
2. **Level 2**: Agents (atomizers, planners, executors)
3. **Level 3**: Profiles (agent selection and settings)
4. **Level 4**: Application configuration

## Quick Start

1. Review `FINAL_ARCHITECTURE.md` for system overview
2. Check `ARCHITECTURE_VERIFICATION.md` for current implementation status
3. Use `ROMA_V2_TASK_PLAN.json` for project tracking
4. See `IMPLEMENTATION_STATUS.md` for detailed Phase 2 roadmap

## Implementation Status

- ✅ **Phase 1 COMPLETE**: Domain layer, application services, agent system
  - 99.3% domain test coverage (146/148 tests passing)
  - 89.8% orchestration test coverage (53/59 tests passing) - Core validated
  - AgentRuntimeService with lazy loading (25x memory improvement)
  - SystemManager orchestration working
  - Real LLM integration via Agno framework
- 🔄 **Phase 2 IN PROGRESS**: Framework entry wiring, toolkit migration
- 📋 **Future Phases**: Full observability, production deployment

## Archive

Legacy documentation has been moved to `archive/` for reference. The documents listed above represent the current, consolidated state of ROMA v2.0 architecture and planning.