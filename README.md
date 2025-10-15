# ROMA DSPy Modules

> Building reusable large-language-model agents on top of [DSPy](https://github.com/stanfordnlp/dspy) with a pragmatic set of task-focused modules. This document explains how every module under `roma_dspy.core.modules` fits together, how to configure them, and how to compose them into production-grade pipelines.

## 🚀 Quick Start

```bash
# One-command setup
just setup

# Or with specific profile
just setup crypto_agent
```

See [SETUP_GUIDE.md](reports/SETUP_GUIDE.md) for detailed setup instructions.

## Table of Contents
- [Conceptual Overview](#conceptual-overview)
- [Installation & Setup](#installation--setup)
- [Quickstart: End-to-End Workflow](#quickstart-end-to-end-workflow)
- [Configuration & Storage](#configuration--storage)
  - [Configuration System](#configuration-system)
  - [Storage Architecture](#storage-architecture)
  - [Profile Management](#profile-management)
- [Toolkits](#toolkits)
  - [Built-in Toolkits](#built-in-toolkits)
  - [Toolkit Configuration](#toolkit-configuration)
  - [Custom Toolkits](#custom-toolkits)
- [REST API](#rest-api)
  - [Starting the Server](#starting-the-server)
  - [API Endpoints](#api-endpoints)
  - [Example Usage](#example-usage)
- [Enhanced CLI](#enhanced-cli)
  - [Command Reference](#command-reference)
  - [Examples](#examples)
- [Core Building Block: `BaseModule`](#core-building-block-basemodule)
  - [Context & LM Management](#context--lm-management)
  - [Working with Tools](#working-with-tools)
  - [Prediction Strategies](#prediction-strategies)
  - [Async Execution](#async-execution)
- [Module Reference](#module-reference)
  - [Atomizer](#atomizer)
  - [Planner](#planner)
  - [Executor](#executor)
  - [Aggregator](#aggregator)
  - [Verifier](#verifier)
- [Advanced Patterns](#advanced-patterns)
  - [Swapping Models at Runtime](#swapping-models-at-runtime)
  - [Per-Call Overrides](#per-call-overrides)
  - [Tool-Only Execution](#tool-only-execution)
- [Testing](#testing)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Glossary](#glossary)

---

## Conceptual Overview
ROMA’s module layer wraps canonical DSPy patterns into purpose-built components that reflect the lifecycle of complex task execution:

1. **Atomizer** decides whether a request can be handled directly or needs decomposition.
2. **Planner** breaks non-atomic goals into an ordered graph of subtasks.
3. **Executor** resolves individual subtasks, optionally routing through function/tool calls.
4. **Aggregator** synthesizes subtask outputs back into a coherent answer.
5. **Verifier** (optional) inspects the aggregate output against the original goal before delivering.

Every module shares the same ergonomics: instantiate it with a language model (LM) or provider string, choose a prediction strategy, then call `.forward()` (or `.aforward()` for async) with the task-specific fields.

All modules ultimately delegate to DSPy signatures defined in `roma_dspy.core.signatures`. This keeps interfaces stable even as the internals evolve.

## Installation & Setup

**Prerequisites:**
- Python 3.12+
- [Just](https://github.com/casey/just) command runner (optional, recommended)

**Quick Setup:**
```bash
# One-command setup with Just
just setup

# Or manual installation
pip install -e .
```

**With REST API Support:**
```bash
pip install -e ".[api]"
```

**With E2B Code Execution:**
```bash
pip install -e ".[e2b]"
```

ROMA depends on DSPy. Make sure any provider-specific environment variables (OpenAI keys, Fireworks credentials, etc.) are already exported before you instantiate an LM.

```bash
export OPENAI_API_KEY=...
export FIREWORKS_API_KEY=...
```

**For API Usage (optional):**
```bash
export DATABASE_URL=postgresql+asyncpg://user:password@localhost/roma
export ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
export REQUIRE_AUTH=false  # Set to true for API key authentication
```

> **Note**: When running inside notebooks or async environments, prefer `dspy.configure(lm=...)` or rely on ROMA's per-call context defaults as shown below.

## Quickstart: End-to-End Workflow
The following example mirrors a typical orchestration loop. It uses three different providers to showcase how easily each module can work with distinct models and strategies.

```python
import dspy
from roma_dspy import Aggregator, Atomizer, Executor, Planner, Verifier, SubTask
from roma_dspy.types import TaskType

# Optional tool that the Executor may call
def get_weather(city: str) -> str:
    """Return a canned weather report for the city."""
    return f"The weather in {city} is sunny."

# Executor geared toward ReAct with a Fireworks model
executor_lm = dspy.LM(
    "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905",
    temperature=0.7,
    cache=True,
)
executor = Executor(
    lm=executor_lm,
    prediction_strategy="react",
    tools=[get_weather],
    context_defaults={"track_usage": True},
)

# Atomizer decides when to branch into planning
atomizer = Atomizer(
    lm=dspy.LM("openrouter/google/gemini-2.5-flash", temperature=0.6, cache=False),
    prediction_strategy="cot",
    context_defaults={"track_usage": True},
)

# Planner produces executable subtasks for non-atomic goals
planner = Planner(
    lm=dspy.LM("openrouter/openai/gpt-4o-mini", temperature=0.85, cache=True),
    prediction_strategy="cot",
    context_defaults={"track_usage": True},
)

aggregator = Aggregator(
    lm=dspy.LM("openrouter/openai/gpt-4o-mini", temperature=0.65),
    prediction_strategy="cot",
)

verifier = Verifier(
    lm=dspy.LM("openrouter/openai/gpt-4o-mini", temperature=0.0),
)

def run_pipeline(goal: str) -> str:
    atomized = atomizer.forward(goal)
    if atomized.is_atomic or atomized.node_type.is_execute:
        execution = executor.forward(goal)
        candidate = execution.output
    else:
        plan = planner.forward(goal)
        results = []
        for idx, subtask in enumerate(plan.subtasks, start=1):
            execution = executor.forward(subtask.goal)
            results.append(
                SubTask(
                    goal=subtask.goal,
                    task_type=subtask.task_type,
                    dependencies=subtask.dependencies,
                )
            )
        aggregated = aggregator.forward(goal, results)
        candidate = aggregated.synthesized_result

    verdict = verifier.forward(goal, candidate)
    if verdict.verdict:
        return candidate
    return f"Verifier flagged the output: {verdict.feedback or 'no feedback returned'}"

print(run_pipeline("Plan a weekend in Barcelona and include a packing list."))
```

Highlights:
- Different modules can run on different LMs and temperatures.
- Tools are provided either at construction or per-call.
- `context_defaults` ensures each `.forward()` call enters a proper `dspy.context()` with the module's LM.

---

## Configuration & Storage

ROMA-DSPy features a comprehensive configuration system and execution-scoped storage architecture designed for production deployments.

### Configuration System

#### Loading Configurations

```python
from roma_dspy.config import load_config

# Load with defaults
config = load_config()

# Load with specific profile
config = load_config(profile="high_quality")

# Load with runtime overrides
config = load_config(
    profile="crypto_agent",
    overrides=["agents.executor.llm.temperature=0.3"]
)
```

#### Configuration Structure

**Agent Configuration:**
```yaml
agents:
  executor:
    llm:
      model: "openai/gpt-4o-mini"
      temperature: 0.7
      max_tokens: 2000
      timeout: 30
    prediction_strategy: "react"
    enabled: true
    toolkits:
      - class_name: "FileToolkit"
        enabled: true
        toolkit_config:
          enable_delete: false
          max_file_size: 10485760  # 10MB
```

**Storage Configuration:**
```yaml
storage:
  type: "local"  # or "s3"
  base_path: "/opt/sentient/storage"  # Local path or S3 bucket
  s3:
    bucket: "roma-dspy-storage"
    region: "us-east-1"
    endpoint_url: null  # For S3-compatible services
```

**Runtime Configuration:**
```yaml
runtime:
  timeout: 300
  max_depth: 5
  enable_logging: true
  log_level: INFO
```

#### Available Profiles

- **`lightweight`**: Minimal resource usage, suitable for testing
- **`high_quality`**: Production-grade with GPT-4 models
- **`tool_enabled`**: Pre-configured with FileToolkit and E2BToolkit
- **`crypto_agent`**: Specialized for crypto/blockchain analysis

See [config/README.md](config/README.md) for detailed configuration documentation.

### Storage Architecture

ROMA-DSPy implements **execution-scoped storage** to ensure complete isolation between task executions.

#### Storage Hierarchy

```
{base_path}/
├── executions/
│   └── {execution_id}/          # Unique per execution
│       ├── .cache/               # LM caching
│       │   └── toolkit/          # Toolkit data storage
│       │       ├── arkham/       # Parquet files for large responses
│       │       ├── binance/
│       │       └── defillama/
│       └── files/                # FileToolkit workspace
│           ├── input.txt
│           └── output.json
└── shared/                       # Shared resources (if needed)
```

#### FileStorage

Every execution gets automatic FileStorage initialization:

```python
from roma_dspy.core.engine.solve import solve

# Storage is created automatically with execution ID
result = solve("Analyze blockchain transactions")
# Files stored at: {base_path}/executions/{auto_generated_id}/
```

**Key Features:**
- ✅ Automatic lifecycle management
- ✅ Complete execution isolation
- ✅ S3-compatible paths
- ✅ Automatic cleanup support
- ✅ No manual configuration required

#### DataStorage (Parquet Integration)

Large toolkit responses are automatically stored in Parquet format:

```python
# Automatic threshold-based storage (default: 100KB)
# Response > 100KB → Parquet file
# Response < 100KB → Direct JSON response

{
  "success": true,
  "data_stored": true,
  "storage_path": ".cache/toolkit/arkham/transfers_12345.parquet",
  "size_kb": 250,
  "summary": {
    "count": 1000,
    "preview": [...]  # First few records
  }
}
```

**Configuration:**
```yaml
agents:
  executor:
    toolkits:
      - class_name: "ArkhamToolkit"
        toolkit_config:
          storage_threshold_kb: 100  # Customize threshold
```

#### S3 Integration

For production deployments with S3:

```yaml
# config/profiles/production.yaml
storage:
  type: "s3"
  base_path: "s3://my-bucket/roma-executions"
  s3:
    bucket: "my-bucket"
    region: "us-east-1"
    access_key_id: ${S3_ACCESS_KEY}
    secret_access_key: ${S3_SECRET_KEY}
```

S3 works transparently via Docker volume mounts - no code changes needed:

```bash
# Docker Compose handles S3 mounting
docker compose --profile s3 up -d
```

See [STORAGE_ARCHITECTURE.md](reports/STORAGE_ARCHITECTURE.md) for complete architecture details.

### Profile Management

#### Creating Custom Profiles

1. Create YAML file in `config/profiles/`:

```yaml
# config/profiles/my_agent.yaml
project: "my-custom-agent"
version: "1.0.0"

agents:
  executor:
    llm:
      model: "openai/gpt-4"
      temperature: 0.5
    prediction_strategy: "react"
    toolkits:
      - class_name: "FileToolkit"
        enabled: true
      - class_name: "E2BToolkit"
        enabled: true
        toolkit_config:
          timeout: 600

storage:
  base_path: "/data/my-agent"

runtime:
  timeout: 300
  max_depth: 6
```

2. Use with setup:

```bash
just setup my_agent
```

3. Use programmatically:

```python
config = load_config(profile="my_agent")
```

---

## Toolkits

ROMA-DSPy includes a powerful toolkit system that extends agent capabilities with external tools and services.

### Built-in Toolkits

#### FileToolkit (Core)

File operations with execution-scoped isolation:

```python
# Automatic in executor configuration
agents:
  executor:
    toolkits:
      - class_name: "FileToolkit"
        enabled: true
        toolkit_config:
          enable_delete: false  # Safety: disable destructive operations
          max_file_size: 10485760  # 10MB limit
```

**Available Operations:**
- `save_file(filename, content)` - Save text/data to file
- `read_file(filename)` - Read file contents
- `list_files(pattern)` - List files matching pattern
- `file_exists(filename)` - Check file existence
- `delete_file(filename)` - Delete file (if enabled)

**Features:**
- ✅ Execution-scoped (automatic isolation)
- ✅ No cross-execution contamination
- ✅ Requires FileStorage (validated automatically)
- ✅ Configurable size limits and permissions

#### E2BToolkit (Code Execution)

Secure code execution in isolated sandboxes:

```yaml
agents:
  executor:
    toolkits:
      - class_name: "E2BToolkit"
        enabled: true
        toolkit_config:
          timeout: 600  # 10 minutes
          max_lifetime_hours: 23.5
          template: "base"  # or custom E2B template
          auto_reinitialize: true
```

**Available Operations:**
- `execute_code(language, code)` - Execute Python/JavaScript/Shell
- `install_package(package_name)` - Install dependencies
- `read_file(path)` - Read files from sandbox
- `write_file(path, content)` - Write files to sandbox

**Use Cases:**
- Data analysis with pandas/numpy
- Web scraping
- Package testing
- Long-running computations

See [E2B_SETUP.md](docs/E2B_SETUP.md) for setup instructions.

#### Crypto Toolkits

Specialized toolkits for blockchain/crypto analysis:

**ArkhamToolkit** - On-chain intelligence:
```yaml
- class_name: "ArkhamToolkit"
  enabled: true
  toolkit_config:
    api_key: ${ARKHAM_API_KEY}
    default_chain: "ethereum"
    enable_analysis: true  # Statistical analysis
```

**BinanceToolkit** - Exchange data:
```yaml
- class_name: "BinanceToolkit"
  enabled: true
  toolkit_config:
    default_market: "spot"
    enable_analysis: true
```

**CoinGeckoToolkit** - Market data:
```yaml
- class_name: "CoinGeckoToolkit"
  enabled: true
  toolkit_config:
    default_vs_currency: "usd"
    enable_analysis: true
```

**DefiLlamaToolkit** - DeFi analytics:
```yaml
- class_name: "DefiLlamaToolkit"
  enabled: true
  toolkit_config:
    enable_pro_features: false
    enable_analysis: true
```

All crypto toolkits include:
- ✅ Automatic Parquet storage for large datasets
- ✅ Built-in statistical analysis
- ✅ Rate limiting
- ✅ Error handling and retries

#### Utility Toolkits

**CalculatorToolkit** - Basic mathematical operations:
```yaml
- class_name: "CalculatorToolkit"
  enabled: true
```

**SerperToolkit** - Web search via Serper API:
```yaml
- class_name: "SerperToolkit"
  enabled: true
  toolkit_config:
    api_key: ${SERPER_API_KEY}
    max_results: 10
```

### Toolkit Configuration

#### Tool Selection

Include/exclude specific tools from a toolkit:

```yaml
agents:
  executor:
    toolkits:
      - class_name: "FileToolkit"
        enabled: true
        include_tools:  # Only these tools
          - "read_file"
          - "save_file"
        exclude_tools:  # Block these tools
          - "delete_file"
```

#### Toolkit-Specific Settings

Each toolkit accepts custom configuration:

```yaml
- class_name: "E2BToolkit"
  toolkit_config:
    timeout: 600
    max_lifetime_hours: 23.5
    template: "base"

- class_name: "FileToolkit"
  toolkit_config:
    enable_delete: false
    max_file_size: 10485760

- class_name: "ArkhamToolkit"
  toolkit_config:
    api_key: ${ARKHAM_API_KEY}
    default_chain: "ethereum"
    storage_threshold_kb: 100  # Parquet threshold
```

### Custom Toolkits

Create custom toolkits by extending `BaseToolkit`:

```python
from roma_dspy.tools.base import BaseToolkit
from typing import Dict, Any

class CustomToolkit(BaseToolkit):
    """My custom toolkit."""

    # Set to True if toolkit needs FileStorage
    REQUIRES_FILE_STORAGE: bool = False

    def _setup_dependencies(self) -> None:
        """Setup external dependencies."""
        pass

    def _initialize_tools(self) -> None:
        """Initialize toolkit-specific configuration."""
        pass

    async def my_tool(self, param: str) -> Dict[str, Any]:
        """My custom tool - auto-registered by BaseToolkit."""
        result = await self._do_something(param)

        # Use storage-enabled response builder
        return await self._build_success_response(
            data=result,
            storage_data_type="my_data",
            storage_prefix="my_tool",
            tool_name="my_tool",
            param=param
        )
```

**Registration:**

```yaml
agents:
  executor:
    toolkits:
      - class_name: "CustomToolkit"
        enabled: true
        toolkit_config:
          my_setting: "value"
```

**Key Guidelines:**
- Inherit from `BaseToolkit`
- Use `_file_storage` for file operations (if `REQUIRES_FILE_STORAGE = True`)
- Use `_data_storage` for automatic Parquet storage
- Use `_build_success_response()` for consistent responses
- Implement `_setup_dependencies()` and `_initialize_tools()`

---

## REST API

ROMA-DSPy exposes a comprehensive REST API for programmatic access to all framework capabilities.

### Starting the Server

```bash
# Development mode with auto-reload
roma-dspy server start --reload

# Production mode
roma-dspy server start --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

**Execution Management** (5 endpoints)
- `POST /api/v1/executions` - Create new execution
- `GET /api/v1/executions` - List executions (with pagination & filtering)
- `GET /api/v1/executions/{id}` - Get execution details
- `GET /api/v1/executions/{id}/status` - Poll execution status
- `POST /api/v1/executions/{id}/cancel` - Cancel running execution

**Checkpoint Management** (4 endpoints)
- `GET /api/v1/executions/{id}/checkpoints` - List checkpoints
- `GET /api/v1/checkpoints/{id}` - Get checkpoint details
- `POST /api/v1/checkpoints/{id}/restore` - Restore from checkpoint
- `DELETE /api/v1/checkpoints/{id}` - Delete checkpoint

**Visualization** (2 endpoints)
- `POST /api/v1/executions/{id}/visualize` - Generate DAG visualization
- `GET /api/v1/executions/{id}/dag` - Get raw DAG snapshot

**Metrics** (2 endpoints)
- `GET /api/v1/executions/{id}/metrics` - Get LM usage metrics
- `GET /api/v1/executions/{id}/costs` - Get cost breakdown

**Health** (1 endpoint)
- `GET /health` - Health check with uptime and storage status

### Example Usage

```python
import httpx

# Create execution
response = httpx.post("http://localhost:8000/api/v1/executions", json={
    "goal": "Research recent ML papers on transformers",
    "max_depth": 3,
    "config_profile": "high_quality"
})
execution_id = response.json()["execution_id"]

# Poll status
status = httpx.get(f"http://localhost:8000/api/v1/executions/{execution_id}/status")
print(status.json())

# Get visualization
viz = httpx.post(f"http://localhost:8000/api/v1/executions/{execution_id}/visualize", json={
    "visualizer_type": "tree",
    "format": "text"
})
print(viz.json()["visualization"])
```

**Features:**
- Async/await with FastAPI
- Pydantic schema validation
- Background task management with 5-second status caching
- Request logging and rate limiting (60 req/min)
- CORS support (configurable)
- Comprehensive error handling

---

## Enhanced CLI

ROMA-DSPy provides a rich command-line interface with 16 commands organized into logical groups.

### Command Reference

**Original Commands** (3)
- `roma-dspy solve <task>` - Execute task locally
- `roma-dspy config` - Display configuration
- `roma-dspy version` - Show version

**Server Management** (2)
- `roma-dspy server start` - Start API server
- `roma-dspy server health` - Check server health

**Execution Management** (5)
- `roma-dspy exec create <task>` - Create execution via API
- `roma-dspy exec list` - List all executions
- `roma-dspy exec get <id>` - Get execution details
- `roma-dspy exec status <id>` - Poll status (supports --watch)
- `roma-dspy exec cancel <id>` - Cancel execution

**Checkpoint Management** (4)
- `roma-dspy checkpoint list <execution_id>` - List checkpoints
- `roma-dspy checkpoint get <checkpoint_id>` - Get checkpoint
- `roma-dspy checkpoint restore <checkpoint_id>` - Restore
- `roma-dspy checkpoint delete <checkpoint_id>` - Delete

**Visualization & Metrics** (2)
- `roma-dspy visualize <id>` - Generate visualization
- `roma-dspy metrics <id>` - Get metrics and costs

### Examples

```bash
# Start API server
roma-dspy server start --reload &

# Create execution
EXEC_ID=$(roma-dspy exec create "Research ML papers" | grep -oE '[a-f0-9-]{36}')

# Watch execution progress
roma-dspy exec status $EXEC_ID --watch

# Visualize completed execution
roma-dspy visualize $EXEC_ID --type tree --output viz.txt

# Get cost breakdown
roma-dspy metrics $EXEC_ID --breakdown

# List checkpoints
roma-dspy checkpoint list $EXEC_ID

# Local execution (backward compatible)
roma-dspy solve "Plan a trip" --profile high_quality --verbose
```

**Features:**
- Rich terminal UI (tables, panels, colors)
- Watch mode for real-time status updates
- File output support for visualizations
- Confirmation prompts for destructive operations
- Comprehensive help text (--help on any command)

---

## Core Building Block: `BaseModule`
All modules inherit from `BaseModule`, located at `roma_dspy/core/modules/base_module.py`. It standardizes:
- signature binding via DSPy prediction strategies,
- LM instantiation and context management,
- tool normalization and merging,
- sync/async entrypoints with safe keyword filtering.

### Context & LM Management
When you instantiate a module, you can either provide an existing `dspy.LM` or let the module build one from a provider string (`model`) and optional keyword arguments (`model_config`).

```python
from roma_dspy import Executor

executor = Executor(
    model="openrouter/openai/gpt-4o-mini",
    model_config={"temperature": 0.5, "cache": True},
)
```

Internally, `BaseModule` ensures that every `.forward()` call wraps the predictor invocation in:

```python
with dspy.context(lm=self._lm, **context_defaults):
    ...
```

You can inspect the effective LM configuration via `get_model_config()` to confirm provider, cache settings, or sanitized kwargs.

### Working with Tools
Tools can be supplied as a list, tuple, or mapping of callables accepted by DSPy’s ReAct/CodeAct strategies.

```python
executor = Executor(tools=[get_weather])
executor.forward("What is the weather in Amman?", tools=[another_function])
```

`BaseModule` automatically deduplicates tools based on object identity and merges constructor defaults with per-call overrides.

### Prediction Strategies
ROMA exposes DSPy's strategies through the `PredictionStrategy` enum (`roma_dspy/types/prediction_strategy.py`). Use either the enum or a case-insensitive string alias:

```python
from roma_dspy.types import PredictionStrategy

planner = Planner(prediction_strategy=PredictionStrategy.CHAIN_OF_THOUGHT)
executor = Executor(prediction_strategy="react")
```

Available options include `Predict`, `ChainOfThought`, `ReAct`, `CodeAct`, `BestOfN`, `Refine`, `Parallel`, `majority`, and more. Strategies that require tools (`ReAct`, `CodeAct`) automatically receive any tools you pass to the module.

### Async Execution
Every module offers an `aforward()` method. When the underlying DSPy predictor supports async (`acall`/`aforward`), ROMA dispatches asynchronously; otherwise, it gracefully falls back to the sync implementation while preserving awaitability.

```python
result = await executor.aforward("Download the latest sales report")
```

## Module Reference

### Atomizer
**Location**: `roma_dspy/core/modules/atomizer.py`

**Purpose**: Decide whether a goal is atomic or needs planning.

**Constructor**:
```python
Atomizer(
    prediction_strategy: Union[PredictionStrategy, str] = "ChainOfThought",
    *,
    lm: Optional[dspy.LM] = None,
    model: Optional[str] = None,
    model_config: Optional[Mapping[str, Any]] = None,
    tools: Optional[Sequence|Mapping] = None,
    **strategy_kwargs,
)
```

**Inputs** (`AtomizerSignature`):
- `goal: str`

**Outputs** (`AtomizerResponse`):
- `is_atomic: bool` — whether the task can run directly.
- `node_type: NodeType` — `PLAN` or `EXECUTE` hint for downstream routing.

**Usage**:
```python
atomized = atomizer.forward("Curate a 5-day Tokyo itinerary with restaurant reservations")
if atomized.is_atomic:
    ...  # send directly to Executor
else:
    ...  # hand off to Planner
```

The Atomizer is strategy-agnostic but typically uses `ChainOfThought` or `Predict`. You can pass hints (e.g., `max_tokens`) via `call_params`:

```python
atomizer.forward(
    "Summarize this PDF",
    call_params={"max_tokens": 200},
)
```

### Planner
**Location**: `roma_dspy/core/modules/planner.py`

**Purpose**: Break a goal into ordered subtasks with optional dependency graph.

**Constructor**: identical pattern as the Atomizer.

**Inputs** (`PlannerSignature`):
- `goal: str`

**Outputs** (`PlannerResult`):
- `subtasks: List[SubTask]` — each has `goal`, `task_type`, and `dependencies`.
- `dependencies_graph: Optional[Dict[str, List[str]]]` — explicit adjacency mapping when returned by the LM.

**Usage**:
```python
plan = planner.forward("Launch a B2B webinar in 6 weeks")
for subtask in plan.subtasks:
    print(subtask.goal, subtask.task_type)
```

`SubTask.task_type` is a `TaskType` enum that follows the ROMA MECE framework (Retrieve, Write, Think, Code Interpret, Image Generation).

### Executor
**Location**: `roma_dspy/core/modules/executor.py`

**Purpose**: Resolve atomic goals, optionally calling tools/functions through DSPy's ReAct, CodeAct, or similar strategies.

**Constructor**: same pattern; the most common strategies are `ReAct`, `CodeAct`, or `ChainOfThought`.

**Inputs** (`ExecutorSignature`):
- `goal: str`

**Outputs** (`ExecutorResult`):
- `output: str | Any`
- `sources: Optional[List[str]]` — provenance or citations.

**Usage**:
```python
execution = executor.forward(
    "Compile a packing list for a 3-day ski trip",
    config={"temperature": 0.4},  # per-call LM override
)
print(execution.output)
```

To expose tools only for certain calls:

```python
execution = executor.forward(
    "What is the weather in Paris?",
    tools=[get_weather],
)
```

### Aggregator
**Location**: `roma_dspy/core/modules/aggregator.py`

**Purpose**: Combine multiple subtask results into a final narrative or decision.

**Constructor**: identical pattern.

**Inputs** (`AggregatorResult` signature):
- `original_goal: str`
- `subtasks_results: List[SubTask]` — usually the planner’s proposals augmented with execution outputs.

**Outputs** (`AggregatorResult` base model):
- `synthesized_result: str`

**Usage**:
```python
aggregated = aggregator.forward(
    original_goal="Plan a data migration",
    subtasks_results=[
        SubTask(goal="Inventory current databases", task_type=TaskType.RETRIEVE),
        SubTask(goal="Draft migration timeline", task_type=TaskType.WRITE),
    ],
)
print(aggregated.synthesized_result)
```

Because it inherits `BaseModule`, you can still attach tools (e.g., a knowledge-base retrieval function) if your aggregation strategy requires external calls.

### Verifier
**Location**: `roma_dspy/core/modules/verifier.py`

**Purpose**: Validate that the synthesized output satisfies the original goal.

**Inputs** (`VerifierSignature`):
- `goal: str`
- `candidate_output: str`

**Outputs**:
- `verdict: bool`
- `feedback: Optional[str]`

**Usage**:
```python
verdict = verifier.forward(
    goal="Draft a GDPR-compliant privacy policy",
    candidate_output=aggregated.synthesized_result,
)
if not verdict.verdict:
    print("Needs revision:", verdict.feedback)
```

## Advanced Patterns

### Swapping Models at Runtime
Use `replace_lm()` to reuse the same module with a different LM (useful for A/B testing or fallbacks).

```python
fast_executor = executor.replace_lm(dspy.LM("openrouter/anthropic/claude-3-haiku"))
```

### Per-Call Overrides
You can alter LM behavior or provide extra parameters without rebuilding the module.

```python
executor.forward(
    "Summarize the meeting notes",
    config={"temperature": 0.1, "max_tokens": 300},
    context={"stop": ["Observation:"]},
)
```

`call_params` (or keyword arguments) are filtered to match the DSPy predictor’s accepted kwargs, preventing accidental errors.

### Tool-Only Execution
If you want deterministic tool routing, you can set a dummy LM (or a very low-temperature model) and pass pure Python callables.

```python
from roma_dspy import Executor

executor = Executor(
    prediction_strategy="code_act",
    lm=dspy.LM("openrouter/openai/gpt-4o-mini", temperature=0.0),
    tools={"get_weather": get_weather, "lookup_user": lookup_user},
)
```

ROMA will ensure both constructor and per-call tools are available to the strategy.

## Testing

ROMA-DSPy includes a comprehensive test suite covering core functionality, API endpoints, and CLI commands.

### Running Tests

```bash
# Unit tests only (fast, no dependencies)
pytest tests/unit/test_api_schemas.py tests/unit/test_execution_service.py -v

# CLI integration tests
pytest tests/integration/test_cli_commands.py -v

# All passing tests
pytest tests/unit/test_api_schemas.py tests/unit/test_execution_service.py tests/integration/test_cli_commands.py -v
```

### Test Coverage

- **Unit Tests**: 36/36 passing (100%)
  - API schema validation (20 tests)
  - ExecutionService lifecycle (16 tests)

- **CLI Tests**: 22/30 passing (73%)
  - Server and execution management
  - Checkpoint operations
  - Visualization and metrics

- **Overall**: 57/90 tests passing (63%)

### Test Features

- Mock-based testing (no database required)
- pytest + pytest-asyncio for async support
- Fast execution (~0.88s for 66 tests)
- Comprehensive fixtures for API testing

See `CHANGELOG.md` for detailed test documentation.

## Troubleshooting & Tips
- **`ValueError: Either provide an existing lm`** — supply `lm=` or `model=` when constructing the module.
- **`Invalid prediction strategy`** — check spelling; strings are case-insensitive but must match a known alias.
- **Caching** — pass `cache=True` on your LM or set it in `model_config` to reutilize previous completions.
- **Async contexts** — when mixing sync and async calls, ensure your event loop is running (e.g., use `asyncio.run`).
- **Tool duplicates** — tools are deduplicated by identity; create distinct functions if you need variations.

## Glossary

### Core Concepts
- **DSPy**: Stanford's declarative framework for prompting, planning, and tool integration.
- **Prediction Strategy**: The DSPy class/function that powers reasoning (CoT, ReAct, etc.).
- **SubTask**: Pydantic model describing a decomposed unit of work (`goal`, `task_type`, `dependencies`).
- **NodeType**: Whether the Atomizer chose to `PLAN` or `EXECUTE`.
- **TaskType**: MECE classification for subtasks (`RETRIEVE`, `WRITE`, `THINK`, `CODE_INTERPRET`, `IMAGE_GENERATION`).
- **Context Defaults**: Keyword arguments provided to `dspy.context(...)` on every call.

### Configuration & Storage
- **FileStorage**: Execution-scoped storage manager providing isolated directories per task execution.
- **DataStorage**: Automatic Parquet storage system for large toolkit responses (threshold-based).
- **Execution ID**: Unique identifier for each task execution, used for storage isolation.
- **Base Path**: Root directory for all storage operations (local path or S3 bucket).
- **Profile**: Named configuration preset (e.g., `lightweight`, `high_quality`, `crypto_agent`).
- **Configuration Override**: Runtime value that supersedes profile/default settings.

### Toolkits
- **BaseToolkit**: Abstract base class for all toolkits providing storage integration and tool registration.
- **REQUIRES_FILE_STORAGE**: Metadata flag indicating a toolkit requires FileStorage (e.g., FileToolkit).
- **Toolkit Config**: Toolkit-specific settings like API keys, timeouts, and thresholds.
- **Tool Selection**: Include/exclude lists to filter which tools from a toolkit are available.
- **Storage Threshold**: Size limit (KB) above which responses are stored in Parquet format.

### Architecture
- **Execution-Scoped Isolation**: Pattern where each execution gets unique storage directory.
- **Parquet Integration**: Automatic columnar storage for large structured data.
- **S3 Compatibility**: Ability to use S3-compatible storage via Docker volume mounts.
- **Tool Registration**: Automatic discovery and registration of toolkit methods as callable tools.

---

Happy building! If you extend or customize a module, keep the signatures aligned so your higher-level orchestration remains stable.

**Additional Resources:**
- [Configuration System](config/README.md) - Detailed configuration documentation
- [Storage Architecture](reports/STORAGE_ARCHITECTURE.md) - Complete storage system design
- [Setup Guide](reports/SETUP_GUIDE.md) - Dynamic setup and profile management
- [E2B Setup](docs/E2B_SETUP.md) - Code execution toolkit setup
- [Deployment Guide](reports/DEPLOYMENT_GUIDE.md) - Production deployment instructions
- [Quick Start](QUICKSTART.md) - 5-minute local setup guide
