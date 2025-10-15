# ROMA-DSPy Execution-Scoped Storage Architecture

## Overview

ROMA-DSPy implements a **fully execution-scoped storage system** that ensures complete isolation between different task executions. All file operations and data storage are automatically scoped to individual execution IDs, preventing cross-execution contamination.

## Architecture Components

### 1. FileStorage (`src/roma_dspy/core/storage/file_storage.py`)

**Purpose**: Provides execution-scoped file system isolation.

**Key Features**:
- Creates unique directory for each execution: `{base_path}/executions/{execution_id}/`
- Automatic directory creation and cleanup
- Thread-safe file operations
- Metadata tracking for all files

**Usage**:
```python
from roma_dspy.core.storage import FileStorage
from roma_dspy.config.schemas.storage import StorageConfig

config = StorageConfig(base_path="/opt/sentient")
storage = FileStorage(config=config, execution_id="exec_12345")

# All operations are scoped to: /opt/sentient/executions/exec_12345/
```

### 2. DataStorage (`src/roma_dspy/tools/utils/storage.py`)

**Purpose**: Automatic threshold-based Parquet storage for large toolkit responses.

**Key Features**:
- Automatic serialization to Parquet when data exceeds threshold (default: 1MB)
- Toolkit-specific subdirectories
- Nested data type organization
- Size tracking and metadata

**Usage**:
```python
from roma_dspy.tools.utils.storage import DataStorage

data_storage = DataStorage(
    file_storage=file_storage,
    toolkit_name="coingecko",
    threshold_kb=500  # Store if > 500KB
)

# Automatic storage if data exceeds threshold
if data_storage.should_store(large_data):
    path, size = await data_storage.store_parquet(
        data=large_data,
        data_type="market_charts",
        prefix="btc_usd_30d"
    )
```

### 3. BaseToolkit (`src/roma_dspy/tools/base/base.py`)

**Purpose**: Base class for all toolkits with automatic storage integration.

**Key Features**:
- Automatic FileStorage injection via constructor
- Optional DataStorage for threshold-based Parquet storage
- `REQUIRES_FILE_STORAGE` flag for toolkits that need FileStorage
- Helper methods for building responses with automatic storage

**Storage Attributes**:
- `_file_storage`: Raw FileStorage instance (for direct access)
- `_data_storage`: DataStorage wrapper (for automatic Parquet storage)

### 4. FileToolkit (`src/roma_dspy/tools/core/file.py`)

**Purpose**: Provides file operations within execution-scoped directories.

**Key Changes** (v2):
- **REQUIRES** FileStorage - no fallback to config
- Automatically uses FileStorage root directory
- No `base_directory` configuration needed
- All file operations isolated to execution directory

**Configuration**:
```yaml
- class_name: FileToolkit
  enabled: true
  toolkit_config:
    enable_delete: false  # Operational setting
    max_file_size: 10485760  # Operational setting (10MB)
    # NO base_directory needed!
```

### 5. ToolkitManager (`src/roma_dspy/tools/base/manager.py`)

**Purpose**: Manages toolkit lifecycle and FileStorage injection.

**Key Features**:
- Validates `REQUIRES_FILE_STORAGE` flag
- Injects FileStorage into all toolkits during creation
- Execution-scoped toolkit caching
- Automatic cleanup after execution

**Validation**:
```python
# ToolkitManager validates FileStorage requirement
requires_file_storage = getattr(toolkit_class, "REQUIRES_FILE_STORAGE", False)
if requires_file_storage and not file_storage:
    raise ValueError(
        f"{class_name} requires FileStorage but none was provided"
    )
```

## Storage Hierarchy

```
{STORAGE_BASE_PATH}/
└── executions/
    ├── exec_001/
    │   ├── data/
    │   │   ├── coingecko/
    │   │   │   ├── market_charts/
    │   │   │   │   └── btc_usd_30d_20250110_123456.parquet
    │   │   │   └── prices/
    │   │   ├── defillama/
    │   │   └── arkham/
    │   ├── artifacts/
    │   ├── test.txt  (FileToolkit operations)
    │   └── report.json  (FileToolkit operations)
    └── exec_002/
        └── ...  (completely isolated)
```

## Configuration

### Storage Configuration (`config/profiles/*.yaml`)

```yaml
# Global storage settings
storage:
  base_path: ${STORAGE_BASE_PATH:-/opt/sentient}
  max_file_size: 104857600  # 100MB

  postgres:
    enabled: ${POSTGRES_ENABLED:-true}
    connection_url: ${DATABASE_URL}
```

### Toolkit Configuration (No base_directory needed!)

```yaml
agents:
  executor:
    toolkits:
      # FileToolkit - NO base_directory!
      - class_name: FileToolkit
        enabled: true
        toolkit_config:
          enable_delete: false
          max_file_size: 10485760  # 10MB

      # Data toolkits - optional storage threshold
      - class_name: CoinGeckoToolkit
        enabled: true
        toolkit_config:
          storage_threshold_kb: 500  # Store responses > 500KB
          enable_analysis: true
```

## S3 Integration

### Docker Environment

In Docker, S3 is mounted as a volume:

```yaml
# docker-compose.yaml
volumes:
  - s3_data:/opt/sentient  # S3 mounted here
```

### Environment Variables

```bash
# S3 configuration
export STORAGE_BASE_PATH=/opt/sentient  # S3 mount point
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
export AWS_REGION=us-east-1
```

### S3 File Paths

All execution files are automatically stored in S3:

```
s3://your-bucket/
└── executions/
    ├── exec_001/
    │   ├── data/  # Parquet files from toolkits
    │   └── ...  # FileToolkit operations
    └── exec_002/
        └── ...
```

## Data Toolkit Integration

All data-fetching toolkits (Arkham, DefiLlama, CoinGecko, Binance) automatically:

1. **Receive FileStorage** via ToolkitManager
2. **Create DataStorage** wrapper with threshold
3. **Auto-store large responses** as Parquet files
4. **Return file_path** instead of inline data when stored

### Example Response

**Small response (< threshold)**:
```json
{
  "success": true,
  "data": {...},  # Inline data
  "toolkit": "CoinGeckoToolkit",
  "execution_id": "exec_123"
}
```

**Large response (> threshold)**:
```json
{
  "success": true,
  "file_path": "executions/exec_123/data/coingecko/market_charts/btc_usd_30d_20250110.parquet",
  "stored": true,
  "size_kb": 1234.5,
  "message": "Response data (1234.5KB) exceeds threshold and has been saved to: ...",
  "toolkit": "CoinGeckoToolkit",
  "execution_id": "exec_123"
}
```

## Key Design Decisions

### 1. **FileStorage is Required for FileToolkit**

**Rationale**: Ensures execution isolation and prevents cross-execution file contamination.

**Implementation**:
- `FileToolkit.REQUIRES_FILE_STORAGE = True`
- ToolkitManager validates this requirement
- No fallback to config-based `base_directory`

### 2. **No base_directory Configuration**

**Rationale**: Base directory should come from execution context, not static config.

**Migration**:
```yaml
# ❌ Old (removed)
- class_name: FileToolkit
  toolkit_config:
    base_directory: ${STORAGE_BASE_PATH}/executions

# ✅ New (automatic)
- class_name: FileToolkit
  toolkit_config:
    enable_delete: false
    max_file_size: 10485760
```

### 3. **Automatic Parquet Storage**

**Rationale**: Large API responses should be stored as files, not passed inline.

**Benefits**:
- Reduces memory usage
- Enables analysis of large datasets
- Persistent storage on S3
- Automatic serialization

### 4. **Execution-Scoped Everything**

**Rationale**: Complete isolation prevents bugs and enables concurrent executions.

**Scope**:
- All file operations
- All toolkit data storage
- All temporary artifacts
- All logs and metadata

## Validation

Run the validation script to verify the storage system:

```bash
uv run python scripts/validate_storage_system.py
```

**Expected Output**:
```
✅ ALL VALIDATIONS PASSED

Summary:
  ✓ FileStorage creates execution-scoped directories
  ✓ FileToolkit requires FileStorage (no fallback)
  ✓ FileToolkit uses execution-scoped directories
  ✓ No base_directory configuration needed
  ✓ File operations work correctly with isolation
```

## Migration Guide

### From Old System

**Step 1**: Remove `base_directory` from toolkit configs:

```yaml
# Remove this parameter
toolkit_config:
  base_directory: ${STORAGE_BASE_PATH}/executions  # DELETE
```

**Step 2**: Ensure FileStorage is provided:

```python
# In solver/engine
file_storage = FileStorage(config=config.storage, execution_id=dag.execution_id)

# ToolkitManager automatically injects FileStorage
tools = await toolkit_manager.get_tools_for_execution(
    execution_id=dag.execution_id,
    file_storage=file_storage,  # Required!
    toolkit_configs=agent_config.toolkits
)
```

**Step 3**: Update tests to provide FileStorage:

```python
# In tests
storage = FileStorage(config=storage_config, execution_id="test_exec")
toolkit = manager._create_toolkit_instance(
    class_name="FileToolkit",
    config=toolkit_config,
    file_storage=storage  # Must provide
)
```

## Benefits

1. **Complete Isolation**: No cross-execution contamination
2. **S3 Compatible**: Works seamlessly with mounted S3 volumes
3. **Automatic Storage**: Large responses stored as Parquet automatically
4. **No Configuration**: Base directories handled automatically
5. **Memory Efficient**: Large datasets stored as files, not in memory
6. **Concurrent Safe**: Multiple executions can run simultaneously
7. **Audit Trail**: All files tracked with execution metadata

## Summary

The ROMA-DSPy storage architecture provides:

- ✅ **Execution-scoped isolation** for all file operations
- ✅ **Automatic FileStorage injection** by ToolkitManager
- ✅ **No manual base_directory configuration** needed
- ✅ **Threshold-based Parquet storage** for large data
- ✅ **S3 integration** via volume mounts
- ✅ **Complete thread safety** for concurrent executions
- ✅ **Validation tooling** to verify correctness

All toolkits automatically receive execution-scoped storage, ensuring clean separation between different task executions.
