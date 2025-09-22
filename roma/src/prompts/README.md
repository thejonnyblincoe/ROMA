# ROMA Prompt Template Variables Reference

This guide provides comprehensive documentation for all template variables available when building Jinja2 prompts in ROMA v2.0. The ContextBuilderService automatically exports all variables to templates, allowing flexible usage based on your specific needs.

## Quick Start

Every template receives a rich set of variables through the `export_template_variables()` method. Use them in your Jinja2 templates like this:

```jinja2
{# Basic variable usage #}
Goal: {{ goal }}
Task Type: {{ task_type }}
Date: {{ current_date }}

{# Conditional sections #}
{% if has_dependencies %}
## Dependencies
{% for dep in dependency_results %}
- {{ dep.goal }} ({{ dep.status }})
{% endfor %}
{% endif %}

{# Safe access with defaults #}
Project: {{ execution_metadata | safe_get("project_name", "Unknown") }}
```

## Variable Categories

### 1. Core Variables (Always Present)

Essential task information that's available in every template:

#### `task` (TaskNode object)
**Type**: TaskNode object
**Description**: Complete task object with all properties
**Example Value**:
```python
TaskNode(
    task_id="task_abc123",
    goal="Analyze cryptocurrency market trends for Q4 2024",
    task_type=TaskType.THINK,
    status=TaskStatus.EXECUTING,
    parent_id="task_xyz789",
    version=2
)
```
**Template Usage**:
```jinja2
Task ID: {{ task.task_id }}
Full goal: {{ task.goal }}
Task version: {{ task.version }}
```

#### `goal` (string)
**Type**: string
**Description**: Human-readable task objective
**Example Values**:
- `"Analyze cryptocurrency market trends for Q4 2024"`
- `"Write a comprehensive report on AI developments"`
- `"Search for recent papers on quantum computing"`
**Template Usage**:
```jinja2
Your mission: {{ goal }}
You are tasked with: {{ goal }}
```

#### `task_type` (TaskType enum)
**Type**: TaskType enum value
**Description**: MECE task classification
**Possible Values**: `RETRIEVE`, `WRITE`, `THINK`, `CODE_INTERPRET`, `IMAGE_GENERATION`
**Example Values**:
- `TaskType.RETRIEVE` → `"RETRIEVE"`
- `TaskType.THINK` → `"THINK"`
- `TaskType.WRITE` → `"WRITE"`
**Template Usage**:
```jinja2
Task type: {{ task_type }}
{% if task_type == "RETRIEVE" %}
Focus on data gathering and research.
{% elif task_type == "THINK" %}
Focus on analysis and reasoning.
{% endif %}
```

#### `task_status` (TaskStatus enum)
**Type**: TaskStatus enum value
**Description**: Current execution state
**Possible Values**: `PENDING`, `READY`, `EXECUTING`, `COMPLETED`, `FAILED`
**Example Values**:
- `TaskStatus.EXECUTING` → `"EXECUTING"`
- `TaskStatus.PENDING` → `"PENDING"`
**Template Usage**:
```jinja2
Current status: {{ task_status }}
{% if task_status == "EXECUTING" %}
This task is currently being processed.
{% endif %}
```

#### `overall_objective` (string)
**Type**: string
**Description**: Root-level goal from user
**Example Values**:
- `"Create a comprehensive market analysis report"`
- `"Research and implement a new ML model"`
- `"Generate a business proposal for sustainable energy"`
**Template Usage**:
```jinja2
Overall objective: {{ overall_objective }}
{% if overall_objective != goal %}
This subtask supports: {{ overall_objective }}
{% endif %}
```

#### `task_id` (string)
**Type**: string (UUID)
**Description**: Unique task identifier
**Example Values**:
- `"task_f47ac10b-58cc-4372-a567-0e02b2c3d479"`
- `"task_6ba7b810-9dad-11d1-80b4-00c04fd430c8"`
**Template Usage**:
```jinja2
Task ID: {{ task_id }}
Reference: {{ task_id[:8] }}...
```

#### `parent_id` (string or null)
**Type**: string (UUID) or null
**Description**: Parent task identifier
**Example Values**:
- `"task_a1b2c3d4-e5f6-7890-abcd-ef1234567890"` (has parent)
- `null` (root task)
**Template Usage**:
```jinja2
{% if parent_id %}
Parent task: {{ parent_id }}
{% else %}
This is a root task.
{% endif %}
```

#### `is_root_task` (boolean)
**Type**: boolean
**Description**: True if this is the top-level task
**Example Values**: `true`, `false`
**Template Usage**:
```jinja2
{% if is_root_task %}
You are working on the main task.
{% else %}
You are working on a subtask.
{% endif %}
```

### 2. Temporal Context (LLM Grounding)

Time-related variables help ground the LLM in current context:

#### `current_date` (string)
**Type**: string (YYYY-MM-DD)
**Description**: Current date in ISO format
**Example Values**: `"2024-09-19"`, `"2024-12-25"`
**Template Usage**:
```jinja2
Today's date: {{ current_date }}
As of {{ current_date }}, consider current events.
```

#### `current_year` (integer)
**Type**: integer
**Description**: Current year as number
**Example Values**: `2024`, `2025`
**Template Usage**:
```jinja2
Current year: {{ current_year }}
In {{ current_year }}, focus on recent developments.
```

#### `current_timestamp` (string)
**Type**: string (ISO timestamp)
**Description**: Precise current time
**Example Values**:
- `"2024-09-19T14:30:00Z"`
- `"2024-09-19T09:15:30.123Z"`
**Template Usage**:
```jinja2
Timestamp: {{ current_timestamp }}
Generated at: {{ current_timestamp }}
```

#### `execution_started_at` (string)
**Type**: string (ISO timestamp)
**Description**: When this execution began
**Example Values**: `"2024-09-19T14:25:00Z"`
**Template Usage**:
```jinja2
Execution started: {{ execution_started_at }}
Running since: {{ execution_started_at }}
```

#### `time_elapsed` (string)
**Type**: string (duration)
**Description**: Time since execution start
**Example Values**:
- `"00:05:30"` (5 minutes 30 seconds)
- `"00:00:45"` (45 seconds)
**Template Usage**:
```jinja2
{% if time_elapsed %}
Time elapsed: {{ time_elapsed }}
{% if time_elapsed > "00:10:00" %}
⚠️ Task running over 10 minutes.
{% endif %}
{% endif %}
```

### 3. Prior Work Context (Task History)

Access to previous task results and accumulated knowledge:

#### `has_prior_work` (boolean)
**Type**: boolean
**Description**: True if any prior results exist
**Example Values**: `true`, `false`
**Template Usage**:
```jinja2
{% if has_prior_work %}
Previous work is available for context.
{% else %}
Starting fresh with no prior context.
{% endif %}
```

#### `parent_results` (list)
**Type**: list of result objects
**Description**: Results from parent tasks
**Example Value**:
```python
[
    {
        "task_id": "task_parent1",
        "goal": "Research market data sources",
        "content": "Found 5 reliable data sources: Bloomberg, Reuters...",
        "summary": "Identified key financial data sources",
        "task_type": "RETRIEVE",
        "execution_time": 120.5
    }
]
```
**Template Usage**:
```jinja2
{% if parent_results %}
## Context from Parent Tasks
{% for result in parent_results %}
**{{ result.goal }}:**
{{ result.content | truncate(200) }}
{% endfor %}
{% endif %}
```

#### `sibling_results` (list)
**Type**: list of result objects
**Description**: Results from sibling tasks
**Example Value**:
```python
[
    {
        "task_id": "task_sibling1",
        "goal": "Collect Q3 market data",
        "content": "Q3 data shows 15% growth in crypto market...",
        "summary": "Q3 market data collected",
        "task_type": "RETRIEVE"
    },
    {
        "task_id": "task_sibling2",
        "goal": "Analyze competitor strategies",
        "content": "Top 3 competitors focusing on DeFi...",
        "summary": "Competitor analysis complete",
        "task_type": "THINK"
    }
]
```
**Template Usage**:
```jinja2
{% if sibling_results %}
## Related Work
{% for result in sibling_results %}
- **{{ result.goal }}**: {{ result.summary }}
{% endfor %}
{% endif %}
```

#### `relevant_results` (list)
**Type**: list of result objects
**Description**: Contextually relevant outputs from any level
**Example Value**: Similar to sibling_results but includes semantically related tasks
**Template Usage**:
```jinja2
{% if relevant_results %}
## Relevant Context
{% for result in relevant_results %}
{{ result.goal }}: {{ result.content | truncate(150) }}
{% endfor %}
{% endif %}
```

#### `knowledge_store` (dict)
**Type**: dictionary
**Description**: Accumulated knowledge base
**Example Value**:
```python
{
    "market_trends": "Cryptocurrency market showing volatility...",
    "key_metrics": {"btc_price": 45000, "eth_price": 3200},
    "data_sources": ["CoinGecko", "CoinMarketCap", "DeFiPulse"],
    "analysis_framework": "Technical and fundamental analysis approach"
}
```
**Template Usage**:
```jinja2
{% if knowledge_store %}
## Available Knowledge
{% for key, value in knowledge_store.items() %}
**{{ key }}**: {{ value | truncate(100) }}
{% endfor %}
{% endif %}
```

### 4. Enhanced Dependency Context

Comprehensive dependency information with validation status:

#### `has_dependencies` (boolean)
**Type**: boolean
**Description**: True if task has dependencies
**Example Values**: `true`, `false`
**Template Usage**:
```jinja2
{% if has_dependencies %}
This task has dependencies that must be satisfied first.
{% else %}
This task can execute immediately.
{% endif %}
```

#### `dependency_count` (integer)
**Type**: integer
**Description**: Number of dependency tasks
**Example Values**: `0`, `3`, `7`
**Template Usage**:
```jinja2
Dependencies: {{ dependency_count }}
{% if dependency_count > 5 %}
This task has many dependencies ({{ dependency_count }}).
{% endif %}
```

#### `dependency_results` (list)
**Type**: list of dependency objects
**Description**: Full dependency outputs with metadata
**Example Value**:
```python
[
    {
        "dependency_id": "task_dep1",
        "goal": "Fetch latest market data",
        "status": "completed",
        "result_summary": "Retrieved 1000 data points from 5 exchanges",
        "full_result": {"data": [...], "timestamp": "2024-09-19T14:20:00Z"},
        "execution_time": 45.2,
        "task_type": "RETRIEVE",
        "metadata": {"source_count": 5, "data_points": 1000}
    },
    {
        "dependency_id": "task_dep2",
        "goal": "Validate data quality",
        "status": "failed",
        "error": "Data validation failed: 15% missing values",
        "task_type": "THINK",
        "retry_count": 2
    }
]
```
**Template Usage**:
```jinja2
{% if dependency_results %}
## Dependency Status
{% for dep in dependency_results %}
**{{ dep.goal }}** ({{ dep.status }})
{% if dep.status == "completed" %}
✅ {{ dep.result_summary }}
{% elif dep.status == "failed" %}
❌ {{ dep.error }}
{% endif %}
{% endfor %}
{% endif %}
```

#### `dependency_validation` (dict)
**Type**: dictionary
**Description**: Validation summary with execution readiness
**Example Value**:
```python
{
    "status": "failed",
    "message": "Execution blocked: 1 dependency failed, 2 pending",
    "completed_count": 2,
    "failed_count": 1,
    "pending_count": 2,
    "can_execute": False
}
```
**Template Usage**:
```jinja2
{% if dependency_validation %}
## Validation Status
{% if dependency_validation.can_execute %}
✅ Ready to execute
{% else %}
⚠️ {{ dependency_validation.message }}
{% endif %}

Summary: {{ dependency_validation.completed_count }} completed,
{{ dependency_validation.failed_count }} failed,
{{ dependency_validation.pending_count }} pending
{% endif %}
```

#### `completed_dependencies` (list)
**Type**: list of strings
**Description**: IDs of successfully completed dependencies
**Example Value**: `["task_dep1", "task_dep3", "task_dep5"]`
**Template Usage**:
```jinja2
{% if completed_dependencies %}
Completed dependencies: {{ completed_dependencies | join(", ") }}
{% endif %}
```

#### `failed_dependencies` (list)
**Type**: list of strings
**Description**: IDs of failed dependencies
**Example Value**: `["task_dep2", "task_dep4"]`
**Template Usage**:
```jinja2
{% if failed_dependencies %}
⚠️ Failed dependencies: {{ failed_dependencies | join(", ") }}
Consider alternative approaches for these failed components.
{% endif %}
```

#### `dependency_chain_valid` (boolean)
**Type**: boolean
**Description**: True if entire dependency chain is valid
**Example Values**: `true`, `false`
**Template Usage**:
```jinja2
{% if dependency_chain_valid %}
All dependencies satisfied. Proceed with confidence.
{% else %}
Dependency issues detected. Consider partial execution or replanning.
{% endif %}
```

### 5. Tools and Capabilities

Information about available tools and agent capabilities:

#### `has_tools` (boolean)
**Type**: boolean
**Description**: True if tools are available
**Example Values**: `true`, `false`
**Template Usage**:
```jinja2
{% if has_tools %}
External tools are available for enhanced capabilities.
{% else %}
Rely on built-in language model capabilities only.
{% endif %}
```

#### `available_tools` (list)
**Type**: list of strings
**Description**: Names of available tools
**Example Value**: `["web_search", "code_interpreter", "image_generator", "data_api", "knowledge_store"]`
**Template Usage**:
```jinja2
{% if available_tools %}
Available tools: {{ available_tools | join(", ") }}

{% if "web_search" in available_tools %}
Use web_search for current information.
{% endif %}
{% if "code_interpreter" in available_tools %}
Use code_interpreter for data analysis.
{% endif %}
{% endif %}
```

#### `toolkit_count` (integer)
**Type**: integer
**Description**: Number of available tool categories
**Example Values**: `0`, `3`, `8`
**Template Usage**:
```jinja2
Available toolkits: {{ toolkit_count }}
{% if toolkit_count > 5 %}
You have extensive tool access ({{ toolkit_count }} categories).
{% endif %}
```

#### `tools_by_category` (dict)
**Type**: dictionary
**Description**: Tools organized by functional category
**Example Value**:
```python
{
    "research": [
        {"name": "web_search", "description": "Search web for current info"},
        {"name": "academic_search", "description": "Search academic papers"}
    ],
    "analysis": [
        {"name": "code_interpreter", "description": "Execute Python code"},
        {"name": "data_processor", "description": "Process structured data"}
    ],
    "generation": [
        {"name": "image_generator", "description": "Create images from text"},
        {"name": "document_writer", "description": "Generate formatted docs"}
    ]
}
```
**Template Usage**:
```jinja2
{% if tools_by_category %}
## Available Tools by Category
{% for category, tools in tools_by_category.items() %}
**{{ category.title() }}:**
{% for tool in tools %}
- **{{ tool.name }}**: {{ tool.description }}
{% endfor %}
{% endfor %}
{% endif %}
```

### 6. Project and Execution Metadata

System-level context and execution information:

#### `execution_metadata` (dict)
**Type**: dictionary
**Description**: Execution context and settings
**Example Value**:
```python
{
    "execution_id": "exec_abc123",
    "project_name": "Market Analysis Q4",
    "max_tokens": 4000,
    "model_name": "gpt-4",
    "temperature": 0.7,
    "started_at": "2024-09-19T14:25:00Z",
    "config_profile": "general_agent"
}
```
**Template Usage**:
```jinja2
{% if execution_metadata %}
Model: {{ execution_metadata | safe_get("model_name", "Unknown") }}
Max tokens: {{ execution_metadata | safe_get("max_tokens", "Not specified") }}
Profile: {{ execution_metadata | safe_get("config_profile", "Default") }}
{% endif %}
```

#### `project_info` (dict)
**Type**: dictionary
**Description**: Project information and context
**Example Value**:
```python
{
    "name": "Cryptocurrency Market Analysis",
    "description": "Comprehensive analysis of crypto markets for Q4 2024",
    "version": "2.1.0",
    "owner": "research_team",
    "created_date": "2024-09-01",
    "tags": ["crypto", "market-analysis", "Q4-2024"]
}
```
**Template Usage**:
```jinja2
{% if project_info %}
Project: {{ project_info.name }}
{% if project_info.description %}
Description: {{ project_info.description }}
{% endif %}
Version: {{ project_info | safe_get("version", "1.0") }}
{% endif %}
```

#### `execution_id` (string)
**Type**: string
**Description**: Unique execution identifier
**Example Value**: `"exec_f47ac10b-58cc-4372-a567-0e02b2c3d479"`
**Template Usage**:
```jinja2
Execution ID: {{ execution_id }}
Session: {{ execution_id[:8] }}...
```

#### `storage_paths` (dict)
**Type**: dictionary
**Description**: Available storage locations
**Example Value**:
```python
{
    "output_dir": "/tmp/roma_output/exec_abc123",
    "cache_dir": "/tmp/roma_cache",
    "artifacts_dir": "/tmp/roma_artifacts/exec_abc123",
    "logs_dir": "/var/log/roma"
}
```
**Template Usage**:
```jinja2
{% if storage_paths %}
Output directory: {{ storage_paths | safe_get("output_dir", "Not configured") }}
Artifacts: {{ storage_paths | safe_get("artifacts_dir", "Not available") }}
{% endif %}
```

### 7. System Context (Constraints & Preferences)

User preferences and system constraints:

#### `constraints` (list)
**Type**: list of strings
**Description**: System constraints and limitations
**Example Value**:
```python
[
    "Maximum execution time: 10 minutes",
    "No external API calls to paid services",
    "Output must be under 5000 tokens",
    "PII data must be anonymized"
]
```
**Template Usage**:
```jinja2
{% if constraints %}
## System Constraints
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}
```

#### `user_preferences` (dict)
**Type**: dictionary
**Description**: User style and format preferences
**Example Value**:
```python
{
    "output_style": "professional",
    "max_length": "comprehensive",
    "format": "markdown",
    "include_sources": True,
    "detail_level": "high",
    "language": "en",
    "tone": "analytical"
}
```
**Template Usage**:
```jinja2
{% if user_preferences %}
Style: {{ user_preferences | safe_get("output_style", "neutral") }}
Format: {{ user_preferences | safe_get("format", "text") }}
{% if user_preferences.include_sources %}
📚 Include sources and references.
{% endif %}
{% endif %}
```

#### `has_constraints` (boolean)
**Type**: boolean
**Description**: True if constraints exist
**Example Values**: `true`, `false`
**Template Usage**:
```jinja2
{% if has_constraints %}
⚠️ Operating under system constraints. See details above.
{% else %}
No specific constraints apply to this task.
{% endif %}
```

#### `configuration` (dict)
**Type**: dictionary
**Description**: System configuration settings
**Example Value**:
```python
{
    "strict_mode": True,
    "debug_enabled": False,
    "max_retries": 3,
    "timeout_seconds": 300,
    "enable_caching": True,
    "log_level": "INFO"
}
```
**Template Usage**:
```jinja2
{% if configuration %}
{% if configuration.strict_mode %}
⚠️ Strict mode enabled. Follow all guidelines precisely.
{% endif %}
{% if configuration.debug_enabled %}
🐛 Debug mode active.
{% endif %}
{% endif %}
```

### 8. Helper Flags and Computed Values

Convenience flags and computed values for template logic:

#### `has_artifacts` (boolean)
**Type**: boolean
**Description**: True if artifacts are available
**Example Values**: `true`, `false`
**Template Usage**:
```jinja2
{% if has_artifacts %}
📎 Previous artifacts available for reference.
{% else %}
No artifacts from previous executions.
{% endif %}
```

#### `context_priority` (dict)
**Type**: dictionary
**Description**: Context prioritization information
**Example Value**:
```python
{
    "priority_level": "high",
    "boost_factors": ["recent", "parent_task", "same_type"],
    "context_score": 0.85,
    "overflow_handling": "truncate_oldest"
}
```
**Template Usage**:
```jinja2
{% if context_priority %}
Context priority: {{ context_priority | safe_get("priority_level", "normal") }}
Score: {{ context_priority | safe_get("context_score", "unknown") }}
{% endif %}
```

#### `task_complexity` (string)
**Type**: string
**Description**: Estimated task complexity level
**Example Values**: `"low"`, `"moderate"`, `"high"`, `"very_high"`
**Template Usage**:
```jinja2
{% if task_complexity %}
Complexity: {{ task_complexity }}
{% if task_complexity == "high" %}
⚠️ Complex task - consider breaking into subtasks.
{% elif task_complexity == "low" %}
✅ Straightforward task - execute directly.
{% endif %}
{% endif %}
```

#### `estimated_duration` (string)
**Type**: string
**Description**: Estimated execution time
**Example Values**:
- `"2-3 minutes"`
- `"5-10 minutes"`
- `"15-20 minutes"`
**Template Usage**:
```jinja2
{% if estimated_duration %}
Estimated duration: {{ estimated_duration }}
{% if "20" in estimated_duration %}
⏰ Long-running task expected.
{% endif %}
{% endif %}
```

## Jinja2 Filters

ROMA provides custom filters for safe and convenient data handling:

### truncate
Truncate text to specified length with ellipsis:
```jinja2
{{ long_description | truncate(150) }}
{{ result.content | truncate(200) }}

{# Example output #}
"This is a very long description that goes on and on with lots of details about the market analysis and findings from multiple sources and..."
```

### format_list
Format lists with custom prefixes:
```jinja2
{{ available_tools | format_list("🔧 ") }}
{{ constraints | format_list("⚠️ ") }}

{# Example output #}
🔧 web_search
🔧 code_interpreter
🔧 image_generator
```

### safe_get
Safely access dictionary values with defaults:
```jinja2
{{ execution_metadata | safe_get("model_name", "Unknown Model") }}
{{ user_preferences | safe_get("output_format", "markdown") }}

{# Example output when key exists #}
"gpt-4"

{# Example output when key missing #}
"Unknown Model"
```

## Template Design Patterns

### 1. Conditional Sections
Use conditional blocks to show sections only when relevant:

```jinja2
{% if has_dependencies %}
## Dependencies ({{ dependency_count }})
{{ dependency_validation.message }}

{% for dep in dependency_results %}
- **{{ dep.goal }}** ({{ dep.status }})
{% endfor %}
{% endif %}

{% if has_prior_work %}
## Previous Context
{% if parent_results %}
Parent work: {{ parent_results | length }} results
{% endif %}
{% if sibling_results %}
Related work: {{ sibling_results | length }} results
{% endif %}
{% endif %}
```

### 2. Safe Data Access
Always check for data existence before using:

```jinja2
{% if parent_results %}
{% for result in parent_results %}
### {{ result.goal | default("Unnamed Task") }}
{% if result.content %}
{{ result.content | truncate(300) }}
{% else %}
No content available.
{% endif %}
Completed in: {{ result.execution_time | default("unknown") }}s
{% endfor %}
{% endif %}
```

### 3. Progressive Disclosure
Show most important information first, details on demand:

```jinja2
# {{ task_type }} Task: {{ goal }}
Status: {{ task_status }} | Complexity: {{ task_complexity | default("unknown") }}

{% if overall_objective != goal %}
Supporting: {{ overall_objective }}
{% endif %}

{% if has_dependencies and dependency_validation %}
Dependencies: {{ dependency_validation.completed_count }}/{{ dependency_count }} ready
{% endif %}

{% if time_elapsed %}
Runtime: {{ time_elapsed }}
{% endif %}

---

{% if has_dependencies %}
<details>
<summary>Dependency Details ({{ dependency_count }})</summary>

{% for dep in dependency_results %}
**{{ dep.goal }}** - {{ dep.status }}
{% if dep.status == "completed" %}
Result: {{ dep.result_summary }}
{% elif dep.status == "failed" %}
Error: {{ dep.error }}
{% endif %}
{% endfor %}
</details>
{% endif %}
```

### 9. Output Schema and Examples (Response Models)

Critical variables for ensuring agent responses match expected Pydantic model structure:

#### Response Model Schemas

Each agent type has an associated Pydantic model with a JSON schema for output validation:

**`atomizer_schema` (dict)**
**Type**: JSON Schema dictionary
**Description**: AtomizerResult model schema for atomization decisions
**Template Usage**:
```jinja2
Expected output format:
```json
{{ atomizer_schema | tojson(indent=2) }}
```

**`planner_schema` (dict)**
**Type**: JSON Schema dictionary
**Description**: PlannerResult model schema for task decomposition
**Template Usage**:
```jinja2
Your response must follow this exact structure:
{{ planner_schema.properties | list | join(", ") }}
```

**`executor_schema` (dict)**
**Type**: JSON Schema dictionary
**Description**: ExecutorResult model schema for task execution
**Template Usage**:
```jinja2
{% if executor_schema %}
## Required Output Fields
{% for field, details in executor_schema.properties.items() %}
- **{{ field }}**: {{ details.description }}
{% endfor %}
{% endif %}
```

Similar patterns apply for:
- `aggregator_schema` - AggregatorResult model schema
- `plan_modifier_schema` - PlanModifierResult model schema
- `subtask_schema` - SubTask model schema

#### Response Model Examples

Each model provides 5 diverse examples for few-shot learning:

**`atomizer_examples` (list)**
**Type**: List of 5 example AtomizerResult objects
**Description**: Diverse atomization decisions across task types
**Template Usage**:
```jinja2
{% if atomizer_examples %}
## Examples of Correct Responses

{% for example in atomizer_examples[:3] %}
### Example {{ loop.index }}
```json
{{ example | tojson(indent=2) }}
```
{% endfor %}
{% endif %}
```

**`planner_examples` (list)**
**Type**: List of 5 example PlannerResult objects
**Description**: Various planning scenarios and decomposition approaches
**Template Usage**:
```jinja2
## Planning Examples

{% for example in planner_examples %}
**Scenario {{ loop.index }}**: {{ example.reasoning[:100] }}...
- Subtasks: {{ example.subtasks | length }}
- Strategy: Dependencies show {{ "sequential" if example.subtasks[0].dependencies else "parallel" }} execution
{% endfor %}
```

Similar patterns for:
- `executor_examples` - ExecutorResult examples
- `aggregator_examples` - AggregatorResult examples
- `plan_modifier_examples` - PlanModifierResult examples
- `subtask_examples` - SubTask examples

#### Best Practices for Schema and Examples

**1. Always Reference Schema**
```jinja2
Your response MUST be valid JSON matching this schema:
{{ atomizer_schema | tojson(indent=2) }}
```

**2. Use Relevant Examples**
```jinja2
{% if task_type == "RETRIEVE" %}
{% set relevant_examples = atomizer_examples | selectattr("examples", "search", "RETRIEVE") %}
Here are examples for {{ task_type }} tasks:
{% for example in relevant_examples[:2] %}
{{ example | tojson(indent=2) }}
{% endfor %}
{% endif %}
```

**3. Emphasize Required Fields**
```jinja2
{% if planner_schema.required %}
REQUIRED fields: {{ planner_schema.required | join(", ") }}
{% endif %}
```

**4. Show Field Descriptions**
```jinja2
## Field Descriptions
{% for field, details in executor_schema.properties.items() %}
**{{ field }}**: {{ details.description }}
{% if field in executor_schema.required %}(REQUIRED){% endif %}
{% endfor %}
```

This comprehensive guide with detailed examples enables you to leverage all available template variables effectively. Remember that all variables are exported automatically - you choose which ones to use based on your template's specific needs.