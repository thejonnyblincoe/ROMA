# ROMA v2.0 Prompt System Architecture

## Overview

ROMA v2.0 features a sophisticated Jinja2-based prompt system that provides intelligent, context-aware prompts for all agent types. The system automatically exports rich context variables and uses template inheritance for maintainable, consistent agent interactions.

## Architecture Components

### 1. PromptTemplateManager

**Location**: `src/roma/infrastructure/prompts/prompt_template_manager.py`

The central orchestrator for all template operations:

```python
from src.roma.infrastructure.prompts.prompt_template_manager import PromptTemplateManager

# Initialize with context builder for variable export
template_manager = PromptTemplateManager(
    templates_dir="/path/to/templates",
    context_builder=context_builder_service
)

# Render agent prompt with all available variables
prompt = await template_manager.render_agent_prompt(
    agent_type="atomizer",
    task_type="retrieve",
    task=task_node,
    task_context=context
)
```

**Key Features**:
- Template caching for performance
- Jinja2 environment with custom filters
- Automatic variable export integration
- Error handling and validation
- Template hot-reloading for development

### 2. ContextBuilderService Integration

**Location**: `src/roma/application/services/context_builder_service.py`

Exports comprehensive template variables through `export_template_variables()`:

```python
# Automatic variable export for templates
template_vars = await context_builder.export_template_variables(task, context)

# Returns 8 categories of variables:
# 1. Core variables (task, goal, task_type, etc.)
# 2. Temporal context (current_date, timestamps)
# 3. Prior work context (parent_results, sibling_results)
# 4. Dependency context (dependency_results, validation)
# 5. Tools and capabilities (available_tools, toolkits)
# 6. Project metadata (execution_id, project_info)
# 7. System metadata (constraints, preferences)
# 8. Helper flags and computed values
```

### 3. Template Directory Structure

```
src/prompts/
├── README.md                    # Template variables documentation
├── base/                        # Base templates for inheritance
│   ├── base_agent.jinja2        # Common agent introduction
│   ├── base_context.jinja2      # Standard context display
│   └── base_instructions.jinja2 # Common output guidelines
├── helpers/                     # Reusable template components
│   ├── temporal_awareness.jinja2
│   ├── task_type_info.jinja2
│   ├── dependency_display.jinja2
│   └── tools_listing.jinja2
├── atomizer/                    # Task atomization decisions
│   ├── retrieve.jinja2
│   ├── write.jinja2
│   ├── think.jinja2
│   ├── code_interpret.jinja2
│   └── image_generation.jinja2
├── planner/                     # Task decomposition plans
├── executor/                    # Task execution
├── aggregator/                  # Result synthesis
└── plan_modifier/              # Plan adjustments
```

## Template Workflow

### 1. Template Resolution

```python
def get_template_path(agent_type: str, task_type: str) -> str:
    return f"{agent_type.lower()}/{task_type.lower()}.jinja2"

# Examples:
# atomizer + RETRIEVE → "atomizer/retrieve.jinja2"
# planner + WRITE → "planner/write.jinja2"
# executor + THINK → "executor/think.jinja2"
```

### 2. Variable Export

```python
# ContextBuilderService exports ALL available variables
template_vars = await context_builder.export_template_variables(task, context)

# Key variable categories:
{
    # Core task information
    "task": task_node,
    "goal": "Analyze market trends",
    "task_type": "THINK",
    "overall_objective": "Create market report",

    # Temporal context
    "current_date": "2024-09-19",
    "current_year": 2024,
    "current_timestamp": "2024-09-19T10:30:00Z",

    # Prior work context
    "has_prior_work": True,
    "parent_results": [...],
    "sibling_results": [...],

    # Dependency context (Enhanced Dependency Resolution)
    "has_dependencies": True,
    "dependency_results": [...],
    "dependency_validation": {...},

    # Available capabilities
    "has_toolkits": True,
    "available_tools": [...],

    # System context
    "execution_metadata": {...},
    "constraints": [...],
    "user_preferences": {...}
}
```

### 3. Template Rendering

```jinja2
{# Template example: atomizer/think.jinja2 #}
{% extends "base/base_agent.jinja2" %}

{% block agent_introduction %}
You are an Atomizer agent specializing in ANALYSIS task decomposition.
Today's date: {{ current_date }}
{% endblock %}

{% block task_information %}
## Current Analysis Task
Goal: {{ goal }}
{% if overall_objective %}Supporting: {{ overall_objective }}{% endif %}
{% endblock %}

{% block context_display %}
{% if has_dependencies %}
{% include "helpers/dependency_display.jinja2" %}
{% endif %}

{% if has_prior_work %}
{% include "base/base_context.jinja2" %}
{% endif %}
{% endblock %}

{% block instructions %}
{% include "base/base_instructions.jinja2" %}
{% endblock %}
```

## Template Variable Categories

### 1. Essential Core Variables (Always Present)

```jinja2
{{ task }}                    # TaskNode object
{{ goal }}                    # Task goal string
{{ task_type }}              # Task type (RETRIEVE, WRITE, etc.)
{{ task_status }}            # Current task status
{{ overall_objective }}      # Root objective
{{ task_id }}               # Unique task identifier
{{ parent_id }}             # Parent task ID (null for root)
{{ is_root_task }}          # Boolean flag
```

### 2. Temporal Context (LLM Grounding)

```jinja2
{{ current_date }}          # YYYY-MM-DD format
{{ current_year }}          # Integer year
{{ current_timestamp }}     # ISO timestamp
{{ execution_started_at }}  # When execution began
{{ time_elapsed }}          # Duration since start
```

### 3. Prior Work Context (Task History)

```jinja2
{{ has_prior_work }}        # Boolean flag
{{ parent_results }}        # List of parent task outputs
{{ sibling_results }}       # List of sibling task outputs
{{ relevant_results }}      # Contextually relevant outputs
{{ knowledge_store }}       # Accumulated knowledge
```

### 4. Enhanced Dependency Context

```jinja2
{{ has_dependencies }}              # Boolean flag
{{ dependency_count }}              # Number of dependencies
{{ dependency_results }}            # Full dependency outputs
{{ dependency_validation }}         # Validation status
{{ completed_dependencies }}        # Successfully completed deps
{{ failed_dependencies }}          # Failed dependencies
{{ dependency_chain_valid }}       # Overall chain validity
```

### 5. Tools and Capabilities

```jinja2
{{ has_toolkits }}          # Boolean flag
{{ available_tools }}       # List of available tools
{{ toolkit_count }}         # Number of available toolkits
{{ tools_by_category }}     # Tools grouped by type
```

### 6. Project and Execution Metadata

```jinja2
{{ execution_metadata }}    # Execution context
{{ project_info }}         # Project information
{{ execution_id }}         # Current execution ID
{{ storage_paths }}        # Available storage locations
```

### 7. System Context (Constraints & Preferences)

```jinja2
{{ constraints }}           # System constraints
{{ user_preferences }}     # User preferences
{{ has_constraints }}      # Boolean flag
{{ configuration }}        # System configuration
```

### 8. Helper Flags and Computed Values

```jinja2
{{ has_artifacts }}        # Boolean flag for artifacts
{{ context_priority }}     # Context prioritization info
{{ task_complexity }}      # Estimated complexity
{{ estimated_duration }}   # Estimated execution time
```

## Custom Jinja2 Filters

### Built-in Filters

```jinja2
{{ long_text | truncate(200) }}           # Truncate to 200 chars
{{ items_list | format_list("- ") }}      # Format as bulleted list
{{ data_dict | safe_get("key", "default") }} # Safe dictionary access
```

### Filter Implementations

```python
def _truncate_filter(self, text: str, length: int = 200) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= length:
        return text
    return text[:length] + "..."

def _format_list_filter(self, items: list, prefix: str = "- ") -> str:
    """Format list as string with prefix for each item."""
    if not items:
        return ""
    return "\n".join([f"{prefix}{item}" for item in items])

def _safe_get_filter(self, obj: dict, key: str, default: str = "") -> str:
    """Safely get value from dict with default fallback."""
    if not isinstance(obj, dict):
        return default
    return str(obj.get(key, default))
```

## Template Design Patterns

### 1. Template Inheritance

```jinja2
{# Base template: base/base_agent.jinja2 #}
<!DOCTYPE prompt>
<prompt>
{% block agent_introduction %}{% endblock %}
{% block task_information %}{% endblock %}
{% block context_display %}{% endblock %}
{% block instructions %}{% endblock %}
</prompt>

{# Specific agent template extends base #}
{% extends "base/base_agent.jinja2" %}
{% block agent_introduction %}
You are a {{ agent_type }} agent...
{% endblock %}
```

### 2. Conditional Rendering

```jinja2
{# Only show section if data exists #}
{% if has_dependencies %}
## Dependencies
{% for dep in dependency_results %}
- {{ dep.goal }} ({{ dep.status }})
{% endfor %}
{% endif %}

{# Conditional warnings #}
{% if dependency_validation.failed_count > 0 %}
⚠️ Warning: {{ dependency_validation.failed_count }} dependencies failed
{% endif %}
```

### 3. Dynamic Content Inclusion

```jinja2
{# Include helpers based on available data #}
{% if has_prior_work %}
{% include "helpers/context_display.jinja2" %}
{% endif %}

{% if has_toolkits %}
{% include "helpers/tools_listing.jinja2" %}
{% endif %}

{# Template composition #}
{% block context_display %}
{% include "base/base_context.jinja2" %}
{% if has_dependencies %}
{% include "helpers/dependency_display.jinja2" %}
{% endif %}
{% endblock %}
```

### 4. Data Formatting

```jinja2
{# Format complex data structures #}
{% if parent_results %}
## Context from Parent Tasks
{% for result in parent_results %}
### {{ result.goal }}
{{ result.content | truncate(300) }}
{% endfor %}
{% endif %}

{# List formatting with custom prefix #}
{{ available_tools | format_list("🔧 ") }}

{# Safe data access #}
Execution started: {{ execution_metadata | safe_get("started_at", "Unknown") }}
```

## Agent-Specific Template Patterns

### 1. Atomizer Templates

Focus on decision-making criteria:

```jinja2
## Atomization Framework for {{ task_type.upper() }}

**ATOMIC {{ task_type.upper() }}** (Execute directly):
{% for indicator in current_task_type_info.atomic_indicators %}
- {{ indicator }}
{% endfor %}

**COMPOSITE {{ task_type.upper() }}** (Requires decomposition):
{% for indicator in current_task_type_info.composite_indicators %}
- {{ indicator }}
{% endfor %}

## Analysis Questions
{% include "helpers/atomization_questions.jinja2" %}
```

### 2. Planner Templates

Focus on decomposition strategy:

```jinja2
## Planning Guidelines
- Break into 3-6 subtasks
- Minimize dependencies for parallel execution
- Self-contained task goals
- Clear task type assignments ({{ ", ".join(task_types_info.keys()) }})

{% if has_dependencies %}
## Dependency Constraints
{% include "helpers/dependency_display.jinja2" %}
{% endif %}
```

### 3. Executor Templates

Focus on execution instructions:

```jinja2
## Task Execution Context
Goal: {{ goal }}
Type: {{ task_type }}

{% if has_toolkits %}
## Available Tools
{% include "helpers/tools_listing.jinja2" %}
{% endif %}

## Output Requirements
{% include "base/base_instructions.jinja2" %}
```

### 4. Aggregator Templates

Focus on synthesis requirements:

```jinja2
## Aggregation Context
{% if sibling_results %}
## Results to Synthesize
{% for result in sibling_results %}
### {{ result.goal }}
{{ result.content | truncate(200) }}
{% endfor %}
{% endif %}

## Synthesis Guidelines
- Maintain factual accuracy
- Preserve key insights
- Create coherent narrative
- Address original objective: {{ overall_objective }}
```

## Performance Considerations

### Template Caching

```python
# Templates are cached after first load
self._template_cache: Dict[str, Template] = {}

# Cache hit for subsequent renders
if template_path in self._template_cache:
    return self._template_cache[template_path]
```

### Variable Export Optimization

```python
# Variables exported once per context build
template_vars = await context_builder.export_template_variables(task, context)

# Reused across multiple template renders
prompt1 = template1.render(**template_vars)
prompt2 = template2.render(**template_vars)
```

### Memory Management

```python
# Clear cache when needed
template_manager.clear_cache()

# Reload specific template
template_manager.reload_template("atomizer/retrieve.jinja2")
```

## Development Workflow

### 1. Template Development

```python
# Check if template exists
if template_manager.template_exists("atomizer/new_task.jinja2"):
    template = template_manager.load_template("atomizer/new_task.jinja2")

# Validate template syntax
is_valid = template_manager.validate_template("atomizer/new_task.jinja2")

# List available templates
templates = template_manager.list_templates("atomizer")
```

### 2. Testing Templates

```python
# Test with sample context
context = {
    "goal": "Test task",
    "task_type": "THINK",
    "current_date": "2024-09-19"
}

rendered = template_manager.render_template("atomizer/think.jinja2", context)
print(rendered)
```

### 3. Template Debugging

```python
# Get cache information
cache_info = template_manager.get_cache_info()
print(f"Cached templates: {cache_info['cached_templates']}")

# Check template paths
template_path = template_manager.get_default_template_path("executor", "RETRIEVE")
print(f"Expected path: {template_path}")
```

## Integration with Agent System

### Agent Runtime Integration

```python
# PromptTemplateManager is injected into AgentRuntimeService
class AgentRuntimeService:
    def __init__(self, prompt_manager: PromptTemplateManager):
        self.prompt_manager = prompt_manager

    async def get_agent_prompt(self, agent_type, task_type, task, context):
        return await self.prompt_manager.render_agent_prompt(
            agent_type, task_type, task, context
        )
```

### SystemManager Integration

```python
# SystemManager coordinates template rendering with execution
class SystemManager:
    async def execute_task(self, task):
        # Context built with all available variables
        context = await self.context_builder.build_context(task)

        # Prompt rendered with full context
        prompt = await self.prompt_manager.render_agent_prompt(
            agent_type, task.task_type.value, task, context
        )

        # Agent executed with rendered prompt
        result = await self.agent_runtime.execute_agent(agent, prompt)
```

## Error Handling

### Template Loading Errors

```python
try:
    template = template_manager.load_template("invalid/path.jinja2")
except FileNotFoundError:
    # Template file doesn't exist
    pass
except TemplateError:
    # Invalid Jinja2 syntax
    pass
```

### Rendering Errors

```python
try:
    rendered = await template_manager.render_agent_prompt(
        "atomizer", "INVALID_TYPE", task, context
    )
except TemplateError as e:
    # Variable missing or rendering failed
    logger.error(f"Template rendering failed: {e}")
```

### Graceful Fallbacks

```python
# PromptTemplateManager provides basic variables if ContextBuilderService unavailable
if self.context_builder:
    template_vars = await self.context_builder.export_template_variables(task, context)
else:
    template_vars = self._get_basic_template_variables(agent_type, task_type, task, context)
```

## Best Practices

### 1. Template Design

- ✅ Use template inheritance for consistency
- ✅ Include conditional sections for optional data
- ✅ Provide clear instructions and examples
- ✅ Use meaningful variable names
- ✅ Add comments for complex logic

### 2. Variable Usage

- ✅ Check existence before using (`{% if variable %}`)
- ✅ Provide fallbacks for optional data
- ✅ Use filters for safe data formatting
- ✅ Leverage helper flags for conditional rendering

### 3. Performance

- ✅ Cache templates after loading
- ✅ Minimize template complexity
- ✅ Reuse variable exports across templates
- ✅ Clear cache periodically in development

### 4. Maintainability

- ✅ Use descriptive template names
- ✅ Document complex template logic
- ✅ Test templates with various input scenarios
- ✅ Version control template changes

This prompt system provides a powerful, flexible foundation for ROMA v2's agent interactions while maintaining consistency, performance, and ease of development.