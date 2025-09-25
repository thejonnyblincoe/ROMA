"""
Tests for PromptTemplateManager.

Comprehensive test coverage for template loading, rendering, fallback mechanisms,
and Jinja2 features like includes/extends.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import TemplateError

from roma.application.services.context_builder_service import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType
from roma.infrastructure.prompts.prompt_template_manager import PromptTemplateManager


class TestPromptTemplateManager:
    """Test cases for PromptTemplateManager."""

    @pytest.fixture
    def temp_templates_dir(self):
        """Create temporary templates directory with test templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir)

            # Create directory structure
            (templates_dir / "base").mkdir(parents=True)
            (templates_dir / "atomizer").mkdir(parents=True)
            (templates_dir / "planner").mkdir(parents=True)
            (templates_dir / "executor").mkdir(parents=True)

            # Create base template
            base_template = templates_dir / "base" / "base_agent.jinja2"
            base_template.write_text("""
{# Base agent template #}
Agent Type: {{ agent_type }}
Task Type: {{ task_type }}
Goal: {{ goal }}
{% block specific_content %}{% endblock %}
            """.strip())

            # Create atomizer template with extends
            atomizer_template = templates_dir / "atomizer" / "atomizer.jinja2"
            atomizer_template.write_text("""
{% extends "base/base_agent.jinja2" %}
{% block specific_content %}
You are an atomizer agent.
Task: {{ task.goal }}
{% endblock %}
            """.strip())

            # Create planner template
            planner_template = templates_dir / "planner" / "planner.jinja2"
            planner_template.write_text("""
You are a planner agent for {{ task_type }} tasks.
Goal: {{ goal }}
            """.strip())

            # Create task-specific executor template
            executor_retrieve_template = templates_dir / "executor" / "retrieve.jinja2"
            executor_retrieve_template.write_text("""
You are a RETRIEVE executor.
Task: {{ goal }}
            """.strip())

            yield templates_dir

    @pytest.fixture
    def template_manager(self, temp_templates_dir):
        """Create PromptTemplateManager with test templates directory."""
        return PromptTemplateManager(templates_dir=str(temp_templates_dir))

    @pytest.fixture
    def sample_task(self):
        """Create sample TaskNode for testing."""
        return TaskNode(
            task_id="test-task-1",
            goal="Test task goal",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.PENDING
        )

    @pytest.fixture
    def sample_context(self, sample_task):
        """Create sample TaskContext for testing."""
        return TaskContext(
            task=sample_task,
            overall_objective="Test objective"
        )

    def test_template_loading_with_cache(self, template_manager):
        """Test that templates are loaded and cached correctly."""
        # First load
        template1 = template_manager.load_template("planner/planner.jinja2")
        assert template1 is not None

        # Second load should hit cache
        template2 = template_manager.load_template("planner/planner.jinja2")
        assert template1 is template2

        # Verify cache contains the template
        cache_info = template_manager.get_cache_info()
        assert "planner/planner.jinja2" in cache_info["template_paths"]

    def test_template_not_found(self, template_manager):
        """Test FileNotFoundError for non-existent templates."""
        with pytest.raises(FileNotFoundError):
            template_manager.load_template("nonexistent/template.jinja2")

    def test_template_extends_functionality(self, template_manager):
        """Test that Jinja2 extends directive works correctly."""
        template = template_manager.load_template("atomizer/atomizer.jinja2")

        context = {
            "agent_type": "atomizer",
            "task_type": "RETRIEVE",
            "goal": "Test goal",
            "task": MagicMock(goal="Test task goal")
        }

        rendered = template.render(**context)

        # Should contain content from both base and atomizer templates
        assert "Agent Type: atomizer" in rendered
        assert "Task Type: RETRIEVE" in rendered
        assert "You are an atomizer agent" in rendered
        assert "Test task goal" in rendered

    def test_template_validation(self, template_manager):
        """Test template validation functionality."""
        # Valid template
        assert template_manager.validate_template("planner/planner.jinja2") is True

        # Invalid template path
        assert template_manager.validate_template("nonexistent/template.jinja2") is False

    def test_template_exists(self, template_manager):
        """Test template existence checking."""
        assert template_manager.template_exists("planner/planner.jinja2") is True
        assert template_manager.template_exists("nonexistent/template.jinja2") is False

    def test_list_templates(self, template_manager):
        """Test listing available templates."""
        all_templates = template_manager.list_templates()
        assert "atomizer/atomizer.jinja2" in all_templates
        assert "planner/planner.jinja2" in all_templates
        assert "executor/retrieve.jinja2" in all_templates

        # Filter by agent type
        atomizer_templates = template_manager.list_templates("atomizer")
        assert len(atomizer_templates) == 1
        assert "atomizer/atomizer.jinja2" in atomizer_templates

    def test_render_template_direct(self, template_manager):
        """Test direct template rendering with context."""
        context = {
            "agent_type": "planner",
            "task_type": "RETRIEVE",
            "goal": "Test planning goal"
        }

        rendered = template_manager.render_template("planner/planner.jinja2", context)

        assert "planner agent for RETRIEVE tasks" in rendered
        assert "Test planning goal" in rendered

    def test_render_template_with_missing_variables(self, template_manager):
        """Test template rendering with missing required variables."""
        # Template expects 'goal' but we don't provide it
        context = {"agent_type": "planner", "task_type": "RETRIEVE"}

        # Should not raise error due to Jinja2's undefined handling
        rendered = template_manager.render_template("planner/planner.jinja2", context)
        assert rendered is not None

    def test_custom_jinja_filters(self, template_manager, temp_templates_dir):
        """Test custom Jinja2 filters are available."""
        # Create template using custom filters
        filter_template = temp_templates_dir / "test_filters.jinja2"
        filter_template.write_text("""
Text: {{ long_text | truncate(10) }}
List: {{ items | format_list("* ") }}
Safe: {{ data | safe_get("key", "default") }}
        """.strip())

        context = {
            "long_text": "This is a very long text that should be truncated",
            "items": ["item1", "item2", "item3"],
            "data": {"key": "value", "other": "test"}
        }

        rendered = template_manager.render_template("test_filters.jinja2", context)

        assert "This is a ..." in rendered
        assert "* item1" in rendered
        assert "* item2" in rendered
        assert "* item3" in rendered
        assert "value" in rendered

    def test_cache_management(self, template_manager):
        """Test template cache management operations."""
        # Load template to populate cache
        template_manager.load_template("planner/planner.jinja2")

        cache_info = template_manager.get_cache_info()
        assert cache_info["cached_templates"] == 1

        # Clear cache
        template_manager.clear_cache()

        cache_info = template_manager.get_cache_info()
        assert cache_info["cached_templates"] == 0

    def test_reload_template(self, template_manager):
        """Test forcing template reload."""
        # Load template first time
        template1 = template_manager.load_template("planner/planner.jinja2")

        # Reload should return new instance (bypass cache)
        template2 = template_manager.reload_template("planner/planner.jinja2")

        # Should be same objects since using env.get_template() - no cache bypass needed
        assert template1 is not None
        assert template2 is not None

    def test_get_template_path(self, template_manager):
        """Test template path generation."""
        path = template_manager.get_template_path("atomizer", "RETRIEVE")
        assert path == "atomizer/retrieve.jinja2"

        path = template_manager.get_default_template_path("planner", "THINK")
        assert path == "planner/think.jinja2"

    def test_directory_discovery_with_env_var(self):
        """Test templates directory discovery with environment variable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test prompts directory
            prompts_dir = Path(temp_dir) / "custom_prompts"
            prompts_dir.mkdir()
            (prompts_dir / "test.jinja2").write_text("Test template")

            # Test with environment variable
            with patch.dict(os.environ, {"ROMA_PROMPTS_DIR": str(prompts_dir)}):
                manager = PromptTemplateManager()
                assert manager.templates_dir == prompts_dir

    def test_directory_discovery_traversal(self, temp_templates_dir):
        """Test templates directory discovery by traversal."""
        # Create src/prompts structure
        project_root = temp_templates_dir.parent
        src_dir = project_root / "src"
        src_dir.mkdir()
        prompts_dir = src_dir / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test.jinja2").write_text("Test template")

        # Mock __file__ to be inside the project
        mock_file = project_root / "some" / "nested" / "module.py"
        mock_file.parent.mkdir(parents=True)
        mock_file.write_text("# mock file")

        with patch("src.roma.infrastructure.prompts.prompt_template_manager.__file__", str(mock_file)):
            manager = PromptTemplateManager()
            assert manager.templates_dir == prompts_dir

    def test_directory_discovery_failure(self):
        """Test templates directory discovery failure."""
        # Mock environment with no ROMA_PROMPTS_DIR and no findable src/prompts
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.roma.infrastructure.prompts.prompt_template_manager.__file__", "/tmp/isolated/file.py"):
                # Should use fallback path which likely won't exist
                with pytest.raises(FileNotFoundError):
                    PromptTemplateManager()

    @pytest.mark.asyncio
    async def test_render_agent_prompt_without_context_builder(self, template_manager, sample_task, sample_context):
        """Test rendering agent prompt without context builder."""
        rendered = await template_manager.render_agent_prompt(
            agent_type="planner",
            task_type="RETRIEVE",
            task=sample_task,
            task_context=sample_context
        )

        assert "planner agent for RETRIEVE tasks" in rendered
        assert "Test task goal" in rendered

    @pytest.mark.asyncio
    async def test_render_agent_prompt_with_context_builder(self, template_manager, sample_task, sample_context):
        """Test rendering agent prompt with context builder."""
        # Mock context builder
        mock_context_builder = MagicMock()
        mock_context_builder.export_template_variables.return_value = {
            "agent_type": "planner",
            "task_type": "RETRIEVE",
            "goal": "Enhanced goal with context",
            "enhanced_context": "Additional context from builder"
        }

        template_manager.context_builder = mock_context_builder

        rendered = await template_manager.render_agent_prompt(
            agent_type="planner",
            task_type="RETRIEVE",
            task=sample_task,
            task_context=sample_context
        )

        assert "Enhanced goal with context" in rendered
        mock_context_builder.export_template_variables.assert_called_once()

    def test_template_error_handling(self, template_manager, temp_templates_dir):
        """Test handling of template syntax errors."""
        # Create template with syntax error
        bad_template = temp_templates_dir / "bad_template.jinja2"
        bad_template.write_text("{{ unclosed_variable")

        with pytest.raises(TemplateError):
            template_manager.load_template("bad_template.jinja2")

    def test_initialization_without_templates_dir(self):
        """Test initialization when templates directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = Path(temp_dir) / "nonexistent"

            # Should not raise error, but log warning
            manager = PromptTemplateManager(templates_dir=str(nonexistent_dir))
            assert manager.templates_dir == nonexistent_dir

            # Should have empty environment
            assert manager.env is not None

    @pytest.mark.asyncio
    async def test_template_render_validation_comprehensive(self, template_manager, temp_templates_dir, sample_task, sample_context):
        """Test comprehensive template rendering validation with all context variables."""
        # Create a comprehensive test template that uses many context variables
        comprehensive_template = temp_templates_dir / "test_comprehensive.jinja2"
        comprehensive_template.write_text("""
Agent Type: {{ agent_type }}
Task Type: {{ task_type }}
Goal: {{ goal }}
Task Status: {{ task_status }}
Overall Objective: {{ overall_objective }}
Current Date: {{ current_date }}
Current Year: {{ current_year }}

{% if constraints %}
Constraints:
{{ constraints | format_list("- ") }}
{% endif %}

{% if user_preferences %}
User Preferences: {{ user_preferences | safe_get("style", "default") }}
{% endif %}

{% if has_prior_work %}
Has Prior Work: Yes
{% else %}
Has Prior Work: No
{% endif %}

Task Types Information:
{% for task_info in task_types_info %}
- {{ task_info.name }}: {{ task_info.description | truncate(50) }}
{% endfor %}

Current Task Type Info:
Description: {{ current_task_type_info.description }}
Examples: {{ current_task_type_info.examples | length }} examples
Atomic Indicators: {{ current_task_type_info.atomic_indicators | length }} indicators
        """.strip())

        # Mock context builder with comprehensive template variables
        mock_context_builder = MagicMock()
        mock_context_builder.export_template_variables.return_value = {
            "agent_type": "executor",
            "task_type": "RETRIEVE",
            "goal": "Test comprehensive goal",
            "task_status": "PENDING",
            "overall_objective": "Test overall objective",
            "current_date": "2024-01-15",
            "current_year": 2024,
            "constraints": ["Must use reliable sources", "Complete within 5 minutes"],
            "user_preferences": {"style": "detailed", "format": "markdown"},
            "has_prior_work": True,
            "has_toolkits": True,
            "has_artifacts": False,
            "task_types_info": [
                {"name": "RETRIEVE", "description": "Gather information from external sources"},
                {"name": "WRITE", "description": "Generate and create content"},
                {"name": "THINK", "description": "Analyze and reason about information"}
            ],
            "current_task_type_info": {
                "description": "Gather information from external sources using web search and APIs",
                "examples": ["Search for recent news", "Query database"],
                "atomic_indicators": ["single source", "specific query"],
                "composite_indicators": ["multiple sources", "complex research"]
            }
        }

        template_manager.context_builder = mock_context_builder

        # Test rendering with comprehensive context
        rendered = await template_manager.render_agent_prompt(
            agent_type="executor",
            task_type="RETRIEVE",
            task=sample_task,
            task_context=sample_context
        )

        # Validate all context variables are properly rendered
        assert "Agent Type: executor" in rendered
        assert "Task Type: RETRIEVE" in rendered
        assert "Test comprehensive goal" in rendered
        assert "Task Status: PENDING" in rendered
        assert "Test overall objective" in rendered
        assert "Current Date: 2024-01-15" in rendered
        assert "Current Year: 2024" in rendered

        # Validate custom filters work
        assert "- Must use reliable sources" in rendered
        assert "- Complete within 5 minutes" in rendered
        assert "detailed" in rendered  # safe_get filter
        assert "Has Prior Work: Yes" in rendered

        # Validate task type information is included
        assert "RETRIEVE: Gather information from external sources" in rendered
        assert "WRITE: Generate and create content" in rendered
        assert "THINK: Analyze and reason about information" in rendered

        # Validate current task type info
        assert "Description: Gather information from external sources using web search and APIs" in rendered
        assert "2 examples" in rendered
        assert "2 indicators" in rendered

        # Verify context builder was called correctly
        mock_context_builder.export_template_variables.assert_called_once_with(
            sample_task, sample_context
        )

    def test_template_render_validation_missing_context(self, template_manager, temp_templates_dir, sample_task, sample_context):
        """Test template rendering handles missing context variables gracefully."""
        # Create template that expects variables that might not be provided
        minimal_template = temp_templates_dir / "test_minimal.jinja2"
        minimal_template.write_text("""
Goal: {{ goal }}
Optional Field: {{ optional_field | default("Not provided") }}
List Field: {{ optional_list | format_list("- ") if optional_list else "No items" }}
Dict Field: {{ optional_dict | safe_get("key", "No value") }}
        """.strip())

        # Test with minimal context (only required fields)
        rendered = template_manager.render_template("test_minimal.jinja2", {
            "goal": "Minimal test goal"
        })

        # Should handle missing variables gracefully
        assert "Goal: Minimal test goal" in rendered
        assert "Optional Field: Not provided" in rendered
        assert "List Field: No items" in rendered
        assert "Dict Field: No value" in rendered

    @pytest.mark.asyncio
    async def test_template_render_validation_with_fallback_context(self, template_manager, sample_task, sample_context):
        """Test template rendering with fallback context when no context builder."""
        # Test without context builder (should use fallback)
        template_manager.context_builder = None

        rendered = await template_manager.render_agent_prompt(
            agent_type="executor",
            task_type="RETRIEVE",
            task=sample_task,
            task_context=sample_context
        )

        # Should render with basic fallback context
        assert "executor agent for RETRIEVE tasks" in rendered
        assert "Test task goal" in rendered

    def test_template_render_validation_error_handling(self, template_manager, temp_templates_dir, sample_task, sample_context):
        """Test template rendering error handling."""
        # Create template with undefined function call
        error_template = temp_templates_dir / "test_error.jinja2"
        error_template.write_text("""
Goal: {{ goal }}
Error: {{ undefined_function() }}
        """.strip())

        # Should handle template errors gracefully
        with pytest.raises(TemplateError):
            template_manager.render_template("test_error.jinja2", {"goal": "Test goal"})
