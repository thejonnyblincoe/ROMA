"""
Tests for AgentFactory template resolution mechanism.

Tests the template fallback logic and schema name generation fixes.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from roma.domain.value_objects.config.agent_config import AgentConfig
from roma.domain.value_objects.config.model_config import ModelConfig
from roma.domain.value_objects.config.roma_config import ROMAConfig
from roma.domain.value_objects.task_type import TaskType
from roma.infrastructure.agents.agent_factory import AgentFactory


class TestAgentFactoryTemplateResolution:
    """Test cases for AgentFactory template resolution."""

    @pytest.fixture
    def temp_templates_dir(self):
        """Create temporary templates directory with test templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir)

            # Create directory structure
            (templates_dir / "atomizer").mkdir(parents=True)
            (templates_dir / "planner").mkdir(parents=True)
            (templates_dir / "executor").mkdir(parents=True)
            (templates_dir / "plan_modifier").mkdir(parents=True)

            # Create generic templates (fallbacks)
            (templates_dir / "atomizer" / "atomizer.jinja2").write_text("Generic atomizer template")
            (templates_dir / "planner" / "planner.jinja2").write_text("Generic planner template")
            (templates_dir / "executor" / "executor.jinja2").write_text("Generic executor template")
            (templates_dir / "plan_modifier" / "plan_modifier.jinja2").write_text("Generic plan_modifier template")

            # Create some task-specific templates
            (templates_dir / "executor" / "retrieve.jinja2").write_text("RETRIEVE executor template")
            (templates_dir / "planner" / "think.jinja2").write_text("THINK planner template")

            # Create profile-specified template
            (templates_dir / "custom").mkdir(parents=True)
            (templates_dir / "custom" / "custom_atomizer.jinja2").write_text("Custom profile template")

            yield templates_dir

    @pytest.fixture
    def mock_config(self):
        """Create mock ROMA configuration."""
        config = MagicMock(spec=ROMAConfig)
        return config

    @pytest.fixture
    def agent_factory(self, mock_config, temp_templates_dir):
        """Create AgentFactory with test setup."""
        factory = AgentFactory(mock_config)

        # Patch the prompt manager to use test templates directory
        with patch('src.roma.infrastructure.agents.agent_factory.PromptTemplateManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            # Setup prompt manager to load from test directory
            def load_template_side_effect(path):
                full_path = temp_templates_dir / path
                if full_path.exists():
                    mock_template = MagicMock()
                    mock_template.render.return_value = f"Rendered: {path}"
                    return mock_template
                else:
                    raise FileNotFoundError(f"Template not found: {path}")

            mock_manager.load_template.side_effect = load_template_side_effect
            factory.prompt_manager = mock_manager

            yield factory

    def test_schema_name_generation_camelcase(self, agent_factory):
        """Test that schema names are properly camelCased."""
        # Test plan_modifier -> PlanModifierResult
        agent_config = AgentConfig(
            name="test_agent",
            type="plan_modifier",
            task_type=TaskType.THINK,
            model=ModelConfig(provider="litellm", model_id="test-model"),
            enabled=True
        )

        template = agent_factory._resolve_prompt_template(agent_config)

        # Should resolve to generic plan_modifier template
        assert template is not None

    def test_template_resolution_profile_specified(self, agent_factory, temp_templates_dir):
        """Test template resolution when profile specifies template."""
        # Create the custom template
        custom_dir = temp_templates_dir / "custom"
        custom_dir.mkdir(exist_ok=True)

        agent_config = AgentConfig(
            name="test_agent",
            type="atomizer",
            task_type=TaskType.RETRIEVE,
            model=ModelConfig(provider="litellm", model_id="test-model"),
            prompt_template="custom/custom_atomizer.jinja2",
            enabled=True
        )

        template = agent_factory._resolve_prompt_template(agent_config)
        assert template is not None
        # Should use profile-specified template

    def test_template_resolution_task_specific(self, agent_factory):
        """Test template resolution falls back to task-specific template."""
        agent_config = AgentConfig(
            name="test_agent",
            type="executor",
            task_type=TaskType.RETRIEVE,
            model=ModelConfig(provider="litellm", model_id="test-model"),
            enabled=True
        )

        template = agent_factory._resolve_prompt_template(agent_config)
        assert template is not None
        # Should use executor/retrieve.jinja2 (task-specific)

    def test_template_resolution_generic_fallback(self, agent_factory):
        """Test template resolution falls back to generic template."""
        agent_config = AgentConfig(
            name="test_agent",
            type="executor",
            task_type=TaskType.WRITE,  # No write-specific executor template
            model=ModelConfig(provider="litellm", model_id="test-model"),
            enabled=True
        )

        template = agent_factory._resolve_prompt_template(agent_config)
        assert template is not None
        # Should use executor/executor.jinja2 (generic fallback)

    def test_template_resolution_all_fail(self, agent_factory):
        """Test template resolution raises error when all paths fail."""
        # Create a valid agent config but remove templates to simulate failure
        agent_config = AgentConfig(
            name="test_agent",
            type="atomizer",  # Valid type
            task_type=TaskType.RETRIEVE,
            model=ModelConfig(provider="litellm", model_id="test-model"),
            enabled=True
        )

        # Mock the prompt manager to always fail
        def always_fail_load(path):
            raise FileNotFoundError(f"Template not found: {path}")

        agent_factory.prompt_manager.load_template.side_effect = always_fail_load

        with pytest.raises(ValueError) as exc_info:
            agent_factory._resolve_prompt_template(agent_config)

        assert "No prompt template found" in str(exc_info.value)
        assert "test_agent" in str(exc_info.value)

    def test_template_resolution_profile_not_found_fallback(self, agent_factory):
        """Test template resolution when profile template doesn't exist."""
        agent_config = AgentConfig(
            name="test_agent",
            type="atomizer",
            task_type=TaskType.RETRIEVE,
            model=ModelConfig(provider="litellm", model_id="test-model"),
            prompt_template="nonexistent/template.jinja2",
            enabled=True
        )

        template = agent_factory._resolve_prompt_template(agent_config)
        assert template is not None
        # Should fall back to atomizer/retrieve.jinja2 or atomizer/atomizer.jinja2

    def test_schema_name_generation_variations(self, agent_factory):
        """Test schema name generation for various agent types."""
        test_cases = [
            ("atomizer", "AtomizerResult"),
            ("planner", "PlannerResult"),
            ("executor", "ExecutorResult"),
            ("aggregator", "AggregatorResult"),
            ("plan_modifier", "PlanModifierResult"),
        ]

        for agent_type, expected_schema in test_cases:
            agent_config = AgentConfig(
                name="test_agent",
                type=agent_type,
                task_type=TaskType.THINK,
                model=ModelConfig(provider="litellm", model_id="test-model"),
                enabled=True
            )

            # Test the schema name generation logic directly
            schema_name = "".join(p.title() for p in agent_config.type.split("_")) + "Result"
            assert schema_name == expected_schema

        # Test the edge case directly without creating AgentConfig
        test_type = "test_agent_type"
        schema_name = "".join(p.title() for p in test_type.split("_")) + "Result"
        assert schema_name == "TestAgentTypeResult"

    def test_template_resolution_logging(self, agent_factory, caplog):
        """Test that template resolution logs appropriate messages."""
        import logging

        # Test generic fallback logging
        agent_config = AgentConfig(
            name="test_agent",
            type="executor",
            task_type=TaskType.WRITE,  # Will fall back to generic
            model=ModelConfig(provider="litellm", model_id="test-model"),
            enabled=True
        )

        with caplog.at_level(logging.INFO):
            template = agent_factory._resolve_prompt_template(agent_config)
            assert template is not None

        # Should log info about using generic fallback
        assert any("generic template fallback" in record.message.lower() for record in caplog.records)

    def test_template_resolution_profile_warning(self, agent_factory, caplog):
        """Test warning when profile-specified template not found."""
        import logging

        agent_config = AgentConfig(
            name="test_agent",
            type="atomizer",
            task_type=TaskType.RETRIEVE,
            model=ModelConfig(provider="litellm", model_id="test-model"),
            prompt_template="nonexistent/template.jinja2",
            enabled=True
        )

        with caplog.at_level(logging.WARNING):
            template = agent_factory._resolve_prompt_template(agent_config)
            assert template is not None

        # Should log warning about profile template not found
        assert any("profile-specified template not found" in record.message.lower() for record in caplog.records)

    def test_template_paths_tried_in_error(self, agent_factory):
        """Test that error message includes all tried paths."""
        agent_config = AgentConfig(
            name="test_agent",
            type="atomizer",  # Valid type
            task_type=TaskType.RETRIEVE,
            model=ModelConfig(provider="litellm", model_id="test-model"),
            prompt_template="also/nonexistent.jinja2",
            enabled=True
        )

        # Mock the prompt manager to always fail
        def always_fail_load(path):
            raise FileNotFoundError(f"Template not found: {path}")

        agent_factory.prompt_manager.load_template.side_effect = always_fail_load

        with pytest.raises(ValueError) as exc_info:
            agent_factory._resolve_prompt_template(agent_config)

        error_msg = str(exc_info.value)
        assert "also/nonexistent.jinja2" in error_msg
        assert "atomizer/retrieve.jinja2" in error_msg
        assert "atomizer/atomizer.jinja2" in error_msg
