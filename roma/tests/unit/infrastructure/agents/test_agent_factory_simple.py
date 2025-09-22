"""
Simplified unit tests for AgentFactory edge cases to improve coverage.

Focus on testing branches and edge cases that aren't covered by integration tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from roma.infrastructure.agents.agent_factory import AgentFactory
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.agent_type import AgentType


class TestAgentFactorySimpleEdgeCases:
    """Simple test cases for AgentFactory edge cases."""

    @pytest.fixture
    def agent_factory(self):
        """Create AgentFactory with mocked config."""
        mock_config = Mock()
        mock_profile = Mock()
        mock_agent_mapping = Mock()

        # Set up the mock chain
        mock_config.profile = mock_profile
        mock_profile.agent_mapping = mock_agent_mapping

        # Configure atomizers mapping
        mock_agent_mapping.atomizers = {
            "THINK": "simple_atomizer",
        }
        mock_agent_mapping.planners = {}
        mock_agent_mapping.executors = {}
        mock_agent_mapping.aggregators = {}

        return AgentFactory(mock_config)

    def test_get_output_schema_unknown_schema(self, agent_factory):
        """Test get_output_schema with unknown schema name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown output schema: UnknownSchema"):
            agent_factory.get_output_schema("UnknownSchema")

    def test_get_output_schema_valid_schemas(self, agent_factory):
        """Test get_output_schema returns correct schemas for valid names."""
        from roma.domain.value_objects.agent_responses import AtomizerResult, PlannerResult

        atomizer_schema = agent_factory.get_output_schema("AtomizerResult")
        assert atomizer_schema == AtomizerResult

        planner_schema = agent_factory.get_output_schema("PlannerResult")
        assert planner_schema == PlannerResult

    def test_get_agent_config_missing_task_type_mapping(self, agent_factory):
        """Test get_agent_config raises error when task type mapping is missing."""
        # The fixture has empty planners dict, so this should raise an error
        with pytest.raises(ValueError, match="No planner configured for task type THINK"):
            agent_factory.get_agent_config(TaskType.THINK, AgentType.PLANNER)

    def test_get_agent_config_missing_agent_type_attribute(self, agent_factory):
        """Test get_agent_config when agent type attribute doesn't exist."""
        # Test with an agent type that doesn't have a mapping configured
        with pytest.raises(ValueError, match="No executor configured for task type THINK"):
            agent_factory.get_agent_config(TaskType.THINK, AgentType.EXECUTOR)

    def test_get_agent_config_string_config(self, agent_factory):
        """Test get_agent_config with string configuration creates minimal config."""
        # Mock the config to return a string
        agent_factory.config.profile.agent_mapping.atomizers = {"THINK": "test_agent"}

        config_dict = agent_factory.get_agent_config(TaskType.THINK, AgentType.ATOMIZER)

        assert config_dict["name"] == "test_agent"
        assert config_dict["type"] == "atomizer"
        assert config_dict["description"] == "Test atomizer agent"
        assert config_dict["model"]["provider"] == "litellm"
        assert config_dict["enabled"] is True
        assert config_dict["task_type"] == TaskType.THINK
        assert config_dict["agent_type"] == "atomizer"

    def test_get_agent_config_dict_config(self, agent_factory):
        """Test get_agent_config with dictionary configuration."""
        test_config = {"name": "dict_agent", "enabled": True, "custom_field": "value"}
        agent_factory.config.profile.agent_mapping.atomizers = {"THINK": test_config}

        config_dict = agent_factory.get_agent_config(TaskType.THINK, AgentType.ATOMIZER)

        assert config_dict["name"] == "dict_agent"
        assert config_dict["enabled"] is True
        assert config_dict["custom_field"] == "value"
        assert config_dict["task_type"] == TaskType.THINK
        assert config_dict["agent_type"] == "atomizer"

    def test_get_agent_config_object_with_model_dump(self, agent_factory):
        """Test get_agent_config with object that has model_dump method."""
        mock_config = Mock()
        mock_config.model_dump.return_value = {"name": "pydantic_agent", "version": "1.0"}

        agent_factory.config.profile.agent_mapping.atomizers = {"THINK": mock_config}

        config_dict = agent_factory.get_agent_config(TaskType.THINK, AgentType.ATOMIZER)

        assert config_dict["name"] == "pydantic_agent"
        assert config_dict["version"] == "1.0"
        assert config_dict["task_type"] == TaskType.THINK
        assert config_dict["agent_type"] == "atomizer"

    def test_get_agent_config_object_with_dict_attribute(self, agent_factory):
        """Test get_agent_config with object that has __dict__ attribute."""
        mock_config = Mock()
        mock_config.__dict__ = {"name": "dataclass_agent", "field": "test"}
        # Remove model_dump to ensure it falls through to __dict__ path
        if hasattr(mock_config, 'model_dump'):
            delattr(mock_config, 'model_dump')

        agent_factory.config.profile.agent_mapping.atomizers = {"THINK": mock_config}

        config_dict = agent_factory.get_agent_config(TaskType.THINK, AgentType.ATOMIZER)

        assert config_dict["name"] == "dataclass_agent"
        assert config_dict["field"] == "test"
        assert config_dict["task_type"] == TaskType.THINK
        assert config_dict["agent_type"] == "atomizer"

    def test_clear_cache(self, agent_factory):
        """Test clear_cache method calls prompt manager."""
        with patch.object(agent_factory.prompt_manager, 'clear_cache') as mock_clear:
            agent_factory.clear_cache()
            mock_clear.assert_called_once()

    def test_get_cache_stats(self, agent_factory):
        """Test get_cache_stats returns formatted statistics."""
        mock_cache_info = {"cached_templates": 5, "cache_hits": 10, "other_field": "ignored"}

        with patch.object(agent_factory.prompt_manager, 'get_cache_info', return_value=mock_cache_info):
            stats = agent_factory.get_cache_stats()

            assert stats["prompt_templates_cached"] == 5
            assert stats["output_schemas_available"] == len(agent_factory.output_schemas)
            # Should not include other fields from cache_info

    @pytest.mark.asyncio
    async def test_initialize_double_call_protection(self, agent_factory):
        """Test that initialize can be called multiple times safely."""
        # Mock the dependencies
        agent_factory.adapter = Mock()
        agent_factory.adapter.initialize = AsyncMock()
        agent_factory.toolkit_manager = Mock()
        agent_factory.toolkit_manager.initialize = AsyncMock()

        # First call
        await agent_factory.initialize()
        assert agent_factory._initialized is True

        # Reset mocks to track second call
        agent_factory.adapter.initialize.reset_mock()
        agent_factory.toolkit_manager.initialize.reset_mock()

        # Second call should not initialize again
        await agent_factory.initialize()

        agent_factory.adapter.initialize.assert_not_called()
        agent_factory.toolkit_manager.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_components_setup(self, agent_factory):
        """Test that initialize properly sets up all components."""
        # Mock the dependencies
        agent_factory.adapter = Mock()
        agent_factory.adapter.initialize = AsyncMock()
        agent_factory.toolkit_manager = Mock()
        agent_factory.toolkit_manager.initialize = AsyncMock()

        await agent_factory.initialize()

        # Should initialize both components
        agent_factory.adapter.initialize.assert_called_once()
        agent_factory.toolkit_manager.initialize.assert_called_once()

        # Should link toolkit manager to adapter
        assert agent_factory.adapter.toolkit_manager == agent_factory.toolkit_manager

        # Should set initialized flag
        assert agent_factory._initialized is True

    @pytest.mark.asyncio
    async def test_create_agent_auto_initialize(self, agent_factory):
        """Test create_agent calls initialize when not initialized."""
        from roma.domain.value_objects.config.agent_config import AgentConfig
        from roma.domain.value_objects.config.model_config import ModelConfig
        from jinja2 import Template

        agent_config = AgentConfig(
            name="test_agent",
            type="atomizer",
            task_type=TaskType.THINK,
            model=ModelConfig(provider="litellm", name="gpt-4o"),
            enabled=True
        )

        # Mock initialize and other dependencies
        agent_factory.initialize = AsyncMock()
        agent_factory.prompt_manager = Mock()
        agent_factory.prompt_manager.load_template.return_value = Template("test")

        with patch('src.roma.infrastructure.agents.agent_factory.ConfigurableAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            await agent_factory.create_agent(agent_config)

            # Should call initialize since _initialized defaults to False
            agent_factory.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_agent_template_fallback(self, agent_factory):
        """Test create_agent uses fallback template when file not found."""
        from roma.domain.value_objects.config.agent_config import AgentConfig
        from roma.domain.value_objects.config.model_config import ModelConfig

        agent_factory._initialized = True  # Skip auto-initialize

        agent_config = AgentConfig(
            name="test_agent",
            type="atomizer",
            task_type=TaskType.THINK,
            model=ModelConfig(provider="litellm", name="gpt-4o"),
            enabled=True
        )

        # Mock prompt manager to raise FileNotFoundError
        agent_factory.prompt_manager = Mock()
        agent_factory.prompt_manager.load_template.side_effect = FileNotFoundError("Template not found")

        with patch('src.roma.infrastructure.agents.agent_factory.ConfigurableAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            agent = await agent_factory.create_agent(agent_config)

            # Should still create agent with fallback template
            assert agent == mock_agent

            # Check that fallback template was used (Template object, check its source)
            call_args = mock_agent_class.call_args
            prompt_template = call_args[1]['prompt_template']
            # For Jinja2 Template, we can render it to check content
            rendered = prompt_template.render(agent_type="atomizer", task_type="THINK", task=Mock(goal="test"))
            assert "You are a atomizer agent" in rendered

    def test_get_output_schema_error_message_includes_available_schemas(self, agent_factory):
        """Test get_output_schema error message includes available schemas."""
        try:
            agent_factory.get_output_schema("UnknownSchema")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            assert "Unknown output schema: UnknownSchema" in error_msg
            assert "Available:" in error_msg
            assert "AtomizerResult" in error_msg
            assert "PlannerResult" in error_msg

    def test_constructor_sets_up_defaults(self):
        """Test AgentFactory constructor initializes all components."""
        mock_config = Mock()
        factory = AgentFactory(mock_config)

        assert factory.config == mock_config
        assert factory.adapter is not None
        assert factory.toolkit_manager is not None
        assert factory.prompt_manager is not None
        assert len(factory.output_schemas) == 5
        assert factory._initialized is False

        # Check all expected schemas are registered
        expected_schemas = [
            "AtomizerResult", "PlannerResult", "ExecutorResult",
            "AggregatorResult", "PlanModifierResult"
        ]
        for schema_name in expected_schemas:
            assert schema_name in factory.output_schemas


class TestAgentFactorySchemaNameGeneration:
    """Test schema name generation edge cases."""

    @pytest.fixture
    def agent_factory(self):
        """Simple agent factory for schema tests."""
        return AgentFactory(Mock())

    @pytest.mark.asyncio
    async def test_schema_name_generation_logic(self, agent_factory):
        """Test that schema name generation follows expected pattern."""
        # Test the actual schema name generation logic directly
        test_cases = [
            ("atomizer", "AtomizerResult"),
            ("planner", "PlannerResult"),
            ("plan_modifier", "PlanModifierResult"),  # This exists in registry
        ]

        for agent_type, expected_schema in test_cases:
            # Generate schema name using same logic as create_agent
            agent_type_clean = agent_type.replace("_", "").title()
            schema_name = f"{agent_type_clean}Result"

            if expected_schema in agent_factory.output_schemas:
                schema_class = agent_factory.get_output_schema(expected_schema)
                assert schema_class is not None

    @pytest.mark.asyncio
    async def test_custom_output_schema_override(self, agent_factory):
        """Test custom output schema name can override default."""
        from roma.domain.value_objects.config.agent_config import AgentConfig
        from roma.domain.value_objects.config.model_config import ModelConfig
        from roma.domain.value_objects.agent_responses import AtomizerResult
        from jinja2 import Template

        agent_factory._initialized = True

        agent_config = AgentConfig(
            name="test_agent",
            type="planner",  # Would normally use PlannerResult
            task_type=TaskType.THINK,
            model=ModelConfig(provider="litellm", name="gpt-4o"),
            enabled=True
        )

        agent_factory.prompt_manager = Mock()
        agent_factory.prompt_manager.load_template.return_value = Template("test")

        with patch('src.roma.infrastructure.agents.agent_factory.ConfigurableAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # Override with AtomizerResult
            await agent_factory.create_agent(agent_config, output_schema_name="AtomizerResult")

            # Should use AtomizerResult instead of PlannerResult
            call_args = mock_agent_class.call_args
            output_schema = call_args[1]['output_schema']
            assert output_schema == AtomizerResult