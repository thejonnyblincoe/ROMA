"""
Agent Factory for ROMA v2.0

Creates configured agents using the factory pattern with proper dependency injection.
"""

from typing import Dict, Any, Type, Optional, Union
from pathlib import Path
import logging

from roma.infrastructure.agents.configurable_agent import ConfigurableAgent
from roma.infrastructure.adapters.agno_adapter import AgnoFrameworkAdapter
from roma.infrastructure.toolkits.agno_toolkit_manager import AgnoToolkitManager
from roma.infrastructure.prompts.prompt_template_manager import PromptTemplateManager
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.config.roma_config import ROMAConfig
from roma.domain.value_objects.config.agent_config import AgentConfig
from roma.domain.value_objects.config.model_config import ModelConfig
from roma.domain.value_objects.agent_responses import (
    AtomizerResult,
    PlannerResult,
    ExecutorResult,
    AggregatorResult,
    PlanModifierResult
)

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating configured agents."""

    def __init__(self, config: ROMAConfig):
        """
        Initialize agent factory.

        Args:
            config: ROMA configuration containing agent configs
        """
        self.config = config
        self.adapter = AgnoFrameworkAdapter()
        self.toolkit_manager = AgnoToolkitManager()
        self.prompt_manager = PromptTemplateManager()

        # Response model registry
        self.output_schemas = {
            "AtomizerResult": AtomizerResult,
            "PlannerResult": PlannerResult,
            "ExecutorResult": ExecutorResult,
            "AggregatorResult": AggregatorResult,
            "PlanModifierResult": PlanModifierResult
        }

        self._initialized = False

    async def initialize(self):
        """Initialize factory components."""
        if self._initialized:
            return

        await self.adapter.initialize()
        await self.toolkit_manager.initialize()

        # Link toolkit manager to adapter
        self.adapter.toolkit_manager = self.toolkit_manager

        self._initialized = True
        logger.info("AgentFactory initialized")

    def get_agent_config(self, task_type: TaskType, agent_type: AgentType) -> Dict[str, Any]:
        """
        Get Hydra-resolved agent configuration from profile.

        Args:
            task_type: Task type enum
            agent_type: Agent type enum

        Returns:
            Resolved agent configuration dictionary
        """
        profile = self.config.profile
        agent_mapping = profile.agent_mapping

        # Get agent config directly from resolved mapping
        mapping_attr = f"{agent_type.value.lower()}s"  # atomizers, planners, executors, etc.
        agent_mapping_dict = getattr(agent_mapping, mapping_attr, {})

        if task_type.value not in agent_mapping_dict:
            raise ValueError(f"No {agent_type.value} configured for task type {task_type.value}")

        # At this point, agent_mapping_dict[task_type.value] contains the full resolved config
        # thanks to Hydra's interpolation
        resolved_config = agent_mapping_dict[task_type.value]

        # Convert to dict if it's a config object
        if hasattr(resolved_config, 'model_dump'):
            config_dict = resolved_config.model_dump()
        elif hasattr(resolved_config, '__dict__'):
            config_dict = resolved_config.__dict__.copy()
        elif isinstance(resolved_config, str):
            # Simple agent name - create minimal config for testing
            config_dict = {
                "name": resolved_config,
                "type": agent_type.value,
                "description": f"Test {agent_type.value} agent",
                "model": {
                    "provider": "litellm",
                    "name": "gpt-4o",
                    "temperature": 0.7
                },
                "enabled": True
            }
        else:
            # Already a dict
            config_dict = dict(resolved_config)

        # Ensure task_type and agent_type are included as enums (not strings)
        config_dict.update({
            "task_type": task_type,  # Keep as TaskType enum
            "agent_type": agent_type.value  # Keep as string since AgentConfig.type is str
        })

        return config_dict

    def _resolve_prompt_template(self, agent_config: AgentConfig):
        """
        Resolve prompt template using fallback mechanism.

        Resolution order:
        1. Profile-specified template (agent_config.prompt_template)
        2. Task-specific template ({agent_type}/{task_type}.jinja2)
        3. Generic agent template ({agent_type}/{agent_type}.jinja2)
        4. Raise error if none found

        Args:
            agent_config: Agent configuration

        Returns:
            Jinja2 Template instance

        Raises:
            ValueError: If no template can be resolved
        """
        template_paths_tried = []

        # 1. Try profile-specified template first
        if agent_config.prompt_template:
            try:
                return self.prompt_manager.load_template(agent_config.prompt_template)
            except FileNotFoundError:
                template_paths_tried.append(agent_config.prompt_template)
                logger.warning(f"Profile-specified template not found: {agent_config.prompt_template}")

        # 2. Try task-specific template
        task_specific_path = f"{agent_config.type}/{agent_config.task_type.value.lower()}.jinja2"
        try:
            return self.prompt_manager.load_template(task_specific_path)
        except FileNotFoundError:
            template_paths_tried.append(task_specific_path)

        # 3. Try generic agent type template
        generic_path = f"{agent_config.type}/{agent_config.type}.jinja2"
        try:
            template = self.prompt_manager.load_template(generic_path)
            logger.info(f"Using generic template fallback: {generic_path} for agent {agent_config.name}")
            return template
        except FileNotFoundError:
            template_paths_tried.append(generic_path)

        # 4. All paths failed - raise error
        raise ValueError(
            f"No prompt template found for agent {agent_config.name}. "
            f"Tried paths: {', '.join(template_paths_tried)}"
        )

    def get_output_schema(self, schema_name: str) -> Type:
        """
        Get output schema class by name.

        Args:
            schema_name: Name of the output schema

        Returns:
            Pydantic model class

        Raises:
            ValueError: If schema is unknown
        """
        if schema_name not in self.output_schemas:
            raise ValueError(f"Unknown output schema: {schema_name}. Available: {list(self.output_schemas.keys())}")
        return self.output_schemas[schema_name]

    async def create_agent(
        self,
        agent_config: AgentConfig,
        output_schema_name: Optional[str] = None
    ) -> ConfigurableAgent:
        """
        Create agent from configuration.

        Args:
            agent_config: AgentConfig object (already validated)
            output_schema_name: Optional override for output schema

        Returns:
            Configured agent instance

        Raises:
            ValueError: If configuration is invalid
        """
        if not self._initialized:
            await self.initialize()

        # Get output schema - prioritize config.output_schema over parameter
        schema_name = agent_config.output_schema or output_schema_name
        if not schema_name:
            # Default schema based on agent type - proper camelization
            schema_name = "".join(p.title() for p in agent_config.type.split("_")) + "Result"

        try:
            output_schema = self.get_output_schema(schema_name)
        except ValueError as e:
            logger.error(f"Invalid output schema for agent {agent_config.name}: {e}")
            raise

        # Create agent
        agent = ConfigurableAgent(
            config=agent_config,
            framework_adapter=self.adapter,
            output_schema=output_schema,
            prompt_template_manager=self.prompt_manager
        )

        # Inject toolkit manager
        agent.toolkit_manager = self.toolkit_manager

        logger.info(f"Created {agent_config.type} agent: {agent_config.name}")
        return agent


    def clear_cache(self) -> None:
        """Clear prompt template cache."""
        self.prompt_manager.clear_cache()
        logger.info("Agent factory cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        cache_info = self.prompt_manager.get_cache_info()
        return {
            "prompt_templates_cached": cache_info["cached_templates"],
            "output_schemas_available": len(self.output_schemas)
        }