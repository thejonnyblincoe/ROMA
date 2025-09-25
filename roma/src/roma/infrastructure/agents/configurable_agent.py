"""
Configurable Agent Implementation for ROMA v2.0

Generic agent implementation that can be configured for any agent type
using Jinja2 templates and Agno's native output_schema support.
"""

import logging
from typing import TYPE_CHECKING, Optional, TypeVar

from roma.domain.entities.task_node import TaskNode
from roma.domain.interfaces.configurable_agent import IConfigurableAgent
from roma.domain.value_objects.config.agent_config import AgentConfig
from roma.infrastructure.adapters.agno_adapter import AgnoFrameworkAdapter
from roma.infrastructure.toolkits.agno_toolkit_manager import AgnoToolkitManager

if TYPE_CHECKING:
    from roma.domain.context.task_context import TaskContext
    from roma.infrastructure.prompts.prompt_template_manager import PromptTemplateManager

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ConfigurableAgent(IConfigurableAgent[T]):
    """
    Configurable agent that delegates to specialized services.

    - PromptTemplateManager renders prompts with full context
    - Agno executes with output_schema for structured output
    - Clean separation of concerns with proper delegation
    """

    def __init__(
        self,
        config: AgentConfig,
        framework_adapter: AgnoFrameworkAdapter,
        output_schema: type[T],
        prompt_template_manager: "PromptTemplateManager",
    ):
        """
        Initialize configurable agent with dependencies.

        Args:
            config: AgentConfig object (already validated)
            framework_adapter: Framework adapter for Agno integration
            output_schema: Pydantic output schema class (Agno's output_schema)
            prompt_template_manager: Manager for template operations
        """
        self.config = config
        self.adapter = framework_adapter
        self.output_schema = output_schema
        self.prompt_manager = prompt_template_manager

        # Injected dependencies
        self.toolkit_manager: AgnoToolkitManager | None = None

        self.agent_name = config.name
        logger.info(f"Initialized {self.agent_name} with output schema {output_schema.__name__}")

    @property
    def name(self) -> str:
        """Alias for agent name to ensure compatibility with callers expecting `.name`.

        Some parts of the system (and external frameworks like Agno) expose the
        agent identifier via a `name` attribute. ConfigurableAgent historically
        stored it as `agent_name`. This property provides a stable `.name` alias
        so the rest of the codebase can consistently access `agent.name`.
        """
        return self.agent_name

    async def run(self, task: TaskNode, context: Optional["TaskContext"] = None) -> T:
        """
        Execute agent with proper separation of concerns.

        Args:
            task: TaskNode to process
            context: Optional TaskContext from ContextBuilderService

        Returns:
            Structured result of type T

        Raises:
            ValueError: If context is None (required for agent execution)
        """
        if context is None:
            raise ValueError("TaskContext is required for agent execution")

        try:
            # Delegate prompt rendering to PromptTemplateManager
            # Use the agent config's prompt_template directly
            if self.config.prompt_template:
                template = self.prompt_manager.load_template(self.config.prompt_template)

                # Get template variables
                if self.prompt_manager.context_builder:
                    template_vars = (
                        await self.prompt_manager.context_builder.export_template_variables(
                            task, context
                        )
                    )
                else:
                    template_vars = self.prompt_manager._get_basic_template_variables(
                        self.config.type,
                        self.config.task_type.value if self.config.task_type else "GENERAL",
                        task,
                        context,
                    )

                prompt = template.render(**template_vars).strip()
            else:
                # Fallback to old method
                prompt = await self.prompt_manager.render_agent_prompt(
                    agent_type=self.config.type,
                    task_type=self.config.task_type.value if self.config.task_type else "GENERAL",
                    task=task,
                    task_context=context,
                )

            # Get tools from configuration
            tools = self._get_tools()

            # Create model configuration
            model_config = self._create_model_config()

            # Execute via Agno with output_schema (Agno handles structured output)
            result = await self.adapter.run(
                prompt=prompt,
                output_schema=self.output_schema,
                tools=tools,
                agent_name=self.agent_name,
                model_config=model_config,
            )

            logger.info(f"{self.agent_name} completed successfully")
            return result

        except Exception as e:
            logger.error(f"{self.agent_name} execution failed: {e}")
            raise

    def _get_tools(self) -> list:
        """Get tool names from configuration."""
        # Use the new config.tools field (List[ToolConfig])
        if not self.config.tools:
            return []

        # Return tool names (strings) instead of toolkit instances
        # The adapter will create toolkits from these names
        tool_names = []
        for tool_config in self.config.tools:
            tool_names.append(tool_config.name)
            logger.debug(
                f"Added tool name {tool_config.name} ({tool_config.type}) to agent {self.agent_name}"
            )

        return tool_names

    def _create_model_config(self):
        """Get ModelConfig from agent configuration."""
        # AgentConfig is a Pydantic model and always has a 'model' field
        if self.config.model is None:
            raise ValueError(f"No model configuration found for agent {self.config.name}")
        return self.config.model

    def get_config(self) -> AgentConfig:
        """Get agent configuration."""
        return self.config

    def get_agent_type(self) -> str:
        """Get agent type from configuration."""
        return self.config.type

    def get_task_type(self) -> str | None:
        """Get task type this agent is configured for."""
        return self.config.task_type.value if self.config.task_type else None

    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.agent_name}', "
            f"type='{self.get_agent_type()}', "
            f"output_schema={self.output_schema.__name__})"
        )
