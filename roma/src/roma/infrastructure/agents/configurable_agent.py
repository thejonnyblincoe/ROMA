"""
Configurable Agent Implementation for ROMA v2.0

Generic agent implementation that can be configured for any agent type
using Jinja2 templates and Agno's native output_schema support.
"""

from typing import TypeVar, Generic, Type, Dict, Any, Optional, Union
import logging
from jinja2 import Template, TemplateError

from src.roma.domain.interfaces.agent import Agent
from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.config.agent_config import AgentConfig
from src.roma.infrastructure.adapters.agno_adapter import AgnoFrameworkAdapter
from src.roma.infrastructure.toolkits.agno_toolkit_manager import AgnoToolkitManager

T = TypeVar('T')
logger = logging.getLogger(__name__)


class ConfigurableAgent(Agent[T], Generic[T]):
    """
    Generic configurable agent using Agno framework.

    This agent can be configured to serve as any agent type (atomizer, planner, etc.)
    by providing appropriate prompt templates and output schemas.
    """

    def __init__(
        self,
        config: AgentConfig,
        framework_adapter: AgnoFrameworkAdapter,
        output_schema: Type[T],
        prompt_template: Union[str, Template]
    ):
        """
        Initialize configurable agent.

        Args:
            config: AgentConfig object (already validated)
            framework_adapter: Framework adapter for Agno integration
            output_schema: Pydantic output schema class (Agno's output_schema)
            prompt_template: Jinja2 template string or Template object
        """
        self.config = config
        self.adapter = framework_adapter
        self.output_schema = output_schema

        try:
            if isinstance(prompt_template, Template):
                self.prompt_template = prompt_template
            else:
                self.prompt_template = Template(prompt_template)
        except TemplateError as e:
            logger.error(f"Invalid prompt template: {e}")
            raise ValueError(f"Invalid prompt template: {e}")

        # Injected dependencies
        self.toolkit_manager: Optional[AgnoToolkitManager] = None

        self.agent_name = config.name
        logger.info(f"Initialized {self.agent_name} with output schema {output_schema.__name__}")

    async def run(self, task: TaskNode, context: Dict[str, Any]) -> T:
        """
        Execute agent with task and context.

        Args:
            task: TaskNode to process
            context: Execution context with relevant data

        Returns:
            Structured result of type T
        """
        try:
            # Render prompt with Jinja2 template
            prompt = self._render_prompt(task, context)

            # Get tools from configuration
            tools = self._get_tools()

            # Create model configuration
            model_config = self._create_model_config()

            # Execute via Agno adapter with native output_schema
            result = await self.adapter.run(
                prompt=prompt,
                output_schema=self.output_schema,
                tools=tools,
                agent_name=self.agent_name,
                model_config=model_config
            )

            logger.info(f"{self.agent_name} completed successfully")
            return result

        except Exception as e:
            logger.error(f"{self.agent_name} execution failed: {e}")
            raise

    def _render_prompt(self, task: TaskNode, context: Dict[str, Any]) -> str:
        """
        Render prompt template with task and context data.

        Args:
            task: TaskNode to include in template
            context: Context data for template

        Returns:
            Rendered prompt string
        """
        try:
            prompt = self.prompt_template.render(
                task=task,
                context=context,
                config=self.config,
                # Add helpful template functions
                task_type=task.task_type.value if task.task_type else "UNKNOWN",
                status=task.status.value if task.status else "UNKNOWN",
                goal=task.goal
            )

            logger.debug(f"Rendered prompt for {self.agent_name}: {prompt[:200]}...")
            return prompt

        except TemplateError as e:
            logger.error(f"Template rendering failed for {self.agent_name}: {e}")
            raise ValueError(f"Template rendering failed: {e}")

    def _get_tools(self) -> list:
        """Get tools from configuration."""
        # Use the new config.tools field (List[ToolConfig])
        if not self.config.tools or not self.toolkit_manager:
            return []

        # Convert ToolConfig objects to toolkit instances using the toolkit manager
        toolkit_instances = []
        for tool_config in self.config.tools:
            try:
                # Get toolkit instance from manager using the tool name
                toolkit_instance = self.toolkit_manager.get_toolkit_for_agent(tool_config.name)

                if toolkit_instance:
                    toolkit_instances.append(toolkit_instance)
                    logger.debug(f"Added toolkit {tool_config.name} ({tool_config.type}) to agent {self.agent_name}")

            except Exception as e:
                logger.error(f"Failed to get toolkit {tool_config.name}: {e}")

        return toolkit_instances

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

    def get_task_type(self) -> Optional[str]:
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