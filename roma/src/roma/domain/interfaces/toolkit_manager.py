"""
Toolkit Manager Interface - ROMA v2.0 Domain Interface.

Abstract interface for toolkit management operations that infrastructure must implement.
"""

from abc import ABC, abstractmethod
from typing import Any


class IToolkitManager(ABC):
    """
    Domain interface for toolkit manager operations.

    Abstract interface that defines all toolkit management operations required by the domain.
    Infrastructure layer must implement this interface to provide tool management capabilities.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the toolkit manager.

        This method should set up any required resources and prepare the toolkit
        for tool registration and execution.
        """

    @abstractmethod
    async def register_tool_config(self, tool_config: Any) -> None:
        """
        Register a tool configuration.

        Args:
            tool_config: Tool configuration object to register

        Raises:
            ToolRegistrationError: If tool registration fails
        """

    @abstractmethod
    async def get_available_tools(self) -> list[str]:
        """
        Get list of available tool names.

        Returns:
            List of registered tool names
        """

    @abstractmethod
    async def get_tool_info(self, tool_name: str) -> dict[str, Any]:
        """
        Get information about a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary with tool information

        Raises:
            ToolNotFoundError: If tool is not found
        """

    @abstractmethod
    async def execute_tool(
        self, tool_name: str, parameters: dict[str, Any], context: dict[str, Any] | None = None
    ) -> Any:
        """
        Execute a tool with given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            context: Optional execution context

        Returns:
            Tool execution result

        Raises:
            ToolNotFoundError: If tool is not found
            ToolExecutionError: If tool execution fails
        """

    @abstractmethod
    async def get_tools_for_task_type(self, task_type: str) -> list[str]:
        """
        Get tools suitable for a specific task type.

        Args:
            task_type: Type of task

        Returns:
            List of tool names suitable for the task type
        """

    @abstractmethod
    async def validate_tool_parameters(self, tool_name: str, parameters: dict[str, Any]) -> bool:
        """
        Validate parameters for a specific tool.

        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate

        Returns:
            True if parameters are valid, False otherwise

        Raises:
            ToolNotFoundError: If tool is not found
        """

    @abstractmethod
    def get_toolkit_stats(self) -> dict[str, Any]:
        """
        Get statistics about the toolkit manager.

        Returns:
            Dictionary with toolkit statistics (registered tools, executions, etc.)
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the toolkit manager and clean up resources.

        This method should properly clean up any resources used by the toolkit
        and registered tools.
        """
