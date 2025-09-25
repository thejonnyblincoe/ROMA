"""
Framework Adapter interface for pluggable agent frameworks.

Defines the contract that framework-specific adapters must implement
to integrate with the ROMA agent runtime.
"""

from abc import ABC, abstractmethod
from typing import Any


class FrameworkAdapter(ABC):
    """
    Abstract interface for framework adapters.

    Enables integration of different agent frameworks (Agno, LangGraph, CrewAI, etc.)
    through a consistent interface.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the framework adapter."""

    @abstractmethod
    async def run(
        self,
        prompt: str,
        output_schema: type,
        tools: list[Any] | None = None,
        agent_name: str = "roma_agent",
        model_config: Any = None,
    ) -> Any:
        """
        Execute a task using the framework.

        Args:
            prompt: The prompt to execute
            output_schema: Expected output schema class
            tools: List of tools/toolkits to use
            agent_name: Name for the agent
            model_config: Model configuration

        Returns:
            Structured result matching output_schema
        """

    @abstractmethod
    def get_framework_name(self) -> str:
        """
        Get the name of this framework.

        Returns:
            Framework name (e.g., 'agno', 'langgraph', 'crewai')
        """
