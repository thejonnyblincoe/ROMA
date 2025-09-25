"""
Agent Runtime Service - Application Layer.

Orchestrates agent runtime lifecycle and coordinates with other services.
This service provides the application-level logic for managing the agent runtime,
connecting it to the event system, and coordinating with configuration and toolkit systems.

Task 1.3.1: Framework-agnostic agent runtime abstraction
"""

import logging
from datetime import datetime
from typing import Any

from roma.domain.context.task_context import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.interfaces.agent_factory import IAgentFactory
from roma.domain.interfaces.agent_runtime_service import IAgentRuntimeService
from roma.domain.interfaces.configurable_agent import IConfigurableAgent
from roma.domain.interfaces.event_publisher import IEventPublisher
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.config.agent_config import AgentConfig
from roma.domain.value_objects.result_envelope import (
    AnyResultEnvelope,
    ExecutionMetrics,
    ResultEnvelope,
)
from roma.domain.value_objects.task_type import TaskType

logger = logging.getLogger(__name__)


class AgentRuntimeService(IAgentRuntimeService):
    """
    Application service for managing agent runtime lifecycle.

    Coordinates between the agent runtime, event system, configuration,
    and toolkit management to provide a complete agent execution environment.
    """

    def __init__(self, agent_factory: IAgentFactory, event_publisher: IEventPublisher):
        """
        Initialize the agent runtime service.

        Args:
            agent_factory: Agent factory for creating configured agents (required)
            event_publisher: Event publisher for runtime events (required)
        """
        self._agent_factory = agent_factory
        self._event_publisher = event_publisher
        self._initialized = False
        self._runtime_agents: dict[str, Any] = {}  # Pre-created agents at startup
        self._runtime_metrics = {"agents_created": 0, "agents_executed": 0, "runtime_errors": 0}

    async def initialize(self) -> None:
        """
        Initialize the agent runtime service.

        Performs startup sequence to initialize agent factory and create all runtime agents.
        """
        if self._initialized:
            logger.warning("Agent runtime service already initialized")
            return

        logger.info("Initializing Agent Runtime Service")

        # Initialize agent factory
        await self._agent_factory.initialize()
        logger.info("Agent factory initialized")

        # Skip creating all agents at startup - use lazy loading instead
        logger.info("Using lazy agent creation - agents will be created on first use")

        self._initialized = True

        # Emit initialization event
        await self._event_publisher.emit_runtime_initialized(
            framework="agno",
            agent_factory_available=True,
            runtime_agents_created=len(self._runtime_agents),
        )

        logger.info(
            f"Agent Runtime Service initialized with {len(self._runtime_agents)} runtime agents"
        )

    async def _create_and_cache_agent(
        self, task_type: TaskType, agent_type: AgentType
    ) -> IConfigurableAgent | None:
        """
        Create and cache agent for specified task and agent types.

        Args:
            task_type: Task type enum
            agent_type: Agent type enum

        Returns:
            IConfigurableAgent instance or None if creation failed
        """
        agent_key = f"{task_type.value}_{agent_type.value}"

        try:
            # Get agent config from factory
            config_dict = self._agent_factory.get_agent_config(task_type, agent_type)
            agent_config = AgentConfig(**config_dict)

            # Create agent
            agent = await self._agent_factory.create_agent(agent_config)

            # Cache the agent
            self._runtime_agents[agent_key] = agent
            self._increment_metric("agents_created")
            logger.debug(f"Created and cached agent: {agent_key}")

            return agent

        except Exception as e:
            logger.error(f"Failed to create agent {agent_key}: {e}")
            return None

    async def _create_all_agents(self) -> None:
        """Create all configured agents at startup and cache them."""
        created_count = 0
        failed_count = 0

        for task_type in TaskType:
            for agent_type in AgentType:
                try:
                    agent = await self._create_and_cache_agent(task_type, agent_type)
                    if agent:
                        created_count += 1
                    else:
                        failed_count += 1
                except Exception:
                    failed_count += 1

        logger.info(
            f"Runtime agent creation complete: {created_count} created, {failed_count} skipped"
        )

    async def shutdown(self) -> None:
        """Shutdown the agent runtime service."""
        if not self._initialized:
            return

        logger.info("Shutting down Agent Runtime Service")

        # Clean up runtime agents
        self._runtime_agents.clear()

        # Emit shutdown event
        await self._event_publisher.emit_runtime_shutdown(metrics=self._runtime_metrics.copy())

        self._initialized = False
        logger.info("Agent Runtime Service shutdown complete")

    async def get_agent(self, task_type: TaskType, agent_type: AgentType) -> IConfigurableAgent:
        """
        Get agent for the specified task type and agent type (lazy creation).

        Args:
            task_type: Task type enum
            agent_type: Agent type enum

        Returns:
            IConfigurableAgent instance (created on first use)

        Raises:
            RuntimeError: If service not initialized or agent config not available
        """
        self._ensure_initialized()

        # Generate agent key
        agent_key = f"{task_type.value}_{agent_type.value}"

        # Check if agent already exists in cache
        if agent_key in self._runtime_agents:
            logger.debug(f"Returning cached agent: {agent_key}")
            return self._runtime_agents[agent_key]

        # Create agent on-demand (lazy loading)
        logger.info(f"Creating agent on-demand: {agent_key}")
        agent = await self._create_and_cache_agent(task_type, agent_type)

        if not agent:
            raise RuntimeError(f"Agent {agent_key} not available")

        return agent

    async def execute_agent(
        self,
        agent: IConfigurableAgent,
        task: TaskNode,
        context: TaskContext | None = None,
        agent_type: AgentType | None = None,
        execution_id: str | None = None,
    ) -> AnyResultEnvelope:
        """
        Execute an agent with the given task and optional TaskContext.

        Args:
            agent: Agent instance to execute
            task: Task to execute
            context: Optional TaskContext with rich multimodal context
            agent_type: Type of agent (ATOMIZER, PLANNER, EXECUTOR, AGGREGATOR) for metadata
            execution_id: Execution ID for session isolation (if supported by framework)

        Returns:
            ResultEnvelope with execution result
        """
        self._ensure_initialized()
        start_time = datetime.now()

        try:
            # Emit execution start event
            await self._event_publisher.emit_agent_execution_started(task, agent, context)

            # Execute ConfigurableAgent directly with TaskContext
            # Pass execution_id for session isolation if supported
            if hasattr(agent, "set_execution_context") and execution_id:
                agent.set_execution_context(execution_id)

            structured_result = await agent.run(task, context)

            # Validate that agent returned a proper result
            if structured_result is None:
                raise ValueError(
                    f"Agent {agent.name} returned None result for task {task.task_id} - this indicates an agent execution problem"
                )

            # Create result envelope with execution metrics
            result_envelope = self._create_result_envelope(
                structured_result=structured_result,
                task=task,
                agent=agent,
                agent_type=agent_type,
                execution_id=execution_id,
                start_time=start_time,
            )

            # Update metrics
            self._increment_metric("agents_executed")

            # Emit execution success event
            await self._event_publisher.emit_agent_execution_completed(task, agent, success=True)

            logger.debug(
                f"Executed agent for task {task.task_id} with context: {context is not None}"
            )
            return result_envelope

        except Exception as e:
            self._increment_metric("runtime_errors")
            await self._event_publisher.emit_agent_execution_failed(task, agent, str(e))
            logger.error(f"Failed to execute agent for task {task.task_id}: {e}")
            raise

    def get_runtime_metrics(self) -> dict[str, Any]:
        """Get runtime performance metrics."""
        return {
            **self._runtime_metrics.copy(),
            "runtime_agents_available": len(self._runtime_agents),
            "framework": "agno",
            "initialized": self._initialized,
        }

    def get_framework_name(self) -> str:
        """Get the current framework name."""
        return "agno"

    def is_initialized(self) -> bool:
        """Check if the runtime service is initialized."""
        return self._initialized

    def _ensure_initialized(self) -> None:
        """Ensure the runtime service is initialized."""
        if not self._initialized:
            raise RuntimeError("Agent runtime service not initialized. Call initialize() first.")

    def _increment_metric(self, metric_name: str) -> None:
        """Increment a runtime metric."""
        if metric_name in self._runtime_metrics:
            self._runtime_metrics[metric_name] += 1
        else:
            logger.warning(f"Unknown metric: {metric_name}")

    def _extract_execution_metrics(
        self, structured_result: Any, start_time: datetime
    ) -> ExecutionMetrics:
        """
        Extract execution metrics from agent result.

        Args:
            structured_result: Agent execution result
            start_time: Execution start time

        Returns:
            ExecutionMetrics with timing and usage data
        """
        execution_time = (datetime.now() - start_time).total_seconds()
        return ExecutionMetrics(
            execution_time=execution_time,
            tokens_used=getattr(structured_result, "tokens_used", 0) if structured_result else 0,
            model_calls=1,
            cost_estimate=getattr(structured_result, "cost_estimate", 0.0)
            if structured_result
            else 0.0,
        )

    def _create_result_envelope(
        self,
        structured_result: Any,
        task: TaskNode,
        agent: IConfigurableAgent,
        agent_type: AgentType | None,
        execution_id: str | None,
        start_time: datetime,
    ) -> ResultEnvelope:
        """
        Create result envelope with execution metrics and metadata.

        Args:
            structured_result: Agent execution result
            task: Task that was executed
            agent: Agent instance that executed the task
            agent_type: Type of agent for metadata
            execution_id: Execution ID for session isolation
            start_time: Execution start time

        Returns:
            Complete result envelope with metrics and metadata
        """
        # Extract execution metrics
        execution_metrics = self._extract_execution_metrics(structured_result, start_time)

        # Keep typed Pydantic result as-is (avoid converting to dict)
        primary_result = structured_result
        output_text = None  # Envelope will provide extract_primary_output()

        # Create result envelope with proper agent type
        return ResultEnvelope.create_success(
            result=primary_result,
            task_id=task.task_id,
            execution_id=f"agent_{task.task_id}",
            agent_type=agent_type or AgentType.EXECUTOR,
            execution_metrics=execution_metrics,
            artifacts=[],  # Agents don't typically create artifacts directly
            output_text=output_text,
            metadata={
                "agent_name": agent.name,
                "agent_type": (agent_type or AgentType.EXECUTOR).value,
                "response_type": type(structured_result).__name__ if structured_result else "None",
                "framework": "agno",
                "execution_id": execution_id or "unknown",
            },
        )
