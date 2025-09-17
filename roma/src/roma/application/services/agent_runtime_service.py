"""
Agent Runtime Service - Application Layer.

Orchestrates agent runtime lifecycle and coordinates with other services.
This service provides the application-level logic for managing the agent runtime,
connecting it to the event system, and coordinating with configuration and toolkit systems.

Task 1.3.1: Framework-agnostic agent runtime abstraction
"""

import logging
from typing import Dict, Any, Optional, List, Union

from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.agent_type import AgentType
from src.roma.domain.value_objects.agent_responses import AtomizerResult
from src.roma.application.services.event_store import InMemoryEventStore
from src.roma.application.services.context_builder_service import TaskContext
from src.roma.infrastructure.agents.agent_factory import AgentFactory

logger = logging.getLogger(__name__)


class AgentRuntimeService:
    """
    Application service for managing agent runtime lifecycle.
    
    Coordinates between the agent runtime, event system, configuration,
    and toolkit management to provide a complete agent execution environment.
    """
    
    def __init__(
        self,
        event_store: Optional[InMemoryEventStore] = None,
        agent_factory: Optional[AgentFactory] = None
    ):
        """
        Initialize the agent runtime service.

        Args:
            event_store: Event store for runtime events
            agent_factory: Agent factory for creating configured agents
        """
        self._event_store = event_store
        self._agent_factory = agent_factory
        self._initialized = False
        self._runtime_agents: Dict[str, Any] = {}  # Pre-created agents at startup
        self._runtime_metrics = {
            "agents_created": 0,
            "agents_executed": 0,
            "runtime_errors": 0
        }
        
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
        if self._agent_factory:
            await self._agent_factory.initialize()
            logger.info("Agent factory initialized")

        # Skip creating all agents at startup - use lazy loading instead
        logger.info("Using lazy agent creation - agents will be created on first use")

        self._initialized = True

        # Emit initialization event
        await self._emit_runtime_event("runtime_initialized", {
            "framework": "agno",
            "agent_factory_available": self._agent_factory is not None,
            "runtime_agents_created": len(self._runtime_agents)
        })

        logger.info(f"Agent Runtime Service initialized with {len(self._runtime_agents)} runtime agents")

    async def _create_all_agents(self) -> None:
        """Create all configured agents at startup and cache them."""
        if not self._agent_factory:
            logger.warning("No agent factory available - skipping agent creation")
            return

        from src.roma.domain.value_objects.task_type import TaskType
        from src.roma.domain.value_objects.agent_type import AgentType

        created_count = 0
        failed_count = 0

        for task_type in TaskType:
            for agent_type in AgentType:
                agent_key = f"{task_type.value}_{agent_type.value}"

                try:
                    # Get agent configuration dictionary from factory
                    config_dict = self._agent_factory.get_agent_config(task_type, agent_type)

                    # Convert dictionary to AgentConfig object
                    from src.roma.domain.value_objects.config.agent_config import AgentConfig
                    agent_config = AgentConfig.from_dict(config_dict)

                    # Create agent using factory with proper AgentConfig object
                    agent = await self._agent_factory.create_agent(agent_config)
                    self._runtime_agents[agent_key] = agent
                    created_count += 1
                    logger.info(f"✅ Created runtime agent: {agent_key}")

                except Exception as e:
                    failed_count += 1
                    logger.debug(f"⚠️  No config for {agent_key}: {e}")
                    # Not an error - some agents may not be configured

        logger.info(f"Runtime agent creation complete: {created_count} created, {failed_count} skipped")
        self._runtime_metrics["agents_created"] = created_count

    async def shutdown(self) -> None:
        """Shutdown the agent runtime service."""
        if not self._initialized:
            return
            
        logger.info("Shutting down Agent Runtime Service")
        
        # Clean up runtime agents
        self._runtime_agents.clear()
        
        # Emit shutdown event
        await self._emit_runtime_event("runtime_shutdown", {
            "metrics": self._runtime_metrics.copy()
        })
        
        self._initialized = False
        logger.info("Agent Runtime Service shutdown complete")
        
    async def get_agent(self, task_type: TaskType, agent_type: AgentType) -> Any:
        """
        Get agent for the specified task type and agent type (lazy creation).

        Args:
            task_type: Task type enum
            agent_type: Agent type enum

        Returns:
            Agent instance (created on first use)

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
        try:
            # Get agent configuration dictionary from factory
            config_dict = self._agent_factory.get_agent_config(task_type, agent_type)

            # Convert dictionary to AgentConfig object
            from src.roma.domain.value_objects.config.agent_config import AgentConfig
            agent_config = AgentConfig.from_dict(config_dict)

            # Create agent using factory with proper AgentConfig object
            agent = await self._agent_factory.create_agent(agent_config)

            # Cache the agent for future use
            self._runtime_agents[agent_key] = agent
            self._runtime_metrics["agents_created"] += 1

            logger.info(f"✅ Created and cached agent on-demand: {agent_key}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent {agent_key}: {e}")
            raise RuntimeError(f"Agent {agent_key} not available - {e}")
            
    async def execute_agent(self, agent: Any, task: TaskNode, context: Optional[TaskContext] = None) -> Dict[str, Any]:
        """
        Execute an agent with the given task and optional TaskContext.

        Args:
            agent: Agent instance to execute
            task: Task to execute
            context: Optional TaskContext with rich multimodal context

        Returns:
            Execution result
        """
        self._ensure_initialized()
        
        try:
            # Emit execution start event
            await self._emit_runtime_event("agent_execution_started", {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "agent_name": getattr(agent, 'name', 'unknown'),
                "context_provided": context is not None,
                "context_files": len([item for item in context.context_items if item.content_type.value in ["IMAGE", "AUDIO", "VIDEO", "FILE"]]) if context else 0
            })

            # Execute ConfigurableAgent directly with TaskContext
            structured_result = await agent.run(task, context)

            # Convert structured response to runtime format
            result = {
                "result": structured_result.model_dump() if hasattr(structured_result, 'model_dump') else structured_result,
                "success": True,
                "agent_name": getattr(agent, 'agent_name', 'ConfigurableAgent'),
                "response_type": type(structured_result).__name__ if structured_result else "None"
            }
            
            # Update metrics
            self._runtime_metrics["agents_executed"] += 1
            
            # Emit execution success event
            await self._emit_runtime_event("agent_execution_completed", {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "agent_name": getattr(agent, 'name', 'unknown'),
                "success": result.get("success", True),
                "artifacts_created": len(result.get("artifacts", []))
            })
            
            logger.debug(f"Executed agent for task {task.task_id} with context: {context is not None}")
            return result
            
        except Exception as e:
            self._runtime_metrics["runtime_errors"] += 1
            await self._emit_runtime_event("agent_execution_failed", {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "agent_name": getattr(agent, 'name', 'unknown'),
                "error": str(e)
            })
            logger.error(f"Failed to execute agent for task {task.task_id}: {e}")
            raise
            
        
    def get_runtime_metrics(self) -> Dict[str, Any]:
        """Get runtime performance metrics."""
        return {
            **self._runtime_metrics.copy(),
            "runtime_agents_available": len(self._runtime_agents),
            "framework": "agno",
            "initialized": self._initialized
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
            
    async def _emit_runtime_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a runtime event."""
        try:
            if self._event_store:
                from src.roma.domain.events.task_events import BaseTaskEvent, utc_now
                from uuid import uuid4

                event = BaseTaskEvent(
                    event_id=str(uuid4()),
                    event_type=f"runtime.{event_type}",
                    task_id=data.get("task_id", "runtime-event"),
                    timestamp=utc_now(),
                    metadata={**data, "framework": "agno"}
                )
                await self._event_store.append(event)
        except Exception as e:
            logger.error(f"Failed to emit runtime event {event_type}: {e}")
            # Don't fail the main operation if event emission fails