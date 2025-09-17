"""
System Manager - ROMA Infrastructure Layer.

Central orchestrator and composition root for general agentic task execution.
Handles any type of task (coding, analysis, research, creative work, etc.) through
intelligent agent decomposition and parallel execution.

Acts as the main entry point and dependency injection container for the system.
Integrates: ContextBuilder + Storage + ToolkitManager + Agent Runtime
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.value_objects.agent_type import AgentType
from src.roma.domain.value_objects.node_type import NodeType
from src.roma.domain.value_objects.config.roma_config import ROMAConfig
from src.roma.domain.value_objects.result_envelope import ResultEnvelope, ExecutionMetrics, AnyResultEnvelope
from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from src.roma.application.services.event_store import InMemoryEventStore
from src.roma.application.services.agent_runtime_service import AgentRuntimeService
from src.roma.application.services.artifact_service import ArtifactService
from src.roma.application.services.context_builder_service import ContextBuilderService, TaskContext
from src.roma.application.orchestration.graph_state_manager import GraphStateManager
from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
from src.roma.application.services.recovery_manager import RecoveryManager
from src.roma.infrastructure.toolkits.agno_toolkit_manager import AgnoToolkitManager
from src.roma.infrastructure.storage.local_storage import LocalFileStorage
from src.roma.infrastructure.storage.storage_interface import StorageConfig as InfraStorageConfig
from src.roma.infrastructure.agents.agent_factory import AgentFactory

logger = logging.getLogger(__name__)


class SystemManager:
    """
    ROMA System Manager - Central orchestrator with clean architecture.

    Manages all system components and provides the main interface for
    executing any type of task through intelligent agent decomposition.
    Configuration is injected directly from Hydra/CLI.
    """
    
    def __init__(self, config: ROMAConfig):
        """
        Initialize system manager with configuration object.
        
        Args:
            config: ROMAConfig object from Hydra CLI
        """
        self.config = config
        self._initialized = False
        
        # Core components
        self._event_store: Optional[InMemoryEventStore] = None
        self._task_graph: Optional[DynamicTaskGraph] = None
        self._graph_state_manager: Optional[GraphStateManager] = None
        self._parallel_execution_engine: Optional[ParallelExecutionEngine] = None
        self._agent_runtime_service: Optional[AgentRuntimeService] = None
        self._toolkit_manager: Optional[AgnoToolkitManager] = None
        self._context_builder: Optional[ContextBuilderService] = None
        self._storage: Optional[LocalFileStorage] = None
        self._artifact_service: Optional[ArtifactService] = None
        self._recovery_manager: Optional[RecoveryManager] = None
        self._agent_factory: Optional[AgentFactory] = None
        
        # System state
        self._current_profile: Optional[str] = None
        self._active_executions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("SystemManager initialized with configuration")
        
    async def initialize(self, profile_name: str) -> None:
        """
        Initialize system with specified profile.
        
        Args:
            profile_name: Name of the profile to initialize with
        """
        if self._initialized:
            logger.warning("SystemManager already initialized")
            return
            
        logger.info(f"Initializing ROMA v2 system with profile: {profile_name}")
        
        try:
            # Initialize components in dependency order
            await self._initialize_event_store()
            await self._initialize_task_graph()
            await self._initialize_storage()
            await self._initialize_artifact_service()
            await self._initialize_context_builder()
            await self._initialize_toolkit_manager()
            await self._initialize_agent_factory()
            await self._initialize_agent_runtime_service()
            await self._initialize_recovery_manager()
            await self._initialize_graph_state_manager()
            await self._initialize_parallel_execution_engine()

            # Load profile from config
            await self._load_profile(profile_name)
            
            self._initialized = True
            self._current_profile = profile_name
            
            logger.info(f"✅ ROMA system initialized with profile: {profile_name}")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            await self._cleanup()
            raise
            
    async def _initialize_event_store(self) -> None:
        """Initialize event store."""
        self._event_store = InMemoryEventStore()
        # No initialization needed for in-memory store
        
    async def _initialize_task_graph(self) -> None:
        """Initialize dynamic task graph."""
        self._task_graph = DynamicTaskGraph()
        
    async def _initialize_storage(self) -> None:
        """Initialize goofys-based storage."""
        mount_path = self.config.storage.mount_path

        # Create infrastructure StorageConfig from domain config
        config = InfraStorageConfig.from_mount_path(mount_path)
        self._storage = LocalFileStorage(config)
        logger.info(f"Storage initialized at: {mount_path}")

    async def _initialize_artifact_service(self) -> None:
        """Initialize artifact service using existing storage."""
        if not self._storage:
            raise RuntimeError("Storage must be initialized before ArtifactService")

        self._artifact_service = ArtifactService(self._storage)
        await self._artifact_service.initialize()
        logger.info("✅ ArtifactService initialized")

    async def _initialize_context_builder(self) -> None:
        """Initialize context builder service."""
        self._context_builder = ContextBuilderService()
        logger.info("Context builder initialized")
        
    async def _initialize_toolkit_manager(self) -> None:
        """Initialize toolkit manager."""
        self._toolkit_manager = AgnoToolkitManager()
        await self._toolkit_manager.initialize()

    async def _initialize_agent_factory(self) -> None:
        """Initialize agent factory with configuration."""
        self._agent_factory = AgentFactory(self.config)
        logger.info("AgentFactory initialized")
        
    async def _initialize_agent_runtime_service(self) -> None:
        """Initialize agent runtime service with dependencies."""
        # Create agent runtime service with simplified dependencies
        self._agent_runtime_service = AgentRuntimeService(
            event_store=self._event_store,
            agent_factory=self._agent_factory
        )

        await self._agent_runtime_service.initialize()
        
    async def _initialize_recovery_manager(self) -> None:
        """Initialize recovery manager with circuit breaker."""
        self._recovery_manager = RecoveryManager()
        logger.info("✅ RecoveryManager initialized")

    async def _initialize_graph_state_manager(self) -> None:
        """Initialize graph state manager for atomic state transitions."""
        self._graph_state_manager = GraphStateManager(
            task_graph=self._task_graph,
            event_store=self._event_store
        )
        logger.info("✅ GraphStateManager initialized")

    async def _initialize_parallel_execution_engine(self) -> None:
        """Initialize parallel execution engine for concurrent task processing."""
        self._parallel_execution_engine = ParallelExecutionEngine(
            task_graph=self._task_graph,
            agent_factory=self._agent_factory,
            context_builder=self._context_builder,
            max_concurrent_tasks=self.config.execution.max_concurrent_tasks
        )
        logger.info("✅ ParallelExecutionEngine initialized")

    async def _load_profile(self, profile_name: str) -> None:
        """Load and validate agent profile configuration from injected config."""
        # Use the profile from ROMAConfig
        profile_config = self.config.profile
        logger.info(f"Profile {profile_name} loaded successfully with config: {profile_config}")
            
    async def execute_task(self, task: str, **options) -> Dict[str, Any]:
        """
        Execute any type of task with full multimodal context support.

        Can handle coding, analysis, research, creative work, data processing,
        or any other task through intelligent agent decomposition.

        Args:
            task: Task description to execute
            **options: Additional execution options

        Returns:
            Execution result dictionary
        """
        if not self._initialized:
            raise RuntimeError("SystemManager not initialized")
            
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        logger.info(f"Starting task execution [{execution_id}]: {task[:50]}...")

        try:
            # Create root task node
            root_task = TaskNode(
                task_id=f"root_{execution_id}",
                goal=task,
                task_type=TaskType.THINK,  # Will be determined by atomizer
                status=TaskStatus.PENDING
            )
            
            # Add to task graph
            await self._task_graph.add_node(root_task)
            
            # Build initial execution context - delegate to ContextBuilderService
            execution_context = await self._context_builder.build_context(
                task=root_task,
                overall_objective=root_task.goal,
                execution_metadata=options
            )
            
            # Track execution with context
            execution_info = {
                "execution_id": execution_id,
                "root_task": root_task,
                "start_time": start_time,
                "status": TaskStatus.EXECUTING,
                "task": task,
                "options": options,
                "context": execution_context
            }
            self._active_executions[execution_id] = execution_info
            
            # Execute task through runtime service with context - now returns ResultEnvelope
            result_envelope = await self._execute_task_with_context(
                root_task, execution_id, execution_context
            )

            # Store artifacts from envelope using ArtifactService
            artifact_refs = await self._artifact_service.store_envelope_artifacts(
                execution_id, result_envelope
            )

            # Update execution tracking
            execution_info["status"] = TaskStatus.COMPLETED
            execution_info["end_time"] = datetime.now()
            execution_info["result_envelope"] = result_envelope
            execution_info["artifact_refs"] = artifact_refs

            # Return standardized ResultEnvelope format
            return {
                "execution_id": execution_id,
                "task": task,
                "status": "completed",
                "result": result_envelope.extract_primary_output(),
                "execution_time": result_envelope.execution_metrics.execution_time,
                "node_count": len(self._task_graph.get_all_nodes()),
                "framework": self._agent_runtime_service.get_framework_name(),
                "artifacts": [artifact.to_dict() for artifact in result_envelope.artifacts],
                "envelope": result_envelope.to_dict()  # Full envelope data
            }
            
        except Exception as e:
            logger.error(f"Task execution failed [{execution_id}]: {e}")
            
            if execution_id in self._active_executions:
                self._active_executions[execution_id]["status"] = TaskStatus.FAILED
                
            raise
            
    async def _execute_task_with_context(
        self,
        task: TaskNode,
        execution_id: str,
        context: TaskContext
    ) -> AnyResultEnvelope:
        """Coordinate task execution through specialized components: atomizer → plan/execute → aggregate."""
        start_time = datetime.now()
        try:
            # Phase 1: Atomizer Decision - delegate to AgentRuntimeService
            await self._graph_state_manager.transition_node_status(task.task_id, TaskStatus.READY)
            await self._graph_state_manager.transition_node_status(task.task_id, TaskStatus.EXECUTING)

            # Get atomizer agent and execute
            atomizer_agent = await self._agent_runtime_service.get_agent(task.task_type, AgentType.ATOMIZER)
            atomizer_result = await self._agent_runtime_service.execute_agent(atomizer_agent, task, context)

            # Parse atomizer result - should be AtomizerResult with node_type field
            if not isinstance(atomizer_result, dict) or "node_type" not in atomizer_result:
                raise ValueError(f"Invalid atomizer result format: {atomizer_result}")

            node_type = NodeType.from_string(atomizer_result["node_type"]) if isinstance(atomizer_result["node_type"], str) else atomizer_result["node_type"]
            logger.info(f"Atomizer decision for task {task.task_id}: {node_type} (is_atomic: {atomizer_result.get('is_atomic')})")

            # Update task node with determined node_type
            updated_task = task.model_copy(update={"node_type": node_type})
            await self._task_graph.update_node(updated_task)

            # Phase 2: Delegate execution to specialized components based on node type
            if node_type == NodeType.EXECUTE:
                # Atomic execution - delegate to AgentRuntimeService
                executor_agent = await self._agent_runtime_service.get_agent(updated_task.task_type, AgentType.EXECUTOR)
                result = await self._agent_runtime_service.execute_agent(executor_agent, updated_task, context)

            elif node_type == NodeType.PLAN:
                # Planned execution - use planner agent to decompose task
                planner_agent = await self._agent_runtime_service.get_agent(updated_task.task_type, AgentType.PLANNER)
                planner_result = await self._agent_runtime_service.execute_agent(planner_agent, updated_task, context)

                # For now, return planner result as a simple implementation
                # TODO: In future, create subtasks and execute them in parallel
                result = planner_result

            else:
                raise ValueError(f"Unknown node type: {node_type}")

            # Task completed successfully - coordinate final state transition
            await self._graph_state_manager.transition_node_status(task.task_id, TaskStatus.COMPLETED)
            await self._recovery_manager.record_success(task.task_id)

            # Create standardized ResultEnvelope
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_metrics = ExecutionMetrics(
                execution_time=execution_time,
                tokens_used=result.get("tokens_used", 0),
                model_calls=result.get("model_calls", 1),
                cost_estimate=result.get("cost_estimate", 0.0)
            )

            # Determine agent type based on execution path
            primary_agent_type = AgentType.EXECUTOR if node_type == NodeType.EXECUTE else AgentType.AGGREGATOR

            # Create result envelope with coordinated execution metadata
            envelope = ResultEnvelope.create_success(
                result=result,
                task_id=task.task_id,
                execution_id=execution_id,
                agent_type=primary_agent_type,
                execution_metrics=execution_metrics,
                artifacts=result.get("artifacts", []) if isinstance(result.get("artifacts"), list) else [],
                output_text=result.get("result", f"Completed: {task.goal}"),
                metadata={
                    "coordinated_by": "SystemManager",
                    "framework": self._agent_runtime_service.get_framework_name(),
                    "execution_path": result.get("execution_path", "coordinated"),
                    "node_type": node_type.value
                }
            )

            return envelope

        except Exception as e:
            logger.error(f"Task execution coordination failed for {task.task_id}: {e}")

            # Use recovery manager to coordinate failure handling
            recovery_result = await self._recovery_manager.handle_failure(task, e)

            if recovery_result.updated_node:
                # Coordinate recovery action
                updated_task = recovery_result.updated_node
                await self._graph_state_manager.transition_node_status(
                    updated_task.task_id,
                    updated_task.status
                )
            else:
                # Mark task as failed if no recovery possible
                await self._graph_state_manager.transition_node_status(task.task_id, TaskStatus.FAILED)

# TODO: In future iterations, create error_envelope with ExecutionMetrics and return instead of raising

            # For now, still raise to maintain error flow, but return envelope in future iterations
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        if not self._initialized:
            return {"status": "not_initialized"}
            
        runtime_metrics = self._agent_runtime_service.get_runtime_metrics()
        
        return {
            "status": "initialized",
            "current_profile": self._current_profile,
            "framework": self._agent_runtime_service.get_framework_name(),
            "active_executions": len(self._active_executions),
            "total_nodes": len(self._task_graph.get_all_nodes()) if self._task_graph else 0,
            "components": {
                "event_store": self._event_store is not None,
                "task_graph": self._task_graph is not None,
                "agent_runtime_service": self._agent_runtime_service is not None,
                "toolkit_manager": self._toolkit_manager is not None
            },
            "runtime_metrics": runtime_metrics
        }
        
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration."""
        errors = []
        warnings = []
        
        # Check if profile is configured
        if not self.config.profile:
            warnings.append("Profile not configured")
            
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
        
    def get_current_profile(self) -> Optional[str]:
        """Get current active profile."""
        return self._current_profile
        
    def get_available_profiles(self) -> List[str]:
        """Get list of available profiles."""
        # Return the current profile name since ROMAConfig only contains one profile
        return [self.config.profile.name] if self.config.profile.name else []
        
    async def switch_profile(self, profile_name: str) -> Dict[str, Any]:
        """Switch to different profile."""
        if not self._initialized:
            raise RuntimeError("SystemManager not initialized")
            
        try:
            logger.info(f"Switching to profile: {profile_name}")
            
            # Clear current state
            self._active_executions.clear()
            
            # Reset task graph
            if self._task_graph:
                self._task_graph = DynamicTaskGraph()
                
            # Load new profile
            await self._load_profile(profile_name)
            self._current_profile = profile_name
            
            return {
                "success": True,
                "profile": profile_name,
                "message": f"Switched to {profile_name}",
                "system_info": self.get_system_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to switch profile: {e}")
            return {
                "success": False,
                "error": str(e),
                "profile": profile_name
            }
            
    async def shutdown(self) -> None:
        """Shutdown system manager and all components."""
        if not self._initialized:
            return
            
        logger.info("Shutting down ROMA SystemManager")
        
        # Cancel active executions
        for execution_info in self._active_executions.values():
            if execution_info["status"] in [TaskStatus.PENDING, TaskStatus.EXECUTING]:
                execution_info["status"] = TaskStatus.FAILED
                
        self._active_executions.clear()
        
        # Shutdown components
        if self._agent_runtime_service:
            await self._agent_runtime_service.shutdown()
            
        if self._event_store:
            await self._event_store.clear()
            
        self._initialized = False
        logger.info("✅ SystemManager shutdown complete")
        
    async def _cleanup(self) -> None:
        """Clean up partially initialized components."""
        logger.info("Cleaning up SystemManager components")
        
        if self._agent_runtime_service:
            try:
                await self._agent_runtime_service.shutdown()
            except Exception as e:
                logger.error(f"Error cleaning up agent runtime service: {e}")
                
        if self._event_store:
            try:
                await self._event_store.clear()
            except Exception as e:
                logger.error(f"Error cleaning up event store: {e}")
                
        if self._recovery_manager:
            try:
                self._recovery_manager.reset_circuit_breaker()
                self._recovery_manager.clear_permanent_failures()
            except Exception as e:
                logger.error(f"Error cleaning up recovery manager: {e}")
                
        self._initialized = False