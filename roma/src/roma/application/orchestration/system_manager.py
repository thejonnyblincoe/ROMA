"""
System Manager - ROMA Infrastructure Layer.

Central orchestrator and composition root for general agentic task execution.
Handles any type of task (coding, analysis, research, creative work, etc.) through
intelligent agent decomposition and parallel execution.

Acts as the main entry point and dependency injection container for the system.
Integrates: ContextBuilder + Storage + ToolkitManager + Agent Runtime
"""

import contextlib
import logging
import uuid
from datetime import datetime
from typing import Any

from roma.application.orchestration.execution_orchestrator import ExecutionOrchestrator
from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.agent_service_registry import AgentServiceRegistry
from roma.application.services.artifact_service import ArtifactService
from roma.application.services.checkpoint_service import CheckpointService
from roma.application.services.context_builder_service import ContextBuilderService
from roma.application.services.deadlock_detector import DeadlockDetector
from roma.application.services.event_publisher import EventPublisher, initialize_event_publisher
from roma.application.services.event_store import InMemoryEventStore
from roma.application.services.execution_context import ExecutionContext
from roma.application.services.hitl_service import HITLService
from roma.application.services.knowledge_store_service import KnowledgeStoreService
from roma.application.services.recovery_manager import RecoveryManager
from roma.domain.entities.task_node import TaskNode
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.interfaces.agent_factory import IAgentFactory
from roma.domain.interfaces.storage import IStorage
from roma.domain.value_objects.config.roma_config import ROMAConfig
from roma.domain.value_objects.storage_config import StorageConfig
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType
from roma.infrastructure.agents.agent_factory import AgentFactory
from roma.infrastructure.persistence.connection_manager import DatabaseConnectionManager
from roma.infrastructure.persistence.postgres_event_store import PostgreSQLEventStore
from roma.infrastructure.persistence.repositories.checkpoint_repository_impl import (
    SQLAlchemyCheckpointRepository,
    SQLAlchemyRecoveryRepository,
)
from roma.infrastructure.persistence.repositories.execution_history_repository_impl import (
    SQLAlchemyExecutionHistoryRepository,
)
from roma.infrastructure.storage.local_storage import LocalFileStorage
from roma.infrastructure.toolkits.agno_toolkit_manager import AgnoToolkitManager

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
        self._event_store: InMemoryEventStore | PostgreSQLEventStore | None = None
        self._event_publisher: EventPublisher | None = None
        self._connection_manager: DatabaseConnectionManager | None = None
        self._task_graph: DynamicTaskGraph | None = None
        self._graph_state_manager: GraphStateManager | None = None
        self._parallel_execution_engine: ParallelExecutionEngine | None = None
        self._execution_orchestrator: ExecutionOrchestrator | None = None
        self._agent_service_registry: AgentServiceRegistry | None = None
        self._agent_runtime_service: AgentRuntimeService | None = None
        self._toolkit_manager: AgnoToolkitManager | None = None
        self._context_builder: ContextBuilderService | None = None
        self._storage: IStorage | None = None
        self._artifact_service: ArtifactService | None = None
        self._knowledge_store: KnowledgeStoreService | None = None
        self._recovery_manager: RecoveryManager | None = None
        self._agent_factory: IAgentFactory | None = None
        self._hitl_service: HITLService | None = None
        self._deadlock_detector: DeadlockDetector | None = None

        # Persistence components
        self._checkpoint_repository: SQLAlchemyCheckpointRepository | None = None
        self._recovery_repository: SQLAlchemyRecoveryRepository | None = None
        self._execution_history_repository: SQLAlchemyExecutionHistoryRepository | None = None
        self._checkpoint_service: Any | None = None

        # System state
        self._current_profile: str | None = None
        self._active_executions: dict[str, dict[str, Any]] = {}
        self._active_contexts: dict[str, ExecutionContext] = {}

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
            await self._initialize_event_publisher()
            await self._initialize_task_graph()
            await self._initialize_storage()
            await self._initialize_artifact_service()
            await self._initialize_knowledge_store()
            await self._initialize_context_builder()
            await self._initialize_toolkit_manager()
            await self._load_tools_registry()
            await self._initialize_agent_factory()
            await self._initialize_agent_runtime_service()
            await self._initialize_recovery_manager()
            await self._initialize_persistence_repositories()
            await self._initialize_checkpoint_service()
            await self._initialize_hitl_service()
            await self._initialize_deadlock_detector()
            await self._initialize_graph_state_manager()
            await self._initialize_parallel_execution_engine()
            await self._initialize_agent_service_registry()
            await self._initialize_execution_orchestrator()

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
        """Initialize event store - PostgreSQL if configured, otherwise in-memory."""
        # Check if PostgreSQL is configured and available
        if (
            self.config.database
            and self.config.database.host
            and self.config.database.host.lower() not in ["localhost", "127.0.0.1"]
            or self.config.database.get_connection_string_from_env()
        ):
            try:
                # Initialize PostgreSQL connection manager
                self._connection_manager = DatabaseConnectionManager(self.config.database)
                await self._connection_manager.initialize()

                # Test connection
                if self._connection_manager.is_healthy():
                    # Initialize PostgreSQL event store
                    self._event_store = PostgreSQLEventStore(self._connection_manager)
                    await self._event_store.initialize()
                    logger.info("✅ PostgreSQL event store initialized")
                else:
                    logger.warning(
                        "PostgreSQL connection unhealthy, falling back to in-memory store"
                    )
                    self._event_store = InMemoryEventStore()
            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL event store: {e}")
                logger.info("Falling back to in-memory event store")
                self._event_store = InMemoryEventStore()
                if self._connection_manager:
                    await self._connection_manager.close()
                    self._connection_manager = None
        else:
            # Use in-memory store for development/testing
            self._event_store = InMemoryEventStore()
            logger.info("✅ In-memory event store initialized")

    async def _initialize_event_publisher(self) -> None:
        """Initialize event publisher with event store."""
        if not self._event_store:
            raise RuntimeError("Event store must be initialized before EventPublisher")

        self._event_publisher = initialize_event_publisher(self._event_store)
        logger.info("✅ EventPublisher initialized")

    async def _initialize_task_graph(self) -> None:
        """Initialize dynamic task graph."""
        self._task_graph = DynamicTaskGraph()
        logger.info("✅ DynamicTaskGraph initialized")

        # Note: Event publishing is handled by GraphStateManager to avoid duplicates

    async def _initialize_storage(self) -> None:
        """Initialize goofys-based storage."""
        mount_path = self.config.storage.mount_path

        # Create StorageConfig from domain config
        config = StorageConfig.from_mount_path(mount_path)
        self._storage = LocalFileStorage(config)
        logger.info(f"Storage initialized at: {mount_path}")

    async def _initialize_artifact_service(self) -> None:
        """Initialize artifact service using existing storage."""
        if not self._storage:
            raise RuntimeError("Storage must be initialized before ArtifactService")

        self._artifact_service = ArtifactService(self._storage)
        await self._artifact_service.initialize()
        logger.info("✅ ArtifactService initialized")

    async def _initialize_knowledge_store(self) -> None:
        """Initialize knowledge store service."""
        self._knowledge_store = KnowledgeStoreService(self._artifact_service)
        logger.info("✅ KnowledgeStoreService initialized")

    async def _initialize_context_builder(self) -> None:
        """Initialize context builder service with knowledge store."""
        self._context_builder = ContextBuilderService(knowledge_store=self._knowledge_store)
        logger.info("✅ ContextBuilderService initialized")

    async def _initialize_toolkit_manager(self) -> None:
        """Initialize toolkit manager."""
        self._toolkit_manager = AgnoToolkitManager()
        await self._toolkit_manager.initialize()

    async def _load_tools_registry(self) -> None:
        """Load and register tools from configuration."""
        logger.info("Loading tools registry from configuration")

        try:
            # Get tools from Hydra config
            tools_config = getattr(self.config, "tools", None)
            if not tools_config:
                logger.warning("No tools configuration found in config")
                return

            # Register each tool category
            tool_count = 0
            for category_name, tools in tools_config.items():
                if category_name == "presets":
                    # Skip presets, they're combinations
                    continue

                logger.debug(f"Loading {category_name} tools")

                if isinstance(tools, dict):
                    for tool_name, tool_config in tools.items():
                        try:
                            # Register the tool config with toolkit manager
                            self._toolkit_manager.register_tool_config(tool_config)
                            tool_count += 1
                            logger.debug(
                                f"Registered tool: {tool_config.name} (type: {tool_config.type})"
                            )
                        except Exception as tool_error:
                            logger.error(f"Failed to register tool {tool_name}: {tool_error}")

            logger.info(f"✅ Tools registry loaded: {tool_count} tools registered")

        except Exception as e:
            logger.error(f"Failed to load tools registry: {e}")
            # Don't fail initialization if tools loading fails
            logger.warning("Continuing without tools registry")

    async def _initialize_agent_factory(self) -> None:
        """Initialize agent factory with configuration."""
        self._agent_factory = AgentFactory(self.config)
        logger.info("✅ AgentFactory initialized")

    async def _initialize_agent_runtime_service(self) -> None:
        """Initialize agent runtime service with dependencies."""
        # Create agent runtime service with simplified dependencies
        self._agent_runtime_service = AgentRuntimeService(
            agent_factory=self._agent_factory, event_publisher=self._event_publisher
        )

        await self._agent_runtime_service.initialize()

    async def _initialize_recovery_manager(self) -> None:
        """Initialize recovery manager with circuit breaker and execution config."""
        self._recovery_manager = RecoveryManager(execution_config=self.config.execution)
        logger.info("✅ RecoveryManager initialized with execution config")

    async def _initialize_persistence_repositories(self) -> None:
        """Initialize persistence repositories if PostgreSQL is available."""
        # Only initialize persistence repositories if connection manager is available
        if self._connection_manager and self._connection_manager.is_healthy():
            try:
                session_factory = self._connection_manager.get_session_factory()

                # Initialize checkpoint repository
                self._checkpoint_repository = SQLAlchemyCheckpointRepository(session_factory)
                logger.info("✅ CheckpointRepository initialized")

                # Initialize recovery repository
                self._recovery_repository = SQLAlchemyRecoveryRepository(session_factory)
                logger.info("✅ RecoveryRepository initialized")

                # Initialize execution history repository
                self._execution_history_repository = SQLAlchemyExecutionHistoryRepository(
                    session_factory
                )
                logger.info("✅ ExecutionHistoryRepository initialized")

                logger.info("✅ All persistence repositories initialized")

            except Exception as e:
                logger.error(f"Failed to initialize persistence repositories: {e}")
                logger.info("Continuing without persistence repositories")
                # Set repositories to None to indicate they're unavailable
                self._checkpoint_repository = None
                self._recovery_repository = None
                self._execution_history_repository = None
        else:
            logger.info("PostgreSQL not available - persistence repositories disabled")
            self._checkpoint_repository = None
            self._recovery_repository = None
            self._execution_history_repository = None

    async def _initialize_checkpoint_service(self) -> None:
        """Initialize checkpoint service if persistence repositories are available."""
        # Only initialize checkpoint service if both repositories are available
        if self._checkpoint_repository and self._recovery_repository:
            try:
                self._checkpoint_service = CheckpointService(
                    checkpoint_repository=self._checkpoint_repository,
                    recovery_repository=self._recovery_repository,
                )
                logger.info("✅ CheckpointService initialized")
            except Exception as e:
                logger.error(f"Failed to initialize CheckpointService: {e}")
                logger.info("Continuing without CheckpointService")
                self._checkpoint_service = None
        else:
            logger.info("Persistence repositories not available - CheckpointService disabled")
            self._checkpoint_service = None

    async def _initialize_hitl_service(self) -> None:
        """Initialize HITL service for human interaction."""
        # HITL service can be disabled by setting enabled=False
        hitl_enabled = getattr(self.config.execution, "hitl_enabled", False)
        if hitl_enabled:
            self._hitl_service = HITLService(
                enabled=True,
                default_timeout_seconds=getattr(self.config.execution, "hitl_timeout_seconds", 300),
            )
            logger.info("✅ HITLService initialized (enabled)")
        else:
            self._hitl_service = None
            logger.info("✅ HITLService initialized (disabled)")

    async def _initialize_deadlock_detector(self) -> None:
        """Initialize deadlock detector for execution monitoring."""
        if not self._task_graph:
            raise RuntimeError("Task graph must be initialized before DeadlockDetector")

        self._deadlock_detector = DeadlockDetector(
            graph=self._task_graph,
            stall_threshold_seconds=getattr(self.config.execution, "deadlock_timeout_seconds", 600),
        )
        logger.info("✅ DeadlockDetector initialized")

    async def _initialize_graph_state_manager(self) -> None:
        """Initialize graph state manager for atomic state transitions."""
        self._graph_state_manager = GraphStateManager(self._task_graph, self._event_publisher)
        logger.info("✅ GraphStateManager initialized")

    async def _initialize_parallel_execution_engine(self) -> None:
        """Initialize parallel execution engine for concurrent task processing."""
        self._parallel_execution_engine = ParallelExecutionEngine(
            state_manager=self._graph_state_manager,
            max_concurrent_tasks=self.config.execution.max_concurrent_tasks,
        )
        logger.info("✅ ParallelExecutionEngine initialized")

    async def _initialize_agent_service_registry(self) -> None:
        """Initialize agent service registry for agent service management."""
        self._agent_service_registry = AgentServiceRegistry(
            agent_runtime_service=self._agent_runtime_service,
            recovery_manager=self._recovery_manager,
            hitl_service=self._hitl_service,
            checkpoint_repository=self._checkpoint_repository,
            recovery_repository=self._recovery_repository,
            execution_history_repository=self._execution_history_repository,
        )
        logger.info("✅ AgentServiceRegistry initialized")

    async def _initialize_execution_orchestrator(self) -> None:
        """Initialize execution orchestrator for main coordination."""
        self._execution_orchestrator = ExecutionOrchestrator(
            graph_state_manager=self._graph_state_manager,
            parallel_engine=self._parallel_execution_engine,
            agent_service_registry=self._agent_service_registry,
            context_builder=self._context_builder,
            recovery_manager=self._recovery_manager,
            event_publisher=self._event_publisher,
            execution_config=self.config.execution,
            deadlock_detector=self._deadlock_detector,
            knowledge_store=self._knowledge_store,
        )
        logger.info("✅ ExecutionOrchestrator initialized")

    async def _load_profile(self, profile_name: str) -> None:
        """Load and validate agent profile configuration from injected config."""
        # Use the profile from ROMAConfig
        profile_config = self.config.profile
        logger.info(f"Profile {profile_name} loaded successfully with config: {profile_config}")

    async def execute_task(self, task: str, **options) -> dict[str, Any]:
        """
        Execute any type of task using the new orchestration architecture.

        Can handle coding, analysis, research, creative work, data processing,
        or any other task through intelligent agent decomposition.

        Args:
            task: Task description to execute
            **options: Additional execution options

        Returns:
            Execution result dictionary compatible with existing API
        """
        if not self._initialized:
            raise RuntimeError("SystemManager not initialized")

        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        execution_orchestrator = None  # Initialize to track creation state
        execution_context = None  # Initialize to track creation state

        logger.info(f"Starting task execution [{execution_id}]: {task[:50]}...")

        try:
            # Create root task node
            root_task = TaskNode(
                task_id=f"root_{execution_id}",
                goal=task,
                task_type=TaskType.THINK,  # Will be determined by atomizer
                status=TaskStatus.PENDING,
            )

            # Create execution context with isolated resources
            execution_context = ExecutionContext(
                execution_id=execution_id,
                base_storage_config=StorageConfig.from_mount_path(self.config.storage.mount_path),
            )
            execution_context.set_config(self.config)
            await execution_context.initialize()

            # Store execution context
            self._active_contexts[execution_id] = execution_context

            # Create execution-specific orchestrator with isolated resources
            execution_orchestrator = ExecutionOrchestrator(
                graph_state_manager=execution_context.graph_state_manager,
                parallel_engine=ParallelExecutionEngine(
                    state_manager=execution_context.graph_state_manager,
                    max_concurrent_tasks=self.config.execution.max_concurrent_tasks,
                    recovery_manager=self._recovery_manager,
                ),
                agent_service_registry=self._agent_service_registry,
                context_builder=execution_context.context_builder,
                recovery_manager=self._recovery_manager,
                event_publisher=execution_context.event_publisher,
                execution_config=self.config.execution,
                deadlock_detector=self._deadlock_detector,
                knowledge_store=execution_context.knowledge_store,
            )

            # Track execution
            execution_info = {
                "execution_id": execution_id,
                "root_task": root_task,
                "start_time": start_time,
                "status": TaskStatus.EXECUTING,
                "task": task,
                "options": options,
                "graph": execution_context.task_graph,
                "orchestrator": execution_orchestrator,
            }
            self._active_executions[execution_id] = execution_info

            # Execute through execution-specific orchestrator
            logger.info(
                f"Delegating execution to isolated ExecutionOrchestrator for task: {task[:100]}..."
            )

            execution_result = await execution_orchestrator.execute(
                root_task=root_task, overall_objective=task, execution_id=execution_id
            )

            # Store artifacts from final result if available
            artifact_refs = []
            if execution_result.final_result:
                artifact_refs = await execution_context.artifact_service.store_envelope_artifacts(
                    execution_result.final_result
                )

            # Update execution tracking
            execution_info["status"] = (
                TaskStatus.COMPLETED if execution_result.success else TaskStatus.FAILED
            )
            execution_info["end_time"] = datetime.now()
            execution_info["execution_result"] = execution_result
            execution_info["artifact_refs"] = artifact_refs

            # Cleanup execution-specific components to prevent memory leaks
            await execution_orchestrator.cleanup_execution(execution_id)

            # Return standardized result
            return self._build_execution_result(
                execution_id=execution_id,
                task=task,
                execution_result=execution_result,
                artifact_refs=artifact_refs,
                options=options,
                execution_orchestrator=execution_orchestrator,
            )

        except Exception as e:
            # Update execution tracking on failure
            if execution_id in self._active_executions:
                execution_info = self._active_executions[execution_id]
                execution_info["status"] = TaskStatus.FAILED
                execution_info["end_time"] = datetime.now()
                execution_info["error"] = str(e)

            logger.error(f"Task execution failed [{execution_id}]: {e}")

            # Cleanup execution-specific components even on failure (only if created)
            if execution_orchestrator is not None:
                try:
                    await execution_orchestrator.cleanup_execution(execution_id)
                except Exception as cleanup_error:
                    logger.error(
                        f"Error during orchestrator cleanup for {execution_id}: {cleanup_error}"
                    )

            # Return error response
            return self._build_error_result(
                execution_id=execution_id,
                task=task,
                error=e,
                start_time=start_time,
                options=options,
            )

        finally:
            # Clean up execution tracking and isolated components
            try:
                if execution_id in self._active_executions:
                    try:
                        self._active_executions[execution_id]["completed_at"] = datetime.now()

                        # Clean up execution-specific components for garbage collection
                        execution_info = self._active_executions[execution_id]
                        execution_info.pop("graph", None)
                        execution_info.pop("orchestrator", None)
                    except Exception as tracking_error:
                        logger.error(
                            f"Error updating execution tracking for {execution_id}: {tracking_error}"
                        )
                    finally:
                        # Always remove from active executions to prevent memory leak
                        with contextlib.suppress(KeyError):
                            # Already removed, ignore
                            del self._active_executions[execution_id]

                # Clean up execution context
                if execution_id in self._active_contexts:
                    try:
                        await self._active_contexts[execution_id].cleanup()
                    except Exception as cleanup_error:
                        logger.error(
                            f"Error cleaning up execution context {execution_id}: {cleanup_error}"
                        )
                    finally:
                        # Always remove from active contexts to prevent memory leak
                        with contextlib.suppress(KeyError):
                            # Already removed, ignore
                            del self._active_contexts[execution_id]
            except Exception as final_cleanup_error:
                # Log any errors in final cleanup but don't re-raise
                logger.error(
                    f"Critical error during final cleanup for {execution_id}: {final_cleanup_error}"
                )

    def get_system_info(self) -> dict[str, Any]:
        """Get comprehensive system information with robust error handling."""
        if not self._initialized:
            return {"status": "not_initialized"}

        # Safely get runtime metrics
        runtime_metrics = {}
        try:
            if self._agent_runtime_service:
                runtime_metrics = self._agent_runtime_service.get_runtime_metrics()
        except Exception as e:
            logger.warning(f"Failed to get runtime metrics: {e}")
            runtime_metrics = {"error": "metrics_unavailable"}

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
                "toolkit_manager": self._toolkit_manager is not None,
            },
            "runtime_metrics": runtime_metrics,
        }

    def validate_configuration(self) -> dict[str, Any]:
        """Validate system configuration."""
        errors = []
        warnings = []

        # Check if profile is configured
        if not self.config.profile:
            warnings.append("Profile not configured")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def get_current_profile(self) -> str | None:
        """Get current active profile."""
        return self._current_profile

    def get_available_profiles(self) -> list[str]:
        """Get list of available profiles."""
        # Return the current profile name since ROMAConfig only contains one profile
        return [self.config.profile.name] if self.config.profile.name else []

    async def switch_profile(self, profile_name: str) -> dict[str, Any]:
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
                "system_info": self.get_system_info(),
            }

        except Exception as e:
            logger.error(f"Failed to switch profile: {e}")
            return {"success": False, "error": str(e), "profile": profile_name}

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
                if hasattr(self._event_store, "close"):
                    await self._event_store.close()
                else:
                    await self._event_store.clear()
            except Exception as e:
                logger.error(f"Error cleaning up event store: {e}")

        if self._connection_manager:
            try:
                await self._connection_manager.close()
            except Exception as e:
                logger.error(f"Error cleaning up connection manager: {e}")

        if self._recovery_manager:
            try:
                self._recovery_manager.reset_circuit_breaker()
                self._recovery_manager.clear_permanent_failures()
            except Exception as e:
                logger.error(f"Error cleaning up recovery manager: {e}")

        self._initialized = False

    # Property accessors for commonly used components
    @property
    def event_store(self):
        """Get the event store component."""
        return self._event_store

    @property
    def task_graph(self):
        """Get the task graph component."""
        return self._task_graph

    @property
    def agent_runtime_service(self):
        """Get the agent runtime service component."""
        return self._agent_runtime_service

    @property
    def execution_orchestrator(self):
        """Get the execution orchestrator component."""
        return self._execution_orchestrator

    @property
    def storage(self):
        """Get the storage component."""
        return self._storage

    @property
    def context_builder(self):
        """Get the context builder component."""
        return self._context_builder

    @property
    def knowledge_store(self):
        """Get the knowledge store component."""
        return self._knowledge_store

    def get_component_status(self) -> dict[str, bool]:
        """Get initialization status of all components."""
        return {
            "event_store": self._event_store is not None,
            "task_graph": self._task_graph is not None,
            "storage": self._storage is not None,
            "agent_runtime_service": self._agent_runtime_service is not None,
            "execution_orchestrator": self._execution_orchestrator is not None,
            "context_builder": self._context_builder is not None,
            "knowledge_store": self._knowledge_store is not None,
            "artifact_service": self._artifact_service is not None,
            "toolkit_manager": self._toolkit_manager is not None,
            "agent_factory": self._agent_factory is not None,
            "recovery_manager": self._recovery_manager is not None,
            "hitl_service": self._hitl_service is not None,
            "graph_state_manager": self._graph_state_manager is not None,
            "parallel_execution_engine": self._parallel_execution_engine is not None,
            "agent_service_registry": self._agent_service_registry is not None,
        }

    def _build_execution_result(
        self,
        execution_id: str,
        task: str,
        execution_result: Any,
        artifact_refs: list[str],
        options: dict[str, Any],
        execution_orchestrator: Any,
    ) -> dict[str, Any]:
        """Build standardized execution result response."""
        return {
            "execution_id": execution_id,
            "task": task,
            "status": "completed" if execution_result.success else "failed",
            "result": (
                execution_result.final_result.extract_primary_output()
                if execution_result.final_result
                else None
            ),
            "execution_time": execution_result.execution_time_seconds,
            "node_count": execution_result.total_nodes,
            "completed_nodes": execution_result.completed_nodes,
            "failed_nodes": execution_result.failed_nodes,
            "iterations": execution_result.iterations,
            "framework": self._get_safe_framework_name(),
            "artifacts": len(artifact_refs),
            "error_details": execution_result.error_details
            if execution_result.has_errors
            else None,
            "orchestration_metrics": execution_orchestrator.get_orchestration_metrics(execution_id),
            # Legacy compatibility fields
            "final_output": (
                execution_result.final_result.extract_primary_output()
                if execution_result.final_result
                else None
            ),
            "hitl_enabled": options.get("enable_hitl", False),
            "framework_result": execution_result.to_dict(),
        }

    def _build_error_result(
        self,
        execution_id: str,
        task: str,
        error: Exception,
        start_time: datetime,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Build standardized error response."""
        return {
            "execution_id": execution_id,
            "task": task,
            "status": "failed",
            "result": None,
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "node_count": 0,
            "error": str(error),
            "framework": self._get_safe_framework_name(),
            "artifacts": 0,
            "final_output": None,
            "hitl_enabled": options.get("enable_hitl", False),
            "framework_result": None,
        }

    def _get_safe_framework_name(self) -> str:
        """Safely get framework name with fallback."""
        try:
            if self._agent_runtime_service:
                return self._agent_runtime_service.get_framework_name()
        except Exception as e:
            logger.warning(f"Failed to get framework name: {e}")
        return "unknown"
