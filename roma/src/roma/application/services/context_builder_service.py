"""
Context Builder Service - ROMA v2.0 Multimodal Context Assembly.

Assembles complete context for agent execution by combining text strings,
file artifacts, and task metadata into structured context objects.
"""

import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.entities.artifacts.base_artifact import BaseArtifact
from src.roma.domain.entities.artifacts.file_artifact import FileArtifact
from src.roma.domain.value_objects.media_type import MediaType
from src.roma.domain.value_objects.task_type import TaskType

logger = logging.getLogger(__name__)


class ContextItem(BaseModel):
    """Single context item with content and metadata."""
    
    model_config = ConfigDict(frozen=True)
    
    item_id: str = Field(default_factory=lambda: str(uuid4()))
    content_type: MediaType
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    
    @classmethod
    def from_text(
        cls, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None, 
        priority: int = 0
    ) -> "ContextItem":
        """Create context item from text content."""
        return cls(
            content_type=MediaType.TEXT,
            content=content,
            metadata=metadata or {},
            priority=priority
        )
    
    @classmethod
    def from_artifact(
        cls, 
        artifact: BaseArtifact, 
        priority: int = 0
    ) -> "ContextItem":
        """Create context item from artifact."""
        return cls(
            content_type=artifact.media_type,
            content=artifact,
            metadata=artifact.metadata,
            priority=priority
        )


class TaskContext(BaseModel):
    """Complete context assembled for agent execution."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, from_attributes=True)
    
    # Core task information
    task: Any  # TaskNode - using Any to avoid Pydantic validation issues
    overall_objective: str
    
    # Context items (ordered by priority)
    context_items: List[ContextItem] = Field(default_factory=list)
    
    # System metadata
    execution_metadata: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    def get_text_content(self) -> List[str]:
        """Extract all text content from context."""
        text_items = [
            item.content for item in self.context_items 
            if item.content_type == MediaType.TEXT
        ]
        return text_items
    
    def get_file_artifacts(self) -> List[FileArtifact]:
        """Extract all file artifacts from context."""
        file_items = []
        for item in self.context_items:
            if item.content_type == MediaType.FILE and isinstance(item.content, FileArtifact):
                file_items.append(item.content)
        return file_items
    
    def get_by_media_type(self, media_type: MediaType) -> List[ContextItem]:
        """Get context items by media type."""
        return [item for item in self.context_items if item.content_type == media_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task": self.task.to_dict(),
            "overall_objective": self.overall_objective,
            "context_items": [
                {
                    "item_id": item.item_id,
                    "content_type": item.content_type.value,
                    "content": (
                        item.content.to_dict() if isinstance(item.content, BaseArtifact) 
                        else str(item.content)
                    ),
                    "metadata": item.metadata,
                    "priority": item.priority
                }
                for item in self.context_items
            ],
            "execution_metadata": self.execution_metadata,
            "constraints": self.constraints,
            "user_preferences": self.user_preferences,
            "text_count": len(self.get_text_content()),
            "file_count": len(self.get_file_artifacts()),
        }


class ContextBuilderService:
    """
    Service for assembling multimodal context for agent execution.
    
    Combines text strings, file artifacts, and metadata into structured
    TaskContext objects that agents can process.
    """
    
    def __init__(self):
        self.logger = logger
    
    async def build_context(
        self,
        task: TaskNode,
        overall_objective: str,
        text_content: Optional[List[str]] = None,
        file_artifacts: Optional[List[FileArtifact]] = None,
        parent_results: Optional[List[str]] = None,
        sibling_results: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        execution_metadata: Optional[Dict[str, Any]] = None
    ) -> TaskContext:
        """
        Build complete context for task execution.
        
        Args:
            task: The task node being executed
            overall_objective: Root goal/objective
            text_content: Additional text content for context
            file_artifacts: File artifacts to include
            parent_results: Results from parent tasks
            sibling_results: Results from sibling tasks
            constraints: Any execution constraints
            user_preferences: User preferences for execution
            execution_metadata: System execution metadata
            
        Returns:
            TaskContext: Complete assembled context
        """
        context_items = []
        
        # Add goal as highest priority text content
        context_items.append(
            ContextItem.from_text(
                content=f"Current Task: {task.goal}",
                metadata={"source": "task_goal", "task_id": task.task_id},
                priority=100
            )
        )
        
        # Add overall objective
        context_items.append(
            ContextItem.from_text(
                content=f"Overall Objective: {overall_objective}",
                metadata={"source": "overall_objective"},
                priority=95
            )
        )
        
        # Add parent results with high priority
        if parent_results:
            for i, result in enumerate(parent_results):
                context_items.append(
                    ContextItem.from_text(
                        content=result,
                        metadata={"source": "parent_result", "index": i},
                        priority=80
                    )
                )
        
        # Add sibling results with medium priority
        if sibling_results:
            for i, result in enumerate(sibling_results):
                context_items.append(
                    ContextItem.from_text(
                        content=result,
                        metadata={"source": "sibling_result", "index": i},
                        priority=60
                    )
                )
        
        # Add additional text content
        if text_content:
            for i, content in enumerate(text_content):
                context_items.append(
                    ContextItem.from_text(
                        content=content,
                        metadata={"source": "additional_text", "index": i},
                        priority=40
                    )
                )
        
        # Add file artifacts
        if file_artifacts:
            for artifact in file_artifacts:
                context_items.append(
                    ContextItem.from_artifact(
                        artifact=artifact,
                        priority=50
                    )
                )
        
        # Sort by priority (highest first)
        context_items.sort(key=lambda x: x.priority, reverse=True)
        
        # Build final context
        context = TaskContext(
            task=task,
            overall_objective=overall_objective,
            context_items=context_items,
            constraints=constraints or [],
            user_preferences=user_preferences or {},
            execution_metadata=execution_metadata or {}
        )
        
        self.logger.info(
            f"Built context for task {task.task_id}: "
            f"{len(context.get_text_content())} text items, "
            f"{len(context.get_file_artifacts())} file artifacts"
        )
        
        return context
    
    async def build_lineage_context(
        self,
        task: TaskNode,
        overall_objective: str,
        parent_chain: List[Dict[str, Any]],
        execution_metadata: Optional[Dict[str, Any]] = None
    ) -> TaskContext:
        """
        Build context with focus on task lineage and parent results.
        
        Args:
            task: Current task node
            overall_objective: Root goal
            parent_chain: Chain of parent task results
            execution_metadata: System metadata
            
        Returns:
            TaskContext: Context focused on lineage
        """
        parent_results = [
            f"Parent Task Result: {parent.get('result', 'No result')}"
            for parent in parent_chain
        ]
        
        return await self.build_context(
            task=task,
            overall_objective=overall_objective,
            parent_results=parent_results,
            execution_metadata=execution_metadata
        )
    
    async def build_rich_context(
        self,
        task: TaskNode,
        overall_objective: str,
        knowledge_store: Dict[str, Any],
        execution_metadata: Optional[Dict[str, Any]] = None
    ) -> TaskContext:
        """
        Build rich context with knowledge store integration.
        
        Args:
            task: Current task node
            overall_objective: Root goal
            knowledge_store: Accumulated knowledge from execution
            execution_metadata: System metadata
            
        Returns:
            TaskContext: Rich context with knowledge store
        """
        # Extract relevant information from knowledge store
        relevant_results = knowledge_store.get("relevant_results", [])
        constraints = knowledge_store.get("constraints", [])
        user_preferences = knowledge_store.get("user_preferences", {})
        
        # Convert knowledge store items to text content
        knowledge_text = []
        for key, value in knowledge_store.items():
            if key not in ["relevant_results", "constraints", "user_preferences"]:
                knowledge_text.append(f"{key}: {value}")
        
        return await self.build_context(
            task=task,
            overall_objective=overall_objective,
            text_content=knowledge_text,
            parent_results=[str(r) for r in relevant_results],
            constraints=constraints,
            user_preferences=user_preferences,
            execution_metadata=execution_metadata
        )
    
    def validate_context(self, context: TaskContext) -> bool:
        """
        Validate that context is properly structured.
        
        Args:
            context: Context to validate
            
        Returns:
            bool: True if context is valid
        """
        try:
            # Check required fields
            if not context.task or not context.overall_objective:
                return False
            
            # Check context items structure
            for item in context.context_items:
                if not item.item_id or not item.content_type:
                    return False
            
            # Validate file artifacts are accessible
            for artifact in context.get_file_artifacts():
                if not artifact.is_accessible():
                    self.logger.warning(f"File artifact not accessible: {artifact.name}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Context validation failed: {e}")
            return False