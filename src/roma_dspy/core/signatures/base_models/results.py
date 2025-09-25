from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from ....types import NodeType
from .subtask import SubTask

class AtomizerResponse(BaseModel):
    """
    Atomizer decision result.

    Determines whether a task should be executed atomically or decomposed into subtasks.
    """
    is_atomic: bool = Field(..., description="True if task can be executed directly")
    node_type: NodeType = Field(..., description="Type of node to process (PLAN or EXECUTE)")


class PlannerResult(BaseModel):
    """
    Planner decomposition result.

    Contains the breakdown of a complex task into executable subtasks.
    """

    subtasks: List[SubTask] = Field(..., min_items=1, description="List of planned subtasks")
    dependencies_graph: Optional[Dict[str, List[str]]] = Field(default=None, description="Task dependency mapping")

    
class AggregatorResult(BaseModel):
    """
    Aggregator synthesis result.

    Contains the synthesis of multiple subtask results into a cohesive output.
    """

    synthesized_result: str = Field(..., min_length=1, description="Final synthesized output")

class ExecutorResult(BaseModel):
    """
    Executor execution result.

    Contains the output of atomic task execution.
    """

    output: Any = Field(..., description="Primary execution result")
    sources: Optional[List[str]] = Field(default_factory=list, description="Information sources used")
