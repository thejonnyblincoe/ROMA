import dspy
from typing import Optional, Dict, List, Any
from src.roma_dspy.signatures.base_models.subtask import SubTask
from src.roma_dspy.signatures.base_models.results import AtomizerResponse, PlannerResult, ExecutorResult
from src.roma_dspy.types.node_type import NodeType


class AtomizerSignature(dspy.Signature):
    """Signature for task atomization."""
    goal: str = dspy.InputField(description="Task to atomize")
    is_atomic: bool = dspy.OutputField(description="True if task can be executed directly")
    node_type: NodeType = dspy.OutputField(description="Type of node to process (PLAN or EXECUTE)")


class PlannerSignature(dspy.Signature):
    """
    Planner decomposition result.

    Contains the breakdown of a complex task into executable subtasks.
    """
    goal: str = dspy.InputField(description="Task that needs to be decomposed into subtasks through planner")
    subtasks: List[SubTask] = dspy.OutputField(description="List of generated subtasks from planner")
    #TODO: This should be revised, it shouldn't go from str to List[str], perhaps it should be int to List[int]
    dependencies_graph: Optional[Dict[str, List[str]]] = dspy.OutputField(default=None, description="Task dependency mapping")

class ExecutorSignature(dspy.Signature):
    """
    Executor execution result.

    Contains the output of atomic task execution.
    """
    goal: str = dspy.InputField(description="Task that needs to be executed")
    output: str = dspy.OutputField(description="Execution result")
    sources: Optional[List[str]] = dspy.OutputField(default_factory=list, description="Information sources used")

class AggregatorResult(dspy.Signature):
    """
    Aggregator synthesis result.

    Contains the synthesis of multiple subtask results into a cohesive output.
    """
    original_goal: str = dspy.InputField(description="Original goal of the task")
    subtasks_results: List[SubTask] = dspy.InputField(description="List of subtask results to synthesize")
    synthesized_result: str = dspy.OutputField(description="Final synthesized output")