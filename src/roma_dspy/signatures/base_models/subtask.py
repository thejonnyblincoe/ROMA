from pydantic import BaseModel, Field
from typing import List
from src.roma_dspy.types.task_type import TaskType


class SubTask(BaseModel):
    """
    Individual subtask in a decomposition plan.
    """

    goal: str = Field(..., min_length=1, description="Precise subtask objective")
    task_type: TaskType = Field(..., description="Type of subtask")
    dependencies: List[str] = Field(default_factory=list, description="List of subtask IDs this depends on")