"""
Recovery Result Value Object.

Domain value object representing the result of a recovery operation.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.recovery_action import RecoveryAction


class RecoveryResult(BaseModel):
    """Result of recovery attempt."""

    model_config = ConfigDict(frozen=True)

    action: RecoveryAction = Field(..., description="Recovery action to take")
    reasoning: str = Field(..., description="Reasoning for the action")
    updated_node: TaskNode | None = Field(
        default=None, description="Updated task node if applicable"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")
