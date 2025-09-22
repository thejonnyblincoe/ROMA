"""
HITL (Human-in-the-Loop) Value Objects for ROMA v2.0.

Contains value objects for human interaction requests and responses.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class HITLRequestType(str, Enum):
    """Types of HITL requests."""

    REPLANNING_APPROVAL = "replanning_approval"
    TASK_GUIDANCE = "task_guidance"
    EXECUTION_PAUSE = "execution_pause"
    DECISION_POINT = "decision_point"


class HITLRequestStatus(str, Enum):
    """Status of HITL requests."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMEOUT = "timeout"


class HITLRequest(BaseModel):
    """Value object representing a HITL request."""

    request_id: str
    request_type: HITLRequestType
    task_id: str
    title: str = ""
    description: str = ""
    context_data: Dict[str, Any] = Field(default_factory=dict)
    suggested_actions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_seconds: Optional[int] = None
    status: HITLRequestStatus = HITLRequestStatus.PENDING
    request_data: Dict[str, Any] = Field(default_factory=dict)


class HITLResponse(BaseModel):
    """Value object representing a HITL response."""

    request_id: str
    status: HITLRequestStatus
    human_feedback: Optional[str] = None
    modified_context: Optional[Dict[str, Any]] = None
    selected_action: Optional[str] = None
    response_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_notes: Optional[str] = None
    response_data: Dict[str, Any] = Field(default_factory=dict)