"""
API Response Schemas.

Defines response models for the ROMA API endpoints.
Presentation layer - only for API output formatting.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class ExecuteResponse(BaseModel):
    """Response model for execution endpoints."""
    
    execution_id: str = Field(
        ...,
        description="Unique identifier for the execution",
        example="exec-001-abc123"
    )
    goal: str = Field(
        ...,
        description="The original research goal",
        example="Research quantum computing applications"
    )
    status: str = Field(
        ...,
        description="Execution status",
        example="completed"
    )
    final_output: str = Field(
        ...,
        description="Final execution result",
        example="Quantum computing has applications in..."
    )
    execution_time: float = Field(
        ...,
        description="Total execution time in seconds",
        example=12.5
    )
    node_count: int = Field(
        ...,
        description="Number of nodes processed",
        example=5
    )
    hitl_enabled: bool = Field(
        ...,
        description="Whether Human-in-the-Loop was enabled"
    )
    framework_result: Dict[str, Any] = Field(
        ...,
        description="Internal framework result data"
    )


class StreamEvent(BaseModel):
    """Response model for streaming events."""
    
    event: str = Field(
        ...,
        description="Event type",
        example="progress"
    )
    goal: Optional[str] = Field(
        None,
        description="Associated goal"
    )
    message: Optional[str] = Field(
        None,
        description="Event message",
        example="Processing node 3 of 5"
    )
    progress: Optional[float] = Field(
        None,
        description="Progress percentage (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    result: Optional[str] = Field(
        None,
        description="Partial or final result"
    )
    timestamp: Optional[str] = Field(
        None,
        description="Event timestamp",
        example="2024-01-01T12:00:00Z"
    )


class SystemInfo(BaseModel):
    """Response model for system information."""
    
    name: str = Field(
        ...,
        description="System name",
        example="ROMA"
    )
    version: str = Field(
        ...,
        description="System version",
        example="2.0.0"
    )
    description: str = Field(
        ...,
        description="System description"
    )
    environment: str = Field(
        ...,
        description="Environment (development, production, etc.)",
        example="development"
    )
    status: str = Field(
        ...,
        description="System status",
        example="running"
    )
    profile: str = Field(
        ...,
        description="Current active profile",
        example="general_agent"
    )
    cache_enabled: bool = Field(
        ...,
        description="Whether caching is enabled"
    )


class ValidationResponse(BaseModel):
    """Response model for configuration validation."""
    
    valid: bool = Field(
        ...,
        description="Whether configuration is valid"
    )
    issues: Dict[str, Any] = Field(
        ...,
        description="Validation issues if any"
    )
    profile: str = Field(
        ...,
        description="Profile that was validated"
    )
    status: str = Field(
        ...,
        description="Validation status",
        example="completed"
    )


class ProfileInfo(BaseModel):
    """Response model for profile information."""
    
    profile_name: str = Field(
        ...,
        description="Profile name",
        example="deep_research_agent"
    )
    description: str = Field(
        ...,
        description="Profile description"
    )
    version: str = Field(
        ...,
        description="Profile version",
        example="2.0.0"
    )
    enabled: bool = Field(
        ...,
        description="Whether profile is enabled"
    )
    status: str = Field(
        ...,
        description="Profile status",
        example="active"
    )
    completeness: Dict[str, Any] = Field(
        ...,
        description="Profile completeness validation results"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(
        "healthy",
        description="Health status"
    )
    service: str = Field(
        "ROMA v2 API",
        description="Service name"
    )
    version: str = Field(
        "2.0.0",
        description="Service version"
    )
    framework: str = Field(
        "operational",
        description="Framework status"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Health check timestamp"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(
        ...,
        description="Error type",
        example="ValidationError"
    )
    message: str = Field(
        ...,
        description="Error message",
        example="Invalid goal format"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Error timestamp"
    )


class SimpleResponse(BaseModel):
    """Simple response model for basic endpoints."""
    
    result: str = Field(
        ...,
        description="Result content"
    )
    status: str = Field(
        ...,
        description="Operation status",
        example="completed"
    )


class StatusResponse(BaseModel):
    """Response model for status endpoints."""
    
    name: str = Field(
        "ROMA",
        description="System name"
    )
    version: str = Field(
        "2.0.0", 
        description="System version"
    )
    status: str = Field(
        "operational",
        description="System status"
    )
    api_version: str = Field(
        "v1_compatible",
        description="API version"
    )
    available_profiles: List[str] = Field(
        ...,
        description="List of available profiles"
    )