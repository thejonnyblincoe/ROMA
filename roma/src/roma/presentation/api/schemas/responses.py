"""
API Response Schemas.

Defines response models for the ROMA API endpoints.
Presentation layer - only for API output formatting.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ExecuteResponse(BaseModel):
    """Response model for execution endpoints."""

    execution_id: str = Field(description="Unique identifier for the execution", examples=["exec-001-abc123"])
    goal: str = Field(
        description="The original research goal",
        examples=["Research quantum computing applications"],
    )
    status: str = Field(description="Execution status", examples=["completed"])
    final_output: str = Field(
        description="Final execution result",
        examples=["Quantum computing has applications in..."],
    )
    execution_time: float = Field(description="Total execution time in seconds", examples=[12.5])
    node_count: int = Field(description="Number of nodes processed", examples=[5])
    hitl_enabled: bool = Field(description="Whether Human-in-the-Loop was enabled")
    framework_result: dict[str, Any] = Field(description="Internal framework result data")


class StreamEvent(BaseModel):
    """Response model for streaming events."""

    event: str = Field(description="Event type", examples=["progress"])
    goal: str | None = Field(default=None, description="Associated goal")
    message: str | None = Field(default=None, description="Event message", examples=["Processing node 3 of 5"])
    progress: float | None = Field(
        default=None, description="Progress percentage (0.0 to 1.0)", ge=0.0, le=1.0
    )
    result: str | None = Field(default=None, description="Partial or final result")
    timestamp: str | None = Field(
        default=None, description="Event timestamp", examples=["2024-01-01T12:00:00Z"]
    )


class SystemInfo(BaseModel):
    """Response model for system information."""

    name: str = Field(description="System name", examples=["ROMA"])
    version: str = Field(description="System version", examples=["2.0.0"])
    description: str = Field(description="System description")
    environment: str = Field(
        description="Environment (development, production, etc.)", examples=["development"]
    )
    status: str = Field(description="System status", examples=["running"])
    profile: str = Field(description="Current active profile", examples=["general_agent"])
    cache_enabled: bool = Field(description="Whether caching is enabled")


class ValidationResponse(BaseModel):
    """Response model for configuration validation."""

    valid: bool = Field(description="Whether configuration is valid")
    issues: dict[str, Any] = Field(description="Validation issues if any")
    profile: str = Field(description="Profile that was validated")
    status: str = Field(description="Validation status", examples=["completed"])


class ProfileInfo(BaseModel):
    """Response model for profile information."""

    profile_name: str = Field(description="Profile name", examples=["deep_research_agent"])
    description: str = Field(description="Profile description")
    version: str = Field(description="Profile version", examples=["2.0.0"])
    enabled: bool = Field(description="Whether profile is enabled")
    status: str = Field(description="Profile status", examples=["active"])
    completeness: dict[str, Any] = Field(description="Profile completeness validation results")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(default="healthy", description="Health status")
    service: str = Field(default="ROMA v2 API", description="Service name")
    version: str = Field(default="2.0.0", description="Service version")
    framework: str = Field(default="operational", description="Framework status")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(), description="Health check timestamp"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(description="Error type", examples=["ValidationError"])
    message: str = Field(description="Error message", examples=["Invalid goal format"])
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(), description="Error timestamp"
    )


class SimpleResponse(BaseModel):
    """Simple response model for basic endpoints."""

    result: str = Field(description="Result content")
    status: str = Field(description="Operation status", examples=["completed"])


class StatusResponse(BaseModel):
    """Response model for status endpoints."""

    name: str = Field(default="ROMA", description="System name")
    version: str = Field(default="2.0.0", description="System version")
    status: str = Field(default="operational", description="System status")
    api_version: str = Field(default="v1_compatible", description="API version")
    available_profiles: list[str] = Field(description="List of available profiles")
