"""
API Request Schemas.

Defines request models for the ROMA API endpoints.
Presentation layer - only for API input validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class ExecuteRequest(BaseModel):
    """Request model for execution endpoints."""
    
    goal: str = Field(
        ..., 
        description="Research goal to execute",
        min_length=1,
        max_length=2000,
        example="Research the latest developments in quantum computing"
    )
    profile: Optional[str] = Field(
        "general_agent", 
        description="Agent profile to use",
        example="deep_research_agent"
    )
    enable_hitl: Optional[bool] = Field(
        False, 
        description="Enable Human-in-the-Loop interactions"
    )
    max_steps: Optional[int] = Field(
        50, 
        description="Maximum execution steps",
        ge=1,
        le=1000
    )
    options: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional execution options"
    )


class StreamRequest(BaseModel):
    """Request model for streaming endpoints."""
    
    goal: str = Field(
        ...,
        description="Research goal to execute with streaming",
        min_length=1,
        max_length=2000
    )
    profile: Optional[str] = Field(
        "general_agent",
        description="Agent profile to use"
    )
    enable_hitl: Optional[bool] = Field(
        False,
        description="Enable Human-in-the-Loop interactions"
    )


class ProfileRequest(BaseModel):
    """Request model for profile operations."""
    
    profile_name: str = Field(
        ...,
        description="Name of the profile",
        min_length=1,
        max_length=100
    )


class ValidationRequest(BaseModel):
    """Request model for configuration validation."""
    
    config_path: Optional[str] = Field(
        None,
        description="Path to configuration file to validate"
    )
    profile: Optional[str] = Field(
        None,
        description="Specific profile to validate"
    )