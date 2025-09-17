"""
Tool Configuration Value Object.

Defines tool/toolkit configuration as a domain value object.
Single source of truth for tool configuration across the system.
"""

from pydantic.dataclasses import dataclass
from pydantic import Field, field_validator
from dataclasses import field
from typing import Dict, Any, Optional, List


@dataclass(frozen=True)
class ToolConfig:
    """Tool configuration value object."""

    name: str
    type: str  # web_search, code_execution, data_api, storage, etc.
    enabled: bool = True
    implementation: Optional[str] = None  # Custom implementation class path
    config: Dict[str, Any] = field(default_factory=dict)
    include_tools: List[str] = field(default_factory=list)  # Tools to include from toolkit
    exclude_tools: List[str] = field(default_factory=list)  # Tools to exclude from toolkit

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Tool name cannot be empty")
        return v.strip()

    @field_validator("type")
    @classmethod
    def validate_tool_type(cls, v: str) -> str:
        """Validate tool type string."""
        valid_types = [
            "web_search", "code_execution", "reasoning", "image_generation",
            "visualization", "data_api", "storage", "knowledge", "utility"
        ]
        if v not in valid_types:
            raise ValueError(f"tool type must be one of {valid_types}, got: {v}")
        return v

    @field_validator("include_tools")
    @classmethod
    def validate_include_tools(cls, v: List[str]) -> List[str]:
        """Validate include tools list."""
        if not isinstance(v, list):
            raise ValueError("include_tools must be a list")
        return [tool.strip() for tool in v if tool.strip()]

    @field_validator("exclude_tools")
    @classmethod
    def validate_exclude_tools(cls, v: List[str]) -> List[str]:
        """Validate exclude tools list."""
        if not isinstance(v, list):
            raise ValueError("exclude_tools must be a list")
        return [tool.strip() for tool in v if tool.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "enabled": self.enabled,
            "implementation": self.implementation,
            "config": dict(self.config),
            "include_tools": list(self.include_tools),
            "exclude_tools": list(self.exclude_tools),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type=data["type"],
            enabled=data.get("enabled", True),
            implementation=data.get("implementation"),
            config=data.get("config", {}),
            include_tools=data.get("include_tools", []),
            exclude_tools=data.get("exclude_tools", []),
        )