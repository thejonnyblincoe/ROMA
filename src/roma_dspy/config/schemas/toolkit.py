"""Toolkit configuration schemas for ROMA-DSPy."""

from pydantic.dataclasses import dataclass
from pydantic import field_validator, model_validator
from typing import List, Dict, Any, Optional


@dataclass
class ToolkitConfig:
    """Configuration for a single toolkit."""

    class_name: str                                    # e.g., "FileToolkit", "CalculatorToolkit"
    enabled: bool = True                               # Whether this toolkit is enabled
    include_tools: Optional[List[str]] = None          # Specific tools to include (None = all available)
    exclude_tools: Optional[List[str]] = None          # Tools to exclude from available tools
    toolkit_config: Optional[Dict[str, Any]] = None    # Toolkit-specific configuration parameters

    def __post_init__(self):
        """Initialize defaults after creation."""
        if self.include_tools is None:
            self.include_tools = []
        if self.exclude_tools is None:
            self.exclude_tools = []
        if self.toolkit_config is None:
            self.toolkit_config = {}

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        """Validate class name is not empty."""
        if not v or not v.strip():
            raise ValueError("Toolkit class name cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_tool_overlap(self):
        """Validate that tools are not both included and excluded."""
        if self.include_tools and self.exclude_tools:
            overlap = set(self.include_tools) & set(self.exclude_tools)
            if overlap:
                raise ValueError(f"Tools cannot be both included and excluded: {overlap}")
        return self