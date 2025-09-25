"""
Toolkit Configuration Value Objects for ROMA v2.

Domain value objects representing toolkit configuration concepts.
Uses Pydantic dataclasses for Hydra structured config compatibility.
"""

from typing import Any

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass


@dataclass
class ToolkitConfig:
    """
    Pydantic dataclass for toolkit configuration.
    Used as Hydra structured config for type-safe toolkit definitions.

    Domain concept representing what a toolkit configuration IS,
    independent of how it's loaded or managed.
    """

    name: str
    type: str  # web_search, analysis, code, data, crypto
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    implementation: str | None = None  # For custom toolkits

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Toolkit name cannot be empty")
        return v.strip()

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid_types = {"web_search", "analysis", "code", "data", "crypto", "generic"}
        if v not in valid_types:
            raise ValueError(f"Invalid toolkit type: {v}. Must be one of {valid_types}")
        return v

    def is_default_agno_toolkit(self) -> bool:
        """Check if this is a default Agno toolkit (no implementation field)."""
        return self.implementation is None

    def is_custom_toolkit(self) -> bool:
        """Check if this is a custom toolkit (has implementation field)."""
        return self.implementation is not None


@dataclass
class AgentToolkitsConfig:
    """
    Pydantic dataclass for agent toolkit configuration.

    Domain concept representing which toolkits should be attached to an agent.
    Independent of the loading mechanism.
    """

    toolkits: list[str | ToolkitConfig] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)  # Alternative format

    @field_validator("toolkits")
    @classmethod
    def validate_toolkits(cls, v: list[str | ToolkitConfig]) -> list[str | ToolkitConfig]:
        if not isinstance(v, list):
            raise ValueError("toolkits must be a list")
        return v
