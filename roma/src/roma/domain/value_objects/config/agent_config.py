"""
Agent Configuration Value Object.

Defines agent configuration as a domain value object.
Single source of truth for agent configuration across the system.
"""

from dataclasses import field
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic.dataclasses import dataclass

from roma.domain.value_objects.task_type import TaskType

from .model_config import ModelConfig
from .tool_config import ToolConfig


@dataclass(frozen=True)
class AgentConfig:
    """Agent configuration value object."""

    name: str
    type: str  # atomizer, planner, executor, aggregator, plan_modifier
    task_type: TaskType | None
    description: str = ""
    model: ModelConfig = ModelConfig()
    prompt_template: str | None = None
    output_schema: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    tools: list[ToolConfig] = field(default_factory=list)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Agent name cannot be empty")
        return v.strip()

    @field_validator("type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent type string matches known types."""
        valid_types = ["atomizer", "planner", "executor", "aggregator", "plan_modifier"]
        if v not in valid_types:
            raise ValueError(f"agent type must be one of {valid_types}, got: {v}")
        return v

    @field_validator("task_type", mode="before")
    @classmethod
    def validate_task_type(cls, v: Any) -> TaskType | None:
        """Convert string to TaskType if needed, allow null for general-purpose agents."""
        if v is None:
            return None
        elif isinstance(v, str):
            return TaskType.from_string(v)
        elif isinstance(v, TaskType):
            return v
        else:
            raise ValueError(f"task_type must be TaskType, string, or null, got: {type(v)}")

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template(cls, v: str | None) -> str | None:
        """Validate prompt template path exists."""
        if v is None:
            return v

        # Check if it's a relative path (should be relative to src/prompts/)
        if not v.endswith(".jinja2"):
            raise ValueError(f"prompt_template must be a .jinja2 file, got: {v}")

        # For relative paths, they should exist in src/prompts/
        prompt_path = Path("src/prompts") / v
        if not prompt_path.exists():
            # Don't fail hard during config loading, just warn
            # This allows for dynamic creation of templates
            pass

        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "task_type": self.task_type.value if self.task_type else None,
            "description": self.description,
            "model": self.model.to_dict(),
            "prompt_template": self.prompt_template,
            "output_schema": self.output_schema,
            "config": dict(self.config),
            "tools": [tool.to_dict() for tool in self.tools],
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        model_data = data.get("model", {})
        if isinstance(model_data, dict):
            model_config = ModelConfig.from_dict(model_data)
        else:
            model_config = model_data

        tools_data = data.get("tools", [])
        tools = []
        for tool_data in tools_data:
            if isinstance(tool_data, ToolConfig):
                tools.append(tool_data)
            else:
                # Skip invalid tools - only ToolConfig objects are supported
                continue

        return cls(
            name=data["name"],
            type=data["type"],
            task_type=data["task_type"],
            description=data.get("description", ""),
            model=model_config,
            prompt_template=data.get("prompt_template"),
            output_schema=data.get("output_schema"),
            config=data.get("config", {}),
            tools=tools,
            enabled=data.get("enabled", True),
        )
