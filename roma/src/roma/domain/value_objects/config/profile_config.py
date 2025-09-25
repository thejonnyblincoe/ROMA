"""
Profile Configuration Value Objects.

Defines agent profile and mapping configurations as domain value objects.
Single source of truth for profile configuration across the system.
"""

from dataclasses import dataclass as dataclass_decorator
from dataclasses import field
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import field_validator
from pydantic.dataclasses import dataclass

from roma.domain.value_objects.config.agent_config import AgentConfig
from roma.domain.value_objects.task_type import TaskType


@dataclass_decorator(frozen=False)  # Allow mutation for post_init instantiation
class AgentMappingConfig:
    """Maps task types to agent configurations."""

    # Accept Any (DictConfig/dict) from Hydra, convert to AgentConfig in post_init
    atomizers: dict[str, Any] = field(default_factory=dict)
    planners: dict[str, Any] = field(default_factory=dict)
    executors: dict[str, Any] = field(default_factory=dict)
    aggregators: dict[str, Any] = field(default_factory=dict)
    plan_modifiers: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Instantiate agent configs and validate task type keys."""
        valid_task_types = [task_type.value for task_type in TaskType] + [
            "AGGREGATE",
            "default",
            "error_recovery",
        ]

        for agent_type in ["atomizers", "planners", "executors", "aggregators", "plan_modifiers"]:
            mapping = getattr(self, agent_type, {})
            if isinstance(mapping, dict):
                instantiated_mapping = {}
                for key, agent_data in mapping.items():
                    # Validate task type key
                    if key not in valid_task_types:
                        raise ValueError(
                            f"Invalid task type key '{key}' in {agent_type}. Valid task types: {valid_task_types}"
                        )

                    # Instantiate agent if it's a dict/DictConfig with _target_
                    if isinstance(agent_data, (dict, DictConfig)) and "_target_" in agent_data:
                        try:
                            instantiated_agent = instantiate(agent_data, _convert_="object")
                            instantiated_mapping[key] = instantiated_agent
                        except Exception as e:
                            raise ValueError(f"Failed to instantiate agent {agent_type}.{key}: {e}") from e
                    else:
                        # Already instantiated or no _target_
                        instantiated_mapping[key] = agent_data

                # Update the mapping with instantiated agents
                object.__setattr__(self, agent_type, instantiated_mapping)

    def get_agent_for_task(self, agent_type: str, task_type: TaskType) -> AgentConfig | None:
        """Get agent config for specific agent type and task type."""
        mapping = getattr(self, agent_type, {})
        return mapping.get(task_type.value, None)

    def has_agent_for_task(self, agent_type: str, task_type: TaskType) -> bool:
        """Check if agent mapping exists for specific agent type and task type."""
        return bool(self.get_agent_for_task(agent_type, task_type))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "atomizers": dict(self.atomizers),
            "planners": dict(self.planners),
            "executors": dict(self.executors),
            "aggregators": dict(self.aggregators),
            "plan_modifiers": dict(self.plan_modifiers),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMappingConfig":
        """Create from dictionary."""
        return cls(
            atomizers=data.get("atomizers", {}),
            planners=data.get("planners", {}),
            executors=data.get("executors", {}),
            aggregators=data.get("aggregators", {}),
            plan_modifiers=data.get("plan_modifiers", {}),
        )


@dataclass(frozen=True)
class ProfileConfig:
    """Agent profile configuration value object."""

    name: str = "default_profile"
    description: str = "Default ROMA profile"
    version: str = "2.0.0"
    agent_mapping: AgentMappingConfig = field(default_factory=lambda: AgentMappingConfig())
    enabled: bool = True

    # Additional profile settings from YAML (optional)
    settings: dict[str, Any] = field(default_factory=dict)  # Execution settings
    multimodal: dict[str, Any] = field(default_factory=dict)  # Multimodal config
    observability: dict[str, Any] = field(default_factory=dict)  # Observability settings
    hitl: dict[str, Any] = field(default_factory=dict)  # Human-in-the-loop settings

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Profile name cannot be empty")
        return v.strip()

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Profile version cannot be empty")
        return v.strip()

    def validate_completeness(self) -> dict[str, Any]:
        """Validate that profile has complete agent mappings for all task types."""
        missing_mappings = {}
        agent_types = ["atomizers", "planners", "executors", "aggregators"]

        for agent_type in agent_types:
            missing_tasks = []
            mapping = getattr(self.agent_mapping, agent_type, {})

            for task_type in TaskType:
                if task_type.value not in mapping:
                    missing_tasks.append(task_type.value)

            if missing_tasks:
                missing_mappings[agent_type] = missing_tasks

        return missing_mappings

    def is_complete(self) -> bool:
        """Check if profile has complete agent mappings."""
        return len(self.validate_completeness()) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "agent_mapping": self.agent_mapping.to_dict(),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileConfig":
        """Create from dictionary."""
        mapping_data = data.get("agent_mapping", {})
        if isinstance(mapping_data, dict):
            agent_mapping = AgentMappingConfig.from_dict(mapping_data)
        else:
            agent_mapping = mapping_data

        return cls(
            name=data.get("name", "default_profile"),
            description=data.get("description", "Default ROMA profile"),
            version=data.get("version", "2.0.0"),
            agent_mapping=agent_mapping,
            enabled=data.get("enabled", True),
            settings=data.get("settings", {}),
            multimodal=data.get("multimodal", {}),
            observability=data.get("observability", {}),
            hitl=data.get("hitl", {}),
        )
