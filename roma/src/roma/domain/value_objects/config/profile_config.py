"""
Profile Configuration Value Objects.

Defines agent profile and mapping configurations as domain value objects.
Single source of truth for profile configuration across the system.
"""

from pydantic.dataclasses import dataclass
from pydantic import Field, field_validator
from dataclasses import field
from typing import Dict, Any
from ..task_type import TaskType


@dataclass(frozen=True)
class AgentMappingConfig:
    """Maps task types to agent names for each agent type."""
    
    atomizers: Dict[str, str] = field(default_factory=dict)
    planners: Dict[str, str] = field(default_factory=dict)
    executors: Dict[str, str] = field(default_factory=dict)
    aggregators: Dict[str, str] = field(default_factory=dict)
    plan_modifiers: Dict[str, str] = field(default_factory=dict)
    
    @field_validator("atomizers", "planners", "executors", "aggregators", "plan_modifiers")
    @classmethod
    def validate_task_type_keys(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate that all keys are valid TaskType values."""
        valid_task_types = [task_type.value for task_type in TaskType]
        
        for key in v.keys():
            if key not in valid_task_types:
                raise ValueError(
                    f"Invalid task type key '{key}'. Valid task types: {valid_task_types}"
                )
        return v
    
    @field_validator("atomizers", "planners", "executors", "aggregators", "plan_modifiers")
    @classmethod
    def validate_agent_names_not_empty(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate that agent names are not empty."""
        for key, agent_name in v.items():
            if not agent_name or len(agent_name.strip()) == 0:
                raise ValueError(f"Agent name cannot be empty for task type '{key}'")
        
        # Clean up agent names
        return {key: agent_name.strip() for key, agent_name in v.items()}
    
    def get_agent_for_task(self, agent_type: str, task_type: TaskType) -> str:
        """Get agent name for specific agent type and task type."""
        mapping = getattr(self, agent_type, {})
        return mapping.get(task_type.value, "")
    
    def has_agent_for_task(self, agent_type: str, task_type: TaskType) -> bool:
        """Check if agent mapping exists for specific agent type and task type."""
        return bool(self.get_agent_for_task(agent_type, task_type))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "atomizers": dict(self.atomizers),
            "planners": dict(self.planners),
            "executors": dict(self.executors),
            "aggregators": dict(self.aggregators),
            "plan_modifiers": dict(self.plan_modifiers),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMappingConfig":
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
    settings: Dict[str, Any] = field(default_factory=dict)  # Execution settings
    multimodal: Dict[str, Any] = field(default_factory=dict)  # Multimodal config
    observability: Dict[str, Any] = field(default_factory=dict)  # Observability settings
    hitl: Dict[str, Any] = field(default_factory=dict)  # Human-in-the-loop settings
    
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
    
    def validate_completeness(self) -> Dict[str, Any]:
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "agent_mapping": self.agent_mapping.to_dict(),
            "enabled": self.enabled,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileConfig":
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