"""
Configuration Loader interface for ROMA v2.

Handles loading and resolving agent configurations from YAML files
with modern Pydantic validation and clean architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ConfigurationLoader(ABC):
    """
    Abstract interface for configuration loading.
    
    Supports modern YAML configuration format with agent profiles
    and task-type specific mappings.
    """
    
    def __init__(self):
        """Initialize configuration loader."""
        self._config_cache = {}
        
    async def load_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration format.
        
        Args:
            config_data: Raw configuration dictionary
            
        Returns:
            Processed configuration
        """
        # Store for resolution
        self._config_cache = config_data
        return config_data
        
    def resolve_config(self, task_type: str, action_verb: str) -> Dict[str, Any]:
        """
        Resolve agent configuration for task type and action.
        
        Args:
            task_type: Task type (RETRIEVE, WRITE, THINK, CODE_INTERPRET, IMAGE_GENERATION)
            action_verb: Agent action (atomizer, planner, executor, aggregator, plan_modifier)
            
        Returns:
            Resolved agent configuration
        """
        # Try to find in cached config
        if "agents" in self._config_cache:
            agent_config = self._config_cache["agents"].get(action_verb, {})
            if agent_config:
                return agent_config
                
        # Default configuration  
        return {
            "name": f"{task_type}_{action_verb}",
            "model": "gpt-4o",
            "temperature": 0.1,
            "task_type": task_type,
            "action_verb": action_verb
        }