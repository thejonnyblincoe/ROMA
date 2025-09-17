"""
Toolkit Registry interface for extensible tool management.

Manages discovery, registration, and health monitoring of toolkits
for agent execution in the ROMA framework.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ToolkitHealthStatus(Enum):
    """Toolkit health status enumeration."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class RegisteredToolkit(BaseModel):
    """Information about a registered toolkit."""
    name: str = Field(..., description="Toolkit name")
    toolkit_type: str = Field(..., description="Type of toolkit")
    description: Optional[str] = Field(None, description="Toolkit description")
    health_status: ToolkitHealthStatus = ToolkitHealthStatus.UNKNOWN
    health_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Health score")
    last_health_check: Optional[str] = Field(None, description="ISO timestamp of last health check")
    available_tools: List[str] = Field(default_factory=list, description="Available tool names")
    supports_validation: bool = False
    is_default_agno: bool = False
    is_custom_implementation: bool = False


class ToolkitRegistry(ABC):
    """
    Abstract interface for toolkit registry.
    
    Manages toolkits that provide capabilities to agents
    (web search, data analysis, code execution, etc.)
    """
    
    def __init__(self):
        """Initialize toolkit registry."""
        self._toolkits = {}
        
    async def discover_toolkits(self) -> List[Any]:
        """
        Discover available toolkits.
        
        Returns:
            List of available toolkit instances
        """
        # Return registered toolkits for now
        return list(self._toolkits.values())
        
    async def register_toolkit(self, toolkit: Any) -> None:
        """
        Register a toolkit.
        
        Args:
            toolkit: Toolkit instance to register
        """
        self._toolkits[toolkit.name] = toolkit
        
    def compose_toolkits(self, required_capabilities: List[str]) -> List[Any]:
        """
        Compose toolkits that provide required capabilities.
        
        Args:
            required_capabilities: List of capability names needed
            
        Returns:
            List of toolkits that together provide all capabilities
        """
        composed = []
        remaining_capabilities = set(required_capabilities)
        
        for toolkit in self._toolkits.values():
            if hasattr(toolkit, 'capabilities'):
                toolkit_caps = set(toolkit.capabilities)
                if toolkit_caps & remaining_capabilities:  # Has some required capabilities
                    composed.append(toolkit)
                    remaining_capabilities -= toolkit_caps
                    
        return composed
        
    def get_toolkit(self, name: str) -> Any:
        """
        Get toolkit by name.
        
        Args:
            name: Toolkit name
            
        Returns:
            Toolkit instance or None
        """
        return self._toolkits.get(name)