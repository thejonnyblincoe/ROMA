"""
Agno Toolkit Manager for ROMA v2.

Provides CRUD operations for both:
1. Default Agno toolkits - imported from Agno
2. Custom toolkits - loaded from custom implementations

Manages toolkit lifecycle and makes them available to agents.
"""

from typing import Dict, Any, List, Optional, Type
import logging
import importlib
from pathlib import Path

from .base_agno_toolkit import (
    BaseAgnoToolkit,
    DefaultAgnoToolkitWrapper
)
from roma.domain.value_objects.config.tool_config import ToolConfig

logger = logging.getLogger(__name__)


class AgnoToolkitManager:
    """
    Manages CRUD operations for Agno toolkits.
    
    Handles both default Agno toolkits and custom implementations,
    making them available to agents based on configuration.
    """
    
    def __init__(self):
        """Initialize toolkit manager."""
        self._toolkits: Dict[str, Any] = {}
        self._default_toolkits: Dict[str, DefaultAgnoToolkitWrapper] = {}
        self._custom_toolkits: Dict[str, BaseAgnoToolkit] = {}
        self._tool_configs: Dict[str, ToolConfig] = {}  # Store ToolConfig instances
        self._initialized = False

    def register_tool_config(self, tool_config: ToolConfig) -> None:
        """Register a ToolConfig for later use."""
        self._tool_configs[tool_config.name] = tool_config
        logger.debug(f"Registered tool config: {tool_config.name}")

    def get_tool_config(self, tool_name: str) -> Optional[ToolConfig]:
        """Get ToolConfig by tool name."""
        return self._tool_configs.get(tool_name)
        
    async def initialize(self) -> None:
        """Initialize the toolkit manager."""
        self._initialized = True
        logger.info("Agno Toolkit Manager initialized")
        
    def is_connected(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized
        
    async def create_toolkit(self, toolkit_spec: Dict[str, Any]) -> Any:
        """
        Create a new toolkit instance.

        Args:
            toolkit_spec: Toolkit specification with name, type, config

        Returns:
            Created toolkit instance
        """
        if not self._initialized:
            await self.initialize()

        toolkit_name = toolkit_spec.get("name")
        if not toolkit_name:
            raise ValueError("Toolkit name is required")

        # Check if toolkit already exists
        if toolkit_name in self._toolkits:
            logger.debug(f"Returning existing toolkit: {toolkit_name}")
            return self._toolkits[toolkit_name]

        # Try to enhance toolkit_spec with registered ToolConfig if available
        tool_config = self.get_tool_config(toolkit_name)
        if tool_config:
            # Merge ToolConfig information into toolkit_spec
            enhanced_spec = {
                "name": tool_config.name,
                "type": tool_config.type,
                "enabled": getattr(tool_config, 'enabled', True),
                **toolkit_spec  # Original spec overrides ToolConfig
            }
            toolkit_spec = enhanced_spec
            logger.debug(f"Enhanced toolkit spec for {toolkit_name} with registered ToolConfig")

        # Check if it's a default Agno toolkit or custom implementation
        if "implementation" in toolkit_spec:
            # Custom toolkit - load from implementation class
            toolkit = await self._create_custom_toolkit(toolkit_spec)
            self._custom_toolkits[toolkit_name] = toolkit
        else:
            # Default Agno toolkit - import from Agno
            toolkit = await self._create_default_toolkit(toolkit_spec)
            self._default_toolkits[toolkit_name] = toolkit

        self._toolkits[toolkit_name] = toolkit
        logger.info(f"Created toolkit: {toolkit_name}")
        return toolkit
        
    async def _create_default_toolkit(self, spec: Dict[str, Any]) -> DefaultAgnoToolkitWrapper:
        """Create default Agno toolkit by importing from Agno."""
        toolkit_name = spec.get("name")
        toolkit_type = spec.get("type", "generic")

        # Create wrapper for default Agno toolkit (now supports include/exclude from config)
        wrapper = DefaultAgnoToolkitWrapper(toolkit_name, spec)
        await wrapper.create()

        logger.info(f"Created default Agno toolkit: {toolkit_name} (type: {toolkit_type})")
        return wrapper
        
    async def _create_custom_toolkit(self, spec: Dict[str, Any]) -> BaseAgnoToolkit:
        """Create custom toolkit by loading implementation class."""
        implementation_path = spec.get("implementation")
        if not implementation_path:
            raise ValueError("Implementation path required for custom toolkit")
            
        try:
            # Import custom toolkit class
            module_path, class_name = implementation_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            toolkit_class = getattr(module, class_name)
            
            # Instantiate custom toolkit
            toolkit_config = spec.get("config", {})
            toolkit = toolkit_class(spec, **toolkit_config)
            
            if hasattr(toolkit, 'create'):
                await toolkit.create()
                
            logger.info(f"Created custom toolkit: {spec.get('name')} from {implementation_path}")
            return toolkit
            
        except Exception as e:
            logger.error(f"Failed to create custom toolkit {spec.get('name')}: {e}")
            raise
            
    async def read_toolkit(self, toolkit_name: str) -> Optional[Any]:
        """
        Read/get a toolkit by name.
        
        Args:
            toolkit_name: Name of toolkit to retrieve
            
        Returns:
            Toolkit instance or None if not found
        """
        return self._toolkits.get(toolkit_name)
        
    async def list_available_toolkits(self) -> List[Dict[str, Any]]:
        """
        List all available toolkits.
        
        Returns:
            List of toolkit information dictionaries
        """
        toolkits = []
        
        for name, toolkit in self._toolkits.items():
            toolkit_info = {
                "name": name,
                "type": getattr(toolkit, "toolkit_type", "unknown"),
                "is_default": getattr(toolkit, "is_default_agno_toolkit", lambda: False)(),
                "is_custom": getattr(toolkit, "is_custom_toolkit", lambda: False)(),
                "enabled": getattr(toolkit, "is_enabled", lambda: True)()
            }
            toolkits.append(toolkit_info)
            
        return toolkits
        
    async def list_default_agno_toolkits(self) -> List[Dict[str, Any]]:
        """List only default Agno toolkits."""
        return [
            {
                "name": name,
                "type": toolkit.toolkit_type,
                "source": "agno_default"
            }
            for name, toolkit in self._default_toolkits.items()
        ]
        
    async def list_custom_toolkits(self) -> List[Dict[str, Any]]:
        """List only custom toolkit implementations.""" 
        return [
            {
                "name": name,
                "type": getattr(toolkit, "toolkit_type", "unknown"),
                "source": "custom_implementation"
            }
            for name, toolkit in self._custom_toolkits.items()
        ]
        
    async def update_toolkit(self, toolkit_name: str, new_config: Dict[str, Any]) -> Any:
        """
        Update toolkit configuration.
        
        Args:
            toolkit_name: Name of toolkit to update
            new_config: New configuration parameters
            
        Returns:
            Updated toolkit instance
        """
        toolkit = self._toolkits.get(toolkit_name)
        if not toolkit:
            raise ValueError(f"Toolkit {toolkit_name} not found")
            
        if hasattr(toolkit, 'update_config'):
            await toolkit.update_config(new_config)
        else:
            logger.warning(f"Toolkit {toolkit_name} does not support config updates")
            
        logger.info(f"Updated toolkit config: {toolkit_name}")
        return toolkit
        
    async def delete_toolkit(self, toolkit_name: str) -> bool:
        """
        Delete/remove a toolkit.
        
        Args:
            toolkit_name: Name of toolkit to delete
            
        Returns:
            True if successfully deleted
        """
        if toolkit_name not in self._toolkits:
            return False
            
        toolkit = self._toolkits[toolkit_name]
        
        # Cleanup toolkit
        if hasattr(toolkit, 'delete'):
            await toolkit.delete()
            
        # Remove from registries
        del self._toolkits[toolkit_name]
        self._default_toolkits.pop(toolkit_name, None)
        self._custom_toolkits.pop(toolkit_name, None)
        
        logger.info(f"Deleted toolkit: {toolkit_name}")
        return True
        
    async def toolkit_exists(self, toolkit_name: str) -> bool:
        """Check if toolkit exists."""
        return toolkit_name in self._toolkits
        
    def get_toolkit_for_agent(self, toolkit_name: str) -> Any:
        """
        Get toolkit instance for adding to an agent.
        
        Args:
            toolkit_name: Name of toolkit
            
        Returns:
            Toolkit instance ready to be added to agent
        """
        toolkit = self._toolkits.get(toolkit_name)
        if not toolkit:
            logger.error(f"Toolkit {toolkit_name} not found for agent")
            return None
            
        # For default Agno toolkits, return the underlying Agno toolkit
        if isinstance(toolkit, DefaultAgnoToolkitWrapper):
            return toolkit.get_agno_toolkit()
            
        # For custom toolkits, return the toolkit itself
        return toolkit

    async def find_toolkits_by_capability(
        self,
        required_tools: List[str],
        preferred_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find toolkits that provide specific capabilities.

        Args:
            required_tools: List of tool names required
            preferred_types: Optional preferred toolkit types

        Returns:
            List of toolkit info dicts that provide required tools
        """
        matching_toolkits = []

        for name, toolkit in self._toolkits.items():
            # Get available tools from toolkit
            toolkit_tools = set()
            if hasattr(toolkit, 'get_available_tools'):
                try:
                    toolkit_tools = set(toolkit.get_available_tools())
                except Exception as e:
                    logger.warning(f"Could not get tools for {name}: {e}")
                    continue

            required_tools_set = set(required_tools)

            # Check if toolkit provides all required tools
            if required_tools_set.issubset(toolkit_tools):
                toolkit_info = {
                    "name": name,
                    "type": getattr(toolkit, "toolkit_type", "unknown"),
                    "is_default": getattr(toolkit, "is_default_agno_toolkit", lambda: False)(),
                    "is_custom": getattr(toolkit, "is_custom_toolkit", lambda: False)(),
                    "available_tools": list(toolkit_tools),
                    "enabled": getattr(toolkit, "is_enabled", lambda: True)()
                }
                matching_toolkits.append(toolkit_info)

        # Sort by preferred types if specified
        if preferred_types:
            def sort_key(toolkit):
                try:
                    return preferred_types.index(toolkit["type"])
                except ValueError:
                    return len(preferred_types)  # Put non-preferred types at end

            matching_toolkits.sort(key=sort_key)

        return matching_toolkits

    async def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get toolkit registry statistics and health summary.

        Returns:
            Dictionary with toolkit statistics
        """
        total_toolkits = len(self._toolkits)

        # Count by type
        type_counts = {}
        custom_count = 0
        default_count = 0
        enabled_count = 0

        for toolkit in self._toolkits.values():
            # Count by type
            toolkit_type = getattr(toolkit, "toolkit_type", "unknown")
            type_counts[toolkit_type] = type_counts.get(toolkit_type, 0) + 1

            # Count by source
            if getattr(toolkit, "is_default_agno_toolkit", lambda: False)():
                default_count += 1
            elif getattr(toolkit, "is_custom_toolkit", lambda: False)():
                custom_count += 1

            # Count enabled
            if getattr(toolkit, "is_enabled", lambda: True)():
                enabled_count += 1

        return {
            "total_toolkits": total_toolkits,
            "enabled_toolkits": enabled_count,
            "default_agno_toolkits": default_count,
            "custom_toolkits": custom_count,
            "toolkit_type_counts": type_counts,
            "availability_percentage": round((enabled_count / total_toolkits * 100) if total_toolkits > 0 else 0, 2),
            "last_updated": self._get_current_timestamp()
        }

    async def get_available_tools(self) -> List[Any]:
        """
        Get list of all available tools from all toolkits.

        Returns:
            List of tool instances from all registered toolkits
        """
        available_tools = []

        for toolkit_name, toolkit in self._toolkits.items():
            try:
                # Get tools from toolkit
                if hasattr(toolkit, 'get_tools'):
                    toolkit_tools = toolkit.get_tools()
                    if toolkit_tools:
                        available_tools.extend(toolkit_tools)
                elif hasattr(toolkit, 'tools'):
                    # Some toolkits might have tools as a direct attribute
                    toolkit_tools = getattr(toolkit, 'tools', [])
                    if toolkit_tools:
                        available_tools.extend(toolkit_tools)
                else:
                    logger.debug(f"Toolkit {toolkit_name} has no accessible tools")

            except Exception as e:
                logger.warning(f"Failed to get tools from toolkit {toolkit_name}: {e}")
                continue

        logger.debug(f"Found {len(available_tools)} total tools from {len(self._toolkits)} toolkits")
        return available_tools

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()