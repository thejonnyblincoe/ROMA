"""
Toolkit Loader for ROMA v2.

Uses Hydra structured configs with Pydantic dataclasses for type-safe toolkit configuration.
Only initializes toolkits when they are attached to agents (lazy initialization).
"""

from typing import Dict, Any, List, Optional, Set, Union
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from roma.domain.value_objects import ToolkitConfig, AgentToolkitsConfig
from roma.infrastructure.toolkits.agno_toolkit_manager import AgnoToolkitManager
from roma.infrastructure.utils.async_file_utils import async_path_exists, async_yaml_load

logger = logging.getLogger(__name__)


class ToolkitLoader:
    """
    Loads toolkits based on Hydra structured configs with Pydantic validation.
    
    Only initializes toolkits when they are actually needed by agents.
    Uses Pydantic dataclasses for type-safe configuration handling.
    """
    
    def __init__(self, hydra_config: Optional[DictConfig] = None):
        """
        Initialize toolkit loader with Hydra-resolved configuration.
        
        Args:
            hydra_config: Hydra-resolved configuration object (optional for testing)
        """
        self._config = hydra_config or {}
        self._manager = AgnoToolkitManager()
        self._initialized_toolkits: Dict[str, Any] = {}
        self._manager_initialized = False
        
    async def _ensure_manager_initialized(self) -> None:
        """Ensure toolkit manager is initialized."""
        if not self._manager_initialized:
            await self._manager.initialize()
            self._manager_initialized = True
            
    def get_agent_toolkit_config(self, agent_name: str) -> AgentToolkitsConfig:
        """
        Get agent toolkit configuration using Pydantic structured config.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Validated agent toolkit configuration
        """
        try:
            # Get agent config from Hydra (assumes structured config)
            agent_config_dict = {}
            
            # Try to find agent config in Hydra structure
            if hasattr(self._config, 'agents') and agent_name in self._config.agents:
                agent_config_dict = self._config.agents[agent_name]
            elif hasattr(self._config, 'profiles') and hasattr(self._config.profiles, 'agents'):
                if agent_name in self._config.profiles.agents:
                    agent_config_dict = self._config.profiles.agents[agent_name]
                    
            # Convert to Pydantic dataclass for validation
            agent_toolkit_config = AgentToolkitsConfig(
                toolkits=agent_config_dict.get('toolkits', []),
                available_tools=agent_config_dict.get('available_tools', [])
            )
            
            return agent_toolkit_config
            
        except Exception as e:
            logger.error(f"Failed to get agent toolkit config for {agent_name}: {e}")
            return AgentToolkitsConfig()
            
    def resolve_toolkit_specs(self, agent_config: AgentToolkitsConfig) -> List[ToolkitConfig]:
        """
        Resolve toolkit specifications from agent configuration.
        
        Args:
            agent_config: Agent toolkit configuration
            
        Returns:
            List of resolved toolkit configurations
        """
        toolkit_specs = []
        
        # Process toolkits list
        for toolkit_ref in agent_config.toolkits:
            if isinstance(toolkit_ref, str):
                # String reference - look up in config
                toolkit_spec = self._lookup_toolkit_config(toolkit_ref)
                if toolkit_spec:
                    toolkit_specs.append(toolkit_spec)
            elif isinstance(toolkit_ref, dict):
                # Direct toolkit specification - convert to ToolkitConfig
                try:
                    toolkit_spec = ToolkitConfig(**toolkit_ref)
                    toolkit_specs.append(toolkit_spec)
                except Exception as e:
                    logger.error(f"Invalid toolkit config: {e}")
                    continue
                    
        # Process available_tools (legacy format)
        for tool_name in agent_config.available_tools:
            toolkit_spec = self._lookup_toolkit_config(tool_name)
            if toolkit_spec:
                toolkit_specs.append(toolkit_spec)
                
        return toolkit_specs
        
    def _lookup_toolkit_config(self, toolkit_name: str) -> Optional[ToolkitConfig]:
        """
        Look up toolkit configuration by name in Hydra structured config.
        
        Args:
            toolkit_name: Name of toolkit to look up
            
        Returns:
            Validated toolkit configuration
        """
        try:
            # Look in entities.tools (assuming structured config)
            if not hasattr(self._config, 'entities') or not hasattr(self._config.entities, 'tools'):
                logger.warning("No entities.tools configuration found")
                return None
                
            tools_config = self._config.entities.tools
            
            # Search through toolkit categories
            for category_name in tools_config:
                category = getattr(tools_config, category_name)
                if hasattr(category, toolkit_name):
                    toolkit_dict = getattr(category, toolkit_name)
                    
                    # Convert to dictionary and add name
                    if hasattr(toolkit_dict, '_content'):
                        # OmegaConf DictConfig
                        config_dict = dict(toolkit_dict)
                    else:
                        config_dict = toolkit_dict
                        
                    config_dict['name'] = toolkit_name
                    
                    # Create Pydantic dataclass with validation
                    return ToolkitConfig(**config_dict)
                    
            logger.warning(f"Toolkit {toolkit_name} not found in structured config")
            return None
            
        except Exception as e:
            logger.error(f"Failed to lookup toolkit config for {toolkit_name}: {e}")
            return None
            
    async def attach_toolkits_to_agent(self, agent: Any, agent_name: str) -> List[str]:
        """
        Attach toolkits to agent based on Hydra structured config.
        
        Args:
            agent: Agent instance
            agent_name: Name of the agent
            
        Returns:
            List of attached toolkit names
        """
        await self._ensure_manager_initialized()
        
        try:
            # Get validated agent toolkit configuration
            agent_config = self.get_agent_toolkit_config(agent_name)
            toolkit_specs = self.resolve_toolkit_specs(agent_config)
            
            attached_toolkits = []
            
            for toolkit_spec in toolkit_specs:
                try:
                    # Initialize toolkit if not already done (lazy initialization)
                    if toolkit_spec.name not in self._initialized_toolkits:
                        toolkit = await self._initialize_toolkit(toolkit_spec)
                        self._initialized_toolkits[toolkit_spec.name] = toolkit
                    else:
                        toolkit = self._initialized_toolkits[toolkit_spec.name]
                        
                    # Attach to agent
                    await self._attach_toolkit_to_agent(agent, toolkit_spec.name)
                    attached_toolkits.append(toolkit_spec.name)
                    
                    logger.info(f"Attached toolkit {toolkit_spec.name} to agent {agent_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to attach toolkit {toolkit_spec.name}: {e}")
                    continue
                    
            logger.info(f"Attached {len(attached_toolkits)} toolkits to agent {agent_name}")
            return attached_toolkits
            
        except Exception as e:
            logger.error(f"Failed to attach toolkits to agent {agent_name}: {e}")
            return []
            
    async def _initialize_toolkit(self, toolkit_spec: ToolkitConfig) -> Any:
        """
        Initialize toolkit from Pydantic configuration.
        
        Args:
            toolkit_spec: Validated toolkit configuration
            
        Returns:
            Initialized toolkit instance
        """
        try:
            # Convert Pydantic dataclass to dict for manager
            toolkit_dict = {
                'name': toolkit_spec.name,
                'type': toolkit_spec.type,
                'enabled': toolkit_spec.enabled,
                'config': toolkit_spec.config
            }
            
            if toolkit_spec.implementation:
                toolkit_dict['implementation'] = toolkit_spec.implementation
                
            toolkit = await self._manager.create_toolkit(toolkit_dict)
            logger.info(f"Initialized toolkit: {toolkit_spec.name}")
            return toolkit
            
        except Exception as e:
            logger.error(f"Failed to initialize toolkit {toolkit_spec.name}: {e}")
            raise
            
    async def _attach_toolkit_to_agent(self, agent: Any, toolkit_name: str) -> None:
        """
        Attach initialized toolkit to agent.
        
        Args:
            agent: Agent instance
            toolkit_name: Name of toolkit to attach
        """
        try:
            # Get toolkit instance for agent
            agent_toolkit = self._manager.get_toolkit_for_agent(toolkit_name)
            
            if agent_toolkit is None:
                logger.error(f"No toolkit instance found: {toolkit_name}")
                return
                
            # Attach to agent using appropriate method
            if hasattr(agent, 'add_toolkit'):
                # Agent has explicit toolkit management
                await agent.add_toolkit(agent_toolkit)
            elif hasattr(agent, '_toolkits'):
                # Agent has toolkit dictionary
                agent._toolkits[toolkit_name] = agent_toolkit
            else:
                # Create toolkit dictionary on agent
                if not hasattr(agent, '_toolkits'):
                    agent._toolkits = {}
                agent._toolkits[toolkit_name] = agent_toolkit
                
            logger.debug(f"Attached toolkit {toolkit_name} to agent")
            
        except Exception as e:
            logger.error(f"Failed to attach toolkit {toolkit_name}: {e}")
            raise
            
    def get_initialized_toolkits(self) -> Dict[str, Any]:
        """Get all initialized toolkits."""
        return self._initialized_toolkits.copy()
        
    async def cleanup_unused_toolkits(self, active_agents: Set[str]) -> None:
        """
        Cleanup toolkits not needed by active agents.
        
        Args:
            active_agents: Set of active agent names
        """
        try:
            # Find toolkits still needed
            needed_toolkits = set()
            
            for agent_name in active_agents:
                agent_config = self.get_agent_toolkit_config(agent_name)
                toolkit_specs = self.resolve_toolkit_specs(agent_config)
                
                for spec in toolkit_specs:
                    needed_toolkits.add(spec.name)
                    
            # Remove unused toolkits
            to_remove = set(self._initialized_toolkits.keys()) - needed_toolkits
            
            for toolkit_name in to_remove:
                try:
                    await self._manager.delete_toolkit(toolkit_name)
                    del self._initialized_toolkits[toolkit_name]
                    logger.info(f"Cleaned up toolkit: {toolkit_name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup toolkit {toolkit_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup toolkits: {e}")
            
    async def load_toolkit_configs(self, config_path: str) -> Dict[str, Any]:
        """
        Load toolkit configurations from YAML file.

        Args:
            config_path: Path to toolkit definitions YAML file

        Returns:
            Dictionary of toolkit configurations
        """
        try:
            config_file = Path(config_path)
            if not await async_path_exists(config_file):
                logger.warning(f"Config file not found: {config_path}")
                return {}

            config_data = await async_yaml_load(config_file)
            return config_data or {}

        except Exception as e:
            logger.error(f"Failed to load toolkit configs from {config_path}: {e}")
            return {}
            
    def categorize_toolkits(self, toolkit_configs: Dict[str, Any]) -> tuple[List[Dict], List[Dict]]:
        """
        Categorize toolkits into default and custom implementations.
        
        Args:
            toolkit_configs: Dictionary of toolkit configurations
            
        Returns:
            Tuple of (default_toolkits, custom_toolkits)
        """
        default_toolkits = []
        custom_toolkits = []
        
        for category_name, category_config in toolkit_configs.items():
            if isinstance(category_config, dict):
                # Check if this is a direct toolkit config (has implementation field directly)
                if 'implementation' in category_config:
                    # This is a custom toolkit config at the top level
                    toolkit_spec = category_config.copy()
                    toolkit_spec['name'] = category_name
                    custom_toolkits.append(toolkit_spec)
                else:
                    # This is a category containing toolkit configs
                    for toolkit_name, toolkit_config in category_config.items():
                        if isinstance(toolkit_config, dict):
                            toolkit_spec = toolkit_config.copy()
                            toolkit_spec['name'] = toolkit_name
                            
                            # Check if it's custom implementation
                            if 'implementation' in toolkit_spec:
                                custom_toolkits.append(toolkit_spec)
                            else:
                                default_toolkits.append(toolkit_spec)
                            
        return default_toolkits, custom_toolkits
        
    async def initialize_default_toolkit(self, config: Dict[str, Any], manager: Any) -> Any:
        """
        Initialize a default Agno toolkit.
        
        Args:
            config: Toolkit configuration
            manager: Toolkit manager instance
            
        Returns:
            Initialized toolkit instance
        """
        return await manager.create_toolkit(config)
        
    async def initialize_custom_toolkit(self, config: Dict[str, Any]) -> Any:
        """
        Initialize a custom toolkit implementation.
        
        Args:
            config: Toolkit configuration with implementation path
            
        Returns:
            Initialized custom toolkit instance
        """
        # Use the manager to create custom toolkit
        await self._ensure_manager_initialized()
        return await self._manager.create_toolkit(config)
        
    async def initialize_and_register_toolkits(
        self, 
        toolkit_configs: Dict[str, Any], 
        registry: Any
    ) -> None:
        """
        Initialize and register all toolkits from configuration.
        
        Args:
            toolkit_configs: Dictionary of toolkit configurations
            registry: Toolkit registry instance
        """
        try:
            default_toolkits, custom_toolkits = self.categorize_toolkits(toolkit_configs)
            
            await self._ensure_manager_initialized()
            
            # Initialize default toolkits
            for toolkit_config in default_toolkits:
                try:
                    toolkit = await self.initialize_default_toolkit(toolkit_config, self._manager)
                    await registry.register_toolkit(toolkit)
                    logger.info(f"Registered default toolkit: {toolkit_config['name']}")
                except Exception as e:
                    logger.error(f"Failed to register default toolkit {toolkit_config['name']}: {e}")
                    
            # Initialize custom toolkits
            for toolkit_config in custom_toolkits:
                try:
                    toolkit = await self.initialize_custom_toolkit(toolkit_config)
                    await registry.register_toolkit(toolkit)
                    logger.info(f"Registered custom toolkit: {toolkit_config['name']}")
                except Exception as e:
                    logger.error(f"Failed to register custom toolkit {toolkit_config['name']}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize and register toolkits: {e}")
            raise