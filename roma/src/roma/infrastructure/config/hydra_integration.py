"""
Hydra Integration - Infrastructure Layer.

Registers domain configuration value objects with Hydra ConfigStore.
This is the infrastructure layer that bridges domain configs with Hydra framework.
"""

from pathlib import Path

from hydra.core.config_store import ConfigStore

from roma.domain.value_objects.config import (
    AgentConfig,
    AgentMappingConfig,
    AppConfig,
    CacheConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    ProfileConfig,
    ROMAConfig,
    SecurityConfig,
)


def discover_profiles(config_dir: str = "config/profiles") -> list[str]:
    """Dynamically discover available profile YAML files."""
    profiles_dir = Path(config_dir)

    if not profiles_dir.exists():
        return []

    profile_names = []
    for yaml_file in profiles_dir.glob("*.yaml"):
        # Remove .yaml extension to get profile name
        profile_name = yaml_file.stem
        if profile_name != "__init__":  # Skip any init files
            profile_names.append(profile_name)

    return profile_names


def register_configs(config_dir: str | None = None) -> ConfigStore:
    """Register configuration schemas with Hydra ConfigStore."""

    cs = ConfigStore.instance()

    # Register main configuration with different name to avoid automatic validation
    cs.store(name="config_schema", node=ROMAConfig)

    # Register component configurations
    cs.store(name="app_config", node=AppConfig)
    cs.store(name="cache_config", node=CacheConfig)
    cs.store(name="logging_config", node=LoggingConfig)
    cs.store(name="security_config", node=SecurityConfig)
    cs.store(name="experiment_config", node=ExperimentConfig)

    # Register profile configurations
    cs.store(name="profile_config", node=ProfileConfig)
    cs.store(name="agent_mapping_config", node=AgentMappingConfig)

    # Register entity configurations
    cs.store(name="agent_config", node=AgentConfig)
    cs.store(name="model_config", node=ModelConfig)

    # Dynamically register profiles based on available YAML files
    if config_dir:
        profile_names = discover_profiles(f"{config_dir}/profiles")
    else:
        # Try common locations
        for potential_dir in ["config", "../config", "../../config"]:
            if Path(f"{potential_dir}/profiles").exists():
                profile_names = discover_profiles(f"{potential_dir}/profiles")
                break
        else:
            profile_names = []

    # Register profile schemas with different names to avoid automatic validation
    # This prevents the deprecated automatic schema matching behavior that causes merge errors
    for profile_name in profile_names:
        # Register schema with "_schema" suffix to avoid automatic matching
        cs.store(group="profiles", name=f"{profile_name}_schema", node=ProfileConfig)

    return cs


# Initialize ConfigStore on module import
cs = register_configs()
