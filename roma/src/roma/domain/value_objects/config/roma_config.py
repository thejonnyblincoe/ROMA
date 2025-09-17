"""
ROMA Main Configuration Value Object.

Defines the main ROMA configuration - Level 4 (Application).
Only contains application-level concerns and references to Level 3 (Profile).
Level 1&2 objects are resolved through dependency injection, not stored here.
"""

from pydantic.dataclasses import dataclass, Field
from pydantic import field_validator
from typing import Dict, Any
from .profile_config import ProfileConfig
from .app_config import AppConfig, CacheConfig, LoggingConfig, SecurityConfig, ExperimentConfig, StorageConfig


@dataclass(frozen=True)
class ROMAConfig:
    """Main ROMA configuration - Level 4 (Application)."""
    
    # App metadata (Level 4) - Use direct type annotation for OmegaConf compatibility
    app: AppConfig = AppConfig()
    
    # Level 3: Profile configuration
    profile: ProfileConfig = ProfileConfig()
    
    # Application-level configurations (Level 4)
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()
    security: SecurityConfig = SecurityConfig()
    storage: StorageConfig = StorageConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    
    # Profile selection (Level 4 concern)
    default_profile: str = "general_profile"
    
    @field_validator("default_profile")
    @classmethod
    def validate_profile_exists(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("default_profile cannot be empty")
        return v.strip()
    
    def validate_profile_completeness(self) -> Dict[str, Any]:
        """Validate that the active profile has complete agent mappings."""
        return self.profile.validate_completeness()
    
    def is_valid(self) -> bool:
        """Check if configuration is valid and complete."""
        return len(self.validate_profile_completeness()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "app": self.app,
            "profile": self.profile.to_dict(),
            "cache": self.cache,
            "logging": self.logging,
            "security": self.security,
            "storage": self.storage,
            "experiment": self.experiment,
            "default_profile": self.default_profile,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROMAConfig":
        """Create from dictionary with Hydra config mapping."""
        # Handle Hydra's structure where profile data is under 'profiles' key
        profile_data = data.get("profile", {})
        if not profile_data and "profiles" in data:
            # Map profiles → profile for Hydra compatibility
            profile_data = data["profiles"]

        return cls(
            app=AppConfig(**data.get("app", {})),
            profile=ProfileConfig.from_dict(profile_data),
            cache=CacheConfig(**data.get("cache", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            security=SecurityConfig(**data.get("security", {})),
            storage=StorageConfig(**data.get("storage", {})),
            experiment=ExperimentConfig(**data.get("experiment", {})),
            default_profile=data.get("default_profile", "general_profile"),
        )