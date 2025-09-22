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
from .database_config import DatabaseConfig
from .execution_config import ExecutionConfig


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
    database: DatabaseConfig = DatabaseConfig(database="roma_db", user="roma_user", password="roma_password")
    experiment: ExperimentConfig = ExperimentConfig()
    execution: ExecutionConfig = ExecutionConfig()
    
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
            "app": self.app.to_dict(),
            "profile": self.profile.to_dict(),
            "cache": self.cache.to_dict(),
            "logging": self.logging.to_dict(),
            "security": self.security.to_dict(),
            "storage": self.storage.to_dict(),
            "database": self.database.to_dict(),
            "experiment": self.experiment.to_dict(),
            "execution": self.execution.to_dict(),
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
            app=AppConfig.from_dict(data.get("app", {})),
            profile=ProfileConfig.from_dict(profile_data),
            cache=CacheConfig.from_dict(data.get("cache", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
            security=SecurityConfig.from_dict(data.get("security", {})),
            storage=StorageConfig.from_dict(data.get("storage", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            experiment=ExperimentConfig.from_dict(data.get("experiment", {})),
            execution=ExecutionConfig.from_dict(data.get("execution", {})),
            default_profile=data.get("default_profile", "general_profile"),
        )