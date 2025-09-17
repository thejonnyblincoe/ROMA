"""
Application Configuration Value Objects.

Defines application-level configuration as domain value objects.
Single source of truth for application configuration across the system.
"""

from pydantic.dataclasses import dataclass
from pydantic import Field, field_validator
from dataclasses import field
from typing import Dict, Any, Optional
import os
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application metadata configuration."""
    
    name: str = "ROMA"
    version: str = "2.0.0"
    description: str = "Research-Oriented Multi-Agent Architecture"
    environment: str = "development"
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = ["development", "staging", "production", "testing"]
        if v not in valid_envs:
            raise ValueError(f"environment must be one of {valid_envs}, got: {v}")
        return v


@dataclass(frozen=True)
class CacheConfig:
    """Caching configuration."""
    
    enabled: bool = True
    cache_type: str = "file"
    cache_dir: str = "./runtime/cache/agent"
    ttl_seconds: int = 7200  # 2 hours
    max_size: int = 500
    
    @field_validator("cache_type")
    @classmethod
    def validate_cache_type(cls, v: str) -> str:
        valid_types = ["file", "redis", "memory"]
        if v not in valid_types:
            raise ValueError(f"cache_type must be one of {valid_types}, got: {v}")
        return v
    
    @field_validator("ttl_seconds")
    @classmethod
    def validate_ttl(cls, v: int) -> int:
        if v < 1 or v > 86400:  # 1 second to 1 day
            raise ValueError(f"ttl_seconds must be 1-86400, got: {v}")
        return v
    
    @field_validator("max_size")
    @classmethod
    def validate_max_size(cls, v: int) -> int:
        if v < 1 or v > 10000:
            raise ValueError(f"max_size must be 1-10000, got: {v}")
        return v
    
    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("cache_dir cannot be empty")
        return v.strip()


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = True
    file_path: str = "./runtime/logs/roma.log"
    file_rotation: str = "10 MB"
    file_retention: int = 3
    console_style: str = "clean"
    module_levels: Dict[str, str] = field(default_factory=dict)
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log level must be one of {valid_levels}, got: {v}")
        return v.upper()
    
    @field_validator("console_style")
    @classmethod
    def validate_console_style(cls, v: str) -> str:
        valid_styles = ["clean", "timestamp", "detailed"]
        if v not in valid_styles:
            raise ValueError(f"console_style must be one of {valid_styles}, got: {v}")
        return v
    
    @field_validator("file_retention")
    @classmethod
    def validate_file_retention(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"file_retention must be 1-100, got: {v}")
        return v
    
    @field_validator("module_levels")
    @classmethod
    def validate_module_levels(cls, v: Dict[str, str]) -> Dict[str, str]:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for module, level in v.items():
            if level.upper() not in valid_levels:
                raise ValueError(f"Invalid log level '{level}' for module '{module}'. Valid levels: {valid_levels}")
        
        # Normalize levels to uppercase
        return {module: level.upper() for module, level in v.items()}


@dataclass(frozen=True)
class SecurityConfig:
    """Security configuration."""
    
    api_keys: Dict[str, str] = field(default_factory=dict)
    encryption_enabled: bool = False
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation: bool = True
    
    @field_validator("encryption_algorithm")
    @classmethod
    def validate_encryption_algorithm(cls, v: str) -> str:
        valid_algorithms = ["AES-256-GCM", "AES-128-GCM", "ChaCha20-Poly1305"]
        if v not in valid_algorithms:
            raise ValueError(f"encryption_algorithm must be one of {valid_algorithms}, got: {v}")
        return v
    
    @field_validator("api_keys")
    @classmethod
    def validate_api_keys(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate that API keys are not empty and follow basic patterns."""
        valid_providers = ["openai", "anthropic", "google", "exa", "binance", "coingecko"]
        
        for provider, key in v.items():
            if provider not in valid_providers:
                raise ValueError(f"Unknown API provider '{provider}'. Valid providers: {valid_providers}")
            
            if key and len(key.strip()) == 0:
                raise ValueError(f"API key for '{provider}' cannot be empty string")
        
        return v
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specific provider, checking environment first."""
        # Check environment variable first
        env_key = f"{provider.upper()}_API_KEY"
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
        
        # Fall back to config value
        return self.api_keys.get(provider)


@dataclass(frozen=True)
class StorageConfig:
    """Storage configuration."""
    
    mount_path: str = "/tmp/roma_storage"
    
    @field_validator("mount_path")
    @classmethod
    def validate_mount_path(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("mount_path cannot be empty")
        return v.strip()


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment tracking configuration."""
    
    base_dir: str = "experiments"
    results_dir: str = "results"
    emergency_backup_dir: str = "emergency_backups"
    configs_dir: str = "configs"
    retention_days: int = 30
    auto_cleanup: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    @field_validator("retention_days")
    @classmethod
    def validate_retention_days(cls, v: int) -> int:
        if v < 1 or v > 365:
            raise ValueError(f"retention_days must be 1-365, got: {v}")
        return v
    
    @field_validator("base_dir", "results_dir", "emergency_backup_dir", "configs_dir")
    @classmethod
    def validate_directories(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Directory path cannot be empty")
        return v.strip()
    
    def get_experiment_path(self, experiment_id: str) -> Path:
        """Get full path for experiment directory."""
        return Path(self.base_dir) / experiment_id
    
    def get_results_path(self, experiment_id: str) -> Path:
        """Get full path for experiment results."""
        return self.get_experiment_path(experiment_id) / self.results_dir