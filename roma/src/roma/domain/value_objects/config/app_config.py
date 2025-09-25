"""
Application Configuration Value Objects.

Defines application-level configuration as domain value objects.
Single source of truth for application configuration across the system.
"""

import os
from dataclasses import field
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic.dataclasses import dataclass


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "environment": self.environment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "ROMA"),
            version=data.get("version", "2.0.0"),
            description=data.get("description", "Research-Oriented Multi-Agent Architecture"),
            environment=data.get("environment", "development"),
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "cache_type": self.cache_type,
            "cache_dir": self.cache_dir,
            "ttl_seconds": self.ttl_seconds,
            "max_size": self.max_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            cache_type=data.get("cache_type", "file"),
            cache_dir=data.get("cache_dir", "./runtime/cache/agent"),
            ttl_seconds=data.get("ttl_seconds", 7200),
            max_size=data.get("max_size", 500),
        )


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
    module_levels: dict[str, str] = field(default_factory=dict)

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
    def validate_module_levels(cls, v: dict[str, str]) -> dict[str, str]:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for module, level in v.items():
            if level.upper() not in valid_levels:
                raise ValueError(
                    f"Invalid log level '{level}' for module '{module}'. Valid levels: {valid_levels}"
                )

        # Normalize levels to uppercase
        return {module: level.upper() for module, level in v.items()}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level,
            "enable_console": self.enable_console,
            "enable_file": self.enable_file,
            "file_path": self.file_path,
            "file_rotation": self.file_rotation,
            "file_retention": self.file_retention,
            "console_style": self.console_style,
            "module_levels": dict(self.module_levels),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        """Create from dictionary."""
        return cls(
            level=data.get("level", "INFO"),
            enable_console=data.get("enable_console", True),
            enable_file=data.get("enable_file", True),
            file_path=data.get("file_path", "./runtime/logs/roma.log"),
            file_rotation=data.get("file_rotation", "10 MB"),
            file_retention=data.get("file_retention", 3),
            console_style=data.get("console_style", "clean"),
            module_levels=data.get("module_levels", {}),
        )


@dataclass(frozen=True)
class SecurityConfig:
    """Security configuration."""

    api_keys: dict[str, str] = field(default_factory=dict)
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
    def validate_api_keys(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that API keys are not empty and follow basic patterns."""
        valid_providers = ["openai", "anthropic", "google", "exa", "binance", "coingecko"]

        for provider, key in v.items():
            if provider not in valid_providers:
                raise ValueError(
                    f"Unknown API provider '{provider}'. Valid providers: {valid_providers}"
                )

            if key and len(key.strip()) == 0:
                raise ValueError(f"API key for '{provider}' cannot be empty string")

        return v

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for specific provider, checking environment first."""
        # Check environment variable first
        env_key = f"{provider.upper()}_API_KEY"
        env_value = os.getenv(env_key)
        if env_value:
            return env_value

        # Fall back to config value
        return self.api_keys.get(provider)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "api_keys": dict(self.api_keys),
            "encryption_enabled": self.encryption_enabled,
            "encryption_algorithm": self.encryption_algorithm,
            "key_rotation": self.key_rotation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityConfig":
        """Create from dictionary."""
        return cls(
            api_keys=data.get("api_keys", {}),
            encryption_enabled=data.get("encryption_enabled", False),
            encryption_algorithm=data.get("encryption_algorithm", "AES-256-GCM"),
            key_rotation=data.get("key_rotation", True),
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mount_path": self.mount_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StorageConfig":
        """Create from dictionary."""
        return cls(
            mount_path=data.get("mount_path", "/tmp/roma_storage"),
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "base_dir": self.base_dir,
            "results_dir": self.results_dir,
            "emergency_backup_dir": self.emergency_backup_dir,
            "configs_dir": self.configs_dir,
            "retention_days": self.retention_days,
            "auto_cleanup": self.auto_cleanup,
            "timestamp_format": self.timestamp_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(
            base_dir=data.get("base_dir", "experiments"),
            results_dir=data.get("results_dir", "results"),
            emergency_backup_dir=data.get("emergency_backup_dir", "emergency_backups"),
            configs_dir=data.get("configs_dir", "configs"),
            retention_days=data.get("retention_days", 30),
            auto_cleanup=data.get("auto_cleanup", True),
            timestamp_format=data.get("timestamp_format", "%Y%m%d_%H%M%S"),
        )
