"""Base configuration schemas for ROMA-DSPy."""

from pydantic.dataclasses import dataclass
from pydantic import field_validator
from typing import Optional


@dataclass
class CacheConfig:
    """DSPy cache system configuration."""
    enabled: bool = True  # Master toggle for cache system

    # Cache layer controls
    enable_disk_cache: bool = True
    enable_memory_cache: bool = True

    # Storage configuration
    disk_cache_dir: str = ".cache/dspy"
    disk_size_limit_bytes: int = 30_000_000_000  # 30GB (DSPy default)
    memory_max_entries: int = 1_000_000  # 1M entries (DSPy default)

    @field_validator("disk_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        """Validate cache directory is not empty."""
        if not v or not v.strip():
            raise ValueError("Cache directory cannot be empty")
        return v.strip()

    @field_validator("disk_size_limit_bytes")
    @classmethod
    def validate_size_limit(cls, v: int) -> int:
        """Validate disk size limit is positive."""
        if v <= 0:
            raise ValueError("Disk size limit must be positive")
        return v

    @field_validator("memory_max_entries")
    @classmethod
    def validate_max_entries(cls, v: int) -> int:
        """Validate memory max entries is positive."""
        if v <= 0:
            raise ValueError("Memory max entries must be positive")
        return v


@dataclass
class LLMConfig:
    """Language model configuration."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # DSPy-native retry and caching
    num_retries: int = 3
    cache: bool = True
    rollout_id: Optional[int] = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within valid range."""
        if not (0.0 <= v <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max_tokens is within valid range."""
        if not (0 < v <= 200000):
            raise ValueError(f"Max tokens must be between 1 and 200000, got {v}")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError(f"Timeout must be positive, got {v}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator("num_retries")
    @classmethod
    def validate_num_retries(cls, v: int) -> int:
        """Validate num_retries is within valid range."""
        if not (0 <= v <= 10):
            raise ValueError(f"num_retries must be between 0 and 10, got {v}")
        return v


@dataclass
class RuntimeConfig:
    """Runtime system configuration."""

    max_concurrency: int = 5
    timeout: int = 30
    verbose: bool = False
    max_depth: int = 5  # For recursive solver depth control
    enable_logging: bool = False  # Separate logging control from verbose
    log_level: str = "INFO"  # Logging level control

    # Cache configuration
    cache: Optional[CacheConfig] = None

    def __post_init__(self):
        """Initialize cache config with defaults if not provided."""
        if self.cache is None:
            self.cache = CacheConfig()

    @property
    def cache_dir(self) -> str:
        """Backward compatibility property for cache_dir."""
        return self.cache.disk_cache_dir if self.cache else ".cache/dspy"

    @field_validator("max_concurrency")
    @classmethod
    def validate_max_concurrency(cls, v: int) -> int:
        """Validate max_concurrency is within valid range."""
        if not (1 <= v <= 50):
            raise ValueError(f"Max concurrency must be between 1 and 50, got {v}")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is within valid range."""
        if not (1 <= v <= 300):
            raise ValueError(f"Timeout must be between 1 and 300 seconds, got {v}")
        return v

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        """Validate cache directory path."""
        if not v or not v.strip():
            raise ValueError("Cache directory cannot be empty")
        return v.strip()

    @field_validator("max_depth")
    @classmethod
    def validate_max_depth(cls, v: int) -> int:
        """Validate max_depth is within valid range."""
        if not (1 <= v <= 20):
            raise ValueError(f"Max depth must be between 1 and 20, got {v}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log_level is a valid logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}, got {v}")
        return v.upper()